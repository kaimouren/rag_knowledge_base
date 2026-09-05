"""
Query Rewriting(核心实体提取增强)

用一次轻量LLM调用,从用户的自然语言问题里提取"核心技术标识符"(可能的
函数名/类名/表名/文件名),辅助解决两类之前已经诊断清楚、但hybrid检索
(向量+BM25+RRF,见hybrid_query.py)本身解决不了的失败模式:

  - 多词稀释: 问题里同时出现多个实体词(比如"vaccine scheduler项目里
    Caregiver类"——"vaccine scheduler"和"Caregiver"两个词一起参与
    BM25/向量匹配),真正的核心实体(Caregiver类)反而被和它不太相关的
    背景词稀释掉,排不进最终top-5。
  - 词汇鸿沟: 用户用自然语言描述("有偏硬币"),但代码里用的是英文标识符
    ("biased_coin");或者query关键词在目标chunk原文里字面没出现(比如
    B类案例里"在SQL中..."这个问题,目标chunk原文只写了"Joins Examples",
    没有"SQL"这个词,BM25/向量都抓不住)。

设计上明确是"增强"不是"替换": hybrid_query.py原有的两路检索(向量+BM25)
完全不动。提取出的核心实体只是作为【第三路检索】的输入——对chunk原文
做大小写不敏感的精确子串匹配(不经过BM25分词器那条路径),匹配结果连同
原来两路一起参与RRF融合,给一个更高的初始权重(因为这是从"核心概念"
精确匹配来的信号,比泛化的语义/词频匹配更确定)。LLM调用失败或返回
格式不对时,静默降级为空实体列表——这是叠加的增强步骤,不能因为它出错
就拖垮原本能正常工作的检索路径。

模型选择说明: 最初设想用Claude Haiku(轻量、低成本),但当前环境没有
配置 ANTHROPIC_API_KEY(和生成/评估阶段一样的约束,见README"已知偏差"
一节——手头没有现成的Anthropic key)。这里用gpt-4o-mini代替,做的是
同样"用小模型控制成本和延迟"的事,不是设计上的改变,只是同一个约束在
这个新模块上的延续,后续拿到Claude key之后按同样的接口换掉即可。

用法:
    python query/query_rewrite.py "问题1" "问题2" ...
"""

import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import jieba
from openai import OpenAI

REWRITE_MODEL = "gpt-4o-mini"
MAX_ENTITIES = 2

# 中英文混合实体的语言边界切分。第一轮实测发现LLM经常把英文技术词和中文
# 修饰语粘成一个复合短语提取出来(比如"SQL查询"、"离散随机变量X"),拿整个
# 复合短语去匹配纯英文的原始语料,几乎不可能匹配上——哪怕其中的"SQL"单独
# 拿出来是能匹配上原文的。这里按"连续CJK字符段"/"连续非CJK字符段"把每个
# 实体切开,非CJK段(英文单词、标识符、数字)保持完整一个token,不用jieba
# 处理(jieba是中文分词器,对英文没有意义,而且"get_initial_centroids"这种
# 标识符本身就该保持完整,拆分交给下游 build_bm25_index.tokenize() 的
# camelCase/下划线子词逻辑去做);CJK段再用jieba切细,不要整个中文短语当
# 一个不可能被逐字匹配上的长字符串。
_CJK_OR_OTHER_RE = re.compile(r"[一-鿿]+|[^一-鿿\s]+")
_CJK_FULL_RE = re.compile(r"[一-鿿]+")


def split_entity_to_tokens(entity: str) -> list:
    tokens = []
    for match in _CJK_OR_OTHER_RE.finditer(entity):
        segment = match.group()
        if _CJK_FULL_RE.fullmatch(segment):
            tokens.extend(w.strip() for w in jieba.cut(segment) if w.strip())
        else:
            tokens.append(segment)
    return tokens

SYSTEM_PROMPT = (
    "你是一个帮助分析课程笔记检索问题的助手。用户会提出一个关于课程代码/"
    "笔记的问题,你需要从中提取【核心技术标识符】——也就是问题里最可能"
    "对应代码或文档里真实存在的函数名、类名、表名、文件名的词或短语,"
    "即使用户是用自然语言描述的,也要联想可能对应的标识符写法(比如"
    "\"有偏硬币\"可能对应代码里的\"biased_coin\",\"K-means\"可能对应"
    "\"get_initial_centroids\"这类相关函数名——如果不确定具体函数名,"
    "保留用户原话里的关键词也可以,不要编造一个听起来合理但你并不确定"
    "存在的具体名字)。\n\n"
    "如果问题里出现多个不同的实体词(比如某个项目名+某个类名),判断哪个"
    "是问题真正想问的核心对象,只保留最核心的1-2个,不要把所有词都同等"
    "列出来——只是背景上下文、不是问题真正对象的词不要放进结果(比如"
    "\"某某项目里Caregiver类是怎么实现的\"这种问法,核心是Caregiver类,"
    "项目名只是背景,不需要放进结果)。\n\n"
    "严格按下面的JSON格式输出,不要有任何其他文字:\n"
    '{"core_entities": ["实体1", "实体2"]}\n\n'
    f"core_entities最多{MAX_ENTITIES}个,如果问题里根本没有可辨认的具体"
    "标识符(纯概念性提问,比如\"什么是整数规划\"),返回空列表。"
)


def extract_query_entities(openai_client, question: str) -> dict:
    """返回:
    {
        "original_query": question,
        "core_entities": [...],       # 提取失败时为空列表,不抛异常
        "entity_tokens": [...],       # core_entities按语言边界拆分后的token,
                                        # 供下游BM25打分直接使用
        "latency_sec": float,
        "usage": {"input_tokens": int, "output_tokens": int},
        "error": Optional[str],
    }
    """
    result = {
        "original_query": question,
        "core_entities": [],
        "entity_tokens": [],
        "latency_sec": 0.0,
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "error": None,
    }
    start = time.perf_counter()
    try:
        response = openai_client.chat.completions.create(
            model=REWRITE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result["latency_sec"] = time.perf_counter() - start
        usage = response.usage
        if usage is not None:
            result["usage"]["input_tokens"] = usage.prompt_tokens
            result["usage"]["output_tokens"] = usage.completion_tokens

        content = response.choices[0].message.content
        parsed = json.loads(content)
        entities = parsed.get("core_entities", [])
        if isinstance(entities, list):
            cleaned = [str(e).strip() for e in entities if str(e).strip()]
            result["core_entities"] = cleaned[:MAX_ENTITIES]
    except Exception as e:
        result["latency_sec"] = time.perf_counter() - start
        result["error"] = f"{type(e).__name__}: {e}"

    for entity in result["core_entities"]:
        result["entity_tokens"].extend(split_entity_to_tokens(entity))

    return result


def main():
    client = OpenAI()
    questions = sys.argv[1:] or [
        "vaccine scheduler项目里Caregiver类是怎么实现的?",
        "R语言里怎么模拟一个有偏硬币(biased coin)?",
        "在SQL中,如何使用不同类型的连接查询两个表的结果?能给我一些具体的例子吗?",
    ]
    for q in questions:
        r = extract_query_entities(client, q)
        print(f"问题: {q}")
        print(f"  core_entities: {r['core_entities']}")
        print(f"  entity_tokens: {r['entity_tokens']}")
        print(
            f"  耗时: {r['latency_sec']:.2f}s  "
            f"tokens(in/out): {r['usage']['input_tokens']}/{r['usage']['output_tokens']}"
        )
        if r["error"]:
            print(f"  [错误] {r['error']}")
        print()


if __name__ == "__main__":
    main()
