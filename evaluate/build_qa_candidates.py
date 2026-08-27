"""
RAGAS评估测试集 - 候选QA对生成(第一步)

从 combined_chunks.json(24297条,文本+代码)分层抽样,每个抽中的chunk用
OpenAI生成一条候选QA对,输出成CSV供人工审核。这一步只产出候选,不直接
进正式测试集——人工筛完之后才决定哪些条目真正进 evaluate/qa_testset.json
(那是下一步的事,这个脚本不碰)。

生成模型用 OpenAI(不是 CLAUDE.md 原定的 Claude API)——用户当前没有
Anthropic key,沿用RAG生成阶段已经在用的选择,保持一致,不是新决策。

分层抽样方案(不是随机乱抽,覆盖依据见下方 STRATA 配置):
  文本知识类(15条): good 5 + degraded 5(其中3条特意选带chunk_quality_flag
    的,专门用来测试"面对残缺内容,系统会不会诚实说明limitation";2条选
    没有flag的,保留"degraded文档里也有正常内容"这个对照) + llamaparse_done 5
  代码知识类(20条): ipynb/py/sql/r/java 各4条
  合计35条,落在用户要求的30-40条区间。

同一层内抽样时优先来自不同文件(用之前 parse_code.py 里验证过的"一个文件
最多出一条,不够再补"逻辑),避免同一份文档反复被抽中,抽样用固定随机种子
保证可复现。

用法:
    python evaluate/build_qa_candidates.py
"""

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "ingest"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from openai import OpenAI

COMBINED_CHUNKS_PATH = Path(__file__).parent.parent / "ingest" / "_test_output" / "combined_chunks.json"
OUTPUT_CSV = Path(__file__).parent / "qa_candidates_for_review.csv"

GENERATION_MODEL = "gpt-4o-mini"
SAMPLE_SEED = 2024

MIN_TEXT_CHUNK_LEN = 150  # 太短的文本chunk(比如只有一个标题)信息量不够,生成不出有意义的问题
MIN_CODE_CHUNK_LEN = 60

# 分层抽样配置: (筛选条件, 目标数量, 分层标签)
TEXT_QUOTAS = {"good": 5, "degraded": 5, "llamaparse_done": 5}
DEGRADED_FLAGGED_QUOTA = 3  # degraded 5条里,至少3条要带chunk_quality_flag
CODE_FORMAT_QUOTA = {"ipynb": 4, "py": 4, "sql": 4, "r": 4, "java": 4}


def load_chunks():
    return json.load(open(COMBINED_CHUNKS_PATH, encoding="utf-8"))


def pick_diverse(chunks, n, rng):
    """尽量一个文件只出一条,不够再从剩余里补,复用 parse_code.py 里验证过的抽样思路。"""
    by_file = {}
    for c in chunks:
        by_file.setdefault(c["metadata"]["file"], []).append(c)

    files = list(by_file.keys())
    rng.shuffle(files)

    picked = []
    for f in files:
        if len(picked) >= n:
            break
        picked.append(rng.choice(by_file[f]))

    if len(picked) < n:
        remaining = [c for c in chunks if c not in picked]
        rng.shuffle(remaining)
        picked.extend(remaining[: n - len(picked)])

    return picked[:n]


def sample_text_chunks(all_chunks, rng):
    text_chunks = [
        c for c in all_chunks
        if c["metadata"]["source_type"] == "text" and len(c["text"]) >= MIN_TEXT_CHUNK_LEN
    ]

    sampled = []
    for quality, quota in TEXT_QUOTAS.items():
        pool = [c for c in text_chunks if c["metadata"]["quality"] == quality]

        if quality == "degraded":
            flagged = [c for c in pool if c["metadata"]["chunk_quality_flag"]]
            unflagged = [c for c in pool if not c["metadata"]["chunk_quality_flag"]]
            flagged_n = min(DEGRADED_FLAGGED_QUOTA, len(flagged))
            unflagged_n = quota - flagged_n
            sampled.extend(pick_diverse(flagged, flagged_n, rng))
            sampled.extend(pick_diverse(unflagged, unflagged_n, rng))
        else:
            sampled.extend(pick_diverse(pool, quota, rng))

    return sampled


def sample_code_chunks(all_chunks, rng):
    code_chunks = [
        c for c in all_chunks
        if c["metadata"]["source_type"] == "code" and len(c["text"]) >= MIN_CODE_CHUNK_LEN
    ]

    sampled = []
    for fmt, quota in CODE_FORMAT_QUOTA.items():
        pool = [c for c in code_chunks if c["metadata"]["chunk_type"].split("_")[0] == fmt]
        sampled.extend(pick_diverse(pool, quota, rng))

    return sampled


# ---------------------------------------------------------------------------
# QA生成
# ---------------------------------------------------------------------------
NORMAL_PROMPT = """你是在给一个"课程笔记RAG系统"构建评估测试集。下面是一段从课程材料(可能是PDF讲义、
作业解答,也可能是Python/R/Java/SQL/Jupyter Notebook代码)里切出来的chunk。

请基于这段内容生成一条问答对:
1. question: 模拟一个真实学生会怎么提问才能得到这段内容里的信息。不要直接摘抄
   原文的句子当问题,要用自己的话组织成自然语言提问(可以中文也可以英文,
   看原文语言习惯选更自然的那种)。避免"这段话说了什么"这类没有信息量的问题,
   问题要具体、有实际检索价值。
2. ground_truth: 基于这段内容能给出的标准答案,内容要来自这段文本本身,不要编造
   文本里没有的信息。

语言要求(重要): question用中文提问,ground_truth的解释性文字也要用中文写,
即使原始chunk内容是英文——不要因为原文是英文就直接照抄英文段落当答案,要用
中文转述/解释原文内容。但代码、SQL语句、原始数据输出这类"本来就不是自然语言"
的内容,照原样保留不用翻译(比如SQL查询语句本身、代码片段、"Affected rows: 4"
这类程序输出),可以在代码/数据前后用中文简要说明。如果原文是英文且question
本身用英文提问会更自然(比如问的是英文课件里的专有名词定义),则question和
ground_truth都保持英文,不要混用。

代码/SQL类问题的特殊要求(重要): 如果question是在问"怎么实现/怎么写/怎么创建
/怎么查询"这类要求生成代码或SQL的问题,ground_truth里必须完整保留chunk原文里
对应的代码/SQL语句本身(用代码块包裹),不能只用中文转述替代代码——转述可以放在
代码前后做解释,但不能省略代码原文。只有当question问的是"结果是什么/值是多少"
这类不要求写代码、只是问某个具体结果的问题时,才可以不放代码,直接给出结果。

返回JSON格式: {{"question": "...", "ground_truth": "..."}}

--- chunk内容 ---
{chunk_text}
"""

FLAGGED_PROMPT = """你是在给一个"课程笔记RAG系统"构建评估测试集,这次要构造一条专门测试"系统面对
残缺/低质量内容时会不会诚实承认局限性"的测试用例。

下面这段chunk被自动质量检测标记为: {flags}
(formula_missing = 公式内容在解析时丢失了; fragmented = 文本被OCR/解析成了碎片化的
乱码; has_unresolved_image = 包含一张没有被解析成文字的图片,注意这跟前两种不一样——
图片本身是存在的,只是这套流程目前没有把图片内容提取出来,不是内容彻底丢失;
regex_derived_lower_precision = 这段代码是用正则表达式近似解析出来的,不像
ast/nbformat那样是精确解析,细节可能有遗漏)

这段内容来源的描述(已经处理成不暴露具体文件路径的自然语言描述): {file}

请生成一条问答对:
1. question: 设计一个问题,这个问题理应能从这段内容找到答案,但由于上面这个质量
   问题,内容实际上是不完整/不可靠的。问题本身要写得像正常提问,不要在问题里暴露
   "我知道这段内容有问题"。
2. ground_truth: 标准答案要如实反映"能回答的部分",对无法覆盖的部分按标记类型
   分别处理,不要一律写成"无法提供可靠答案",并且在提到"建议查阅原文"时统一用
   上面给的 {file} 这个描述性说法引用来源(比如"具体公式请查阅{file}"),不要
   自己编造更具体的文件名/路径/页码——{file} 这个说法就是允许暴露的精度上限:
   - 如果标记包含 formula_missing: 明确说公式内容缺失/无法可靠还原,并提示
     "具体公式请查阅{file}"。
   - 如果标记包含 fragmented(且不含formula_missing): 明确说该部分内容因OCR/
     解析问题变成碎片化乱码,无法可靠还原,同样提示"建议查阅{file}"。
   - 如果标记包含 has_unresolved_image: 不要说"无法回答",而是用引用式的说法,
     比如"原文此处附有图表/图片,具体内容请参考{file}中的图片"——图片是
     存在的,只是这套chunk文本里没有解析出来,措辞上要体现"可以去看原图"
     而不是"内容没了"。
   - 如果标记包含 regex_derived_lower_precision: 能确认无误的代码部分正常
     作答,提示无法确认的细节部分"建议查阅{file}中的完整代码确认"。
   一个chunk可能同时有多个标记,按上面的规则分别处理对应的部分即可。绝对不要
   编造具体页码,也不要在 {file} 之外额外补充更具体的路径/文件名细节。

语言要求(重要): question用中文提问,ground_truth的解释性文字也要用中文写,
即使原始chunk内容是英文——不要因为原文是英文就直接照抄英文段落当答案,要用
中文转述/解释原文内容(包括"哪部分内容缺失/建议参考原图"这类说明也要用中文)。
但代码、SQL语句、原始数据输出这类"本来就不是自然语言"的内容,照原样保留不用
翻译。如果原文是英文且question本身用英文提问会更自然,则question和ground_truth
都保持英文,不要混用。

代码/SQL类问题的特殊要求(重要): 如果question是在问"怎么实现/怎么写/怎么创建
/怎么查询"这类要求生成代码或SQL的问题,ground_truth里(在诚实说明质量问题的
前提下)也要尽量保留chunk原文里能确认无误的那部分代码/SQL语句本身(用代码块
包裹),不是无条件删掉代码只留转述——除非代码本身就是被标记问题影响到不可靠
的那部分,那种情况下才不放代码,直接说明这部分不可靠。

返回JSON格式: {{"question": "...", "ground_truth": "..."}}

--- chunk内容(注意:这段内容本身可能包含乱码/占位符,这是预期的,不是你的输入错误) ---
{chunk_text}
"""


def describe_source(file_path: str, course: str) -> str:
    """把 'CSE 416\\Class14\\14.pdf' 这种带反斜杠的原始路径,转成
    '课程材料(CSE 416,Class14相关)' 这种不暴露具体路径结构、但仍然
    有辨识度的自然语言描述——比"课程原始文件"这种纯泛化说法更有用
    (用户能大致定位去哪门课的哪个作业/讲义里找),但又不会把内部文件
    组织结构原样写进ground_truth正文。"""
    if not file_path:
        return f"{course}的课程材料" if course else "课程原始材料"

    parts = [p for p in file_path.replace("/", "\\").split("\\") if p]
    # parts[0] 通常就是course本身,取它之后、文件名之前的那一级作为上下文
    # (比如 HW4、Class14、教材),没有中间层就只用course。
    context = parts[1] if len(parts) > 2 else None

    if context:
        return f"{course}({context}相关)的课程材料" if course else f"{context}相关的课程材料"
    return f"{course}的课程材料" if course else "课程原始材料"


def generate_qa(client, chunk) -> dict:
    meta = chunk["metadata"]
    flags = meta.get("chunk_quality_flag")

    if flags:
        source_desc = describe_source(meta.get("file"), meta.get("course"))
        prompt = FLAGGED_PROMPT.format(flags=",".join(flags), file=source_desc, chunk_text=chunk["text"])
    else:
        prompt = NORMAL_PROMPT.format(chunk_text=chunk["text"])

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    return result


FIELDNAMES = [
    "chunk_id", "source_type", "chunk_type", "quality", "chunk_quality_flag",
    "file", "name", "chunk_text", "generated_question", "generated_ground_truth", "error",
    "test_category", "abstention_type",
]

# test_category 区分两种评估用途,不能混在一起跑标准RAGAS指标:
#   standard: 正常QA,可以直接用 faithfulness/answer_relevancy/context_precision
#     这些标准RAGAS指标评估。
#   abstention: ground_truth 本身就是"拒答/说明局限性"(带 chunk_quality_flag 走
#     FLAGGED_PROMPT生成的那批),reference answer不是一个完整信息答案,而是"诚实
#     承认做不到"本身就是期望行为。这类不能直接套标准RAGAS指标算分——系统就算
#     答对了,也会因为reference是"我不知道"而在语义相似度类指标上被扣分,需要
#     单独设计"是否诚实说明局限性"这类指标来评估,不是这个脚本的范围,这里只
#     负责把两类数据分开标注,不混进同一个评分池子。
#
# abstention_type 是对 abstention 组的再细分,人工审核后发现"abstention"内部
# 其实混了两种不同性质的样本,笼统算一个平均分会掩盖真正想测的信号:
#   no_content: 真的答不出来,ground_truth只是"请查阅原始材料",没有实质内容。
#     这类测的是"信息真的不足时,系统会不会诚实说不知道"。
#   confident_with_caveat: chunk里其实有能确认无误的完整代码/答案,ground_truth
#     以此为主体,只是结尾加了一句"部分细节可能因质量问题缺失,建议核实"这类
#     免责声明。这类测的是"系统在信息基本充分时,会不会不必要地过度免责"——
#     跟no_content是两种不同的能力测试,不该混进同一个统计口径。
# 这个再细分需要人工判断内容"是否实质可用",没有做成自动规则(比如"chunk_text
# 长度"这类简单启发式区分不出来,试过但效果不好),所以脚本本身不会自动填充
# 这一列,是人工审核后手动标注的结果——如果重新生成新一批数据,这一列需要
# 重新人工过一遍,不能假设脚本会自动产出。


def main():
    rng = random.Random(SAMPLE_SEED)
    all_chunks = load_chunks()
    print(f"chunk总数: {len(all_chunks)}")

    text_samples = sample_text_chunks(all_chunks, rng)
    code_samples = sample_code_chunks(all_chunks, rng)
    samples = text_samples + code_samples
    print(f"文本知识类抽样: {len(text_samples)} 条")
    print(f"代码知识类抽样: {len(code_samples)} 条")
    print(f"合计: {len(samples)} 条候选")

    client = OpenAI()
    rows = []

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        f.flush()

        for i, chunk in enumerate(samples, 1):
            meta = chunk["metadata"]
            row = {
                "chunk_id": chunk["chunk_id"],
                "source_type": meta["source_type"],
                "chunk_type": meta["chunk_type"],
                "quality": meta.get("quality") or "",
                "chunk_quality_flag": ",".join(meta["chunk_quality_flag"]) if meta.get("chunk_quality_flag") else "",
                "file": meta.get("file") or "",
                "name": meta.get("name") or "",
                "chunk_text": chunk["text"],
                "generated_question": "",
                "generated_ground_truth": "",
                "error": "",
                "test_category": "abstention" if meta.get("chunk_quality_flag") else "standard",
            }
            try:
                qa = generate_qa(client, chunk)
                row["generated_question"] = qa["question"]
                row["generated_ground_truth"] = qa["ground_truth"]
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                print(f"  [{i}/{len(samples)}] [失败] {chunk['chunk_id']}: {row['error']}", flush=True)

            writer.writerow(row)
            f.flush()
            rows.append(row)

            if not row["error"]:
                print(f"  [{i}/{len(samples)}] {chunk['chunk_id']} ({meta['source_type']}/{meta['chunk_type']}) 生成完成", flush=True)

    errors = [r for r in rows if r["error"]]
    print(f"\n完成: {len(rows)} 条候选, {len(errors)} 条生成失败")
    print(f"已保存至: {OUTPUT_CSV}")
    print("\n这是候选清单,不是正式测试集——请人工审核筛选后再确定最终进入 evaluate/qa_testset.json 的条目。")


if __name__ == "__main__":
    main()
