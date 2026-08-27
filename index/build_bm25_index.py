"""
BM25 索引构建: 跟 build_simple_index.py(Chroma向量索引)并行的稀疏检索索引,
不是替代关系,是给 query/hybrid_query.py 做 RRF 融合用的另一路检索。

范围演进说明: 原来只覆盖代码知识类,现在跟 build_simple_index.py 一样改成
读 merge_chunks.py 产出的合并数据集(combined_chunks.json,文本+代码共
24297条),chunk_id 直接沿用数据集里已经编好的 txt_00000/code_00000,
保证跟 Chroma 索引的 id 完全对齐——RRF融合是靠id把两边的排名匹配起来的。

分词方式: 正则分词(提取 [a-zA-Z_][a-zA-Z0-9_]* 形式的token),只匹配ASCII
字母,不处理中文分词。合并后实测中文占比仍然只有0.07%-0.14%(课程材料本身
以英文讲义/教材为主),所以延用ASCII分词,没有临时改成中文分词器的必要;
真要支持中文检索还是得换jieba之类的分词器,这里先如实标注局限性,不是遗漏。

修订(排查"biased coin"检索失败案例后发现的真实bug,不是阈值能解决的
问题): 原来的分词器把下划线当成标识符的一部分,"biased_coin"被当成
一个不可分割的token。查询问"biased coin"(自然语言,空格分开)时,
永远匹配不上代码里连写的"biased_coin",实测这个case的BM25分数是
精确的0.0(全库倒数,不是"分数低"而是"完全没有任何token重叠")。
现在改成: 每个identifier除了保留完整token(维持"get_initial_centroids"
这种整词精确匹配的能力),再按下划线和驼峰边界拆出子词一并加入
(如"biased_coin"->[biased_coin, biased, coin],"getUsername"->
[getusername, get, username]),让自然语言提问也能对上代码标识符里的
语义子词。只有真能拆出>=2个子词时才追加,单个词不重复计数,避免虚增
词频。这会增加索引体积和标识符密集文本的token频率,是有意为之的
取舍,不是副作用。

直接效果(修复前后实测对比,同一个"R语言里怎么模拟一个有偏硬币(biased
coin)?"查询):
  - 修复前: biased_coin对应chunk的BM25分数=0.0000,排名#24033(24297条
    记录里倒数第4),RRF融合池里完全挤不进去,LLM在没有真实依据的情况下
    编造了一段不存在的sample()示例代码回答——这是比"检索不到"更危险的
    失败模式(自信地给错误答案,而不是承认不知道)。
  - 修复后: 该chunk分数=17.5819,排名#2-3,进入最终RRF top-5,生成回答
    完整引用了真实的biased_coin函数原文。

已知的连带影响(修复后做10问题回归测试时发现,不是新bug,是分词器改动
本身固有的性质): "CSE 416的作业里有没有实现K-means相关的代码?"这道题
从"正确回答"变成了"诚实拒答"。机制是: 子词拆分给几乎每个chunk都增加了
token数量,这会改变BM25公式依赖的全局统计量(比如平均文档长度),从而
连带影响了跟biased_coin完全无关的其他查询的排名——具体到这道题,原本
进入BM25 top-5的3个k-means代码notebook chunk,修复后只剩1个,被来自
另一节课(层次聚类讲义)的2个通用标题chunk挤了进来,导致最终RRF池里
代码证据不够,LLM选择回答"没有足够信息"而不是像之前那样给出正确答案。

这个连带影响判定为"在安全边界内,暂不处理": 退化方向是"正确→诚实拒答",
不是"正确→编造错误答案"——之前建立的"检索不到就不编"这道安全网(见
query/simple_query.py 和 query/hybrid_query.py 的 prompt 设计)仍然
在生效,只是召回率(recall)打了折扣,没有引入新的幻觉风险。任何分词器
改动都会影响全局词频统计,不可能只精确修复一个查询而不触碰任何其他
查询,这是可解释、可预期的代价,不是遗漏或需要立刻回滚的回归。如果后续
要收窄这个影响面,方向是给近似重复的文档(比如CSE416课件在多个文件夹
下的重复副本)做去重,而不是回退这次分词器修复。

用法:
    python index/build_bm25_index.py
"""

import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rank_bm25 import BM25Okapi

from common import load_combined_chunks, sanitize_metadata

BM25_INDEX_PATH = Path(__file__).parent / "bm25_index.pkl"

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _split_subwords(identifier: str) -> list:
    subwords = []
    for part in identifier.split("_"):
        if not part:
            continue
        for camel_part in _CAMEL_BOUNDARY_RE.sub(" ", part).split(" "):
            if camel_part:
                subwords.append(camel_part.lower())
    return subwords


def tokenize(text: str) -> list:
    tokens = []
    for raw in _TOKEN_RE.findall(text):
        tokens.append(raw.lower())
        subwords = _split_subwords(raw)
        if len(subwords) > 1:
            tokens.extend(subwords)
    return tokens


def main():
    print("=== 加载合并后的chunk数据集 ===")
    chunks = load_combined_chunks()
    print(f"总chunk数: {len(chunks)}")

    print("\n=== 分词 ===")
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]

    print("=== 构建 BM25 索引 ===")
    bm25 = BM25Okapi(tokenized_corpus)

    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [sanitize_metadata(c["metadata"]) for c in chunks]

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "ids": ids,
                "texts": texts,
                "metadatas": metadatas,
            },
            f,
        )

    print(f"\n完成。BM25索引共 {len(chunks)} 条记录,存于 {BM25_INDEX_PATH}")


if __name__ == "__main__":
    main()
