"""
Hybrid检索: Chroma向量检索 + BM25稀疏检索,用RRF(Reciprocal Rank Fusion)融合

背景: simple_query.py 纯向量检索在3个测试问题上跑偏(K-means、Caregiver类、
Caregivers表SQL——后者是"文件对了但chunk错了"),这些失败模式的共同点是
问题里包含精确的标识符(函数名/类名/表名),而向量相似度对这类精确关键词
不敏感,BM25(基于词频的稀疏检索)正好擅长这个。

RRF 公式: score(chunk) = sum( 1 / (k + rank_i) ),对每个检索器(向量/BM25)
各自给出的排名 rank_i(从1开始)算一个倒数分,两个检索器的倒数分相加,
最后按总分排序。k=60 是RRF论文/业界的标准取值,不是调出来的。

设计要点: 两个检索器各自先取 top-10,再融合出最终 top-5——不是每个各出
2-3个再拼凑,是让两边都有足够候选参与竞争,防止只有一边强的信号被另一边
的小候选池排除掉。

生成阶段逻辑完全复用 simple_query.py 的 generate_answer(),不做改动,
这次变化只在检索这一层。

Query Rewriting(可选增强,默认关闭): 见 query_rewrite.py 的模块docstring。
简单说是用一次轻量LLM调用从问题里提取"核心实体"(可能的函数名/类名/
表名),按语言边界拆分成token后,复用主BM25索引的get_scores()对这批
token单独打一次分,分数高的chunk作为【第三路】候选(带更高初始权重)
一起参与RRF融合——不改变、不替换原有的向量+BM25两路检索,只是叠加。

第一轮实现(v1)用的是"大小写不敏感精确子串匹配+命中计数"打分,实测
发现两个问题:(1) LLM提取出的中英混合复合词(比如"SQL查询")按整个
短语去匹配,匹配不上只包含"SQL"这个词根的英文原文;(2) 命中计数对
"Caregiver"这类常见英文单词没有特异性判断,导致在无关文档里的偶然
命中被同等对待,挤掉了原本正确的结果(具体案例和数据见
evaluate/query_rewrite_eval.py 的测试记录)。v2改成:提取出的实体先
按语言边界拆分(query_rewrite.py::split_entity_to_tokens()),再直接
喂给主BM25索引打分——常见词自然因为在全库文档频率高、IDF低被打低分,
不需要额外发明"这个词罕不罕见"的规则,复用的是BM25本身已经内置的机制。

设为可选而非默认开启,是因为效果还在验证阶段(见
evaluate/query_rewrite_eval.py 的测试结果和结论),没有足够把握说
"利大于弊"之前不改变线上默认行为。

用法:
    python query/hybrid_query.py                # 跑内置对比测试问题
    python query/hybrid_query.py "自定义问题"      # 跑单个问题
"""

import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "index"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from build_bm25_index import tokenize, BM25_INDEX_PATH
from query_rewrite import extract_query_entities
from simple_query import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    generate_answer,
)

RETRIEVE_TOP_K = 10  # 每个检索器各自先取多少候选
FINAL_TOP_K = 5  # RRF融合后最终取多少
RRF_K = 60  # RRF标准常数

ENTITY_TOP_K = 10  # 实体BM25打分这一路先取多少候选,和另外两路对齐
# 实体匹配结果的RRF权重倍数(>1表示比同排名的向量/BM25信号更值得信任,
# 因为这是从LLM提取出的"核心概念"单独打分,不是泛化的语义/词频匹配)。
# 1.5是初始估计值,不是精调出来的,具体要不要调整、调多少,要看
# evaluate/query_rewrite_eval.py 的实测效果再定。
ENTITY_WEIGHT = 1.5

# 之前纯向量检索的3个失败案例 + 2个成功案例,一起跑,既看有没有修复,
# 也看有没有把原本对的搞错
TEST_QUESTIONS = [
    ("失败案例", "CSE 416的作业里有没有实现K-means相关的代码?"),
    ("失败案例", "vaccine scheduler项目里Caregiver类是怎么实现的?"),
    ("失败案例(文件对chunk错)", "有没有创建Caregivers表的SQL语句?"),
    ("成功案例", "get_initial_centroids这个函数是做什么的?"),
    ("成功案例", "R语言里怎么模拟一个有偏硬币(biased coin)?"),
]


def load_resources():
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = chroma_client.get_collection(COLLECTION_NAME)

    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_store = pickle.load(f)

    openai_client = OpenAI()
    return embed_model, collection, bm25_store, openai_client


def retrieve_vector_ids(embed_model, collection, question: str, top_k=RETRIEVE_TOP_K):
    query_embedding = embed_model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results["ids"][0]  # 已经按相似度排好序


def retrieve_bm25_ids(bm25_store, question: str, top_k=RETRIEVE_TOP_K):
    tokens = tokenize(question)
    scores = bm25_store["bm25"].get_scores(tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [bm25_store["ids"][i] for i in ranked_indices]


def retrieve_entity_bm25_matches(bm25_store, entity_tokens, top_k=ENTITY_TOP_K):
    """query rewriting的第三路检索(v2,替换掉v1的"精确子串匹配+命中计数"
    方案——见本文件模块docstring "Query Rewriting" 一节的v1/v2对比说明)。

    entity_tokens 是 query_rewrite.py::extract_query_entities() 已经按
    语言边界拆分过的token列表,这里直接喂给build_bm25_index.tokenize()
    (会做进一步的camelCase/下划线子词拆分,和主BM25检索路径用的是同一套
    规则)再调用bm25.get_scores()——不是重新发明一套打分逻辑,是把提取出
    的实体token当成一个"增强query"复用主索引本身的打分机制,常见词自然
    因为IDF低被打低分。

    entity_tokens里如果有中文token(jieba切出来的),这里的tokenize()
    不会产出任何token给它们——build_bm25_index.tokenize()的正则只识别
    ASCII标识符模式,对中文字符没有匹配规则,这是当前BM25索引本身英文
    为主的选型导致的(LIMITATIONS.md已记录的已知局限),不是这里新引入
    的bug。意味着这一路实际有效果的是entity_tokens里的英文/技术词部分,
    纯中文短语部分即使被jieba切细了,也拿不到能打分的token。

    score<=0的候选不返回——BM25对完全不相关的query token本来就会打0分
    或负分(idf<0的情况),这些不该被当成"匹配到了"混进候选池。
    """
    if not entity_tokens:
        return []

    query_tokens = []
    for t in entity_tokens:
        query_tokens.extend(tokenize(t))
    if not query_tokens:
        return []

    scores = bm25_store["bm25"].get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    ids = bm25_store["ids"]
    return [ids[i] for i in ranked_indices if scores[i] > 0]


def rrf_fuse(vector_ids, bm25_ids, entity_ids=None, entity_weight=ENTITY_WEIGHT, k=RRF_K, final_top_k=FINAL_TOP_K):
    """entity_ids 是query rewriting那一路的结果,默认None(不传就是原来
    向量+BM25两路融合,行为完全不变——旧调用方式 rrf_fuse(vector_ids,
    bm25_ids) 不受影响)。传了的话按entity_weight给更高的初始权重再并入
    同一个RRF融合池,不是单独排序后拼接。
    """
    scores = defaultdict(float)
    for rank, cid in enumerate(vector_ids, start=1):
        scores[cid] += 1 / (k + rank)
    for rank, cid in enumerate(bm25_ids, start=1):
        scores[cid] += 1 / (k + rank)
    if entity_ids:
        for rank, cid in enumerate(entity_ids, start=1):
            scores[cid] += entity_weight * (1 / (k + rank))

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:final_top_k]


def retrieve_hybrid(question: str, embed_model, collection, bm25_store, openai_client, use_query_rewrite=False):
    """在原有向量+BM25两路检索基础上,可选叠加query rewriting第三路。
    返回 (fused, rewrite_info)——rewrite_info 在 use_query_rewrite=False
    时是 None,不会白白多花一次LLM调用。
    """
    vector_ids = retrieve_vector_ids(embed_model, collection, question)
    bm25_ids = retrieve_bm25_ids(bm25_store, question)

    entity_ids = None
    rewrite_info = None
    if use_query_rewrite:
        rewrite_info = extract_query_entities(openai_client, question)
        entity_ids = retrieve_entity_bm25_matches(bm25_store, rewrite_info["entity_tokens"])

    fused = rrf_fuse(vector_ids, bm25_ids, entity_ids=entity_ids)
    return fused, rewrite_info


def lookup_by_id(bm25_store, chunk_id: str):
    idx = bm25_store["ids"].index(chunk_id)
    return bm25_store["texts"][idx], bm25_store["metadatas"][idx]


def build_prompt(question: str, items) -> str:
    """items: [(text, metadata), ...],跟 simple_query.build_prompt 用同样的prompt格式,
    方便公平对比生成质量。"""
    context_blocks = []
    for i, (doc, meta) in enumerate(items, 1):
        source = f"[来源{i}] {meta.get('file', '?')} | {meta.get('chunk_type', '?')}"
        if meta.get("name"):
            source += f" | {meta['name']}"
        context_blocks.append(f"{source}\n```\n{doc}\n```")

    context = "\n\n".join(context_blocks)
    return (
        "你是一个帮助学生查阅课程代码资料的助手。请只根据下面提供的代码片段回答问题,"
        "如果片段里没有足够信息回答,请明确说明。回答时请说明依据来自哪个来源(比如[来源1])。\n\n"
        f"### 检索到的代码片段:\n{context}\n\n"
        f"### 问题:\n{question}\n"
    )


def format_source_line(meta) -> str:
    name_part = f" | {meta['name']}" if meta.get("name") else ""
    return f"{meta.get('file', '?')} | {meta.get('chunk_type', '?')}{name_part}"


def run_comparison(embed_model, collection, bm25_store, openai_client, label: str, question: str):
    print(f"\n{'=' * 100}")
    print(f"[{label}] {question}")
    print(f"{'=' * 100}")

    vector_ids = retrieve_vector_ids(embed_model, collection, question)
    bm25_ids = retrieve_bm25_ids(bm25_store, question)
    fused = rrf_fuse(vector_ids, bm25_ids)

    print(f"\n--- 纯向量检索 top-5(旧结果) ---")
    for cid in vector_ids[:FINAL_TOP_K]:
        _, meta = lookup_by_id(bm25_store, cid)
        print(f"  {cid}  {format_source_line(meta)}")

    print(f"\n--- BM25检索 top-5(参考) ---")
    for cid in bm25_ids[:FINAL_TOP_K]:
        _, meta = lookup_by_id(bm25_store, cid)
        print(f"  {cid}  {format_source_line(meta)}")

    print(f"\n--- Hybrid(RRF融合)top-5(新结果) ---")
    hybrid_items = []
    for cid, score in fused:
        text, meta = lookup_by_id(bm25_store, cid)
        hybrid_items.append((text, meta))
        print(f"  {cid}  rrf_score={score:.5f}  {format_source_line(meta)}")

    prompt = build_prompt(question, hybrid_items)
    answer = generate_answer(openai_client, prompt)
    print(f"\nHybrid检索下的生成回答:\n{answer}")


def main():
    embed_model, collection, bm25_store, openai_client = load_resources()
    print(f"Chroma记录数: {collection.count()}, BM25记录数: {len(bm25_store['ids'])}")

    if len(sys.argv) > 1:
        questions = [("自定义", " ".join(sys.argv[1:]))]
    else:
        questions = TEST_QUESTIONS

    for label, q in questions:
        run_comparison(embed_model, collection, bm25_store, openai_client, label, q)


if __name__ == "__main__":
    main()
