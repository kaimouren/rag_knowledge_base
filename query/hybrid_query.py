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


def rrf_fuse(vector_ids, bm25_ids, k=RRF_K, final_top_k=FINAL_TOP_K):
    scores = defaultdict(float)
    for rank, cid in enumerate(vector_ids, start=1):
        scores[cid] += 1 / (k + rank)
    for rank, cid in enumerate(bm25_ids, start=1):
        scores[cid] += 1 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:final_top_k]


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
