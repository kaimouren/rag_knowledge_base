"""
最简版检索+生成: 验证"向量检索 -> 拼prompt -> LLM生成回答"这条链路能跑通

不做优化: 单路向量检索(没有BM25/hybrid),没有rerank,prompt也是最朴素的
拼接方式。目的只是验证 build_simple_index.py 建好的索引能不能真的支撑
一次问答,不是最终版本。

生成模型用 OpenAI(不是 CLAUDE.md 里原定的 Claude API)——这是当前这一步
的临时选择(手头有现成的OpenAI key,先跑通链路),生成模块单独封装成
一个函数(generate_answer),后续要换回 Claude API 只需要改这一个函数,
不影响检索部分。

用法:
    python query/simple_query.py                  # 跑内置的5个测试问题
    python query/simple_query.py "自定义问题"        # 跑单个问题
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "ingest"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

CHROMA_DB_DIR = Path(__file__).parent.parent / "index" / "chroma_db"
COLLECTION_NAME = "knowledge_chunks_v0"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
GENERATION_MODEL = "gpt-4o-mini"

TEST_QUESTIONS = [
    "get_initial_centroids这个函数是做什么的?",
    "CSE 416的作业里有没有实现K-means相关的代码?",
    "vaccine scheduler项目里Caregiver类是怎么实现的?",
    "有没有创建Caregivers表的SQL语句?",
    "R语言里怎么模拟一个有偏硬币(biased coin)?",
]


def load_resources():
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    openai_client = OpenAI()  # 从环境变量 OPENAI_API_KEY 读取
    return embed_model, collection, openai_client


def retrieve(embed_model, collection, question: str, top_k: int = TOP_K):
    query_embedding = embed_model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results


def build_prompt(question: str, results) -> str:
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_blocks = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
        source = f"[来源{i}] {meta.get('file', '?')} | {meta.get('chunk_type', '?')}"
        if meta.get("name"):
            source += f" | {meta['name']}"
        context_blocks.append(f"{source}\n```\n{doc}\n```")

    context = "\n\n".join(context_blocks)
    prompt = (
        "你是一个帮助学生查阅课程代码资料的助手。请只根据下面提供的代码片段回答问题,"
        "如果片段里没有足够信息回答,请明确说明。回答时请说明依据来自哪个来源(比如[来源1])。\n\n"
        f"### 检索到的代码片段:\n{context}\n\n"
        f"### 问题:\n{question}\n"
    )
    return prompt


def generate_answer(openai_client, prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def answer_question(embed_model, collection, openai_client, question: str):
    results = retrieve(embed_model, collection, question)
    prompt = build_prompt(question, results)
    answer = generate_answer(openai_client, prompt)

    print(f"\n{'=' * 90}")
    print(f"问题: {question}")
    print(f"{'=' * 90}")
    print(f"\n回答:\n{answer}")

    print(f"\n检索来源(top-{TOP_K}):")
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        name_part = f" | {meta['name']}" if meta.get("name") else ""
        print(f"  - {meta.get('file', '?')} | {meta.get('chunk_type', '?')}{name_part}  (距离={dist:.3f})")


def main():
    embed_model, collection, openai_client = load_resources()
    print(f"索引记录数: {collection.count()}")

    if len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]
    else:
        questions = TEST_QUESTIONS

    for q in questions:
        answer_question(embed_model, collection, openai_client, q)


if __name__ == "__main__":
    main()
