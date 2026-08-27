"""
Chroma向量索引构建

范围演进说明: 最初这个脚本只覆盖代码知识类(ipynb/py/sql/r/java,3091条chunk),
目的是先验证"存进去→检索出来→生成回答"这条端到端链路能不能跑通。现在文本
知识类(439个PDF,21206条chunk)也完成了chunking并跟代码知识类合并进了统一
schema(merge_chunks.py的输出combined_chunks.json,共24297条),这个脚本
改成直接读那份合并数据,不再自己现跑parse_code.py——索引构建和chunk生成
彻底解耦,重跑索引不需要重新解析源文件。

embedding模型选择依据(扩大范围后重新核实过,不是沿用旧结论): 实测合并后的
语料里,文本知识类chunk中文占比0.07%,代码知识类chunk中文占比0.14%,两边
都跟最初评估代码知识类时的结论(0.11%)一个数量级——课程材料本身以英文
讲义/教材为主,所以继续用轻量英文模型 all-MiniLM-L6-v2,没有临时切换成
bge-large-zh-v1.5的必要。

metadata 处理: combined_chunks.json 里的 chunk 比最初的代码chunk多了
position 这种嵌套dict字段(文本知识类才有,记录字符偏移量估算值),Chroma
metadata 不接受嵌套dict,common.sanitize_metadata() 现在会把dict转成JSON
字符串存(向量检索用不到这个字段做过滤,只是为了不丢信息)。

用法:
    python index/build_simple_index.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "ingest"))
sys.path.insert(0, str(Path(__file__).parent))

import chromadb
from sentence_transformers import SentenceTransformer

from common import load_combined_chunks, sanitize_metadata

CHROMA_DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "knowledge_chunks_v0"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


def main():
    print("=== 加载合并后的chunk数据集 ===")
    chunks = load_combined_chunks()
    print(f"总chunk数: {len(chunks)}")

    print(f"\n=== 加载embedding模型: {EMBEDDING_MODEL} ===")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"\n=== 初始化 Chroma (collection={COLLECTION_NAME}) ===")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    print(f"\n=== 向量化 + 写入 (batch={BATCH_SIZE}) ===")
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        ids = [c["chunk_id"] for c in batch]
        metadatas = [sanitize_metadata(c["metadata"]) for c in batch]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        if (i // BATCH_SIZE) % 20 == 0 or i + BATCH_SIZE >= len(chunks):
            print(f"  已写入 {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print(f"\n完成。collection '{COLLECTION_NAME}' 共 {collection.count()} 条记录,存于 {CHROMA_DB_DIR}")


if __name__ == "__main__":
    main()
