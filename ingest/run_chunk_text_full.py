"""
文本知识类全量chunking: 对 quality_manifest.json 里 quality in
{good, llamaparse_done, degraded} 的全部文档跑 chunk_text.py 里验证过的
切分逻辑(MarkdownHeaderTextSplitter + 二次切分 + chunk_quality_flag检测),
输出统一schema的chunk数据集。

quality=="pending" 的6个文档(5个deferred_low_priority + 1个
deferred_budget_timeout)显式跳过,不在这次构建范围内,但完整保留在
quality_manifest.json里,后续可以按pending_reason筛出来单独补处理。

用法:
    python ingest/run_chunk_text_full.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from chunk_text import load_doc_index, chunk_document

OUTPUT_PATH = Path(__file__).parent / "_test_output" / "text_chunks.json"


def chunk_to_dict(chunk) -> dict:
    return {"chunk_id": chunk.chunk_id, "text": chunk.text, "metadata": chunk.metadata}


def main():
    docs = load_doc_index()
    doc_items = list(docs.items())
    print(f"待处理文档数: {len(doc_items)}")

    all_chunks = []
    errors = []
    start_all = time.time()

    for i, (pdf_rel, doc_info) in enumerate(doc_items, 1):
        try:
            chunks = chunk_document(pdf_rel, doc_info)
            all_chunks.extend(chunks)
        except Exception as e:
            errors.append((pdf_rel, f"{type(e).__name__}: {e}"))
            print(f"  [{i}/{len(doc_items)}] [异常] {pdf_rel}: {type(e).__name__}: {e}", flush=True)
            continue

        if i % 40 == 0 or i == len(doc_items):
            print(f"  已完成 {i}/{len(doc_items)}, 累计chunk数 {len(all_chunks)}", flush=True)

    # chunk_id 目前是每个文档内部从0开始编号的,全量合并后会撞号,
    # 这里重新编一遍全局唯一id,原来的文档内序号仍保留在metadata里
    # (没有专门存,但可以从同一文档内chunk出现顺序反推,不是关键信息)
    for i, chunk in enumerate(all_chunks):
        chunk.chunk_id = f"txt_{i:05d}"

    total_elapsed = time.time() - start_all

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump([chunk_to_dict(c) for c in all_chunks], f, ensure_ascii=False, indent=2)

    print(f"\n完成: {len(doc_items)}个文档, 总耗时 {total_elapsed:.1f}s")
    print(f"处理失败: {len(errors)} 个")
    for pdf_rel, err in errors:
        print(f"  {pdf_rel}: {err}")
    print(f"\n总chunk数: {len(all_chunks)}")
    print(f"平均每文档chunk数: {len(all_chunks) / max(1, len(doc_items) - len(errors)):.1f}")
    print(f"已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
