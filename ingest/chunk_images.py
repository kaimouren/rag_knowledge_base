"""
图片类chunking: 把 ingest/join_image_page_idx.py 产出的
image_processing_full.csv(11115条,OCR文字/VLM描述+已补页码)转成跟
文本/代码/数据摘要三类对齐的统一chunk schema。

chunk_type取值(按这张图片实际用了哪种信号命名,不是随便定的):
  - image_ocr: 只有OCR文字(EasyOCR平均置信度>0.85那一档,印刷体/清晰
    截图,OCR够用)
  - image_vlm: 只有VLM描述(<0.5那一档,照片/图表/手写内容)
  - image_ocr_vlm: 两者都有(0.5-0.85灰色地带,两种信号都保留)

text字段构造: 复用之前文本/代码chunk已经验证过的"文件名前缀"方案
(chunk_text_mineru.py::indexed_text同一个道理——文件名里的关键词能被
关键词检索命中),后面按信号类型加中文标签分段拼接,让下游LLM看到chunk
内容时知道"这段是OCR原文"还是"这段是VLM描述",不是混在一起降低可读性。

position/page_start-page_end: 直接读CSV里join_image_page_idx.py已经
补好的page_idx(单页,page_start==page_end==page_idx+1,跟文本chunk一样
按1-based展示,内部存储沿用MinerU的0-based page_idx语义,不做二次转换
引入误差)。standalone来源的76张,以及mineru来源里16张没能join上的
(0.14%,原因见join_image_page_idx.py的说明),page_start/page_end都是
None,如实标注,不编造页码。

quality="good"/extraction_confidence="medium": 按用户在这一步明确给定
的固定值,不是从图片内容动态判断出来的(不同于文本chunk继承文档级别的
quality)——图片这条产线目前没有"整体质量分层"的概念(不像最初文本知识类
分good/degraded/llamaparse_done三档),所有图片统一给这两个值。

用法:
    python ingest/chunk_images.py
"""

import csv
import json
from dataclasses import asdict
from pathlib import Path

from chunk_text import TextChunk

INGEST_DIR = Path(__file__).parent
CSV_PATH = INGEST_DIR / "_test_output" / "image_processing_full.csv"
OUTPUT_JSON = INGEST_DIR / "_test_output" / "image_chunks.json"


def build_text(file_name: str, ocr_text: str, vlm_desc: str) -> str:
    parts = [file_name]
    if ocr_text:
        parts.append(f"[图片OCR文字]\n{ocr_text}")
    if vlm_desc:
        parts.append(f"[图片描述]\n{vlm_desc}")
    return "\n".join(parts)


def chunk_type_of(tier: str) -> str:
    return {"ocr_only": "image_ocr", "vlm_only": "image_vlm", "both": "image_ocr_vlm"}[tier]


def build_chunk(row: dict, idx: int) -> TextChunk:
    page_idx = row["page_idx"]
    has_page = page_idx not in ("", None)
    page_start = int(page_idx) if has_page else None
    page_end = page_start

    file_name = Path(row["file"]).name
    text = build_text(file_name, row["ocr_text"], row["vlm_desc"])

    if has_page:
        note = "页码来自MinerU content_list.json的page_idx(0-based),单张图片对应单页"
        citation = f"{row['course']} / {file_name} p.{page_start + 1}"
    elif row["source"] == "standalone":
        note = "独立图片文件,不属于任何PDF,没有页码概念"
        citation = f"{row['course']} / {file_name}"
    else:
        note = "该图片在content_list.json里找不到对应block(极少数情况,见join_image_page_idx.py说明),退化为无页码"
        citation = f"{row['course']} / {file_name}"

    return TextChunk(
        chunk_id="",
        text=text,
        metadata={
            "source_type": "image",
            "file": row["file"],
            "course": row["course"],
            "chunk_type": chunk_type_of(row["tier"]),
            "name": None,
            "position": {
                "page_start": page_start + 1 if has_page else None,
                "page_end": page_end + 1 if has_page else None,
                "estimated": False,
                "note": note,
            },
            "source_citation": citation,
            "parsed_by": row["source"],  # "mineru" 或 "standalone"
            "ocr_confidence": float(row["ocr_conf"]) if row["ocr_conf"] else None,
            "chunk_quality_flag": None,
            "quality": "good",
            "extraction_confidence": "medium",
        },
    )


def main():
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig")))
    print(f"图片处理数据总行数: {len(rows)}")

    chunks = [build_chunk(r, i) for i, r in enumerate(rows)]
    for i, c in enumerate(chunks):
        c.chunk_id = f"img_{i:05d}"

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)

    from collections import Counter
    type_counts = Counter(c.metadata["chunk_type"] for c in chunks)
    with_page = sum(1 for c in chunks if c.metadata["position"]["page_start"] is not None)
    print(f"chunk_type分布: {dict(type_counts)}")
    print(f"有页码: {with_page}, 无页码: {len(chunks) - with_page}")
    print(f"\n已写入 {OUTPUT_JSON}: {len(chunks)} 个chunk")


if __name__ == "__main__":
    main()
