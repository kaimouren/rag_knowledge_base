"""
文本知识类chunking: 把 quality_manifest.json 里已经生成好的document级别
markdown(Docling/LlamaParse输出)切分成chunk级别,转成跟代码知识类
(parse_code.py)对齐的统一schema。

两层切分:
  1) MarkdownHeaderTextSplitter 按 #/##/### 标题切第一刀,保留章节标题
     (对应schema里的 name 字段),这一刀天然贴合文档的语义结构。
  2) 每个标题段落内部再用 RecursiveCharacterTextSplitter 做二次切分,
     控制chunk大小,避免大章节整个变成一个超长chunk。
     CHUNK_SIZE=700、CHUNK_OVERLAP=100: 落在用户要求的500-800字符区间
     中段,overlap取约15%,避免句子/公式被硬切断后两边chunk都读不出
     完整语义(这是RecursiveCharacterTextSplitter的标准用法,不是精调出来的)。

chunk_type 的判定规则(用户原话"用章节层级判断,比如heading_section/
paragraph",但没给出精确定义,这里是我的解读,已在下方注释里说明):
  一个标题段落如果本身长度没超过CHUNK_SIZE、没有被二次切分,判定为
  heading_section(这个chunk就是完整的一节);如果超长被切成了多块,
  每一块判定为paragraph(这个chunk只是某一节的一部分,不能代表整节)。
  这个区分是为了让下游知道"这个chunk读到的是不是这一节的全部内容"。

quality vs chunk_quality_flag 是两个独立字段,不是一个字段的"降级":
  - quality: 直接继承文档级别(quality_manifest.json)的 good/degraded/
    llamaparse_done,取值空间跟chunk_quality_flag完全不同,不存在互相
    覆盖的问题。
  - chunk_quality_flag: chunk粒度独立检测出的具体问题(可能有多个,存成
    列表;什么都没检测到就是None)。选择列表而不是单值,是因为逐项检测出
    的问题不是互斥的(一个chunk完全可能同时公式缺失又包含未解析图片),
    单值会丢信息。
  两者的价值差异: 之前分析发现,即便一份文档整体被判定为"good",其中
  个别chunk仍可能踩到"formula-not-decoded"或"<!-- image -->"(只是没多到
  拉低整份文档的判定),chunk级别的检测能补上文档级别看不到的这层细节,
  这正是"质量标记下沉到chunk级别"要解决的问题。

位置信息(position)的真实情况: 实测确认 Docling 和 LlamaParse 输出的
markdown里都没有任何页码标记(用grep在样本文件里搜"page"关键词和常见
分页符,一个都没有),所以没法给出准确的"第几页到第几页"。这里退而求其次,
用chunk在原始文档全文里的字符起止位置(char_start/char_end)做近似定位,
显式标注 estimated=true,不假装这是页码。

用法(先跑小范围测试,不是全量):
    python ingest/chunk_text.py
"""

import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from quality_router import line_fragmentation_ratio, FRAGMENT_RATIO_THRESHOLD

DATA_ROOT = Path(r"E:\project")
INGEST_DIR = Path(__file__).parent
MANIFEST_PATH = INGEST_DIR / "_test_output" / "quality_manifest.json"
FINAL_REPORT_CSV = INGEST_DIR / "_test_output" / "final_report.csv"

HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

FORMULA_PLACEHOLDER = "formula-not-decoded"
IMAGE_PLACEHOLDER = "<!-- image -->"


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 文档定位: 从 quality_manifest.json + final_report.csv 里找到每个PDF对应的
# markdown文件路径、course、quality、来源说明
# ---------------------------------------------------------------------------
def load_doc_index():
    manifest = json.load(open(MANIFEST_PATH, encoding="utf-8"))
    final_rows = {r["file"]: r for r in csv.DictReader(open(FINAL_REPORT_CSV, encoding="utf-8-sig"))}

    docs = {}
    skipped_pending = 0
    for pdf_rel, entry in manifest.items():
        if entry["quality"] == "pending":
            # 统一的"暂缓处理"状态,不是解析失败——涵盖两类原因
            # (pending_reason: deferred_low_priority公开教材/期刊论文,
            # deferred_budget_timeout预算内暂缓的超时文件),显式跳过,
            # 不进入本次chunking范围,但完整记录在manifest里,后续可按
            # pending_reason筛选出来单独补处理。
            skipped_pending += 1
            continue

        if entry["quality"] == "llamaparse_done":
            md_rel = entry["llamaparse_md_file"]
            md_path = INGEST_DIR / md_rel
        else:
            row = final_rows.get(pdf_rel)
            if not row or not row.get("md_file"):
                skipped_pending += 1
                continue
            md_path = INGEST_DIR / row["md_file"]

        docs[pdf_rel] = {
            "md_path": md_path,
            "course": entry["course"],
            "quality": entry["quality"],
            "source_note": entry.get("docling_reasons") or entry.get("note") or "",
        }

    print(f"文档定位: 可chunking {len(docs)} 个, 暂缓处理(pending)跳过 {skipped_pending} 个")
    return docs


# ---------------------------------------------------------------------------
# chunk级别质量检测
# ---------------------------------------------------------------------------
def detect_chunk_quality_flags(text: str) -> Optional[List[str]]:
    flags = []
    if FORMULA_PLACEHOLDER in text:
        flags.append("formula_missing")
    if IMAGE_PLACEHOLDER in text:
        flags.append("has_unresolved_image")

    frag_ratio = line_fragmentation_ratio(text)
    if frag_ratio is not None and frag_ratio > FRAGMENT_RATIO_THRESHOLD:
        flags.append("fragmented")

    return flags if flags else None


# ---------------------------------------------------------------------------
# 切分主逻辑
# ---------------------------------------------------------------------------
def chunk_document(pdf_rel: str, doc_info: dict) -> List[TextChunk]:
    full_text = doc_info["md_path"].read_text(encoding="utf-8")

    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False)
    header_sections = header_splitter.split_text(full_text)

    char_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    chunks: List[TextChunk] = []
    search_cursor = 0
    chunk_idx = 0

    for section in header_sections:
        section_text = section.page_content
        # 章节标题取最细的一级(h3 > h2 > h1),没有标题的开头段落就是None
        heading = section.metadata.get("h3") or section.metadata.get("h2") or section.metadata.get("h1")

        if len(section_text) <= CHUNK_SIZE:
            pieces = [section_text]
            chunk_type = "heading_section"
        else:
            pieces = char_splitter.split_text(section_text)
            chunk_type = "paragraph"

        for piece in pieces:
            # 用片段的前50个字符在原文里定位真实字符偏移量,定位不到就退化成
            # "上一个chunk结束的地方"累加估算——两种情况都标注estimated=true,
            # 不假装这是精确页码。
            probe = piece[:50]
            found_at = full_text.find(probe, search_cursor)
            if found_at == -1:
                found_at = full_text.find(probe)
            if found_at != -1:
                char_start = found_at
                char_end = found_at + len(piece)
                search_cursor = char_end
            else:
                char_start = search_cursor
                char_end = search_cursor + len(piece)
                search_cursor = char_end

            # quality_flags 检测和position定位都要用原始piece(不含文件名前缀),
            # 不然"formula-not-decoded"这类检测和在full_text里找偏移量都会失真。
            quality_flags = detect_chunk_quality_flags(piece)

            chunk_id = f"txt_{chunk_idx:05d}"
            chunk_idx += 1

            # 把文件名(不含路径,比如"HW2.pdf")拼进可索引文本——跟之前修复
            # Caregivers SQL案例(parse_code.py的parse_sql)同一套方案: BM25
            # 查询里如果提到"SQL"这种词,但chunk原文本身不含这个字面词
            # (比如一段纯markdown内容,只是恰好在一份.sql文件里),会导致
            # 检索完全找不到——加上文件名能让文件名里的关键词(比如"sql"、
            # "HW2"这类)也参与匹配,缓解这类词汇鸿沟问题。这个前缀只影响
            # 存进索引/展示给LLM的text字段,不影响上面的质量检测和定位逻辑。
            indexed_text = f"{Path(pdf_rel).name}\n{piece}"

            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=indexed_text,
                    metadata={
                        "source_type": "text",
                        "file": pdf_rel,
                        "course": doc_info["course"],
                        "chunk_type": chunk_type,
                        "name": heading,
                        "position": {
                            "char_start": char_start,
                            "char_end": char_end,
                            "estimated": True,
                            "note": "Docling/LlamaParse输出markdown不含页码标记,用字符偏移量近似定位,不是页码",
                        },
                        "quality": doc_info["quality"],
                        "chunk_quality_flag": quality_flags,
                        "extraction_confidence": "medium",
                        "source_note": doc_info["source_note"],
                    },
                )
            )

    return chunks


# ---------------------------------------------------------------------------
# 测试入口: 挑几个文档跑,不做全量
# ---------------------------------------------------------------------------
TEST_DOCS = [
    (r"ECON 484\HW2\HW2.pdf", "good"),
    (r"ECON 484\Some Notes on Interpreting Predictive Black-box Models.pdf", "good"),
    (r"IEOR E4573 Deep Learning for NLP\HW2\NLPDL_HW_2B_Solutions.pdf", "degraded"),
    (r"IEOR E4106 Stochastic Model\教材\Chapter 6.pdf", "degraded"),
    (r"CSE 416\HW5\LR5.pdf", "llamaparse_done"),
]


def print_chunk_summary(pdf_rel: str, expected_quality: str, chunks: List[TextChunk]):
    print(f"\n{'=' * 100}")
    print(f"文档: {pdf_rel}  (预期quality={expected_quality})")
    print(f"{'=' * 100}")
    print(f"切出 {len(chunks)} 个chunk")

    type_counts = {}
    flag_counts = {}
    headings = []
    for c in chunks:
        type_counts[c.metadata["chunk_type"]] = type_counts.get(c.metadata["chunk_type"], 0) + 1
        flags = c.metadata["chunk_quality_flag"]
        if flags:
            for f in flags:
                flag_counts[f] = flag_counts.get(f, 0) + 1
        if c.metadata["name"] and c.metadata["name"] not in headings:
            headings.append(c.metadata["name"])

    print(f"chunk_type分布: {type_counts}")
    print(f"识别到的章节标题({len(headings)}个): {headings[:10]}{' ...' if len(headings) > 10 else ''}")
    print(f"chunk_quality_flag命中统计: {flag_counts if flag_counts else '(全部为None,没检测到问题)'}")

    print(f"\n--- 前3个chunk完整样例 ---")
    for c in chunks[:3]:
        print(f"\n  [{c.chunk_id}] chunk_type={c.metadata['chunk_type']}  name={c.metadata['name']!r}")
        print(f"  position={c.metadata['position']}")
        print(f"  quality={c.metadata['quality']}  chunk_quality_flag={c.metadata['chunk_quality_flag']}")
        print(f"  text预览: {c.text[:150]!r}")

    flagged_chunks = [c for c in chunks if c.metadata["chunk_quality_flag"]]
    if flagged_chunks:
        print(f"\n--- 命中quality_flag的chunk样例(最多3个) ---")
        for c in flagged_chunks[:3]:
            print(f"\n  [{c.chunk_id}] flag={c.metadata['chunk_quality_flag']}")
            print(f"  text预览: {c.text[:200]!r}")


if __name__ == "__main__":
    docs = load_doc_index()
    print(f"manifest里共有 {len(docs)} 个可定位到markdown文件的文档")

    for pdf_rel, expected_quality in TEST_DOCS:
        if pdf_rel not in docs:
            print(f"\n[跳过] manifest里找不到: {pdf_rel}")
            continue
        doc_info = docs[pdf_rel]
        chunks = chunk_document(pdf_rel, doc_info)
        print_chunk_summary(pdf_rel, expected_quality, chunks)
