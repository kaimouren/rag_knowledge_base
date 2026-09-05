"""
文本知识类chunking(MinerU版): 用parse_text_mineru.py产出的content_list.json
(每个内容块自带精确page_idx)替代之前"字符偏移量估算"的位置信息,给每个
chunk标注真实的页码范围,而不是estimated=true的近似值。

两层切分算法本身跟chunk_text.py(Docling/LlamaParse版)完全一样,复用同一个
split_into_pieces()——不重新发明切分逻辑,只是这一版能把切出来的每个片段
准确定位到第几页。

页码怎么算出来的(build_page_anchors): content_list.json里的每个文字类
块(type=="text"/"equation"/"code")都带着自己的page_idx(从0开始,MinerU
模型直接输出的,不是猜的)。用块内容的前40个字符作为probe,在.md全文里
按顺序定位,找到就记一个(char_pos, page_idx)锚点。之后对chunk切出来的
任意char_start/char_end,找"锚点里pos<=这个偏移量的最后一个"对应的page_idx
就是这段文字所在的页——不是逐字符精确到"这个chunk横跨第3-4页的具体哪个
字符换的页",但块级别的粒度(实测每份文档能定位98%+的文字块)对700字符量级
的chunk来说已经足够细。

用法:
    python ingest/chunk_text_mineru.py --test    # 只跑mineru_parse_report.csv
                                                   # 里已经成功解析的文件,先看结果
    python ingest/chunk_text_mineru.py            # 全量跑(等parse_text_mineru.py
                                                   # 全量跑完之后)
"""

import argparse
import bisect
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from chunk_text import TextChunk, split_into_pieces, detect_chunk_quality_flags

INGEST_DIR = Path(__file__).parent
REPORT_CSV = INGEST_DIR / "_test_output" / "mineru_parse_report.csv"
BATCH_DIR = INGEST_DIR / "_test_output" / "mineru_batch"
OUTPUT_JSON = INGEST_DIR / "_test_output" / "text_chunks_mineru.json"

PROBE_LEN = 40

# 公式可信度风险标记(两个独立信号,不是同一件事的两种叫法):
#
# 1) formula_low_confidence: MinerU的middle.json给每个"inline_equation"
#    (行内公式)span都标了识别置信度score,但"interline_equation"(独立成行/
#    多行的大公式块,对应content_list.json里的"equation"类型block)一个都
#    没有score——这是实测确认的、MinerU pipeline backend本身的结构性
#    局限,不是解析失败(见下方load_inline_formula_min_scores的实测结果:
#    全库439个文档随机抽15个,624个inline_equation span全部有score,
#    252个interline_equation span全部没有score,0/252)。
#    阈值0.6是全库29473个inline公式score分布的约第12百分位(min=0.45,
#    median=0.87),没有天然的双峰分界点,选一个偏保守的整数分位点,
#    不是精调出来的。
#
# 2) formula_unverifiable: 这次人工抽查发现,真正的"格式规整但语义错误"
#    风险几乎全部出在interline公式上(见LIMITATIONS.md"公式可信度风险"
#    一节的抽查记录:一份手写笔记文档里人工核对11个公式,4个语义错误但
#    格式规整——这类"看起来对但是错的"全部是interline类型;而inline
#    公式基本能通过score检测出来)。既然interline公式完全没有score信号,
#    这个标记不是"低置信度",而是如实标注"这个chunk包含一段完全无法
#    自动判断对错的公式内容"——不是解决问题,是让下游知道这里有盲区。
FORMULA_LOW_CONFIDENCE_THRESHOLD = 0.6


def load_doc_index():
    rows = list(csv.DictReader(open(REPORT_CSV, encoding="utf-8-sig")))
    docs = {}
    for r in rows:
        if r["status"] != "ok":
            continue
        idx = int(r["idx"])
        stem = f"{idx:05d}"
        docs[r["file"]] = {
            "idx": idx,
            "md_path": BATCH_DIR / stem / "ocr" / f"{stem}.md",
            "content_list_path": BATCH_DIR / stem / "ocr" / f"{stem}_content_list.json",
            "middle_json_path": BATCH_DIR / stem / "ocr" / f"{stem}_middle.json",
            "course": r["course"],
            "old_quality": r["old_quality"],
        }
    return docs


def load_inline_formula_min_scores(middle_json_path: Path) -> dict:
    """扫middle.json里所有inline_equation span,返回每页最低score
    (dict: page_idx -> min_score)。interline_equation span实测(见上方
    模块常量注释)全部没有score字段,这里天然不会统计到它们,不是遗漏。
    """
    try:
        data = json.loads(middle_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    min_scores = {}
    for page in data.get("pdf_info", []):
        page_idx = page.get("page_idx")
        for block in page.get("preproc_blocks", []):
            for b in block.get("blocks", [block]):
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("type") == "inline_equation" and "score" in span:
                            prev = min_scores.get(page_idx)
                            if prev is None or span["score"] < prev:
                                min_scores[page_idx] = span["score"]
    return min_scores


def pages_with_equation_blocks(content_list: list) -> set:
    """返回含有interline/display公式(content_list里的"equation"类型block)
    的page_idx集合——这些完全没有置信度信号(见上方模块常量注释),
    只能标"存在、无法自动判断对错",不能标"低/高置信度"。
    """
    return {b["page_idx"] for b in content_list if b.get("type") == "equation"}


def build_page_anchors(full_text: str, content_list: list):
    """返回 (anchors, skipped) —— anchors是按char_pos升序排列的(char_pos,page_idx)
    列表, skipped是content_list里有文字但在.md全文里没找到对应位置的块数
    (不是bug,是OCR/公式转写偶尔会让markdown导出跟content_list原文轻微不一致,
    找不到就跳过这个块,不影响其它块的锚点,也不影响chunk本身的切分)。
    """
    anchors = []
    cursor = 0
    skipped = 0

    for block in content_list:
        btype = block.get("type")
        if btype in ("text", "equation") and block.get("text", "").strip():
            text = block["text"]
        elif btype == "code" and block.get("code_body", "").strip():
            text = block["code_body"]
        else:
            continue

        probe = text[:PROBE_LEN].strip()
        if not probe:
            continue

        pos = full_text.find(probe, cursor)
        if pos == -1:
            pos = full_text.find(probe)
        if pos == -1:
            skipped += 1
            continue

        anchors.append((pos, block["page_idx"]))
        cursor = pos + len(probe)

    anchors.sort(key=lambda a: a[0])
    return anchors, skipped


def page_at(anchors, char_offset: int):
    if not anchors:
        return None
    positions = [a[0] for a in anchors]
    i = bisect.bisect_right(positions, char_offset) - 1
    if i < 0:
        return anchors[0][1]
    return anchors[i][1]


def chunk_document(pdf_rel: str, doc_info: dict):
    full_text = doc_info["md_path"].read_text(encoding="utf-8")
    content_list = json.loads(doc_info["content_list_path"].read_text(encoding="utf-8"))
    anchors, skipped_anchors = build_page_anchors(full_text, content_list)

    inline_min_scores = load_inline_formula_min_scores(doc_info["middle_json_path"])
    equation_pages = pages_with_equation_blocks(content_list)

    chunks = []
    for item in split_into_pieces(full_text):
        piece = item["piece"]
        char_start, char_end = item["char_start"], item["char_end"]

        page_start = page_at(anchors, char_start)
        page_end = page_at(anchors, max(char_start, char_end - 1))

        quality_flags = detect_chunk_quality_flags(piece) or []
        if page_start is not None:
            chunk_pages = range(page_start, page_end + 1)
            if equation_pages.intersection(chunk_pages):
                quality_flags = quality_flags + ["formula_unverifiable"]
            page_scores = [inline_min_scores[p] for p in chunk_pages if p in inline_min_scores]
            if page_scores and min(page_scores) < FORMULA_LOW_CONFIDENCE_THRESHOLD:
                quality_flags = quality_flags + ["formula_low_confidence"]
        quality_flags = quality_flags or None

        indexed_text = f"{Path(pdf_rel).name}\n{piece}"

        chunks.append(TextChunk(
            chunk_id="",  # 全局编号在merge阶段统一分配,这里先留空
            text=indexed_text,
            metadata={
                "source_type": "text",
                "file": pdf_rel,
                "course": doc_info["course"],
                "chunk_type": item["chunk_type"],
                "name": item["heading"],
                "position": {
                    "page_start": page_start,
                    "page_end": page_end,
                    "char_start": char_start,
                    "char_end": char_end,
                    "estimated": page_start is None,
                    "note": (
                        "页码来自MinerU content_list.json的page_idx(0-based),"
                        "按块级锚点定位,不是字符偏移量估算"
                        if page_start is not None else
                        "该文档在.md全文里找不到任何可用锚点(极少数情况),退化为无页码"
                    ),
                },
                "source_citation": (
                    f"{doc_info['course']} / {Path(pdf_rel).name}"
                    + (f" p.{page_start + 1}" if page_start is not None and page_start == page_end
                       else f" p.{page_start + 1}-{page_end + 1}" if page_start is not None
                       else "")
                ),
                "parsed_by": "mineru",
                "previous_docling_quality": doc_info["old_quality"],
                "chunk_quality_flag": quality_flags,
                "extraction_confidence": "medium",
            },
        ))

    return chunks, skipped_anchors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="只跑前5个文档,打印结果不写文件")
    args = parser.parse_args()

    docs = load_doc_index()
    print(f"可chunking文档数(mineru解析成功的): {len(docs)}")

    items = list(docs.items())
    if args.test:
        items = items[:5]

    all_chunks = []
    total_skipped = 0
    for pdf_rel, doc_info in items:
        chunks, skipped = chunk_document(pdf_rel, doc_info)
        total_skipped += skipped
        all_chunks.extend(chunks)

        if args.test:
            print(f"\n{'=' * 90}\n{pdf_rel}  (course={doc_info['course']})")
            print(f"切出 {len(chunks)} 个chunk, 锚点缺失块数 {skipped}")
            for c in chunks[:3]:
                pos = c.metadata["position"]
                print(f"  [{c.metadata['chunk_type']}] page {pos['page_start']}-{pos['page_end']} "
                      f"| {c.metadata['source_citation']} | {c.text[:60]!r}")

    # 全局重新编号,跟chunk_text.py的txt_00000规则保持一致空间(避免和旧chunk_id撞车,
    # 这里换个前缀txtm_,合并阶段再决定最终编号方案)
    for i, c in enumerate(all_chunks):
        c.chunk_id = f"txtm_{i:05d}"

    if not args.test:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in all_chunks], f, ensure_ascii=False, indent=2)
        print(f"\n已写入 {OUTPUT_JSON}: {len(all_chunks)} 个chunk, "
              f"锚点缺失块数合计 {total_skipped}")
    else:
        print(f"\n[测试模式,未写文件] 累计 {len(all_chunks)} 个chunk, 锚点缺失块数合计 {total_skipped}")


if __name__ == "__main__":
    main()
