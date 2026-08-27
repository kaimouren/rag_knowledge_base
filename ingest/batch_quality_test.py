"""
批量抽样测试: Docling 解析 + quality_router 路由判断

背景: parse_text.py / quality_router.py 目前只在人工精选的 5 个 PDF 上验证过。
这个脚本把验证范围扩大到"文本知识类"里随机抽取的 30-50 个 PDF,覆盖不同
课程、不同文件大小,检验 quality_router 的判断阈值在更大样本下是否还站得住,
而不是只在 5 个精心挑选的样本上凑巧对。

抽样策略: 按课程(顶层目录)分层 + 按文件大小分层(小/中/大),
用固定随机种子(SAMPLE_SEED)保证可复现 —— CLAUDE.md 里"可复现性"原则要求
相同输入产出一致结果,方便你后续复查同一批样本。

用法:
    python ingest/batch_quality_test.py
"""

import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from route_files import EXCLUDE_DIRS
from quality_router import FileInfo, check_quality

DATA_ROOT = Path(r"E:\project")
OUTPUT_DIR = Path(__file__).parent / "_test_output" / "batch"
REPORT_CSV = Path(__file__).parent / "_test_output" / "batch_report.csv"

SAMPLE_SEED = 42
TARGET_SAMPLE_SIZE = 40
MIN_PER_COURSE = 3  # 每门课至少抽到几个,避免样本多的课程把小课程挤没


def collect_all_pdfs():
    pdfs = []
    for p in DATA_ROOT.rglob("*.pdf"):
        if not p.is_file() or p.name.startswith("._"):
            continue
        rel_parts = p.relative_to(DATA_ROOT).parts
        if any(part in EXCLUDE_DIRS for part in rel_parts[:-1]):
            continue
        pdfs.append(p)
    return pdfs


def size_bucket(size_bytes: int) -> str:
    if size_bytes < 300_000:
        return "small"
    if size_bytes < 3_000_000:
        return "medium"
    return "large"


def stratified_sample(pdfs, target_size: int, seed: int):
    """按课程分层抽样,课程内部再按大小分桶抽样,尽量覆盖不同来源和体量。"""
    rng = random.Random(seed)

    by_course = {}
    for p in pdfs:
        course = p.relative_to(DATA_ROOT).parts[0]
        by_course.setdefault(course, []).append(p)

    n_courses = len(by_course)
    base_quota = max(MIN_PER_COURSE, target_size // n_courses)

    sampled = []
    for course, files in by_course.items():
        quota = min(base_quota, len(files))
        buckets = {"small": [], "medium": [], "large": []}
        for f in files:
            buckets[size_bucket(f.stat().st_size)].append(f)

        # 尽量从每个大小桶里各抽一部分,桶内随机
        picked = []
        bucket_names = [b for b in buckets if buckets[b]]
        per_bucket = max(1, quota // max(1, len(bucket_names)))
        for b in bucket_names:
            rng.shuffle(buckets[b])
            picked.extend(buckets[b][:per_bucket])
        # 配额没抽满就从剩余文件里随机补
        remaining = quota - len(picked)
        if remaining > 0:
            leftover = [f for f in files if f not in picked]
            rng.shuffle(leftover)
            picked.extend(leftover[:remaining])

        sampled.extend(picked[:quota])

    rng.shuffle(sampled)
    return sampled[:target_size] if len(sampled) > target_size else sampled


def get_page_count(pdf_path: Path):
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(pdf_path)).pages)
    except Exception:
        return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_pdfs = collect_all_pdfs()
    print(f"文本知识类 PDF 总数: {len(all_pdfs)}")

    sample = stratified_sample(all_pdfs, TARGET_SAMPLE_SIZE, SAMPLE_SEED)
    print(f"本次抽样: {len(sample)} 个文件(种子={SAMPLE_SEED},可复现)\n")

    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    rows = []

    for i, pdf_path in enumerate(sample, 1):
        rel = pdf_path.relative_to(DATA_ROOT)
        print(f"[{i}/{len(sample)}] {rel}")

        start = time.time()
        try:
            result = converter.convert(pdf_path)
            markdown = result.document.export_to_markdown()
        except Exception as e:
            print(f"  [解析失败] {type(e).__name__}: {e}")
            rows.append(
                {
                    "index": i,
                    "course": rel.parts[0],
                    "file": str(rel),
                    "size_kb": round(pdf_path.stat().st_size / 1024, 1),
                    "pages": "",
                    "elapsed_s": "",
                    "needs_llamaparse": "PARSE_ERROR",
                    "reasons": f"{type(e).__name__}: {e}",
                    "garbled_ratio": "",
                    "formula_density": "",
                    "output_density": "",
                    "md_file": "",
                }
            )
            continue
        elapsed = time.time() - start

        safe_name = f"{i:03d}_{pdf_path.stem}"[:120]
        md_path = OUTPUT_DIR / f"{safe_name}.md"
        md_path.write_text(markdown, encoding="utf-8")

        page_count = get_page_count(pdf_path)
        info = FileInfo(
            path=pdf_path,
            file_size_bytes=pdf_path.stat().st_size,
            page_count=page_count,
        )
        decision = check_quality(info, markdown)

        rows.append(
            {
                "index": i,
                "course": rel.parts[0],
                "file": str(rel),
                "size_kb": round(pdf_path.stat().st_size / 1024, 1),
                "pages": page_count or "",
                "elapsed_s": round(elapsed, 1),
                "needs_llamaparse": decision.needs_llamaparse,
                "reasons": "; ".join(decision.reasons),
                "garbled_ratio": decision.metrics.get("garbled_ratio"),
                "formula_density": decision.metrics.get(
                    "formula_density_per_1k_chars"
                ),
                "output_density": decision.metrics.get("output_density"),
                "md_file": str(md_path.relative_to(Path(__file__).parent)),
            }
        )
        verdict = "转LlamaParse" if decision.needs_llamaparse else "Docling可用"
        print(f"  {verdict} | 耗时{elapsed:.1f}s | {decision.reasons}")

    with open(REPORT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    flagged = sum(1 for r in rows if r["needs_llamaparse"] is True)
    errors = sum(1 for r in rows if r["needs_llamaparse"] == "PARSE_ERROR")
    print(f"\n完成: {len(rows)} 个文件, {flagged} 个标记转LlamaParse, {errors} 个解析报错")
    print(f"完整报告已保存至: {REPORT_CSV}")


if __name__ == "__main__":
    main()
