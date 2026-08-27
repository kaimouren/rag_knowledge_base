"""
文本知识类解析测试脚本(Docling)

CLAUDE.md "文件分类与处理策略 -> 1. 文本知识类" 规定:
常规 pdf/docx/pptx/txt/md/rmd/html 走 Docling(本地开源)解析为 markdown。
本脚本先只跑 5 个 PDF 样本,把 Docling 的解析结果打印出来,用于人工评估解析质量,
再决定要不要加 LlamaParse 兜底路由(复杂扫描件/多栏版面)。

用法:
    python ingest/parse_text.py
"""

import time
from pathlib import Path

from docling.document_converter import DocumentConverter

DATA_ROOT = Path(r"E:\project")

# 从"文本知识类"里挑的 5 个 PDF 样本,覆盖不同版面类型,方便对比 Docling 表现:
#   - 手写/扫描作业提交(单栏,可能含手写体)
#   - Colab notebook 导出(代码块 + 文本混排)
#   - 考试卷(纯文本,格式规整)
#   - arXiv 论文(双栏学术排版,公式密集)
#   - 大文件扫描讲义(23MB,测试大文件/可能的扫描件表现)
SAMPLE_PDFS = [
    DATA_ROOT / "CSE 414" / "100719939" / "Part1 copy.pdf",
    DATA_ROOT / "CSE 416" / "416场外援助" / "Copy of HW5 Deep Learning - Colaboratory.pdf",
    DATA_ROOT / "ECON 484" / "484_Final.pdf",
    DATA_ROOT / "IEOR E4004 Optimization Models & Methods" / "2202.00138.pdf",
    DATA_ROOT / "IEOR E4106 Stochastic Model" / "1 Review of Conditional Probability.pdf",
]

OUTPUT_DIR = Path(__file__).parent / "_test_output"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    converter = DocumentConverter()

    for i, pdf_path in enumerate(SAMPLE_PDFS, 1):
        print("=" * 100)
        print(f"[{i}/{len(SAMPLE_PDFS)}] {pdf_path.relative_to(DATA_ROOT)}")
        print("=" * 100)

        if not pdf_path.exists():
            print(f"[跳过] 文件不存在: {pdf_path}")
            continue

        start = time.time()
        try:
            result = converter.convert(pdf_path)
            markdown = result.document.export_to_markdown()
        except Exception as e:
            print(f"[解析失败] {type(e).__name__}: {e}")
            continue
        elapsed = time.time() - start

        out_file = OUTPUT_DIR / f"{i:02d}_{pdf_path.stem}.md"
        out_file.write_text(markdown, encoding="utf-8")

        print(f"耗时 {elapsed:.1f}s | 输出 {len(markdown)} 字符 | 已保存至 {out_file}")
        print("-" * 100)
        print(markdown)
        print()


if __name__ == "__main__":
    main()
