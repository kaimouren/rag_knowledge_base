"""
LlamaParse 小范围验证(阶段一: 只跑3-5个文件,不批量)

背景: quality_router 判定"刚需课程"(ieor 4742 deep learning + CSE 416)里有
54个文件 Docling 解析质量不够,计划转 LlamaParse。真正批量跑之前,先挑几个
有代表性的文件做小范围验证,确认三件事:
  1) 公式是否真的被解析成可读的LaTeX(这是转LlamaParse最核心的目的)
  2) 扫描件/乱码问题是否真的改善了
  3) 消耗的credits是否符合预期(页数 x 3,Cost-effective档位)

用的是 tier="cost_effective"(3 credits/页),不是更贵的 agentic(10)/
agentic_plus(45) —— 这个参数值是从 LlamaParse v2 官方文档确认的,不是猜的。

API key 从项目根目录的 .env 读取(LLAMA_CLOUD_API_KEY),不硬编码在代码里,
.env 已经加进 .gitignore,不会被误提交。

用法:
    python ingest/llamaparse_test.py
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from llama_cloud import LlamaCloud

DATA_ROOT = Path(r"E:\project")
OUTPUT_DIR = Path(__file__).parent / "_test_output" / "llamaparse_test"
CREDITS_PER_PAGE = 3
TIER = "cost_effective"

# 精选4个文件,覆盖两种失败模式,页数都不大,控制这次验证的成本
# (总计27页,预计81 credits,对10000额度基本无感,但足够看出效果差异):
#   - 纯公式密集: LR3.pdf(公式密度8.47/千字符,全批里最高)
#   - 纯乱码/OCR失败: LR2.pdf, LR1.pdf(短行碎片化40%+)
#   - 两种问题都有: LR5.pdf
TEST_FILES = [
    (r"CSE 416\HW3\LR3.pdf", r"_test_output\stage2\0117_LR3.md", "公式密集"),
    (r"CSE 416\Class4\LR2.pdf", r"_test_output\stage2\0082_LR2.md", "乱码/OCR失败"),
    (r"CSE 416\CSE416✅\LR1.pdf", r"_test_output\stage2\0109_LR1.md", "乱码/OCR失败"),
    (r"CSE 416\HW5\LR5.pdf", r"_test_output\stage1\0010_LR5.md", "公式密集+乱码"),
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = LlamaCloud()  # 从环境变量 LLAMA_CLOUD_API_KEY 读取

    total_pages = 0
    print(f"=== LlamaParse 小范围验证: {len(TEST_FILES)} 个文件, tier={TIER} ===\n")

    for rel_path, docling_md_rel, category in TEST_FILES:
        pdf_path = DATA_ROOT / rel_path
        print(f"[{category}] {rel_path}")

        start = time.time()
        try:
            uploaded = client.files.create(file=str(pdf_path), purpose="parse")
            result = client.parsing.parse(
                file_id=uploaded.id,
                tier=TIER,
                version="latest",
                expand=["markdown"],
            )
        except Exception as e:
            print(f"  [失败] {type(e).__name__}: {e}\n")
            continue
        elapsed = time.time() - start

        pages = result.markdown.pages
        page_count = len(pages)
        total_pages += page_count
        markdown = "\n\n".join(p.markdown for p in pages)

        safe_name = Path(rel_path).stem
        out_path = OUTPUT_DIR / f"{safe_name}_llamaparse.md"
        out_path.write_text(markdown, encoding="utf-8")

        docling_md_path = Path(__file__).parent / docling_md_rel

        print(f"  耗时 {elapsed:.1f}s | {page_count}页 | 预计credits={page_count * CREDITS_PER_PAGE} "
              f"| 已保存至 {out_path.relative_to(Path(__file__).parent)}")
        print(f"  对比文件: Docling={docling_md_path.relative_to(Path(__file__).parent)} vs LlamaParse={out_path.name}")
        print()

    expected_credits = total_pages * CREDITS_PER_PAGE
    print(f"=== 汇总 ===")
    print(f"总页数: {total_pages}")
    print(f"预计消耗credits(页数x3): {expected_credits}")
    print(f"请去 LlamaCloud 控制台核对实际扣费是否等于这个数字,以此验证credits计费口径是否符合预期。")


if __name__ == "__main__":
    main()
