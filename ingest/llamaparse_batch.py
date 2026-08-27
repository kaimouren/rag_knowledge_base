"""
LlamaParse 批量处理: 刚需课程(ieor 4742 deep learning + CSE 416)剩余的
50个文件(4个已经在 llamaparse_test.py 里跑过并验证效果达标,这里排除)。

预计: 3162页 x 3 credits/页(cost_effective档) = 9486 credits,加上已消耗
的81,总计9567,在10000免费额度内,留433 credits缓冲。

工程要求(跟之前 Docling 批处理一致的三点):
  1) try/except 包住单文件处理,某个文件解析失败不中断整批(尤其是真金白银
     在消耗credits,更不能因为中途崩溃导致状态不明、不知道扣了多少费)
  2) 每处理完一个文件立刻写入CSV并flush,防止中途中断丢结果
  3) 显示进度,同时累计打印"已消耗credits"方便随时跟预算比对

用法:
    python ingest/llamaparse_batch.py
"""

import csv
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from llama_cloud import LlamaCloud

DATA_ROOT = Path(r"E:\project")
OUTPUT_DIR = Path(__file__).parent / "_test_output" / "llamaparse_batch"
REPORT_CSV = Path(__file__).parent / "_test_output" / "llamaparse_batch_report.csv"

CREDITS_PER_PAGE = 3
TIER = "cost_effective"
PRIORITY_COURSES = {"ieor 4742 deep learning", "CSE 416"}

ALREADY_DONE = {
    r"CSE 416\HW3\LR3.pdf",
    r"CSE 416\Class4\LR2.pdf",
    r"CSE 416\CSE416✅\LR1.pdf",
    r"CSE 416\HW5\LR5.pdf",
}

FIELDNAMES = ["course", "file", "pages", "credits", "elapsed_s", "md_file", "error"]


def collect_targets():
    rows = list(csv.DictReader(open(
        Path(__file__).parent / "_test_output" / "final_report.csv", encoding="utf-8-sig"
    )))
    targets = [
        r for r in rows
        if r["needs_llamaparse"] == "True"
        and r["course"] in PRIORITY_COURSES
        and r["file"] not in ALREADY_DONE
    ]
    return targets


def process_one(client, row) -> dict:
    rel_path = row["file"]
    pdf_path = DATA_ROOT / rel_path
    out = {
        "course": row["course"],
        "file": rel_path,
        "pages": "",
        "credits": "",
        "elapsed_s": "",
        "md_file": "",
        "error": "",
    }
    start = time.time()
    try:
        uploaded = client.files.create(file=str(pdf_path), purpose="parse")
        result = client.parsing.parse(
            file_id=uploaded.id,
            tier=TIER,
            version="latest",
            expand=["markdown"],
        )
        pages = result.markdown.pages
        page_count = len(pages)
        markdown = "\n\n".join(p.markdown for p in pages)

        safe_name = Path(rel_path).stem
        # 同名文件可能出现在不同文件夹(比如多份 vaccine-scheduler-java-main 的副本),
        # 用课程+文件名前缀区分,避免互相覆盖
        out_path = OUTPUT_DIR / f"{row['course']}__{safe_name}.md".replace("/", "_").replace("\\", "_")
        out_path.write_text(markdown, encoding="utf-8")

        out["pages"] = page_count
        out["credits"] = page_count * CREDITS_PER_PAGE
        out["md_file"] = str(out_path.relative_to(Path(__file__).parent))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"

    out["elapsed_s"] = round(time.time() - start, 1)
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = collect_targets()
    print(f"待处理文件数: {len(targets)}", flush=True)

    client = LlamaCloud()
    cumulative_credits = 0
    error_count = 0
    start_all = time.time()

    with open(REPORT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        f.flush()

        for i, row in enumerate(targets, 1):
            result = process_one(client, row)
            writer.writerow(result)
            f.flush()

            if result["error"]:
                error_count += 1
                print(f"  [{i}/{len(targets)}] [失败] {result['file']}: {result['error']}", flush=True)
            else:
                cumulative_credits += result["credits"]
                print(
                    f"  [{i}/{len(targets)}] {result['file']} | {result['pages']}页 | "
                    f"{result['elapsed_s']}s | 累计credits={cumulative_credits}",
                    flush=True,
                )

    total_elapsed = time.time() - start_all
    print(f"\n批量处理完成: {len(targets)}个文件, 总耗时 {total_elapsed:.1f}s ({total_elapsed/60:.1f}分钟)", flush=True)
    print(f"成功: {len(targets) - error_count}, 失败: {error_count}", flush=True)
    print(f"本批累计消耗credits: {cumulative_credits}", flush=True)
    print(f"加上之前小范围验证的81 credits, 总计: {cumulative_credits + 81} credits", flush=True)


if __name__ == "__main__":
    main()
