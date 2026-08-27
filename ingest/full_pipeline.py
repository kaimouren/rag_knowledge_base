"""
quality_router 全流程批处理: Docling解析 + 质量路由,覆盖"文本知识类"全部PDF

分两阶段跑(用户要求):
  阶段一 (python full_pipeline.py 1): 抽50个新样本(不含之前 batch_quality_test.py
      已测试过的36个)先验证稳定性(耗时/异常/分类分布),避免拿全量数据练手却在
      中途才发现某个环节有问题。
  阶段二 (python full_pipeline.py 2): 阶段一确认没问题后,处理剩余全部PDF。

工程考虑(对应用户提出的三点要求):
  1) 超时/崩溃保护: Docling.convert() 放在独立子进程里跑,外层用
     concurrent.futures 加超时(PER_FILE_TIMEOUT)。单个文件的转换异常
     (文件损坏等)会被 try/except 捕获记录,不会导致整批中断;如果是真的
     卡死超时,会强制终止那个卡住的子进程并重启一个新的,后续文件不受影响。
  2) 增量写入: 每处理完一个文件立刻 writerow + flush,不是攒到最后一次性写,
     中途中断(比如手动Ctrl+C、断电)已处理的结果不会丢。
  3) 进度可见: 每处理完一个文件都打印一行,异常单独高亮;每10个额外打印
     一次简短进度汇总。

阶段一/二互不重跑已经处理过的文件: 会读取 batch_report.csv(第一批36个)和
stage1_report.csv,自动排除,确保 445 个 PDF 总共只被 Docling 解析一次,
不浪费算力。
"""

import csv
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DATA_ROOT = Path(r"E:\project")
OUTPUT_DIR = Path(__file__).parent / "_test_output"

# 阈值依据: 之前36个样本里最长耗时是112.9s(CNNs.pdf, 71页图多的PPT),
# 180s留了约1.6倍安全边际,既不会因为正常的大文件被误杀,也不会让真正
# 卡死的文件无限占用整个批次的时间。
PER_FILE_TIMEOUT = 180

CREDITS_PER_PAGE = 3
FREE_QUOTA = 10000

FIELDNAMES = [
    "course",
    "file",
    "size_kb",
    "pages",
    "elapsed_s",
    "needs_llamaparse",
    "reasons",
    "garbled_ratio",
    "fragmentation_ratio",
    "formula_density",
    "output_density",
    "md_file",
    "error",
]


def collect_all_pdfs():
    from route_files import EXCLUDE_DIRS

    pdfs = []
    for p in DATA_ROOT.rglob("*.pdf"):
        if not p.is_file() or p.name.startswith("._"):
            continue
        rel_parts = p.relative_to(DATA_ROOT).parts
        if any(part in EXCLUDE_DIRS for part in rel_parts[:-1]):
            continue
        pdfs.append(p)
    return pdfs


def files_in_csv(csv_path: Path) -> set:
    tested = set()
    if csv_path.exists():
        with open(csv_path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                tested.add(row["file"])
    return tested


def size_bucket(size_bytes: int) -> str:
    if size_bytes < 300_000:
        return "small"
    if size_bytes < 3_000_000:
        return "medium"
    return "large"


def stratified_sample(pdfs, target_size, seed, exclude_rel_paths):
    """跟 batch_quality_test.py 同样的分层抽样逻辑(按课程+文件大小),
    额外排除已经测试过的文件,避免重复解析。"""
    rng = random.Random(seed)
    pdfs = [p for p in pdfs if str(p.relative_to(DATA_ROOT)) not in exclude_rel_paths]

    by_course = {}
    for p in pdfs:
        course = p.relative_to(DATA_ROOT).parts[0]
        by_course.setdefault(course, []).append(p)

    n_courses = max(1, len(by_course))
    base_quota = max(1, target_size // n_courses)

    sampled = []
    for course, files in by_course.items():
        quota = min(base_quota, len(files))
        buckets = {"small": [], "medium": [], "large": []}
        for f in files:
            buckets[size_bucket(f.stat().st_size)].append(f)
        picked = []
        names = [b for b in buckets if buckets[b]]
        per_bucket = max(1, quota // max(1, len(names))) if names else 0
        for b in names:
            rng.shuffle(buckets[b])
            picked.extend(buckets[b][:per_bucket])
        remaining = quota - len(picked)
        if remaining > 0:
            leftover = [f for f in files if f not in picked]
            rng.shuffle(leftover)
            picked.extend(leftover[:remaining])
        sampled.extend(picked[:quota])

    rng.shuffle(sampled)
    return sampled[:target_size] if len(sampled) > target_size else sampled


# ---------------------------------------------------------------------------
# 子进程 worker: 把 Docling 转换隔离在独立进程里,这样超时/崩溃只影响这一个
# 文件,不会拖垮整个批处理。DocumentConverter() 只在 worker 启动时初始化一次
# (放在 initializer 里),避免每个文件都重新加载一遍模型。
# ---------------------------------------------------------------------------
_worker_converter = None


def _init_worker():
    global _worker_converter
    from docling.document_converter import DocumentConverter

    _worker_converter = DocumentConverter()


def _convert_in_worker(pdf_path_str: str) -> str:
    result = _worker_converter.convert(pdf_path_str)
    return result.document.export_to_markdown()


def get_page_count(pdf_path: Path):
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(pdf_path)).pages)
    except Exception:
        return None


def _kill_and_restart(executor: ProcessPoolExecutor) -> ProcessPoolExecutor:
    """强制终止卡死的 worker 子进程并重建一个新的 executor。
    ProcessPoolExecutor.shutdown() 本身不会打断正在运行的任务,所以需要
    直接拿到底层子进程对象调用 terminate()(用的是内部属性 _processes,
    没有更干净的公开API,但这是社区里处理"超时后真正杀掉进程"的通用做法)。
    """
    for p in list(executor._processes.values()):
        p.terminate()
    executor.shutdown(wait=False)
    return ProcessPoolExecutor(max_workers=1, initializer=_init_worker)


def process_file(executor, pdf_path: Path, md_output_dir: Path, idx: int):
    from quality_router import FileInfo, check_quality

    rel = pdf_path.relative_to(DATA_ROOT)
    row = {k: "" for k in FIELDNAMES}
    row["course"] = rel.parts[0]
    row["file"] = str(rel)
    row["size_kb"] = round(pdf_path.stat().st_size / 1024, 1)

    start = time.time()
    needs_restart = False
    try:
        future = executor.submit(_convert_in_worker, str(pdf_path))
        markdown = future.result(timeout=PER_FILE_TIMEOUT)
    except FutureTimeoutError:
        row["elapsed_s"] = round(time.time() - start, 1)
        # 修订(阶段一实测后发现的问题): 之前把超时文件记成 ERROR,会导致它们
        # 的页数没被统计进最终的credits预算——阶段一里两个超时文件事后查证
        # 分别是128页和138页,不是文件损坏,就是单纯很大导致Docling处理不完。
        # Docling 3分钟处理不完本身就是"这份文件需要更强处理能力"的信号,
        # 直接判定为需要转LlamaParse,页数用pypdf独立读取(不依赖Docling,
        # 所以超时也能拿到),这样最终报告不会漏掉这些大文件的credits开销。
        row["pages"] = get_page_count(pdf_path) or ""
        row["needs_llamaparse"] = True
        row["reasons"] = f"Docling处理超时(>{PER_FILE_TIMEOUT}s),大概率是超大/超复杂文件,直接判定需要转LlamaParse"
        row["error"] = f"TIMEOUT (>{PER_FILE_TIMEOUT}s)"
        return row, True
    except Exception as e:
        row["elapsed_s"] = round(time.time() - start, 1)
        row["needs_llamaparse"] = "ERROR"
        row["error"] = f"{type(e).__name__}: {e}"
        return row, needs_restart

    row["elapsed_s"] = round(time.time() - start, 1)

    try:
        safe_name = f"{idx:04d}_{pdf_path.stem}"[:120]
        md_path = md_output_dir / f"{safe_name}.md"
        md_path.write_text(markdown, encoding="utf-8")
        row["md_file"] = str(md_path.relative_to(Path(__file__).parent))

        page_count = get_page_count(pdf_path)
        info = FileInfo(
            path=pdf_path,
            file_size_bytes=pdf_path.stat().st_size,
            page_count=page_count,
        )
        decision = check_quality(info, markdown)

        row["pages"] = page_count or ""
        row["needs_llamaparse"] = decision.needs_llamaparse
        row["reasons"] = "; ".join(decision.reasons)
        row["garbled_ratio"] = decision.metrics.get("garbled_ratio")
        row["fragmentation_ratio"] = decision.metrics.get("fragmentation_ratio")
        row["formula_density"] = decision.metrics.get("formula_density_per_1k_chars")
        row["output_density"] = decision.metrics.get("output_density")
    except Exception as e:
        row["needs_llamaparse"] = "ERROR"
        row["error"] = f"post-process {type(e).__name__}: {e}"

    return row, needs_restart


def run_stage(files, csv_path: Path, md_output_dir: Path, stage_name: str):
    md_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {stage_name}: {len(files)} 个文件 ===", flush=True)

    executor = ProcessPoolExecutor(max_workers=1, initializer=_init_worker)
    start_all = time.time()
    error_count = 0
    timeout_count = 0
    flagged_count = 0

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        f.flush()

        for i, pdf_path in enumerate(files, 1):
            row, needs_restart = process_file(executor, pdf_path, md_output_dir, i)
            writer.writerow(row)
            f.flush()

            is_timeout = "TIMEOUT" in row.get("error", "")
            if is_timeout:
                timeout_count += 1
                flagged_count += 1  # 超时按需要转LlamaParse计入
                print(f"  [{i}/{len(files)}] [超时,判定转LlamaParse] {row['file']}: {row['error']}", flush=True)
            elif row["needs_llamaparse"] == "ERROR":
                error_count += 1
                print(f"  [{i}/{len(files)}] [异常] {row['file']}: {row['error']}", flush=True)
            else:
                if row["needs_llamaparse"] is True or row["needs_llamaparse"] == "True":
                    flagged_count += 1
                if i % 10 == 0 or i == len(files):
                    print(
                        f"  已完成 {i}/{len(files)} (最新: {row['file']}, "
                        f"{row['elapsed_s']}s, "
                        f"{'转LlamaParse' if row['needs_llamaparse'] in (True, 'True') else 'Docling可用'})",
                        flush=True,
                    )

            if needs_restart:
                print(f"  [重建worker] 上一个文件超时,终止卡死的子进程并重启", flush=True)
                executor = _kill_and_restart(executor)

    executor.shutdown(wait=True)
    total_elapsed = time.time() - start_all
    print(
        f"\n{stage_name} 完成: 总耗时 {total_elapsed:.1f}s ({total_elapsed / 60:.1f}分钟), "
        f"异常 {error_count} 个(其中超时 {timeout_count} 个), "
        f"标记转LlamaParse {flagged_count} 个",
        flush=True,
    )
    return total_elapsed, error_count, timeout_count, flagged_count


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "1"

    all_pdfs = collect_all_pdfs()
    prior_tested = files_in_csv(OUTPUT_DIR / "batch_report.csv")
    print(f"文本知识类 PDF 总数: {len(all_pdfs)}, 第一批已测试: {len(prior_tested)}", flush=True)

    if stage == "1":
        sample = stratified_sample(all_pdfs, 50, seed=123, exclude_rel_paths=prior_tested)
        run_stage(
            sample,
            OUTPUT_DIR / "stage1_report.csv",
            OUTPUT_DIR / "stage1",
            "阶段一(50个新样本,验证稳定性)",
        )

    elif stage == "2":
        stage1_tested = files_in_csv(OUTPUT_DIR / "stage1_report.csv")
        exclude = prior_tested | stage1_tested
        remaining = [p for p in all_pdfs if str(p.relative_to(DATA_ROOT)) not in exclude]
        print(f"阶段二剩余文件数: {len(remaining)}", flush=True)
        run_stage(
            remaining,
            OUTPUT_DIR / "stage2_report.csv",
            OUTPUT_DIR / "stage2",
            "阶段二(剩余全部PDF)",
        )
    else:
        print(f"未知阶段参数: {stage}(应为 1 或 2)")
        sys.exit(1)
