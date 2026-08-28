"""
parse_code.py 全量跑批: 覆盖"代码知识类"全部256个文件(ipynb/py/sql/r/java)

跟 quality_router 那批 Docling 处理不同,这里都是纯本地文本/语法解析,速度快
(不需要模型推理),不需要像 full_pipeline.py 那样上子进程超时保护 —— 但
用户明确要求了单文件异常隔离 + 增量写入 + 进度显示,这几点还是照做,防止
个别文件(比如损坏的ipynb JSON、编码诡异的源文件)意外中断整批。

用法:
    python ingest/run_parse_code_full.py
"""

import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from route_files import EXCLUDE_DIRS
from parse_code import PARSERS, DATA_ROOT, _course_of, _rel

OUTPUT_DIR = Path(__file__).parent / "_test_output"
REPORT_CSV = OUTPUT_DIR / "parse_code_full_report.csv"

FIELDNAMES = ["format", "course", "file", "size_bytes", "chunk_count", "chunk_types", "error"]


def collect_files(ext: str):
    files = []
    for p in DATA_ROOT.rglob(f"*{ext}"):
        if not p.is_file() or p.name.startswith("._"):
            continue
        rel_parts = p.relative_to(DATA_ROOT).parts
        if len(rel_parts) < 2:
            # 直接躺在 DATA_ROOT 根目录下、不在任何课程子目录里的文件,
            # 只可能是项目自身的脚本(比如根目录的 app.py)——EXCLUDE_DIRS
            # 是按目录名排除的,对这种"没有父目录"的根级文件天然失效
            # (之前 evaluate/ 目录漏排除导致课程代码里混进项目自己的评估
            # 脚本,修完那个bug之后,同一类问题又在 app.py 上重现了才发现
            # 这里还有个结构性漏洞:只要有文件直接放在DATA_ROOT根目录,
            # 不管叫什么名字,都会绕过EXCLUDE_DIRS)
            continue
        if any(part in EXCLUDE_DIRS for part in rel_parts[:-1]):
            continue
        files.append(p)
    return files


def process_one(ext: str, path: Path) -> dict:
    row = {
        "format": ext,
        "course": _course_of(path),
        "file": _rel(path),
        "size_bytes": path.stat().st_size,
        "chunk_count": "",
        "chunk_types": "",
        "error": "",
    }
    try:
        chunks = PARSERS[ext](path)
        row["chunk_count"] = len(chunks)
        type_counts = Counter(c.metadata.get("chunk_type", "?") for c in chunks)
        row["chunk_types"] = ";".join(f"{k}:{v}" for k, v in sorted(type_counts.items()))
    except Exception as e:
        row["chunk_count"] = 0
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def run():
    all_files = []
    for ext in PARSERS:
        for p in collect_files(ext):
            all_files.append((ext, p))

    print(f"代码知识类文件总数: {len(all_files)}", flush=True)
    for ext in PARSERS:
        n = sum(1 for e, _ in all_files if e == ext)
        print(f"  {ext}: {n} 个", flush=True)

    start_all = time.time()
    rows = []

    with open(REPORT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        f.flush()

        for i, (ext, path) in enumerate(all_files, 1):
            row = process_one(ext, path)
            writer.writerow(row)
            f.flush()
            rows.append(row)

            if row["error"]:
                print(f"  [{i}/{len(all_files)}] [异常] {row['file']}: {row['error']}", flush=True)
            elif i % 20 == 0 or i == len(all_files):
                print(f"  已完成 {i}/{len(all_files)} (最新: {row['file']}, {row['chunk_count']}个chunk)", flush=True)

    total_elapsed = time.time() - start_all
    print(f"\n全量解析完成: {len(all_files)}个文件, 总耗时 {total_elapsed:.1f}s", flush=True)
    return rows, total_elapsed


def summarize(rows, total_elapsed):
    print("\n" + "=" * 90)
    print("汇总报告")
    print("=" * 90)

    by_format = defaultdict(list)
    for r in rows:
        by_format[r["format"]].append(r)

    print(f"\n总耗时: {total_elapsed:.1f}s\n")
    print(f"{'格式':<8}{'文件数':>8}{'成功':>8}{'失败':>8}{'chunk总数':>12}{'平均chunk/文件':>16}")
    print("-" * 70)
    for ext, group in by_format.items():
        ok = [r for r in group if not r["error"]]
        failed = [r for r in group if r["error"]]
        total_chunks = sum(int(r["chunk_count"]) for r in ok)
        avg = total_chunks / len(ok) if ok else 0
        print(f"{ext:<8}{len(group):>8}{len(ok):>8}{len(failed):>8}{total_chunks:>12}{avg:>16.1f}")

    # --- 失败文件列表 ---
    all_failed = [r for r in rows if r["error"]]
    print(f"\n处理失败的文件: {len(all_failed)} 个")
    for r in all_failed:
        print(f"  [{r['format']}] {r['file']}: {r['error']}")

    # --- 边界情况检测 ---
    print("\n--- 边界情况检测 ---")
    STRUCTURED_TYPES = {
        ".py": {"py_function", "py_class"},
        ".r": {"r_function"},
        ".java": {"java_method", "java_constructor"},
    }

    anomaly_count = 0
    for ext, group in by_format.items():
        ok = [r for r in group if not r["error"]]
        counts = [int(r["chunk_count"]) for r in ok]
        if not counts:
            continue
        counts_sorted = sorted(counts)
        median = counts_sorted[len(counts_sorted) // 2]

        for r in ok:
            cc = int(r["chunk_count"])
            flags = []

            if cc == 0:
                flags.append("0个chunk(整份文件没提取到任何内容)")

            if ext in STRUCTURED_TYPES and r["size_bytes"] > 200:
                type_counts = dict(
                    (kv.split(":")[0], int(kv.split(":")[1])) for kv in r["chunk_types"].split(";") if kv
                )
                structured = sum(v for k, v in type_counts.items() if k in STRUCTURED_TYPES[ext])
                if structured == 0:
                    flags.append(f"文件有{r['size_bytes']}字节但0个结构化chunk(函数/类/方法一个都没匹配到,"
                                 f"全部落进了toplevel,可能是正则/ast没覆盖到的书写风格)")

            if median > 0 and cc > max(10, median * 5):
                flags.append(f"chunk数({cc})远高于该格式中位数({median})的5倍,可能切分过细或文件本身异常大")

            if flags:
                anomaly_count += 1
                print(f"  [{ext}] {r['file']} (共{cc}个chunk): {'; '.join(flags)}")

    if anomaly_count == 0:
        print("  未发现异常情况")
    else:
        print(f"\n共发现 {anomaly_count} 个需要关注的边界情况")


if __name__ == "__main__":
    rows, elapsed = run()
    summarize(rows, elapsed)
