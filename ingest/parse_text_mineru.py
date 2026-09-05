"""
文本知识类解析: 全量445个PDF改用MinerU重新解析(替代之前Docling/LlamaParse
混合方案里"页码信息在导出时被拍平丢失"的那部分)。

背景(调查结论,详见对话记录/LIMITATIONS.md):
  - Docling: DoclingDocument对象本身每个item都带精确prov.page_no,
    export_to_markdown()也支持page_break_placeholder参数——但当初调用时
    没传任何参数,只存了拍平后的markdown字符串,页码信息在那一步被丢弃,
    要拿回来必须重新跑一次Docling(本地免费,只是CPU时间成本)。
  - LlamaParse: API返回本身就是分页的(result.markdown.pages),但两个脚本
    存盘前直接"\n\n".join()拍平,且事后用client.parsing.get(job_id)实测
    确认服务端解析结果已过期(只剩job状态,markdown字段是None)——免费
    捞不回来,要拿回来必须重新花credits调API。

与其分别打两次不同代价的补丁,这次统一改用MinerU重新解析全部445个文件:
MinerU的content_list.json本身就有精确的page_idx(实测确认,见下方
build_page_index()),不需要额外传参数或事后追加请求;而且之前的小范围
评估(已发布的"MinerU Trial Proofs" artifact)显示MinerU在公式转LaTeX和
扫描/手写OCR上普遍优于Docling,只在表格提取上有一个反例(LR2.pdf)。

运行参数微调(用户要求"微调一下MinerU",这里调的是调用方式和硬件,不是
训练/微调模型权重):
  1) GPU: 实测RTX 5070(Blackwell架构,sm_120)之前装的是CPU-only build的
     torch,装不了就用不了MINERU_DEVICE_MODE=cuda。换成官方cu128构建后
     GPU可用,在公式密集/长文档上加速明显(25页/250处公式的样本:
     CPU 292秒 -> GPU 40秒,约7倍);短文档提升有限(模型加载是固定开销,
     单文件调用摊不掉,比如1页的样本CPU 24.5秒 -> GPU 25秒,几乎没变化)。
  2) 批量喂文件: 正是因为模型加载是固定开销,每个文件单独起一次mineru
     进程会让这个开销被文件数放大——实测5个文件单独跑总计156秒,
     一次性把5个文件所在目录喂给mineru只need 51秒(同一次加载,处理完
     就进行下一个)。所以这里把445个文件分成BATCH_SIZE一批,每批一次
     mineru调用,而不是445次单独调用。

用法:
    python ingest/parse_text_mineru.py --test   # 先跑12个文件验证,不动全量
    python ingest/parse_text_mineru.py          # 全量跑剩余(444个,排除pending)
    (支持断点续跑: 已经在 mineru_parse_report.csv 里记录过status=ok的文件会跳过)
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DATA_ROOT = Path(r"E:\project")
INGEST_DIR = Path(__file__).parent
FINAL_REPORT_CSV = INGEST_DIR / "_test_output" / "final_report.csv"
QUALITY_MANIFEST = INGEST_DIR / "_test_output" / "quality_manifest.json"

STAGE_DIR = INGEST_DIR / "_test_output" / "mineru_stage"
OUTPUT_DIR = INGEST_DIR / "_test_output" / "mineru_batch"
REPORT_CSV = INGEST_DIR / "_test_output" / "mineru_parse_report.csv"

BATCH_SIZE = 20
TEST_LIMIT = 12

MINERU_EXE = str(Path(sys.executable).parent / "mineru.exe")
FIELDNAMES = ["idx", "file", "course", "old_quality", "pages_old", "status", "error", "elapsed_s"]


def load_targets(limit=None):
    manifest = json.load(open(QUALITY_MANIFEST, encoding="utf-8"))
    rows = list(csv.DictReader(open(FINAL_REPORT_CSV, encoding="utf-8-sig")))

    targets = []
    for idx, r in enumerate(rows):
        rel = r["file"]
        entry = manifest.get(rel, {})
        if entry.get("quality") == "pending":
            # 沿用quality_manifest.json里已经做过的"暂缓处理"决定,这次不主动纳入,
            # 跟chunk_text.py load_doc_index()里跳过pending的逻辑保持一致
            continue
        targets.append({
            "idx": idx,
            "file": rel,
            "course": entry.get("course", r.get("course", "")),
            "old_quality": entry.get("quality", ""),
            "pages_old": r.get("pages", ""),
        })

    if limit:
        # 测试模式覆盖good/degraded/llamaparse_done三种,不是简单取前N个
        by_quality = {}
        for t in targets:
            by_quality.setdefault(t["old_quality"], []).append(t)
        picked = []
        per_bucket = max(1, limit // max(1, len(by_quality)))
        for bucket in by_quality.values():
            picked.extend(bucket[:per_bucket])
        return picked[:limit]

    return targets


def load_done_idx():
    if not REPORT_CSV.exists():
        return set()
    rows = list(csv.DictReader(open(REPORT_CSV, encoding="utf-8-sig")))
    return {int(r["idx"]) for r in rows if r["status"] == "ok"}


def stage_batch(batch):
    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    STAGE_DIR.mkdir(parents=True)
    for t in batch:
        src = DATA_ROOT / t["file"]
        dst = STAGE_DIR / f"{t['idx']:05d}.pdf"
        shutil.copyfile(src, dst)


def run_mineru_batch():
    env = {**os.environ, "MINERU_DEVICE_MODE": "cuda", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    cmd = [MINERU_EXE, "-p", str(STAGE_DIR), "-o", str(OUTPUT_DIR), "-b", "pipeline", "-m", "ocr", "-l", "en"]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return result.returncode, result.stdout, result.stderr


def check_output(idx) -> bool:
    stem = f"{idx:05d}"
    content_list = OUTPUT_DIR / stem / "ocr" / f"{stem}_content_list.json"
    md_file = OUTPUT_DIR / stem / "ocr" / f"{stem}.md"
    if not (content_list.exists() and md_file.exists()):
        return False
    try:
        data = json.loads(content_list.read_text(encoding="utf-8"))
        return isinstance(data, list) and len(data) > 0
    except (json.JSONDecodeError, OSError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="只跑12个文件验证,不动全量")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = load_targets(limit=TEST_LIMIT if args.test else None)
    done_idx = load_done_idx()
    remaining = [t for t in targets if t["idx"] not in done_idx]

    print(f"目标文件数: {len(targets)}, 已完成(跳过): {len(done_idx & {t['idx'] for t in targets})}, "
          f"待处理: {len(remaining)}", flush=True)

    if not remaining:
        print("没有待处理文件,退出。", flush=True)
        return

    write_header = not REPORT_CSV.exists()
    with open(REPORT_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for batch_start in range(0, len(remaining), args.batch_size):
            batch = remaining[batch_start:batch_start + args.batch_size]
            batch_no = batch_start // args.batch_size + 1
            total_batches = (len(remaining) + args.batch_size - 1) // args.batch_size
            print(f"\n=== 批次 {batch_no}/{total_batches} ({len(batch)}个文件) ===", flush=True)

            stage_batch(batch)
            start = time.time()
            returncode, stdout, stderr = run_mineru_batch()
            elapsed = time.time() - start
            per_file_elapsed = round(elapsed / len(batch), 1)

            ok_count = 0
            for t in batch:
                success = check_output(t["idx"])
                writer.writerow({
                    "idx": t["idx"],
                    "file": t["file"],
                    "course": t["course"],
                    "old_quality": t["old_quality"],
                    "pages_old": t["pages_old"],
                    "status": "ok" if success else "failed",
                    "error": "" if success else f"batch_returncode={returncode}",
                    "elapsed_s": per_file_elapsed,
                })
                ok_count += int(success)
            f.flush()

            print(f"  批次耗时 {elapsed:.1f}s | 成功 {ok_count}/{len(batch)} | "
                  f"平均每文件 {per_file_elapsed}s", flush=True)
            if ok_count < len(batch):
                print(f"  [警告] 本批次有失败文件,mineru返回码={returncode}", flush=True)
                if stderr.strip():
                    print(f"  stderr末尾: {stderr[-500:]}", flush=True)

    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)

    all_rows = list(csv.DictReader(open(REPORT_CSV, encoding="utf-8-sig")))
    ok_total = sum(1 for r in all_rows if r["status"] == "ok")
    print(f"\n=== 全部完成 ===\n累计成功 {ok_total}/{len(all_rows)}", flush=True)


if __name__ == "__main__":
    main()
