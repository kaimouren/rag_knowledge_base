"""
图片类处理(全量): 对MinerU解析时提取的图片(19445张,分布在各文档的
ocr/images/目录下)和"图片类"独立文件(77个,route_files.py路由结果)
做去重+过滤+OCR/VLM路由描述,产出可供后续chunking使用的图片描述数据集。

背景与规模摸底(已完成,数字来自实测,不是估算):
  - 19445张MinerU图片,内容哈希去重后剩17433张(2012张完全重复,常见于
    同一份讲义被多个学生上传到各自文件夹的情况,或者PPT模板里重复出现
    的装饰性banner)
  - 77个独立图片文件,1个是HEIC伪装成.jpg打不开,76个有效且互不重复
  - 过滤掉<5KB(大概率图标/装饰元素)和宽高<50px的,最终剩 11115 张
    (11039 mineru + 76 standalone)——这是真正需要处理的规模,比原始
    19522张少43%

路由逻辑(15-20张抽样验证过,见对话记录,不是拍脑袋定的阈值):
  - EasyOCR平均置信度 > 0.85: 只用OCR文字(印刷体/清晰截图,OCR够用,
    比VLM更精确,能保留具体数值,VLM反而会"总结"掉细节)
  - < 0.5: 只用VLM描述(照片/图表/手写内容,OCR要么读不出来,要么读出来
    是乱码,VLM能给出正确的语义描述)
  - 0.5-0.85: 两者都做,都存下来(灰色地带,人工判断哪个更有用)
  - 已知局限: 这个阈值基于EasyOCR的置信度分数,分数本身是"OCR有多确信
    自己读对了",不是"这张图片本身有没有信息量"的直接度量,抽样测试中
    也发现少数图表因为轴标签文字清晰而拿到偏高置信度,但图表趋势本身
    还是需要VLM才能读出来——已经用"两者都做"覆盖了这个中间地带,不是
    完全没考虑到这个风险。

VLM模型: gpt-5.6-luna(不是最初讨论的Claude vision,项目里没有
ANTHROPIC_API_KEY;也不是最早测试用的gpt-4o-mini/gpt-6-astra——
经过多个候选模型的实测token消耗+定价对比+3张图表类描述的人工核对
后选定,人工核对方法是直接打开原图核实VLM给出的具体数值/趋势判断,
比如核对一张柱状图"最高点在第17个patch、数值范围5.3-5.9"这类具体
断言是否属实,3张图表类样本(柱状图、ROC曲线、决策边界散点图)描述
全部准确,没有编造)。

用法:
    python ingest/process_images.py --test    # 只跑18张抽样,不动全量
    python ingest/process_images.py           # 全量跑11115张
    (支持断点续跑: 已经写进output CSV的图片会跳过)
"""

import argparse
import base64
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import easyocr
import numpy as np
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

load_dotenv(Path(__file__).parent.parent / ".env")

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from route_files import EXCLUDE_DIRS

DATA_ROOT = Path(r"E:\project")
INGEST_DIR = Path(__file__).parent
MINERU_BATCH = INGEST_DIR / "_test_output" / "mineru_batch"
MINERU_REPORT_CSV = INGEST_DIR / "_test_output" / "mineru_parse_report.csv"

OUTPUT_CSV = INGEST_DIR / "_test_output" / "image_processing_full.csv"

MIN_BYTES = 5 * 1024
MIN_DIM = 50
HIGH_CONF = 0.85
LOW_CONF = 0.5

VLM_MODEL = "gpt-5.6-luna"
PROMPT = (
    "This image is embedded in a university course PDF (lecture slide, homework, or exam). "
    "Describe what it shows in 2-4 sentences: is it a screenshot of text/code/UI, a chart/plot "
    "(name axes and what it shows), a diagram/schematic, a photo, or decorative art. "
    "If it contains readable text, quote the key text. If it's a chart, describe the data trend."
)

# gpt-5.6-luna定价(2026-09核实,见 developers.openai.com/api/docs/pricing)
PRICE_PER_1M_INPUT = 0.20
PRICE_PER_1M_OUTPUT = 1.20

FIELDNAMES = [
    "path", "source", "course", "file", "size", "w", "h", "hash",
    "ocr_conf", "ocr_box_count", "ocr_char_count", "ocr_text",
    "tier", "vlm_desc", "vlm_in_tok", "vlm_out_tok", "vlm_reasoning_tok", "error",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_mineru_images():
    rows = list(csv.DictReader(open(MINERU_REPORT_CSV, encoding="utf-8-sig")))
    idx_to_doc = {int(r["idx"]): r for r in rows if r["status"] == "ok"}

    records = []
    for img_path in MINERU_BATCH.glob("*/ocr/images/*"):
        idx = None
        for part in img_path.parts:
            if part.isdigit():
                idx = int(part)
                break
        doc = idx_to_doc.get(idx)
        if not doc:
            continue
        try:
            size = img_path.stat().st_size
            with Image.open(img_path) as im:
                w, h = im.size
        except (UnidentifiedImageError, OSError):
            continue
        records.append({
            "path": str(img_path), "source": "mineru", "course": doc["course"], "file": doc["file"],
            "size": size, "w": w, "h": h,
        })
    return records


def collect_standalone_images():
    records = []
    for p in DATA_ROOT.rglob("*"):
        if not p.is_file():
            continue
        parts = p.relative_to(DATA_ROOT).parts
        if any(part in EXCLUDE_DIRS or part.startswith(".") for part in parts):
            continue
        if p.suffix.lower() in (".png", ".jpg", ".jpeg") and len(parts) > 1:
            try:
                size = p.stat().st_size
                with Image.open(p) as im:
                    w, h = im.size
            except (UnidentifiedImageError, OSError):
                continue
            records.append({
                "path": str(p), "source": "standalone", "course": parts[0], "file": str(Path(*parts)),
                "size": size, "w": w, "h": h,
            })
    return records


def dedup_and_filter(records):
    by_hash = {}
    for r in records:
        r["hash"] = sha256_of(Path(r["path"]))
        by_hash.setdefault(r["hash"], r)  # 保留第一个出现的副本,重复的丢弃
    kept = [r for r in by_hash.values() if r["size"] >= MIN_BYTES and r["w"] >= MIN_DIM and r["h"] >= MIN_DIM]
    return kept


def tier_of(conf: float) -> str:
    if conf > HIGH_CONF:
        return "ocr_only"
    if conf < LOW_CONF:
        return "vlm_only"
    return "both"


def load_done_paths():
    if not OUTPUT_CSV.exists():
        return set()
    rows = list(csv.DictReader(open(OUTPUT_CSV, encoding="utf-8-sig")))
    return {r["path"] for r in rows if not r.get("error")}


def call_vlm(client, path, ext):
    # max_completion_tokens=1200,不是600: 全量跑完后发现48张图片(占VLM调用
    # 的0.54%)返回了空字符串,usage显示out_tok==reasoning_tok==600——luna是
    # 推理模型,少数图片会把整个600token预算耗在看不见的内部推理上,一个字
    # 没写出来,但token照样计费。跟之前测gpt-5-nano时发现的坑是同一个机制。
    # 600->1200后这48张重跑全部拿到了正常输出,不是完美解法(理论上仍可能有
    # 极少数图片推理更久),但目前观测到的失败样本全部被覆盖了。
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=VLM_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
                    ],
                }],
                max_completion_tokens=1200,
            )
            usage = resp.usage
            reasoning = usage.completion_tokens_details.reasoning_tokens if usage.completion_tokens_details else 0
            return resp.choices[0].message.content, usage.prompt_tokens, usage.completion_tokens, reasoning, None
        except Exception as e:
            if attempt < 4:
                time.sleep(10 * (attempt + 1))
            else:
                return "", 0, 0, 0, f"{type(e).__name__}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="只跑18张(复用之前抽样测试用的同一批),不动全量")
    args = parser.parse_args()

    print("=== 收集图片(mineru + standalone) ===", flush=True)
    mineru_records = collect_mineru_images()
    standalone_records = collect_standalone_images()
    print(f"mineru原始文件数: {len(mineru_records)}, standalone: {len(standalone_records)}", flush=True)

    print("=== 去重+过滤(计算哈希,可能需要几分钟) ===", flush=True)
    all_kept = dedup_and_filter(mineru_records) + dedup_and_filter(standalone_records)
    print(f"去重过滤后总数: {len(all_kept)}", flush=True)

    if args.test:
        all_kept = all_kept[:18]

    done = load_done_paths()
    remaining = [r for r in all_kept if r["path"] not in done]
    print(f"已完成(跳过): {len(done & {r['path'] for r in all_kept})}, 待处理: {len(remaining)}", flush=True)

    reader = easyocr.Reader(["en"], gpu=True)
    client = OpenAI()

    write_header = not OUTPUT_CSV.exists()
    total_vlm_calls = 0
    total_in_tok = 0
    total_out_tok = 0

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for i, r in enumerate(remaining, 1):
            path = r["path"]
            row = dict(r)
            row.update({"ocr_conf": "", "ocr_box_count": "", "ocr_char_count": "", "ocr_text": "",
                        "tier": "", "vlm_desc": "", "vlm_in_tok": "", "vlm_out_tok": "",
                        "vlm_reasoning_tok": "", "error": ""})
            try:
                # 16张图片(全部是standalone来源)在全量跑的时候在这里报过
                # AttributeError: 'NoneType' object has no attribute 'shape'——
                # EasyOCR内部用cv2.imread(path)读文件,这16张PIL能正常打开
                # (前面的dedup_and_filter已经用PIL读过一遍w/h,没报错),但
                # cv2.imread返回None,不是Unicode路径的问题(16张里一半是纯
                # ASCII文件名),更像是这几个PNG/JPG的编码细节cv2解码器认不出
                # 来。绕开的方法很直接:不传路径字符串,改成用PIL读成numpy
                # array再传给readtext,EasyOCR接受这种输入,PIL的解码器更宽容。
                with Image.open(path) as im:
                    img_array = np.array(im.convert("RGB"))
                ocr_out = reader.readtext(img_array)
                texts = [t[1] for t in ocr_out]
                confs = [t[2] for t in ocr_out]
                full_text = " ".join(texts)
                avg_conf = sum(confs) / len(confs) if confs else 0.0

                row["ocr_conf"] = round(avg_conf, 4)
                row["ocr_box_count"] = len(confs)
                row["ocr_char_count"] = len(full_text)
                row["ocr_text"] = full_text
                tier = tier_of(avg_conf)
                row["tier"] = tier

                if tier in ("vlm_only", "both"):
                    ext = Path(path).suffix.lower().lstrip(".")
                    if ext == "jpg":
                        ext = "jpeg"
                    desc, in_tok, out_tok, reasoning_tok, err = call_vlm(client, path, ext)
                    row["vlm_desc"] = desc
                    row["vlm_in_tok"] = in_tok
                    row["vlm_out_tok"] = out_tok
                    row["vlm_reasoning_tok"] = reasoning_tok
                    if err:
                        row["error"] = err
                    else:
                        total_vlm_calls += 1
                        total_in_tok += in_tok
                        total_out_tok += out_tok
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"

            writer.writerow(row)
            f.flush()

            if i % 50 == 0 or i == len(remaining):
                running_cost = (total_in_tok * PRICE_PER_1M_INPUT + total_out_tok * PRICE_PER_1M_OUTPUT) / 1e6
                print(f"  [{i}/{len(remaining)}] VLM调用累计={total_vlm_calls}, "
                      f"累计token(in/out)={total_in_tok}/{total_out_tok}, "
                      f"累计成本=${running_cost:.4f}", flush=True)

    final_cost = (total_in_tok * PRICE_PER_1M_INPUT + total_out_tok * PRICE_PER_1M_OUTPUT) / 1e6
    print(f"\n=== 完成 ===", flush=True)
    print(f"本次处理: {len(remaining)} 张, VLM调用: {total_vlm_calls} 次", flush=True)
    print(f"本次VLM总token: input={total_in_tok}, output={total_out_tok}", flush=True)
    print(f"本次VLM实际成本: ${final_cost:.4f}", flush=True)
    print(f"结果已保存至: {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
