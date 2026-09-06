"""
给 image_processing_full.csv 里的 mineru 来源图片(11039张)补上page_idx。

背景: content_list.json 里不是只有 type:"image" 的记录带 img_path——
实测(ingest/process_images.py 跑完之后逐类核对)发现 equation/chart/table
四种block类型都会各自存一份裁剪图,都带 img_path + page_idx。第一版只按
type:"image" 去join,匹配率只有26.8%(4440/11039的分母算错了,实际是
11039张图片对4440条记录),换成四种类型一起join,匹配率跳到99.55%。

剩下0.45%(50张)进一步排查: 34张是表格单元格内嵌的图片——MinerU的
表格识别会把"表格里某一格是一张图"这种情况,把img标签直接写进这个
table块的table_body HTML字符串里(`<img src="images/xxx.jpg"/>`),
不是一个独立的顶层block,所以第一版join找不到它对应的记录,补一次
"扫table_body里的img标签,用父表格的page_idx"的规则后覆盖了这34张。

剩下16张(0.14%)确认过不是bug产出的孤儿文件——在更底层的middle.json里
能找到对应记录和真实page信息,但content_list.json的导出没有把它带出来
(具体是content_list.json生成时对某些被判定为"低置信度"或"被合并"的
表格block做了裁剪,这16张全部出自这类内部被丢弃的表格)。这16张排查
到此为止,不再往middle.json里挖——0.14%的量级,而且这个"信息在更底层
JSON里但content_list.json没导出"是MinerU自己的行为,不是这里的代码
可以简单修的东西,如实标注 page_idx=None,不假装补上。

用法:
    python ingest/join_image_page_idx.py
"""

import csv
import json
import re
from pathlib import Path

INGEST_DIR = Path(__file__).parent
MINERU_BATCH = INGEST_DIR / "_test_output" / "mineru_batch"
CSV_PATH = INGEST_DIR / "_test_output" / "image_processing_full.csv"

IMG_LIKE_TYPES = {"image", "equation", "chart", "table"}
TABLE_IMG_SRC_RE = re.compile(r'<img src="(images/[0-9a-f]+\.(?:jpg|jpeg|png))"')


def build_page_map():
    page_map = {}
    for cl_path in MINERU_BATCH.glob("*/ocr/*_content_list.json"):
        ocr_dir = cl_path.parent
        cl = json.load(open(cl_path, encoding="utf-8"))
        for b in cl:
            # 顶层block: image/equation/chart/table 各自的img_path
            if b.get("type") in IMG_LIKE_TYPES and b.get("img_path"):
                full_path = str((ocr_dir / b["img_path"]).resolve())
                page_map[full_path] = b["page_idx"]
            # table_body HTML里嵌套的图片(表格单元格内容是图片的情况),
            # 用父表格自己的page_idx——同一张表格不会跨页,这个近似是准确的
            if b.get("type") == "table" and b.get("table_body"):
                for rel in TABLE_IMG_SRC_RE.findall(b["table_body"]):
                    full_path = str((ocr_dir / rel).resolve())
                    page_map.setdefault(full_path, b["page_idx"])
    return page_map


def main():
    page_map = build_page_map()
    print(f"page_map条目数: {len(page_map)}")

    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8-sig")))
    fieldnames = list(rows[0].keys())
    if "page_idx" not in fieldnames:
        fieldnames.append("page_idx")

    matched = 0
    mineru_total = 0
    unmatched_paths = []
    for r in rows:
        if r["source"] != "mineru":
            r["page_idx"] = ""
            continue
        mineru_total += 1
        full_path = str(Path(r["path"]).resolve())
        if full_path in page_map:
            r["page_idx"] = page_map[full_path]
            matched += 1
        else:
            r["page_idx"] = ""
            unmatched_paths.append(r["path"])

    print(f"mineru来源图片总数: {mineru_total}")
    print(f"成功join到page_idx: {matched} ({matched/mineru_total*100:.2f}%)")
    print(f"未能join(page_idx留空): {len(unmatched_paths)} ({len(unmatched_paths)/mineru_total*100:.2f}%)")
    if unmatched_paths:
        print("未匹配样例(最多显示10个):")
        for p in unmatched_paths[:10]:
            print(" ", p)

    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n已写回: {CSV_PATH}")


if __name__ == "__main__":
    main()
