"""
基于预算约束的质量分层决策

背景: full_pipeline.py 跑完全部445个"文本知识类"PDF后,quality_router判定
218个(49%)需要转LlamaParse,总计8373页,按3 credits/页要25119 credits,
远超10000免费额度(详见对话记录/final_report.csv)。不可能全部转,需要按
课程分优先级,这是一个预算约束下的主动工程决策,不是"漏处理"。

优先级决策(人工评审后确定,不是脚本自动算的):
  刚需课程 = ieor 4742 deep learning + CSE 416
  理由: 求职方向相关度高(深度学习/机器学习),这两门课加起来3189页、
        9567 credits,压着10000免费额度线内(留433 credits缓冲)。
  IEOR E4106 Stochastic Model 虽然是这次统计里 credits 占比最高的
  (5613,单科22.3%),但明确排除在优先级之外——不是技术原因,是本次
  预算分配的取舍结果。

本模块产出一份"文件级质量清单"(quality_manifest.json),给每个被
quality_router判定过的PDF标注三种状态之一:
  - good:              未被标记,Docling输出可直接用
  - pending_llamaparse: 优先课程里被标记的文件,后续会转LlamaParse重新解析
                        (本模块只做标记决策,不实际调用LlamaParse API)
  - degraded:           非优先课程里被标记的文件,保留现有Docling输出,
                        但标注质量降级 + 明确的降级原因,供下游 chunk/index/
                        generate 模块读取使用(比如检索时降权、生成回答时
                        附加"内容来源质量有限"提示、或者干脆排除出检索范围
                        —— 具体怎么用留给对应模块决定,这里只负责打标签)

用法:
    python ingest/quality_manifest.py

---
【后续状态变更,2026-09】本模块产出的 quality_manifest.json 仍然如实记录了
"当时"那次Docling+LlamaParse预算分配决策,继续保留作为历史记录——但它的
good/degraded/llamaparse_done三态标签已经不再描述"现在索引里的文本chunk
是怎么产出的"。439个文档(6个pending除外)全部改用MinerU重新解析,不再有
"部分文档用更贵的工具"这种差异,三态分层本身依赖的前提(不同文档走了不同
质量的工具)不再成立。

现在的质量信号来源改成:
  - ingest/mineru_parse_report.csv: 记录MinerU解析成功/失败(目前439/439成功)
  - chunk_quality_flag(chunk_text_mineru.py产出,chunk粒度而非文档粒度):
    fragmented(行碎片化)、formula_unverifiable(chunk含独立公式块,MinerU
    对这类公式的识别结果没有任何置信度信号,无法自动判断对错——见
    LIMITATIONS.md"公式可信度风险"一节)、formula_low_confidence(chunk所在页
    有行内公式被MinerU自己的OCR置信度判定为低分,阈值0.6)
  - previous_docling_quality(chunk_text_mineru.py保留的历史字段): 直接
    继承自本manifest的旧good/degraded/llamaparse_done标签,纯粹作为"这份
    文档在旧pipeline下曾经被怎么评估过"的历史参考,不代表当前质量判断
  - 注意: detect_chunk_quality_flags()里的formula_missing、
    has_unresolved_image两个flag是针对Docling输出的占位符字符串
    ("formula-not-decoded"、"<!-- image -->")设计的,MinerU的markdown
    输出不产生这两种占位符(公式要么被转成真实LaTeX,要么整个不出现;
    图片是"![](images/...)"格式),实测439个文档的chunk里这两个flag
    出现次数为0——不是没检测到问题,是这两个flag对MinerU产线而言是
    已知失效的检测逻辑,如实记录在这里,不是自然为0。
"""

import csv
import json
from pathlib import Path

REPORT_CSV = Path(__file__).parent / "_test_output" / "final_report.csv"
MANIFEST_JSON = Path(__file__).parent / "_test_output" / "quality_manifest.json"

CREDITS_PER_PAGE = 3
FREE_QUOTA = 10000

# 人工评审确定的优先级,不是算法自动选的——这是求职方向相关性的主观判断,
# 换一批课程或者换一个人选,这个集合就该重新定。
PRIORITY_COURSES = {"ieor 4742 deep learning", "CSE 416"}


def build_manifest():
    rows = list(csv.DictReader(open(REPORT_CSV, encoding="utf-8-sig")))

    manifest = {}
    stats = {"good": 0, "pending_llamaparse": 0, "degraded": 0}
    pending_pages = degraded_pages = 0

    for r in rows:
        file_path = r["file"]
        flagged = r["needs_llamaparse"] == "True"
        course = r["course"]
        pages = int(float(r["pages"])) if r["pages"] else None

        if not flagged:
            status = "good"
            entry = {
                "quality": "good",
                "course": course,
                "pages": pages,
            }
        elif course in PRIORITY_COURSES:
            status = "pending_llamaparse"
            entry = {
                "quality": "pending_llamaparse",
                "course": course,
                "pages": pages,
                "docling_reasons": r["reasons"],
                "note": "该课程被判定为求职方向刚需课程,已优先分配LlamaParse预算,"
                        "等待实际调用LlamaParse重新解析(本清单只记录决策,尚未执行)。",
            }
            if pages:
                pending_pages += pages
        else:
            status = "degraded"
            entry = {
                "quality": "degraded",
                "course": course,
                "pages": pages,
                "docling_reasons": r["reasons"],
                "degraded_reason": "budget_constraint",
                "note": (
                    f"quality_router 判定 Docling 解析质量不足(原因: {r['reasons']}),"
                    f"但受 LlamaParse 免费额度(10000 credits)限制,{course} 未被列入"
                    f"本轮优先课程,主动决定不转LlamaParse、保留现有Docling输出。"
                    f"这是预算约束下的工程取舍,不是遗漏——如果后续额度充足或该课程"
                    f"优先级上调,可以用同一批Docling输出文件重新触发LlamaParse转换。"
                ),
            }
            if pages:
                degraded_pages += pages

        manifest[file_path] = entry
        stats[status] += 1

    return manifest, stats, pending_pages, degraded_pages


def print_summary(stats, pending_pages, degraded_pages):
    total = sum(stats.values())
    print(f"文件总数: {total}")
    print(f"  good (Docling可用):             {stats['good']:>4} 个")
    print(f"  pending_llamaparse (优先课程):   {stats['pending_llamaparse']:>4} 个, {pending_pages} 页, "
          f"约 {pending_pages * CREDITS_PER_PAGE} credits")
    print(f"  degraded (预算内降级):           {stats['degraded']:>4} 个, {degraded_pages} 页")

    pending_credits = pending_pages * CREDITS_PER_PAGE
    print(f"\n优先课程预计消耗: {pending_credits} / {FREE_QUOTA} credits "
          f"({pending_credits / FREE_QUOTA:.1%}), 剩余 {FREE_QUOTA - pending_credits} credits")

    print(f"\n降级文件按课程分布:")
    from collections import Counter
    rows = list(csv.DictReader(open(REPORT_CSV, encoding="utf-8-sig")))
    degraded_by_course = Counter()
    for r in rows:
        if r["needs_llamaparse"] == "True" and r["course"] not in PRIORITY_COURSES:
            degraded_by_course[r["course"]] += 1
    for course, n in degraded_by_course.most_common():
        print(f"  {course}: {n} 个")


if __name__ == "__main__":
    manifest, stats, pending_pages, degraded_pages = build_manifest()

    with open(MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"已生成质量清单: {MANIFEST_JSON}\n")
    print_summary(stats, pending_pages, degraded_pages)
