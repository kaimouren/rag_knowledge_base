"""
"final/期末"相关材料 QA测试集生成 + 检索效果对照

目的: 专门测试RAG系统在"期末考试/final相关材料"这类内容上的表现。
筛选范围: combined_chunks.json里 metadata.file 包含"final"(不区分
大小写)或"期末"的全部chunk——注意这个筛选是按用户字面要求做的,"final"
不只会命中"期末考试"相关文件,也会命中"Final Project"(期末大作业,
跟考试是两回事)这类文件夹,统计阶段会把命中的完整文件列表打印出来,
方便先确认覆盖范围符不符合预期,再决定要不要往下生成QA。

QA生成的prompt、质量要求、CSV schema全部复用 build_qa_candidates.py
(NORMAL_PROMPT/FLAGGED_PROMPT/describe_source/generate_qa/FIELDNAMES),
不重新发明一套——唯一的区别是这里的筛选范围限定在final/期末相关chunk,
抽样时按课程分层(build_qa_candidates.py原来是按quality/代码格式分层,
这里换成按课程分层,因为这次的目标是"尽量覆盖不同课程",不是"覆盖不同
质量等级/语言")。

跟 build_qa_candidates.py 的另一个区别: 生成完QA对后,不停下来等人工
审核,而是直接对每个问题跑一遍 hybrid_query.py 的检索+生成(和线上
问答路径完全一样的代码路径,不是另外模拟的),把"系统实际检索到的来源"
和"系统实际生成的回答"一起写进同一份CSV,跟"LLM生成的候选标准答案"
放在一起方便直接对照。生成的标准答案仅供参考,不代表绝对正确——完整
的正式审核流程(人工过一遍、标注abstention_type)跟 build_qa_candidates.py
产出的候选集一样,这里没有省略,只是没有阻塞在这一步。

产出是候选测试集,不是正式测试集,单独存成一个CSV,不会混入
qa_candidates_for_review.csv。

用法:
    python evaluate/build_final_qa_eval.py
"""

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "query"))

from build_qa_candidates import (  # noqa: E402
    COMBINED_CHUNKS_PATH,
    FIELDNAMES as BASE_FIELDNAMES,
    generate_qa,
)
from hybrid_query import (  # noqa: E402
    build_prompt,
    generate_answer,
    load_resources,
    lookup_by_id,
    retrieve_bm25_ids,
    retrieve_vector_ids,
    rrf_fuse,
)

OUTPUT_CSV = Path(__file__).parent / "qa_candidates_final_review.csv"
SAMPLE_SEED = 2024
TARGET_N = 18  # 用户要求的15-20条区间,取中间值
MIN_CHUNK_LEN = 100  # final相关chunk池子本来就小,门槛不宜像原脚本那么高(150/60)
                       # 否则可能抽不够数,统一放宽到100

FIELDNAMES = BASE_FIELDNAMES + ["retrieved_sources", "system_answer"]


def is_final_related(file_path: str) -> bool:
    if not file_path:
        return False
    return "final" in file_path.lower() or "期末" in file_path


def load_final_chunks():
    all_chunks = json.load(open(COMBINED_CHUNKS_PATH, encoding="utf-8"))
    return [c for c in all_chunks if is_final_related(c["metadata"].get("file"))]


def print_coverage_stats(chunks):
    files = sorted({c["metadata"]["file"] for c in chunks})
    by_course_files = {}
    by_course_count = {}
    for c in chunks:
        course = c["metadata"].get("course") or "?"
        by_course_files.setdefault(course, set()).add(c["metadata"]["file"])
        by_course_count[course] = by_course_count.get(course, 0) + 1

    print("=" * 90)
    print("覆盖范围统计")
    print("=" * 90)
    print(f"命中chunk总数: {len(chunks)}")
    print(f"命中文件数: {len(files)}")
    print("\n按课程分布(文件数 / chunk数):")
    for course, fileset in sorted(by_course_files.items(), key=lambda kv: -len(kv[1])):
        print(f"  {course}: {len(fileset)}个文件, {by_course_count[course]}个chunk")

    print("\n命中的完整文件列表(确认覆盖范围是否合理,注意'final'会同时命中"
          "'期末考试'和'Final Project期末大作业'这两类不同性质的文件夹):")
    for f in files:
        print(f"  {f}")
    print()


def stratified_sample_by_course(chunks, n, rng):
    """按课程轮流抽,每个课程内部先按文件去重(一个文件先只出一条,不够
    再补),尽量覆盖不同课程、不要全从一个文件里抽——跟用户的要求对应,
    不是复用build_qa_candidates.py按quality/代码格式分层那套(这次目标
    维度不一样)。
    """
    usable = [c for c in chunks if len(c["text"]) >= MIN_CHUNK_LEN]

    by_course = {}
    for c in usable:
        by_course.setdefault(c["metadata"].get("course") or "?", []).append(c)

    courses = list(by_course.keys())
    rng.shuffle(courses)

    per_course_pool = {}
    for course, pool in by_course.items():
        by_file = {}
        for c in pool:
            by_file.setdefault(c["metadata"]["file"], []).append(c)
        files = list(by_file.keys())
        rng.shuffle(files)
        per_course_pool[course] = [rng.choice(by_file[f]) for f in files]

    picked = []
    picked_ids = set()
    round_idx = 0
    while len(picked) < n:
        progressed = False
        for course in courses:
            pool = per_course_pool[course]
            if round_idx < len(pool):
                c = pool[round_idx]
                if c["chunk_id"] not in picked_ids:
                    picked.append(c)
                    picked_ids.add(c["chunk_id"])
                    progressed = True
                if len(picked) >= n:
                    break
        round_idx += 1
        if not progressed:
            break

    return picked[:n]


def run_system(question, embed_model, collection, bm25_store, openai_client):
    vector_ids = retrieve_vector_ids(embed_model, collection, question)
    bm25_ids = retrieve_bm25_ids(bm25_store, question)
    fused = rrf_fuse(vector_ids, bm25_ids)

    items = []
    source_lines = []
    for cid, score in fused:
        text, meta = lookup_by_id(bm25_store, cid)
        items.append((text, meta))
        name_part = f" | {meta['name']}" if meta.get("name") else ""
        source_lines.append(f"{cid} | {meta.get('file', '?')} | {meta.get('chunk_type', '?')}{name_part}")

    prompt = build_prompt(question, items)
    answer = generate_answer(openai_client, prompt)
    return "; ".join(source_lines), answer


def main():
    rng = random.Random(SAMPLE_SEED)

    final_chunks = load_final_chunks()
    print_coverage_stats(final_chunks)

    samples = stratified_sample_by_course(final_chunks, TARGET_N, rng)
    print(f"抽样结果: {len(samples)} 条候选(目标{TARGET_N}条)")
    sampled_courses = sorted({c["metadata"].get("course") for c in samples})
    print(f"覆盖课程: {sampled_courses}\n")

    embed_model, collection, bm25_store, openai_client = load_resources()

    rows = []
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        f.flush()

        for i, chunk in enumerate(samples, 1):
            meta = chunk["metadata"]
            row = {
                "chunk_id": chunk["chunk_id"],
                "source_type": meta["source_type"],
                "chunk_type": meta["chunk_type"],
                "quality": meta.get("quality") or "",
                "chunk_quality_flag": ",".join(meta["chunk_quality_flag"]) if meta.get("chunk_quality_flag") else "",
                "file": meta.get("file") or "",
                "name": meta.get("name") or "",
                "chunk_text": chunk["text"],
                "generated_question": "",
                "generated_ground_truth": "",
                "error": "",
                "test_category": "abstention" if meta.get("chunk_quality_flag") else "standard",
                "abstention_type": "",
                "retrieved_sources": "",
                "system_answer": "",
            }
            try:
                qa = generate_qa(openai_client, chunk)
                row["generated_question"] = qa["question"]
                row["generated_ground_truth"] = qa["ground_truth"]

                sources, answer = run_system(qa["question"], embed_model, collection, bm25_store, openai_client)
                row["retrieved_sources"] = sources
                row["system_answer"] = answer
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                print(f"  [{i}/{len(samples)}] [失败] {chunk['chunk_id']}: {row['error']}", flush=True)

            writer.writerow(row)
            f.flush()
            rows.append(row)

            if not row["error"]:
                print(f"  [{i}/{len(samples)}] {chunk['chunk_id']} ({meta['source_type']}/{meta['chunk_type']}, {meta.get('course')}) 完成", flush=True)

    errors = [r for r in rows if r["error"]]
    print(f"\n完成: {len(rows)} 条候选, {len(errors)} 条生成/检索失败")
    print(f"已保存至: {OUTPUT_CSV}")
    print("\n这是候选清单,不是正式测试集——generated_ground_truth仅供参考,请对照retrieved_sources/system_answer人工审核。")


if __name__ == "__main__":
    main()
