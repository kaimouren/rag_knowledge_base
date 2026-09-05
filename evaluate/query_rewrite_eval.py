"""
Query Rewriting 效果验证

目的: query_rewrite.py + hybrid_query.py 里新增的"实体BM25打分第三路"
是可选增强(默认关闭),这个脚本就是决定"要不要把它设成默认开启"的
证据来源——不是随便跑一下看着还行就完事,要给出可以对比的具体数字。

v1 -> v2 改动: v1版本(实体精确子串匹配+命中计数)实测发现两个问题——
中英复合词("SQL查询")按整体匹配不上纯英文原文、常见英文单词
("Caregiver")没有特异性判断导致误匹配挤占正确结果(具体案例见下面
V1_DIAG_RESULTS/V1_TQ_PROXY_METRICS的注释)。v2版本改成: 实体先按
语言边界拆分(query_rewrite.py::split_entity_to_tokens()),再复用主
BM25索引的get_scores()打分。v1的完整测试代码已经被v2覆盖替换,没办法
重新跑一遍——这里把v1那次跑出来的真实结果记成常量,跟v2这次的结果放
在同一张表里对比,不是凭印象补的数字。baseline(完全不用rewrite)的
结果在v1/v2两次跑法下应该是同一个数(rewrite不介入时代码路径完全不变),
这里以v2这次实际跑出的baseline为准。

测试集分两部分:
  1. 14个已知诊断案例(evaluate/_diag_text_miss_data.json,之前排查
     "文本知识类检索命中率只有6.7%"那次生成的,每条都有明确的目标
     chunk_id): A类语言不匹配7个 + B类词汇鸿沟3个(txt_05591等,这次
     任务点名要测的) + C类检索池稀释4个。对每一条都精确算"目标chunk
     有没有进最终top-5"(hit@5),加rewriting前后各跑一次,直接对比。
     这些案例全部是中文提问,同时覆盖了"中文案例"这个要求。
  2. hybrid_query.py::TEST_QUESTIONS(5个,K-means/Caregiver/Caregivers
     SQL/get_initial_centroids/biased_coin,代码知识类),这几个之前是
     靠人工看top-5来源判断对不对,没有存"标准答案chunk_id"这种结构化
     数据,所以这里只能做加rewriting前后的top-5来源定性对比,人工判断。

同时统计query rewriting这一步新增的延迟和token成本(相对原有hybrid
检索是纯新增的一次LLM调用),这样"要不要默认开启"是拿真实的收益和
成本一起权衡出来的,不是只看效果不看代价。

用法:
    python evaluate/query_rewrite_eval.py
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "query"))

from hybrid_query import (  # noqa: E402
    TEST_QUESTIONS,
    load_resources,
    lookup_by_id,
    retrieve_bm25_ids,
    retrieve_entity_bm25_matches,
    retrieve_vector_ids,
    rrf_fuse,
)
from query_rewrite import extract_query_entities  # noqa: E402

DIAG_FILE = Path(__file__).parent / "_diag_text_miss_data.json"

# gpt-4o-mini 官方定价(写这个脚本时查的,后续可能变,这里只是给个数量级
# 参考,不是精确账单来源)
PRICE_PER_1M_INPUT = 0.15
PRICE_PER_1M_OUTPUT = 0.60

# v1(精确子串匹配+命中计数)那次实际跑出来的14个诊断案例hit@5结果,
# 全部False->False(仍不中)除了txt_00382改善成True。这次(v2)在
# print_diag_comparison()里跟这个对比。
V1_DIAG_HIT_AFTER = {
    "txt_16272": False,
    "txt_05129": False,
    "txt_05591": False,
    "txt_10161": False,
    "txt_05618": False,
    "txt_17615": False,
    "txt_00382": True,
    "txt_15857": False,
    "txt_10244": False,
    "txt_06560": False,
    "txt_02341": False,
    "txt_15110": False,
    "txt_20002": False,
    "txt_17596": False,
}

# v1那次TEST_QUESTIONS的3个代理指标(从实际打印出的top-5来源人工数出来的,
# 不是重新跑的):
#   caregivers_sql_count: "有没有创建Caregivers表的SQL语句?"这条,
#     top-5里chunk_type=="sql_statement"的个数(越高越好,这是v1退化的
#     那个案例:baseline=3, v1=1)
#   caregiver_class_hit: "Caregiver类是怎么实现的?"这条,top-5里有没有
#     真正的Caregiver.java文件(baseline=0, v1=0,一直没修复)
V1_TQ_PROXY = {
    "caregivers_sql_count": 1,
    "caregiver_class_hit": 0,
}


def load_diag_cases():
    return json.load(open(DIAG_FILE, encoding="utf-8"))


def _rank_of(fused, target_chunk_id):
    ids = [cid for cid, _ in fused]
    return ids.index(target_chunk_id) + 1 if target_chunk_id in ids else None


def run_diag_regression(embed_model, collection, bm25_store, openai_client):
    cases = load_diag_cases()
    known_ids = set(bm25_store["ids"])
    rows = []
    calls = []  # (latency_sec, usage)

    for case in cases:
        q = case["question"]
        target = case["chunk_id"]

        if target not in known_ids:
            print(f"  [跳过] {target} 在当前索引里找不到了(可能是重新分块后id漂移),不参与本次统计")
            continue

        vector_ids = retrieve_vector_ids(embed_model, collection, q)
        bm25_ids = retrieve_bm25_ids(bm25_store, q)
        fused_before = rrf_fuse(vector_ids, bm25_ids)

        rewrite_info = extract_query_entities(openai_client, q)
        calls.append((rewrite_info["latency_sec"], rewrite_info["usage"]))

        entity_ids = retrieve_entity_bm25_matches(bm25_store, rewrite_info["entity_tokens"])
        fused_after = rrf_fuse(vector_ids, bm25_ids, entity_ids=entity_ids)

        rank_before = _rank_of(fused_before, target)
        rank_after = _rank_of(fused_after, target)

        rows.append(
            {
                "chunk_id": target,
                "question": q,
                "core_entities": rewrite_info["core_entities"],
                "entity_tokens": rewrite_info["entity_tokens"],
                "hit_before": rank_before is not None,
                "hit_after": rank_after is not None,
                "rank_before": rank_before,
                "rank_after": rank_after,
            }
        )

    return rows, calls


def print_diag_report(rows):
    print(f"\n{'chunk_id':<12}{'命中(前)':<8}{'命中(后)':<8}{'排名(前)':<10}{'排名(后)':<10}{'结论':<6} core_entities  question")
    print("-" * 130)
    improved = regressed = same_hit = same_miss = 0
    for r in rows:
        before, after = r["hit_before"], r["hit_after"]
        if not before and after:
            tag, improved = "改善", improved + 1
        elif before and not after:
            tag, regressed = "退化", regressed + 1
        elif before and after:
            tag, same_hit = "仍中", same_hit + 1
        else:
            tag, same_miss = "仍不中", same_miss + 1
        print(
            f"{r['chunk_id']:<12}{str(before):<8}{str(after):<8}"
            f"{str(r['rank_before']):<10}{str(r['rank_after']):<10}{tag:<6}"
            f"{r['core_entities']}  {r['question'][:36]}"
        )
    total = len(rows)
    print("-" * 130)
    print(f"改善(不中->中): {improved}  退化(中->不中): {regressed}  "
          f"仍命中: {same_hit}  仍未命中: {same_miss}  (共{total}条)")


def print_diag_comparison(rows):
    print("\n" + "=" * 100)
    print("三方对比: baseline(不用rewrite) vs v1(子串匹配) vs v2(BM25打分)  ——14个诊断案例hit@5")
    print("=" * 100)
    print(f"{'chunk_id':<12}{'baseline':<10}{'v1':<10}{'v2(本次)':<10}结论")
    print("-" * 80)
    v1_hits = v2_hits = baseline_hits = 0
    v2_only_improve = v1_only_improve = both_improve = neither = 0
    for r in rows:
        cid = r["chunk_id"]
        baseline = r["hit_before"]
        v1 = V1_DIAG_HIT_AFTER.get(cid)
        v2 = r["hit_after"]
        if baseline:
            baseline_hits += 1
        if v1:
            v1_hits += 1
        if v2:
            v2_hits += 1

        if v1 is None:
            note = "(v1无此案例数据)"
        elif v1 and v2:
            note, both_improve = "v1/v2都命中", both_improve + 1
        elif v1 and not v2:
            note, v1_only_improve = "只有v1命中(v2退步)", v1_only_improve + 1
        elif v2 and not v1:
            note, v2_only_improve = "只有v2命中(v2比v1好)", v2_only_improve + 1
        else:
            note, neither = "两版都不中", neither + 1

        print(f"{cid:<12}{str(baseline):<10}{str(v1):<10}{str(v2):<10}{note}")

    print("-" * 80)
    print(f"命中总数: baseline={baseline_hits}  v1={v1_hits}  v2={v2_hits}  (满分14)")
    print(
        f"相对baseline的净收益: v1净增{v1_hits - baseline_hits}个  v2净增{v2_hits - baseline_hits}个"
    )
    print(f"v2相对v1: 只有v2命中(v2更好)={v2_only_improve}  只有v1命中(v2更差)={v1_only_improve}  "
          f"两版都命中={both_improve}  两版都不命中={neither}")


def _caregivers_sql_count(fused, bm25_store):
    count = 0
    for cid, _ in fused:
        _, meta = lookup_by_id(bm25_store, cid)
        if meta.get("chunk_type") == "sql_statement":
            count += 1
    return count


def _caregiver_class_hit(fused, bm25_store):
    for cid, _ in fused:
        _, meta = lookup_by_id(bm25_store, cid)
        file = meta.get("file") or ""
        if file.endswith("model\\Caregiver.java") or file.endswith("model/Caregiver.java"):
            return True
    return False


def run_test_questions_comparison(embed_model, collection, bm25_store, openai_client):
    print("\n" + "=" * 100)
    print("Part 2: TEST_QUESTIONS 定性对比(代码知识类,之前人工判断过对错,没有存标准chunk_id)")
    print("=" * 100)
    calls = []
    proxy = {}
    for label, q in TEST_QUESTIONS:
        vector_ids = retrieve_vector_ids(embed_model, collection, q)
        bm25_ids = retrieve_bm25_ids(bm25_store, q)
        fused_before = rrf_fuse(vector_ids, bm25_ids)

        rewrite_info = extract_query_entities(openai_client, q)
        calls.append((rewrite_info["latency_sec"], rewrite_info["usage"]))
        entity_ids = retrieve_entity_bm25_matches(bm25_store, rewrite_info["entity_tokens"])
        fused_after = rrf_fuse(vector_ids, bm25_ids, entity_ids=entity_ids)

        print(f"\n[{label}] {q}")
        print(f"  core_entities: {rewrite_info['core_entities']}  entity_tokens: {rewrite_info['entity_tokens']}")

        print("  --- 加rewrite前 top-5 ---")
        for cid, score in fused_before:
            _, meta = lookup_by_id(bm25_store, cid)
            name_part = f" | {meta['name']}" if meta.get("name") else ""
            print(f"    {cid}  {meta.get('file', '?')} | {meta.get('chunk_type', '?')}{name_part}")

        print("  --- 加rewrite后 top-5 ---")
        for cid, score in fused_after:
            _, meta = lookup_by_id(bm25_store, cid)
            name_part = f" | {meta['name']}" if meta.get("name") else ""
            print(f"    {cid}  {meta.get('file', '?')} | {meta.get('chunk_type', '?')}{name_part}")

        if "Caregivers表的SQL" in q:
            proxy["caregivers_sql_count_baseline"] = _caregivers_sql_count(fused_before, bm25_store)
            proxy["caregivers_sql_count_v2"] = _caregivers_sql_count(fused_after, bm25_store)
        if "Caregiver类" in q:
            proxy["caregiver_class_hit_baseline"] = _caregiver_class_hit(fused_before, bm25_store)
            proxy["caregiver_class_hit_v2"] = _caregiver_class_hit(fused_after, bm25_store)

    return calls, proxy


def print_tq_comparison(proxy):
    print("\n" + "=" * 100)
    print("三方对比: baseline vs v1 vs v2  ——TEST_QUESTIONS的2个可量化代理指标")
    print("=" * 100)
    print(
        f"{'指标':<38}{'baseline':<10}{'v1':<10}{'v2(本次)':<10}"
    )
    print("-" * 80)
    print(
        f"{'Caregivers-SQL: top5里sql_statement数':<38}"
        f"{str(proxy.get('caregivers_sql_count_baseline')):<10}"
        f"{str(V1_TQ_PROXY['caregivers_sql_count']):<10}"
        f"{str(proxy.get('caregivers_sql_count_v2')):<10}"
    )
    print(
        f"{'Caregiver类: top5命中真正Caregiver.java':<38}"
        f"{str(proxy.get('caregiver_class_hit_baseline')):<10}"
        f"{str(V1_TQ_PROXY['caregiver_class_hit']):<10}"
        f"{str(proxy.get('caregiver_class_hit_v2')):<10}"
    )


def print_cost_report(all_calls):
    latencies = [c[0] for c in all_calls]
    usages = [c[1] for c in all_calls]
    total_input = sum(u["input_tokens"] for u in usages)
    total_output = sum(u["output_tokens"] for u in usages)
    n = len(latencies)

    print("\n" + "=" * 100)
    print("Part 3: query rewriting新增的延迟/成本(相对原有hybrid检索是纯新增的一次LLM调用)")
    print("=" * 100)
    print(f"调用次数: {n}")
    print(
        f"延迟: 平均 {statistics.mean(latencies):.2f}s  中位数 {statistics.median(latencies):.2f}s  "
        f"最大 {max(latencies):.2f}s  最小 {min(latencies):.2f}s"
    )
    print(f"token总量: input={total_input}  output={total_output}")
    est_cost = total_input / 1e6 * PRICE_PER_1M_INPUT + total_output / 1e6 * PRICE_PER_1M_OUTPUT
    print(f"预估成本(按gpt-4o-mini官方定价估算,非精确账单): ${est_cost:.4f}  (共{n}次调用)")
    print(f"平均每次调用预估成本: ${est_cost / n:.6f}")


def main():
    embed_model, collection, bm25_store, openai_client = load_resources()
    print(f"索引记录数: {len(bm25_store['ids'])}")

    print("\n" + "=" * 100)
    print("Part 1: 14个已知诊断案例回归测试(A类7+B类3+C类4,全部中文提问)")
    print("=" * 100)
    diag_rows, diag_calls = run_diag_regression(embed_model, collection, bm25_store, openai_client)
    print_diag_report(diag_rows)
    print_diag_comparison(diag_rows)

    tq_calls, tq_proxy = run_test_questions_comparison(embed_model, collection, bm25_store, openai_client)
    print_tq_comparison(tq_proxy)

    print_cost_report(diag_calls + tq_calls)


if __name__ == "__main__":
    main()
