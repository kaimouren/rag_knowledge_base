"""
RAGAS评估 - 对35条最终测试集(qa_candidates_for_review.csv)跑完整评估

环境说明(重要,venv重建时需要重新处理): ragas==0.4.3 有个打包bug,
ragas/llms/base.py 硬import了 langchain_community.chat_models.vertexai
这个模块,但当前 langchain-community(0.4.x)已经把VertexAI相关集成移出去了,
导致ragas完全导入不了。本机venv里已经对安装好的ragas包做了本地patch
(把这两个import包在try/except里,失败就用占位类替代——我们用OpenAI不用
VertexAI,不影响功能),patch内容见 .venv/Lib/site-packages/ragas/llms/base.py
开头。这个patch不在项目代码里,venv重建后会丢,需要重新打一遍。

裁判LLM/embedding: 用OpenAI(gpt-4o-mini + text-embedding-3-small),不是
CLAUDE.md原定的Claude API——用户当前没有Anthropic key,延续RAG生成阶段
已经在用的选择。embedding_factory 必须显式传 interface="legacy",默认的
"auto"模式在当前ragas版本下会返回一个不兼容旧版AnswerRelevancy/Faithfulness
指标类的embeddings接口(缺 embed_query 方法),实测验证过这个坑,不是猜的。

四个指标为什么不是对所有35条一视同仁地跑(详见跟用户的讨论,不是我自己
拍脑袋加的限制):
  - Faithfulness: 只比对"回答"和"检索到的context",不用ground_truth,
    对abstention类型完全安全,全部35条都正常跑。
  - Context Precision / Context Recall: 用ground_truth做参照,对
    no_content(3条,ground_truth本身极短、只有一句"查阅原文件")算出来的
    分数区分度低,但不是错的,照常跑,报告里会标注"参考价值有限"。
  - Answer Relevancy: 查过RAGAS官方文档,这个指标的计算机制(从回答反推
    问题、算跟原问题的余弦相似度)明确会惩罚"不完整/回避性"的回答——
    对no_content这3条(标准答案本身就是"信息不足,请查阅原文"这种诚实
    拒答)用这个指标算出来的低分不代表系统表现差,反而可能是系统诚实的
    体现,直接套用会产生系统性误判。所以no_content这3条不跑标准
    AnswerRelevancy,换成下面的自定义 abstention_honesty 检查。

用法(分两阶段,阶段一先把检索+生成结果存下来,阶段二单独跑RAGAS打分,
中间失败不用重跑贵的那部分):
    python evaluate/run_ragas_eval.py pipeline   # 阶段一: 跑hybrid_query,存结果
    python evaluate/run_ragas_eval.py ragas       # 阶段二: 读结果,跑RAGAS打分
    python evaluate/run_ragas_eval.py report      # 阶段三: 分组统计+出报告
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "query"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ingest"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

TESTSET_CSV = Path(__file__).parent / "qa_candidates_for_review.csv"
PIPELINE_RESULTS_PATH = Path(__file__).parent / "pipeline_results.json"
RAGAS_RESULTS_CSV = Path(__file__).parent / "ragas_results.csv"

GENERATION_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"


# ---------------------------------------------------------------------------
# 阶段一: 对每条测试用例跑 hybrid_query.py 的完整检索+生成流程
# ---------------------------------------------------------------------------
def run_pipeline():
    from hybrid_query import (
        load_resources, retrieve_vector_ids, retrieve_bm25_ids, rrf_fuse,
        lookup_by_id, build_prompt, generate_answer,
    )

    rows = list(csv.DictReader(open(TESTSET_CSV, encoding="utf-8-sig")))
    print(f"测试集共 {len(rows)} 条")

    embed_model, collection, bm25_store, openai_client = load_resources()
    results = []

    for i, row in enumerate(rows, 1):
        question = row["generated_question"]
        vector_ids = retrieve_vector_ids(embed_model, collection, question)
        bm25_ids = retrieve_bm25_ids(bm25_store, question)
        fused = rrf_fuse(vector_ids, bm25_ids)

        items = [lookup_by_id(bm25_store, cid) for cid, _score in fused]
        contexts = [text for text, _meta in items]
        retrieved_ids = [cid for cid, _score in fused]

        prompt = build_prompt(question, items)
        response = generate_answer(openai_client, prompt)

        results.append({
            "chunk_id": row["chunk_id"],
            "test_category": row["test_category"],
            "abstention_type": row["abstention_type"],
            "source_type": row["source_type"],
            "chunk_type": row["chunk_type"],
            "quality": row["quality"],
            "question": question,
            "reference": row["generated_ground_truth"],
            "retrieved_ids": retrieved_ids,
            "target_retrieved": row["chunk_id"] in retrieved_ids,
            "contexts": contexts,
            "response": response,
        })
        print(f"  [{i}/{len(rows)}] {row['chunk_id']} ({row['test_category']}) 完成, "
              f"目标chunk是否被检索到={row['chunk_id'] in retrieved_ids}")

    with open(PIPELINE_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    hit = sum(1 for r in results if r["target_retrieved"])
    print(f"\n完成: {len(results)} 条, 目标chunk被检索到 {hit}/{len(results)} 条")
    print(f"已保存至: {PIPELINE_RESULTS_PATH}")


# ---------------------------------------------------------------------------
# 阶段二: 跑RAGAS打分
# ---------------------------------------------------------------------------
def make_ragas_llm_and_embeddings():
    from openai import OpenAI
    from ragas.llms import llm_factory
    from ragas.embeddings import embedding_factory

    client = OpenAI()
    llm = llm_factory(GENERATION_MODEL, provider="openai", client=client)
    # interface="legacy" 是踩坑后确认必须显式传的参数,见文件头注释
    embeddings = embedding_factory(
        provider="openai", model=EMBEDDING_MODEL, client=client, interface="legacy"
    )
    return client, llm, embeddings


ABSTENTION_HONESTY_PROMPT = """判断下面这个"生成回答"是否诚实地表达了"信息不足/无法给出可靠答案/建议
查阅原始材料"这类意思,而不是编造了一个看似完整但缺乏依据的答案。

问题: {question}
生成回答: {response}

只返回JSON: {{"is_honest_abstention": true或false, "reason": "一句话说明理由"}}
"""


def check_abstention_honesty(client, question: str, response: str) -> dict:
    resp = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": ABSTENTION_HONESTY_PROMPT.format(question=question, response=response)}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def run_ragas():
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextPrecisionWithReference, LLMContextRecall

    results = json.load(open(PIPELINE_RESULTS_PATH, encoding="utf-8"))
    client, llm, embeddings = make_ragas_llm_and_embeddings()

    no_content = [r for r in results if r["abstention_type"] == "no_content"]
    others = [r for r in results if r["abstention_type"] != "no_content"]
    print(f"no_content(跳过标准answer_relevancy): {len(no_content)} 条")
    print(f"其余(跑完整4指标): {len(others)} 条")

    # --- 其余32条: 完整4个标准指标 ---
    def to_ragas_row(r):
        return {
            "user_input": r["question"],
            "response": r["response"],
            "retrieved_contexts": r["contexts"],
            "reference": r["reference"],
        }

    print("\n=== 跑 standard + confident_with_caveat (32条, 4个指标) ===")
    ds_others = EvaluationDataset.from_list([to_ragas_row(r) for r in others])
    result_others = evaluate(
        dataset=ds_others,
        metrics=[Faithfulness(), AnswerRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=llm, embeddings=embeddings,
    )
    df_others = result_others.to_pandas()
    for i, r in enumerate(others):
        r["faithfulness"] = float(df_others.iloc[i]["faithfulness"])
        r["answer_relevancy"] = float(df_others.iloc[i]["answer_relevancy"])
        r["context_precision"] = float(df_others.iloc[i]["llm_context_precision_with_reference"])
        r["context_recall"] = float(df_others.iloc[i]["context_recall"])
        r["abstention_honesty"] = ""

    # --- no_content 3条: Faithfulness+Precision+Recall正常跑,answer_relevancy换成自定义检查 ---
    print("\n=== 跑 no_content (3条, 跳过标准answer_relevancy) ===")
    ds_nc = EvaluationDataset.from_list([to_ragas_row(r) for r in no_content])
    result_nc = evaluate(
        dataset=ds_nc,
        metrics=[Faithfulness(), LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=llm, embeddings=embeddings,
    )
    df_nc = result_nc.to_pandas()
    for i, r in enumerate(no_content):
        r["faithfulness"] = float(df_nc.iloc[i]["faithfulness"])
        r["answer_relevancy"] = None  # 明确留空,不套用会产生误判的标准指标
        r["context_precision"] = float(df_nc.iloc[i]["llm_context_precision_with_reference"])
        r["context_recall"] = float(df_nc.iloc[i]["context_recall"])
        honesty = check_abstention_honesty(client, r["question"], r["response"])
        r["abstention_honesty"] = honesty["is_honest_abstention"]
        r["abstention_honesty_reason"] = honesty["reason"]
        print(f"  {r['chunk_id']}: is_honest_abstention={honesty['is_honest_abstention']} ({honesty['reason']})")

    all_results = others + no_content
    with open(PIPELINE_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n打分完成,已写回 {PIPELINE_RESULTS_PATH}")


# ---------------------------------------------------------------------------
# 阶段三: 分组统计
# ---------------------------------------------------------------------------
def run_report():
    import pandas as pd

    results = json.load(open(PIPELINE_RESULTS_PATH, encoding="utf-8"))
    df = pd.DataFrame(results)

    # 保存逐条明细
    detail_cols = [
        "chunk_id", "test_category", "abstention_type", "source_type", "chunk_type", "quality",
        "target_retrieved", "faithfulness", "answer_relevancy", "context_precision", "context_recall",
        "abstention_honesty",
    ]
    df[detail_cols].to_csv(RAGAS_RESULTS_CSV, index=False, encoding="utf-8-sig")
    print(f"逐条明细已保存至: {RAGAS_RESULTS_CSV}\n")

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    def summarize(group_df, label):
        print(f"\n--- {label} (n={len(group_df)}) ---")
        for m in metrics:
            vals = group_df[m].dropna()
            if len(vals) == 0:
                print(f"  {m}: (无有效数据)")
            else:
                print(f"  {m}: mean={vals.mean():.4f}  (n={len(vals)})")
        hit_rate = group_df["target_retrieved"].mean()
        print(f"  目标chunk检索命中率: {hit_rate:.1%}")

    print("=" * 80)
    print("按 test_category 分组 (standard vs abstention)")
    print("=" * 80)
    for cat, g in df.groupby("test_category"):
        summarize(g, f"test_category={cat}")

    print("\n" + "=" * 80)
    print("abstention 内部再拆 (no_content vs confident_with_caveat)")
    print("=" * 80)
    abst = df[df["test_category"] == "abstention"]
    for t, g in abst.groupby("abstention_type"):
        summarize(g, f"abstention_type={t}")
        if t == "no_content":
            honest_rate = g["abstention_honesty"].apply(lambda x: x is True).mean()
            print(f"  诚实承认局限性比例(abstention_honesty=true): {honest_rate:.1%}")

    print("\n" + "=" * 80)
    print("按 source_type 分组 (文本 vs 代码)")
    print("=" * 80)
    for st, g in df.groupby("source_type"):
        summarize(g, f"source_type={st}")

    print("\n" + "=" * 80)
    print("按 quality 分组 (good/degraded/llamaparse_done,只对文本知识类有意义)")
    print("=" * 80)
    text_df = df[df["source_type"] == "text"]
    for q, g in text_df.groupby("quality"):
        if q:
            summarize(g, f"quality={q}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "pipeline"
    if stage == "pipeline":
        run_pipeline()
    elif stage == "ragas":
        run_ragas()
    elif stage == "report":
        run_report()
    else:
        print(f"未知阶段: {stage}(应为 pipeline/ragas/report)")
        sys.exit(1)
