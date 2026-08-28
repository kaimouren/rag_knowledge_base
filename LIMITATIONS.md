# 已知局限性 / 未处理问题记录

这份文档诚实记录当前RAG系统里已知的检索缺陷和数据处理层面的取舍。原则和 `CLAUDE.md` 一致:能修的已经修了,修了之后的实际效果(包括没达到预期的情况)如实记录;暂不修的说清楚为什么现阶段不投入,以及权衡依据是什么。

## 1. 文本知识类的文件名索引修复(已实施,效果有限——如实记录)

### 背景

在35条RAGAS测试集的诊断中发现,文本知识类的检索命中率只有6.7%(15条里1条命中目标chunk)。对14个未命中案例做了逐条根因排查(取全库精确余弦相似度 + BM25精确排名,不只看top-10),分成三类:

- **A类 语言不匹配**(7例):中文query在当前BM25分词器下产出空token列表,向量相似度也很低(embedding模型是纯英文的all-MiniLM-L6-v2)
- **B类 词汇鸿沟**(3例:`txt_05591`、`txt_05618`、`txt_20002`):query里的关键词在目标chunk原文里字面上没出现
- **C类 检索池被稀释**(4例):目标chunk分数/排名其实不差,只是没挤进最终top-5

### 修复方案与实施

对B类,复用了之前修 Caregivers SQL 案例(`ingest/parse_code.py::parse_sql()`)时用的方案——把文件名拼进可索引文本。具体改动:`ingest/chunk_text.py::chunk_document()` 里,每个文本chunk存入索引的正文,现在会在最前面加一行不含路径的源文件名(比如 `Section 2 Worksheet.pdf`),质量检测(`detect_chunk_quality_flags`)和position定位仍然基于不含文件名的原始内容,避免失真。改完后重新跑了全流程:`run_chunk_text_full.py` → `merge_chunks.py` → `build_bm25_index.py` → `build_simple_index.py`,chunk总数 24311(文本21206 + 代码3105),索引全部重建。

### 实测结果(如实记录:未达到预期)

用B类的3个案例重新测了BM25排名和向量排名:

| chunk_id | 查询关键词 | BM25排名(修复前→后) | 向量排名(修复前→后) |
|---|---|---|---|
| txt_05591 | sql | 5857 → 5858 | 1469 → 1167 |
| txt_05618 | sql | 360 → 456 | 2773 → 2993 |
| txt_20002 | x | 1116 → 1125 | 1678 → 1284 |

**结论:三个案例里,BM25排名没有一个变好(两个还略微变差),向量排名两升一降,且没有一个进入最终top-5。** 根因排查后发现:这3个chunk对应的源文件名本身并不包含缺失的关键词——`txt_05591`/`txt_05618` 来自 `Section 2 Worksheet.pdf`/`Worksheet Solutions.docx.pdf`,`txt_20002` 来自 `IEOR4404_HW4.pdf`,文件名里都没有 "sql" 或和目标query强相关的词。这和最初参照的 Caregivers SQL 案例不同——那次是文件本身就叫类似"...SQL..."这样的名字,文件名天然带着缺失的关键词;这3个文本案例只是主题相关但命名方式不同,文件名修复补不上这个缺口。

另外,给全部21206个文本chunk统一加前缀,会小幅改变BM25的全局统计量(文档长度分布、IDF),这正是之前修分词器时观察到的K-means副作用同一种机制——对个别case可能有轻微负面影响,这里两个case排名变差就是这个机制的体现。

**是否保留这次改动**:保留。理由是:(1) 这个方案本身是正确的通用改进——文件名经常携带课程、作业编号等语义信息,能帮助本次未覆盖到的其他查询;(2) 对全局的负面影响量级很小,属于已知的、可解释的权衡,和分词器修复时的处理原则一致;(3) 但对这3个具体audit case没有解决问题,说明"词汇鸿沟"这个问题还没有根本性方案,如果要真正修复需要query expansion或更换多语言/领域适配的embedding模型,现阶段不在本轮投入范围内。

## 2. A类:中文query的检索局限(未处理,记录评估结论)

当前embedding模型(all-MiniLM-L6-v2)针对英文优化,中文query的向量检索和BM25分词均存在已知局限——这是基于语料库以英文为主(courses材料中文占比<1%)这一现状做出的性价比选择。如果后续中文查询需求增加,可考虑更换为multilingual-e5-large等多语言模型,但需要重新评估整体检索效果(担心引入新的副作用,类似之前BM25分词器修复引发的K-means案例连带影响)。

## 3. C类:检索池稀释(未处理,记录评估结论)

部分查询存在检索池被稀释的现象(真实相关内容排名接近但未进入最终top-5)。已知的解法包括query rewriting或扩大候选池,经评估后判断当前阶段暂不投入,原因是候选池扩大可能引入新噪声,且没有独立于语义/关键词检索之外的根本性解法(和之前K-means案例分析结论一致)。

## 4. 数据覆盖范围:6个文件主动排除/搁置

445个文本知识类PDF中,**439个**进入知识库构建流程(chunk库、索引、检索全部覆盖),**6个**因预算约束主动排除/搁置,完整名单和排除原因记录在 `ingest/_test_output/quality_manifest.json`(`quality: "pending"` + `pending_reason` 字段):

- **5个:`pending_reason: "deferred_low_priority"`** —公开教材/期刊论文,内容网络可查、非课程原创,优先级判断后本轮暂不处理:
  - `ECON 484\Polya_HowToSolveIt.pdf`
  - `ECON 484\casi.pdf`
  - `ECON 484\ISLRv2_website.pdf`
  - `ECON 484\First Week Reading\1177704711.pdf`(JSTOR期刊文章)
  - `IEOR E4106 Stochastic Model\Introduction to Probability Models, 11th Edition" by Ross.pdf`
- **1个:`pending_reason: "deferred_budget_timeout"`** —课程原创材料,Docling处理超时,需要额外LlamaParse预算才能重新解析,本轮预算内暂缓:
  - `IEOR E4004 Optimization Models & Methods\L6\IEOR4004-Class6.pdf`

> 注:manifest里没有单独的 `"excluded"` 状态——按之前的决定,所有排除/搁置的文件统一标记为 `quality: "pending"`,用 `pending_reason` 区分"低优先级搁置"和"预算/超时搁置"两种原因,已核对该字段在6条记录里都完整一致(`quality`/`pending_reason`/`note`/`course`/`pages` 均齐全),无需补齐。
