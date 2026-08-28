# 课程笔记 RAG 知识库系统

基于个人课程笔记(讲义、作业、代码)构建的检索增强生成(RAG)系统。用于面试展示端到端工程能力，不是生产系统。

**关于"AI Agent"这个词**：这个项目里没有自主决策/工具调用循环的模块，是一条按固定步骤走的RAG流水线（解析→分块→索引→检索→生成→评估），请不要把它理解成agent。

## 架构

```
数据摄入(ingest/)
  route_files.py      按后缀名把文件分成 文本知识类/代码知识类/数据摘要类/图片类/排除类
  parse_text.py        PDF/DOCX/PPTX 用 Docling 解析成markdown
  quality_router.py    检测Docling解析质量(乱码/公式丢失/碎片化),决定是否需要转LlamaParse
  llamaparse_test.py / llamaparse_batch.py   质量不达标的文档转LlamaParse重新解析
  quality_manifest.py  汇总每个文件的处理路径决策(good/degraded/llamaparse_done/pending/excluded)
  parse_code.py        ipynb(nbformat)/py(ast)/sql(分号切分)/r,java(正则+括号计数器) 解析成chunk
  chunk_text.py         文本知识类的markdown按标题+长度二次切分成chunk,标注chunk级质量flag
  merge_chunks.py       文本chunk+代码chunk 合并成统一schema

索引(index/)
  build_simple_index.py   chunk向量化存入Chroma
  build_bm25_index.py     chunk建BM25稀疏索引(自定义分词器,处理下划线/驼峰命名拆分)

检索+生成(query/)
  simple_query.py   纯向量检索 -> 拼prompt -> LLM生成回答
  hybrid_query.py    向量检索 + BM25检索,RRF融合排序 -> 生成回答

评估(evaluate/)
  build_qa_candidates.py   从chunk库分层抽样,LLM生成候选QA测试集,人工审核筛选
  run_ragas_eval.py        跑RAGAS四项指标(faithfulness/answer_relevancy/context_precision/context_recall),
                            分组统计(standard vs abstention测试用例分开评估,详见脚本内注释)
```

## 环境配置

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash;其他平台按需调整
pip install -r requirements.txt
cp .env.example .env
```

`.env` 需要的key：

```
ANTHROPIC_API_KEY=...
LLAMA_CLOUD_API_KEY=...
OPENAI_API_KEY=...
```

**`.env.example` 里只列了前两个，`OPENAI_API_KEY` 没写进去（当前生成/评估环节实际用的是OpenAI而不是Claude API，是开发过程中的临时选择，见下方"已知偏差"），自己pull下来之后手动加上这一行、填真实key即可，不影响其他配置方式。**

## 数据说明

`ingest/`, `index/`, `evaluate/` 目录下的中间产物(解析结果、索引文件、生成的测试集数据)不在这个仓库里——都是从个人课程笔记衍生出来的内容，`.gitignore` 里排除了。这个仓库只包含处理这些数据用的**代码**，没有数据本身。

## 已知偏差 / 局限性(诚实记录，不是全部修复完的状态)

- **生成阶段用的是OpenAI，不是CLAUDE.md原定的Claude API**——开发过程中手头没有现成的Anthropic key，先用OpenAI把链路跑通，`query/simple_query.py`把生成逻辑单独封装成一个函数，理论上换回Claude API只需要改这一处。
- **r/java 的代码解析用正则表达式+括号计数器，不是真正的语法分析**——精度明显低于 ipynb(nbformat) 和 py(ast) 这两条路径，已经在 `parse_code.py` 里用 `extraction_confidence`/`chunk_quality_flag` 字段显式标注了这个精度差异，不是没发现。
- **BM25分词器对纯中文query支持有限**——评估阶段实测发现，纯中文提问会导致BM25分词器产出空token列表（详见 `index/build_bm25_index.py` 的注释和评估记录），这是当前embedding模型（all-MiniLM-L6-v2，纯英文模型）选型和BM25分词器共同的局限，还没有修复。
- **ragas==0.4.3 有个已知的第三方包兼容性bug**（硬import了已被上游移除的VertexAI集成），本地venv里打了patch才能跑，`evaluate/run_ragas_eval.py` 文件头有完整说明，venv重建后需要重新打这个patch。
- **RAGAS评估集只有35条**，覆盖文本知识类三种质量状态(good/degraded/llamaparse_done)和代码知识类五种格式，样本量小，评估结果只能看方向，不是统计意义上的严谨结论。

更详细的检索命中率根因排查记录(中文query局限、检索池稀释、文件名索引修复的实测效果)见 [LIMITATIONS.md](LIMITATIONS.md)。
