# 项目背景:课程笔记 RAG 知识库系统

## 项目目标
构建一个基于个人课程笔记的检索增强生成(RAG)系统,用于面试展示端到端工程能力。
核心原则:**技术描述必须诚实、可被面试追问验证**——清楚区分"设计"和"实现",不夸大能力(例如不要把这个项目描述成"AI Agent",除非真的加了具备自主决策/工具调用循环的模块)。

## 数据概况
- 数据源目录:`E:\project`,包含多门课程的笔记、作业、代码
- 原始文件总数 2604 个,清理 `._*`(macOS AppleDouble元数据,1264个)和 `.DS_Store`(160个)后剩余 1180 个有效文件(不含 `.claude` 工具配置目录;实测数字来自 `ingest/route_files.py` 的运行结果,可复现)
- 按 `ingest/route_files.py` 的五类路由统计(合计 1180):
  - 文本知识类 523个:PDF 445、TXT 52、MD 7、HTML 6、RMD 6、DOCX 5、PPTX 2
  - 代码知识类 256个:Jupyter Notebook 161、SQL 35、Java 34、R 17、Python 8、LP线性规划模型 1
  - 数据摘要类 152个:CSV 135、XLSX 17
  - 图片类 77个:PNG 61、JPG 15、JPEG 1
  - 排除类 172个:CLASS 36、XML(IDE配置) 32、PKL 22、IML 16、.gitignore 14、JAR 7、LICENSE样板文本 6、ZIP 4、DLL 4、.RData 4、M4A 4、YML(作业提交元数据) 3、MP4 3、text8语料库(约100MB×2) 2、JSON 1、EXE 1、.idea/.name 1、模型checkpoint残留 1

## 文件分类与处理策略

### 1. 文本知识类(核心内容,完整解析)
格式:pdf, docx, pptx, txt, md, rmd, html
- 常规文档 → **Docling**(本地开源,免费,处理单栏结构清晰的文档)
- 复杂扫描件/多栏布局/图表密集文档 → **LlamaParse**(云端API,VLM理解能力更强,有免费额度)
- 需要一个路由判断逻辑:根据文档特征(页数、是否检测到扫描件/复杂版面)决定走哪个工具

### 2. 代码知识类(完整索引,和文本知识库一起做)
格式:ipynb, py, r, sql, java
- **Jupyter Notebook**:用 `nbformat` 库本地解析(.ipynb本身是JSON格式,不需要OCR)
  - markdown cell 和 code cell 分开处理
  - code cell 作为完整单元保留,不按字符数硬切(避免语义丢失)
  - 是否保留执行输出(outputs)需要判断,若包含关键结果(如实验数值)则保留但单独标记类型
- **py/r/sql/java**:按语言语法结构解析(函数/类/存储过程为单位),保留docstring和注释作为独立可检索单元,代码块本身作为完整chunk不切碎

### 3. 数据摘要类(不做全量索引)
格式:csv, xlsx
- 只提取:表头字段名、行列规模、文件名/路径上下文
- 用这些信息生成一段摘要文本(如"该文件包含航班数据,字段包括出发地、到达地、日期..."),把摘要作为可检索的chunk
- 不索引具体数据内容本身

### 4. 图片类(视情况处理)
格式:png, jpg, jpeg
- 如果是课件截图/公式图/图表 → 走本地OCR(如 easyocr,支持中英文)
- 纯装饰性图片可以跳过

### 5. 排除类(不处理)
格式:class, jar, dll, exe, xml(IDE配置文件如.idea相关), iml, zip, pkl, m4a, mp4
以及所有 `._*` 和 `.DS_Store` 文件(需要在ingestion前批量清理)

## 整体技术架构(全流程)

```
数据摄入(按类型路由到不同解析器)
    ↓
统一JSON schema(text, source, file_type, metadata...)
    ↓
分块(Chunking)
  - 文本:MarkdownHeaderTextSplitter做结构切分 + RecursiveCharacterTextSplitter二次切分
  - 代码:按函数/cell为单位,不跨语义边界切分
  - (进阶)Contextual Retrieval:为每个chunk生成上下文说明前缀(Anthropic方法,提升检索时的语义完整性)
    ↓
索引构建
  - 向量索引:Chroma + sentence-transformers(如 BAAI/bge-large-zh-v1.5,支持中英双语)
  - 稀疏索引:BM25(rank_bm25库)
    ↓
Hybrid检索
  - 向量检索 + BM25并行检索
  - RRF(Reciprocal Rank Fusion)融合排序
  - (可选)cross-encoder重排序进一步提升精度
    ↓
生成(Generation)
  - 调用 Claude API 生成回答
  - Prompt设计需要求模型标注引用来源(source attribution)
    ↓
评估(Evaluation)
  - RAGAS框架:faithfulness、context precision/recall、answer relevancy
  - 需要人工构建QA测试集(20-50条起步)
    ↓
前端(Streamlit)
  - 聊天式问答界面
  - 展示检索到的来源片段,增强可解释性
```

## 关键工程原则
1. **成本控制**:优先用本地开源工具(Docling、本地embedding模型),复杂场景才调用云端API(LlamaParse、Claude API)
2. **模块化**:ingest / chunk / index / retrieve / generate / evaluate 各自独立成文件,方便单独测试和面试时逐模块讲解
3. **可复现性**:相同输入应产出一致的检索结果
4. **诚实性**:每个技术决策都要能说清楚"为什么这样做",拒绝为了简历好看而夸大功能范围

## 面试相关背景(供代码注释/README风格参考)
用户目标岗位:AI Application Engineer / Data Scientist / ML Engineer / Data Engineer
这个项目用于展示端到端RAG工程能力,需要在代码结构和文档中体现清晰的架构决策,便于面试时逐步讲解设计思路。
