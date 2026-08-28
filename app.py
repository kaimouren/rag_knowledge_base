"""
Streamlit前端: 复用 query/hybrid_query.py 的检索+生成逻辑,加一层问答UI。

只做本地演示用,不考虑线上部署(没有登录/多用户/持久化会话这些)。
检索/生成逻辑本身不在这里重写,全部从 hybrid_query.py 导入——这个文件
只负责UI交互和"把检索结果展示成人能看懂的来源列表"这件事。

本地启动:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "query"))
sys.path.insert(0, str(ROOT_DIR / "index"))

from hybrid_query import (  # noqa: E402
    build_prompt,
    generate_answer,
    load_resources,
    lookup_by_id,
    retrieve_bm25_ids,
    retrieve_vector_ids,
    rrf_fuse,
)

# 已经验证过检索效果的问题(见 hybrid_query.py::TEST_QUESTIONS),演示时
# 点一下就能查,不用现场打字
EXAMPLE_QUESTIONS = [
    "get_initial_centroids这个函数是做什么的?",
    "R语言里怎么模拟一个有偏硬币(biased coin)?",
    "有没有创建Caregivers表的SQL语句?",
    "vaccine scheduler项目里Caregiver类是怎么实现的?",
    "CSE 416的作业里有没有实现K-means相关的代码?",
]

# chunk_quality_flag -> 给用户看的中文提示。这5个是目前全库实际出现过的
# 全部取值(见 chunk_text.py / parse_code.py / summarize_data.py 里各自
# 的检测逻辑),不是随便编的,新增flag类型时要记得同步补充这里
FLAG_MESSAGES = {
    "formula_missing": "此来源可能存在公式解析缺失(原文档解析时数学公式未被正确识别)",
    "has_unresolved_image": "此来源包含未解析的图片内容(图片本身没有转成文字)",
    "fragmented": "此来源文本存在碎片化迹象(常见于扫描件/复杂排版OCR质量不佳的情况)",
    "regex_derived_lower_precision": "此来源(R/Java代码)由正则表达式近似解析得到,精度低于AST精确解析",
    "possible_no_header": "检测到该数据文件可能没有表头行,展示的是首行原始内容,不一定是真实字段名",
}

MAX_CHUNK_CHARS_SHOWN = 2000


def _normalize_flags(raw):
    """chunk_quality_flag 在 combined_chunks.json 里是list,但存进BM25/Chroma
    之前会经过 index/common.py::sanitize_metadata() 转成逗号拼接的字符串
    (Chroma metadata不接受list类型)。这里读的是BM25 store,拿到的就是那个
    字符串,不能直接 for flag in flags 遍历——那样会把字符串拆成一个个
    字符(比如"formula_missing"变成"f""o""r""m"...)。按逗号切回列表,同时
    兼容万一以后传进来的就是真list的情况。
    """
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    return [f for f in str(raw).split(",") if f]


@st.cache_resource(show_spinner="正在加载模型和索引(第一次启动会慢一些)...")
def get_resources():
    embed_model, collection, bm25_store, openai_client = load_resources()
    metadatas = bm25_store["metadatas"]
    stats = {
        "total_chunks": len(bm25_store["ids"]),
        "courses": sorted({m.get("course") for m in metadatas if m.get("course")}),
        "files": len({m.get("file") for m in metadatas if m.get("file")}),
    }
    return embed_model, collection, bm25_store, openai_client, stats


def render_sources(items):
    st.markdown("### 检索来源")
    for cid, score, text, meta in items:
        label = meta.get("file", "?")
        if meta.get("name"):
            label += f"  ·  {meta['name']}"
        with st.expander(label):
            st.caption(
                f"课程: {meta.get('course', '?')}  |  "
                f"类型: {meta.get('source_type', '?')}/{meta.get('chunk_type', '?')}  |  "
                f"chunk_id: {cid}  |  RRF分数: {score:.5f}"
            )

            flags = _normalize_flags(meta.get("chunk_quality_flag"))
            for flag in flags:
                st.warning(f"⚠️ {FLAG_MESSAGES.get(flag, flag)}")

            shown_text = text
            if len(text) > MAX_CHUNK_CHARS_SHOWN:
                shown_text = text[:MAX_CHUNK_CHARS_SHOWN] + f"\n...(已截断,原文共{len(text)}字符)"
            st.text(shown_text)


def run_query(question: str, embed_model, collection, bm25_store, openai_client):
    with st.spinner("检索中..."):
        vector_ids = retrieve_vector_ids(embed_model, collection, question)
        bm25_ids = retrieve_bm25_ids(bm25_store, question)
        fused = rrf_fuse(vector_ids, bm25_ids)

        items = []
        for cid, score in fused:
            text, meta = lookup_by_id(bm25_store, cid)
            items.append((cid, score, text, meta))

        prompt = build_prompt(question, [(text, meta) for _, _, text, meta in items])
        answer = generate_answer(openai_client, prompt)

    st.markdown("### 回答")
    st.write(answer)
    render_sources(items)


def main():
    st.set_page_config(page_title="课程笔记RAG知识库", page_icon="📚")
    st.title("📚 课程笔记 RAG 知识库")
    st.caption("基于个人课程笔记(讲义/作业/代码/数据集)构建的检索增强问答系统")

    try:
        embed_model, collection, bm25_store, openai_client, stats = get_resources()
    except Exception as e:
        st.error(
            f"资源加载失败: {type(e).__name__}: {e}\n\n"
            f"请检查 .env 里的 OPENAI_API_KEY 是否已配置,以及 index/ 下的索引文件是否存在。"
        )
        st.stop()

    with st.sidebar:
        st.header("知识库概况")
        st.metric("chunk总数", stats["total_chunks"])
        st.metric("覆盖文件数", stats["files"])
        st.metric("覆盖课程数", len(stats["courses"]))
        with st.expander("课程列表"):
            for c in stats["courses"]:
                st.write(f"- {c}")
        st.divider()
        st.markdown(
            "**检索方式**: 向量检索(sentence-transformers)+ BM25稀疏检索"
            "并行召回,用RRF(Reciprocal Rank Fusion)融合排序,不是单一检索"
            "方式。检索到的片段会连同来源信息一起交给LLM生成回答。"
        )

    if "question_input" not in st.session_state:
        st.session_state.question_input = ""
    if "should_query" not in st.session_state:
        st.session_state.should_query = False

    st.markdown("**示例问题**(点击直接查询,之前验证过检索效果):")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, eq in zip(cols, EXAMPLE_QUESTIONS):
        short_label = eq if len(eq) <= 14 else eq[:14] + "..."
        if col.button(short_label, key=f"example_{eq}", help=eq):
            st.session_state.question_input = eq
            st.session_state.should_query = True

    question = st.text_input("输入你的问题", key="question_input")
    ask_clicked = st.button("提问", type="primary")

    should_query = ask_clicked or st.session_state.should_query
    st.session_state.should_query = False

    if should_query:
        if not question.strip():
            st.warning("请输入问题,或者点击上面的示例问题。")
        else:
            run_query(question, embed_model, collection, bm25_store, openai_client)


if __name__ == "__main__":
    main()
