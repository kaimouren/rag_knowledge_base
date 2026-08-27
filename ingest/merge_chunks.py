"""
把文本知识类chunk(run_chunk_text_full.py的输出)和代码知识类chunk
(parse_code.py的输出)转成同一套统一schema,合并成一份完整数据集。

字段映射说明(代码知识类原本没有这些字段,这里是新增的映射逻辑,不是
简单复制):
  - source_type: 固定填 "code"(文本那边是"text")
  - position: 代码知识类目前没有字符偏移量这类通用定位,不同格式各自的
    定位方式差异很大(ipynb用cell_index,sql用statement_index,py/r/java
    没有显式定位,只能靠name隐式排序)——这里如实保留原始定位信号,
    不是编一个假的统一格式出来凑数。
  - quality / chunk_quality_flag / extraction_confidence: 这是这次合并
    里唯一做了实质性判断的地方。之前对比两条产线schema时发现代码知识类
    完全没有可靠性维度的字段,尽管parse_code.py自己的注释里明确写了
    "ipynb/py/sql用ast/nbformat精确解析,r/java用正则+括号计数器近似解析,
    精度明显更低"——这次合并把这个已知的精度差异显式转成字段:
      ipynb_*/py_*/sql_* 系列 -> extraction_confidence="high", 无flag
      r_*/java_* 系列         -> extraction_confidence="medium",
                                  chunk_quality_flag=["regex_derived_lower_precision"]
    这不是新分析,是把 parse_code.py 里已经写清楚但没有变成数据字段的
    结论,补成下游可以直接按字段筛选的形式。
  - quality: 代码解析不走Docling/LlamaParse那条质量管线,好坏程度维度
    跟文本知识类不是一回事,这里如实填 None,不假装有一个"good"状态。

用法:
    python ingest/merge_chunks.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from parse_code import PARSERS
from run_parse_code_full import collect_files

TEXT_CHUNKS_PATH = Path(__file__).parent / "_test_output" / "text_chunks.json"
OUTPUT_PATH = Path(__file__).parent / "_test_output" / "combined_chunks.json"

HIGH_CONFIDENCE_TYPES = {
    "ipynb_code", "ipynb_markdown", "ipynb_merged",
    "py_function", "py_class", "py_toplevel", "py_parse_error",
    "sql_statement",
}
REGEX_DERIVED_TYPES = {
    "r_function", "r_toplevel",
    "java_method", "java_constructor", "java_toplevel",
}


def code_chunk_to_unified(chunk, idx: int) -> dict:
    chunk_type = chunk.metadata.get("chunk_type", "")

    if chunk_type in REGEX_DERIVED_TYPES:
        extraction_confidence = "medium"
        chunk_quality_flag = ["regex_derived_lower_precision"]
        source_note = "r/java用正则表达式+括号计数器近似解析函数/方法边界,精度低于ast/nbformat,详见parse_code.py注释里的局限性说明"
    else:
        extraction_confidence = "high"
        chunk_quality_flag = None
        source_note = ""

    position = None
    if "cell_index" in chunk.metadata:
        position = {"locator_type": "cell_index", "value": chunk.metadata["cell_index"]}
    elif "statement_index" in chunk.metadata:
        position = {"locator_type": "statement_index", "value": chunk.metadata["statement_index"]}

    return {
        "chunk_id": f"code_{idx:05d}",
        "text": chunk.text,
        "metadata": {
            "source_type": "code",
            "file": chunk.metadata.get("file"),
            "course": chunk.metadata.get("course"),
            "chunk_type": chunk_type,
            "name": chunk.metadata.get("name"),
            "position": position,
            "quality": None,  # 代码解析不走Docling/LlamaParse质量管线,这个维度不适用
            "chunk_quality_flag": chunk_quality_flag,
            "extraction_confidence": extraction_confidence,
            "source_note": source_note,
        },
    }


def collect_code_chunks_unified():
    code_chunks = []
    idx = 0
    for ext, parser in PARSERS.items():
        for path in collect_files(ext):
            try:
                chunks = parser(path)
            except Exception:
                continue
            for c in chunks:
                if not c.text or not c.text.strip():
                    continue
                code_chunks.append(code_chunk_to_unified(c, idx))
                idx += 1
    return code_chunks


def main():
    text_chunks = json.load(open(TEXT_CHUNKS_PATH, encoding="utf-8"))
    print(f"文本知识类chunk数: {len(text_chunks)}")

    code_chunks = collect_code_chunks_unified()
    print(f"代码知识类chunk数: {len(code_chunks)}")

    combined = text_chunks + code_chunks
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"\n合并后总chunk数: {len(combined)}")
    print(f"已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
