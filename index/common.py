"""
索引构建共享逻辑

build_simple_index.py(Chroma向量索引)和 build_bm25_index.py(BM25索引)
必须用完全相同顺序、相同ID的chunk集合,RRF融合排序时才能靠id对齐两边
结果——所以把"收集全部代码chunk"这一步抽出来公用,而不是两边各写一份,
避免以后改动 parse_code.py 后两边遍历顺序不小心跑偏、id对不上。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "ingest"))

from parse_code import PARSERS
from run_parse_code_full import collect_files

COMBINED_CHUNKS_PATH = Path(__file__).parent.parent / "ingest" / "_test_output" / "combined_chunks.json"


def load_combined_chunks():
    """加载 merge_chunks.py 产出的统一schema chunk数据集(文本+代码,24297条),
    build_simple_index.py 和 build_bm25_index.py 都从这里读,保证两边id/顺序一致。
    每条记录已经有 chunk_id(txt_00000/code_00000),直接沿用,不用common.chunk_id()
    重新编号。
    """
    with open(COMBINED_CHUNKS_PATH, encoding="utf-8") as f:
        return json.load(f)


def collect_all_chunks(verbose: bool = True):
    chunks = []
    for ext, parser in PARSERS.items():
        files = collect_files(ext)
        if verbose:
            print(f"  {ext}: {len(files)} 个文件")
        for path in files:
            try:
                chunks.extend(parser(path))
            except Exception as e:
                if verbose:
                    print(f"    [跳过,解析失败] {path}: {type(e).__name__}: {e}")

    chunks = [c for c in chunks if c.text and c.text.strip()]
    return chunks


def chunk_id(index: int) -> str:
    return f"chunk_{index:05d}"


def sanitize_metadata(metadata: dict) -> dict:
    """Chroma metadata 只接受 str/int/float/bool,不接受 None/list/dict。
    合并文本知识类chunk后新增了 position 这种嵌套dict字段(比如
    {"char_start":0,"char_end":283,"estimated":true,...}),原来只处理
    list 和 None 的版本不够用了,加一条 dict -> JSON字符串 的转换。"""
    clean = {}
    for k, v in metadata.items():
        if v is None:
            continue
        if isinstance(v, list):
            clean[k] = ",".join(str(x) for x in v)
        elif isinstance(v, dict):
            clean[k] = json.dumps(v, ensure_ascii=False)
        else:
            clean[k] = v
    return clean
