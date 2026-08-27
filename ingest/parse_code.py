"""
代码知识类解析模块

CLAUDE.md "文件分类与处理策略 -> 2. 代码知识类" 规定: ipynb/py/r/sql/java
按语言语法结构解析(函数/类/存储过程为单位),保留 docstring 和注释作为独立
可检索单元,代码块本身作为完整 chunk 不切碎。

四种格式用了三种不同精度的解析手段,精度从高到低:
  - ipynb: nbformat 直接读取结构化 JSON,100% 准确,不是"解析"而是"读取"。
  - py:    ast 模块做语法树分析,和 Python 解释器用的是同一套语法规则,
           理论上能处理所有合法 Python 代码(装饰器、异步函数、嵌套定义等)。
  - sql:   没有用真正的 SQL 解析器(court 项目体量下不值得引入 sqlparse 之类
           的依赖),用了一个感知字符串/注释边界的分号切分器,能正确处理
           字符串里出现分号的情况,但不理解 SQL 语法本身。
  - r/java: 没有对应语言的官方 ast 模块可用(R 有 `parser::getParseData` 但
           要装 R 环境,Java 需要 javalang 等三方库),这里用正则表达式找函数/
           方法的"签名行",再用一个字符串/注释感知的括号计数器找函数体的
           结束位置。这是全模块里精度最低的部分,具体局限见各自函数的注释。

统一输出 Chunk(text, metadata) 结构,不区分格式来源,方便下游分块/索引
模块统一处理,不需要为每种语言单独写接入逻辑。
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

DATA_ROOT = Path(r"E:\project")


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def _course_of(path: Path) -> str:
    try:
        return path.relative_to(DATA_ROOT).parts[0]
    except ValueError:
        return ""


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(DATA_ROOT))
    except ValueError:
        return str(path)


# ===========================================================================
# 1) Jupyter Notebook (.ipynb) — nbformat
# ===========================================================================
# 输出裁剪上限: 有些作业 cell 会 print 出很长的训练日志(比如之前测试 Docling
# 时见过的逐 epoch loss 打印),整段塞进 chunk 里既没有检索价值又会让向量
# embedding 的语义被稀释,所以超过这个长度就截断并标注。
MAX_OUTPUT_CHARS = 2000


def _extract_cell_outputs(cell: dict) -> str:
    """把 code cell 的 outputs 拼成一段文字,不单独成 chunk,而是附加在代码
    chunk 后面。图片类输出(matplotlib图等)没有文本内容,只留一个占位说明。
    """
    pieces = []
    for out in cell.get("outputs", []):
        out_type = out.get("output_type")
        if out_type == "stream":
            text = "".join(out.get("text", []))
            if text.strip():
                pieces.append(text)
        elif out_type in ("execute_result", "display_data"):
            data = out.get("data", {})
            if "text/plain" in data:
                text = "".join(data["text/plain"])
                if text.strip():
                    pieces.append(text)
            elif data:
                # 只有图片/HTML等非文本输出,没有可检索的文字内容
                mime_types = ", ".join(data.keys())
                pieces.append(f"[非文本输出: {mime_types}]")
        elif out_type == "error":
            ename = out.get("ename", "")
            evalue = out.get("evalue", "")
            pieces.append(f"[执行报错] {ename}: {evalue}")

    combined = "\n".join(p.strip() for p in pieces if p.strip())
    if len(combined) > MAX_OUTPUT_CHARS:
        combined = combined[:MAX_OUTPUT_CHARS] + f"\n...(输出过长,已截断,原长度{len(combined)}字符)"
    return combined


def parse_ipynb(path: Path) -> List[Chunk]:
    import nbformat

    nb = nbformat.read(str(path), as_version=4)
    course = _course_of(path)
    rel = _rel(path)

    chunks: List[Chunk] = []
    pending_markdown: Optional[tuple] = None  # (text, cell_index)

    def flush_pending_markdown():
        nonlocal pending_markdown
        if pending_markdown is not None:
            text, idx = pending_markdown
            if text.strip():
                chunks.append(
                    Chunk(
                        text=text,
                        metadata={
                            "file": rel,
                            "course": course,
                            "chunk_type": "ipynb_markdown",
                            "cell_index": idx,
                        },
                    )
                )
            pending_markdown = None

    for i, cell in enumerate(nb.cells):
        cell_type = cell.get("cell_type")
        source = cell.get("source", "")

        if cell_type == "markdown":
            # 连续两个 markdown cell: 前一个显然没有紧跟的 code cell 可合并,
            # 先把它单独落盘,再把当前这个记为待合并候选
            flush_pending_markdown()
            pending_markdown = (source, i)

        elif cell_type == "code":
            if not source.strip():
                flush_pending_markdown()
                continue

            outputs_text = _extract_cell_outputs(cell)
            code_block = source
            if outputs_text:
                code_block = f"{source}\n\n输出:\n{outputs_text}"

            if pending_markdown is not None and pending_markdown[1] == i - 1:
                # markdown 紧跟着这个 code cell,大概率是在解释下面的代码,合并
                md_text, md_idx = pending_markdown
                merged_text = f"{md_text}\n\n```python\n{code_block}\n```"
                chunks.append(
                    Chunk(
                        text=merged_text,
                        metadata={
                            "file": rel,
                            "course": course,
                            "chunk_type": "ipynb_merged",
                            "cell_index": [md_idx, i],
                        },
                    )
                )
                pending_markdown = None
            else:
                flush_pending_markdown()
                chunks.append(
                    Chunk(
                        text=code_block,
                        metadata={
                            "file": rel,
                            "course": course,
                            "chunk_type": "ipynb_code",
                            "cell_index": i,
                        },
                    )
                )
        else:
            # raw cell 等罕见类型: 跳过,但打断"紧跟"关系
            flush_pending_markdown()

    flush_pending_markdown()
    return chunks


# ===========================================================================
# 2) Python (.py) — ast
# ===========================================================================
def parse_py(path: Path) -> List[Chunk]:
    course = _course_of(path)
    rel = _rel(path)
    source = path.read_text(encoding="utf-8", errors="replace")

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        return [
            Chunk(
                text=source,
                metadata={
                    "file": rel,
                    "course": course,
                    "chunk_type": "py_parse_error",
                    "error": f"SyntaxError: {e}",
                },
            )
        ]

    chunks: List[Chunk] = []
    toplevel_segments: List[str] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            segment = ast.get_source_segment(source, node) or ""
            chunks.append(
                Chunk(
                    text=segment,
                    metadata={
                        "file": rel,
                        "course": course,
                        "chunk_type": "py_function",
                        "name": node.name,
                        "docstring": bool(ast.get_docstring(node)),
                    },
                )
            )
        elif isinstance(node, ast.ClassDef):
            segment = ast.get_source_segment(source, node) or ""
            chunks.append(
                Chunk(
                    text=segment,
                    metadata={
                        "file": rel,
                        "course": course,
                        "chunk_type": "py_class",
                        "name": node.name,
                        "docstring": bool(ast.get_docstring(node)),
                    },
                )
            )
        else:
            segment = ast.get_source_segment(source, node)
            if segment:
                toplevel_segments.append(segment)

    if toplevel_segments:
        chunks.append(
            Chunk(
                text="\n\n".join(toplevel_segments),
                metadata={
                    "file": rel,
                    "course": course,
                    "chunk_type": "py_toplevel",
                    "name": None,
                },
            )
        )

    return chunks


# ===========================================================================
# 3) SQL (.sql) — 分号切分(带字符串/注释边界感知)
# ===========================================================================
# 没有引入真正的 SQL 解析器(sqlparse等三方库),分号切分本身足够应付课程作业
# 里的建表/查询语句,但下面这个切分器至少做到了"字符串字面量/注释里的分号
# 不会被误切",这是最容易踩坑的地方。
#
# 局限性(没有处理,按需再补):
#   - 不识别 T-SQL 的 GO 批次分隔符
#   - 不处理美元符号引用($$...$$,PostgreSQL函数体常用)
#   - 单引号内的转义按标准SQL的''(两个单引号)处理,不认反斜杠转义(MySQL方言)
def split_sql_statements(text: str) -> List[str]:
    statements = []
    current = []
    in_single = in_double = in_line_comment = in_block_comment = False
    i, n = 0, len(text)

    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_line_comment:
            current.append(c)
            if c == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            current.append(c)
            if c == "*" and nxt == "/":
                current.append(nxt)
                i += 2
                in_block_comment = False
                continue
            i += 1
            continue
        if in_single:
            current.append(c)
            if c == "'":
                if nxt == "'":  # 标准SQL的''转义
                    current.append(nxt)
                    i += 2
                    continue
                in_single = False
            i += 1
            continue
        if in_double:
            current.append(c)
            if c == '"':
                in_double = False
            i += 1
            continue

        if c == "-" and nxt == "-":
            in_line_comment = True
            current.append(c)
            i += 1
            continue
        if c == "/" and nxt == "*":
            in_block_comment = True
            current.append(c)
            i += 1
            continue
        if c == "'":
            in_single = True
            current.append(c)
            i += 1
            continue
        if c == '"':
            in_double = True
            current.append(c)
            i += 1
            continue
        if c == ";":
            statements.append("".join(current))
            current = []
            i += 1
            continue

        current.append(c)
        i += 1

    tail = "".join(current)
    if tail.strip():
        statements.append(tail)

    return [s.strip() for s in statements if s.strip()]


_SQL_TYPE_RE = re.compile(r"\s*(\w+)(?:\s+(\w+))?", re.IGNORECASE)
_CREATE_SUBTYPES = {"TABLE", "INDEX", "VIEW", "DATABASE", "SCHEMA", "TRIGGER", "PROCEDURE", "FUNCTION"}


_SQL_LEADING_COMMENT_RE = re.compile(r"^\s*(--[^\n]*\n|/\*.*?\*/\s*)+", re.DOTALL)


def detect_sql_statement_type(stmt: str) -> str:
    # 语句前面常常带一整段注释(比如解释这条语句用途的说明),跳过注释再
    # 找关键字,不然第一个"词"会是注释文字而不是SQL关键字,判成UNKNOWN。
    stripped = _SQL_LEADING_COMMENT_RE.sub("", stmt)
    m = _SQL_TYPE_RE.match(stripped)
    if not m:
        return "UNKNOWN"
    kw = m.group(1).upper()
    second = (m.group(2) or "").upper()
    if kw == "CREATE" and second in _CREATE_SUBTYPES:
        return f"CREATE {second}"
    return kw


def parse_sql(path: Path) -> List[Chunk]:
    course = _course_of(path)
    rel = _rel(path)
    source = path.read_text(encoding="utf-8", errors="replace")

    chunks = []
    for idx, stmt in enumerate(split_sql_statements(source)):
        # 文件名之前只存在metadata里,不参与检索——排查"Caregivers表SQL"这个
        # 检索失败案例时发现,查询词"SQL"字面上根本不出现在SQL语句本身里
        # (DDL语句不会写"sql"这个单词),但会大量出现在无关Java文件里(变量名
        # sql),导致真正的目标语句排到第31-39名(BM25 top-10截断之外)。
        # 把文件名拼进可索引文本,让"create.sql"里的"sql"token也算进这个chunk,
        # 缓解这个词汇鸿沟(不保证完全解决,因为核心问题是词汇不匹配,不是
        # 缺失索引线索)。
        indexed_text = f"{path.name}\n{stmt}"
        chunks.append(
            Chunk(
                text=indexed_text,
                metadata={
                    "file": rel,
                    "course": course,
                    "chunk_type": "sql_statement",
                    "statement_index": idx,
                    "statement_type": detect_sql_statement_type(stmt),
                },
            )
        )
    return chunks


# ===========================================================================
# 4) R (.r) / Java (.java) — 正则找函数签名 + 括号计数器找函数体边界
# ===========================================================================
# 局限性说明(相比 ast 精度明显更低,重要):
#   - 正则只匹配"看起来像"函数/方法签名的独立一行,签名如果跨越多行书写
#     (比如参数很多每个参数单独一行)会匹配失败,导致漏掉该函数。
#   - R 函数体如果不用大括号(单表达式函数,如 `f <- function(x) x + 1`)
#     只能退化成"取到当前行末尾",多行的无大括号函数体会被截断。
#   - Java 正则用了非贪婪匹配猜测"返回类型 + 方法名"的边界,遇到复杂泛型
#     嵌套(如 Map<String, List<Map<Integer,String>>>)或注解跨行书写时
#     可能误判方法名或匹配失败。
#   - 都不会区分字符串/字符字面量里出现的花括号 vs 真正的代码花括号——
#     不对,下面 _scan_matching_brace 其实做了这个区分(跳过字符串和注释里
#     的花括号),但仍然是基于字符扫描的启发式,不是真正的语法分析,遇到
#     极端情况(比如未转义的畸形字符串)可能计数错乱。
#
# 出于以上原因,这一层解析结果的可靠性明显低于 py/ipynb,更适合先跑出来
# 人工抽查,而不是直接信任拿去做下游索引。


def _scan_matching_brace(text: str, open_idx: int, line_comment: str, block_comment: Optional[tuple]) -> int:
    """从 text[open_idx] == '{' 开始,找到与之匹配的 '}' 的下标(闭区间,包含)。
    扫描时跳过字符串/字符字面量和注释里的花括号,避免被内容误导。
    找不到匹配时返回 len(text) - 1(退化为到文件末尾)。
    """
    depth = 0
    i = open_idx
    n = len(text)
    in_single = in_double = in_line_comment = in_block_comment = False

    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_line_comment:
            if c == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if block_comment and c == block_comment[1][0] and text[i : i + len(block_comment[1])] == block_comment[1]:
                i += len(block_comment[1])
                in_block_comment = False
                continue
            i += 1
            continue
        if in_single:
            if c == "\\":
                i += 2
                continue
            if c == "'":
                in_single = False
            i += 1
            continue
        if in_double:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_double = False
            i += 1
            continue

        if line_comment and text[i : i + len(line_comment)] == line_comment:
            in_line_comment = True
            i += len(line_comment)
            continue
        if block_comment and text[i : i + len(block_comment[0])] == block_comment[0]:
            in_block_comment = True
            i += len(block_comment[0])
            continue
        if c == "'":
            in_single = True
            i += 1
            continue
        if c == '"':
            in_double = True
            i += 1
            continue
        if c == "{":
            depth += 1
            i += 1
            continue
        if c == "}":
            depth -= 1
            if depth == 0:
                return i
            i += 1
            continue
        i += 1

    return n - 1


def _scan_matching_paren(text: str, open_idx: int) -> int:
    """跟 _scan_matching_brace 类似,但找 '(' 对应的 ')' ,不做字符串/注释
    感知(函数参数列表里出现字符串字面量含括号的情况很少见,从简处理)。"""
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return n - 1


R_FUNC_HEADER_RE = re.compile(r"^[ \t]*([a-zA-Z_.][\w.]*)\s*(?:<-|=)\s*function\s*\(", re.MULTILINE)


def parse_r(path: Path) -> List[Chunk]:
    course = _course_of(path)
    rel = _rel(path)
    source = path.read_text(encoding="utf-8", errors="replace")

    chunks = []
    covered_spans = []  # [(start, end), ...] 用于最后提取"未覆盖"的顶层代码

    for m in R_FUNC_HEADER_RE.finditer(source):
        name = m.group(1)
        open_paren_idx = m.end() - 1  # 正则以 "(" 结尾
        close_paren_idx = _scan_matching_paren(source, open_paren_idx)

        # 跳过空白,寻找函数体的 '{'
        j = close_paren_idx + 1
        while j < len(source) and source[j] in " \t\r\n":
            j += 1

        if j < len(source) and source[j] == "{":
            body_end = _scan_matching_brace(source, j, line_comment="#", block_comment=None)
            end = body_end + 1
        else:
            # 没有大括号的单表达式函数体: 退化为取到当前行末尾(见局限性说明)
            newline_idx = source.find("\n", close_paren_idx)
            end = newline_idx if newline_idx != -1 else len(source)

        start = m.start()
        segment = source[start:end]
        covered_spans.append((start, end))
        chunks.append(
            Chunk(
                text=segment,
                metadata={
                    "file": rel,
                    "course": course,
                    "chunk_type": "r_function",
                    "name": name,
                },
            )
        )

    toplevel = _extract_uncovered(source, covered_spans)
    if toplevel.strip():
        chunks.append(
            Chunk(
                text=toplevel,
                metadata={"file": rel, "course": course, "chunk_type": "r_toplevel", "name": None},
            )
        )

    return chunks


JAVA_METHOD_HEADER_RE = re.compile(
    r"^[ \t]*(?:@\w+(?:\([^)]*\))?\s*)*"  # 可选注解,如 @Override
    r"(?:public|private|protected)\s+"
    r"(?:static\s+)?(?:final\s+)?(?:synchronized\s+)?(?:abstract\s+)?"
    r"(?:<[\w\s,]+>\s*)?"  # 可选泛型方法类型参数,如 <T>
    r"[\w<>\[\],.]+\s+"  # 返回类型
    r"(\w+)\s*\(",
    re.MULTILINE,
)

JAVA_CLASS_NAME_RE = re.compile(r"\bclass\s+(\w+)")


def _java_constructor_re(source: str) -> Optional[re.Pattern]:
    """构造方法没有返回类型,长得像"修饰符 + 类名 + (",跟普通方法的正则区分不开,
    必须单独处理。先扫一遍文件里出现的所有 class 名(包括嵌套类,比如
    Caregiver.java 里的 CaregiverBuilder/CaregiverGetter 内部类),再用这些
    名字动态拼一个"修饰符 + 这些类名之一 + (" 的正则。
    局限性: 如果构造方法名跟某个成员方法恰好重名(理论上不可能,Java语法
    不允许),或者类名相关的正则转义有遗漏字符,可能漏判/误判,但对课程作业
    这种规模的代码基本够用。
    """
    class_names = set(JAVA_CLASS_NAME_RE.findall(source))
    if not class_names:
        return None
    alternation = "|".join(re.escape(n) for n in sorted(class_names, key=len, reverse=True))
    return re.compile(
        r"^[ \t]*(?:@\w+(?:\([^)]*\))?\s*)*"
        r"(?:public|private|protected)\s+"
        rf"({alternation})\s*\(",
        re.MULTILINE,
    )


def _find_class_spans(source: str) -> List[tuple]:
    """找出文件里每个 class 声明的花括号范围(open_idx, close_idx, class_name),
    包括嵌套类(比如 Caregiver.java 里的 CaregiverBuilder/CaregiverGetter)。
    用于给普通方法chunk标注"这是哪个类的方法" —— 方法体本身通常不会重复
    写类名(比如 getUsername() 方法体里只有 return username,没有"Caregiver"
    这个词),纯文本检索(BM25)和语义检索都抓不到"这个方法属于Caregiver类"
    这层信息,所以要在chunk文本里显式补上。
    """
    spans = []
    for m in JAVA_CLASS_NAME_RE.finditer(source):
        name = m.group(1)
        brace_idx = source.find("{", m.end())
        if brace_idx == -1:
            continue
        close_idx = _scan_matching_brace(source, brace_idx, line_comment="//", block_comment=("/*", "*/"))
        spans.append((brace_idx, close_idx, name))
    return spans


def _enclosing_class(pos: int, class_spans: List[tuple]) -> Optional[str]:
    """返回包含 pos 位置的最内层(区间最小的)class名,找不到就返回None。"""
    candidates = [(end - start, name) for start, end, name in class_spans if start <= pos <= end]
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]


def _extract_java_member(source: str, m: "re.Match", chunk_type: str, name: str):
    open_paren_idx = m.end() - 1
    close_paren_idx = _scan_matching_paren(source, open_paren_idx)

    j = close_paren_idx + 1
    # 跳过可能的 throws 子句和空白,找方法体的 '{' 或抽象方法的 ';'
    rest_after_paren = source[j : j + 200]
    throws_match = re.match(r"\s*throws\s+[\w<>,.\s]+", rest_after_paren)
    if throws_match:
        j += throws_match.end()
    while j < len(source) and source[j] in " \t\r\n":
        j += 1

    if j < len(source) and source[j] == "{":
        body_end = _scan_matching_brace(source, j, line_comment="//", block_comment=("/*", "*/"))
        end = body_end + 1
    elif j < len(source) and source[j] == ";":
        # 接口方法 / 抽象方法声明,没有方法体
        end = j + 1
    else:
        end = j

    start = m.start()
    return Chunk(
        text=source[start:end],
        metadata={"chunk_type": chunk_type, "name": name},
    ), (start, end)


def parse_java(path: Path) -> List[Chunk]:
    course = _course_of(path)
    rel = _rel(path)
    source = path.read_text(encoding="utf-8", errors="replace")

    # 普通方法和构造方法的签名匹配规则不同(构造方法没有返回类型),
    # 分别找出所有候选起始位置,按出现顺序合并处理,避免重复覆盖同一段代码。
    candidates = []  # (start_pos, match, chunk_type)
    for m in JAVA_METHOD_HEADER_RE.finditer(source):
        candidates.append((m.start(), m, "java_method"))

    ctor_re = _java_constructor_re(source)
    if ctor_re:
        for m in ctor_re.finditer(source):
            candidates.append((m.start(), m, "java_constructor"))

    candidates.sort(key=lambda c: c[0])
    class_spans = _find_class_spans(source)

    chunks = []
    covered_spans = []
    last_end = -1
    for start_pos, m, chunk_type in candidates:
        if start_pos < last_end:
            continue  # 跟上一个已提取的成员重叠(理论上不该发生),跳过
        name = m.group(1)
        chunk, span = _extract_java_member(source, m, chunk_type, name)

        # 构造方法名本身就是类名(比如 "Caregiver(...)"),已经自带类名信息,
        # 不需要再加前缀;普通方法(getUsername等)方法体里通常不会出现类名,
        # 需要显式补上,否则检索时找不到"这是哪个类的方法"的线索。
        if chunk_type == "java_method":
            enclosing = _enclosing_class(start_pos, class_spans)
            if enclosing:
                chunk.text = f"这是 {enclosing} 类的方法:\n\n{chunk.text}"

        chunk.metadata["file"] = rel
        chunk.metadata["course"] = course
        chunks.append(chunk)
        covered_spans.append(span)
        last_end = span[1]

    toplevel = _extract_uncovered(source, covered_spans)
    if toplevel.strip():
        chunks.append(
            Chunk(
                text=toplevel,
                metadata={"file": rel, "course": course, "chunk_type": "java_toplevel", "name": None},
            )
        )

    return chunks


def _extract_uncovered(source: str, spans: List[tuple]) -> str:
    """把已经被函数/方法chunk覆盖的区间挖掉,剩下的部分(import、字段声明、
    类头、脚本式顶层代码等)拼成一个"顶层代码"chunk,避免这些内容被静默丢弃。
    这一步是在用户原始需求(只针对Python提出)基础上,为 R/Java 做的同款
    补充,目的是不遗漏没被正则匹配到的内容,而不是范围蔓延。
    """
    spans = sorted(spans)
    pieces = []
    cursor = 0
    for start, end in spans:
        if start > cursor:
            pieces.append(source[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(source):
        pieces.append(source[cursor:])
    return "".join(pieces).strip()


# ===========================================================================
# 测试入口: 每种格式抽 5 个样本,解析结果打印出来供人工检查
# ===========================================================================
PARSERS = {
    ".ipynb": parse_ipynb,
    ".py": parse_py,
    ".sql": parse_sql,
    ".r": parse_r,
    ".java": parse_java,
}

SAMPLES_PER_FORMAT = 5


def _collect_files(ext: str) -> List[Path]:
    from route_files import EXCLUDE_DIRS

    files = []
    for p in DATA_ROOT.rglob(f"*{ext}"):
        if not p.is_file() or p.name.startswith("._"):
            continue
        rel_parts = p.relative_to(DATA_ROOT).parts
        if any(part in EXCLUDE_DIRS for part in rel_parts[:-1]):
            continue
        files.append(p)
    return files


def _pick_diverse_samples(files: List[Path], n: int) -> List[Path]:
    """尽量从不同课程各挑一个,课程数不够n个时再从剩余文件里补齐。"""
    import random

    rng = random.Random(2024)
    by_course = {}
    for f in files:
        by_course.setdefault(_course_of(f), []).append(f)

    courses = list(by_course.keys())
    rng.shuffle(courses)

    picked = []
    for course in courses:
        if len(picked) >= n:
            break
        pool = by_course[course]
        picked.append(rng.choice(pool))

    if len(picked) < n:
        remaining = [f for f in files if f not in picked]
        rng.shuffle(remaining)
        picked.extend(remaining[: n - len(picked)])

    return picked[:n]


def _print_chunk(chunk: Chunk, max_chars: int = 1200):
    text = chunk.text
    truncated = len(text) > max_chars
    shown = text[:max_chars]
    print(f"    metadata: {chunk.metadata}")
    print(f"    ---")
    print("    " + shown.replace("\n", "\n    "))
    if truncated:
        print(f"    ...(共{len(text)}字符,已截断显示)")
    print()


def run_format_test(ext: str):
    parser = PARSERS[ext]
    files = _collect_files(ext)
    print(f"\n{'#' * 90}")
    print(f"# 格式: {ext}  (共{len(files)}个文件,抽样测试{SAMPLES_PER_FORMAT}个)")
    print(f"{'#' * 90}")

    if not files:
        print("  [无文件]")
        return

    samples = _pick_diverse_samples(files, SAMPLES_PER_FORMAT)

    for path in samples:
        rel = _rel(path)
        print(f"\n{'=' * 90}")
        print(f"文件: {rel}")
        print(f"{'=' * 90}")
        try:
            chunks = parser(path)
        except Exception as e:
            print(f"  [解析失败] {type(e).__name__}: {e}")
            continue

        print(f"  共切出 {len(chunks)} 个chunk")
        for i, chunk in enumerate(chunks):
            print(f"\n  --- chunk {i + 1}/{len(chunks)} ---")
            _print_chunk(chunk)


if __name__ == "__main__":
    import sys

    for ext in [".ipynb", ".py", ".sql", ".r", ".java"]:
        run_format_test(ext)
