"""
数据摘要类(csv/xlsx)处理模块

CLAUDE.md "文件分类与处理策略 -> 3. 数据摘要类" 规定: 不做全量索引,只提取
表头字段名、行列规模、文件名/路径上下文,生成一段摘要文本作为可检索chunk,
不索引具体数据内容本身(比如某一行的实际数值)。

两种格式读取方式不同:
  - csv: 用标准库 csv 模块流式读取,只取表头行 + 逐行计数,不会把整个文件
    内容都载入内存里保留下来。
  - xlsx: 用 openpyxl 的 read_only 模式打开,行列规模走 worksheet.max_row/
    max_column(读的是sheet的dimension元数据,不遍历全部单元格),表头走
    第一行,避免像 bp-stats-review-2022-all-data.xlsx 这种几万行的大文件
    被完整加载进内存。

产出 chunk 的 metadata 里 quality/chunk_quality_flag/extraction_confidence
三个字段的取值和文本知识类/代码知识类不是一回事:
  - quality: 数据摘要类没有Docling/LlamaParse那套"解析质量好坏"的概念,
    统一给 "good"(提取表头信息本身不存在"解析失败但还能用"的中间状态)。
  - chunk_quality_flag: 默认 None,但有一个例外——用 csv.Sniffer().has_header()
    检测csv首行是不是真的表头,发现有几个文件(flight-dataset目录下的
    carriers.csv/months.csv/weekdays.csv/flights-small.csv)首行其实是
    数据不是字段名,这种情况标 ["possible_no_header"],摘要文本里也如实
    说明"首行原始内容"而不是"字段包括",不把数据值硬套成字段名。
  - extraction_confidence: 统一 "high"——读表头字段名和行列数是确定性操作
    (csv.reader读第一行/openpyxl读sheet dimension),不存在像r/java代码解析
    那种"正则近似、可能出错"的模糊地带。

用法:
    python ingest/summarize_data.py --test    # 抽样10-15个文件,打印摘要供人工检查
    python ingest/summarize_data.py           # 全量跑152个文件,输出chunk数据集
"""

import csv
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from route_files import EXCLUDE_DIRS  # noqa: E402

DATA_ROOT = Path(r"E:\project")
OUTPUT_PATH = Path(__file__).parent / "_test_output" / "data_summary_chunks.json"
ERRORS_PATH = Path(__file__).parent / "_test_output" / "data_summary_errors.json"

# csv编码尝试顺序: utf-8-sig能处理不带BOM和带BOM两种情况,gb18030覆盖国内
# 课程数据里可能出现的中文csv,latin-1永不抛UnicodeDecodeError,作为兜底
# (即使解出来是乱码,至少不会让整个文件处理失败)
CSV_ENCODINGS = ["utf-8-sig", "gb18030", "latin-1"]

# 单个chunk里最多展示的字段数,超过的话只展示前N个+提示总数,避免个别
# 超宽表格(几百列)把摘要文本撑得过长,稀释向量embedding的语义
MAX_FIELDS_SHOWN = 60

# 单个workbook最多展示的sheet数(逐sheet列字段),超过的话只处理前N个+
# 提示还有多少个未展开,同样是为了避免摘要文本失控
MAX_SHEETS_SHOWN = 10


def course_of(path: Path) -> str:
    try:
        return path.relative_to(DATA_ROOT).parts[0]
    except ValueError:
        return ""


def rel_of(path: Path) -> str:
    try:
        return str(path.relative_to(DATA_ROOT))
    except ValueError:
        return str(path)


def collect_files() -> List[Path]:
    files = []
    for path in DATA_ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(DATA_ROOT).parts
        if len(rel_parts) < 2:
            # 根目录下没有父目录的文件不算课程数据,同 route_files.route()
            # 和 run_parse_code_full.collect_files() 的处理
            continue
        if any(part in EXCLUDE_DIRS for part in rel_parts[:-1]):
            continue
        if path.suffix.lower() in (".csv", ".xlsx"):
            files.append(path)
    return sorted(files)


def _sniff_has_header(sample_text: str) -> Optional[bool]:
    """用标准库 csv.Sniffer 判断首行是不是真的表头(而不是数据行)。
    实测发现flight-dataset目录下有几个csv(carriers.csv/months.csv/
    weekdays.csv/flights-small.csv)根本没有表头行,第一行就是真实数据——
    如果不做这个检测,摘要文本会把"02Q, Titan Airways"这种数据值当成
    字段名写进去,反而误导检索。Sniffer基于首行和后续行的类型/长度模式
    差异做判断,不是100%准确,判断不了的情况(csv.Error)返回None,不强行
    下结论。
    """
    try:
        return csv.Sniffer().has_header(sample_text)
    except csv.Error:
        return None


def _read_csv_header_and_count(path: Path):
    """返回 (header: List[str], row_count: int, encoding_used: str,
    has_header: Optional[bool])。
    row_count 是"表头之外的行数"(不管首行实际是不是表头,都按原始文件
    结构来算,是否需要+1由调用方根据has_header判断)。用流式迭代计数,
    不把整份文件内容保留在内存里。
    """
    last_err = None
    for enc in CSV_ENCODINGS:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                sample = f.read(4096)
                f.seek(0)
                reader = csv.reader(f)
                header = next(reader)
                row_count = sum(1 for _ in reader)
            has_header = _sniff_has_header(sample)
            return header, row_count, enc, has_header
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except StopIteration:
            # 空文件(只有表头或完全空)
            return [], 0, enc, None
    raise last_err


def summarize_csv(path: Path):
    """返回 (摘要文本, chunk_quality_flag)。"""
    header, row_count, enc, has_header = _read_csv_header_and_count(path)
    fields = [h.strip() for h in header if h and h.strip()]

    course = course_of(path)
    name = path.name
    field_text = _format_fields(fields)
    col_count = len(header)

    if has_header is False:
        # 首行实为数据,不当作字段名展示,避免"字段包括:02Q, Titan Airways"
        # 这种把数据值误标成字段名的情况——如实说明检测结果,原始首行内容
        # 仍然展示出来(对检索仍有一定价值,比如能匹配到"Titan Airways"这种
        # 查询),但不冠以"字段"之名。
        text = (
            f"{name} 是 {course} 课程的数据集(CSV格式),"
            f"包含 {row_count + 1} 行、{col_count} 列。"
            f"检测发现该文件没有表头行(首行是数据而非字段名),"
            f"首行原始内容示例:{field_text}。"
        )
        return text, ["possible_no_header"]

    text = (
        f"{name} 是 {course} 课程的数据集(CSV格式),"
        f"包含 {row_count} 行、{col_count} 列。字段包括:{field_text}。"
    )
    return text, None


def _format_fields(fields: List[str]) -> str:
    if not fields:
        return "(未识别到字段名)"
    if len(fields) <= MAX_FIELDS_SHOWN:
        return ", ".join(fields)
    shown = ", ".join(fields[:MAX_FIELDS_SHOWN])
    return f"{shown} 等共{len(fields)}个字段(仅展示前{MAX_FIELDS_SHOWN}个)"


def summarize_xlsx(path: Path) -> str:
    import openpyxl

    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    try:
        sheet_names = wb.sheetnames
        course = course_of(path)
        name = path.name

        parts = [f"{name} 是 {course} 课程的数据集(Excel格式)"]

        if len(sheet_names) > 1:
            # sheet名称列表和下面的逐sheet字段展开用同一个截断上限,避免像
            # bp-stats-review-2022-all-data.xlsx(83个sheet)这种文件生成的
            # 摘要文本长到稀释向量embedding语义
            shown_names = sheet_names[:MAX_SHEETS_SHOWN]
            names_text = ", ".join(shown_names)
            if len(sheet_names) > MAX_SHEETS_SHOWN:
                names_text += f" 等共{len(sheet_names)}个sheet(仅展示前{MAX_SHEETS_SHOWN}个)"
            parts.append(f",包含 {len(sheet_names)} 个sheet:{names_text}")
        parts.append("。")

        sheets_to_show = sheet_names[:MAX_SHEETS_SHOWN]
        for sheet_name in sheets_to_show:
            ws = wb[sheet_name]
            row_count = ws.max_row or 0
            col_count = ws.max_column or 0
            header_row = None
            if row_count > 0:
                try:
                    header_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True))
                except StopIteration:
                    header_row = None
            fields = [str(v).strip() for v in (header_row or []) if v is not None and str(v).strip()]
            # 数据行数不含表头行(如果表头占了第1行)
            data_row_count = max(0, row_count - 1) if header_row else row_count

            field_text = _format_fields(fields)
            if len(sheet_names) > 1:
                parts.append(
                    f"sheet '{sheet_name}' 有 {data_row_count} 行、{col_count} 列,"
                    f"字段包括:{field_text}。"
                )
            else:
                parts.append(f"包含 {data_row_count} 行、{col_count} 列。字段包括:{field_text}。")

        if len(sheet_names) > MAX_SHEETS_SHOWN:
            parts.append(f"(还有 {len(sheet_names) - MAX_SHEETS_SHOWN} 个sheet未展开)")

        return "".join(parts)
    finally:
        wb.close()


def build_chunk(path: Path, idx: int) -> Optional[dict]:
    ext = path.suffix.lower()
    chunk_quality_flag = None
    if ext == ".csv":
        text, chunk_quality_flag = summarize_csv(path)
    elif ext == ".xlsx":
        text = summarize_xlsx(path)
    else:
        return None

    return {
        "chunk_id": f"data_{idx:05d}",
        "text": text,
        "metadata": {
            "source_type": "data_summary",
            "file": rel_of(path),
            "course": course_of(path),
            "chunk_type": "data_summary",
            "name": path.stem,
            "position": None,
            "quality": "good",
            "chunk_quality_flag": chunk_quality_flag,
            "extraction_confidence": "high",
            "source_note": "",
        },
    }


def run(files: List[Path], label: str):
    chunks = []
    errors = []
    for i, path in enumerate(files, 1):
        try:
            chunk = build_chunk(path, i)
            if chunk is not None:
                chunks.append(chunk)
        except Exception as e:
            errors.append({"file": rel_of(path), "error": f"{type(e).__name__}: {e}"})
            print(f"  [{i}/{len(files)}] [异常] {rel_of(path)}: {type(e).__name__}: {e}")
            continue

        if i % 40 == 0 or i == len(files):
            print(f"  已完成 {i}/{len(files)}, 累计chunk数 {len(chunks)}")

    print(f"\n{label}: {len(files)} 个文件, 成功 {len(chunks)} 个, 失败 {len(errors)} 个")
    if errors:
        print("失败明细:")
        for e in errors:
            print(f"  {e['file']}: {e['error']}")

    return chunks, errors


def main():
    test_mode = "--test" in sys.argv
    all_files = collect_files()
    print(f"扫描到数据摘要类文件: {len(all_files)} 个 (csv={sum(1 for f in all_files if f.suffix.lower()=='.csv')}, "
          f"xlsx={sum(1 for f in all_files if f.suffix.lower()=='.xlsx')})")

    if test_mode:
        csvs = [f for f in all_files if f.suffix.lower() == ".csv"]
        xlsxs = [f for f in all_files if f.suffix.lower() == ".xlsx"]

        # 明确把之前发现的大文件塞进测试集,覆盖"大文件读取要快"这个要求
        big_file = next((f for f in xlsxs if "bp-stats-review" in f.name), None)

        sample = csvs[:7] + xlsxs[:6]
        if big_file and big_file not in sample:
            sample.append(big_file)

        print(f"\n=== 测试模式: 抽样 {len(sample)} 个文件 ===\n")
        chunks, errors = run(sample, "测试批次")

        print("\n=== 生成的摘要文本 ===\n")
        for c in chunks:
            print(f"[{c['chunk_id']}] {c['metadata']['file']}")
            print(f"  {c['text']}")
            print()
        return

    chunks, errors = run(all_files, "全量批次")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    with open(ERRORS_PATH, "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    print(f"\n已保存chunk至: {OUTPUT_PATH}")
    print(f"已保存失败记录至: {ERRORS_PATH}")


if __name__ == "__main__":
    main()
