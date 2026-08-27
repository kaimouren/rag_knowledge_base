"""
文档质量检测 + 路由分流模块

背景: 在 ingest/parse_text.py 里用 Docling 测试了 5 个 PDF 样本后发现,
Docling 对数字原生文档(考卷、论文)效果很好,但对扫描/手写文档、公式密集
文档表现很差(具体样本见本文件底部自测部分)。这个模块的作用是: 拿到 Docling
解析出的 markdown 文本,自动判断这次解析是否"能用",不能用的就标记为
"建议转 LlamaParse 重新解析",而不需要人工逐份检查。

设计上刻意用了多个独立信号、任一命中就标记,而不是单一综合分数,原因是:
实测发现"扫描件乱码"和"输出密度过低"这两类失败模式几乎不重叠
(见文件底部 __main__ 的实测数据),单一信号无法同时覆盖两种失败模式。

四个信号对应 CLAUDE.md 里 Docling / LlamaParse 路由判断逻辑的具体实现。

---
第二轮修订(在 30-50 个随机抽样的 PDF 上批量验证后发现的问题):
最初只在 5 个精选样本上校准,"输出密度过低"这条阈值在扩大到36个随机样本后
暴露出系统性误判 —— 大量 PPT 导出的课件(大字体+大量留白+图表)会被
误判为"解析失败",实际上文字提取得很好,只是内容本身就少。人工抽查
12个"仅密度触发"的样本,4个当中只有1个是真失败,3个都是正常课件。
同时发现一个漏判: 一份中文手写笔记乱码文档完全没被抓到,因为信号a)只
统计英文字母 token,对中文乱码没有识别能力。

针对这两个问题的修订:
1) 新增信号d)"短行碎片化比例",不区分语言,专门补上中文/其他非拉丁
   文字乱码的检测盲区。
2) "输出密度过低"改为两级判断: 密度极低(<100字符/页)时,大概率是
   整页图片/空白,直接判定失败;密度中等偏低(100-500字符/页)时,
   必须同时有信号a)或信号d)的乱码证据佐证才判定失败,否则只是
   正常课件内容少,不误判。
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 信号 a) 乱码/OCR失败检测: "看起来像正常单词"的比例
# ---------------------------------------------------------------------------
# 判定一个英文字母 token 是否"像正常单词",用的是不依赖词典的启发式规则
# (项目目标机器可能没有联网权限装大词典,而且课程笔记里大量专业术语/人名
# 词典也覆盖不到,所以用字符层面的规律而不是"是否是英文单词"来判断):
#
#   1) token 长度 > MAX_PLAUSIBLE_LEN(25): 判定为不像单词
#      依据: 英文单词很少超过25个字符。Docling 对手写/扫描件识别失败时,
#      经常把原本有空格的多个单词粘连成一长串(例如实测样本1里的
#      "Sincewewanttomakesurethepansyanoperat",38个字符),这是"丢空格"
#      这种失败模式的典型特征。
#   2) token 不含任何元音字母(aeiouy): 判定为不像单词
#      依据: 英文单词(含罕见的无元音词如 "by")几乎都含元音,y 也算作
#      元音以覆盖 "my/why/try" 这类词。实测样本1里的乱码 token 如
#      "RzlDRMRMRMWzlDRzMRzMW" 完全不含元音,这是 OCR 把图案/噪点误判成
#      字母时的典型特征。
#   3) 除首字母外还有 >=2 个大写字母,且整个token不是全大写(排除缩写词
#      如 SQL/ACID/GPU): 判定为不像单词
#      依据: 正常英文大小写只会出现"首字母大写"或"全大写缩写"两种模式,
#      内部随机夹杂大写(如样本1的 "RzlDRMRMRM")是 OCR 把不同笔画误判成
#      不同大小写字母的典型特征。
#
# 长度 < MIN_PLAUSIBLE_LEN(3) 的 token(如 "a" "is" "of")不做判断,直接
# 记为"正常",因为这类短词太常见、信息量低,强行判断反而容易引入噪声。

MIN_PLAUSIBLE_LEN = 3
MAX_PLAUSIBLE_LEN = 25
VOWELS = set("aeiouAEIOUy")

# 阈值依据(实测数据,见文件底部 __main__ 的校准结果):
#   样本1(已知OCR失败,扫描手写作业)  garbled_ratio = 0.184
#   样本2(已知基本正常,Colab导出)    garbled_ratio = 0.005
#   样本3(已知正常,数字原生考卷)     garbled_ratio = 0.006
#   样本4(已知正常,arXiv论文)        garbled_ratio = 0.003
#   样本5(已知OCR失败,扫描手写笔记)  garbled_ratio = 0.018  <- 这个信号
#       没能命中样本5(它的乱码更多是"看起来像短单词的错误识别",而不是
#       "丢空格连成长串"或"无元音噪点"),所以样本5要靠下面信号b/c补上,
#       这正是"三信号任一命中"这种设计的原因,而不是只依赖这一个信号。
# 阈值定在 0.08(8%): 比正常样本的最大值(0.6%)高出一个数量级,
# 同时比样本1的实测值(18.4%)留了一倍以上安全边际。
GARBLED_RATIO_THRESHOLD = 0.08

# token 总数少于这个值时,比例统计噪声太大,不做该项判断(标记为"样本量不足")
MIN_TOKENS_FOR_SIGNAL = 20


def garbled_word_ratio(text: str) -> Optional[float]:
    """返回"不像正常单词"的 token 占比;token 总数太少时返回 None。"""
    tokens = re.findall(r"[A-Za-z]+", text)
    if len(tokens) < MIN_TOKENS_FOR_SIGNAL:
        return None

    bad = 0
    for tok in tokens:
        if len(tok) < MIN_PLAUSIBLE_LEN:
            continue
        if len(tok) > MAX_PLAUSIBLE_LEN:
            bad += 1
            continue
        if tok.isupper():
            continue  # 缩写词例外(SQL, ACID, GPU...)
        if not any(c in VOWELS for c in tok):
            bad += 1
            continue
        internal_upper = sum(1 for c in tok[1:] if c.isupper())
        if internal_upper >= 2:
            bad += 1

    return bad / len(tokens)


# ---------------------------------------------------------------------------
# 信号 b) 公式密集检测: "formula-not-decoded" 占位符密度
# ---------------------------------------------------------------------------
# Docling 检测到公式区域但默认不做公式识别,会插入 "<!-- formula-not-decoded -->"
# 占位符。这个信号跟"扫描质量"无关 —— 即使原文是数字原生、排版工整的
# PDF,只要公式多,Docling 输出里公式内容也是完全丢失的,所以单独作为
# 一类问题,而不是并进信号a。
#
# 阈值依据(实测数据): 每1000字符里的占位符个数
#   样本3(2个公式,考卷) 0.44   样本4(9个公式,论文) 0.08   都明显偏低
#   样本5(11个公式,概率论手写笔记) 2.49   明显偏高
# 用户提出的"每1000字符超过2次"这个经验阈值,刚好卡在 0.44/0.08 和 2.49
# 之间,分离效果很好,予以采用。
FORMULA_DENSITY_THRESHOLD = 2.0  # 每 1000 字符

FORMULA_PLACEHOLDER = "formula-not-decoded"


def formula_density(text: str) -> float:
    """每 1000 字符里 formula-not-decoded 占位符出现的次数。"""
    if not text:
        return 0.0
    count = text.count(FORMULA_PLACEHOLDER)
    return count / (len(text) / 1000)


# ---------------------------------------------------------------------------
# 信号 d) 短行碎片化检测(语言无关,用于补上信号a)的中文/非拉丁文盲区)
# ---------------------------------------------------------------------------
# 信号a)只统计 [A-Za-z]+ token,对中文/其他非拉丁文乱码完全"失明"——
# 实测中一份中文手写笔记(大量数学符号+中文夹杂)乱码得很彻底,但因为
# 乱码内容大部分是中文/符号,信号a)的乱码词占比只有 1.8%(远低于8%阈值),
# 完全没被抓到。
#
# 这里用一个不依赖语言的特征: OCR 识别失败时,不管中英文,都容易把内容
# 切成很多孤立的短碎片(一两个字/符号独占一行),而不是正常的连贯语句。
# 具体做法: 统计"非结构性"行(排除 markdown 的 <!--、|、#、``` 这类语法
# 行)里,长度 <= MAX_FRAGMENT_LEN(3个字符)的短行占比。
#
# 阈值依据(实测数据,按行统计,过滤掉 markdown 结构行后):
#   样本2(正常,Colab导出)     0.067
#   样本3(正常,考卷)          0.000
#   样本4(正常,论文)          0.006
#   样本5(已知乱码,概率论笔记) 0.154
#   NLP整理.pdf(中文乱码,批量测试中发现的漏判) 0.18
# 正常样本上限 0.067,乱码样本下限 0.154,中间空隙很大,阈值定在 0.10,
# 两边都留了约50%以上的安全边际。
MAX_FRAGMENT_LEN = 3
FRAGMENT_RATIO_THRESHOLD = 0.10
MIN_LINES_FOR_SIGNAL = 20  # 有效行数太少时统计噪声大,不做该项判断

_STRUCTURAL_LINE_PREFIXES = ("<!--", "|", "#", "```")


def line_fragmentation_ratio(text: str) -> Optional[float]:
    """返回"短碎片行"占非结构性行的比例;有效行数太少时返回 None。"""
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith(_STRUCTURAL_LINE_PREFIXES)
    ]
    if len(lines) < MIN_LINES_FOR_SIGNAL:
        return None

    short = sum(1 for line in lines if len(line) <= MAX_FRAGMENT_LEN)
    return short / len(lines)


# ---------------------------------------------------------------------------
# 信号 c) 整体解析失败检测: 输出密度(每页字符数)异常低
# ---------------------------------------------------------------------------
# 优先用"每页字符数"而不是"每字节文件大小字符数",因为实测发现按文件
# 大小算的密度在"已知正常"的样本之间波动就有将近3倍(样本2/3/4分别是
# 38/32/93 字符每KB),原因是 PDF 里嵌入图片会显著拉大文件体积但跟文字
# 内容量无关,不是稳定的基准。按页数算就稳定得多:
#   样本1(1页,OCR失败但输出量不少)  1489 字符/页  <- 这个信号故意
#       "漏判"样本1,因为它的失败模式是"内容错但字数不少",不是"内容缺失",
#       该由信号a负责识别,不是这里的职责。
#   样本2(27页,正常) 1407 字符/页
#   样本3(3页,正常)  1511 字符/页
#   样本4(36页,正常) 2974 字符/页
#   样本5(15页,OCR失败) 295 字符/页  <- 明显低于其余样本一个数量级
#
# 阈值定在 500 字符/页: 比四个"正常"样本的最小值(1407)留了近3倍安全
# 边际,同时明显高于样本5的实测值(295),分离度足够。
#
# 修订(批量抽样后): 500 这条线本身没错,但"低于500就判定失败"这个
# 用法在36个随机样本上验证时暴露了问题 —— PPT 导出的课件(大字体、
# 大量留白、图表多)正常情况下密度也会明显低于500(实测多份已知内容
# 完整、可读性良好的课件密度在127~476之间),会被误判。真正"解析失败
# 到几乎没输出"的样本(如一页纯图片的文件)密度通常在100以下,跟"课件
# 本来就少字"的密度区间有明显差别,所以拆成两档:
#   < SEVERE_CHARS_PER_PAGE(100): 几乎没提取到文本,直接判定失败,
#       不需要其他信号佐证(比如整页是图片,没有乱码可言,信号a/d
#       天然也测不出东西)。
#   [SEVERE_CHARS_PER_PAGE, MIN_CHARS_PER_PAGE) 即 [100, 500):
#       密度偏低但不算离谱,必须同时命中信号a)或信号d)的乱码证据,
#       才判定为解析失败;否则大概率只是正常课件,不标记。
SEVERE_CHARS_PER_PAGE = 100
MIN_CHARS_PER_PAGE = 500

# 拿不到页数时(比如 docx/pptx,或 PDF 页数提取失败)的兜底信号,只用
# 文件大小估算,置信度低 —— 上面已经说明按字节的密度波动很大,这里只
# 挡最离谱的情况(比如几十MB的文件只解析出几百字符),阈值刻意设得很松,
# 没有用这5个样本单独校准,后续遇到更多 docx/pptx 样本后应重新标定。
MIN_CHARS_PER_KB_FALLBACK = 3.0


@dataclass
class DensityResult:
    value: float
    unit: str  # "chars/page" 或 "chars/KB(低置信度兜底)"
    severity: str  # "normal" | "moderate_low"(需要佐证) | "severe_low"(直接判定失败)


def output_density(
    text_len: int, page_count: Optional[int], file_size_bytes: int
) -> DensityResult:
    if page_count and page_count > 0:
        value = text_len / page_count
        unit = "chars/page"
        severe_threshold, moderate_threshold = SEVERE_CHARS_PER_PAGE, MIN_CHARS_PER_PAGE
    else:
        # 拿不到页数的兜底(docx/pptx等): 按字节估算,置信度低,阈值同比例放宽
        kb = max(file_size_bytes / 1024, 1e-9)
        value = text_len / kb
        unit = "chars/KB(低置信度兜底)"
        severe_threshold = MIN_CHARS_PER_KB_FALLBACK / 3
        moderate_threshold = MIN_CHARS_PER_KB_FALLBACK

    if value < severe_threshold:
        severity = "severe_low"
    elif value < moderate_threshold:
        severity = "moderate_low"
    else:
        severity = "normal"

    return DensityResult(value, unit, severity)


# ---------------------------------------------------------------------------
# 汇总: 三个信号任一命中 -> 标记为需要转 LlamaParse
# ---------------------------------------------------------------------------
@dataclass
class FileInfo:
    path: Path
    file_size_bytes: int
    page_count: Optional[int] = None


@dataclass
class RouteDecision:
    file_path: Path
    needs_llamaparse: bool
    reasons: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def check_quality(info: FileInfo, markdown_text: str) -> RouteDecision:
    reasons = []
    metrics = {}

    ratio = garbled_word_ratio(markdown_text)
    metrics["garbled_ratio"] = ratio
    garbled_hit = ratio is not None and ratio > GARBLED_RATIO_THRESHOLD
    if garbled_hit:
        reasons.append(
            f"疑似扫描件/OCR失败: 乱码词占比 {ratio:.1%} > 阈值 {GARBLED_RATIO_THRESHOLD:.0%}"
        )

    frag = line_fragmentation_ratio(markdown_text)
    metrics["fragmentation_ratio"] = frag
    frag_hit = frag is not None and frag > FRAGMENT_RATIO_THRESHOLD
    if frag_hit:
        reasons.append(
            f"短行碎片化严重(可能是非拉丁文乱码): 短行占比 {frag:.1%} > 阈值 {FRAGMENT_RATIO_THRESHOLD:.0%}"
        )

    density_f = formula_density(markdown_text)
    metrics["formula_density_per_1k_chars"] = density_f
    if density_f > FORMULA_DENSITY_THRESHOLD:
        reasons.append(
            f"公式密集: {density_f:.2f} 个/千字符 > 阈值 {FORMULA_DENSITY_THRESHOLD:.1f}"
        )

    dens = output_density(len(markdown_text), info.page_count, info.file_size_bytes)
    metrics["output_density"] = f"{dens.value:.1f} {dens.unit}"
    if dens.severity == "severe_low":
        reasons.append(f"输出密度极低: {dens.value:.1f} {dens.unit},大概率整页图片或彻底解析失败")
    elif dens.severity == "moderate_low":
        if garbled_hit or frag_hit:
            reasons.append(
                f"输出密度偏低({dens.value:.1f} {dens.unit})且有乱码佐证,判定为解析失败"
            )
        # else: 密度偏低但没有乱码佐证,大概率是正常课件内容少,不标记

    return RouteDecision(
        file_path=info.path,
        needs_llamaparse=bool(reasons),
        reasons=reasons,
        metrics=metrics,
    )


def print_report(decisions: list):
    print(f"{'文件':<55} {'结论':<12} 原因")
    print("-" * 110)
    for d in decisions:
        verdict = "转LlamaParse" if d.needs_llamaparse else "Docling可用"
        name = d.file_path.name
        if len(name) > 53:
            name = name[:50] + "..."
        print(f"{name:<55} {verdict:<12} {'; '.join(d.reasons) if d.reasons else '-'}")
        print(f"{'':<55} {'':<12} 指标: {d.metrics}")


# ---------------------------------------------------------------------------
# 自测: 在 parse_text.py 测试过的 5 个样本上验证检测逻辑
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from parse_text import SAMPLE_PDFS, OUTPUT_DIR

    try:
        from pypdf import PdfReader
    except ImportError:
        PdfReader = None

    def get_page_count(pdf_path: Path) -> Optional[int]:
        if PdfReader is None:
            return None
        try:
            return len(PdfReader(str(pdf_path)).pages)
        except Exception:
            return None

    # 已知结论(来自人工审阅 parse_text.py 的输出,见对话历史):
    # 样本1、5 是 OCR/扫描失败案例,应被标记为需要转 LlamaParse;
    # 样本3、4 是数字原生文档,解析质量好,不应被误判。
    EXPECTED = {1: True, 2: None, 3: False, 4: False, 5: True}  # None = 不做强制断言

    decisions = []
    all_pass = True
    for i, pdf_path in enumerate(SAMPLE_PDFS, 1):
        md_path = OUTPUT_DIR / f"{i:02d}_{pdf_path.stem}.md"
        if not md_path.exists():
            print(f"[跳过] 找不到 {md_path},请先运行 parse_text.py")
            continue

        markdown_text = md_path.read_text(encoding="utf-8")
        info = FileInfo(
            path=pdf_path,
            file_size_bytes=pdf_path.stat().st_size,
            page_count=get_page_count(pdf_path),
        )
        decision = check_quality(info, markdown_text)
        decisions.append(decision)

        expected = EXPECTED.get(i)
        if expected is not None:
            ok = decision.needs_llamaparse == expected
            all_pass = all_pass and ok
            tag = "PASS" if ok else "FAIL"
            print(
                f"[自测 {tag}] 样本{i}: 预期 needs_llamaparse={expected}, "
                f"实际={decision.needs_llamaparse}"
            )

    print()
    print_report(decisions)

    # -------------------------------------------------------------------
    # 回归测试: 批量抽样(batch_quality_test.py, 36个随机样本)人工抽查后
    # 确认的结论,防止以后调阈值时又把这几个已知案例改错。
    # -------------------------------------------------------------------
    BATCH_DIR = OUTPUT_DIR / "batch"
    BATCH_EXPECTED = {
        # (原始PDF相对路径, 批量测试里保存的md文件名) -> 预期 needs_llamaparse
        (
            r"CSE 414\俊杰的\vaccine-scheduler-java-main\out\production\vaccine-scheduler-java-main\resources\design.pdf",
            "024_design.md",
        ): True,  # 整页只有 <!-- image -->,真实失败
        (
            r"IEOR E4540 Data Mining\SARA-RT-slides.pdf",
            "023_SARA-RT-slides.md",
        ): False,  # 图多字少的正常研究报告slides,曾被密度信号误判
        (
            r"CSE 414\俊杰的\LEC22_parallel_processing.pdf",
            "011_LEC22_parallel_processing.md",
        ): False,  # 正常课件PPT,曾被密度信号误判
        (
            r"IEOR E4573 Deep Learning for NLP\Lecture 1.pdf",
            "006_Lecture 1.md",
        ): False,  # 正常课件PPT,曾被密度信号误判
        (
            r"IEOR E4573 Deep Learning for NLP\NLP整理.pdf",
            "008_NLP整理.md",
        ): True,  # 中文手写笔记乱码,曾被信号a)的拉丁文盲区漏判
    }

    if BATCH_DIR.exists():
        print("\n--- 批量抽样回归测试(修复密度误判/中文漏判后) ---")
        for (rel_pdf, md_name), expected in BATCH_EXPECTED.items():
            pdf_path = Path(r"E:\project") / rel_pdf
            md_path = BATCH_DIR / md_name
            if not md_path.exists():
                print(f"[跳过] 找不到 {md_path}")
                continue
            markdown_text = md_path.read_text(encoding="utf-8")
            info = FileInfo(
                path=pdf_path,
                file_size_bytes=pdf_path.stat().st_size if pdf_path.exists() else 0,
                page_count=get_page_count(pdf_path) if pdf_path.exists() else None,
            )
            decision = check_quality(info, markdown_text)
            ok = decision.needs_llamaparse == expected
            all_pass = all_pass and ok
            tag = "PASS" if ok else "FAIL"
            print(
                f"[回归 {tag}] {md_name}: 预期={expected} 实际={decision.needs_llamaparse} "
                f"原因={decision.reasons}"
            )
    else:
        print(f"\n[提示] 未找到 {BATCH_DIR},跳过批量回归测试(先运行 batch_quality_test.py)")

    print()
    print("全部自测通过" if all_pass else "存在自测失败,请检查阈值/逻辑")
