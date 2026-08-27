"""
文件路由脚本(仅统计,不解析内容)

按 CLAUDE.md 中的"文件分类与处理策略"规则,把 DATA_ROOT 下的每个文件
按后缀名归到五个类别之一: 文本知识类 / 代码知识类 / 数据摘要类 / 图片类 / 排除类。
未在 CLAUDE.md 中定义的后缀归入"未分类",用于人工确认分类规则是否需要补充。

用法:
    python ingest/route_files.py [数据根目录,默认 E:\\project]
"""

import sys
from collections import defaultdict
from pathlib import Path

# 与项目自身相关、不属于课程数据的目录,扫描时跳过
# .venv: 项目虚拟环境(为跑 Docling 等依赖新建的),里面 site-packages 装了
# 大量第三方库源码,不排除的话会把 torch/transformers 这些库代码当成课程
# 代码扫进来 —— parse_code.py 抽样测试 .py 格式时就是因为漏排除这个目录,
# 意外抽到了 transformers/scipy 的源码文件才发现这个问题。
EXCLUDE_DIRS = {".claude", ".git", "ingest", "__MACOSX", ".venv", "index", "query"}

# CLAUDE.md "文件分类与处理策略" 一节的映射
CATEGORY_MAP = {
    # 1. 文本知识类
    ".pdf": "文本知识类",
    ".docx": "文本知识类",
    ".pptx": "文本知识类",
    ".txt": "文本知识类",
    ".md": "文本知识类",
    ".rmd": "文本知识类",
    ".html": "文本知识类",
    # 2. 代码知识类
    ".ipynb": "代码知识类",
    ".py": "代码知识类",
    ".r": "代码知识类",
    ".sql": "代码知识类",
    ".java": "代码知识类",
    # 3. 数据摘要类
    ".csv": "数据摘要类",
    ".xlsx": "数据摘要类",
    # 4. 图片类
    ".png": "图片类",
    ".jpg": "图片类",
    ".jpeg": "图片类",
    # 5. 排除类
    ".class": "排除类",
    ".jar": "排除类",
    ".dll": "排除类",
    ".exe": "排除类",
    ".xml": "排除类",
    ".iml": "排除类",
    ".zip": "排除类",
    ".pkl": "排除类",
    ".m4a": "排除类",
    ".mp4": "排除类",
    ".json": "排除类",  # 目前样本均为 IDE/编辑器配置(如 .vscode/settings.json)
    ".yml": "排除类",  # 样本为作业提交系统自动生成的元数据(提交人/时间戳/分数),无知识内容
    ".lp": "代码知识类",  # 线性规划(Linear Programming)模型文件,优化课程的代码资产
}

# pathlib 对 ".gitignore" 这类"纯点开头"或完全无后缀的文件名, suffix 会算成
# 空字符串,所以按文件名(而不是后缀)单独匹配
FILENAME_OVERRIDES = {
    ".gitignore": "排除类",
    ".Rhistory": "排除类",  # RStudio 自动生成的命令历史,非内容
    ".RData": "排除类",  # R 会话工作区二进制快照,非可读内容
    ".name": "排除类",  # IntelliJ .idea/.name 项目名文件
    "LICENSE": "排除类",  # 开源模板仓库自带的许可证样板文本,非课程内容
    "text8": "排除类",  # 标准 Wikipedia 语料基准数据集(约100MB),非个人笔记
}

# 文件名前缀匹配(用于同一类会有编号后缀的情况,如多个 checkpoint 文件)
FILENAME_PREFIX_OVERRIDES = {
    "model-checkpoint": "排除类",  # 深度学习模型权重文件,二进制
}

CATEGORY_ORDER = ["文本知识类", "代码知识类", "数据摘要类", "图片类", "排除类", "未分类"]

SAMPLES_PER_EXT = 3


def is_junk(name: str) -> bool:
    """理论上 ingestion 前应已清理干净,这里做一次防御性跳过。"""
    return name.startswith("._") or name == ".DS_Store"


def route(data_root: Path):
    # 结构: category -> ext -> {"count": int, "samples": [relpath, ...]}
    stats = defaultdict(lambda: defaultdict(lambda: {"count": 0, "samples": []}))
    skipped_junk = 0
    total = 0

    for path in data_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in path.relative_to(data_root).parts[:-1]):
            continue
        if is_junk(path.name):
            skipped_junk += 1
            continue

        total += 1
        prefix_hit = next(
            (p for p in FILENAME_PREFIX_OVERRIDES if path.name.startswith(p)), None
        )
        if path.name in FILENAME_OVERRIDES:
            ext = path.name
            category = FILENAME_OVERRIDES[path.name]
        elif prefix_hit:
            ext = f"{prefix_hit}*"
            category = FILENAME_PREFIX_OVERRIDES[prefix_hit]
        else:
            ext = path.suffix.lower() or "(无扩展名)"
            category = CATEGORY_MAP.get(ext, "未分类")

        bucket = stats[category][ext]
        bucket["count"] += 1
        if len(bucket["samples"]) < SAMPLES_PER_EXT:
            bucket["samples"].append(str(path.relative_to(data_root)))

    return stats, total, skipped_junk


def print_report(stats, total, skipped_junk):
    print(f"扫描到有效文件: {total} 个(跳过残留垃圾文件 {skipped_junk} 个)\n")

    grand_total = 0
    for category in CATEGORY_ORDER:
        ext_stats = stats.get(category)
        if not ext_stats:
            continue
        cat_count = sum(v["count"] for v in ext_stats.values())
        grand_total += cat_count
        print(f"=== {category} ({cat_count} 个文件) ===")
        for ext, info in sorted(ext_stats.items(), key=lambda kv: -kv[1]["count"]):
            samples = " | ".join(info["samples"])
            print(f"  {ext:<10} {info['count']:>5}  例: {samples}")
        print()

    if grand_total != total:
        print(f"[警告] 分类合计 {grand_total} 与扫描总数 {total} 不一致,请检查逻辑\n")

    if "未分类" in stats:
        print("提示: 以上'未分类'后缀不在 CLAUDE.md 的分类规则里,"
              "需要人工确认应归入哪一类或补充规则。")


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"E:\project")
    if not root.is_dir():
        print(f"目录不存在: {root}")
        sys.exit(1)

    stats, total, skipped_junk = route(root)
    print_report(stats, total, skipped_junk)
