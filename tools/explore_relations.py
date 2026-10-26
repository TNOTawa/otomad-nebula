"""v0.4.5 高维意外关系发现 CLI（观察性，不写 DB/地图）。

用法（项目根目录）::

    python tools/explore_relations.py
    python tools/explore_relations.py --top 50 --k 10 --mode all
    python tools/explore_relations.py --mode a --out logs/unexpected_relations.json
    python tools/explore_relations.py --mode c --top 30

读取 full_mean_v1（joint/audio/video）、temporal_stats_v1（joint）、
dense_visual_clip_v2（joint）、dense_visual_dino_v2（joint）的 canonical 共同作品，
按高维余弦排名百分位计算：

- 模式 A：joint Top-k 候选的 U = bar_p_joint - (bar_p_audio + bar_p_video) / 2；
- 模式 B：temporal Top-k 候选的 U = bar_p_temporal - bar_p_full_mean；
- 模式 C：PE-AV/CLIP/DINO mutual Top-k 对比（common 仅含包含 PE-AV 的 >=2 空间
  mutual；visual_vs_peav = CLIP+DINO mutual 但 PE-AV 不 mutual；单模型只看恰好
  一个空间 mutual）。

输出 logs/unexpected_relations.json + 长表 CSV；跨模型比较只用排名百分位 p，原始
余弦仅作展示。不写数据库、不建地图、不加载模型、不访问网络。
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nebula import discovery  # noqa: E402
from nebula.config import load_config  # noqa: E402
from nebula.db import Catalog  # noqa: E402
from nebula.errors import NebulaError  # noqa: E402
from nebula.log import configure_logging  # noqa: E402

DEFAULT_CONFIG = "config.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/explore_relations.py",
        description=(
            "v0.4.5 高维意外关系发现：排名百分位下的模式 A/B/C"
            "（joint vs audio/video、temporal vs full_mean、PE-AV/CLIP/DINO mutual 对比）。"
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="配置文件路径（默认取 NEBULA_CONFIG 环境变量，否则当前目录 config.yaml）",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=discovery.DEFAULT_TOP_N,
        help=f"每个模式/类别输出条数（默认 {discovery.DEFAULT_TOP_N}）",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=discovery.DEFAULT_K,
        help=f"高维 Top-k 邻域（默认 {discovery.DEFAULT_K}）",
    )
    parser.add_argument(
        "--mode",
        choices=discovery.MODE_CHOICES,
        default="all",
        help="输出模式（all/a/b/c，默认 all）",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="JSON 输出路径（默认 <logs>/unexpected_relations.json）",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="CSV 输出路径（默认与 JSON 同目录同名 .csv）",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="日志级别（默认取 config.logging.level）",
    )
    return parser


def _configure_console() -> None:
    """Windows 控制台可能是 GBK：避免标题中的不可编码字符让 --mode 全部输出崩溃。"""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            with contextlib.suppress(ValueError, OSError):
                reconfigure(errors="replace")


def _resolve_config_path(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    env_value = os.environ.get("NEBULA_CONFIG")
    if env_value:
        return Path(env_value).expanduser()
    return Path.cwd() / DEFAULT_CONFIG


def main(argv: list[str] | None = None) -> int:
    _configure_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(_resolve_config_path(args.config))
    except NebulaError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2
    configure_logging(args.log_level or config.logging.level)

    catalog = Catalog(config.paths.db_path)
    try:
        catalog.initialize()
        report = discovery.discover(
            catalog,
            config,
            top_n=args.top,
            k=args.k,
            mode=args.mode,
        )
    except NebulaError as exc:
        print(f"意外关系发现失败: {exc}", file=sys.stderr)
        return 1
    finally:
        catalog.close()

    json_path = (
        Path(args.out).expanduser()
        if args.out
        else config.paths.logs / "unexpected_relations.json"
    )
    csv_path = (
        Path(args.csv).expanduser() if args.csv else json_path.with_suffix(".csv")
    )
    try:
        written_json, written_csv = discovery.write_reports(
            report, json_path=json_path, csv_path=csv_path
        )
    except NebulaError as exc:
        print(f"写入意外关系报告失败: {exc}", file=sys.stderr)
        return 1

    discovery.print_summary(
        report, json_path=written_json, csv_path=written_csv
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
