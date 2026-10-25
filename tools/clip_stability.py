"""片段稳定性报告 CLI（A3）。

用法（项目根目录）::

    .\\.venv\\Scripts\\python.exe tools/clip_stability.py
    .\\.venv\\Scripts\\python.exe tools/clip_stability.py --config config.yaml --k 5
    .\\.venv\\Scripts\\python.exe tools/clip_stability.py --version <embedding_version> --out-dir logs

行为：只读 data/embeddings 里已保存的 clip_joint / work_joint 向量，计算留一稳定性，
输出 <out-dir>/clip_stability.json 与 <out-dir>/clip_stability.csv（默认 out-dir 为配置的 logs 目录）。
不加载模型、不重新编码、不访问网络、不写数据库；本报告不参与星云坐标与作品质量评价。

参数：
    --config    配置文件路径（默认取 NEBULA_CONFIG 环境变量，否则当前目录 config.yaml）
    --version   向量版本（默认配置派生的 embedding_version）
    --k         Top-k 近邻数（默认 5）
    --out-dir   报告输出目录（默认配置里的 logs）
    --work-ids  只评估指定 work_id（可选；近邻空间仍使用全部 canonical 作品）
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nebula.config import load_config  # noqa: E402
from nebula.db import Catalog  # noqa: E402
from nebula.errors import NebulaError  # noqa: E402
from nebula.log import configure_logging  # noqa: E402
from nebula.stability import DEFAULT_K, stability_report, write_report  # noqa: E402

JSON_FILENAME = "clip_stability.json"
CSV_FILENAME = "clip_stability.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/clip_stability.py",
        description="A3 片段稳定性报告：留一重聚合 + Top-k 近邻变化（只读已保存向量）。",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="配置文件路径（默认取 NEBULA_CONFIG 环境变量，否则当前目录 config.yaml）",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="向量版本（默认使用配置派生的 embedding_version）",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Top-k 近邻数（默认 {DEFAULT_K}）",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="报告输出目录（默认使用配置里的 logs 目录）",
    )
    parser.add_argument(
        "--work-ids",
        nargs="+",
        type=int,
        default=None,
        help="只评估指定 work_id（默认评估全部 canonical 作品；近邻空间仍用全部 canonical）",
    )
    return parser


def _resolve_config_path(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    env_value = os.environ.get("NEBULA_CONFIG")
    if env_value:
        return Path(env_value).expanduser()
    return Path.cwd() / "config.yaml"


def _print_summary(report: dict, *, json_path: Path, csv_path: Path) -> None:
    print("片段稳定性报告（A3）")
    print(f"  向量版本: {report.get('embedding_version')}")
    print(
        f"  Top-k: {report.get('k')}（有效 {report.get('effective_k')}）"
        f"；不稳定阈值: {report.get('unstable_threshold')}"
    )
    print(
        f"  canonical 作品: {report.get('n_canonical_works')}"
        f"；排除 duplicate: {report.get('n_excluded_duplicates')}"
    )
    print(
        f"  评估作品: {report.get('n_evaluated')} 部"
        f"；跳过: {report.get('n_skipped')} 部"
    )
    print(f"  平均 Top-k 重叠率: {float(report.get('mean_overlap', 0.0)):.4f}")
    print(
        f"  不稳定作品数: {report.get('unstable_count')} / {report.get('n_evaluated')}"
        f"（阈值 {float(report.get('unstable_threshold', 0.0)):.2f}）"
    )
    print(f"  平均余弦漂移: {float(report.get('mean_cosine_drift', 0.0)):.6f}")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    skipped = report.get("skipped", [])
    if skipped:
        print(f"  跳过明细（前 5 条，共 {len(skipped)} 条）:")
        for item in skipped[:5]:
            print(f"    work_id={item.get('work_id')}: {item.get('reason')}")
    if not report.get("works"):
        print("  注意：没有可评估的作品，报告只包含汇总与跳过信息。")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(_resolve_config_path(args.config))
    except NebulaError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2
    configure_logging(config.logging.level)

    catalog = Catalog(config.paths.db_path)
    try:
        catalog.initialize()
        report = stability_report(
            catalog,
            config,
            version=args.version,
            k=args.k,
            work_ids=args.work_ids,
        )
    except NebulaError as exc:
        print(f"稳定性报告失败: {exc}", file=sys.stderr)
        return 1
    finally:
        catalog.close()

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else config.paths.logs
    json_path = out_dir / JSON_FILENAME
    csv_path = out_dir / CSV_FILENAME
    try:
        write_report(report, json_path=json_path, csv_path=csv_path)
    except NebulaError as exc:
        print(f"写入报告失败: {exc}", file=sys.stderr)
        return 1

    _print_summary(report, json_path=json_path, csv_path=csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
