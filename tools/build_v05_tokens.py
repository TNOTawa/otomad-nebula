"""v0.5-alpha 局部音画 token 数据集构建 CLI。

从已有落盘数据生成 data/v05_tokens/<version>/<work_id>.npz + manifest.json：

- PE-AV 窗口向量 / clip_times、事件扫描 event_scan.npz、CLIP/DINOv2 帧级向量；
- 固定 0.5s × 16 bin/段，段数 = PE-AV 窗口数（最多 128，超长截断）；
- 只处理 canonical + completed 作品；不加载 PE-AV / CLIP / DINOv2，不训练、不联网。

用法（项目根目录）::

    python tools/build_v05_tokens.py
    python tools/build_v05_tokens.py --version v05_alpha_v1 --force
    python tools/build_v05_tokens.py --limit 5 --work-ids 12 34 56
    python tools/build_v05_tokens.py --sequence-version <seq> --clip-frame-version <v> --dino-frame-version <v>

真实数据规模预估（110 部、4130 段、CLIP 8fps）：约 150 MB 未压缩 npz（平均
约 1.3 MB/部、最多 128 段约 4.6 MB/部）；manifest 约几十 KB。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nebula import v05_data  # noqa: E402
from nebula.config import load_config  # noqa: E402
from nebula.db import Catalog  # noqa: E402
from nebula.errors import NebulaError  # noqa: E402
from nebula.log import configure_logging  # noqa: E402

DEFAULT_CONFIG = "config.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/build_v05_tokens.py",
        description=(
            "v0.5-alpha 数据管道：从 PE-AV 窗口 / 事件扫描 / CLIP+DINOv2 帧向量构建"
            "局部音画 token npz 数据集（不重新编码）。"
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="配置文件路径（默认取 NEBULA_CONFIG 环境变量，否则当前目录 config.yaml）",
    )
    parser.add_argument(
        "--version",
        default=v05_data.V05_DATASET_VERSION,
        help=f"数据集版本（默认 {v05_data.V05_DATASET_VERSION}）",
    )
    parser.add_argument(
        "--sequence-version",
        default=None,
        help="PE-AV 序列版本（默认 config.embedding_version）",
    )
    parser.add_argument(
        "--clip-frame-version",
        default=None,
        help="CLIP 帧编码版本目录名（默认按 config.visual_dense.encoders 解析）",
    )
    parser.add_argument(
        "--dino-frame-version",
        default=None,
        help="DINOv2 帧编码版本目录名（默认按 config.visual_dense.encoders 解析）",
    )
    parser.add_argument(
        "--force", action="store_true", help="已存在的 <work_id>.npz 也重建"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="最多构建多少部作品（调试/试跑）"
    )
    parser.add_argument(
        "--work-ids",
        nargs="+",
        type=int,
        default=None,
        help="只构建指定 work_id（可多值；默认全部 canonical completed）",
    )
    parser.add_argument(
        "--max-segments",
        type=_parse_max_segments,
        default=v05_data.V05_MAX_SEGMENTS,
        metavar="N|none",
        help=(
            "每部作品最大段数（默认 128，保持 alpha 兼容；传 none/0 表示全量不截断，"
            "v0.5-beta 用）"
        ),
    )
    return parser


def _parse_max_segments(value: str) -> int | None:
    text = str(value).strip().lower()
    if text in ("none", "null", "0", "all", "full"):
        return None
    try:
        number = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--max-segments 必须是正整数或 none，收到 {value!r}"
        ) from exc
    if number < 1:
        raise argparse.ArgumentTypeError(f"--max-segments 必须 >= 1，收到 {number}")
    return number


def _resolve_config_path(value: str | None) -> Path:
    import os

    if value:
        return Path(value).expanduser()
    env_value = os.environ.get("NEBULA_CONFIG")
    if env_value:
        return Path(env_value).expanduser()
    return Path.cwd() / DEFAULT_CONFIG


def _print_summary(summary: dict) -> None:
    print("v0.5-alpha 局部音画 token 数据集")
    print(f"  dataset_version: {summary['dataset_version']}")
    print(f"  目录: {summary['dataset_dir']}")
    print(
        f"  sequence={summary['sequence_version']}"
        f"  clip_frame={summary['clip_frame_version']}"
        f"  dino_frame={summary['dino_frame_version']}"
    )
    print(
        f"  bin={v05_data.V05_BIN_SECONDS:g}s × {v05_data.V05_BINS_PER_SEGMENT}/段"
        f"  max_segments={summary.get('max_segments')}"
    )
    print(
        f"  候选={summary['n_candidates']}  新建={summary['n_built']}"
        f"  已存在={summary['n_existing']}  缺失源={summary['n_missing']}"
        f"  失败={summary['n_failed']}  截断={summary['n_truncated']}"
        f"  耗时={summary['elapsed_s']:.2f}s"
    )
    print(f"  manifest: {summary['manifest_path']}")
    missing = summary.get("missing") or []
    if missing:
        print(f"  缺失源（前 5 条，共 {len(missing)}）:")
        for item in missing[:5]:
            print(f"    work_id={item['work_id']}: {'; '.join(item['reasons'])}")
    failures = summary.get("failures") or []
    if failures:
        print(f"  失败（前 5 条，共 {len(failures)}）:")
        for item in failures[:5]:
            print(f"    work_id={item['work_id']}: {item['error']}")
    if summary.get("truncated_work_ids"):
        print(f"  截断作品 work_id: {summary['truncated_work_ids']}")


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
        summary = v05_data.build_dataset(
            catalog,
            config,
            version=args.version,
            sequence_version=args.sequence_version,
            clip_frame_version=args.clip_frame_version,
            dino_frame_version=args.dino_frame_version,
            force=bool(args.force),
            limit=args.limit,
            work_ids=args.work_ids,
            max_segments=args.max_segments,
        )
    except NebulaError as exc:
        print(f"v0.5 token 数据集构建失败: {exc}", file=sys.stderr)
        return 1
    finally:
        catalog.close()

    _print_summary(summary)
    if summary["n_failed"] or summary["n_missing"]:
        print(
            f"存在 {summary['n_failed']} 个失败、{summary['n_missing']} 个缺失源；"
            "修复后可用 --force/同命令重跑。",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
