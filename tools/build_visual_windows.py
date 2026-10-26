"""v0.4.3.1 从已保存帧特征重建窗口层（窗口计划层分离，不下载/不解码/不跑模型）。

用法：

    python tools/build_visual_windows.py                     # 默认窗口计划（config）
    python tools/build_visual_windows.py --force             # 强制重建
    python tools/build_visual_windows.py --dry-run           # 只统计将处理项
    python tools/build_visual_windows.py --window-seconds 5 --overlap-seconds 0 \
        --weight-scheme equal                                # 新窗口计划（独立版本目录/DB 行）
    python tools/build_visual_windows.py --frame-version <slug> --limit 5
    python tools/build_visual_windows.py --all --fps 8       # 扫描 data/visual_dense 全部版本

实现读取 data/visual_dense/<frame_version>/<work_id>/dense_visual.npz 的帧级向量，
按窗口计划（window_seconds + overlap_seconds + weight_scheme）池化，
输出 data/visual_windows/<window_plan_version>/<frame_version>/<work_id>/
（windows.npy / window_std.npy / window_weights.npy / manifest.json），
并登记 visual_window_plans / visual_window_sequences。窗口权重 coverage-v1 使用
sampling.coverage_weights（覆盖计数积分，sum=1），equal 为 1/T。

不下载视频、不解码、不加载视觉模型；已完成的窗口层默认跳过，--force 才重建。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nebula.config import load_config  # noqa: E402
from nebula.db import Catalog  # noqa: E402
from nebula.errors import NebulaError  # noqa: E402
from nebula.visual_pipeline import (  # noqa: E402
    WEIGHT_SCHEMES,
    build_visual_windows,
)

LOGGER = logging.getLogger("build_visual_windows")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_visual_windows.py",
        description="从已保存的帧级特征重建窗口层（窗口计划层分离，不跑模型）",
    )
    parser.add_argument("--config", metavar="PATH", default=None, help="配置文件路径（默认当前目录 config.yaml）")
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        metavar="SEC",
        help="窗口时长（秒）；默认 config.media.clip_seconds",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=None,
        metavar="SEC",
        help="相邻窗口重叠（秒）；默认 config.sampling.overlap_seconds",
    )
    parser.add_argument(
        "--weight-scheme",
        choices=WEIGHT_SCHEMES,
        default=None,
        help="窗口权重方案：coverage-v1（默认，覆盖计数积分）| equal（1/T）",
    )
    parser.add_argument(
        "--frame-version",
        action="append",
        default=None,
        metavar="SLUG",
        help="只处理指定帧编码版本（可重复）",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="扫描 data/visual_dense 下所有帧版本目录（含 DB 未登记的版本会报告并跳过）",
    )
    parser.add_argument("--fps", type=float, default=None, help="只处理该帧率的帧版本")
    parser.add_argument("--limit", type=int, default=None, help="最多处理多少部作品（按 work_id 升序）")
    parser.add_argument("--force", action="store_true", help="重建已完成的窗口层")
    parser.add_argument("--dry-run", action="store_true", help="只统计将处理项，不写 DB/文件")
    parser.add_argument("--quiet", action="store_true", help="不打印逐作品进度")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=None,
        help="日志级别（默认取 config.logging.level）",
    )
    return parser


def _print_summary(result: dict) -> None:
    plan = result["plan"]
    versions = result["frame_versions"]
    print(f"窗口计划: {result['window_plan_version']}（{plan.label}）")
    print(f"帧编码版本 {len(versions)} 个: {', '.join(versions) if versions else '（无）'}")
    print(f"canonical 作品 {result['works']} 部，候选组合 {result['total_pairs']} 个")
    if result["dry_run"]:
        print(
            f"dry-run：将处理 {result['would_process']} 个 (帧版本, 作品)，"
            f"跳过已完成 {result['skipped']} 个，缺帧 {result['missing_frames']} 个；"
            f"耗时 {result['elapsed_s']:.2f}s"
        )
        return
    print(
        f"完成 {result['created']} 个窗口序列（共 {result['total_windows']} 窗口），"
        f"跳过已完成 {result['skipped']} 个，缺帧 {result['missing_frames']} 个，"
        f"失败 {len(result['failed'])} 个，耗时 {result['elapsed_s']:.2f}s"
    )
    for failure in result["failed"][:10]:
        detail = failure.get("error") or failure.get("reason") or ""
        frame_version = failure.get("frame_version", "?")
        work_id = failure.get("work_id")
        prefix = f"  - frame_version={frame_version}"
        if work_id is not None:
            prefix += f" work_id={work_id}"
        print(f"{prefix}: {detail}")
    if len(result["failed"]) > 10:
        print(f"  ... 其余 {len(result['failed']) - 10} 条见日志")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        config = load_config(args.config) if args.config else load_config()
        logging.basicConfig(
            level=getattr(
                logging,
                str(args.log_level or config.logging.level).upper(),
                logging.INFO,
            ),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        config.ensure_directories()

        def progress(text: str) -> None:
            if not args.quiet:
                print(text, flush=True)

        with Catalog(config.paths.db_path) as catalog:
            catalog.initialize()
            result = build_visual_windows(
                catalog,
                config,
                window_seconds=args.window_seconds,
                overlap_seconds=args.overlap_seconds,
                weight_scheme=args.weight_scheme,
                frame_versions=args.frame_version,
                all_versions=args.all,
                fps=args.fps,
                limit=args.limit,
                force=args.force,
                dry_run=args.dry_run,
                progress=progress,
            )
        print()
        _print_summary(result)
        return 1 if result["failed"] else 0
    except NebulaError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"错误：文件系统操作失败：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("操作已取消", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
