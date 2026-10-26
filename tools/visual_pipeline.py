"""v0.4.3 密集视觉批处理命令行入口。

命令：

    python tools/visual_pipeline.py run          # 批量生成 canonical 作品的密集视觉序列
    python tools/visual_pipeline.py status       # 各视觉版本状态计数与存储占用
    python tools/visual_pipeline.py list         # 已登记的视觉版本列表
    python tools/visual_pipeline.py clean-temp   # 清理视觉任务全部完成的 temp 目录

run 默认处理全部 canonical + completed 作品，每个配置编码器跑主帧率（8fps），
并动态选取 subset_count 部作品追加 30fps 子集任务；已完成的 (work, version) 自动跳过。
不加载模型时（例如 --help / status / list）不会触发任何网络或 GPU。
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
    clean_completed_temp_dirs,
    run_visual_pipeline,
    visual_status,
)

LOGGER = logging.getLogger("visual_pipeline_cli")


def _format_bytes(value: int) -> str:
    number = float(max(0, int(value)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if number < 1024.0 or unit == "TB":
            return f"{number:.2f} {unit}" if unit != "B" else f"{int(number)} B"
        number /= 1024.0
    return f"{int(value)} B"


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        metavar="PATH",
        default=argparse.SUPPRESS,
        help="配置文件路径（默认当前目录 config.yaml）",
    )
    common.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=argparse.SUPPRESS,
        help="日志级别（默认取 config.logging.level）",
    )
    parser = argparse.ArgumentParser(
        prog="visual_pipeline.py",
        description="v0.4.3 密集视觉语义批处理（canonical 作品，断点续跑）",
        parents=[common],
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="子命令")

    run_parser = subparsers.add_parser(
        "run",
        parents=[common],
        help="批量生成密集视觉序列（主帧率 + 30fps 子集）",
        description="对 canonical + completed 作品批量执行冻结视觉编码与窗口池化。",
    )
    run_parser.add_argument("--limit", type=int, default=None, help="最多处理的作品数（按 work_id 升序）")
    run_parser.add_argument(
        "--work-ids",
        type=int,
        nargs="+",
        default=None,
        metavar="ID",
        help="只处理指定 work_id（仍要求 canonical + completed）",
    )
    run_parser.add_argument(
        "--no-subset",
        action="store_true",
        help="不生成 30fps 子集任务（只跑主帧率版本）",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印选中作品/任务数/版本，不写 DB、不建文件、不下载",
    )
    run_parser.add_argument(
        "--quiet",
        action="store_true",
        help="不打印逐作品进度，只打印最终摘要",
    )
    run_parser.set_defaults(handler=_cmd_run)

    status_parser = subparsers.add_parser(
        "status",
        parents=[common],
        help="查看各视觉版本状态计数与存储占用",
        description="汇总 visual_versions / visual_sequences 的 completed/in_progress/pending/failed 计数。",
    )
    status_parser.set_defaults(handler=_cmd_status)

    list_parser = subparsers.add_parser(
        "list",
        parents=[common],
        help="列出已登记的视觉版本",
        description="打印 visual_versions 表（model/revision/fps/size/preprocess/dim）。",
    )
    list_parser.set_defaults(handler=_cmd_list)

    clean_parser = subparsers.add_parser(
        "clean-temp",
        parents=[common],
        help="清理视觉任务全部完成的 temp/work_XXXXXX 目录",
        description="只删除该作品所有需要的视觉任务全部 completed 且文件齐全的临时目录。",
    )
    clean_parser.set_defaults(handler=_cmd_clean_temp)
    return parser


def _cmd_run(catalog: Catalog, config, args: argparse.Namespace) -> int:
    def progress(text: str) -> None:
        if not args.quiet:
            print(text, flush=True)

    summary = run_visual_pipeline(
        catalog,
        config,
        limit=args.limit,
        work_ids=args.work_ids,
        include_subset=not args.no_subset,
        dry_run=args.dry_run,
        progress=progress,
    )
    print()
    print(summary.summary_text())
    if args.dry_run:
        print("dry-run：未写入数据库、未创建文件、未下载；去掉 --dry-run 才会真正执行。")
    return 0


def _cmd_status(catalog: Catalog, config, args: argparse.Namespace) -> int:
    payload = visual_status(catalog, config)
    versions = payload["versions"]
    if not versions:
        print("还没有登记任何视觉序列版本（先运行 tools/visual_pipeline.py run）。")
        return 0
    print(f"密集视觉版本状态（canonical + completed 作品期望数：{payload['expected_works']}）")
    header = (
        f"{'version':<46} {'model':<34} {'fps':>5} {'size':>5} {'prep':>9} "
        f"{'dim':>5} {'完成':>4} {'进行':>4} {'待办':>4} {'失败':>4} {'存储':>10}"
    )
    print(header)
    print("-" * len(header))
    for version in versions:
        counts = version["status_counts"]
        model = str(version["model_id"] or "")
        if len(model) > 32:
            model = "…" + model[-31:]
        prep = str(version["preprocessing"] or "")
        print(
            f"{str(version['version']):<46} {model:<34} {float(version['fps']):>5.0f} "
            f"{int(version['size']):>5} {prep:>9} {str(version['dim'] or '-'):>5} "
            f"{counts.get('completed', 0):>4} {counts.get('in_progress', 0):>4} "
            f"{counts.get('pending', 0):>4} {counts.get('failed', 0):>4} "
            f"{_format_bytes(version['storage_bytes']):>10}"
        )
        if version["missing_files"]:
            print(f"  ⚠ 有 {version['missing_files']} 个 completed 记录的文件缺失")
    print()
    print(f"总存储占用：{_format_bytes(payload['total_storage_bytes'])}")
    return 0


def _cmd_list(catalog: Catalog, config, args: argparse.Namespace) -> int:
    versions = catalog.list_visual_versions()
    if not versions:
        print("还没有登记任何视觉序列版本。")
        return 0
    header = f"{'version':<46} {'model':<34} {'revision':<42} {'fps':>5} {'size':>5} {'prep':>9} {'dim':>5}"
    print(header)
    print("-" * len(header))
    for record in versions:
        print(
            f"{str(record['version']):<46} {str(record['model_id']):<34} "
            f"{str(record['revision']):<42} {float(record['fps']):>5.0f} "
            f"{int(record['size']):>5} {str(record['preprocessing']):>9} "
            f"{str(record.get('dim') or '-'):>5}"
        )
    print()
    print(f"共 {len(versions)} 个视觉序列版本。")
    return 0


def _cmd_clean_temp(catalog: Catalog, config, args: argparse.Namespace) -> int:
    result = clean_completed_temp_dirs(catalog, config)
    print(
        f"clean-temp 完成：删除 {result['removed_dirs']} 个目录，"
        f"保留 {result['kept_dirs']} 个（未完成），跳过 {result['skipped_dirs']} 个，"
        f"释放 {_format_bytes(result['freed_bytes'])}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    config_path = getattr(args, "config", None)
    log_level = getattr(args, "log_level", None)
    try:
        config = load_config(config_path) if config_path else load_config()
        logging.basicConfig(
            level=getattr(logging, str(log_level or config.logging.level).upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        config.ensure_directories()
        with Catalog(config.paths.db_path) as catalog:
            catalog.initialize()
            return int(args.handler(catalog, config, args))
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
