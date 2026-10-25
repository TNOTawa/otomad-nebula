"""音MAD 星云 · 流水线命令行。

只暴露方案文档要求的简单操作：

  python pipeline.py process-local        处理 input/local 下的本地视频（P0）
  python pipeline.py run-pending          继续处理尚未完成的作品（可限定来源）
  python pipeline.py retry 12 34          单独重试指定 work_id
  python pipeline.py ablate --limit 3     P0 音画消融排查实验
  python pipeline.py status               查看作品/向量/星云状态
  python pipeline.py clean-temp           清理失败残留的临时目录

全局选项 --config 可指定配置文件；默认使用项目根目录 config.yaml。
"""

from __future__ import annotations

import argparse
import json
import sys

from nebula.config import load_config
from nebula.errors import NebulaError
from nebula.log import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline.py",
        description="音MAD星云计划：下载 -> 切片 -> PE-AV 编码 -> 保存 -> 清理",
    )
    parser.add_argument("--config", default=None, help="配置文件路径（默认 ./config.yaml）")
    sub = parser.add_subparsers(dest="command", required=True)

    local = sub.add_parser("process-local", help="扫描 input/local 并处理本地测试视频")
    local.add_argument("--limit", type=int, default=None, help="最多处理多少部（试跑用）")
    local.add_argument("--no-retry-failed", action="store_true", help="不重试之前失败的作品")
    local.add_argument(
        "--verify-sync", type=int, default=3, help="对前 N 部作品抽查音画同步（0 表示关闭）"
    )
    local.add_argument(
        "--test-embedder", action="store_true", help="使用测试用 HashTestEmbedder（仅调试）"
    )

    pending = sub.add_parser("run-pending", help="继续处理 pending/failed 作品")
    pending.add_argument("--limit", type=int, default=None, help="最多处理多少部")
    pending.add_argument(
        "--source", choices=["all", "local", "bilibili"], default="all", help="限定作品来源"
    )
    pending.add_argument("--no-retry-failed", action="store_true", help="不重试 failed 作品")
    pending.add_argument(
        "--test-embedder", action="store_true", help="使用测试用 HashTestEmbedder（仅调试）"
    )

    retry = sub.add_parser("retry", help="重试指定作品（可一次多个 work_id）")
    retry.add_argument("work_ids", nargs="+", type=int, help="作品内部 ID")
    retry.add_argument(
        "--test-embedder", action="store_true", help="使用测试用 HashTestEmbedder（仅调试）"
    )

    ablate = sub.add_parser("ablate", help="P0 消融实验：静音/遮蔽画面后联合向量是否变化")
    ablate.add_argument("--limit", type=int, default=3, help="抽查作品数")
    ablate.add_argument(
        "--work-id", dest="work_ids", type=int, action="append", help="指定作品 ID（可重复）"
    )
    ablate.add_argument(
        "--test-embedder", action="store_true", help="使用测试用 HashTestEmbedder（仅调试）"
    )

    sub.add_parser("status", help="查看作品、向量与星云状态")
    sub.add_parser("clean-temp", help="清空 temp 下流水线临时目录")
    return parser


def _make_pipeline(config, *, test_embedder: bool):
    from nebula.pipeline import Pipeline

    if test_embedder:
        from nebula.embed import HashTestEmbedder

        return Pipeline(config, embedder=HashTestEmbedder(), allow_test_embedder=True)
    return Pipeline(config)


def _print_failures(stats) -> None:
    failures = [o for o in stats.outcomes if o.status == "failed"]
    for outcome in failures:
        print(f"  失败 work_id={outcome.work_id}: {outcome.message}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        configure_logging(
            config.logging.level, log_file=config.paths.logs / f"pipeline_{args.command}.log"
        )
    except NebulaError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2

    try:
        if args.command == "process-local":
            from nebula.ingest import register_local_videos

            with _make_pipeline(config, test_embedder=args.test_embedder) as pipeline:
                if pipeline.embedder.is_test_only:
                    print("警告：使用测试用嵌入器，生成的向量不能用于真实观察", file=sys.stderr)
                summary = register_local_videos(pipeline.catalog, config)
                print(
                    f"本地扫描: {summary.files} 个视频（新登记 {summary.created}，已存在 {summary.existing}）"
                )
                stats = pipeline.process_pending(
                    source="local",
                    limit=args.limit,
                    retry_failed=not args.no_retry_failed,
                    verify_sync=args.verify_sync,
                )
                print(stats.summary())
                _print_failures(stats)
                if pipeline.sync_reports:
                    ok = sum(1 for item in pipeline.sync_reports if item.get("ok"))
                    print(
                        f"音画同步抽查: {ok}/{len(pipeline.sync_reports)} 条通过，"
                        f"详见 {config.paths.logs / 'sync_report.json'}"
                    )
                return 0 if stats.failed == 0 else 1

        if args.command == "run-pending":
            with _make_pipeline(config, test_embedder=args.test_embedder) as pipeline:
                if pipeline.embedder.is_test_only:
                    print("警告：使用测试用嵌入器，生成的向量不能用于真实观察", file=sys.stderr)
                stats = pipeline.process_pending(
                    source=None if args.source == "all" else args.source,
                    limit=args.limit,
                    retry_failed=not args.no_retry_failed,
                )
                print(stats.summary())
                _print_failures(stats)
                return 0 if stats.failed == 0 else 1

        if args.command == "retry":
            with _make_pipeline(config, test_embedder=args.test_embedder) as pipeline:
                stats = pipeline.retry_works(args.work_ids)
                print(stats.summary())
                _print_failures(stats)
                return 0 if stats.failed == 0 else 1

        if args.command == "ablate":
            with _make_pipeline(config, test_embedder=args.test_embedder) as pipeline:
                results = pipeline.ablate(limit=args.limit, work_ids=args.work_ids)
                for item in results:
                    if "error" in item:
                        print(f"work_id={item['work_id']} 失败: {item['error']}", file=sys.stderr)
                    else:
                        print(
                            f"work_id={item['work_id']} 原始 vs 静音 cos="
                            f"{item['joint_vs_mute_cosine']:.4f}，原始 vs 纯黑 cos="
                            f"{item['joint_vs_black_cosine']:.4f}"
                        )
                print(f"报告: {config.paths.logs / 'ablation_report.json'}")
                return 0 if all("error" not in item for item in results) else 1

        if args.command == "status":
            with _make_pipeline(config, test_embedder=args.test_embedder) as pipeline:
                print(json.dumps(pipeline.status(), ensure_ascii=False, indent=2))
                return 0

        if args.command == "clean-temp":
            with _make_pipeline(config, test_embedder=args.test_embedder) as pipeline:
                removed = pipeline.clean_temp()
                print(f"已清理 {removed} 个临时条目")
                return 0

        parser.error(f"未知子命令: {args.command}")
        return 2
    except NebulaError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
