"""音MAD 星云 · 流水线命令行。

只暴露方案文档要求的简单操作：

  python pipeline.py process-local        处理 input/local 下的本地视频（P0）
  python pipeline.py run-pending          继续处理尚未完成的作品（可限定来源/榜单）
                                           例: run-pending --source bilibili --list-key votes_2025 --list-limit 30
  python pipeline.py retry 12 34          单独重试指定 work_id
  python pipeline.py timings              查看最近批次的单部分阶段耗时（A4）
  python pipeline.py ablate --limit 3     P0 音画消融排查实验
  python pipeline.py status               查看作品/向量/星云状态
  python pipeline.py clean-temp           清理失败残留的临时目录

全局选项 --config 可指定配置文件；默认使用项目根目录 config.yaml。
v0.4 采样选项（process-local / run-pending / retry 均可用）：
  --sampling {legacy,full_coverage}  覆盖 config.sampling.mode
  --clip-height H                    切片时把视频缩放到高度 H（如 336，生成 _h336 序列版本）
  --allow-long                       full_coverage 下允许处理超过 media.long_video_seconds 的超长视频
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

from nebula.config import load_config
from nebula.errors import NebulaError
from nebula.log import configure_logging


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    """给跑批子命令添加 v0.4 采样/切片/超长视频选项。"""

    parser.add_argument(
        "--sampling",
        choices=["legacy", "full_coverage"],
        default=None,
        help="覆盖 config.sampling.mode：legacy=旧 5 片段；full_coverage=全片覆盖",
    )
    parser.add_argument(
        "--clip-height",
        type=int,
        default=None,
        help="切片时把视频缩放到该高度（如 336，生成 _h336 序列版本）；默认用配置值",
    )
    parser.add_argument(
        "--allow-long",
        action="store_true",
        help="full_coverage 下允许处理超过 media.long_video_seconds 的超长视频",
    )


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
    _add_run_options(local)

    pending = sub.add_parser("run-pending", help="继续处理 pending/failed 作品")
    pending.add_argument("--limit", type=int, default=None, help="最多处理多少部")
    pending.add_argument(
        "--source", choices=["all", "local", "bilibili"], default="all", help="限定作品来源"
    )
    pending.add_argument(
        "--list-key", default=None, help="只处理某个榜单中的作品（如 votes_2025），按排名排序"
    )
    pending.add_argument(
        "--list-limit",
        type=int,
        default=None,
        help="配合 --list-key：只取榜单前 N 条（先截取再过滤已完成）",
    )
    pending.add_argument("--no-retry-failed", action="store_true", help="不重试 failed 作品")
    pending.add_argument(
        "--test-embedder", action="store_true", help="使用测试用 HashTestEmbedder（仅调试）"
    )
    _add_run_options(pending)

    retry = sub.add_parser("retry", help="重试指定作品（可一次多个 work_id）")
    retry.add_argument("work_ids", nargs="+", type=int, help="作品内部 ID")
    retry.add_argument(
        "--test-embedder", action="store_true", help="使用测试用 HashTestEmbedder（仅调试）"
    )
    _add_run_options(retry)

    ablate = sub.add_parser("ablate", help="P0 消融实验：静音/遮蔽画面后联合向量是否变化")
    ablate.add_argument("--limit", type=int, default=3, help="抽查作品数")
    ablate.add_argument(
        "--work-id", dest="work_ids", type=int, action="append", help="指定作品 ID（可重复）"
    )
    ablate.add_argument(
        "--test-embedder", action="store_true", help="使用测试用 HashTestEmbedder（仅调试）"
    )

    timings = sub.add_parser("timings", help="查看分阶段耗时（A4，读取数据库）")
    timings.add_argument("--run-id", type=int, default=None, help="批次 ID（默认最近一批）")
    timings.add_argument("--limit-runs", type=int, default=5, help="列出最近多少个批次")

    sub.add_parser("status", help="查看作品、向量与星云状态")
    sub.add_parser("clean-temp", help="清空 temp 下流水线临时目录")
    return parser


def _make_pipeline(
    config,
    *,
    test_embedder: bool,
    sampling_mode: str | None = None,
    clip_height: int | None = None,
    allow_long: bool = False,
):
    """构造 Pipeline；--clip-height 通过替换 frozen config.media 实现。"""

    from nebula.pipeline import Pipeline

    if clip_height is not None:
        if clip_height < 2 or clip_height % 2 != 0:
            raise NebulaError(f"--clip-height 必须是 >=2 的偶数，实际 {clip_height}")
        config = replace(config, media=replace(config.media, clip_height=int(clip_height)))
    if test_embedder:
        from nebula.embed import HashTestEmbedder

        return Pipeline(
            config,
            embedder=HashTestEmbedder(),
            allow_test_embedder=True,
            sampling_mode=sampling_mode,
            allow_long=allow_long,
        )
    return Pipeline(config, sampling_mode=sampling_mode, allow_long=allow_long)


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

            with _make_pipeline(
                config,
                test_embedder=args.test_embedder,
                sampling_mode=args.sampling,
                clip_height=args.clip_height,
                allow_long=args.allow_long,
            ) as pipeline:
                if pipeline.embedder.is_test_only:
                    print("警告：使用测试用嵌入器，生成的向量不能用于真实观察", file=sys.stderr)
                print(
                    f"采样模式: {pipeline.sampling_mode}；序列版本: {pipeline.embedding_version}"
                )
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
            with _make_pipeline(
                config,
                test_embedder=args.test_embedder,
                sampling_mode=args.sampling,
                clip_height=args.clip_height,
                allow_long=args.allow_long,
            ) as pipeline:
                if pipeline.embedder.is_test_only:
                    print("警告：使用测试用嵌入器，生成的向量不能用于真实观察", file=sys.stderr)
                print(
                    f"采样模式: {pipeline.sampling_mode}；序列版本: {pipeline.embedding_version}"
                )
                stats = pipeline.process_pending(
                    source=None if args.source == "all" else args.source,
                    limit=args.limit,
                    retry_failed=not args.no_retry_failed,
                    list_key=args.list_key,
                    list_limit=args.list_limit,
                )
                print(stats.summary())
                _print_failures(stats)
                return 0 if stats.failed == 0 else 1

        if args.command == "retry":
            with _make_pipeline(
                config,
                test_embedder=args.test_embedder,
                sampling_mode=args.sampling,
                clip_height=args.clip_height,
                allow_long=args.allow_long,
            ) as pipeline:
                print(
                    f"采样模式: {pipeline.sampling_mode}；序列版本: {pipeline.embedding_version}"
                )
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

        if args.command == "timings":
            from nebula.db import Catalog

            catalog = Catalog(config.paths.db_path)
            catalog.initialize()
            try:
                runs = catalog.list_pipeline_runs(limit=args.limit_runs)
                print("最近流水线批次:")
                for run in runs:
                    total = run["total_s"]
                    print(
                        f"  run_id={run['id']} version={run['embedding_version']} "
                        f"selected={run['n_selected']} completed={run['n_completed']} "
                        f"failed={run['n_failed']} model_load_s={run['model_load_s']} "
                        f"total_s={None if total is None else round(total, 2)}"
                    )
                run_id = args.run_id
                if run_id is None and runs:
                    run_id = int(runs[0]["id"])
                if run_id is not None:
                    rows = catalog.list_work_timings(run_id=run_id)
                    print(f"\nrun_id={run_id} 单部分阶段耗时（{len(rows)} 行）:")
                    print(
                        "work_id  total_s  download   probe  extract  preprocess  forward"
                        "  events    save"
                    )
                    for row in rows:
                        print(
                            f"{row['work_id']:>7}  {row['total_s']:>7.2f}  "
                            f"{row['download_s']:>8.2f}  {row['probe_s']:>6.2f}  "
                            f"{row['extract_s']:>7.2f}  {row['preprocess_s']:>10.2f}  "
                            f"{row['forward_s']:>7.2f}  {row['events_s']:>7.2f}  "
                            f"{row['save_s']:>4.2f}"
                        )
                return 0
            finally:
                catalog.close()

        if args.command == "status":
            with _make_pipeline(
                config, test_embedder=getattr(args, "test_embedder", False)
            ) as pipeline:
                print(json.dumps(pipeline.status(), ensure_ascii=False, indent=2))
                return 0

        if args.command == "clean-temp":
            with _make_pipeline(
                config, test_embedder=getattr(args, "test_embedder", False)
            ) as pipeline:
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
