"""音画错位实验 CLI（A6）：10 部真实作品 + 真实 PE-AV，回答同步敏感度问题。

用法（项目根目录）::

    .\\.venv\\Scripts\\python.exe tools/desync_experiment.py
    .\\.venv\\Scripts\\python.exe tools/desync_experiment.py --limit 10
    .\\.venv\\Scripts\\python.exe tools/desync_experiment.py --work-ids 1 2 3 --out-dir logs

行为：只读打开 data/catalog.sqlite（URI mode=ro，不写 data/），取 source=local 且
status=completed 的前 10 部作品（或 --work-ids 指定）；按正式采样参数切片段，
生成 A reference / quality（crf 28 再编码基线）/ B 音频+0.5s / C 音频+1.0s /
D 同作品其他片段音频交换 变体；用真实 PEAVEmbedder（device 走 config.model.device）
编码并计算 Δ = 1 - cos(joint/audio/video)。

输出：<out-dir>/av_desync_experiment.json（完整明细 + 聚合统计 + 结论）
      <out-dir>/av_desync_experiment.csv（每个片段×变体一行）
默认 out-dir 为配置里的 logs 目录；临时片段写在 temp/ 下并在结束后清理。
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nebula.avsync import (  # noqa: E402
    DEFAULT_SHIFTS_S,
    QUALITY_DEFAULT_CRF,
    VariantBuild,
    build_report,
    build_work_variants,
    evaluate_variants,
    format_summary,
    write_report,
)
from nebula.config import Config, load_config  # noqa: E402
from nebula.embed import PEAVEmbedder  # noqa: E402
from nebula.errors import NebulaError  # noqa: E402
from nebula.log import configure_logging  # noqa: E402
from nebula.media import probe_media  # noqa: E402
from nebula.sampling import plan_clips  # noqa: E402

JSON_FILENAME = "av_desync_experiment.json"
CSV_FILENAME = "av_desync_experiment.csv"
DEFAULT_LIMIT = 10

logger = logging.getLogger("desync_experiment")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/desync_experiment.py",
        description="A6 音画错位实验：真实 PE-AV 编码，对比普通再编码 / 音频偏移 / 音频交换。",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="配置文件路径（默认取 NEBULA_CONFIG 环境变量，否则当前目录 config.yaml）",
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, help=f"作品数（默认 {DEFAULT_LIMIT}）"
    )
    parser.add_argument(
        "--work-ids",
        nargs="+",
        type=int,
        default=None,
        help="指定 work_id（必须是本地已完成作品；给了该参数时不再套用 --limit）",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="报告输出目录（默认使用配置里的 logs 目录）",
    )
    parser.add_argument("--keep", action="store_true", help="保留 temp/ 下的变体文件")
    return parser


def _resolve_config_path(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    env_value = os.environ.get("NEBULA_CONFIG")
    if env_value:
        return Path(env_value).expanduser()
    return Path.cwd() / "config.yaml"


def load_local_completed_works(
    db_path: Path, *, limit: int | None = None, work_ids: list[int] | None = None
) -> list[dict[str, Any]]:
    """只读打开 SQLite，返回 source=local 且 completed 的作品（按 id 升序）。

    使用 URI mode=ro，任何写操作都会失败；不调用 Catalog 以避免 WAL 等写入。
    """

    if not db_path.is_file():
        raise NebulaError(f"数据库不存在: {db_path}")
    uri = db_path.resolve().as_uri() + "?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=30.0)
    except sqlite3.Error as exc:
        raise NebulaError(f"只读打开数据库失败（{db_path}）：{exc}") from exc
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT id, title, local_path, duration_s FROM works "
            "WHERE source = 'local' AND status = 'completed' AND local_path IS NOT NULL "
            "ORDER BY id"
        ).fetchall()
    except sqlite3.Error as exc:
        raise NebulaError(f"查询本地已完成作品失败：{exc}") from exc
    finally:
        connection.close()
    works = [dict(row) for row in rows]
    if work_ids is not None:
        wanted = list(dict.fromkeys(int(value) for value in work_ids))
        by_id = {int(work["id"]): work for work in works}
        missing = [work_id for work_id in wanted if work_id not in by_id]
        if missing:
            raise NebulaError(f"指定的 work_id 不是本地已完成作品: {missing}")
        works = [by_id[work_id] for work_id in wanted]
    elif limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise NebulaError(f"--limit 必须是正整数，实际 {limit!r}")
        works = works[:limit]
    if not works:
        raise NebulaError("没有可用的本地已完成作品")
    return works


def _run_experiment(
    config: Config,
    *,
    limit: int,
    work_ids: list[int] | None,
    keep: bool,
    out_dir: Path,
) -> dict[str, Any]:
    """执行完整实验，返回 {report, json_path, csv_path, elapsed_s, encode_s, temp_dir}。"""

    started = time.perf_counter()
    work_list = load_local_completed_works(
        config.paths.db_path, limit=limit, work_ids=work_ids
    )
    temp_root = Path(
        tempfile.mkdtemp(prefix="nebula_desync_", dir=str(config.paths.temp))
    )
    logger.info("临时目录: %s", temp_root)
    logger.info("参与实验的作品: %s", [int(work["id"]) for work in work_list])

    embedder = PEAVEmbedder(config.model)
    builds: list[VariantBuild] = []
    work_meta: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    encode_s = 0.0
    try:
        for index, work in enumerate(work_list, start=1):
            work_id = int(work["id"])
            local_path = str(work["local_path"])
            title = work.get("title") or Path(local_path).stem
            source = config.local_work_path(local_path)
            if not source.is_file():
                skipped.append(
                    {"work_id": work_id, "reason": f"本地视频不存在: {source}"}
                )
                logger.warning("works_id=%s 本地视频不存在，已跳过: %s", work_id, source)
                continue
            work_started = time.perf_counter()
            try:
                info = probe_media(
                    source,
                    ffprobe=config.media.ffprobe,
                    timeout_s=config.media.probe_timeout_seconds,
                )
                plans = plan_clips(
                    info.duration_s,
                    clip_seconds=config.media.clip_seconds,
                    clips_per_work=config.media.clips_per_work,
                    min_duration_seconds=config.media.min_duration_seconds,
                )
                build = build_work_variants(
                    source,
                    info,
                    plans,
                    work_id=work_id,
                    out_dir=temp_root / f"work_{work_id:06d}",
                    media=config.media,
                )
            except NebulaError as exc:
                skipped.append({"work_id": work_id, "reason": str(exc)})
                logger.warning("work_id=%s 变体构建失败，已跳过: %s", work_id, exc)
                continue
            builds.append(build)
            work_meta.append(
                {
                    "work_id": work_id,
                    "title": title,
                    "local_path": local_path,
                    "duration_s": info.duration_s,
                }
            )
            for reason in build.skipped:
                skipped.append({"work_id": work_id, "reason": reason})
            logger.info(
                "作品 %d/%d 变体构建完成: work_id=%s 片段=%d 用时=%.1fs（%s）",
                index,
                len(work_list),
                work_id,
                len(build.clips),
                time.perf_counter() - work_started,
                title,
            )
        if not builds:
            raise NebulaError("没有任何作品成功构建变体，无法进行音画错位实验")

        logger.info(
            "开始 PE-AV 编码: %d 部作品 / %d 段 / %d 对变体",
            len(builds),
            sum(len(build.clips) for build in builds),
            sum(len(clip.pairs) for build in builds for clip in build.clips),
        )
        encode_started = time.perf_counter()
        results = evaluate_variants(builds, embedder=embedder)
        encode_s = time.perf_counter() - encode_started

        model_info = {
            "checkpoint": config.model.checkpoint,
            "version": config.model.version,
            "revision": config.model.revision,
            "embedder": type(embedder).__name__,
            "is_test_only": bool(embedder.is_test_only),
            "device": getattr(embedder, "_device", None),
            "autocast_bfloat16": bool(getattr(embedder, "_use_autocast", False)),
            "model_load_s": round(float(getattr(embedder, "model_load_s", 0.0)), 2),
            "batch_size": int(config.model.batch_size),
        }
        config_info = {
            "clip_seconds": float(config.media.clip_seconds),
            "clips_per_work": int(config.media.clips_per_work),
            "quality_crf": int(QUALITY_DEFAULT_CRF),
            "shifts_s": [float(value) for value in DEFAULT_SHIFTS_S],
            "batch_size": int(config.model.batch_size),
        }
        elapsed = time.perf_counter() - started
        report = build_report(
            embedding_version=config.embedding_version,
            model_info=model_info,
            config_info=config_info,
            work_meta=work_meta,
            results=results,
            skipped=skipped,
            extra_notes=[
                f"共耗时 {elapsed:.1f}s，其中 PE-AV 编码 {encode_s:.1f}s。",
            ],
        )
        json_path = out_dir / JSON_FILENAME
        csv_path = out_dir / CSV_FILENAME
        write_report(report, json_path=json_path, csv_path=csv_path)
        return {
            "report": report,
            "json_path": json_path,
            "csv_path": csv_path,
            "elapsed_s": elapsed,
            "encode_s": encode_s,
            "temp_dir": temp_root,
        }
    finally:
        embedder.close()
        if keep:
            logger.info("按 --keep 保留变体临时目录: %s", temp_root)
        else:
            shutil.rmtree(temp_root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(_resolve_config_path(args.config))
    except NebulaError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2
    configure_logging(config.logging.level)
    out_dir = (
        Path(args.out_dir).expanduser() if args.out_dir else config.paths.logs
    )

    try:
        outcome = _run_experiment(
            config,
            limit=args.limit,
            work_ids=args.work_ids,
            keep=args.keep,
            out_dir=out_dir,
        )
    except NebulaError as exc:
        print(f"音画错位实验失败: {exc}", file=sys.stderr)
        return 1

    report = outcome["report"]
    json_path = outcome["json_path"]
    csv_path = outcome["csv_path"]
    print(format_summary(report))
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print(
        f"  耗时: 总计 {outcome['elapsed_s']:.1f}s"
        f"（其中 PE-AV 编码 {outcome['encode_s']:.1f}s）"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
