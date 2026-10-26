"""v0.4.3 密集视觉预处理对照实验 CLI（stretch vs letterbox，不写 DB）。

背景：用户要求正式全量前修正 scale=224:224 的宽高比拉伸问题。本工具对指定少量
本地已完成作品，用同一个冻结视觉编码器分别按 stretch（拉伸，旧行为）与
letterbox（保持比例 + 黑边，候选）预处理提取帧级向量，按生产窗口（默认 8s
全片覆盖窗口）池化后，在同一向量空间比较两种预处理的窗口向量余弦、帧数与
解码/编码耗时，输出 JSON 与中文摘要，供 Lead 决策。

用法（项目根目录）::

    .\\.venv\\Scripts\\python.exe tools/dense_preprocess_compare.py
    .\\.venv\\Scripts\\python.exe tools/dense_preprocess_compare.py --limit 3 ^
        --encoder clip-vit-base-patch16 --fps 8 --max-windows 5 --json
    .\\.venv\\Scripts\\python.exe tools/dense_preprocess_compare.py --work-ids 1 2 3

说明：

- 只读打开 data/catalog.sqlite（sqlite3 URI mode=ro），不写 DB、不写向量文件；
  唯一输出是 JSON 报告（默认 logs/dense_preprocess_compare.json）；
- 编码器 revision 必须固定（--revision 或 config.visual_dense.encoders）；
- 同一编码器实例复用（模型只加载一次），耗时对比看 decode_s/encode_s；
- 真实模型权重由调用方准备（HF_ENDPOINT 等环境变量自行设置），本工具不下载。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from nebula.config import Config, load_config  # noqa: E402
from nebula.dense_visual import (  # noqa: E402
    DenseVisualError,
    HFImageEncoder,
    extract_dense_features,
    pool_windows,
    visual_version_slug,
)
from nebula.errors import NebulaError  # noqa: E402
from nebula.log import configure_logging  # noqa: E402
from nebula.sampling import plan_full_coverage  # noqa: E402
from nebula.util import now_utc_iso  # noqa: E402

ENCODER_ALIASES: dict[str, str] = {
    "clip-vit-base-patch16": "openai/clip-vit-base-patch16",
    "clip-vit-base-patch32": "openai/clip-vit-base-patch32",
    "clip-vit-large-patch14": "openai/clip-vit-large-patch14",
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-base": "facebook/dinov2-base",
}
"""常见编码器别名 -> Hugging Face model id。"""

PREPROCESS_PAIR: tuple[str, str] = ("stretch", "letterbox")
"""对照的两种预处理方式。"""

logger = logging.getLogger("dense_preprocess_compare")


def resolve_encoder_id(value: str) -> str:
    """把编码器别名解析成 Hugging Face model id；未知输入原样返回。"""

    raw = str(value or "").strip()
    if not raw:
        raise DenseVisualError("--encoder 不能为空")
    return ENCODER_ALIASES.get(raw, raw)


def resolve_encoder_spec(
    config: Config, *, encoder: str | None, revision: str | None
) -> tuple[str, str]:
    """确定 (model_id, revision)：缺省取配置里第一个编码器，revision 必须固定。"""

    if encoder:
        model_id = resolve_encoder_id(encoder)
    else:
        if not config.visual_dense.encoders:
            raise DenseVisualError("config.visual_dense.encoders 为空，无法确定默认编码器")
        model_id = config.visual_dense.encoders[0].model
    if revision:
        return model_id, revision
    for entry in config.visual_dense.encoders:
        if entry.model == model_id:
            return model_id, entry.revision
    raise DenseVisualError(
        f"编码器 {model_id} 没有固定 revision：请用 --revision 指定，"
        "或先在 config.visual_dense.encoders 登记 model + revision"
    )


def _readonly_connection(db_path: Path) -> sqlite3.Connection:
    """以只读 URI 打开 SQLite，保证本工具不会写数据库。"""

    if not db_path.is_file():
        raise NebulaError(f"数据库不存在: {db_path}")
    uri = db_path.resolve().as_uri() + "?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=30.0)
    except sqlite3.Error as exc:
        raise NebulaError(f"只读打开数据库失败（{db_path}）：{exc}") from exc
    connection.row_factory = sqlite3.Row
    return connection


def load_local_completed_works(
    db_path: Path, *, work_ids: Sequence[int] | None = None, limit: int | None = None
) -> list[dict[str, Any]]:
    """只读查询 source=local 且 completed 的作品（按 id 升序）。

    work_ids 给定时按给定顺序过滤（缺失即报错）；否则应用 limit。
    """

    connection = _readonly_connection(db_path)
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


def _windows_for(
    config: Config, duration_s: float, *, max_windows: int | None
) -> list[tuple[float, float]]:
    """按生产参数生成 8s 覆盖窗口；max_windows 可截断用于快速 pilot。"""

    windows = plan_full_coverage(
        float(duration_s),
        clip_seconds=config.media.clip_seconds,
        overlap_seconds=config.sampling.overlap_seconds,
        min_duration_seconds=config.media.min_duration_seconds,
    )
    pairs = [(float(window.start_s), float(window.end_s)) for window in windows]
    if max_windows is not None:
        if max_windows < 1:
            raise NebulaError(f"--max-windows 必须 >= 1，实际 {max_windows}")
        pairs = pairs[:max_windows]
    if not pairs:
        raise NebulaError(f"作品时长 {duration_s:.3f}s 没有可用窗口")
    return pairs


def _cosine_stats(reference: np.ndarray, other: np.ndarray) -> dict[str, float]:
    """逐窗口余弦（输入已逐行归一化，余弦=点积）。"""

    if reference.shape != other.shape:
        raise DenseVisualError(
            f"两种预处理的窗口向量形状不一致: {reference.shape} != {other.shape}"
        )
    values = np.sum(reference.astype(np.float64) * other.astype(np.float64), axis=1)
    values = np.clip(values, -1.0, 1.0)
    return {
        "mean": round(float(values.mean()), 6),
        "median": round(float(np.median(values)), 6),
        "min": round(float(values.min()), 6),
        "max": round(float(values.max()), 6),
        "std": round(float(values.std(ddof=0)), 6),
    }


def compare_work(
    config: Config,
    encoder: HFImageEncoder,
    work: dict[str, Any],
    *,
    fps: float,
    size: int,
    batch_size: int,
    max_windows: int | None,
) -> dict[str, Any]:
    """对一部作品跑 stretch 与 letterbox，返回窗口余弦与耗时对照。"""

    work_id = int(work["id"])
    title = work.get("title") or Path(str(work["local_path"])).stem
    source = config.local_work_path(str(work["local_path"]))
    if not source.is_file():
        raise NebulaError(f"本地视频不存在: {source}")
    records: dict[str, dict[str, Any]] = {}
    pooled: dict[str, np.ndarray] = {}
    window_pairs: list[tuple[float, float]] | None = None
    for mode in PREPROCESS_PAIR:
        wall_started = time.perf_counter()
        features = extract_dense_features(
            source,
            encoder,
            fps=fps,
            size=size,
            batch_size=batch_size,
            ffmpeg=config.media.ffmpeg,
            preprocess=mode,
            timeout_s=config.media.extract_timeout_seconds,
        )
        wall_s = time.perf_counter() - wall_started
        if window_pairs is None:
            window_pairs = _windows_for(
                config, features.duration_s, max_windows=max_windows
            )
        pooled[mode] = pool_windows(features, window_pairs)
        records[mode] = {
            "version": features.version,
            "version_slug": visual_version_slug(features.version),
            "preprocess": features.preprocess,
            "n_frames": int(features.n_frames),
            "duration_s": round(float(features.duration_s), 3),
            "dim": int(features.dim),
            "decode_s": round(float(features.decode_s), 3),
            "encode_s": round(float(features.encode_s), 3),
            "model_load_s": round(float(features.model_load_s), 3),
            "wall_s": round(float(wall_s), 3),
        }
        del features
    assert window_pairs is not None
    cosine = _cosine_stats(pooled[PREPROCESS_PAIR[0]], pooled[PREPROCESS_PAIR[1]])
    decode_ratio = (
        records["letterbox"]["decode_s"] / records["stretch"]["decode_s"]
        if records["stretch"]["decode_s"] > 1e-9
        else None
    )
    encode_ratio = (
        records["letterbox"]["encode_s"] / records["stretch"]["encode_s"]
        if records["stretch"]["encode_s"] > 1e-9
        else None
    )
    logger.info(
        "work_id=%s 对照完成：窗口余弦 mean=%.6f min=%.6f，帧数 %s/%s",
        work_id,
        cosine["mean"],
        cosine["min"],
        records["stretch"]["n_frames"],
        records["letterbox"]["n_frames"],
    )
    return {
        "work_id": work_id,
        "title": title,
        "local_path": str(work["local_path"]),
        "duration_s": records["stretch"]["duration_s"],
        "n_frames": records["stretch"]["n_frames"],
        "n_windows": len(window_pairs),
        "window_shape": [int(value) for value in pooled["stretch"].shape],
        "frames_equal": records["stretch"]["n_frames"] == records["letterbox"]["n_frames"],
        "modes": records,
        "cosine": cosine,
        "timing_ratio": {
            "letterbox_over_stretch_decode": (
                None if decode_ratio is None else round(float(decode_ratio), 4)
            ),
            "letterbox_over_stretch_encode": (
                None if encode_ratio is None else round(float(encode_ratio), 4)
            ),
        },
    }


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """聚合多部作品的对照结果。"""

    valid = [item for item in results if "cosine" in item]
    if not valid:
        return {}
    means = [float(item["cosine"]["mean"]) for item in valid]
    mins = [float(item["cosine"]["min"]) for item in valid]
    decode_ratios = [
        float(item["timing_ratio"]["letterbox_over_stretch_decode"])
        for item in valid
        if item["timing_ratio"]["letterbox_over_stretch_decode"] is not None
    ]
    encode_ratios = [
        float(item["timing_ratio"]["letterbox_over_stretch_encode"])
        for item in valid
        if item["timing_ratio"]["letterbox_over_stretch_encode"] is not None
    ]
    return {
        "n_works": len(valid),
        "window_cosine_mean": round(float(np.mean(means)), 6),
        "window_cosine_min": round(float(np.min(mins)), 6),
        "all_windows_identical": bool(np.min(mins) >= 0.999999),
        "files_expected": PREPROCESS_PAIR,
        "frames_equal_all": all(bool(item["frames_equal"]) for item in valid),
        "mean_decode_ratio_letterbox_over_stretch": (
            round(float(np.mean(decode_ratios)), 4) if decode_ratios else None
        ),
        "mean_encode_ratio_letterbox_over_stretch": (
            round(float(np.mean(encode_ratios)), 4) if encode_ratios else None
        ),
        "versions": {mode: sorted({item["modes"][mode]["version"] for item in valid}) for mode in PREPROCESS_PAIR},
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    except OSError as exc:
        raise NebulaError(f"写入对照报告失败（{path}）：{exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


def _print_summary(report: dict[str, Any], *, out_path: Path) -> None:
    model = report.get("model") or {}
    print("密集视觉预处理对照（stretch vs letterbox）")
    print(f"  编码器: {model.get('model')}（revision={model.get('revision')}）")
    print(
        f"  fps={report['config']['fps']} size={report['config']['size']} "
        f"batch_size={report['config']['batch_size']} "
        f"max_windows={report['config']['max_windows']}"
    )
    for item in report.get("works", []):
        if "cosine" not in item:
            print(f"  work_id={item.get('work_id')}: 失败（{item.get('error')}）")
            continue
        cosine = item["cosine"]
        stretch = item["modes"]["stretch"]
        letterbox = item["modes"]["letterbox"]
        print(
            f"  work_id={item['work_id']} 帧={item['n_frames']} 窗口={item['n_windows']} "
            f"余弦 mean={cosine['mean']:.6f} min={cosine['min']:.6f}；"
            f"decode {stretch['decode_s']:.2f}s/{letterbox['decode_s']:.2f}s "
            f"encode {stretch['encode_s']:.2f}s/{letterbox['encode_s']:.2f}s"
        )
    summary = report.get("summary") or {}
    if summary:
        print(
            f"  汇总: {summary['n_works']} 部；窗口余弦 mean={summary['window_cosine_mean']:.6f} "
            f"min={summary['window_cosine_min']:.6f}；帧数一致={summary['frames_equal_all']}；"
            f"decode 比值={summary['mean_decode_ratio_letterbox_over_stretch']}，"
            f"encode 比值={summary['mean_encode_ratio_letterbox_over_stretch']}"
        )
    print(f"  JSON: {out_path}")
    for note in report.get("notes", []):
        print(f"  注意: {note}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/dense_preprocess_compare.py",
        description="密集视觉预处理对照：stretch vs letterbox 的窗口向量/耗时（只读 DB，不写向量）。",
    )
    parser.add_argument("--config", default=None, help="配置文件路径（默认 ./config.yaml）")
    parser.add_argument(
        "--work-ids", nargs="+", type=int, default=None, help="指定 work_id（本地已完成作品）"
    )
    parser.add_argument(
        "--limit", type=int, default=3, help="未指定 --work-ids 时取前 N 部作品（默认 3）"
    )
    parser.add_argument(
        "--encoder",
        default=None,
        help="编码器别名或 Hugging Face model id（默认 config.visual_dense.encoders[0]）",
    )
    parser.add_argument("--revision", default=None, help="模型 revision（默认配置固定 revision）")
    parser.add_argument("--fps", type=float, default=None, help="帧级采样率（默认配置）")
    parser.add_argument("--size", type=int, default=None, help="输入边长（默认配置）")
    parser.add_argument("--batch-size", type=int, default=None, help="每批编码帧数（默认配置）")
    parser.add_argument(
        "--max-windows", type=int, default=None, help="每部作品最多池化窗口数（默认全部）"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="JSON 报告路径（默认 logs/dense_preprocess_compare.json）",
    )
    parser.add_argument("--json", action="store_true", help="同时把 JSON 报告打印到 stdout")
    return parser


def _resolve_config_path(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    env_value = os.environ.get("NEBULA_CONFIG")
    if env_value:
        return Path(env_value).expanduser()
    return Path.cwd() / "config.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(_resolve_config_path(args.config))
    except NebulaError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2
    configure_logging(config.logging.level)
    encoder: HFImageEncoder | None = None
    try:
        settings = config.visual_dense
        model_id, revision = resolve_encoder_spec(
            config, encoder=args.encoder, revision=args.revision
        )
        fps = float(args.fps if args.fps is not None else settings.fps)
        size = int(args.size if args.size is not None else settings.size)
        batch_size = int(
            args.batch_size if args.batch_size is not None else settings.batch_size
        )
        works = load_local_completed_works(
            config.paths.db_path, work_ids=args.work_ids, limit=args.limit
        )
        out_path = (
            Path(args.out).expanduser()
            if args.out
            else config.paths.logs / "dense_preprocess_compare.json"
        )
        encoder = HFImageEncoder(
            model_id,
            revision=revision,
            cache_dir=settings.cache_dir,
            device=settings.device,
            dtype=settings.dtype,
        )
        results: list[dict[str, Any]] = []
        for position, work in enumerate(works, start=1):
            logger.info(
                "对照作品 %d/%d: work_id=%s", position, len(works), work["id"]
            )
            try:
                results.append(
                    compare_work(
                        config,
                        encoder,
                        work,
                        fps=fps,
                        size=size,
                        batch_size=batch_size,
                        max_windows=args.max_windows,
                    )
                )
            except NebulaError as exc:
                logger.warning("work_id=%s 对照失败: %s", work["id"], exc)
                results.append(
                    {"work_id": int(work["id"]), "error": str(exc), "title": work.get("title")}
                )
        report = {
            "experiment": "dense_preprocess_compare",
            "created_at": now_utc_iso(),
            "model": {"model": model_id, "revision": revision},
            "config": {
                "fps": fps,
                "size": size,
                "batch_size": batch_size,
                "preprocess_pair": list(PREPROCESS_PAIR),
                "max_windows": args.max_windows,
                "clip_seconds": float(config.media.clip_seconds),
                "overlap_seconds": float(config.sampling.overlap_seconds),
            },
            "works": results,
            "summary": summarize(results),
            "notes": [
                "stretch=旧行为（scale=size:size 拉伸，破坏宽高比）；"
                "letterbox=保持比例缩小 + 黑边补齐。",
                "窗口余弦=同一编码器同一作品两种预处理池化向量的逐窗口 cosine；"
                "1.0 表示窗口向量完全一致。",
                "模型只加载一次；耗时对比看 decode_s/encode_s（不含模型加载），"
                "首轮 stretch 的 model_load_s 可能非零。",
            ],
        }
        if not report["summary"]:
            raise NebulaError("所有作品对照都失败，无法生成有效报告")
        _write_json_atomic(out_path, report)
    except NebulaError as exc:
        print(f"预处理对照失败: {exc}", file=sys.stderr)
        return 1
    finally:
        if encoder is not None:
            encoder.close()

    if args.json:
        print(json.dumps(report, ensure_ascii=False))
    else:
        _print_summary(report, out_path=out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
