"""v0.4.1 事件扫描手动验证工具（对一个本地视频打印窗口特征表/聚合）。

用途：t3 交付后供 Lead/人工快速检查 nebula.events 在真实视频上的表现：

- 用真实 ffmpeg 单次解码全片，打印音频/视频/音画序列的形状与采样率；
- 按全片覆盖窗口（优先 t1 的 sampling.plan_full_coverage；尚未交付时用等宽回退）
  打印每个窗口的 32 维特征关键列；
- 打印 32 维窗口特征的整片均值/标准差，以及 cut / onset 事件总量；
- 可选 --json 输出完整窗口特征（便于 diff/归档）、--save-bundle 写出事件 bundle。

不下载视频、不写数据库、不加载模型、不占用 GPU。

用法::

    .venv/Scripts/python.exe tools/events_probe.py input/local/xxx.mp4
    .venv/Scripts/python.exe tools/events_probe.py xxx.mp4 --max-rows 5 --json logs/events_probe.json
    .venv/Scripts/python.exe tools/events_probe.py xxx.mp4 --save-bundle temp/events_demo
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nebula.config import Config, load_config  # noqa: E402
from nebula.errors import NebulaError  # noqa: E402
from nebula.events import (  # noqa: E402
    AUDIO_FEATURE_NAMES,
    VIDEO_FEATURE_NAMES,
    WINDOW_FEATURE_NAMES,
    EventFeatures,
    save_events_bundle,
    scan_events,
)
from nebula.media import probe_media  # noqa: E402

LOGGER = logging.getLogger("events_probe")

# 表格中展示的窗口特征列（其余 32 维见 JSON / 聚合输出）
_TABLE_COLUMNS: tuple[str, ...] = (
    "window_duration_s",
    "audio_onset_rate",
    "audio_onset_mean",
    "audio_rms_mean",
    "video_frame_diff_mean",
    "video_cut_rate",
    "video_flicker_rate",
    "av_max_correlation",
    "av_best_lag_s",
)


@dataclass(frozen=True)
class _ProbePlan:
    """等宽回退窗口（字段与 CoverageClip 契约一致，duck typing）。"""

    index: int
    start_s: float
    duration_s: float
    valid_duration_s: float
    weight: float

    @property
    def end_s(self) -> float:
        return round(self.start_s + self.duration_s, 6)


def _fallback_plans(
    duration_s: float, *, clip_seconds: float, overlap_seconds: float
) -> tuple[_ProbePlan, ...]:
    """sampling.plan_full_coverage 尚未交付时的等宽覆盖回退（仅供探针展示）。"""

    if duration_s <= clip_seconds:
        return (_ProbePlan(0, 0.0, round(duration_s, 3), round(duration_s, 3), 1.0),)
    step_max = clip_seconds - overlap_seconds
    if step_max <= 0:
        step_max = clip_seconds
    delta = duration_s - clip_seconds
    count = int(math.ceil(delta / step_max)) + 1
    step = delta / (count - 1)
    starts = [round(index * step, 3) for index in range(count - 1)]
    starts.append(round(duration_s - clip_seconds, 3))
    ends = [min(duration_s, start + clip_seconds) for start in starts]
    # 按覆盖计数计算权重（与契约定义一致）：每段覆盖计数 c(t)，权重 = ∫(1/c)/duration
    boundaries = sorted({0.0, duration_s, *starts, *ends})
    total_weight = [0.0] * count
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        if right - left <= 1e-9:
            continue
        covering = [
            index
            for index, (start, end) in enumerate(zip(starts, ends, strict=True))
            if start <= left + 1e-9 and end >= right - 1e-9
        ]
        if not covering:
            continue
        share = (right - left) / len(covering)
        for index in covering:
            total_weight[index] += share
    plans = []
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        weight = total_weight[index] / duration_s if duration_s > 0 else 1.0 / count
        plans.append(
            _ProbePlan(
                index=index,
                start_s=start,
                duration_s=round(end - start, 3),
                valid_duration_s=round(end - start, 3),
                weight=round(weight, 9),
            )
        )
    return tuple(plans)


def _build_plans(
    config: Config,
    duration_s: float,
    *,
    clip_seconds: float,
    overlap_seconds: float,
) -> tuple[list[Any], str]:
    """优先使用 t1 的 plan_full_coverage；尚不可用时回退并给出提示。"""

    try:
        from nebula.sampling import plan_full_coverage  # type: ignore[attr-defined]
    except ImportError:
        plan_full_coverage = None
    if plan_full_coverage is not None:
        plans = plan_full_coverage(
            duration_s,
            clip_seconds=clip_seconds,
            overlap_seconds=overlap_seconds,
            min_duration_seconds=config.media.min_duration_seconds,
        )
        return list(plans), "sampling.plan_full_coverage"
    plans = _fallback_plans(
        duration_s,
        clip_seconds=clip_seconds,
        overlap_seconds=overlap_seconds,
    )
    return list(plans), "fallback_uniform_coverage（t1 plan_full_coverage 未交付）"


def _format_seconds(value: Any) -> str:
    return f"{float(value):.3f}"


def _window_table(features: EventFeatures, plans: list[Any], max_rows: int) -> None:
    indexes = {name: WINDOW_FEATURE_NAMES.index(name) for name in _TABLE_COLUMNS}
    header = (
        f"{'#':>3} {'start':>8} {'end':>8} {'dur':>7} {'weight':>7} | "
        + " | ".join(f"{name[:22]:>22}" for name in _TABLE_COLUMNS)
    )
    print(header)
    print("-" * len(header))
    rows = features.window_features
    limit = rows.shape[0] if max_rows <= 0 else min(max_rows, rows.shape[0])
    for index in range(limit):
        plan = plans[index]
        start = float(getattr(plan, "start_s", 0.0))
        end = float(getattr(plan, "end_s", start + float(getattr(plan, "duration_s", 0.0))))
        weight = getattr(plan, "weight", None)
        weight_text = f"{float(weight):.4f}" if weight is not None else "-"
        values = " | ".join(
            f"{float(rows[index, indexes[name]]):>22.5f}" for name in _TABLE_COLUMNS
        )
        print(
            f"{index:>3} {start:>8.3f} {end:>8.3f} {end - start:>7.3f} {weight_text:>7} | {values}"
        )
    if limit < rows.shape[0]:
        print(f"... 其余 {rows.shape[0] - limit} 个窗口省略（--max-rows 调整 / --json 输出完整数据）")


def _print_aggregate(features: EventFeatures) -> None:
    mean = features.window_features.mean(axis=0)
    std = features.window_features.std(axis=0)
    print()
    print("窗口特征整片聚合（32 维，固定顺序）：")
    for index, name in enumerate(WINDOW_FEATURE_NAMES):
        print(f"  {index:>2} {name:<31} mean={float(mean[index]):>12.6f}  std={float(std[index]):>12.6f}")


def _event_summary(features: EventFeatures) -> dict[str, Any]:
    onset = features.audio[:, AUDIO_FEATURE_NAMES.index("onset_strength")]
    threshold = float(onset.mean() + onset.std())
    onset_events = int(np.count_nonzero(onset > threshold))
    cuts = int(np.count_nonzero(features.video[:, VIDEO_FEATURE_NAMES.index("cut")] > 0.5))
    return {
        "onset_events": onset_events,
        "video_cuts": cuts,
        "onset_rate_per_s": onset_events / features.duration_s if features.duration_s > 0 else 0.0,
        "cut_rate_per_s": cuts / features.duration_s if features.duration_s > 0 else 0.0,
    }


def _window_payload(features: EventFeatures, plans: list[Any]) -> list[dict[str, Any]]:
    """把每个窗口的计划信息与 32 维特征整理成 JSON 行。"""

    payload: list[dict[str, Any]] = []
    for index, plan in enumerate(plans):
        plan_weight = getattr(plan, "weight", None)
        payload.append(
            {
                "index": int(getattr(plan, "index", index)),
                "start_s": float(getattr(plan, "start_s", 0.0)),
                "duration_s": float(getattr(plan, "duration_s", 0.0)),
                "weight": None if plan_weight is None else float(plan_weight),
                "features": [float(value) for value in features.window_features[index]],
            }
        )
    return payload


def _write_json(
    path: Path,
    *,
    features: EventFeatures,
    plans: list[Any],
    plan_source: str,
    events: dict[str, Any],
) -> None:
    payload = {
        "version": features.version,
        "plan_source": plan_source,
        "duration_s": features.duration_s,
        "audio_rate_hz": features.audio_rate_hz,
        "video_fps": features.video_fps,
        "video_fps_effective": features.video_fps_effective,
        "shapes": {
            "audio": list(features.audio.shape),
            "video": list(features.video.shape),
            "av": list(features.av.shape),
            "window_features": list(features.window_features.shape),
        },
        "feature_names": {
            "audio": list(AUDIO_FEATURE_NAMES),
            "video": list(VIDEO_FEATURE_NAMES),
            "window": list(WINDOW_FEATURE_NAMES),
        },
        "events": events,
        "aggregate": {
            "mean": [float(value) for value in features.window_features.mean(axis=0)],
            "std": [float(value) for value in features.window_features.std(axis=0)],
        },
        "windows": _window_payload(features, plans),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="events_probe.py",
        description="对一个本地视频打印 v0.4.1 高频事件扫描的窗口特征与聚合（不下载/不写 DB）",
    )
    parser.add_argument("video", help="本地视频路径")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.yaml"),
        help="配置文件路径（默认项目根 config.yaml）",
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=None,
        help="覆盖窗口长度（默认 config.media.clip_seconds）",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=None,
        help="窗口重叠秒数（默认 config.sampling.overlap_seconds）",
    )
    parser.add_argument("--audio-rate", type=int, default=None, help="音频事件率（默认 config.events.audio_rate_hz）")
    parser.add_argument("--video-fps", type=int, default=None, help="视频事件帧率（默认 config.events.video_fps）")
    parser.add_argument("--max-rows", type=int, default=20, help="表格最多展示窗口数；0 表示全部（默认 20）")
    parser.add_argument("--json", dest="json_path", default=None, help="把完整窗口特征写入该 JSON 文件")
    parser.add_argument("--save-bundle", default=None, help="把事件 bundle 保存到该目录（可选）")
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="日志级别（默认 WARNING，避免刷屏）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        config = load_config(args.config)
        video = Path(args.video)
        if not video.is_file():
            print(f"错误：视频不存在: {video}", file=sys.stderr)
            return 1
        info = probe_media(
            video,
            ffprobe=config.media.ffprobe,
            timeout_s=config.media.probe_timeout_seconds,
        )
        if not info.has_video:
            print(f"错误：视频没有视频流: {video}", file=sys.stderr)
            return 1
        clip_seconds = (
            float(args.clip_seconds)
            if args.clip_seconds is not None
            else config.media.clip_seconds
        )
        overlap_seconds = (
            float(args.overlap)
            if args.overlap is not None
            else config.sampling.overlap_seconds
        )
        plans, plan_source = _build_plans(
            config,
            info.duration_s,
            clip_seconds=clip_seconds,
            overlap_seconds=overlap_seconds,
        )
        print(f"视频: {video}")
        print(
            f"时长: {info.duration_s:.3f}s | 窗口: {len(plans)} 个（{plan_source}）"
            f" | 视频流: {info.width}x{info.height} fps={info.fps} | 音轨: {'有' if info.has_audio else '无'}"
        )
        started = time.monotonic()
        features = scan_events(
            video,
            plans,
            ffmpeg=config.media.ffmpeg,
            audio_rate_hz=args.audio_rate or config.events.audio_rate_hz,
            video_fps=args.video_fps or config.events.video_fps,
            timeout_s=max(config.media.extract_timeout_seconds, 120.0),
        )
        elapsed = time.monotonic() - started
        print(
            f"事件序列: audio={features.audio.shape} @ {features.audio_rate_hz:.1f}Hz, "
            f"video={features.video.shape} @ {features.video_fps_effective:.2f}fps, "
            f"av={features.av.shape}, windows={features.window_features.shape}"
        )
        events = _event_summary(features)
        print(
            f"事件总量: onset={events['onset_events']}（{events['onset_rate_per_s']:.2f}/s）, "
            f"video cuts={events['video_cuts']}（{events['cut_rate_per_s']:.2f}/s）, "
            f"扫描耗时 {elapsed:.2f}s"
        )
        print()
        _window_table(features, plans, int(args.max_rows))
        _print_aggregate(features)
        if args.json_path:
            _write_json(
                Path(args.json_path),
                features=features,
                plans=plans,
                plan_source=plan_source,
                events=events,
            )
            print(f"\nJSON 已写入: {args.json_path}")
        if args.save_bundle:
            save_events_bundle(Path(args.save_bundle), features)
            print(f"事件 bundle 已写入: {args.save_bundle}")
        return 0
    except (NebulaError, OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        LOGGER.exception("事件探针失败")
        return 1
    except KeyboardInterrupt:
        print("已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
