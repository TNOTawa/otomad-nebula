"""ffmpeg 切片 preset 基准（A5-2，只读实验，不改默认配置）。

固定选择 input/local 中时长最长的 3 部真实视频（按路径排序确定），每部按
生产参数（config.media.clip_seconds / clips_per_work）切 5 段，比较
media.video_preset = veryfast（当前默认）与 ultrafast 的：

- extract_clip 每片段 / 总耗时（预热后测量，传入 MediaInfo 跳过重复 ffprobe）；
- 产物 mp4 总大小与平均单片段大小；
- 相对提速比与体积膨胀比。

preset 只通过 dataclasses.replace 在内存中覆盖，不会修改 config.yaml。
输出人类可读表格 + JSON；JSON 同时写入 --json-out（默认 temp/bench_ffmpeg.json）。

用法（项目根目录）：

    .\\.venv\\Scripts\\python.exe tools/bench_ffmpeg.py
    .\\.venv\\Scripts\\python.exe tools/bench_ffmpeg.py --json-out temp/bench_ffmpeg.json
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nebula.config import Config, load_config  # noqa: E402
from nebula.errors import MediaError  # noqa: E402
from nebula.media import MediaInfo, extract_clip, probe_media  # noqa: E402
from nebula.sampling import plan_clips  # noqa: E402

logger = logging.getLogger("bench_ffmpeg")

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".m4v"}


def iter_video_files(input_local: Path) -> list[Path]:
    """递归列出本地视频文件，按 POSIX 路径排序保证选择确定。"""

    return sorted(
        (
            path
            for path in input_local.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=lambda path: path.as_posix(),
    )


def pick_benchmark_videos(
    config: Config, *, count: int = 3, clip_seconds: float = 8.0, clips_per_work: int = 5
) -> list[tuple[Path, MediaInfo]]:
    """固定挑出“时长最长、足以切满 clips_per_work 段”的前 count 部视频。"""

    candidates: list[tuple[Path, MediaInfo]] = []
    for path in iter_video_files(config.paths.input_local):
        try:
            info = probe_media(
                path,
                ffprobe=config.media.ffprobe,
                timeout_s=config.media.probe_timeout_seconds,
            )
        except MediaError as exc:
            logger.warning("跳过无法探测的视频 %s: %s", path.name, exc)
            continue
        plans = plan_clips(
            info.duration_s,
            clip_seconds=clip_seconds,
            clips_per_work=clips_per_work,
            min_duration_seconds=config.media.min_duration_seconds,
        )
        if len(plans) < clips_per_work:
            continue
        candidates.append((path, info))
    if not candidates:
        raise MediaError("input/local 中没有可切满指定段数的本地视频")
    candidates.sort(key=lambda item: (-item[1].duration_s, item[0].as_posix()))
    return candidates[:count]


def _extract_batch(
    source: Path,
    info: MediaInfo,
    plans,
    out_dir: Path,
    media,
) -> dict[str, object]:
    """用给定 media 配置切一批片段并计时；返回可序列化摘要。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    per_clip_s: list[float] = []
    mp4_bytes = 0
    wav_bytes = 0
    for plan in plans:
        started = time.perf_counter()
        clip = extract_clip(source, plan, out_dir, media=media, info=info)
        per_clip_s.append(time.perf_counter() - started)
        mp4_bytes += clip.video_path.stat().st_size
        wav_bytes += clip.audio_path.stat().st_size
    total_s = sum(per_clip_s)
    return {
        "seconds_per_clip": [round(value, 4) for value in per_clip_s],
        "seconds_total": round(total_s, 3),
        "seconds_mean_per_clip": round(total_s / len(per_clip_s), 4),
        "mp4_bytes": mp4_bytes,
        "wav_bytes": wav_bytes,
        "mp4_mb": round(mp4_bytes / 1e6, 2),
    }


def _print_table(results: list[dict[str, object]]) -> None:
    header = f"{'video':<36} {'preset':<10} {'total_s':>8} {'s/clip':>8} {'mp4_MB':>8} {'wav_MB':>8}"
    print(header)
    print("-" * len(header))
    for row in results:
        for preset, data in row["presets"].items():  # type: ignore[union-attr]
            name = str(row["video"])
            if len(name) > 35:
                name = "..." + name[-32:]
            print(
                f"{name:<36} {preset:<10} "
                f"{data['seconds_total']:>8.2f} {data['seconds_mean_per_clip']:>8.3f} "
                f"{data['mp4_mb']:>8.2f} {data['wav_bytes'] / 1e6:>8.2f}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="extract_clip 的 ffmpeg preset 基准")
    parser.add_argument("--videos", type=int, default=3, help="参与基准的真实视频数（默认 3）")
    parser.add_argument("--clips", type=int, default=5, help="每部视频片段数（默认 5）")
    parser.add_argument("--clip-seconds", type=float, default=8.0, help="片段秒数（默认 8）")
    parser.add_argument(
        "--presets", default="veryfast,ultrafast", help="逗号分隔的 preset 列表"
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "temp" / "bench_ffmpeg.json",
        help="JSON 输出路径",
    )
    parser.add_argument("--keep", action="store_true", help="保留临时工作目录")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(REPO_ROOT / "config.yaml")
    presets = [item.strip() for item in args.presets.split(",") if item.strip()]
    workdir = Path(tempfile.mkdtemp(prefix="nebula_bench_ffmpeg_", dir=str(config.paths.temp)))
    logger.info("工作目录: %s", workdir)
    evidence: dict[str, object] = {
        "clips_per_video": args.clips,
        "clip_seconds": args.clip_seconds,
        "default_video_preset": config.media.video_preset,
        "video_crf": config.media.video_crf,
        "presets": presets,
        "videos": [],
    }
    try:
        picked = pick_benchmark_videos(
            config,
            count=args.videos,
            clip_seconds=args.clip_seconds,
            clips_per_work=args.clips,
        )
        logger.info("固定选择 %s 部视频: %s", len(picked), [path.name for path, _ in picked])
        evidence["selected_videos"] = [
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "duration_s": round(info.duration_s, 2),
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
            }
            for path, info in picked
        ]

        # 预热：先各跑一个片段，避免 ffmpeg 首次启动 / 源文件冷读干扰计时
        first_path, first_info = picked[0]
        first_plan = plan_clips(
            first_info.duration_s,
            clip_seconds=args.clip_seconds,
            clips_per_work=args.clips,
        )[0]
        for preset in presets:
            _extract_batch(
                first_path,
                first_info,
                [first_plan],
                workdir / "warmup" / preset,
                replace(config.media, video_preset=preset),
            )

        rows: list[dict[str, object]] = []
        for index, (path, info) in enumerate(picked):
            plans = plan_clips(
                info.duration_s,
                clip_seconds=args.clip_seconds,
                clips_per_work=args.clips,
                min_duration_seconds=config.media.min_duration_seconds,
            )
            row: dict[str, object] = {
                "video": path.relative_to(REPO_ROOT).as_posix(),
                "duration_s": round(info.duration_s, 2),
                "width": info.width,
                "height": info.height,
                "presets": {},
            }
            for preset in presets:
                media = replace(config.media, video_preset=preset)
                stats = _extract_batch(
                    path,
                    info,
                    plans,
                    workdir / f"v{index:02d}" / preset,
                    media,
                )
                row["presets"][preset] = stats  # type: ignore[index]
                logger.info(
                    "v%d %s: %.2fs（%.3fs/片段）mp4=%.2fMB",
                    index,
                    preset,
                    stats["seconds_total"],
                    stats["seconds_mean_per_clip"],
                    stats["mp4_mb"],
                )
            rows.append(row)
        evidence["results"] = rows

        summary: dict[str, object] = {}
        for preset in presets:
            totals = [
                row["presets"][preset]["seconds_total"]  # type: ignore[index]
                for row in rows
            ]
            sizes = [
                row["presets"][preset]["mp4_mb"]  # type: ignore[index]
                for row in rows
            ]
            summary[preset] = {
                "mean_seconds_total": round(sum(totals) / len(totals), 3),
                "mean_mp4_mb": round(sum(sizes) / len(sizes), 2),
            }
        if "veryfast" in summary and "ultrafast" in summary:
            baseline = summary["veryfast"]["mean_seconds_total"]  # type: ignore[index]
            fast = summary["ultrafast"]["mean_seconds_total"]  # type: ignore[index]
            summary["speedup_ultrafast_vs_veryfast"] = round(baseline / fast, 3)
            summary["size_ratio_ultrafast_vs_veryfast"] = round(
                summary["ultrafast"]["mean_mp4_mb"] / summary["veryfast"]["mean_mp4_mb"], 3  # type: ignore[index]
            )
        evidence["summary"] = summary

        _print_table(rows)
        print()
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("JSON 已写入 %s", args.json_out)
        evidence["ok"] = True
        return 0
    finally:
        if args.keep:
            logger.info("按 --keep 保留工作目录: %s", workdir)
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
