"""临时片段分辨率对照实验（A5-3，只读实验，不改默认配置 / 向量版本）。

固定选择 input/local 中时长最长的 3 部真实视频（与 bench_ffmpeg.py 相同规则），
每部按生产参数切 5 段 8 秒片段，比较：

- 方案 A（现状）：extract_clip 输出原始分辨率 mp4 + wav；
- 方案 B（候选）：切片时直接缩放到高度 336（-vf scale=-2:336）的 mp4 + wav。
  这是在工具内用等价的单条 ffmpeg 命令模拟“未来给 extract_clip 增加可选
  scale_height”的场景，不修改 extract_clip 接口与 config.yaml。

测量：
- 两种做法的 extract 耗时、mp4 体积；
- 真实 PE-AV 编码耗时（A4 的 batch.preprocess_s / forward_s 与墙钟）；
- 片段级 joint / video 向量余弦与最大绝对差、作品级聚合向量余弦；
- 基于数据给出“是否建议采用预缩放”的结论。

输出人类可读表格 + JSON；JSON 同时写入 --json-out（默认 temp/bench_preprocess.json）。

用法（项目根目录）：

    .\\.venv\\Scripts\\python.exe tools/bench_preprocess.py
    .\\.venv\\Scripts\\python.exe tools/bench_preprocess.py --keep
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from nebula.config import Config, load_config  # noqa: E402
from nebula.embed import PEAVEmbedder, aggregate_work_vector  # noqa: E402
from nebula.errors import MediaError  # noqa: E402
from nebula.media import MediaInfo, extract_clip, probe_media  # noqa: E402
from nebula.sampling import ClipPlan, plan_clips  # noqa: E402

logger = logging.getLogger("bench_preprocess")

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


def _run_ffmpeg(command: list[str], *, what: str, timeout_s: float) -> None:
    """执行 ffmpeg 命令；失败时抛带 stderr 摘要的 MediaError。"""

    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        raise MediaError(f"找不到 ffmpeg 可执行文件: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"{what}超时（>{timeout_s:g}s）") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-800:]
        raise MediaError(f"{what}失败（退出码 {proc.returncode}）：{tail}")


def _extract_original_batch(
    source: Path, info: MediaInfo, plans: list[ClipPlan], out_dir: Path, media
) -> dict[str, object]:
    """现状做法：extract_clip 原始分辨率切片，逐片段计时。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    per_clip_s: list[float] = []
    for plan in plans:
        started = time.perf_counter()
        clip = extract_clip(source, plan, out_dir, media=media, info=info)
        per_clip_s.append(time.perf_counter() - started)
        clips.append(clip)
    total_s = sum(per_clip_s)
    mp4_bytes = sum(clip.video_path.stat().st_size for clip in clips)
    return {
        "clips": clips,
        "seconds_per_clip": [round(value, 4) for value in per_clip_s],
        "seconds_total": round(total_s, 3),
        "mp4_bytes": mp4_bytes,
        "mp4_mb": round(mp4_bytes / 1e6, 2),
    }


def _extract_scaled_batch(
    source: Path,
    info: MediaInfo,
    plans: list[ClipPlan],
    out_dir: Path,
    media,
    *,
    scale_height: int,
) -> dict[str, object]:
    """候选做法：单条 ffmpeg 命令切片并缩放到高度 scale_height（模拟未来接口）。"""

    if not info.has_audio:
        raise MediaError(f"缩放实验要求源视频带音轨: {source}")
    if scale_height <= 0 or scale_height % 2 != 0:
        raise MediaError(f"scale_height 必须是正偶数: {scale_height!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rate = int(media.audio_sample_rate)
    channels = int(media.audio_channels)
    per_clip_s: list[float] = []
    pairs: list[tuple[Path, Path]] = []
    for plan in plans:
        video_path = out_dir / f"clip_{plan.index:02d}.mp4"
        audio_path = out_dir / f"clip_{plan.index:02d}.wav"
        command = [
            str(media.ffmpeg),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-ss",
            f"{plan.start_s:.3f}",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-t",
            f"{plan.duration_s:.3f}",
            "-vf",
            f"scale=-2:{scale_height}",
            "-c:v",
            "libx264",
            "-preset",
            str(media.video_preset),
            "-crf",
            str(int(media.video_crf)),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(video_path),
            "-map",
            "0:a:0",
            "-t",
            f"{plan.duration_s:.3f}",
            "-vn",
            "-c:a",
            "pcm_s16le",
            "-ar",
            str(rate),
            "-ac",
            str(channels),
            str(audio_path),
        ]
        started = time.perf_counter()
        _run_ffmpeg(command, what=f"缩放切片 {plan.index} ", timeout_s=media.extract_timeout_seconds)
        per_clip_s.append(time.perf_counter() - started)
        for path in (video_path, audio_path):
            if not path.is_file() or path.stat().st_size <= 0:
                raise MediaError(f"缩放切片未生成有效文件: {path}")
        pairs.append((video_path, audio_path))
    total_s = sum(per_clip_s)
    mp4_bytes = sum(video.stat().st_size for video, _ in pairs)
    return {
        "pairs": pairs,
        "seconds_per_clip": [round(value, 4) for value in per_clip_s],
        "seconds_total": round(total_s, 3),
        "mp4_bytes": mp4_bytes,
        "mp4_mb": round(mp4_bytes / 1e6, 2),
    }


def _encode_batch(
    embedder: PEAVEmbedder, pairs: list[tuple[Path, Path]]
) -> dict[str, object]:
    """真实 PE-AV 编码一批片段，返回向量与 A4 分阶段耗时。"""

    started = time.perf_counter()
    batch = embedder.encode_pairs(pairs)
    wall_s = time.perf_counter() - started
    return {
        "batch": batch,
        "wall_seconds": round(wall_s, 3),
        "preprocess_s": round(batch.preprocess_s, 3),
        "forward_s": round(batch.forward_s, 3),
    }


def _compare_vectors(
    original: np.ndarray, scaled: np.ndarray
) -> tuple[list[float], float]:
    """逐行余弦 + 最大绝对差；输入已逐行归一化，余弦即点积。"""

    cosines = [float(np.dot(original[row], scaled[row])) for row in range(original.shape[0])]
    max_diff = float(np.abs(original - scaled).max())
    return cosines, max_diff


def _print_table(rows: list[dict[str, object]], scale_height: int) -> None:
    header = (
        f"{'video':<32} {'ext_orig':>9} {'ext_scale':>9} {'x':>5} "
        f"{'prep_orig':>9} {'prep_scale':>10} {'joint_cos_min':>13} {'work_cos':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        name = str(row["video"])
        if len(name) > 31:
            name = "..." + name[-28:]
        print(
            f"{name:<32} "
            f"{row['extract_original_s']:>9.2f} {row['extract_scaled_s']:>9.2f} "
            f"{row['extract_speedup']:>5.2f} "
            f"{row['preprocess_original_s']:>9.3f} {row['preprocess_scaled_s']:>10.3f} "
            f"{row['joint_cosine_min']:>13.6f} {row['work_cosine']:>8.6f}"
        )
    print(f"（缩放高度 {scale_height}px；prep = PE-AV processor 预处理秒数）")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="切片分辨率（预缩放）对照实验")
    parser.add_argument("--videos", type=int, default=3, help="参与实验的真实视频数（默认 3）")
    parser.add_argument("--clips", type=int, default=5, help="每部视频片段数（默认 5）")
    parser.add_argument("--clip-seconds", type=float, default=8.0, help="片段秒数（默认 8）")
    parser.add_argument("--scale-height", type=int, default=336, help="候选方案缩放高度（默认 336）")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "temp" / "bench_preprocess.json",
        help="JSON 输出路径",
    )
    parser.add_argument("--keep", action="store_true", help="保留临时工作目录")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(REPO_ROOT / "config.yaml")
    workdir = Path(tempfile.mkdtemp(prefix="nebula_bench_preprocess_", dir=str(config.paths.temp)))
    logger.info("工作目录: %s", workdir)
    evidence: dict[str, object] = {
        "clips_per_video": args.clips,
        "clip_seconds": args.clip_seconds,
        "scale_height": args.scale_height,
        "default_video_preset": config.media.video_preset,
        "candidate_note": (
            "方案 B 在工具内用单条 ffmpeg 命令（-vf scale=-2:高度）模拟未来 "
            "extract_clip 可选 scale_height 的切片耗时，未修改接口与 config.yaml"
        ),
        "results": [],
    }
    embedder = PEAVEmbedder(config.model)
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

        rows: list[dict[str, object]] = []
        for index, (path, info) in enumerate(picked):
            plans = list(
                plan_clips(
                    info.duration_s,
                    clip_seconds=args.clip_seconds,
                    clips_per_work=args.clips,
                    min_duration_seconds=config.media.min_duration_seconds,
                )
            )
            original = _extract_original_batch(
                path, info, plans, workdir / f"v{index:02d}" / "original", config.media
            )
            scaled = _extract_scaled_batch(
                path,
                info,
                plans,
                workdir / f"v{index:02d}" / "scaled",
                config.media,
                scale_height=args.scale_height,
            )
            original_clips = original["clips"]
            original_pairs = [(clip.video_path, clip.audio_path) for clip in original_clips]
            scaled_pairs = scaled["pairs"]
            assert isinstance(original_pairs, list) and isinstance(scaled_pairs, list)

            encoded_original = _encode_batch(embedder, original_pairs)
            encoded_scaled = _encode_batch(embedder, scaled_pairs)
            batch_original = encoded_original["batch"]
            batch_scaled = encoded_scaled["batch"]

            joint_cosines, joint_max_diff = _compare_vectors(
                batch_original.joint, batch_scaled.joint
            )
            video_cosines, video_max_diff = _compare_vectors(
                batch_original.video, batch_scaled.video
            )
            audio_cosines, audio_max_diff = _compare_vectors(
                batch_original.audio, batch_scaled.audio
            )
            work_cosine = float(
                np.dot(
                    aggregate_work_vector(batch_original.joint),
                    aggregate_work_vector(batch_scaled.joint),
                )
            )
            extract_speedup = (
                float(original["seconds_total"]) / float(scaled["seconds_total"])
            )
            row: dict[str, object] = {
                "video": path.relative_to(REPO_ROOT).as_posix(),
                "duration_s": round(info.duration_s, 2),
                "width": info.width,
                "height": info.height,
                "extract_original_s": original["seconds_total"],
                "extract_scaled_s": scaled["seconds_total"],
                "extract_speedup": round(extract_speedup, 3),
                "mp4_original_mb": original["mp4_mb"],
                "mp4_scaled_mb": scaled["mp4_mb"],
                "mp4_ratio": round(float(scaled["mp4_mb"]) / float(original["mp4_mb"]), 3),
                "preprocess_original_s": encoded_original["preprocess_s"],
                "preprocess_scaled_s": encoded_scaled["preprocess_s"],
                "forward_original_s": encoded_original["forward_s"],
                "forward_scaled_s": encoded_scaled["forward_s"],
                "encode_wall_original_s": encoded_original["wall_seconds"],
                "encode_wall_scaled_s": encoded_scaled["wall_seconds"],
                "joint_cosine_min": round(min(joint_cosines), 8),
                "joint_max_abs_diff": joint_max_diff,
                "video_cosine_min": round(min(video_cosines), 8),
                "video_max_abs_diff": video_max_diff,
                "audio_cosine_min": round(min(audio_cosines), 8),
                "audio_max_abs_diff": audio_max_diff,
                "work_cosine": round(work_cosine, 8),
            }
            rows.append(row)
            logger.info(
                "v%d 完成: extract %.2fs vs %.2fs (x%.2f)，joint_min_cos=%.8f work_cos=%.8f",
                index,
                row["extract_original_s"],
                row["extract_scaled_s"],
                extract_speedup,
                row["joint_cosine_min"],
                work_cosine,
            )
        evidence["results"] = rows
        evidence["model_load_s"] = round(embedder.model_load_s, 2)
        evidence["runtime"] = {
            "device": embedder._device,  # type: ignore[attr-defined]
            "autocast_bfloat16": embedder._use_autocast,  # type: ignore[attr-defined]
        }

        n = len(rows)
        summary = {
            "mean_extract_original_s": round(
                sum(float(row["extract_original_s"]) for row in rows) / n, 3
            ),
            "mean_extract_scaled_s": round(
                sum(float(row["extract_scaled_s"]) for row in rows) / n, 3
            ),
            "extract_speedup": round(
                sum(float(row["extract_original_s"]) for row in rows)
                / sum(float(row["extract_scaled_s"]) for row in rows),
                3,
            ),
            "mean_preprocess_original_s": round(
                sum(float(row["preprocess_original_s"]) for row in rows) / n, 3
            ),
            "mean_preprocess_scaled_s": round(
                sum(float(row["preprocess_scaled_s"]) for row in rows) / n, 3
            ),
            "mean_mp4_original_mb": round(
                sum(float(row["mp4_original_mb"]) for row in rows) / n, 2
            ),
            "mean_mp4_scaled_mb": round(
                sum(float(row["mp4_scaled_mb"]) for row in rows) / n, 2
            ),
            "joint_cosine_min": min(float(row["joint_cosine_min"]) for row in rows),
            "work_cosine_min": min(float(row["work_cosine"]) for row in rows),
        }
        summary["preprocess_speedup"] = round(
            float(summary["mean_preprocess_original_s"])
            / float(summary["mean_preprocess_scaled_s"]),
            3,
        )
        mean_forward_original = sum(float(row["forward_original_s"]) for row in rows) / n
        mean_forward_scaled = sum(float(row["forward_scaled_s"]) for row in rows) / n
        mean_total_original = (
            float(summary["mean_extract_original_s"])
            + float(summary["mean_preprocess_original_s"])
            + mean_forward_original
        )
        mean_total_scaled = (
            float(summary["mean_extract_scaled_s"])
            + float(summary["mean_preprocess_scaled_s"])
            + mean_forward_scaled
        )
        summary["mean_forward_original_s"] = round(mean_forward_original, 3)
        summary["mean_forward_scaled_s"] = round(mean_forward_scaled, 3)
        summary["mean_estimated_total_per_work_original_s"] = round(mean_total_original, 2)
        summary["mean_estimated_total_per_work_scaled_s"] = round(mean_total_scaled, 2)
        summary["estimated_total_speedup"] = round(mean_total_original / mean_total_scaled, 3)
        speedup = float(summary["estimated_total_speedup"])
        joint_cosine = float(summary["joint_cosine_min"])
        work_cosine = float(summary["work_cosine_min"])
        if joint_cosine >= 0.999 and speedup >= 1.3:
            recommendation = (
                "建议：端到端提速明显且向量漂移很小（片段级 joint cosine >= 0.999）。"
                "若采用需 Lead 决策 embedding_version 语义（缩放会改变向量，不能与未缩放"
                "向量混用），并整批重跑 P0。"
            )
        elif speedup >= 1.3:
            recommendation = (
                f"可选：端到端提速约 {speedup:.2f}x，但片段级 joint cosine 最低 "
                f"{joint_cosine:.6f}、作品级 {work_cosine:.6f}，向量有可测漂移。"
                "是否接受由 Lead 按 P0 对“形态相似”的要求决策；若采用必须更换 "
                "embedding_version 并整批重跑。"
            )
        else:
            recommendation = (
                f"不建议：端到端提速有限（{speedup:.2f}x），不值得引入向量漂移与"
                "版本语义变更。"
            )
        summary["recommendation"] = recommendation
        evidence["summary"] = summary

        _print_table(rows, args.scale_height)
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
        embedder.close()
        if args.keep:
            logger.info("按 --keep 保留工作目录: %s", workdir)
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
