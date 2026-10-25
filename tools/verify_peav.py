"""PE-AV 真实模型验证脚本（T1 手工证据，不属于默认测试）。

用法（在项目根目录执行）：

    .\\.venv\\Scripts\\python.exe tools/verify_peav.py
    .\\.venv\\Scripts\\python.exe tools/verify_peav.py --keep --clip-seconds 4

步骤：

1. 用 ffmpeg 生成 12 秒、含 48k 正弦音轨的测试视频；
2. probe_media -> plan_clips -> extract_clip 切 2 段；
3. PEAVEmbedder 加载真实权重，一次 batch 编码 2 段；
4. 校验 joint/audio/video 形状一致、逐行范数约等于 1、重复运行一致、
   batch_size=1 与默认 batch 结果一致；
5. 消融：同一片段分别换成静音 wav / 纯黑视频，计算与原片段的余弦相似度，
   并检查“只换音频不影响视频向量、只换画面不影响音频向量”；
6. 以 JSON 形式输出全部证据（形状 / 范数 / 最大绝对差 / 消融余弦 / 耗时）。
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

# 允许直接执行 tools/verify_peav.py（脚本所在目录不在 sys.path 中）：
# 把项目根目录加入 sys.path 后再导入 nebula。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from nebula.config import load_config  # noqa: E402
from nebula.embed import PEAVEmbedder, aggregate_work_vector  # noqa: E402
from nebula.media import (  # noqa: E402
    black_video_clip,
    clip_duration_report,
    extract_clip,
    mute_audio_clip,
    probe_media,
)
from nebula.sampling import plan_clips  # noqa: E402


def _make_sample_video(target: Path) -> Path:
    """生成 12 秒 testsrc2 + 440Hz 正弦的测试视频（160x120@10fps）。"""

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("缺少 ffmpeg")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=10:duration=12",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=12",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(target),
        ],
        check=True,
        capture_output=True,
    )
    return target


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left.reshape(-1), right.reshape(-1)))


def _max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(left - right).max())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PE-AV 真实模型验证（T1 证据脚本）")
    parser.add_argument("--clip-seconds", type=float, default=4.0, help="每段片段秒数（默认 4）")
    parser.add_argument("--keep", action="store_true", help="保留临时工作目录以便人工检查")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(REPO_ROOT / "config.yaml")
    workdir = Path(tempfile.mkdtemp(prefix="nebula_peav_verify_", dir=str(config.paths.temp)))
    logger = logging.getLogger("verify_peav")
    logger.info("工作目录: %s", workdir)
    evidence: dict[str, object] = {"workdir": str(workdir), "checkpoint": config.model.checkpoint}
    embedder = PEAVEmbedder(config.model)
    try:
        sample = _make_sample_video(workdir / "sample.mp4")
        info = probe_media(sample, ffprobe=config.media.ffprobe)
        evidence["sample"] = {
            "duration_s": info.duration_s,
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "has_audio": info.has_audio,
        }
        plans = plan_clips(
            info.duration_s, clip_seconds=args.clip_seconds, clips_per_work=2
        )
        clips = [extract_clip(sample, plan, workdir / "clips", media=config.media) for plan in plans]
        reports = [
            clip_duration_report(clip.video_path, clip.audio_path, ffprobe=config.media.ffprobe)
            for clip in clips
        ]
        evidence["clips"] = [
            {
                "index": clip.index,
                "start_s": clip.start_s,
                "duration_s": clip.duration_s,
                "duration_report": report,
            }
            for clip, report in zip(clips, reports, strict=True)
        ]
        logger.info("已切出 %s 段，音画时长报告: %s", len(clips), reports)

        pairs = [(clip.video_path, clip.audio_path) for clip in clips]
        started = time.time()
        batch = embedder.encode_pairs(pairs)
        first_encode_s = time.time() - started
        logger.info("首次编码（含模型加载）耗时 %.1fs", first_encode_s)
        norms = {
            "joint": np.linalg.norm(batch.joint, axis=1).tolist(),
            "audio": None if batch.audio is None else np.linalg.norm(batch.audio, axis=1).tolist(),
            "video": None if batch.video is None else np.linalg.norm(batch.video, axis=1).tolist(),
        }
        evidence["runtime"] = {
            "device": str(getattr(embedder, "_device", "unknown")),
            "autocast_bfloat16": bool(getattr(embedder, "_use_autocast", False)),
        }
        evidence["batch"] = {
            "shape_joint": list(batch.joint.shape),
            "shape_audio": None if batch.audio is None else list(batch.audio.shape),
            "shape_video": None if batch.video is None else list(batch.video.shape),
            "dtype_joint": str(batch.joint.dtype),
            "norms": norms,
            "first_encode_seconds": round(first_encode_s, 2),
        }
        assert batch.audio is not None and batch.video is not None, "模型应同时返回单模态向量"
        assert batch.audio.shape == batch.joint.shape
        assert batch.video.shape == batch.joint.shape
        for name, values in norms.items():
            assert values is not None
            assert np.allclose(np.asarray(values), 1.0, atol=1e-4), (name, values)

        started = time.time()
        repeated = embedder.encode_pairs(pairs)
        repeat_s = time.time() - started
        repeat_diff = _max_abs_diff(batch.joint, repeated.joint)
        evidence["repeat"] = {
            "joint_max_abs_diff": repeat_diff,
            "seconds": round(repeat_s, 2),
        }
        logger.info("重复运行最大绝对差 %.3e（耗时 %.1fs）", repeat_diff, repeat_s)
        assert repeat_diff <= 1e-5, "同一输入重复编码应一致"

        original_batch_size = embedder._batch_size  # type: ignore[attr-defined]
        try:
            embedder._batch_size = 1  # type: ignore[attr-defined]
            started = time.time()
            chunked = embedder.encode_pairs(pairs)
            chunk_s = time.time() - started
        finally:
            embedder._batch_size = original_batch_size  # type: ignore[attr-defined]
        chunk_diff = _max_abs_diff(batch.joint, chunked.joint)
        chunk_cosine_min = min(
            _cosine(batch.joint[row], chunked.joint[row]) for row in range(batch.joint.shape[0])
        )
        # bfloat16 autocast 下不同 batch 组合会选到不同的 GPU kernel，逐元素差可到 1e-3 量级；
        # 用余弦相似度约束语义一致性，float32 路径则保持严格阈值。
        chunk_tolerance = 5e-3 if getattr(embedder, "_use_autocast", False) else 1e-5
        evidence["batching"] = {
            "batch_size_default": original_batch_size,
            "batch_size_one_joint_max_abs_diff": chunk_diff,
            "batch_size_one_joint_min_cosine": chunk_cosine_min,
            "tolerance": chunk_tolerance,
            "seconds": round(chunk_s, 2),
        }
        logger.info(
            "batch_size=1 与默认 batch 的最大绝对差 %.3e（最小余弦 %.6f）",
            chunk_diff,
            chunk_cosine_min,
        )
        assert chunk_diff <= chunk_tolerance, "分批编码结果应与单批一致"
        assert chunk_cosine_min >= 0.9999, "分批编码结果的余弦相似度应接近 1"

        work_vector = aggregate_work_vector(batch.joint)
        evidence["work_vector"] = {
            "shape": list(work_vector.shape),
            "dtype": str(work_vector.dtype),
            "norm": float(np.linalg.norm(work_vector)),
        }

        clip = clips[0]
        muted_audio = mute_audio_clip(clip.duration_s, workdir / "muted.wav", media=config.media)
        black_video = black_video_clip(
            clip.duration_s, workdir / "black.mp4", media=config.media
        )
        started = time.time()
        muted_batch = embedder.encode_pairs([(clip.video_path, muted_audio)])
        black_batch = embedder.encode_pairs([(black_video, clip.audio_path)])
        ablation_s = time.time() - started
        # bf16 autocast 下 batch 组成不同会用不同 GPU kernel，单模态不变性也放宽到 5e-3；
        # float32 路径（CPU / CUDA fp32）仍用 1e-5 严格阈值。
        inert_tolerance = 5e-3 if getattr(embedder, "_use_autocast", False) else 1e-5
        ablation = {
            "joint_cosine_muted": _cosine(batch.joint[0], muted_batch.joint[0]),
            "joint_cosine_black": _cosine(batch.joint[0], black_batch.joint[0]),
            "audio_cosine_muted": _cosine(batch.audio[0], muted_batch.audio[0]),
            "video_cosine_black": _cosine(batch.video[0], black_batch.video[0]),
            "video_cosine_muted_invariance": _cosine(batch.video[0], muted_batch.video[0]),
            "audio_cosine_black_invariance": _cosine(batch.audio[0], black_batch.audio[0]),
            "video_max_abs_diff_muted": _max_abs_diff(batch.video[0], muted_batch.video[0]),
            "audio_max_abs_diff_black": _max_abs_diff(batch.audio[0], black_batch.audio[0]),
            "inert_tolerance": inert_tolerance,
            "seconds": round(ablation_s, 2),
        }
        evidence["ablation"] = ablation
        logger.info("消融余弦: %s", json.dumps(ablation, ensure_ascii=False))
        assert ablation["video_max_abs_diff_muted"] <= inert_tolerance, "只换音频时视频向量应不变"
        assert ablation["audio_max_abs_diff_black"] <= inert_tolerance, "只换画面时音频向量应不变"
        assert ablation["video_cosine_muted_invariance"] >= 0.9999
        assert ablation["audio_cosine_black_invariance"] >= 0.9999
        assert ablation["audio_cosine_muted"] < 0.99, "静音后音频向量应明显变化"
        assert ablation["video_cosine_black"] < 0.99, "纯黑后视频向量应明显变化"
        assert ablation["joint_cosine_muted"] < 0.99, "静音后联合向量应明显变化"
        assert ablation["joint_cosine_black"] < 0.99, "纯黑后联合向量应明显变化"

        evidence["ok"] = True
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
        return 0
    finally:
        embedder.close()
        if args.keep:
            logger.info("按 --keep 保留工作目录: %s", workdir)
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
