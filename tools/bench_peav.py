"""PE-AV GPU / CPU 推理基准（T1 证据脚本）。

用法（项目根目录）：

    .\\.venv\\Scripts\\python.exe tools/bench_peav.py

覆盖 Lead 关心的生产参数：
- 40 秒测试视频按 plan_clips 切 5 段 8 秒片段（与正式 P0 参数一致）；
- 模型加载耗时；
- 单片段预处理 / 前向耗时；
- batch_size = 1/2/3/4/5 编码 5 段的耗时、每片段耗时、峰值显存；
- batch_size=5 是否 OOM，以及最大安全 batch；
- 同一 batch 重复编码的最大绝对差；
- joint / audio / video 形状、逐行范数、是否含 NaN/Inf；
- CUDA bfloat16 autocast 与 float32 输出的余弦 / 最大绝对差；
- 8 秒片段的静音 / 纯黑消融余弦；
- 按 33 部作品 × 5 片段外推流水线推理耗时。

stdout 只输出 JSON 证据（可重定向到文件），日志走 stderr。
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

from nebula.config import load_config  # noqa: E402
from nebula.embed import PEAVEmbedder  # noqa: E402
from nebula.media import (  # noqa: E402
    black_video_clip,
    extract_clip,
    mute_audio_clip,
    probe_media,
)
from nebula.sampling import plan_clips  # noqa: E402

logger = logging.getLogger("bench_peav")


def _make_sample_video(target: Path, *, duration_s: int = 40) -> Path:
    """生成 duration_s 秒、320x240@25fps + 48k 正弦的测试视频。"""

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
            f"testsrc2=size=320x240:rate=25:duration={duration_s}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=48000:duration={duration_s}",
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


def _gpu_mem(torch) -> dict[str, float]:
    """返回当前/峰值显存（MB）；CPU 设备返回空字典。"""

    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1e6, 1),
    }


def _reset_peak(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _check_batch(batch, *, expected: int, where: str) -> dict[str, object]:
    """检查形状 / 范数 / NaN，返回可序列化的摘要。"""

    assert batch.joint.shape[0] == expected, (where, batch.joint.shape)
    assert batch.audio is not None and batch.video is not None, where
    assert batch.audio.shape == batch.joint.shape == batch.video.shape, (
        where,
        batch.joint.shape,
        batch.audio.shape,
        batch.video.shape,
    )
    summary: dict[str, object] = {
        "shape_joint": list(batch.joint.shape),
        "shape_audio": list(batch.audio.shape),
        "shape_video": list(batch.video.shape),
        "dtype": str(batch.joint.dtype),
    }
    for name, array in (
        ("joint", batch.joint),
        ("audio", batch.audio),
        ("video", batch.video),
    ):
        assert np.all(np.isfinite(array)), f"{where} {name} 含 NaN/Inf"
        norms = np.linalg.norm(array, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), (where, name, norms)
        summary[f"norm_{name}"] = [round(float(value), 7) for value in norms]
        summary[f"finite_{name}"] = True
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PE-AV GPU/CPU 推理基准")
    parser.add_argument("--clip-seconds", type=float, default=8.0, help="片段秒数（默认 8）")
    parser.add_argument("--clips", type=int, default=5, help="每部作品片段数（默认 5）")
    parser.add_argument("--keep", action="store_true", help="保留临时工作目录")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(REPO_ROOT / "config.yaml")
    workdir = Path(tempfile.mkdtemp(prefix="nebula_peav_bench_", dir=str(config.paths.temp)))
    logger.info("工作目录: %s", workdir)
    video_seconds = int(max(40, args.clip_seconds * args.clips + 8))
    evidence: dict[str, object] = {
        "checkpoint": config.model.checkpoint,
        "configured_batch_size": config.model.batch_size,
        "configured_device": config.model.device,
        "configured_dtype": config.model.dtype,
        "clip_seconds": args.clip_seconds,
        "clips": args.clips,
    }
    embedder = PEAVEmbedder(config.model)
    try:
        try:
            torch = embedder._torch  # type: ignore[attr-defined]
            started = time.time()
            embedder._ensure_loaded()  # type: ignore[attr-defined]
            load_s = time.time() - started
            torch = embedder._torch  # type: ignore[attr-defined]
            evidence["runtime"] = {
                "device": embedder._device,  # type: ignore[attr-defined]
                "autocast_bfloat16": embedder._use_autocast,  # type: ignore[attr-defined]
                "load_seconds": round(load_s, 2),
                "gpu_name": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            }
            logger.info("模型加载完成: %s", json.dumps(evidence["runtime"], ensure_ascii=False))

            sample = _make_sample_video(workdir / "sample.mp4", duration_s=video_seconds)
            info = probe_media(sample, ffprobe=config.media.ffprobe)
            plans = plan_clips(
                info.duration_s, clip_seconds=args.clip_seconds, clips_per_work=args.clips
            )
            assert len(plans) == args.clips, (len(plans), args.clips)
            started = time.time()
            clips = [
                extract_clip(sample, plan, workdir / "clips", media=config.media)
                for plan in plans
            ]
            evidence["extraction_seconds"] = round(time.time() - started, 2)
            pairs = [(clip.video_path, clip.audio_path) for clip in clips]
            logger.info("已切出 %s 段 %.1fs 片段", len(clips), args.clip_seconds)

            # ---- 单片段：预处理 / 前向拆分 ----
            processor = embedder._processor  # type: ignore[attr-defined]
            model = embedder._model  # type: ignore[attr-defined]
            device = embedder._device  # type: ignore[attr-defined]
            use_autocast = embedder._use_autocast  # type: ignore[attr-defined]
            video_path, audio_path = pairs[0]
            started = time.time()
            inputs = processor(
                videos=[str(video_path)],
                audio=[str(audio_path)],
                sampling_rate=48000,
                return_tensors="pt",
                padding=True,
            )
            preprocess_s = time.time() - started
            model_inputs = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in dict(inputs).items()
            }
            _reset_peak(torch)
            started = time.time()
            with torch.inference_mode():
                if use_autocast:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = model(**model_inputs)
                else:
                    outputs = model(**model_inputs)
            single_forward_s = time.time() - started
            single_mem = _gpu_mem(torch)
            single_norm = float(outputs.audio_video_embeds.float().norm(dim=-1)[0])
            evidence["single_clip"] = {
                "preprocess_seconds": round(preprocess_s, 2),
                "forward_seconds": round(single_forward_s, 2),
                "joint_unnormalized_norm": round(single_norm, 4),
                "gpu_mem_after_forward": single_mem,
            }
            logger.info("单片段: 预处理 %.2fs 前向 %.2fs 显存 %s", preprocess_s, single_forward_s, single_mem)

            # ---- 不同 batch_size 编码 5 段 ----
            batch_results: dict[str, object] = {}
            max_safe_batch = 0
            for batch_size in (1, 2, 3, 4, 5):
                embedder._batch_size = batch_size  # type: ignore[attr-defined]
                _reset_peak(torch)
                started = time.time()
                try:
                    batch = embedder.encode_pairs(pairs)
                    elapsed = time.time() - started
                    summary = _check_batch(batch, expected=len(pairs), where=f"batch_size={batch_size}")
                    summary["seconds_total"] = round(elapsed, 2)
                    summary["seconds_per_clip"] = round(elapsed / len(pairs), 2)
                    summary["gpu_mem"] = _gpu_mem(torch)
                    batch_results[f"batch_{batch_size}"] = summary
                    max_safe_batch = max(max_safe_batch, batch_size)
                    logger.info(
                        "batch_size=%s: %.2fs（%.2fs/片段）显存 %s",
                        batch_size,
                        elapsed,
                        elapsed / len(pairs),
                        summary["gpu_mem"],
                    )
                except RuntimeError as exc:
                    if not _is_oom(exc):
                        raise
                    batch_results[f"batch_{batch_size}"] = {
                        "oom": True,
                        "error": str(exc).splitlines()[0][:300],
                    }
                    logger.warning("batch_size=%s OOM: %s", batch_size, str(exc).splitlines()[0])
                    _reset_peak(torch)
            evidence["batch_sizes"] = batch_results
            evidence["max_safe_batch"] = max_safe_batch
            assert max_safe_batch >= 1, "batch_size=1 也 OOM，GPU 无法使用"

            # ---- 生产 batch 下重复一致性 ----
            production = min(config.model.batch_size, max_safe_batch)
            embedder._batch_size = production  # type: ignore[attr-defined]
            first = embedder.encode_pairs(pairs)
            second = embedder.encode_pairs(pairs)
            repeat_diff = _max_abs_diff(first.joint, second.joint)
            repeat_cosine_min = min(
                _cosine(first.joint[row], second.joint[row]) for row in range(first.joint.shape[0])
            )
            evidence["repeat"] = {
                "batch_size": production,
                "joint_max_abs_diff": repeat_diff,
                "joint_min_cosine": repeat_cosine_min,
            }
            logger.info("重复一致性: 最大绝对差 %.3e 最小余弦 %.8f", repeat_diff, repeat_cosine_min)

            # ---- bf16 autocast vs float32（同设备） ----
            if use_autocast:
                embedder._use_autocast = False  # type: ignore[attr-defined]
                try:
                    fp32 = embedder.encode_pairs([pairs[0]])
                finally:
                    embedder._use_autocast = use_autocast  # type: ignore[attr-defined]
                bf16 = embedder.encode_pairs([pairs[0]])
                evidence["dtype_compare"] = {
                    "joint_cosine_bf16_vs_fp32": _cosine(fp32.joint[0], bf16.joint[0]),
                    "joint_max_abs_diff": _max_abs_diff(fp32.joint[0], bf16.joint[0]),
                    "fp32_finite": bool(np.all(np.isfinite(fp32.joint))),
                    "bf16_finite": bool(np.all(np.isfinite(bf16.joint))),
                }
                logger.info("dtype 对比: %s", json.dumps(evidence["dtype_compare"], ensure_ascii=False))

            # ---- 8 秒片段消融 ----
            clip = clips[0]
            muted = mute_audio_clip(clip.duration_s, workdir / "muted.wav", media=config.media)
            black = black_video_clip(clip.duration_s, workdir / "black.mp4", media=config.media)
            started = time.time()
            muted_batch = embedder.encode_pairs([(clip.video_path, muted)])
            black_batch = embedder.encode_pairs([(black, clip.audio_path)])
            evidence["ablation"] = {
                "joint_cosine_muted": _cosine(
                    first.joint[0], muted_batch.joint[0]
                ),
                "joint_cosine_black": _cosine(first.joint[0], black_batch.joint[0]),
                "audio_cosine_muted": _cosine(first.audio[0], muted_batch.audio[0]),
                "video_cosine_black": _cosine(first.video[0], black_batch.video[0]),
                "video_max_abs_diff_muted": _max_abs_diff(
                    first.video[0], muted_batch.video[0]
                ),
                "audio_max_abs_diff_black": _max_abs_diff(
                    first.audio[0], black_batch.audio[0]
                ),
                "seconds": round(time.time() - started, 2),
            }
            assert evidence["ablation"]["joint_cosine_muted"] < 0.99
            assert evidence["ablation"]["joint_cosine_black"] < 0.99
            logger.info("消融: %s", json.dumps(evidence["ablation"], ensure_ascii=False))

            # ---- 流水线外推（33 部作品 × 5 片段） ----
            per_work_s = float(batch_results[f"batch_{production}"]["seconds_total"])  # type: ignore[index]
            evidence["projection"] = {
                "per_work_seconds_at_batch": round(per_work_s, 2),
                "33_works_minutes": round(33 * per_work_s / 60.0, 1),
            }
        except Exception as exc:  # noqa: BLE001 - 基准脚本要保留已收集的部分证据
            evidence["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            evidence["ok"] = "error" not in evidence
            print(json.dumps(evidence, ensure_ascii=False, indent=2))
            sys.stdout.flush()
        return 0
    finally:
        embedder.close()
        if args.keep:
            logger.info("按 --keep 保留工作目录: %s", workdir)
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
