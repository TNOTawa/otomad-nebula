r"""v0.5-alpha 局部音画 token 数据集（全部来自已有落盘数据）。

按 docs/plans/v0.5-alpha-temporal-model.md 第 2 节实现输入层：

- PE-AV 窗口向量：data/embeddings/<sequence_version>/<work_id>/clip_{joint,audio,video}.npy
  (W,1024)，组成 paav (W,3,1024)；
- 窗口时间/权重：clip_times.npy (W,4 = start_s, duration_s, valid_duration_s, weight)；
- CLIP / DINOv2 帧向量：data/visual_dense/<frame_version>/<work_id>/dense_visual.npz 的
  per_frame (N,512/384) + manifest fps/duration；
- 音频事件：event_scan.npz 的 audio/av (100Hz,6)；视频事件：video (30fps,6)；
- 局部 token：每段固定 V05_BINS_PER_SEGMENT=16 个 0.5s bin，段数 W = PE-AV 窗口数
  （最多 V05_MAX_SEGMENTS=128，超长作品截断并记录）。

每个 bin 的构造（与设计文档一致）：

- 视觉：帧时间 i/fps 落在 [start+k*0.5, start+(k+1)*0.5) 的帧向量先 L2 再均值；
  无帧则零向量 + mask=0；视觉 mask 要求 CLIP 与 DINOv2 都有效（两个 token 流共用）；
- 音频：100Hz audio 6 维 + av 6 维在同一时间 bin 内的 mean/std -> 24 维，顺序为
  [audio_mean(6), audio_std(6), av_mean(6), av_std(6)]；
- 视觉事件：video 6 维在同一时间 bin 内的 mean/std -> 12 维，顺序为
  [video_mean(6), video_std(6)]；
- mask_visual 要求 CLIP 与 DINOv2 两个视觉 token 流在同一 bin 都有效（共用 mask）；
- 缺事件/缺视觉 bundle 时对应 token 全零 + mask=0，绝不静默伪造数据。

产物：data/v05_tokens/<dataset_version>/<work_id>.npz + manifest.json；
npz 特征为 float16、mask 为 uint8、segment_times 为 float32，duration_s/fps 为标量；
manifest 记录源版本、bin 参数、维度、逐作品段数与截断情况、run 统计。

本模块只读已有向量/事件/视觉 bundle，不加载 torch / PE-AV / CLIP / DINOv2，
不访问网络；所有写入都先写同目录临时文件再 os.replace（原子写）。
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import Config
from .db import Catalog
from .dense_visual import (
    DenseVisualError,
    compress_version,
    load_bundle,
    visual_version_slug,
)
from .errors import NebulaError
from .events import EVENT_VERSION, EventScanError, load_events_bundle
from .util import now_utc_iso, slugify

LOGGER = logging.getLogger(__name__)

V05_DATASET_VERSION = "v05_alpha_v1"
V05_BIN_SECONDS = 0.5
V05_BINS_PER_SEGMENT = 16
V05_MAX_SEGMENTS = 128

#: 固定 token 维度契约（设计文档第 2 节）
PAAV_DIM = 1024
VISUAL_CLIP_DIM = 512
VISUAL_DINO_DIM = 384
AUDIO_EVENT_DIM = 6
AV_EVENT_DIM = 6
VIDEO_EVENT_DIM = 6
AUDIO_TOKEN_DIM = 2 * AUDIO_EVENT_DIM + 2 * AV_EVENT_DIM  # audio mean/std + av mean/std
VISUAL_EVENT_TOKEN_DIM = 2 * VIDEO_EVENT_DIM

FEATURE_KEYS: tuple[str, ...] = (
    "paav",
    "visual_clip",
    "visual_dino",
    "audio",
    "visual_event",
)
MASK_KEYS: tuple[str, ...] = ("mask_visual", "mask_audio", "mask_visual_event")
REQUIRED_TOKEN_KEYS: tuple[str, ...] = (
    *FEATURE_KEYS,
    *MASK_KEYS,
    "segment_times",
    "segment_mask",
    "duration_s",
    "fps",
)
OPTIONAL_TOKEN_KEYS: tuple[str, ...] = ("n_segments_original", "truncated")

VISUAL_DENSE_DIRNAME = "visual_dense"
TOKENS_DIRNAME = "v05_tokens"
_MANIFEST_NAME = "manifest.json"
_EPS = 1e-12


class V05DataError(NebulaError):
    """v0.5-alpha 数据管道错误（源缺失、形状/版本不兼容、写入失败等）。"""


# --------------------------------------------------------------------------- 路径
def _safe_component(value: str, *, what: str) -> str:
    text = str(value or "").strip()
    if not text or text in (".", ".."):
        raise V05DataError(f"{what} 不能为空")
    if "/" in text or "\\" in text or ":" in text:
        raise V05DataError(f"{what} 包含非法路径字符: {text!r}")
    if slugify(text) != text:
        raise V05DataError(
            f"{what} 必须是文件系统安全标签（字母/数字/._-），收到: {text!r}"
        )
    return text


def dataset_dir(config: Config, version: str) -> Path:
    safe = _safe_component(version, what="dataset_version")
    return config.paths.data / TOKENS_DIRNAME / safe


def token_path(config: Config, version: str, work_id: int) -> Path:
    return dataset_dir(config, version) / f"{int(work_id):06d}.npz"


def dataset_manifest_path(config: Config, version: str) -> Path:
    return dataset_dir(config, version) / _MANIFEST_NAME


def load_dataset_manifest(config: Config, version: str) -> dict[str, Any] | None:
    path = dataset_manifest_path(config, version)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V05DataError(f"数据集 manifest 读取失败（{path}）：{exc}") from exc
    if not isinstance(payload, dict):
        raise V05DataError(f"数据集 manifest 顶层必须是对象: {path}")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + chr(10),
        encoding="utf-8",
    )
    os.replace(temp, path)


# --------------------------------------------------------------------------- 源解析
def resolve_default_frame_versions(
    config: Config,
) -> tuple[str | None, str | None]:
    """按 config.visual_dense.encoders 解析默认 (clip_frame_version, dino_frame_version)。"""

    clip_version: str | None = None
    dino_version: str | None = None
    for encoder in config.visual_dense.encoders:
        model = str(encoder.model).lower()
        if "clip" in model:
            kind = "clip"
        elif "dinov2" in model or "dino" in model:
            kind = "dino"
        else:
            continue
        raw = compress_version(
            f"{encoder.model}@{encoder.revision}",
            float(config.visual_dense.fps),
            int(config.visual_dense.size),
            str(config.visual_dense.preprocess),
        )
        version = visual_version_slug(raw)
        if kind == "clip" and clip_version is None:
            clip_version = version
        elif kind == "dino" and dino_version is None:
            dino_version = version
    return clip_version, dino_version


def _sequence_work_dir(config: Config, sequence_version: str, work_id: int) -> Path:
    return config.paths.embeddings / sequence_version / f"{int(work_id):06d}"


def _visual_work_dir(config: Config, frame_version: str, work_id: int) -> Path:
    return config.paths.visual_dense / frame_version / f"{int(work_id):06d}"


def _load_matrix(
    path: Path,
    *,
    what: str,
    columns: int | None = None,
) -> np.ndarray:
    if not path.is_file():
        raise V05DataError(f"缺少 {what}: {path}")
    try:
        array = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    except (OSError, ValueError, EOFError) as exc:
        raise V05DataError(f"读取 {what} 失败（{path}）：{exc}") from exc
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise V05DataError(f"{what} 必须是二维非空矩阵，收到 {array.shape}")
    if columns is not None and int(array.shape[1]) != int(columns):
        raise V05DataError(
            f"{what} 维度应为 {columns}，收到 {array.shape[1]}（{path}）"
        )
    if not np.all(np.isfinite(array)):
        raise V05DataError(f"{what} 包含 NaN/Inf: {path}")
    return np.ascontiguousarray(array, dtype=np.float32)


def _normalize_rows(matrix: np.ndarray, *, what: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    bad = (~np.isfinite(norms)) | (norms <= _EPS)
    if np.any(bad):
        row = int(np.flatnonzero(bad)[0])
        raise V05DataError(f"{what} 第 {row} 行范数为 0 或非有限")
    return (values / norms[:, None]).astype(np.float32)


def _segment_duration(config: Config, work_dir: Path, clip_times: np.ndarray) -> float:
    manifest_path = work_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        duration = payload.get("duration_s") if isinstance(payload, dict) else None
        if isinstance(duration, (int, float)) and not isinstance(duration, bool):
            value = float(duration)
            if math.isfinite(value) and value > 0:
                return value
    ends = np.asarray(clip_times[:, 0], dtype=np.float64) + np.asarray(
        clip_times[:, 1], dtype=np.float64
    )
    value = float(np.max(ends)) if ends.size else 0.0
    if not math.isfinite(value) or value <= 0:
        raise V05DataError(f"无法确定作品时长: {work_dir}")
    return value


def _load_visual_frames(
    config: Config,
    frame_version: str | None,
    work_id: int,
    *,
    expected_dim: int,
    what: str,
) -> tuple[np.ndarray | None, float | None]:
    if not frame_version:
        return None, None
    work_dir = _visual_work_dir(config, _safe_component(frame_version, what=what), work_id)
    if not work_dir.is_dir():
        LOGGER.warning("work_id=%s 缺少 %s 视觉 bundle 目录: %s", work_id, what, work_dir)
        return None, None
    try:
        bundle = load_bundle(work_dir)
    except DenseVisualError as exc:
        raise V05DataError(f"work_id={work_id} 的 {what} 视觉 bundle 损坏: {exc}") from exc
    if bundle is None:
        LOGGER.warning("work_id=%s 缺少 %s 视觉 bundle: %s", work_id, what, work_dir)
        return None, None
    features = bundle.features
    if int(features.dim) != int(expected_dim):
        raise V05DataError(
            f"work_id={work_id} 的 {what} 帧维度 {features.dim} != 期望 {expected_dim}"
        )
    per_frame = np.asarray(features.per_frame, dtype=np.float32)
    if per_frame.ndim != 2 or per_frame.shape[0] < 1 or not np.all(np.isfinite(per_frame)):
        raise V05DataError(f"work_id={work_id} 的 {what} per_frame 形状/数值非法")
    fps = float(features.fps)
    if not math.isfinite(fps) or fps <= 0:
        raise V05DataError(f"work_id={work_id} 的 {what} fps 非法: {fps}")
    return per_frame, fps


# --------------------------------------------------------------------------- bin 映射
def _bin_slices(times: np.ndarray, starts: np.ndarray) -> list[list[tuple[int, int]]]:
    """(W, bins) 的索引切片；第 s 段第 k 个 bin 的成员是 times 落在区间内的样本。"""

    bin_count = V05_BINS_PER_SEGMENT
    slices: list[list[tuple[int, int]]] = []
    for start_s in starts:
        row: list[tuple[int, int]] = []
        for bin_index in range(bin_count):
            low = float(start_s) + bin_index * V05_BIN_SECONDS
            high = low + V05_BIN_SECONDS
            left = int(np.searchsorted(times, low, side="left"))
            right = int(np.searchsorted(times, high, side="left"))
            row.append((left, right))
        slices.append(row)
    return slices


def _visual_tokens(
    per_frame: np.ndarray,
    fps: float,
    segment_times: np.ndarray,
    *,
    expected_dim: int,
    what: str,
) -> tuple[np.ndarray, np.ndarray]:
    """帧级向量 -> (W,16,D) token 与 (W,16) mask（帧时间 i/fps，bin 内 L2 后均值）。"""

    frames = _normalize_rows(per_frame, what=f"{what} per_frame")
    if int(frames.shape[1]) != int(expected_dim):
        raise V05DataError(f"{what} per_frame 维度 {frames.shape[1]} != {expected_dim}")
    frame_times = np.arange(frames.shape[0], dtype=np.float64) / float(fps)
    starts = np.asarray(segment_times[:, 0], dtype=np.float64)
    slices = _bin_slices(frame_times, starts)
    tokens = np.zeros(
        (len(slices), V05_BINS_PER_SEGMENT, expected_dim), dtype=np.float32
    )
    mask = np.zeros((len(slices), V05_BINS_PER_SEGMENT), dtype=np.uint8)
    for segment_index, row in enumerate(slices):
        for bin_index, (left, right) in enumerate(row):
            if right <= left:
                continue
            tokens[segment_index, bin_index] = frames[left:right].mean(axis=0)
            mask[segment_index, bin_index] = 1
    return tokens.astype(np.float16), mask


def _event_tokens(
    matrix: np.ndarray,
    rate: float,
    segment_times: np.ndarray,
    *,
    what: str,
) -> tuple[np.ndarray, np.ndarray]:
    """高频序列 -> (W,16,2C) mean/std token 与 mask（时间 = i/rate）。"""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1:
        raise V05DataError(f"{what} 必须是二维非空序列，收到 {values.shape}")
    if not math.isfinite(float(rate)) or rate <= 0:
        raise V05DataError(f"{what} 的采样率非法: {rate}")
    times = np.arange(values.shape[0], dtype=np.float64) / float(rate)
    starts = np.asarray(segment_times[:, 0], dtype=np.float64)
    slices = _bin_slices(times, starts)
    channels = int(values.shape[1])
    tokens = np.zeros(
        (len(slices), V05_BINS_PER_SEGMENT, 2 * channels), dtype=np.float32
    )
    mask = np.zeros((len(slices), V05_BINS_PER_SEGMENT), dtype=np.uint8)
    for segment_index, row in enumerate(slices):
        for bin_index, (left, right) in enumerate(row):
            if right <= left:
                continue
            chunk = values[left:right]
            tokens[segment_index, bin_index, :channels] = chunk.mean(axis=0)
            tokens[segment_index, bin_index, channels:] = chunk.std(axis=0)
            mask[segment_index, bin_index] = 1
    return tokens.astype(np.float16), mask


def _audio_tokens(
    audio: np.ndarray,
    av: np.ndarray,
    rate: float,
    segment_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    audio_values = np.asarray(audio, dtype=np.float64)
    av_values = np.asarray(av, dtype=np.float64)
    if audio_values.shape != av_values.shape:
        raise V05DataError(
            f"audio 与 av 序列形状不一致: {audio_values.shape} != {av_values.shape}"
        )
    audio_token, audio_mask = _event_tokens(audio_values, rate, segment_times, what="audio")
    av_token, av_mask = _event_tokens(av_values, rate, segment_times, what="av")
    combined = np.concatenate([audio_token.astype(np.float32), av_token.astype(np.float32)], axis=2)
    mask = (audio_mask & av_mask).astype(np.uint8)
    return combined.astype(np.float16), mask


# --------------------------------------------------------------------------- 单作品 token
def build_work_tokens(
    config: Config,
    work_id: int,
    *,
    sequence_version: str | None = None,
    clip_frame_version: str | None = None,
    dino_frame_version: str | None = None,
) -> dict[str, np.ndarray]:
    """构建一部作品的局部音画 token（不写盘）；缺数据用零向量 + mask=0。"""

    work_id = int(work_id)
    if work_id < 1:
        raise V05DataError(f"work_id 非法: {work_id}")
    sequence = _safe_component(
        sequence_version or config.embedding_version, what="sequence_version"
    )
    work_dir = _sequence_work_dir(config, sequence, work_id)

    joint = _load_matrix(
        work_dir / "clip_joint.npy", what="clip_joint", columns=PAAV_DIM
    )
    clip_times = _load_matrix(work_dir / "clip_times.npy", what="clip_times")
    if clip_times.shape[1] != 4:
        raise V05DataError(
            f"clip_times 必须是 (W,4)，收到 {clip_times.shape}（{work_dir}）"
        )
    if clip_times.shape[0] != joint.shape[0]:
        raise V05DataError(
            f"clip_times 行数 {clip_times.shape[0]} 与 clip_joint 行数 {joint.shape[0]} 不一致"
        )

    audio = video = None
    audio_path = work_dir / "clip_audio.npy"
    video_path = work_dir / "clip_video.npy"
    if audio_path.is_file():
        audio = _load_matrix(audio_path, what="clip_audio", columns=PAAV_DIM)
        if audio.shape[0] != joint.shape[0]:
            raise V05DataError("clip_audio 行数与 clip_joint 不一致")
    if video_path.is_file():
        video = _load_matrix(video_path, what="clip_video", columns=PAAV_DIM)
        if video.shape[0] != joint.shape[0]:
            raise V05DataError("clip_video 行数与 clip_joint 不一致")

    n_segments_original = int(joint.shape[0])
    n_segments = min(n_segments_original, V05_MAX_SEGMENTS)
    truncated = n_segments < n_segments_original
    if truncated:
        LOGGER.info(
            "work_id=%s 段数 %d > %d，截断到 %d 段",
            work_id,
            n_segments_original,
            V05_MAX_SEGMENTS,
            n_segments,
        )
    duration_s = _segment_duration(config, work_dir, clip_times)
    joint = joint[:n_segments]
    clip_times = clip_times[:n_segments]
    zeros = np.zeros((n_segments, PAAV_DIM), dtype=np.float32)
    paav = np.stack(
        [joint, audio[:n_segments] if audio is not None else zeros, video[:n_segments] if video is not None else zeros],
        axis=1,
    ).astype(np.float16)

    segment_times = np.asarray(clip_times, dtype=np.float32)
    segment_mask = (
        (segment_times[:, 1] > 0.0) & (segment_times[:, 2] > 0.0)
    ).astype(np.uint8)

    clip_version, dino_version = resolve_default_frame_versions(config)
    if clip_frame_version is None:
        clip_frame_version = clip_version
    if dino_frame_version is None:
        dino_frame_version = dino_version
    clip_frames, clip_fps = _load_visual_frames(
        config,
        clip_frame_version,
        work_id,
        expected_dim=VISUAL_CLIP_DIM,
        what="visual_clip",
    )
    dino_frames, dino_fps = _load_visual_frames(
        config,
        dino_frame_version,
        work_id,
        expected_dim=VISUAL_DINO_DIM,
        what="visual_dino",
    )
    if clip_fps is not None and dino_fps is not None and abs(clip_fps - dino_fps) > 1e-9:
        raise V05DataError(
            f"work_id={work_id} 的 CLIP fps={clip_fps} 与 DINOv2 fps={dino_fps} 不一致"
        )
    fps = clip_fps if clip_fps is not None else dino_fps
    if fps is None:
        fps = float(config.visual_dense.fps)
    if not math.isfinite(fps) or fps <= 0:
        raise V05DataError(f"fps 非法: {fps}")

    empty_visual = np.zeros(
        (n_segments, V05_BINS_PER_SEGMENT, VISUAL_CLIP_DIM), dtype=np.float16
    )
    if clip_frames is not None:
        visual_clip, mask_clip = _visual_tokens(
            clip_frames,
            float(clip_fps),
            segment_times,
            expected_dim=VISUAL_CLIP_DIM,
            what="visual_clip",
        )
    else:
        visual_clip = empty_visual
        mask_clip = np.zeros((n_segments, V05_BINS_PER_SEGMENT), dtype=np.uint8)
    empty_dino = np.zeros(
        (n_segments, V05_BINS_PER_SEGMENT, VISUAL_DINO_DIM), dtype=np.float16
    )
    if dino_frames is not None:
        visual_dino, mask_dino = _visual_tokens(
            dino_frames,
            float(dino_fps),
            segment_times,
            expected_dim=VISUAL_DINO_DIM,
            what="visual_dino",
        )
    else:
        visual_dino = empty_dino
        mask_dino = np.zeros((n_segments, V05_BINS_PER_SEGMENT), dtype=np.uint8)
    mask_visual = (mask_clip & mask_dino).astype(np.uint8)

    events = load_events_bundle(work_dir)
    if events is None:
        audio_token = np.zeros(
            (n_segments, V05_BINS_PER_SEGMENT, AUDIO_TOKEN_DIM), dtype=np.float16
        )
        visual_event = np.zeros(
            (n_segments, V05_BINS_PER_SEGMENT, VISUAL_EVENT_TOKEN_DIM), dtype=np.float16
        )
        mask_audio = np.zeros((n_segments, V05_BINS_PER_SEGMENT), dtype=np.uint8)
        mask_visual_event = np.zeros((n_segments, V05_BINS_PER_SEGMENT), dtype=np.uint8)
    else:
        if events.version != EVENT_VERSION:
            raise V05DataError(
                f"work_id={work_id} 事件版本 {events.version} 与当前 {EVENT_VERSION} 不一致"
            )
        audio_token, mask_audio = _audio_tokens(
            events.audio, events.av, float(events.audio_rate_hz), segment_times
        )
        visual_event, mask_visual_event = _event_tokens(
            events.video, float(events.video_fps_effective), segment_times, what="video"
        )

    tokens: dict[str, np.ndarray] = {
        "paav": paav,
        "visual_clip": visual_clip,
        "visual_dino": visual_dino,
        "audio": audio_token.astype(np.float16),
        "visual_event": visual_event.astype(np.float16),
        "mask_visual": mask_visual,
        "mask_audio": mask_audio,
        "mask_visual_event": mask_visual_event,
        "segment_times": segment_times,
        "segment_mask": segment_mask,
        "duration_s": np.asarray(duration_s, dtype=np.float64),
        "fps": np.asarray(float(fps), dtype=np.float64),
        "n_segments_original": np.asarray(n_segments_original, dtype=np.int64),
        "truncated": np.asarray(1 if truncated else 0, dtype=np.uint8),
    }
    return tokens


# --------------------------------------------------------------------------- 保存 / 读取
def _validated_tokens(tokens: Mapping[str, Any]) -> dict[str, np.ndarray]:
    missing = [key for key in REQUIRED_TOKEN_KEYS if key not in tokens]
    if missing:
        raise V05DataError(f"token 缺少必需数组: {', '.join(missing)}")
    paav = np.asarray(tokens["paav"], dtype=np.float16)
    if paav.ndim != 3 or paav.shape[1:] != (3, PAAV_DIM):
        raise V05DataError(f"paav 形状应为 (W,3,{PAAV_DIM})，收到 {paav.shape}")
    n_segments = int(paav.shape[0])
    if n_segments < 1 or n_segments > V05_MAX_SEGMENTS:
        raise V05DataError(f"段数非法: {n_segments}（1..{V05_MAX_SEGMENTS}）")
    expected_shapes = {
        "visual_clip": (n_segments, V05_BINS_PER_SEGMENT, VISUAL_CLIP_DIM),
        "visual_dino": (n_segments, V05_BINS_PER_SEGMENT, VISUAL_DINO_DIM),
        "audio": (n_segments, V05_BINS_PER_SEGMENT, AUDIO_TOKEN_DIM),
        "visual_event": (n_segments, V05_BINS_PER_SEGMENT, VISUAL_EVENT_TOKEN_DIM),
        "mask_visual": (n_segments, V05_BINS_PER_SEGMENT),
        "mask_audio": (n_segments, V05_BINS_PER_SEGMENT),
        "mask_visual_event": (n_segments, V05_BINS_PER_SEGMENT),
        "segment_times": (n_segments, 4),
        "segment_mask": (n_segments,),
    }
    for key, shape in expected_shapes.items():
        value = np.asarray(tokens[key])
        if value.shape != shape:
            raise V05DataError(f"{key} 形状应为 {shape}，收到 {value.shape}")
        if key.startswith("mask_"):
            if value.dtype != np.uint8:
                value = value.astype(np.uint8)
        elif key == "segment_times":
            value = value.astype(np.float32)
        else:
            value = value.astype(np.float16)
            if not np.all(np.isfinite(value)):
                raise V05DataError(f"{key} 包含 NaN/Inf")
        tokens = {**tokens, key: value}
    for key in ("duration_s", "fps"):
        value = np.asarray(tokens[key], dtype=np.float64)
        if value.ndim != 0 or not math.isfinite(float(value)) or float(value) <= 0:
            raise V05DataError(f"{key} 必须是正的有限标量，收到 {value!r}")
        tokens = {**tokens, key: value}
    n_original = np.asarray(
        tokens.get("n_segments_original", n_segments), dtype=np.int64
    )
    if n_original.ndim != 0 or int(n_original) < n_segments:
        raise V05DataError(f"n_segments_original 非法: {n_original!r}")
    truncated = np.asarray(tokens.get("truncated", int(n_original) > n_segments), dtype=np.uint8)
    tokens = {
        **tokens,
        "n_segments_original": n_original,
        "truncated": truncated.astype(np.uint8),
    }
    return tokens


def save_work_tokens(
    config: Config,
    version: str,
    work_id: int,
    tokens: Mapping[str, Any],
) -> Path:
    """原子保存一部作品的 token npz，返回路径。"""

    arrays = _validated_tokens(tokens)
    target = token_path(config, version, work_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = None
    temp_path: Path | None = None
    try:
        handle, name = tempfile.mkstemp(
            prefix=f".{target.stem}.", suffix=".tmp.npz", dir=str(target.parent)
        )
        temp_path = Path(name)
        with os.fdopen(handle, "wb") as stream:
            np.savez(stream, **arrays)
        os.replace(temp_path, target)
    except (OSError, ValueError) as exc:
        raise V05DataError(f"写入 token npz 失败（{target}）：{exc}") from exc
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:  # pragma: no cover - 清理失败不改变主结果
                LOGGER.warning("无法删除 token 临时文件: %s", temp_path)
    return target


def load_work_tokens(
    config: Config, version: str, work_id: int
) -> dict[str, np.ndarray] | None:
    """读取一部作品的 token npz；文件缺失返回 None，损坏抛 V05DataError。"""

    path = token_path(config, version, work_id)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            arrays = {key: np.asarray(payload[key]) for key in payload.files}
    except (OSError, ValueError, EOFError) as exc:
        raise V05DataError(f"读取 token npz 失败（{path}）：{exc}") from exc
    missing = [key for key in REQUIRED_TOKEN_KEYS if key not in arrays]
    if missing:
        raise V05DataError(
            f"token npz 缺少必需数组（{path}）: {', '.join(missing)}"
        )
    return arrays


# --------------------------------------------------------------------------- 数据集构建
def _dataset_versions(
    config: Config,
    *,
    sequence_version: str | None,
    clip_frame_version: str | None,
    dino_frame_version: str | None,
) -> tuple[str, str | None, str | None]:
    sequence = _safe_component(
        sequence_version or config.embedding_version, what="sequence_version"
    )
    default_clip, default_dino = resolve_default_frame_versions(config)
    clip_version = (
        _safe_component(clip_frame_version, what="clip_frame_version")
        if clip_frame_version
        else default_clip
    )
    dino_version = (
        _safe_component(dino_frame_version, what="dino_frame_version")
        if dino_frame_version
        else default_dino
    )
    return sequence, clip_version, dino_version


def _missing_source_reasons(
    config: Config, sequence_version: str, work_id: int
) -> list[str]:
    work_dir = _sequence_work_dir(config, sequence_version, work_id)
    reasons: list[str] = []
    if not (work_dir / "clip_joint.npy").is_file():
        reasons.append("缺少 clip_joint.npy")
    if not (work_dir / "clip_times.npy").is_file():
        reasons.append("缺少 clip_times.npy")
    return reasons


def _candidate_work_ids(
    catalog: Catalog,
    config: Config,
    *,
    sequence_version: str,
    work_ids: Sequence[int] | None,
) -> tuple[list[int], list[dict[str, Any]]]:
    """返回 (候选 work_id 升序, 显式 --work-ids 的问题列表)。"""

    duplicates = catalog.get_duplicate_work_ids()
    problems: list[dict[str, Any]] = []
    if work_ids is not None:
        seen: set[int] = set()
        candidates: list[int] = []
        for raw in work_ids:
            try:
                work_id = int(raw)
            except (TypeError, ValueError):
                problems.append({"work_id": raw, "reasons": ["work_id 非法"]})
                continue
            if work_id in seen:
                continue
            seen.add(work_id)
            work = catalog.get_work(work_id)
            if work is None:
                problems.append({"work_id": work_id, "reasons": ["作品不存在"]})
                continue
            if int(work_id) in duplicates:
                problems.append({"work_id": work_id, "reasons": ["work 是 duplicate"]})
                continue
            if str(work.get("status")) != "completed":
                problems.append(
                    {
                        "work_id": work_id,
                        "reasons": [f"状态不是 completed: {work.get('status')}"],
                    }
                )
                continue
            candidates.append(work_id)
        return sorted(candidates), problems
    rows = catalog.list_works_with_embedding(sequence_version, kind="clip_joint")
    candidates = [
        int(row["id"])
        for row in rows
        if str(row.get("status")) == "completed" and int(row["id"]) not in duplicates
    ]
    return sorted(candidates), problems


def build_dataset(
    catalog: Catalog,
    config: Config,
    *,
    version: str = V05_DATASET_VERSION,
    sequence_version: str | None = None,
    clip_frame_version: str | None = None,
    dino_frame_version: str | None = None,
    force: bool = False,
    limit: int | None = None,
    work_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """只处理 canonical + completed 作品，构建 v0.5-alpha token 数据集并汇总结果。"""

    started = now_utc_iso()
    started_monotonic = time.perf_counter()
    dataset_version = _safe_component(version, what="dataset_version")
    sequence, clip_version, dino_version = _dataset_versions(
        config,
        sequence_version=sequence_version,
        clip_frame_version=clip_frame_version,
        dino_frame_version=dino_frame_version,
    )
    if catalog.get_embedding_version(sequence) is None:
        raise V05DataError(
            f"序列版本未登记: {sequence}（先运行 pipeline.py 或显式传 --sequence-version）"
        )
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise V05DataError(f"limit 必须是 >= 1 的整数，收到 {limit!r}")

    candidates, problems = _candidate_work_ids(
        catalog, config, sequence_version=sequence, work_ids=work_ids
    )
    if limit is not None:
        candidates = candidates[:limit]

    existing_manifest = None if force else load_dataset_manifest(config, dataset_version)
    works_record: dict[int, dict[str, Any]] = {}
    if isinstance(existing_manifest, dict):
        for item in existing_manifest.get("works", ()):  # type: ignore[union-attr]
            if isinstance(item, dict) and item.get("work_id") is not None:
                works_record[int(item["work_id"])] = dict(item)

    missing: list[dict[str, Any]] = list(problems)
    failures: list[dict[str, Any]] = []
    existing: list[int] = []
    built: list[int] = []
    truncated: list[int] = []
    for work_id in candidates:
        target = token_path(config, dataset_version, work_id)
        if target.is_file() and not force:
            existing.append(work_id)
            continue
        reasons = _missing_source_reasons(config, sequence, work_id)
        if reasons:
            missing.append({"work_id": work_id, "reasons": reasons})
            continue
        try:
            tokens = build_work_tokens(
                config,
                work_id,
                sequence_version=sequence,
                clip_frame_version=clip_version,
                dino_frame_version=dino_version,
            )
            save_work_tokens(config, dataset_version, work_id, tokens)
            n_segments = int(np.asarray(tokens["paav"]).shape[0])
            n_original = int(np.asarray(tokens["n_segments_original"]).item())
            is_truncated = bool(int(np.asarray(tokens["truncated"]).item()))
            if is_truncated:
                truncated.append(work_id)
            works_record[work_id] = {
                "work_id": int(work_id),
                "path": str(token_path(config, dataset_version, work_id)),
                "n_segments": n_segments,
                "n_segments_original": n_original,
                "truncated": is_truncated,
                "has_visual": bool(np.asarray(tokens["mask_visual"]).any()),
                "has_events": bool(
                    np.asarray(tokens["mask_audio"]).any()
                    or np.asarray(tokens["mask_visual_event"]).any()
                ),
            }
            built.append(work_id)
            LOGGER.info(
                "work_id=%s token 完成：%d 段（原始 %d，截断=%s）",
                work_id,
                n_segments,
                n_original,
                is_truncated,
            )
        except (V05DataError, DenseVisualError, EventScanError, OSError, ValueError) as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append({"work_id": work_id, "error": message})
            LOGGER.error("work_id=%s token 构建失败: %s", work_id, message)

    manifest_payload = {
        "dataset_version": dataset_version,
        "created_at": now_utc_iso(),
        "run_started_at": started,
        "sequence_version": sequence,
        "clip_frame_version": clip_version,
        "dino_frame_version": dino_version,
        "event_version": EVENT_VERSION,
        "bin_seconds": V05_BIN_SECONDS,
        "bins_per_segment": V05_BINS_PER_SEGMENT,
        "max_segments": V05_MAX_SEGMENTS,
        "dims": {
            "paav": [3, PAAV_DIM],
            "visual_clip": [V05_BINS_PER_SEGMENT, VISUAL_CLIP_DIM],
            "visual_dino": [V05_BINS_PER_SEGMENT, VISUAL_DINO_DIM],
            "audio": [V05_BINS_PER_SEGMENT, AUDIO_TOKEN_DIM],
            "visual_event": [V05_BINS_PER_SEGMENT, VISUAL_EVENT_TOKEN_DIM],
            "segment_times": [4],
        },
        "dtypes": {
            "features": "float16",
            "masks": "uint8",
            "segment_times": "float32",
            "scalars": "float64",
        },
        "token_layout": {
            "paav": ["joint", "audio", "video"],
            "audio": [
                "audio_mean_6",
                "audio_std_6",
                "av_mean_6",
                "av_std_6",
            ],
            "visual_event": ["video_mean_6", "video_std_6"],
            "visual": ["clip_512", "dino_384"],
        },
        "counts": {
            "n_candidates": len(candidates),
            "n_built": len(built),
            "n_existing": len(existing),
            "n_missing": len(missing),
            "n_failed": len(failures),
            "n_truncated": len(truncated),
        },
        "works": [works_record[work_id] for work_id in sorted(works_record)],
        "missing": missing,
        "failures": failures,
        "truncated_work_ids": truncated,
    }
    manifest_file = dataset_manifest_path(config, dataset_version)
    _atomic_write_json(manifest_file, manifest_payload)
    return {
        "dataset_version": dataset_version,
        "dataset_dir": str(dataset_dir(config, dataset_version)),
        "manifest_path": str(manifest_file),
        "sequence_version": sequence,
        "clip_frame_version": clip_version,
        "dino_frame_version": dino_version,
        "n_candidates": len(candidates),
        "n_built": len(built),
        "n_existing": len(existing),
        "n_missing": len(missing),
        "n_failed": len(failures),
        "n_truncated": len(truncated),
        "n_works_in_manifest": len(works_record),
        "missing": missing,
        "failures": failures,
        "truncated_work_ids": truncated,
        "elapsed_s": round(time.perf_counter() - started_monotonic, 3),
        "built_work_ids": built,
        "existing_work_ids": existing,
    }
