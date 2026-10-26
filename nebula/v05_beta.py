"""v0.5-beta：局部音画配对数据 + 双塔对比模型 + 严格验证（研究产物）。

与 v0.5-alpha 的区别：
- 不再训练 4.3M 三头 V05AlphaModel；最小双塔 MLP（视觉/音频各 -> 128d L2）+
  单一局部 InfoNCE / 多候选匹配目标；
- 视觉 bin 来自全量 token（v05_data 的 visual_clip/visual_dino + mask_visual，
  max_segments=None，不截断）；音频 bin 来自 t1 的 data/audio_dense/<audio_version>/
  <work_id>/audio.npz（bin_features/bin_mask/bin_times；本模块有适配层，contract 变化
  时只需改 load_audio_bins）；
- 配对构造固定且可复现：正样本 (A_t,V_t)，负样本同作品 t+Δ（Δ=1,2,4,8,16 bins）；
  视觉/音频近重复（余弦 > near_dup）或时间位置歧义则跳过；候选顺序随机打乱，
  不使用作品 ID/位置/外部元数据作为特征。
- 按 canonical work_id 确定性划分 train/val/test（默认 ffoor(0.7N)/floor(0.2N)/剩余，
  23 部 -> 16/4/3；110 部 -> 77/22/11，可用 --split-counts 覆盖并记录），
  train/val/test 不重叠，排除 work_links duplicate。
- 只用 train 拟合标准化；val 选模型；test 只评一次；≥3 seeds；按作品重采样给
  bootstrap 95% 区间；报告随机 / 冻结投影 / 平凡时间同步基线、时间偏移曲线、
  消融（音频置零/视觉置零/音频替换）、坍缩与数值稳定性。
- 产物：data/models/<version>_<seed>/checkpoint.pt、logs/<version>_<seed>_metrics.json、
  logs/<version>_summary.json、logs/<version>_candidates_manifest.json；不注册 DB、
  不建 map、不写 data/representations。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .config import Config
from .db import Catalog
from .errors import NebulaError
from .util import now_utc_iso

LOGGER = logging.getLogger(__name__)

BETA_MODEL_VERSION = "v05_beta"
BETA_DATASET_VERSION = "v05_beta_v1"
AUDIO_DENSE_DIRNAME = "audio_dense"
BETA_BIN_SECONDS = 0.5
DEFAULT_OFFSETS = (1, 2, 4, 8, 16)
DEFAULT_NEAR_DUP_COS = 0.95
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_HIDDEN = 128
DEFAULT_EMBED_DIM = 128
DEFAULT_TEMPERATURE = 0.07
DEFAULT_PATIENCE = 5
DEFAULT_BOOTSTRAP = 200
DEFAULT_SPLIT_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
SMOKE_WORKS = 12
SMOKE_BINS = 40
SMOKE_VISUAL_DIM = 64
SMOKE_AUDIO_DIM = 32
AUDIO_FEATURE_KEYS = ("bin_features", "features", "bins")
AUDIO_MASK_KEYS = ("bin_mask", "mask")
AUDIO_TIME_KEYS = ("bin_times", "times", "bin_time")
_ERROR_LIMIT = 2000


class V05BetaError(NebulaError):
    """v0.5-beta 数据/训练的可预期错误。"""


# ---------------------------------------------------------------- 配置 / 划分


@dataclass(frozen=True)
class BetaConfig:
    version: str = BETA_MODEL_VERSION
    dataset_version: str = BETA_DATASET_VERSION
    audio_version: str | None = None
    visual_encoder: str = "clip"
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    hidden: int = DEFAULT_HIDDEN
    embed_dim: int = DEFAULT_EMBED_DIM
    temperature: float = DEFAULT_TEMPERATURE
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    split_seed: int = DEFAULT_SPLIT_SEED
    split_counts: tuple[int, int, int] | None = None
    offsets: tuple[int, ...] = DEFAULT_OFFSETS
    near_dup: float = DEFAULT_NEAR_DUP_COS
    bootstrap: int = DEFAULT_BOOTSTRAP
    patience: int = DEFAULT_PATIENCE
    device: str = "auto"
    limit: int | None = None
    exclude_work_ids: tuple[int, ...] = ()
    max_segments: int | None = None
    smoke: bool = False
    smoke_works: int = SMOKE_WORKS
    smoke_bins: int = SMOKE_BINS
    smoke_visual_dim: int = SMOKE_VISUAL_DIM
    smoke_audio_dim: int = SMOKE_AUDIO_DIM

    def validate(self) -> "BetaConfig":
        if not str(self.version).strip():
            raise V05BetaError("--version 不能为空")
        if not str(self.dataset_version).strip():
            raise V05BetaError("--dataset-version 不能为空")
        if self.visual_encoder not in ("clip", "dino"):
            raise V05BetaError(
                f"visual_encoder 必须是 clip 或 dino，收到 {self.visual_encoder!r}"
            )
        if int(self.epochs) < 1:
            raise V05BetaError(f"epochs 必须 >= 1，收到 {self.epochs}")
        if int(self.batch_size) < 1:
            raise V05BetaError(f"batch_size 必须 >= 1，收到 {self.batch_size}")
        if float(self.lr) <= 0 or float(self.weight_decay) < 0:
            raise V05BetaError("lr 必须 > 0 且 weight_decay >= 0")
        if int(self.hidden) < 1 or int(self.embed_dim) < 1:
            raise V05BetaError("hidden/embed_dim 必须 >= 1")
        if not 0.0 < float(self.temperature) <= 1.0:
            raise V05BetaError(f"temperature 必须在 (0,1]，收到 {self.temperature}")
        if not self.seeds:
            raise V05BetaError("至少需要一个训练 seed")
        if len(set(int(value) for value in self.seeds)) != len(self.seeds):
            raise V05BetaError("seeds 不能重复")
        offsets = tuple(int(value) for value in self.offsets)
        if not offsets or any(value < 1 for value in offsets):
            raise V05BetaError("offsets 必须是 >= 1 的整数")
        if len(set(offsets)) != len(offsets):
            raise V05BetaError("offsets 不能重复")
        if not 0.0 <= float(self.near_dup) <= 1.0:
            raise V05BetaError(f"near_dup 必须在 [0,1]，收到 {self.near_dup}")
        if int(self.bootstrap) < 0:
            raise V05BetaError("bootstrap 必须 >= 0")
        if int(self.patience) < 0:
            raise V05BetaError("patience 必须 >= 0")
        if self.device not in ("auto", "cpu", "cuda"):
            raise V05BetaError(f"device 必须是 auto/cpu/cuda，收到 {self.device!r}")
        if self.limit is not None and int(self.limit) < 1:
            raise V05BetaError(f"limit 必须 >= 1，收到 {self.limit}")
        excluded = tuple(int(value) for value in self.exclude_work_ids)
        if any(value < 1 for value in excluded) or len(set(excluded)) != len(excluded):
            raise V05BetaError("exclude_work_ids 必须是互不重复的正整数")
        if self.split_counts is not None:
            counts = tuple(int(value) for value in self.split_counts)
            if len(counts) != 3 or any(value < 1 for value in counts):
                raise V05BetaError("split_counts 必须是 3 个 >= 1 的整数")
        if self.max_segments is not None and int(self.max_segments) < 1:
            raise V05BetaError("max_segments 必须为 None 或 >= 1")
        return self

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["seeds"] = [int(value) for value in self.seeds]
        payload["offsets"] = [int(value) for value in self.offsets]
        payload["split_counts"] = (
            None if self.split_counts is None else [int(v) for v in self.split_counts]
        )
        return payload


@dataclass(frozen=True)
class BetaSplit:
    train_ids: tuple[int, ...]
    val_ids: tuple[int, ...]
    test_ids: tuple[int, ...]
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "train_work_ids": [int(value) for value in self.train_ids],
            "val_work_ids": [int(value) for value in self.val_ids],
            "test_work_ids": [int(value) for value in self.test_ids],
            "counts": [len(self.train_ids), len(self.val_ids), len(self.test_ids)],
        }


def _hash_unit(work_id: int, seed: int) -> float:
    digest = hashlib.sha256(
        f"v05beta-split:{int(seed)}:{int(work_id)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def split_work_ids(
    work_ids: Sequence[int],
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    split_counts: Sequence[int] | None = None,
) -> BetaSplit:
    """按 work_id hash 确定性划分 train/val/test（不重叠）。

    默认分配：train=floor(0.7N)，val=floor(0.2N)，test=剩余（23 部 -> 16/4/3；
    110 部 -> 77/22/11）。可用 split_counts 显式指定（如 70/20/20 部）。
    至少保证 val/test 各 1 部（N>=3 时）。
    """

    ids = sorted({int(value) for value in work_ids})
    total = len(ids)
    if total < 3:
        raise V05BetaError(f"至少需要 3 部作品才能划分 train/val/test，收到 {total}")
    if split_counts is not None:
        train_count, val_count, test_count = (int(value) for value in split_counts)
        if train_count < 1 or val_count < 1 or test_count < 1:
            raise V05BetaError("split_counts 每项都必须 >= 1")
        if train_count + val_count + test_count != total:
            raise V05BetaError(
                f"split_counts 之和不等于作品数 {total}: "
                f"{train_count}+{val_count}+{test_count}"
            )
    else:
        train_count = max(1, int(total * TRAIN_RATIO))
        val_count = max(1, int(total * VAL_RATIO))
        test_count = total - train_count - val_count
        if test_count < 1:
            train_count = max(1, train_count - (1 - test_count))
            test_count = 1
    ranked = sorted(ids, key=lambda work_id: (_hash_unit(work_id, seed), work_id))
    train_ids = tuple(sorted(ranked[:train_count]))
    val_ids = tuple(sorted(ranked[train_count : train_count + val_count]))
    test_ids = tuple(sorted(ranked[train_count + val_count :]))
    return BetaSplit(
        train_ids=train_ids, val_ids=val_ids, test_ids=test_ids, seed=int(seed)
    )


# ---------------------------------------------------------------- 音频适配层（t1 audio_dense）


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def resolve_audio_version(
    config: Config, requested: str | None = None
) -> tuple[str, Path, dict[str, Any] | None]:
    """解析 data/audio_dense 版本目录；返回 (version, dir, version_manifest)。"""

    root = config.data_path(AUDIO_DENSE_DIRNAME)
    if not root.is_dir():
        raise V05BetaError(
            f"音频目录不存在: {root}；请先运行 t1 的 tools/audio_dense.py 生成"
        )
    if requested:
        version = str(requested).strip()
        version_dir = root / version
        if not version_dir.is_dir():
            raise V05BetaError(
                f"音频版本不存在: {version_dir}（可用版本: "
                f"{', '.join(sorted(p.name for p in root.iterdir() if p.is_dir())) or '无'}）"
            )
    else:
        versions = sorted(p.name for p in root.iterdir() if p.is_dir())
        if len(versions) != 1:
            raise V05BetaError(
                "无法自动确定 audio_version，请用 --audio-version 指定；"
                f"data/audio_dense 下版本: {versions or '无'}"
            )
        version = versions[0]
        version_dir = root / version
    manifest = _load_json(version_dir / "manifest.json")
    if manifest is None:
        manifest = _load_json(root / f"{version}_manifest.json")
    return version, version_dir, manifest


def _audio_file_candidates(version_dir: Path, work_id: int) -> list[Path]:
    wid = f"{int(work_id):06d}"
    return [
        version_dir / wid / "audio.npz",
        version_dir / wid / "features.npz",
        version_dir / f"{wid}.npz",
        version_dir / f"{wid}_audio.npz",
    ]


def _first_array(payload: Mapping[str, Any], keys: Sequence[str]) -> np.ndarray | None:
    for key in keys:
        if key in payload:
            return np.asarray(payload[key])
    return None


def _audio_hop_from_manifest(manifest: Mapping[str, Any] | None) -> float | None:
    if not isinstance(manifest, Mapping):
        return None
    for key in (
        "bin_seconds",
        "bin_hop_seconds",
        "hop_seconds",
        "hop",
        "bin",
        "bin_size_seconds",
    ):
        value = manifest.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0:
            return float(value)
    return None


def load_audio_bins(
    config: Config,
    audio_version: str,
    work_id: int,
    *,
    version_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """读取一部作品的密集音频 bin 特征（适配 t1 audio_dense contract）。

    支持 data/audio_dense/<version>/<work_id>/audio.npz 与扁平 <work_id>.npz；
    必需 bin_features (B,D)，可选 bin_mask (B,)、bin_times (B,) 或 (B,2)；
    缺 times 时按版本 manifest 的 hop/bin 秒数（默认 0.5s）推断 bin 起点。
    """

    version = str(audio_version)
    root = config.data_path(AUDIO_DENSE_DIRNAME)
    version_dir = root / version
    if version_manifest is None:
        version_manifest = _load_json(version_dir / "manifest.json")
        if version_manifest is None:
            version_manifest = _load_json(root / f"{version}_manifest.json")
    path = next(
        (candidate for candidate in _audio_file_candidates(version_dir, work_id) if candidate.is_file()),
        None,
    )
    if path is None:
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            arrays = {key: np.asarray(payload[key]) for key in payload.files}
    except (OSError, ValueError) as exc:
        raise V05BetaError(f"读取音频 bin 失败（{path}）: {exc}") from exc
    features = _first_array(arrays, AUDIO_FEATURE_KEYS)
    if features is None:
        raise V05BetaError(f"音频 npz 缺少 bin_features（{path}）")
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise V05BetaError(f"bin_features 必须是二维 (B,D)，收到 {features.shape}（{path}）")
    if not np.all(np.isfinite(features)):
        raise V05BetaError(f"bin_features 含 NaN/Inf（{path}）")
    mask_array = _first_array(arrays, AUDIO_MASK_KEYS)
    if mask_array is None:
        mask = np.ones(features.shape[0], dtype=bool)
    else:
        mask = np.asarray(mask_array).reshape(-1).astype(bool)
        if mask.shape[0] != features.shape[0]:
            raise V05BetaError(
                f"bin_mask 长度 {mask.shape[0]} 与 bin_features 行数 {features.shape[0]} 不一致"
            )
    times_array = _first_array(arrays, AUDIO_TIME_KEYS)
    hop = _audio_hop_from_manifest(version_manifest)
    if times_array is None:
        hop = hop or BETA_BIN_SECONDS
        times = np.arange(features.shape[0], dtype=np.float64) * float(hop)
    else:
        times_array = np.asarray(times_array, dtype=np.float64)
        if times_array.ndim == 2 and times_array.shape[1] >= 1:
            times = times_array[:, 0]
        elif times_array.ndim == 1:
            times = times_array
        else:
            raise V05BetaError(f"bin_times 形状非法: {times_array.shape}（{path}）")
        if times.shape[0] != features.shape[0]:
            raise V05BetaError("bin_times 长度与 bin_features 行数不一致")
        if not np.all(np.isfinite(times)):
            raise V05BetaError(f"bin_times 含 NaN/Inf（{path}）")
        if hop is None and times.size >= 2:
            diffs = np.diff(np.sort(times))
            positive = diffs[diffs > 1e-9]
            if positive.size:
                hop = float(np.median(positive))
    hop = hop or BETA_BIN_SECONDS
    duration = arrays.get("duration")
    return {
        "path": str(path),
        "features": features,
        "mask": mask,
        "times": times.astype(np.float64),
        "hop_seconds": float(hop),
        "duration_s": (
            float(np.asarray(duration).reshape(-1)[0]) if duration is not None else None
        ),
        "dim": int(features.shape[1]),
        "version": version,
    }


# ---------------------------------------------------------------- 视觉 bin / 对齐 / 配对


@dataclass
class WorkPair:
    work_id: int
    visual: np.ndarray
    audio: np.ndarray
    times: np.ndarray
    duration_s: float
    visual_dim: int
    audio_dim: int
    coverage: dict[str, float]
    # 0.5s bin 事件统计标量（simple_sync 基线用；None 表示无事件特征）：
    # - visual_change: 相邻配对 bin 的 video frame_diff/motion/edge 变化 + 段内 std；
    # - audio_change: 相邻配对 bin 的 audio onset/rms/spectral_flux 正向变化 + 段内 std；
    # - event_valid: 该 bin 的 audio/visual_event mask 是否都有效。
    visual_change: np.ndarray | None = None
    audio_change: np.ndarray | None = None
    event_valid: np.ndarray | None = None

    @property
    def n_bins(self) -> int:
        return int(self.visual.shape[0])


def flatten_visual_bins(
    tokens: Mapping[str, np.ndarray], visual_encoder: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 v05_data token 的 (W,16,D) 视觉 bin 展平为 (W*16,D) + mask + 绝对时间。"""

    key = "visual_clip" if visual_encoder == "clip" else "visual_dino"
    visual = np.asarray(tokens[key], dtype=np.float32)
    mask = np.asarray(tokens["mask_visual"], dtype=bool)
    segment_times = np.asarray(tokens["segment_times"], dtype=np.float64)
    if visual.ndim != 3 or mask.ndim != 2 or segment_times.ndim != 2:
        raise V05BetaError(
            f"visual token 形状非法: {visual.shape}/{mask.shape}/{segment_times.shape}"
        )
    segments, bins, dim = visual.shape
    if mask.shape != (segments, bins) or segment_times.shape[0] != segments:
        raise V05BetaError("visual/mask/segment_times 形状不一致")
    starts = segment_times[:, 0]
    bin_times = starts[:, None] + np.arange(bins, dtype=np.float64)[None, :] * BETA_BIN_SECONDS
    return (
        visual.reshape(segments * bins, dim),
        mask.reshape(-1),
        bin_times.reshape(-1),
    )


def _nearest_audio_indices(
    visual_times: np.ndarray,
    visual_mask: np.ndarray,
    audio_times: np.ndarray,
    audio_mask: np.ndarray,
    hop_seconds: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """每个合法视觉 bin 选择最近的合法音频 bin（不跨越明显时间错位）。"""

    n_visual = int(visual_times.size)
    indices = np.full(n_visual, -1, dtype=np.int64)
    tolerance = max(BETA_BIN_SECONDS, float(hop_seconds)) * 0.5 + 1e-6
    valid_audio = np.flatnonzero(audio_mask)
    if valid_audio.size == 0:
        return indices, {"matched": 0, "tolerance_s": tolerance}
    audio_valid_times = audio_times[valid_audio]
    order = np.argsort(audio_valid_times, kind="mergesort")
    sorted_times = audio_valid_times[order]
    sorted_indices = valid_audio[order]
    matched = 0
    ambiguous = 0
    for position in np.flatnonzero(visual_mask):
        time_value = visual_times[position]
        insert = int(np.searchsorted(sorted_times, time_value, side="left"))
        candidates = []
        if insert < sorted_times.size:
            candidates.append(insert)
        if insert > 0:
            candidates.append(insert - 1)
        if not candidates:
            continue
        best = min(candidates, key=lambda index: abs(sorted_times[index] - time_value))
        diff = abs(float(sorted_times[best]) - float(time_value))
        if diff <= tolerance:
            indices[position] = int(sorted_indices[best])
            matched += 1
        elif diff <= tolerance * 2.0:
            ambiguous += 1
    return indices, {
        "matched": matched,
        "ambiguous_time": ambiguous,
        "tolerance_s": tolerance,
    }


def build_work_pair(
    tokens: Mapping[str, np.ndarray],
    audio: Mapping[str, Any],
    *,
    work_id: int,
    visual_encoder: str,
) -> tuple[WorkPair | None, dict[str, Any]]:
    """对齐视觉 0.5s bin 与密集音频 bin，形成一部作品的配对序列。"""

    visual, visual_mask, visual_times = flatten_visual_bins(tokens, visual_encoder)
    indices, stats = _nearest_audio_indices(
        visual_times,
        visual_mask,
        np.asarray(audio["times"], dtype=np.float64),
        np.asarray(audio["mask"], dtype=bool),
        float(audio["hop_seconds"]),
    )
    picked = np.flatnonzero(indices >= 0)
    details = {
        "visual_bins": int(visual.shape[0]),
        "visual_valid": int(visual_mask.sum()),
        "audio_bins": int(np.asarray(audio["features"]).shape[0]),
        "audio_valid": int(np.asarray(audio["mask"]).sum()),
        "matched": int(picked.size),
        **stats,
    }
    if picked.size < 2:
        details["reason"] = "matched_bins_lt_2"
        return None, details
    pair_times = visual_times[picked]
    order = np.argsort(pair_times, kind="mergesort")
    picked = picked[order]
    pair_times = pair_times[order]
    audio_indices = indices[picked]
    # 保留严格递增的音频 bin（重复/回退视为时间歧义）
    keep = np.ones(picked.size, dtype=bool)
    keep[1:] = np.diff(audio_indices) > 0
    dropped = int((~keep).sum())
    picked = picked[keep]
    pair_times = pair_times[keep]
    audio_indices = audio_indices[keep]
    if picked.size < 2:
        details["reason"] = "strictly_increasing_lt_2"
        return None, details
    visual_features = visual[picked]
    audio_features = np.asarray(audio["features"], dtype=np.float32)[audio_indices]
    duration = float(tokens.get("duration_s", np.asarray(0.0)).reshape(-1)[0])
    coverage = {
        "visual_valid_ratio": float(visual_mask.mean()) if visual_mask.size else 0.0,
        "audio_valid_ratio": (
            float(np.asarray(audio["mask"]).mean())
            if np.asarray(audio["mask"]).size
            else 0.0
        ),
        "matched_ratio": float(picked.size / max(1, int(visual_mask.sum()))),
        "dropped_non_increasing": dropped,
    }
    visual_change, audio_change, event_valid = event_change_scalars(tokens, picked)
    pair = WorkPair(
        work_id=int(work_id),
        visual=visual_features.astype(np.float32),
        audio=audio_features.astype(np.float32),
        times=pair_times.astype(np.float64),
        duration_s=duration,
        visual_dim=int(visual_features.shape[1]),
        audio_dim=int(audio_features.shape[1]),
        coverage=coverage,
        visual_change=visual_change,
        audio_change=audio_change,
        event_valid=event_valid,
    )
    details.update(
        {
            "paired_bins": pair.n_bins,
            "visual_dim": pair.visual_dim,
            "audio_dim": pair.audio_dim,
            "coverage": coverage,
            "event_valid_ratio": (
                float(event_valid.mean()) if event_valid is not None and event_valid.size else None
            ),
        }
    )
    return pair, details


# ---------------------------------------------------------------- 候选构造


@dataclass
class CandidateSet:
    work_id: int
    visual: np.ndarray
    candidates: tuple[np.ndarray, ...]
    labels: np.ndarray
    offsets: tuple[np.ndarray, ...]
    anchor_times: np.ndarray
    # simple_sync 基线用：anchor 的视觉变化标量 / 每个候选音频 bin 的起始变化标量；
    # None 表示该数据集没有事件特征（基线报告 unavailable，而不是伪造 0）。
    visual_change: np.ndarray | None = None
    candidate_audio_change: tuple[np.ndarray, ...] | None = None
    candidate_audio_valid: tuple[np.ndarray, ...] | None = None
    anchor_valid: np.ndarray | None = None

    @property
    def n_anchors(self) -> int:
        return int(self.visual.shape[0])

    @property
    def mean_candidates(self) -> float:
        if not self.candidates:
            return 0.0
        return float(np.mean([len(item) for item in self.candidates]))


# v05_data 事件 token 维度：visual_event = [video_mean(6), video_std(6)]；
# audio = [audio_mean(6), audio_std(6), av_mean(6), av_std(6)]；
# 音频 6 维顺序 = [rms, log_energy, spectral_flux, onset_strength, centroid, flatness]。
_VIDEO_FRAME_DIFF = 0
_VIDEO_MOTION = 1
_VIDEO_EDGE = 2
_AUDIO_RMS = 0
_AUDIO_FLUX = 2
_AUDIO_ONSET = 3


def event_change_scalars(
    tokens: Mapping[str, np.ndarray], picked: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """从 v05_data 事件 token 提取每对 bin 的视觉变化/音频起始标量。

    近似定义（事件特征本身是 bin 内 30fps/100Hz 统计，不是真实 onset 时间戳）：
    - visual_change[i] = |Δframe_diff| + |Δmotion| + |Δedge|（相对前一个配对 bin）
      + ||video_std[frame_diff,motion,edge]||；
    - audio_change[i] = max(0, Δonset_strength) + max(0, Δrms) + max(0, Δspectral_flux)
      （相对前一个配对 bin）+ ||audio_std[rms,log_energy,flux]||；
    - event_valid = mask_audio & mask_visual_event（任一无效则标量置 0 并标记无效）。
    故意只用事件统计、不学习、不使用模型权重/作品 ID/位置。
    """

    audio_token = tokens.get("audio")
    visual_event = tokens.get("visual_event")
    mask_audio = tokens.get("mask_audio")
    mask_visual_event = tokens.get("mask_visual_event")
    if (
        audio_token is None
        or visual_event is None
        or mask_audio is None
        or mask_visual_event is None
    ):
        return None, None, None
    audio_array = np.asarray(audio_token, dtype=np.float32)
    visual_event_array = np.asarray(visual_event, dtype=np.float32)
    if audio_array.ndim != 3 or visual_event_array.ndim != 3:
        return None, None, None
    if audio_array.shape[-1] < 12 or visual_event_array.shape[-1] < 9:
        return None, None, None
    flat_audio = audio_array.reshape(-1, audio_array.shape[-1])[picked]
    flat_visual = visual_event_array.reshape(-1, visual_event_array.shape[-1])[picked]
    mask_audio_flat = np.asarray(mask_audio, dtype=bool).reshape(-1)[picked]
    mask_visual_flat = np.asarray(mask_visual_event, dtype=bool).reshape(-1)[picked]
    valid = mask_audio_flat & mask_visual_flat
    count = int(picked.size)
    visual_change = np.zeros(count, dtype=np.float32)
    audio_change = np.zeros(count, dtype=np.float32)
    for index in range(count):
        if not bool(valid[index]):
            continue
        previous = index - 1 if index > 0 else index
        visual_delta = (
            abs(float(flat_visual[index, _VIDEO_FRAME_DIFF]) - float(flat_visual[previous, _VIDEO_FRAME_DIFF]))
            + abs(float(flat_visual[index, _VIDEO_MOTION]) - float(flat_visual[previous, _VIDEO_MOTION]))
            + abs(float(flat_visual[index, _VIDEO_EDGE]) - float(flat_visual[previous, _VIDEO_EDGE]))
        )
        visual_std = float(np.linalg.norm(flat_visual[index, 6:9]))
        visual_change[index] = np.float32(visual_delta + visual_std)
        audio_delta = (
            max(0.0, float(flat_audio[index, _AUDIO_ONSET]) - float(flat_audio[previous, _AUDIO_ONSET]))
            + max(0.0, float(flat_audio[index, _AUDIO_RMS]) - float(flat_audio[previous, _AUDIO_RMS]))
            + max(0.0, float(flat_audio[index, _AUDIO_FLUX]) - float(flat_audio[previous, _AUDIO_FLUX]))
        )
        audio_std = float(np.linalg.norm(flat_audio[index, 6:9]))
        audio_change[index] = np.float32(audio_delta + audio_std)
    return visual_change, audio_change, valid


def _cosine_rows(
    first: np.ndarray, second: np.ndarray
) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(first, second) / denominator)


def construct_candidates(
    pair: WorkPair,
    *,
    offsets: Sequence[int],
    near_dup: float,
    rng: np.random.Generator,
) -> tuple[CandidateSet | None, dict[str, Any]]:
    """为每部作品构造多候选局部音画匹配样本（正样本 + 时间偏移负样本）。"""

    n_bins = pair.n_bins
    stats = {
        "anchors": n_bins,
        "kept": 0,
        "skipped_no_negative": 0,
        "skipped_ambiguous_time": 0,
        "skipped_near_dup": 0,
        "candidates": 0,
    }
    visuals: list[np.ndarray] = []
    candidate_features: list[np.ndarray] = []
    labels: list[int] = []
    offset_arrays: list[np.ndarray] = []
    anchor_times: list[float] = []
    has_events = (
        pair.visual_change is not None
        and pair.audio_change is not None
        and pair.event_valid is not None
    )
    visual_changes: list[float] = []
    candidate_changes: list[np.ndarray] = []
    candidate_valids: list[np.ndarray] = []
    anchor_valids: list[bool] = []
    for anchor in range(n_bins):
        visual_anchor = pair.visual[anchor]
        audio_anchor = pair.audio[anchor]
        chosen_audio = [audio_anchor]
        chosen_offsets = [0]
        chosen_changes = [float(pair.audio_change[anchor])] if has_events else []
        chosen_valids = [bool(pair.event_valid[anchor])] if has_events else []
        for delta in offsets:
            target = anchor + int(delta)
            if target >= n_bins:
                continue
            gap = float(pair.times[target] - pair.times[anchor])
            expected = float(delta) * BETA_BIN_SECONDS
            if abs(gap - expected) > BETA_BIN_SECONDS * 0.5 + 1e-6:
                stats["skipped_ambiguous_time"] += 1
                continue
            if _cosine_rows(visual_anchor, pair.visual[target]) > near_dup:
                stats["skipped_near_dup"] += 1
                continue
            if _cosine_rows(audio_anchor, pair.audio[target]) > near_dup:
                stats["skipped_near_dup"] += 1
                continue
            chosen_audio.append(pair.audio[target])
            chosen_offsets.append(int(delta))
            if has_events:
                chosen_changes.append(float(pair.audio_change[target]))
                chosen_valids.append(bool(pair.event_valid[target]))
        if len(chosen_audio) < 2:
            stats["skipped_no_negative"] += 1
            continue
        permutation = rng.permutation(len(chosen_audio))
        ordered_audio = np.stack(chosen_audio, axis=0)[permutation]
        ordered_offsets = np.asarray(chosen_offsets, dtype=np.int64)[permutation]
        label = int(np.flatnonzero(ordered_offsets == 0)[0])
        visuals.append(visual_anchor)
        candidate_features.append(ordered_audio.astype(np.float32))
        labels.append(label)
        offset_arrays.append(ordered_offsets)
        anchor_times.append(float(pair.times[anchor]))
        if has_events:
            change_array = np.asarray(chosen_changes, dtype=np.float32)[permutation]
            valid_array = np.asarray(chosen_valids, dtype=bool)[permutation]
            candidate_changes.append(change_array)
            candidate_valids.append(valid_array)
            visual_changes.append(float(pair.visual_change[anchor]))
            anchor_valids.append(bool(pair.event_valid[anchor]))
        stats["candidates"] += len(chosen_audio)
    stats["kept"] = len(visuals)
    if not visuals:
        return None, {**stats, "reason": "no_anchor_with_negative"}
    candidate_set = CandidateSet(
        work_id=pair.work_id,
        visual=np.stack(visuals).astype(np.float32),
        candidates=tuple(candidate_features),
        labels=np.asarray(labels, dtype=np.int64),
        offsets=tuple(offset_arrays),
        anchor_times=np.asarray(anchor_times, dtype=np.float64),
        visual_change=(
            np.asarray(visual_changes, dtype=np.float32) if has_events else None
        ),
        candidate_audio_change=tuple(candidate_changes) if has_events else None,
        candidate_audio_valid=tuple(candidate_valids) if has_events else None,
        anchor_valid=np.asarray(anchor_valids, dtype=bool) if has_events else None,
    )
    return candidate_set, stats


# ---------------------------------------------------------------- 标准化 / 模型 / 损失


def fit_standardizer(pairs: Sequence[WorkPair]) -> dict[str, np.ndarray]:
    """只用 train 作品拟合视觉/音频逐维 mean/std（std 下限 1e-6）。"""

    if not pairs:
        raise V05BetaError("没有 train 作品可用于拟合标准化")
    visual = np.concatenate([pair.visual for pair in pairs], axis=0).astype(np.float64)
    audio = np.concatenate([pair.audio for pair in pairs], axis=0).astype(np.float64)
    return {
        "visual_mean": visual.mean(axis=0).astype(np.float32),
        "visual_std": np.maximum(visual.std(axis=0), 1e-6).astype(np.float32),
        "audio_mean": audio.mean(axis=0).astype(np.float32),
        "audio_std": np.maximum(audio.std(axis=0), 1e-6).astype(np.float32),
    }


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - 环境缺 torch
        raise V05BetaError("v0.5-beta 训练需要 torch（uv pip install -e '.[ml]'）") from exc
    return torch


def _device_for(cfg: BetaConfig) -> Any:
    torch = _torch()
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def build_model(
    *,
    visual_dim: int,
    audio_dim: int,
    standardizer: Mapping[str, np.ndarray],
    hidden: int = DEFAULT_HIDDEN,
    embed_dim: int = DEFAULT_EMBED_DIM,
    dropout: float = 0.1,
) -> Any:
    """双塔 MLP（视觉/音频各 2 层 -> embed_dim，L2 归一化）。"""

    torch = _torch()
    nn = torch.nn

    class BetaTwoTower(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual_dim = int(visual_dim)
            self.audio_dim = int(audio_dim)
            self.embed_dim = int(embed_dim)
            self.register_buffer(
                "visual_mean", torch.from_numpy(np.asarray(standardizer["visual_mean"], dtype=np.float32))
            )
            self.register_buffer(
                "visual_std", torch.from_numpy(np.asarray(standardizer["visual_std"], dtype=np.float32))
            )
            self.register_buffer(
                "audio_mean", torch.from_numpy(np.asarray(standardizer["audio_mean"], dtype=np.float32))
            )
            self.register_buffer(
                "audio_std", torch.from_numpy(np.asarray(standardizer["audio_std"], dtype=np.float32))
            )
            self.visual_net = nn.Sequential(
                nn.Linear(self.visual_dim, int(hidden)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden), self.embed_dim),
            )
            self.audio_net = nn.Sequential(
                nn.Linear(self.audio_dim, int(hidden)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden), self.embed_dim),
            )

        def forward(self, visual: Any, audio: Any) -> tuple[Any, Any]:
            visual_input = (visual - self.visual_mean) / self.visual_std
            audio_input = (audio - self.audio_mean) / self.audio_std
            visual_embedding = nn.functional.normalize(
                self.visual_net(visual_input), dim=-1, eps=1e-8
            )
            audio_embedding = nn.functional.normalize(
                self.audio_net(audio_input), dim=-1, eps=1e-8
            )
            return visual_embedding, audio_embedding

    return BetaTwoTower()


def infonce_loss(
    visual_embedding: Any,
    audio_embedding: Any,
    candidate_mask: Any,
    labels: Any,
    *,
    temperature: float,
) -> Any:
    """多候选 InfoNCE：每个 anchor 的视觉向量对候选音频做 softmax 分类。"""

    torch = _torch()
    scores = torch.einsum("bd,bkd->bk", visual_embedding, audio_embedding)
    scores = scores / float(temperature)
    scores = scores.masked_fill(~candidate_mask.bool(), -1e9)
    return torch.nn.functional.cross_entropy(scores, labels.long())


def collate_anchors(
    sets: Sequence[CandidateSet],
    picks: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """把若干 (set, anchor) 拼成一批（候选数 padding 到 batch 内最大）。"""

    torch = _torch()
    if not picks:
        raise V05BetaError("collate_anchors 需要至少一个 anchor")
    max_candidates = max(len(sets[set_index].candidates[anchor]) for set_index, anchor in picks)
    visual_dim = int(sets[0].visual.shape[1])
    audio_dim = int(sets[0].candidates[0].shape[1])
    visual = np.zeros((len(picks), visual_dim), dtype=np.float32)
    audio = np.zeros((len(picks), max_candidates, audio_dim), dtype=np.float32)
    candidate_mask = np.zeros((len(picks), max_candidates), dtype=bool)
    labels = np.zeros((len(picks),), dtype=np.int64)
    offsets = np.full((len(picks), max_candidates), -1, dtype=np.int64)
    work_ids = np.zeros((len(picks),), dtype=np.int64)
    for row, (set_index, anchor) in enumerate(picks):
        candidate_set = sets[set_index]
        visual[row] = candidate_set.visual[anchor]
        candidates = candidate_set.candidates[anchor]
        audio[row, : candidates.shape[0]] = candidates
        candidate_mask[row, : candidates.shape[0]] = True
        labels[row] = int(candidate_set.labels[anchor])
        offsets[row, : candidates.shape[0]] = candidate_set.offsets[anchor]
        work_ids[row] = candidate_set.work_id
    return {
        "visual": torch.from_numpy(visual),
        "audio": torch.from_numpy(audio),
        "candidate_mask": torch.from_numpy(candidate_mask),
        "labels": torch.from_numpy(labels),
        "offsets": torch.from_numpy(offsets),
        "work_ids": torch.from_numpy(work_ids),
    }


def _to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


# ---------------------------------------------------------------- 指标 / 评估


#: 坍缩检查对全量 anchor 做 O(n^2) 相似度会爆内存；确定性等距抽样足够作为诊断。
COLLAPSE_SAMPLE_MAX_ROWS = 4096


def _collapse_sample(matrix: np.ndarray) -> tuple[np.ndarray, int, bool]:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        return array, 0, False
    rows = int(array.shape[0])
    if rows <= COLLAPSE_SAMPLE_MAX_ROWS:
        return array, rows, False
    positions = np.linspace(0, rows - 1, num=COLLAPSE_SAMPLE_MAX_ROWS, dtype=np.int64)
    return array[positions], rows, True


def _effective_rank(matrix: np.ndarray) -> float | None:
    array, total_rows, sampled = _collapse_sample(matrix)
    if array.ndim != 2 or array.shape[0] < 2:
        return None
    centered = array - array.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    denominator = float(np.sum(singular**2))
    if denominator <= 1e-12:
        return 0.0
    return float(np.sum(singular) ** 2 / denominator)


def _pairwise_cosine(matrix: np.ndarray) -> dict[str, float | None]:
    array, total_rows, sampled = _collapse_sample(matrix)
    if array.ndim != 2 or array.shape[0] < 2:
        return {"mean": None, "std": None, "n_rows": total_rows, "sampled": sampled}
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    normalized = array / np.maximum(norms, 1e-9)
    similarity = normalized @ normalized.T
    mask = ~np.eye(similarity.shape[0], dtype=bool)
    values = similarity[mask]
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "n_rows": total_rows,
        "sampled": sampled,
    }


def _bootstrap_ci(
    per_work: Mapping[int, tuple[int, int]],
    *,
    seed: int,
    samples: int,
) -> dict[str, Any] | None:
    works = sorted(per_work)
    if not works:
        return None
    correct = np.asarray([per_work[work][0] for work in works], dtype=np.float64)
    totals = np.asarray([per_work[work][1] for work in works], dtype=np.float64)
    if np.any(totals <= 0):
        valid = totals > 0
        if not valid.any():
            return None
        works = [work for work, keep in zip(works, valid, strict=True) if keep]
        correct = correct[valid]
        totals = totals[valid]
    mean = float(correct.sum() / totals.sum())
    if int(samples) <= 0:
        return {"mean": mean, "ci95": [mean, mean], "n_works": len(works)}
    rng = np.random.default_rng([int(seed), 424242])
    estimates: list[float] = []
    for _ in range(int(samples)):
        picks = rng.integers(0, len(works), size=len(works))
        bootstrap_correct = float(correct[picks].sum())
        bootstrap_total = float(totals[picks].sum())
        if bootstrap_total > 0:
            estimates.append(bootstrap_correct / bootstrap_total)
    if not estimates:
        return {"mean": mean, "ci95": [mean, mean], "n_works": len(works)}
    low, high = np.percentile(np.asarray(estimates), [2.5, 97.5])
    return {"mean": mean, "ci95": [float(low), float(high)], "n_works": len(works)}


def _embedding_matrices(
    model: Any,
    sets: Sequence[CandidateSet],
    *,
    device: Any,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    torch = _torch()
    visual_rows: list[np.ndarray] = []
    audio_rows: list[np.ndarray] = []
    anchors = [(index, anchor) for index, item in enumerate(sets) for anchor in range(item.n_anchors)]
    with torch.no_grad():
        for start in range(0, len(anchors), int(batch_size)):
            batch = collate_anchors(sets, anchors[start : start + int(batch_size)])
            batch = _to_device(batch, device)
            visual_embedding, audio_embedding = model(batch["visual"], batch["audio"])
            visual_rows.append(visual_embedding.detach().cpu().numpy())
            # 只收集模型输出的候选音频 embedding（128d），不收集原始 768d 特征：
            # 全量 anchor 的原始特征会占用数 GB 并使成对相似度爆内存。
            audio_rows.append(
                audio_embedding[batch["candidate_mask"]].detach().cpu().numpy()
            )
    return (
        np.concatenate(visual_rows, axis=0) if visual_rows else np.zeros((0, 1)),
        np.concatenate(audio_rows, axis=0) if audio_rows else np.zeros((0, 1)),
    )


def _as_embedding_matrices(
    model: Any, sets: Sequence[CandidateSet], *, device: Any, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    return _embedding_matrices(model, sets, device=device, batch_size=batch_size)


def evaluate_candidates(
    model: Any,
    sets: Sequence[CandidateSet],
    *,
    device: Any,
    temperature: float,
    batch_size: int,
) -> dict[str, Any]:
    """Top-1 / Recall@K / 正负分数差 / 时间偏移曲线 / 作品级准确率。"""

    torch = _torch()
    model.eval()
    anchors = [(index, anchor) for index, item in enumerate(sets) for anchor in range(item.n_anchors)]
    correct = 0
    total = 0
    recall_hits = {1: 0, 3: 0, 5: 0}
    recall_totals = {1: 0, 3: 0, 5: 0}
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    offset_gap: dict[int, list[float]] = {}
    offset_correct: dict[int, list[bool]] = {}
    per_work: dict[int, list[int]] = {}
    with torch.no_grad():
        for start in range(0, len(anchors), int(batch_size)):
            picks = anchors[start : start + int(batch_size)]
            batch = collate_anchors(sets, picks)
            batch = _to_device(batch, device)
            visual_embedding, audio_embedding = model(batch["visual"], batch["audio"])
            scores = torch.einsum("bd,bkd->bk", visual_embedding, audio_embedding)
            scores = scores.masked_fill(~batch["candidate_mask"], -1e9)
            labels = batch["labels"]
            prediction = scores.argmax(dim=-1)
            matched = prediction == labels
            correct += int(matched.sum().item())
            total += int(labels.numel())
            positive = scores.gather(1, labels.unsqueeze(1)).squeeze(1)
            positive_scores.extend(float(value) for value in positive.detach().cpu())
            rank = (scores > positive.unsqueeze(1)).sum(dim=1)
            candidate_counts = batch["candidate_mask"].sum(dim=1)
            for k in recall_hits:
                valid_rows = candidate_counts >= k
                if bool(valid_rows.any()):
                    recall_totals[k] += int(valid_rows.sum().item())
                    hits = (rank < k) & valid_rows
                    recall_hits[k] += int(hits.sum().item())
            offsets = batch["offsets"].detach().cpu().numpy()
            score_rows = scores.detach().cpu().numpy()
            label_rows = labels.detach().cpu().numpy()
            work_rows = batch["work_ids"].detach().cpu().numpy()
            for row in range(label_rows.size):
                label_index = int(label_rows[row])
                anchor_positive = float(score_rows[row, label_index])
                for column in range(score_rows.shape[1]):
                    offset = int(offsets[row, column])
                    if offset <= 0:
                        continue
                    candidate_score = float(score_rows[row, column])
                    offset_gap.setdefault(offset, []).append(anchor_positive - candidate_score)
                    offset_correct.setdefault(offset, []).append(anchor_positive > candidate_score)
                    negative_scores.append(candidate_score)
            for row, work_id in enumerate(work_rows):
                bucket = per_work.setdefault(int(work_id), [0, 0])
                bucket[0] += int(score_rows[row].argmax() == label_rows[row])
                bucket[1] += 1
    per_work_metrics = {
        str(work_id): (values[0] / values[1] if values[1] else 0.0)
        for work_id, values in per_work.items()
    }
    offset_curve = {
        str(offset): {
            "accuracy": float(np.mean(offset_correct[offset])) if offset_correct[offset] else None,
            "score_gap": float(np.mean(offset_gap[offset])) if offset_gap[offset] else None,
            "count": len(offset_gap[offset]),
        }
        for offset in sorted(offset_gap)
    }
    visual_matrix, audio_matrix = _as_embedding_matrices(
        model, sets, device=device, batch_size=batch_size
    )
    positive_mean = float(np.mean(positive_scores)) if positive_scores else None
    negative_mean = float(np.mean(negative_scores)) if negative_scores else None
    return {
        "top1": float(correct / total) if total else None,
        "n_anchors": total,
        "n_works": len(per_work),
        "candidate_mean": (
            float(np.mean([item.mean_candidates for item in sets])) if sets else None
        ),
        "recall_at_1": float(recall_hits[1] / recall_totals[1]) if recall_totals[1] else None,
        "recall_at_3": float(recall_hits[3] / recall_totals[3]) if recall_totals[3] else None,
        "recall_at_5": float(recall_hits[5] / recall_totals[5]) if recall_totals[5] else None,
        "positive_score_mean": positive_mean,
        "negative_score_mean": negative_mean,
        "positive_negative_gap": (
            float(positive_mean - negative_mean)
            if positive_mean is not None and negative_mean is not None
            else None
        ),
        "offset_curve": offset_curve,
        "per_work_accuracy": per_work_metrics,
        "collapse": {
            "visual_effective_rank": _effective_rank(visual_matrix),
            "audio_effective_rank": _effective_rank(audio_matrix),
            "visual_pairwise_cosine": _pairwise_cosine(visual_matrix),
            "audio_pairwise_cosine": _pairwise_cosine(audio_matrix),
            "visual_embedding_dim": int(visual_matrix.shape[1]) if visual_matrix.size else 0,
            "collapse_sample_max_rows": COLLAPSE_SAMPLE_MAX_ROWS,
        },
    }


def _random_baseline(sets: Sequence[CandidateSet]) -> float | None:
    if not sets:
        return None
    chances = [1.0 / len(item.candidates[anchor]) for item in sets for anchor in range(item.n_anchors)]
    return float(np.mean(chances)) if chances else None




def _zscore_with_valid(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按 work 内有效值做 z-score；无效位置填 0，返回 (z, 归一化统计)。"""

    array = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    selected = array[mask]
    if selected.size == 0:
        return np.zeros_like(array), np.asarray([0.0, 1.0])
    mean = float(selected.mean())
    std = float(selected.std())
    if not math.isfinite(std) or std <= 1e-12:
        return np.zeros_like(array), np.asarray([mean, 0.0])
    return (array - mean) / std, np.asarray([mean, std])


def simple_sync_baseline(sets: Sequence[CandidateSet]) -> dict[str, Any]:
    """简单音画同步基线：音频起始变化 vs 视觉变化强度（不学习、不用模型权重/作品 ID）。

    使用与训练/测试完全相同的固定候选集。每个 works 内：
    - 对 anchor 的 visual_change 与候选音频的 audio_change 各自做 z-score（只用有效值）；
    - 候选得分 = -|z(visual_change_anchor) - z(audio_change_candidate)|（负绝对差）；
    - 无效候选得分 -inf，正样本无效的 anchor 跳过；Top-1/Recall 按最大得分。
    事件特征来自 v05_data 0.5s bin 的 audio(24 维含 audio/av mean/std) 与
    visual_event(12 维含 video mean/std)：起始/变化是 bin 内 100Hz/30fps 统计的
    正向差分近似，**不是**采样级 onset 时间戳（限制见 metrics 的 approximation 字段）。
    """

    empty = {
        "top1": None,
        "recall_at_1": None,
        "recall_at_3": None,
        "recall_at_5": None,
        "n_anchors": 0,
        "n_works": 0,
        "per_work_accuracy": {},
        "definition": "z-scored visual_change vs audio_change negative absolute difference on fixed candidates",
        "uses_model": False,
        "uses_work_id": False,
        "note": "",
    }
    if not sets:
        empty["note"] = "没有候选集"
        return empty
    if any(
        item.visual_change is None
        or item.candidate_audio_change is None
        or item.candidate_audio_valid is None
        or item.anchor_valid is None
        for item in sets
    ):
        empty["note"] = "事件特征不可用（WorkPair 未携带 audio/visual_event 统计），simple_sync 无法计算"
        return empty
    correct = 0
    total = 0
    recall_hits = {1: 0, 3: 0, 5: 0}
    recall_totals = {1: 0, 3: 0, 5: 0}
    per_work: dict[str, float] = {}
    for item in sets:
        visual_change = np.asarray(item.visual_change, dtype=np.float64)
        anchor_valid = np.asarray(item.anchor_valid, dtype=bool)
        candidate_changes = [
            np.asarray(array, dtype=np.float64) for array in item.candidate_audio_change or ()
        ]
        candidate_valid = [
            np.asarray(array, dtype=bool) for array in item.candidate_audio_valid or ()
        ]
        if len(candidate_changes) != item.n_anchors:
            continue
        pooled_values: list[np.ndarray] = []
        for values, valid in zip(candidate_changes, candidate_valid, strict=True):
            selected = values[valid]
            if selected.size:
                pooled_values.append(selected)
        if not pooled_values:
            continue
        pooled = np.concatenate(pooled_values)
        if pooled.size < 2 or float(pooled.std()) <= 1e-12:
            continue
        z_values, _ = _zscore_with_valid(visual_change, anchor_valid)
        audio_mean = float(pooled.mean())
        audio_std = float(pooled.std())
        work_correct = 0
        work_total = 0
        for anchor in range(item.n_anchors):
            if not bool(anchor_valid[anchor]):
                continue
            values = candidate_changes[anchor]
            valid = candidate_valid[anchor]
            label = int(item.labels[anchor])
            if label < 0 or label >= values.size or not bool(valid[label]):
                continue
            scores = np.full(values.size, -np.inf, dtype=np.float64)
            z_audio = (values - audio_mean) / audio_std
            scores[valid] = -np.abs(z_values[anchor] - z_audio[valid])
            if not np.isfinite(scores).any():
                continue
            prediction = int(np.argmax(scores))
            positive_score = float(scores[label])
            rank = int((scores > positive_score).sum())
            correct += int(prediction == label)
            total += 1
            work_correct += int(prediction == label)
            work_total += 1
            for k in recall_hits:
                recall_totals[k] += 1
                recall_hits[k] += int(rank < k)
        if work_total:
            per_work[str(item.work_id)] = float(work_correct / work_total)
    empty["n_anchors"] = total
    empty["n_works"] = len(per_work)
    empty["per_work_accuracy"] = per_work
    if total:
        empty["top1"] = float(correct / total)
        empty["recall_at_1"] = float(recall_hits[1] / recall_totals[1])
        empty["recall_at_3"] = (
            float(recall_hits[3] / recall_totals[3]) if recall_totals[3] else None
        )
        empty["recall_at_5"] = (
            float(recall_hits[5] / recall_totals[5]) if recall_totals[5] else None
        )
    empty["note"] = (
        "简单音画同步基线（确定性、无学习、无模型权重/无作品 ID）；"
        "audio_change/visual_change 是 0.5s bin 内事件统计的正向差分近似，不是采样级 onset"
    )
    return empty


def _temporal_sync_baseline(sets: Sequence[CandidateSet]) -> dict[str, Any]:
    """平凡时间同步基线：按候选时间偏移 |Δ| 最小者选正样本（必然 1.0，仅作上界记录）。"""

    correct = 0
    total = 0
    for item in sets:
        for anchor in range(item.n_anchors):
            offsets = item.offsets[anchor]
            prediction = int(np.argmin(np.abs(offsets)))
            correct += int(prediction == int(item.labels[anchor]))
            total += 1
    return {
        "top1": float(correct / total) if total else None,
        "note": "trivial time-index baseline; equals 1.0 by construction, only an upper-bound sanity check",
    }


def _replace_audio(
    sets: Sequence[CandidateSet], *, mode: str
) -> list[CandidateSet]:
    """消融：音频置零 / 时间偏移（候选音频轮转到同一作品下一 anchor）。"""

    replaced: list[CandidateSet] = []
    for item in sets:
        candidates = []
        for anchor in range(item.n_anchors):
            original = item.candidates[anchor]
            if mode == "audio_zero":
                candidates.append(np.zeros_like(original))
            elif mode == "time_shift":
                source_anchor = (anchor + 1) % item.n_anchors if item.n_anchors > 1 else anchor
                source = item.candidates[source_anchor]
                if source.shape[0] >= original.shape[0]:
                    candidates.append(source[: original.shape[0]].copy())
                else:
                    padded = np.zeros_like(original)
                    padded[: source.shape[0]] = source
                    candidates.append(padded)
            else:
                candidates.append(original.copy())
        replaced.append(
            CandidateSet(
                work_id=item.work_id,
                visual=item.visual.copy(),
                candidates=tuple(candidates),
                labels=item.labels.copy(),
                offsets=tuple(array.copy() for array in item.offsets),
                anchor_times=item.anchor_times.copy(),
            )
        )
    return replaced


def _replace_audio_across_works(sets: Sequence[CandidateSet]) -> list[CandidateSet]:
    """消融：把候选音频替换为下一部作品的音频（保持视觉不变）。"""

    replaced: list[CandidateSet] = []
    count = len(sets)
    for index, item in enumerate(sets):
        source_item = sets[(index + 1) % count]
        candidates = []
        for anchor in range(item.n_anchors):
            original = item.candidates[anchor]
            source_anchor = anchor % source_item.n_anchors
            source = source_item.candidates[source_anchor]
            if source.shape[0] >= original.shape[0]:
                candidates.append(source[: original.shape[0]].copy())
            else:
                padded = np.zeros_like(original)
                padded[: source.shape[0]] = source
                candidates.append(padded)
        replaced.append(
            CandidateSet(
                work_id=item.work_id,
                visual=item.visual.copy(),
                candidates=tuple(candidates),
                labels=item.labels.copy(),
                offsets=tuple(array.copy() for array in item.offsets),
                anchor_times=item.anchor_times.copy(),
            )
        )
    return replaced


def evaluate_ablations(
    model: Any,
    sets: Sequence[CandidateSet],
    *,
    device: Any,
    temperature: float,
    batch_size: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    normal = evaluate_candidates(
        model, sets, device=device, temperature=temperature, batch_size=batch_size
    )
    results["normal"] = {"top1": normal["top1"], "positive_negative_gap": normal["positive_negative_gap"]}
    for mode in ("audio_zero", "time_shift"):
        ablated = _replace_audio(sets, mode=mode)
        metrics = evaluate_candidates(
            model, ablated, device=device, temperature=temperature, batch_size=batch_size
        )
        results[mode] = {
            "top1": metrics["top1"],
            "positive_negative_gap": metrics["positive_negative_gap"],
        }
    replacement_metrics = evaluate_candidates(
        model,
        _replace_audio_across_works(sets),
        device=device,
        temperature=temperature,
        batch_size=batch_size,
    )
    results["audio_replacement"] = {
        "top1": replacement_metrics["top1"],
        "positive_negative_gap": replacement_metrics["positive_negative_gap"],
        "note": "候选音频替换为下一部作品的音频",
    }
    # 视觉置零：直接对候选集复制后把 visual 置零
    visual_zero_sets = [
        CandidateSet(
            work_id=item.work_id,
            visual=np.zeros_like(item.visual),
            candidates=item.candidates,
            labels=item.labels,
            offsets=item.offsets,
            anchor_times=item.anchor_times,
        )
        for item in sets
    ]
    metrics = evaluate_candidates(
        model, visual_zero_sets, device=device, temperature=temperature, batch_size=batch_size
    )
    results["visual_zero"] = {
        "top1": metrics["top1"],
        "positive_negative_gap": metrics["positive_negative_gap"],
    }
    return results


# ---------------------------------------------------------------- 数据集构建


@dataclass
class BetaDataset:
    pairs: dict[int, WorkPair]
    candidate_sets: dict[int, CandidateSet]
    split: BetaSplit
    standardizer: dict[str, np.ndarray]
    info: dict[str, Any]
    candidate_manifest: dict[str, Any]


def _canonical_works(catalog: Catalog) -> list[dict[str, Any]]:
    duplicates = catalog.get_duplicate_work_ids()
    works = [
        work
        for work in catalog.list_works()
        if int(work["id"]) not in duplicates
        and str(work.get("status")) == "completed"
        and str(work.get("source")) in ("local", "bilibili")
    ]
    works.sort(key=lambda work: int(work["id"]))
    return works


def _build_synthetic_dataset(cfg: BetaConfig) -> BetaDataset:
    """CPU smoke：已知隐含配对的合成视觉/音频 bin（非真实数据）。"""

    rng = np.random.default_rng(int(cfg.split_seed))
    latent_dim = 8
    pairs: dict[int, WorkPair] = {}
    visual_dim = int(cfg.smoke_visual_dim)
    audio_dim = int(cfg.smoke_audio_dim)
    visual_weight = rng.normal(size=(latent_dim, visual_dim)).astype(np.float32)
    audio_weight = rng.normal(size=(latent_dim, audio_dim)).astype(np.float32)
    work_ids = list(range(1, int(cfg.smoke_works) + 1))
    for work_id in work_ids:
        bins = max(8, int(cfg.smoke_bins) + int(work_id) % 5 - 2)
        latent = rng.normal(size=(bins, latent_dim)).astype(np.float32)
        visual = latent @ visual_weight + 0.15 * rng.normal(size=(bins, visual_dim)).astype(np.float32)
        audio = latent @ audio_weight + 0.15 * rng.normal(size=(bins, audio_dim)).astype(np.float32)
        times = np.arange(bins, dtype=np.float64) * BETA_BIN_SECONDS
        visual_change = np.zeros(bins, dtype=np.float32)
        audio_change = np.zeros(bins, dtype=np.float32)
        if bins > 1:
            visual_change[1:] = np.linalg.norm(np.diff(visual, axis=0), axis=1).astype(np.float32)
            audio_change[1:] = np.linalg.norm(np.diff(audio, axis=0), axis=1).astype(np.float32)
        pairs[work_id] = WorkPair(
            work_id=work_id,
            visual=visual,
            audio=audio,
            times=times,
            duration_s=float(bins) * BETA_BIN_SECONDS,
            visual_dim=visual_dim,
            audio_dim=audio_dim,
            coverage={"visual_valid_ratio": 1.0, "audio_valid_ratio": 1.0, "matched_ratio": 1.0},
            visual_change=visual_change,
            audio_change=audio_change,
            event_valid=np.ones(bins, dtype=bool),
        )
    split = split_work_ids(work_ids, seed=cfg.split_seed, split_counts=cfg.split_counts)
    standardizer = fit_standardizer([pairs[work_id] for work_id in split.train_ids])
    candidate_sets: dict[int, CandidateSet] = {}
    candidate_stats: dict[int, dict[str, Any]] = {}
    for work_id in work_ids:
        candidate_rng = np.random.default_rng([int(cfg.split_seed), int(work_id)])
        candidate_set, stats = construct_candidates(
            pairs[work_id],
            offsets=cfg.offsets,
            near_dup=cfg.near_dup,
            rng=candidate_rng,
        )
        if candidate_set is not None:
            candidate_sets[work_id] = candidate_set
        candidate_stats[work_id] = stats
    if not candidate_sets:
        raise V05BetaError("合成数据没有可用候选对")
    info = {
        "dataset_version": "synthetic",
        "audio_version": "synthetic",
        "visual_encoder": cfg.visual_encoder,
        "synthetic": True,
        "n_works_available": len(candidate_sets),
        "n_pairs": len(pairs),
        "n_with_tokens": len(pairs),
        "n_with_audio": len(pairs),
        "visual_dim": visual_dim,
        "audio_dim": audio_dim,
        "bin_seconds": BETA_BIN_SECONDS,
        "split": split.to_dict(),
        "limitations": ["smoke 合成数据：仅用于 CPU 链路验证，不能作为真实指标"],
    }
    candidate_manifest = {
        "dataset_version": "synthetic",
        "audio_version": "synthetic",
        "created_at": now_utc_iso(),
        "synthetic": True,
        "offsets": [int(value) for value in cfg.offsets],
        "near_dup_cosine_threshold": float(cfg.near_dup),
        "bin_seconds": BETA_BIN_SECONDS,
        "split": split.to_dict(),
        "works": {
            str(work_id): {
                "n_bins": pairs[work_id].n_bins,
                "n_anchors": (
                    candidate_sets[work_id].n_anchors if work_id in candidate_sets else 0
                ),
                "candidate_stats": candidate_stats[work_id],
            }
            for work_id in work_ids
            if work_id in candidate_sets
        },
        "split_counts_limit": None,
    }
    return BetaDataset(
        pairs={work_id: pairs[work_id] for work_id in candidate_sets},
        candidate_sets=candidate_sets,
        split=split,
        standardizer=standardizer,
        info=info,
        candidate_manifest=candidate_manifest,
    )


def build_beta_dataset(
    config: Config,
    catalog: Catalog,
    cfg: BetaConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> BetaDataset:
    """加载全量 token + 密集音频，构造局部配对候选集与确定性划分。"""

    cfg = cfg.validate()
    if cfg.smoke:
        return _build_synthetic_dataset(cfg)
    works = _canonical_works(catalog)
    if cfg.limit is not None:
        works = works[: int(cfg.limit)]
    excluded_ids = {int(value) for value in cfg.exclude_work_ids}
    if excluded_ids:
        works = [work for work in works if int(work["id"]) not in excluded_ids]
    if not works:
        raise V05BetaError("没有 canonical + completed 作品")
    audio_version, audio_dir, audio_manifest = resolve_audio_version(
        config, cfg.audio_version
    )
    from . import v05_data

    pairs: dict[int, WorkPair] = {}
    work_details: dict[int, dict[str, Any]] = {}
    missing_tokens: list[int] = []
    missing_audio: list[int] = []
    pair_failures: list[dict[str, Any]] = []
    for index, work in enumerate(works, start=1):
        work_id = int(work["id"])
        tokens = v05_data.load_work_tokens(config, cfg.dataset_version, work_id)
        if tokens is None:
            missing_tokens.append(work_id)
            continue
        if cfg.max_segments is not None:
            for key in v05_data.REQUIRED_TOKEN_KEYS:
                if key in tokens and np.asarray(tokens[key]).ndim >= 1 and key != "duration_s":
                    tokens[key] = np.asarray(tokens[key])[: int(cfg.max_segments)]
        audio = load_audio_bins(
            config,
            audio_version,
            work_id,
            version_manifest=audio_manifest,
        )
        if audio is None:
            missing_audio.append(work_id)
            continue
        pair, details = build_work_pair(
            tokens, audio, work_id=work_id, visual_encoder=cfg.visual_encoder
        )
        work_details[work_id] = details
        if pair is None:
            pair_failures.append({"work_id": work_id, "reason": details.get("reason", "pair_failed")})
            continue
        pairs[work_id] = pair
        if progress:
            progress(
                f"[{index}/{len(works)}] work_id={work_id} 配对 bin={pair.n_bins} "
                f"覆盖率={pair.coverage.get('matched_ratio', 0):.3f}"
            )
    if not pairs:
        raise V05BetaError(
            "没有可用音画配对作品："
            f"pairs=0，缺 token={len(missing_tokens)}，缺 audio={len(missing_audio)}，"
            f"配对失败={len(pair_failures)}；请检查 dataset/audio 版本"
        )
    available_ids = sorted(pairs)
    candidate_sets: dict[int, CandidateSet] = {}
    candidate_stats: dict[int, dict[str, Any]] = {}
    candidate_failures: list[dict[str, Any]] = []
    for work_id in available_ids:
        candidate_rng = np.random.default_rng([int(cfg.split_seed), int(work_id)])
        candidate_set, stats = construct_candidates(
            pairs[work_id],
            offsets=cfg.offsets,
            near_dup=cfg.near_dup,
            rng=candidate_rng,
        )
        candidate_stats[work_id] = stats
        if candidate_set is not None:
            candidate_sets[work_id] = candidate_set
        else:
            candidate_failures.append(
                {"work_id": int(work_id), "reason": str(stats.get("reason") or "no_candidates")}
            )
    if not candidate_sets:
        raise V05BetaError("没有可用候选对（candidate_sets 为空）")
    # 划分只基于“有可用候选对”的作品：否则某作品进入 split 却没有候选，
    # 会 silently 从评估中消失，并让 n_works_available 与 split 计数不一致。
    usable_ids = sorted(candidate_sets)
    split = split_work_ids(usable_ids, seed=cfg.split_seed, split_counts=cfg.split_counts)
    standardizer = fit_standardizer([pairs[work_id] for work_id in split.train_ids])
    limitations: list[str] = []
    if len(usable_ids) < 110:
        limitations.append(
            f"可用作品 {len(usable_ids)}/110 部（有音画配对与候选集）；"
            f"划分 {len(split.train_ids)}/{len(split.val_ids)}/{len(split.test_ids)}；"
            "缺少音频、token 或无候选的作品不进入 split/训练/评估"
        )
    if candidate_failures:
        limitations.append(
            "候选集为空的作品: "
            + ", ".join(f"{item['work_id']}({item['reason']})" for item in candidate_failures)
        )
    if excluded_ids:
        limitations.append(
            "显式排除的作品: "
            + ", ".join(str(value) for value in sorted(excluded_ids))
            + "（原因记录在 --exclude-work-ids 调用方）"
        )
    limitations.extend(
        [
            "音频来自 t1 audio_dense 适配层（bin_features/bin_mask/bin_times 或 manifest hop 推断）",
            "候选负样本仅同作品 t+Δ；不使用作品 ID/位置/外部元数据作为特征",
            "未训练 Cross-Attention 第二阶段；本轮只做最小双塔 InfoNCE",
        ]
    )
    audio_manifest_summary = None
    if isinstance(audio_manifest, Mapping):
        audio_manifest_summary = {
            key: audio_manifest.get(key)
            for key in (
                "version",
                "model",
                "revision",
                "sample_rate",
                "context",
                "hop",
                "layer",
                "pool",
                "bin",
                "normalize",
                "created_at",
            )
        }
    info = {
        "dataset_version": cfg.dataset_version,
        "audio_version": audio_version,
        "audio_dir": str(audio_dir),
        "audio_manifest": audio_manifest_summary,
        "visual_encoder": cfg.visual_encoder,
        "synthetic": False,
        "n_works_total": len(works),
        "n_works_available": len(usable_ids),
        "n_works_with_pairs": len(available_ids),
        "n_pairs": len(pairs),
        "n_with_tokens": len(pairs) + len(pair_failures),
        "n_with_audio": len(pairs) + len(pair_failures),
        "missing_tokens": missing_tokens,
        "missing_audio": missing_audio,
        "pair_failures": pair_failures,
        "candidate_failures": candidate_failures,
        "excluded_work_ids": sorted(excluded_ids),
        "visual_dim": int(next(iter(pairs.values())).visual_dim),
        "audio_dim": int(next(iter(pairs.values())).audio_dim),
        "bin_seconds": BETA_BIN_SECONDS,
        "split": split.to_dict(),
        "limitations": limitations,
        "work_details": {str(work_id): details for work_id, details in work_details.items()},
    }
    candidate_manifest = {
        "dataset_version": cfg.dataset_version,
        "audio_version": audio_version,
        "visual_encoder": cfg.visual_encoder,
        "created_at": now_utc_iso(),
        "synthetic": False,
        "offsets": [int(value) for value in cfg.offsets],
        "near_dup_cosine_threshold": float(cfg.near_dup),
        "bin_seconds": BETA_BIN_SECONDS,
        "split_seed": int(cfg.split_seed),
        "split": split.to_dict(),
        "max_segments": cfg.max_segments,
        "works": {
            str(work_id): {
                "n_bins": pairs[work_id].n_bins,
                "n_anchors": (
                    candidate_sets[work_id].n_anchors
                    if work_id in candidate_sets
                    else 0
                ),
                "candidate_stats": candidate_stats[work_id],
                "coverage": pairs[work_id].coverage,
            }
            for work_id in available_ids
        },
        "limitations": limitations,
    }
    return BetaDataset(
        pairs={work_id: pairs[work_id] for work_id in usable_ids},
        candidate_sets=candidate_sets,
        split=split,
        standardizer=standardizer,
        info=info,
        candidate_manifest=candidate_manifest,
    )

# ---------------------------------------------------------------- 训练 / 产物


@dataclass
class BetaSeedResult:
    seed: int
    metrics: dict[str, Any]
    checkpoint_path: Path
    metrics_path: Path
    elapsed_s: float

    @property
    def test_top1(self) -> float | None:
        return self.metrics["test"]["top1"]


@dataclass
class BetaRunResult:
    summary: dict[str, Any]
    results: tuple[BetaSeedResult, ...]
    summary_path: Path
    candidates_manifest_path: Path


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(tmp_path, target)
    finally:
        tmp_path.unlink(missing_ok=True)
    return target


def _atomic_torch_save(torch: Any, payload: Mapping[str, Any], path: Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            torch.save(dict(payload), stream)
        os.replace(tmp_path, target)
    finally:
        tmp_path.unlink(missing_ok=True)
    return target


def _sets_for(bundle: BetaDataset, work_ids: Sequence[int]) -> list[CandidateSet]:
    return [bundle.candidate_sets[int(work_id)] for work_id in work_ids if int(work_id) in bundle.candidate_sets]


def _anchor_picks(sets: Sequence[CandidateSet]) -> list[tuple[int, int]]:
    return [
        (set_index, anchor)
        for set_index, item in enumerate(sets)
        for anchor in range(item.n_anchors)
    ]


def _bootstrap_accuracy_ci(
    per_work_accuracy: Mapping[str, float], *, seed: int, samples: int
) -> dict[str, Any] | None:
    works = sorted(per_work_accuracy)
    if not works:
        return None
    values = np.asarray([float(per_work_accuracy[work]) for work in works], dtype=np.float64)
    mean = float(values.mean())
    if int(samples) <= 0:
        return {"mean": mean, "ci95": [mean, mean], "n_works": len(works)}
    rng = np.random.default_rng([int(seed), 20251026])
    estimates = []
    for _ in range(int(samples)):
        picks = rng.integers(0, values.size, size=values.size)
        estimates.append(float(values[picks].mean()))
    low, high = np.percentile(np.asarray(estimates), [2.5, 97.5])
    return {"mean": mean, "ci95": [float(low), float(high)], "n_works": len(works)}


def _train_epoch(
    model: Any,
    sets: Sequence[CandidateSet],
    picks: Sequence[tuple[int, int]],
    *,
    cfg: BetaConfig,
    seed: int,
    epoch: int,
    optimizer: Any,
    device: Any,
) -> dict[str, float]:
    _torch()
    model.train()
    rng = np.random.default_rng([int(seed), int(epoch)])
    order = rng.permutation(len(picks))
    losses: list[float] = []
    max_grad = 0.0
    for start in range(0, len(order), int(cfg.batch_size)):
        batch_picks = [picks[int(index)] for index in order[start : start + int(cfg.batch_size)]]
        batch = collate_anchors(sets, batch_picks)
        batch = _to_device(batch, device)
        visual_embedding, audio_embedding = model(batch["visual"], batch["audio"])
        loss = infonce_loss(
            visual_embedding,
            audio_embedding,
            batch["candidate_mask"],
            batch["labels"],
            temperature=cfg.temperature,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = 0.0
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            grad_norm += float(parameter.grad.detach().norm(2) ** 2)
        max_grad = max(max_grad, math.sqrt(grad_norm))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {
        "train_loss": float(np.mean(losses)) if losses else float("nan"),
        "max_grad_norm": max_grad,
        "n_batches": len(losses),
    }


def _evaluate_loss(
    model: Any,
    sets: Sequence[CandidateSet],
    *,
    cfg: BetaConfig,
    device: Any,
) -> float | None:
    torch = _torch()
    model.eval()
    picks = _anchor_picks(sets)
    if not picks:
        return None
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, len(picks), int(cfg.batch_size)):
            batch = collate_anchors(sets, picks[start : start + int(cfg.batch_size)])
            batch = _to_device(batch, device)
            visual_embedding, audio_embedding = model(batch["visual"], batch["audio"])
            loss = infonce_loss(
                visual_embedding,
                audio_embedding,
                batch["candidate_mask"],
                batch["labels"],
                temperature=cfg.temperature,
            )
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else None


def _model_param_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _artifact_version(cfg: BetaConfig) -> str:
    """smoke 默认版本自动加 _smoke，避免覆盖真实 v05-beta 产物。"""

    if cfg.smoke and cfg.version == BETA_MODEL_VERSION:
        return f"{cfg.version}_smoke"
    return cfg.version


def train_beta_seed(
    config: Config,
    bundle: BetaDataset,
    cfg: BetaConfig,
    seed: int,
    *,
    device: Any | None = None,
    progress: Callable[[str], None] | None = None,
) -> BetaSeedResult:
    """训练一个 seed；val 选模型，test 只评一次；保存 checkpoint/metrics。"""

    torch = _torch()
    cfg = cfg.validate()
    resolved_device = device if device is not None else _device_for(cfg)
    started = time.perf_counter()
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats(resolved_device)
    train_sets = _sets_for(bundle, bundle.split.train_ids)
    val_sets = _sets_for(bundle, bundle.split.val_ids)
    test_sets = _sets_for(bundle, bundle.split.test_ids)
    if not train_sets or not val_sets or not test_sets:
        raise V05BetaError(
            "train/val/test 至少各需要一部有候选对的作品："
            f"{len(train_sets)}/{len(val_sets)}/{len(test_sets)}"
        )
    model = build_model(
        visual_dim=int(bundle.info["visual_dim"]),
        audio_dim=int(bundle.info["audio_dim"]),
        standardizer=bundle.standardizer,
        hidden=int(cfg.hidden),
        embed_dim=int(cfg.embed_dim),
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    train_picks = _anchor_picks(train_sets)
    if not train_picks:
        raise V05BetaError("train anchors 为空")
    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_val: float | None = None
    best_gap: float | None = None
    patience = 0
    early_stopped = False
    max_grad = 0.0
    for epoch in range(1, int(cfg.epochs) + 1):
        train_metrics = _train_epoch(
            model,
            train_sets,
            train_picks,
            cfg=cfg,
            seed=seed,
            epoch=epoch,
            optimizer=optimizer,
            device=resolved_device,
        )
        max_grad = max(max_grad, train_metrics["max_grad_norm"])
        val_metrics = evaluate_candidates(
            model,
            val_sets,
            device=resolved_device,
            temperature=float(cfg.temperature),
            batch_size=int(cfg.batch_size),
        )
        val_loss = _evaluate_loss(model, val_sets, cfg=cfg, device=resolved_device)
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["train_loss"],
            "max_grad_norm": train_metrics["max_grad_norm"],
            "val_top1": val_metrics["top1"],
            "val_loss": val_loss,
            "val_positive_negative_gap": val_metrics["positive_negative_gap"],
        }
        history.append(record)
        if progress:
            progress(
                f"seed={seed} epoch {epoch}/{cfg.epochs}: loss={record['train_loss']:.4f} "
                f"val_top1={record['val_top1']} gap={record['val_positive_negative_gap']}"
            )
        improved = False
        if val_metrics["top1"] is not None:
            if best_val is None or val_metrics["top1"] > best_val + 1e-4:
                improved = True
            elif abs(val_metrics["top1"] - best_val) <= 1e-4:
                gap = val_metrics["positive_negative_gap"]
                if gap is not None and (best_gap is None or gap > best_gap + 1e-6):
                    improved = True
        if improved:
            best_val = float(val_metrics["top1"])
            best_gap = val_metrics["positive_negative_gap"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if cfg.patience and patience >= int(cfg.patience):
                early_stopped = True
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    test_metrics = evaluate_candidates(
        model,
        test_sets,
        device=resolved_device,
        temperature=float(cfg.temperature),
        batch_size=int(cfg.batch_size),
    )
    ablations = evaluate_ablations(
        model,
        test_sets,
        device=resolved_device,
        temperature=float(cfg.temperature),
        batch_size=int(cfg.batch_size),
    )
    # 基线：随机 / simple_sync（事件统计相关性，不学习）/ 平凡时间同步 / 冻结投影
    random_top1 = _random_baseline(test_sets)
    simple_sync = simple_sync_baseline(test_sets)
    temporal_sync = _temporal_sync_baseline(test_sets)
    frozen_model = build_model(
        visual_dim=int(bundle.info["visual_dim"]),
        audio_dim=int(bundle.info["audio_dim"]),
        standardizer=bundle.standardizer,
        hidden=int(cfg.hidden),
        embed_dim=int(cfg.embed_dim),
    ).to(resolved_device)
    frozen_model.eval()
    frozen_metrics = evaluate_candidates(
        frozen_model,
        test_sets,
        device=resolved_device,
        temperature=float(cfg.temperature),
        batch_size=int(cfg.batch_size),
    )
    baseline_values = {
        "random_top1": random_top1,
        "simple_sync_top1": simple_sync["top1"],
        "simple_sync_recall_at_1": simple_sync["recall_at_1"],
        "simple_sync_recall_at_3": simple_sync["recall_at_3"],
        "simple_sync_recall_at_5": simple_sync["recall_at_5"],
        "simple_sync_n_anchors": simple_sync["n_anchors"],
        "simple_sync_definition": simple_sync["definition"],
        "simple_sync_note": simple_sync["note"],
        "simple_sync_uses_model": bool(simple_sync["uses_model"]),
        "frozen_projection_top1": frozen_metrics["top1"],
        "temporal_sync_top1": temporal_sync["top1"],
        "temporal_sync_note": temporal_sync["note"],
    }
    test_top1 = test_metrics["top1"]
    baseline_values["delta_vs_random"] = (
        float(test_top1 - random_top1)
        if test_top1 is not None and random_top1 is not None
        else None
    )
    baseline_values["delta_vs_frozen_projection"] = (
        float(test_top1 - frozen_metrics["top1"])
        if test_top1 is not None and frozen_metrics["top1"] is not None
        else None
    )
    baseline_values["delta_vs_simple_sync"] = (
        float(test_top1 - simple_sync["top1"])
        if test_top1 is not None and simple_sync["top1"] is not None
        else None
    )
    bootstrap_ci = _bootstrap_accuracy_ci(
        test_metrics.get("per_work_accuracy") or {},
        seed=int(seed),
        samples=int(cfg.bootstrap),
    )
    cuda_peak = None
    if resolved_device.type == "cuda":
        cuda_peak = float(torch.cuda.max_memory_allocated(resolved_device) / (1024**2))
    elapsed = time.perf_counter() - started
    artifact_version = _artifact_version(cfg)
    checkpoint_path = (
        config.data_path("models") / f"{artifact_version}_{int(seed)}" / "checkpoint.pt"
    )
    metrics_path = config.paths.logs / f"{artifact_version}_{int(seed)}_metrics.json"
    model_config = {
        "visual_dim": int(bundle.info["visual_dim"]),
        "audio_dim": int(bundle.info["audio_dim"]),
        "hidden": int(cfg.hidden),
        "embed_dim": int(cfg.embed_dim),
        "temperature": float(cfg.temperature),
    }
    dataset_summary = {
        "dataset_version": bundle.info["dataset_version"],
        "audio_version": bundle.info["audio_version"],
        "audio_manifest": bundle.info.get("audio_manifest"),
        "visual_encoder": bundle.info["visual_encoder"],
        "synthetic": bool(bundle.info["synthetic"]),
        "n_works_available": len(bundle.candidate_sets),
        "visual_dim": int(bundle.info["visual_dim"]),
        "audio_dim": int(bundle.info["audio_dim"]),
        "bin_seconds": BETA_BIN_SECONDS,
        "split": bundle.split.to_dict(),
        "limitations": bundle.info.get("limitations", []),
        "missing_tokens": bundle.info.get("missing_tokens", []),
        "missing_audio": bundle.info.get("missing_audio", []),
        "pair_failures": bundle.info.get("pair_failures", []),
    }
    candidate_summary = {
        "offsets": [int(value) for value in cfg.offsets],
        "near_dup_cosine_threshold": float(cfg.near_dup),
        "train_anchors": sum(item.n_anchors for item in train_sets),
        "val_anchors": sum(item.n_anchors for item in val_sets),
        "test_anchors": sum(item.n_anchors for item in test_sets),
        "test_mean_candidates": float(
            np.mean([item.mean_candidates for item in test_sets])
        )
        if test_sets
        else None,
        "candidate_manifest": str(
            config.paths.logs / f"{artifact_version}_candidates_manifest.json"
        ),
    }
    metrics = {
        "version": artifact_version,
        "seed": int(seed),
        "created_at": now_utc_iso(),
        "elapsed_s": elapsed,
        "config": cfg.to_dict(),
        "dataset": dataset_summary,
        "candidates": candidate_summary,
        "model": {**model_config, "params": _model_param_count(model)},
        "training": {
            "epochs_requested": int(cfg.epochs),
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "best_val_top1": best_val,
            "best_val_positive_negative_gap": best_gap,
            "early_stopped": early_stopped,
            "history": history,
            "device": str(resolved_device),
            "cuda_peak_memory_mb": cuda_peak,
            "max_grad_norm": max_grad,
        },
        "test": {
            "top1": test_top1,
            "n_anchors": test_metrics["n_anchors"],
            "n_works": test_metrics["n_works"],
            "candidate_mean": test_metrics["candidate_mean"],
            "recall_at_1": test_metrics["recall_at_1"],
            "recall_at_3": test_metrics["recall_at_3"],
            "recall_at_5": test_metrics["recall_at_5"],
            "positive_score_mean": test_metrics["positive_score_mean"],
            "negative_score_mean": test_metrics["negative_score_mean"],
            "positive_negative_gap": test_metrics["positive_negative_gap"],
            "offset_curve": test_metrics["offset_curve"],
            "collapse": test_metrics["collapse"],
            "per_work_accuracy": test_metrics["per_work_accuracy"],
            "bootstrap": bootstrap_ci,
        },
        "baselines": baseline_values,
        "ablations": ablations,
        "gates": {
            "top1_above_random": (
                None
                if test_top1 is None or random_top1 is None
                else bool(test_top1 > random_top1)
            ),
            "top1_above_frozen_projection": (
                None
                if test_top1 is None or frozen_metrics["top1"] is None
                else bool(test_top1 > frozen_metrics["top1"])
            ),
            "top1_above_simple_sync": (
                None
                if test_top1 is None or simple_sync["top1"] is None
                else bool(test_top1 > simple_sync["top1"])
            ),
        },
    }
    checkpoint = {
        "version": artifact_version,
        "seed": int(seed),
        "config": cfg.to_dict(),
        "dataset": dataset_summary,
        "candidates": candidate_summary,
        "model_config": model_config,
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "standardizer": {
            key: np.asarray(value) for key, value in bundle.standardizer.items()
        },
        "optimizer": optimizer.state_dict(),
        "epoch": best_epoch,
        "best_val_top1": best_val,
        "test_top1": test_top1,
        "created_at": now_utc_iso(),
    }
    _atomic_torch_save(torch, checkpoint, checkpoint_path)
    _atomic_write_json(metrics_path, metrics)
    return BetaSeedResult(
        seed=int(seed),
        metrics=metrics,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        elapsed_s=elapsed,
    )


def run_beta_training(
    config: Config,
    catalog: Catalog,
    cfg: BetaConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> BetaRunResult:
    """构建候选数据集 -> 多 seed 训练/验证 -> 产物落盘（不写 DB/不建 map）。"""

    started = time.perf_counter()
    cfg = cfg.validate()
    bundle = build_beta_dataset(config, catalog, cfg, progress=progress)
    artifact_version = _artifact_version(cfg)
    candidates_manifest_path = _atomic_write_json(
        config.paths.logs / f"{artifact_version}_candidates_manifest.json",
        bundle.candidate_manifest,
    )
    results: list[BetaSeedResult] = []
    for seed in cfg.seeds:
        if progress:
            progress(f"=== 开始 seed={seed}（{len(results) + 1}/{len(cfg.seeds)}）===")
        result = train_beta_seed(
            config,
            bundle,
            cfg,
            int(seed),
            progress=progress,
        )
        results.append(result)
    per_seed = [
        {
            "seed": result.seed,
            "test_top1": result.metrics["test"]["top1"],
            "recall_at_1": result.metrics["test"]["recall_at_1"],
            "recall_at_5": result.metrics["test"]["recall_at_5"],
            "positive_negative_gap": result.metrics["test"]["positive_negative_gap"],
            "best_epoch": result.metrics["training"]["best_epoch"],
            "best_val_top1": result.metrics["training"]["best_val_top1"],
            "bootstrap": result.metrics["test"]["bootstrap"],
            "random_top1": result.metrics["baselines"]["random_top1"],
            "simple_sync_top1": result.metrics["baselines"]["simple_sync_top1"],
            "simple_sync_recall_at_3": result.metrics["baselines"][
                "simple_sync_recall_at_3"
            ],
            "simple_sync_recall_at_5": result.metrics["baselines"][
                "simple_sync_recall_at_5"
            ],
            "frozen_projection_top1": result.metrics["baselines"][
                "frozen_projection_top1"
            ],
            "elapsed_s": result.elapsed_s,
            "checkpoint": str(result.checkpoint_path),
            "metrics": str(result.metrics_path),
        }
        for result in results
    ]
    top1_values = [
        float(item["test_top1"]) for item in per_seed if item["test_top1"] is not None
    ]
    aggregate = {
        "n_seeds": len(results),
        "test_top1_mean": float(np.mean(top1_values)) if top1_values else None,
        "test_top1_std": float(np.std(top1_values)) if top1_values else None,
        "test_top1_min": float(np.min(top1_values)) if top1_values else None,
        "test_top1_max": float(np.max(top1_values)) if top1_values else None,
        "random_top1_mean": float(
            np.mean(
                [
                    item["random_top1"]
                    for item in per_seed
                    if item["random_top1"] is not None
                ]
            )
        )
        if any(item["random_top1"] is not None for item in per_seed)
        else None,
        "frozen_projection_top1_mean": float(
            np.mean(
                [
                    item["frozen_projection_top1"]
                    for item in per_seed
                    if item["frozen_projection_top1"] is not None
                ]
            )
        )
        if any(item["frozen_projection_top1"] is not None for item in per_seed)
        else None,
        "simple_sync_top1_mean": float(
            np.mean(
                [
                    item["simple_sync_top1"]
                    for item in per_seed
                    if item["simple_sync_top1"] is not None
                ]
            )
        )
        if any(item["simple_sync_top1"] is not None for item in per_seed)
        else None,
    }
    aggregate["delta_vs_random"] = (
        float(aggregate["test_top1_mean"] - aggregate["random_top1_mean"])
        if aggregate["test_top1_mean"] is not None
        and aggregate["random_top1_mean"] is not None
        else None
    )
    aggregate["delta_vs_frozen_projection"] = (
        float(
            aggregate["test_top1_mean"] - aggregate["frozen_projection_top1_mean"]
        )
        if aggregate["test_top1_mean"] is not None
        and aggregate["frozen_projection_top1_mean"] is not None
        else None
    )
    aggregate["delta_vs_simple_sync"] = (
        float(aggregate["test_top1_mean"] - aggregate["simple_sync_top1_mean"])
        if aggregate["test_top1_mean"] is not None
        and aggregate["simple_sync_top1_mean"] is not None
        else None
    )
    summary = {
        "version": artifact_version,
        "created_at": now_utc_iso(),
        "elapsed_s": time.perf_counter() - started,
        "config": cfg.to_dict(),
        "dataset": {
            "dataset_version": bundle.info["dataset_version"],
            "audio_version": bundle.info["audio_version"],
            "audio_manifest": bundle.info.get("audio_manifest"),
            "visual_encoder": bundle.info["visual_encoder"],
            "synthetic": bool(bundle.info["synthetic"]),
            "n_works_available": len(bundle.candidate_sets),
            "visual_dim": int(bundle.info["visual_dim"]),
            "audio_dim": int(bundle.info["audio_dim"]),
            "split": bundle.split.to_dict(),
            "limitations": bundle.info.get("limitations", []),
            "missing_tokens": bundle.info.get("missing_tokens", []),
            "missing_audio": bundle.info.get("missing_audio", []),
            "pair_failures": bundle.info.get("pair_failures", []),
        },
        "candidates_manifest": str(candidates_manifest_path),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "artifacts": {
            "summary": str(config.paths.logs / f"{artifact_version}_summary.json"),
            "candidates_manifest": str(candidates_manifest_path),
            "checkpoints": [str(result.checkpoint_path) for result in results],
            "metrics": [str(result.metrics_path) for result in results],
        },
        "notes": [
            "研究产物：不注册 representations 表、不建 map、不写 DB",
            "test 只在训练/选模型完成后评一次；val 用于选 epoch",
            "候选负样本仅同作品时间偏移；不使用作品 ID/位置/外部元数据作为特征",
        ],
    }
    _atomic_write_json(config.paths.logs / f"{artifact_version}_summary.json", summary)
    return BetaRunResult(
        summary=summary,
        results=tuple(results),
        summary_path=config.paths.logs / f"{artifact_version}_summary.json",
        candidates_manifest_path=candidates_manifest_path,
    )

