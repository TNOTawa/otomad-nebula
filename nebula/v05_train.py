"""v0.5-alpha 训练与验证（研究用；不注册 DB、不建地图）。

按 docs/plans/v0.5-alpha-temporal-model.md §4-§5 实现：

- 输入：data/v05_tokens/<dataset_version>/<work_id>.npz（task-33 数据层产物）；
  训练端不重建 token，只读落盘数据。
- 模型：优先 nebula.v05_model.V05AlphaModel（task-34 契约）；--smoke 使用本模块内
  的微型合成模型 + 合成 tokens，便于 CPU 快速回归训练/验证/产物链路。
- 训练：AdamW(lr=1e-3, wd=0.01)，batch=4 部作品；每个 batch 依次计算
  masked segment / AV mismatch / order recovery 三个自监督目标并反传。
- 验证：mask cosine/MSE 与常量 Full Mean 基线、AV accuracy/AUC、order accuracy、
  反转/打乱段序的时间敏感性、work vector 坍缩/有效秩/与 Full Mean 的 Top-5 Jaccard、
  duration R² 与 author balanced accuracy 泄漏检查。
- 产物：data/models/<version>/checkpoint.pt、logs/<version>_metrics.json、
  data/representations/transformer_alpha_v1/<sequence_version>/vectors.npy
  （仅研究产物，不写 representations 表、不建 map）。
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .config import Config
from .db import Catalog
from .errors import NebulaError
from .util import now_utc_iso

LOGGER = logging.getLogger(__name__)

DEFAULT_VERSION = "v05_alpha_v1"
DEFAULT_DATASET_VERSION = "v05_alpha_v1"
REPRESENTATION_NAME = "transformer_alpha_v1"
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 0.01
TRAIN_RATIO = 90 / 110  # 90 训练 / 20 验证（110 部 canonical 作品）
MASK_RATIO = 0.2
DEFAULT_MAX_SEGMENTS = 128
DEFAULT_ORDER_K = 3
DEFAULT_PATIENCE = 5
PAAV_DIM = 1024
CLIP_DIM = 512
DINO_DIM = 384
AUDIO_DIM = 24
VISUAL_EVENT_DIM = 12
V05_BINS = 16
ORDER_PERMUTATIONS = tuple(itertools.permutations(range(DEFAULT_ORDER_K)))
ORDER_CLASSES = len(ORDER_PERMUTATIONS)
MASK_WEIGHT = 1.0
AV_WEIGHT = 0.5
ORDER_WEIGHT = 0.5
_SEGMENT_KEYS = (
    "paav",
    "visual_clip",
    "visual_dino",
    "audio",
    "visual_event",
    "mask_visual",
    "mask_audio",
    "mask_visual_event",
    "segment_times",
    "segment_mask",
)
_ERROR_LIMIT = 2000


class V05TrainError(NebulaError):
    """v0.5-alpha 训练/验证的可预期错误（数据缺失、模型依赖未就绪等）。"""


# ---------------------------------------------------------------- 配置与划分


@dataclass(frozen=True)
class TrainConfig:
    """一次训练运行的超参与产物路径。"""

    version: str = DEFAULT_VERSION
    dataset_version: str = DEFAULT_DATASET_VERSION
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    seed: int = DEFAULT_SEED
    mask_ratio: float = MASK_RATIO
    max_segments: int = DEFAULT_MAX_SEGMENTS
    patience: int = DEFAULT_PATIENCE
    device: str = "auto"
    limit: int | None = None
    smoke: bool = False
    hidden: int | None = None
    local_layers: int | None = None
    temporal_layers: int | None = None
    dropout: float | None = None
    w_mask: float = MASK_WEIGHT
    w_av: float = AV_WEIGHT
    w_order: float = ORDER_WEIGHT

    def validate(self) -> "TrainConfig":
        if not str(self.version).strip():
            raise V05TrainError("--version 不能为空")
        if not str(self.dataset_version).strip():
            raise V05TrainError("--dataset-version 不能为空")
        if int(self.epochs) < 1:
            raise V05TrainError(f"epochs 必须 >= 1，收到 {self.epochs}")
        if int(self.batch_size) < 1:
            raise V05TrainError(f"batch_size 必须 >= 1，收到 {self.batch_size}")
        if float(self.lr) <= 0:
            raise V05TrainError(f"lr 必须 > 0，收到 {self.lr}")
        if float(self.weight_decay) < 0:
            raise V05TrainError(f"weight_decay 必须 >= 0，收到 {self.weight_decay}")
        if not 0.0 < float(self.mask_ratio) < 1.0:
            raise V05TrainError(f"mask_ratio 必须在 (0,1)，收到 {self.mask_ratio}")
        if int(self.max_segments) < 4:
            raise V05TrainError(f"max_segments 必须 >= 4，收到 {self.max_segments}")
        if int(self.patience) < 0:
            raise V05TrainError(f"patience 必须 >= 0，收到 {self.patience}")
        if self.device not in ("auto", "cpu", "cuda"):
            raise V05TrainError(f"device 必须是 auto/cpu/cuda，收到 {self.device!r}")
        if self.limit is not None and int(self.limit) < 1:
            raise V05TrainError(f"limit 必须 >= 1，收到 {self.limit}")
        if min(self.w_mask, self.w_av, self.w_order) < 0:
            raise V05TrainError("损失权重不能为负")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Split:
    """确定性 work_id 划分。"""

    train_ids: tuple[int, ...]
    val_ids: tuple[int, ...]
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "train_work_ids": [int(value) for value in self.train_ids],
            "val_work_ids": [int(value) for value in self.val_ids],
        }


def _hash_fraction(work_id: int, seed: int) -> float:
    digest = hashlib.sha256(f"v05-split:{seed}:{int(work_id)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def split_work_ids(
    work_ids: Sequence[int],
    *,
    seed: int = DEFAULT_SEED,
    val_ratio: float = 1.0 - TRAIN_RATIO,
) -> Split:
    """按 work_id 的确定性 hash 划分 train/val（默认约 90/20）。"""

    ids = sorted({int(value) for value in work_ids})
    if not ids:
        raise V05TrainError("没有可划分的作品")
    if not 0.0 < float(val_ratio) < 1.0:
        raise V05TrainError(f"val_ratio 必须在 (0,1)，收到 {val_ratio}")
    ranked = sorted(ids, key=lambda work_id: (_hash_fraction(work_id, seed), work_id))
    val_count = max(1, int(round(len(ids) * float(val_ratio))))
    val_count = min(val_count, len(ids) - 1) if len(ids) > 1 else 0
    val_ids = tuple(sorted(ranked[:val_count]))
    train_ids = tuple(sorted(ranked[val_count:]))
    if not train_ids:
        raise V05TrainError("划分后训练集为空，请增加作品数")
    return Split(train_ids=train_ids, val_ids=val_ids, seed=int(seed))


# ---------------------------------------------------------------- 合成数据（smoke / 未就绪依赖）

def synthetic_tokens(work_id: int, *, seed: int = DEFAULT_SEED) -> dict[str, np.ndarray]:
    """生成与 task-33 契约同 shape 的确定性合成 token（仅 smoke/测试用）。"""

    rng = np.random.default_rng([int(seed), int(work_id) % (2**31)])
    segments = 4 + int(work_id) % 9
    paav = rng.normal(size=(segments, 3, PAAV_DIM)).astype(np.float32)
    paav /= np.maximum(np.linalg.norm(paav, axis=-1, keepdims=True), 1e-9)
    visual_clip = rng.normal(size=(segments, V05_BINS, CLIP_DIM)).astype(np.float32)
    visual_clip /= np.maximum(
        np.linalg.norm(visual_clip, axis=-1, keepdims=True), 1e-9
    )
    visual_dino = rng.normal(size=(segments, V05_BINS, DINO_DIM)).astype(np.float32)
    visual_dino /= np.maximum(
        np.linalg.norm(visual_dino, axis=-1, keepdims=True), 1e-9
    )
    audio = rng.normal(size=(segments, V05_BINS, AUDIO_DIM)).astype(np.float32)
    visual_event = rng.normal(size=(segments, V05_BINS, VISUAL_EVENT_DIM)).astype(
        np.float32
    )
    mask_visual = (rng.random((segments, V05_BINS)) > 0.1).astype(np.uint8)
    mask_audio = (rng.random((segments, V05_BINS)) > 0.15).astype(np.uint8)
    mask_visual_event = (rng.random((segments, V05_BINS)) > 0.2).astype(np.uint8)
    visual_clip *= mask_visual[..., None]
    visual_dino *= mask_visual[..., None]
    audio *= mask_audio[..., None]
    visual_event *= mask_visual_event[..., None]
    segment_times = np.zeros((segments, 4), dtype=np.float32)
    segment_times[:, 0] = np.arange(segments, dtype=np.float32) * 8.0
    segment_times[:, 1] = 8.0
    segment_times[:, 2] = 8.0
    segment_times[:, 3] = 1.0 / segments
    segment_mask = np.ones((segments,), dtype=np.uint8)
    return {
        "paav": paav,
        "visual_clip": visual_clip,
        "visual_dino": visual_dino,
        "audio": audio,
        "visual_event": visual_event,
        "mask_visual": mask_visual,
        "mask_audio": mask_audio,
        "mask_visual_event": mask_visual_event,
        "segment_times": segment_times,
        "segment_mask": segment_mask,
        "duration_s": np.float32(segments * 8.0),
        "fps": np.float32(2.0),
    }


def _read_saved_tokens(path: Path) -> dict[str, np.ndarray]:
    """读取 task-33 落盘的 token npz（不依赖 v05_data 函数签名）。"""

    try:
        with np.load(path, allow_pickle=False) as data:
            payload = {key: np.asarray(data[key]) for key in data.files}
    except (OSError, ValueError) as exc:
        raise V05TrainError(f"读取 token 文件失败（{path}）: {exc}") from exc
    missing = [
        key
        for key in _SEGMENT_KEYS + ("duration_s",)
        if key not in payload
    ]
    if missing:
        raise V05TrainError(f"token 文件缺少字段 {missing}（{path}）")
    return payload


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


def load_dataset(
    config: Config,
    catalog: Catalog,
    cfg: TrainConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    """加载训练数据：works / tokens / duration / author 与数据元信息。"""

    works = _canonical_works(catalog)
    if cfg.limit is not None:
        works = works[: int(cfg.limit)]
    if not works:
        raise V05TrainError("没有 canonical + completed 作品可用于训练")
    work_ids = [int(work["id"]) for work in works]
    durations = {
        int(work["id"]): float(work["duration_s"] or 0.0) for work in works
    }
    authors = {
        int(work["id"]): str(work.get("author_name") or work.get("author_uid") or "")
        for work in works
    }
    tokens: dict[int, dict[str, np.ndarray]] = {}
    token_dir = config.data_path("v05_tokens") / cfg.dataset_version
    missing: list[int] = []
    synthetic = bool(cfg.smoke)
    if synthetic:
        for work_id in work_ids:
            tokens[work_id] = synthetic_tokens(work_id, seed=cfg.seed)
        if progress:
            progress(f"smoke 模式：生成 {len(tokens)} 部作品的合成 tokens")
    else:
        if not token_dir.is_dir():
            raise V05TrainError(
                f"token 数据集目录不存在: {token_dir}；"
                "请先运行 tools/build_v05_tokens.py 或加 --smoke"
            )
        try:
            from . import v05_data
        except ImportError:  # pragma: no cover - 依赖未就绪
            v05_data = None
        loader = getattr(v05_data, "load_work_tokens", None) if v05_data else None
        for work_id in work_ids:
            path = token_dir / f"{work_id:06d}.npz"
            if not path.is_file():
                missing.append(work_id)
                continue
            if callable(loader):
                try:
                    payload = loader(config, cfg.dataset_version, work_id)
                except NebulaError as exc:
                    raise V05TrainError(
                        f"读取 task-33 token 数据集失败（work_id={work_id}）: {exc}"
                    ) from exc
                if payload is None:
                    missing.append(work_id)
                    continue
                tokens[work_id] = dict(payload)
            else:
                tokens[work_id] = _read_saved_tokens(path)
        if progress:
            progress(
                f"读取 token 数据集 {cfg.dataset_version}: {len(tokens)} 部"
                f"（缺失 {len(missing)}）"
            )
        if not tokens:
            raise V05TrainError(
                f"数据集 {token_dir} 中没有可用 token 文件；"
                "请先运行 tools/build_v05_tokens.py 或加 --smoke"
            )
    # 截断超长作品（前 max_segments 段）
    truncated = 0
    for payload in tokens.values():
        segments = int(np.asarray(payload["paav"]).shape[0])
        if segments > cfg.max_segments:
            truncated += 1
            for key in _SEGMENT_KEYS:
                array = np.asarray(payload[key])
                payload[key] = array[: cfg.max_segments]
    manifest_summary: dict[str, Any] | None = None
    sequence_version = config.full_sequence_version
    try:
        from . import v05_data as _v05_data
    except ImportError:  # pragma: no cover - 依赖未就绪
        _v05_data = None
    manifest_loader = (
        getattr(_v05_data, "load_dataset_manifest", None) if _v05_data else None
    )
    if callable(manifest_loader) and not synthetic:
        try:
            manifest = manifest_loader(config, cfg.dataset_version)
        except NebulaError as exc:
            LOGGER.warning("读取数据集 manifest 失败: %s", exc)
            manifest = None
        if isinstance(manifest, dict):
            sequence_version = str(
                manifest.get("sequence_version") or sequence_version
            )
            manifest_summary = {
                key: manifest.get(key)
                for key in (
                    "version",
                    "created_at",
                    "sequence_version",
                    "clip_frame_version",
                    "dino_frame_version",
                    "bin_seconds",
                )
            }
            manifest_summary["counts"] = manifest.get("counts")
    info = {
        "dataset_version": cfg.dataset_version,
        "sequence_version": sequence_version,
        "source_manifest": manifest_summary,
        "synthetic": synthetic,
        "n_works": len(tokens),
        "n_missing_tokens": len(missing),
        "missing_work_ids": missing,
        "truncated_works": truncated,
        "work_ids": sorted(tokens),
        "durations": durations,
        "authors": authors,
    }
    return tokens, info


# ---------------------------------------------------------------- collate / 任务构造


def _torch_module() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - 环境缺 torch
        raise V05TrainError("v0.5-alpha 训练需要 torch（uv pip install -e '.[ml]'）") from exc
    return torch


def collate_works(
    works: Sequence[Mapping[str, np.ndarray]],
    *,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
) -> dict[str, Any]:
    """把变长 W 的作品 padding 到 batch 内最大 W（不超过 max_segments）。"""

    torch = _torch_module()
    if not works:
        raise V05TrainError("collate 需要至少一个作品")
    max_w = min(
        int(max_segments),
        max(int(np.asarray(work["paav"]).shape[0]) for work in works),
    )
    if max_w < 1:
        raise V05TrainError("collate 的 max_segments 非法")
    batch_size = len(works)
    shapes = {
        "paav": (batch_size, max_w, 3, PAAV_DIM),
        "visual_clip": (batch_size, max_w, V05_BINS, CLIP_DIM),
        "visual_dino": (batch_size, max_w, V05_BINS, DINO_DIM),
        "audio": (batch_size, max_w, V05_BINS, AUDIO_DIM),
        "visual_event": (batch_size, max_w, V05_BINS, VISUAL_EVENT_DIM),
        "mask_visual": (batch_size, max_w, V05_BINS),
        "mask_audio": (batch_size, max_w, V05_BINS),
        "mask_visual_event": (batch_size, max_w, V05_BINS),
        "segment_times": (batch_size, max_w, 4),
        "segment_mask": (batch_size, max_w),
    }
    buffers = {
        "paav": np.zeros(shapes["paav"], dtype=np.float32),
        "visual_clip": np.zeros(shapes["visual_clip"], dtype=np.float32),
        "visual_dino": np.zeros(shapes["visual_dino"], dtype=np.float32),
        "audio": np.zeros(shapes["audio"], dtype=np.float32),
        "visual_event": np.zeros(shapes["visual_event"], dtype=np.float32),
        "mask_visual": np.zeros(shapes["mask_visual"], dtype=np.uint8),
        "mask_audio": np.zeros(shapes["mask_audio"], dtype=np.uint8),
        "mask_visual_event": np.zeros(shapes["mask_visual_event"], dtype=np.uint8),
        "segment_times": np.zeros(shapes["segment_times"], dtype=np.float32),
        "segment_mask": np.zeros(shapes["segment_mask"], dtype=np.uint8),
    }
    for row, work in enumerate(works):
        segments = min(int(np.asarray(work["paav"]).shape[0]), max_w)
        for key in _SEGMENT_KEYS:
            array = np.asarray(work[key])
            buffers[key][row, :segments] = array[:segments]
    tensor_map = {
        "paav": buffers["paav"],
        "visual_clip": buffers["visual_clip"],
        "visual_dino": buffers["visual_dino"],
        "audio": buffers["audio"],
        "visual_event": buffers["visual_event"],
        "mask_visual": buffers["mask_visual"].astype(bool),
        "mask_audio": buffers["mask_audio"].astype(bool),
        "mask_visual_event": buffers["mask_visual_event"].astype(bool),
        "segment_times": buffers["segment_times"],
        "segment_mask": buffers["segment_mask"].astype(bool),
        "masked_segments": np.zeros((batch_size, max_w), dtype=bool),
    }
    return {
        key: torch.from_numpy(np.ascontiguousarray(value))
        for key, value in tensor_map.items()
    }


def make_mask(
    segment_mask: np.ndarray,
    *,
    rng: np.random.Generator,
    ratio: float = MASK_RATIO,
) -> np.ndarray:
    """按比例 mask 有效段（至少 1 段，且尽量留 1 段上下文）。"""

    mask = np.asarray(segment_mask, dtype=bool).copy()
    out = np.zeros_like(mask)
    for row in range(mask.shape[0]):
        valid = np.flatnonzero(mask[row])
        if valid.size == 0:
            continue
        count = max(1, int(round(valid.size * float(ratio))))
        count = min(count, valid.size if valid.size == 1 else valid.size - 1)
        chosen = rng.choice(valid, size=count, replace=False)
        out[row, chosen] = True
    return out


def _batched_works(
    tokens: Mapping[int, Mapping[str, np.ndarray]],
    work_ids: Sequence[int],
    *,
    batch_size: int,
) -> list[list[int]]:
    ids = list(work_ids)
    return [ids[index : index + int(batch_size)] for index in range(0, len(ids), int(batch_size))]


def _tensor_batch(
    tokens: Mapping[int, Mapping[str, np.ndarray]],
    work_ids: Sequence[int],
    cfg: TrainConfig,
) -> dict[str, Any]:
    works = [tokens[int(work_id)] for work_id in work_ids]
    return collate_works(works, max_segments=cfg.max_segments)


def make_order_batch(
    tokens: Mapping[int, Mapping[str, np.ndarray]],
    work_ids: Sequence[int],
    *,
    rng: np.random.Generator,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
) -> tuple[dict[str, Any] | None, Any | None]:
    """构造 order recovery batch：选中 3 个连续有效段并随机排列。

    与 task-34 模型契约一致：batch["order_indices"] (B,order_k) 指定排列后的段下标，
    模型用这些段的 temporal token 做 6 分类；目标类别是排列在 ORDER_PERMUTATIONS
    中的下标（即段下标相对起始位置的偏移排列）。
    没有可用作品时返回 (None, None)，训练端跳过该目标。
    """

    torch = _torch_module()
    chosen: list[Mapping[str, np.ndarray]] = []
    indices: list[list[int]] = []
    targets: list[int] = []
    for work_id in work_ids:
        work = tokens[int(work_id)]
        segments = int(np.asarray(work["paav"]).shape[0])
        if segments < DEFAULT_ORDER_K:
            continue
        segment_mask = np.asarray(work["segment_mask"], dtype=bool)
        starts = [
            start
            for start in range(segments - DEFAULT_ORDER_K + 1)
            if segment_mask[start : start + DEFAULT_ORDER_K].all()
        ]
        if not starts:
            continue
        start = int(starts[int(rng.integers(0, len(starts)))])
        class_index = int(rng.integers(0, ORDER_CLASSES))
        permutation = ORDER_PERMUTATIONS[class_index]
        chosen.append(work)
        indices.append([start + int(offset) for offset in permutation])
        targets.append(class_index)
    if not chosen:
        return None, None
    batch = collate_works(chosen, max_segments=max_segments)
    batch["order_indices"] = torch.tensor(indices, dtype=torch.long)
    return batch, torch.tensor(targets, dtype=torch.long)


def make_av_batch(
    batch: Mapping[str, Any],
    *,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], Any]:
    """构造 AV mismatch 输入：一半作品把音频替换为 batch 内其它作品的音频。

    返回 (新 batch, labels)，labels=1 表示视觉/音频匹配，0 表示错位。
    至少保留一个正例与一个负例（batch_size>=2 时）。
    """

    torch = _torch_module()
    result = dict(batch)
    labels = (rng.random(int(batch["segment_mask"].shape[0])) < 0.5).astype(np.float32)
    batch_size = labels.shape[0]
    if batch_size >= 2:
        if bool(labels.all()):
            labels[0] = 0.0
        if bool((labels == 0).all()):
            labels[0] = 1.0
    audio = batch["audio"].clone()
    mask_audio = batch["mask_audio"].clone()
    if batch_size >= 2:
        for row in range(batch_size):
            if labels[row] == 0.0:
                source = (row + 1) % batch_size
                audio[row] = batch["audio"][source]
                mask_audio[row] = batch["mask_audio"][source]
    else:
        # batch=1：负例按段平移到相邻段（至少 1 段错位）；正例保持真实音频
        shift = 1 if audio.shape[1] > 1 else 0
        if shift and labels[0] == 0.0:
            audio = torch.roll(audio, shifts=shift, dims=1)
            mask_audio = torch.roll(mask_audio, shifts=shift, dims=1)
        elif shift == 0:
            labels[0] = 0.0
    result["audio"] = audio
    result["mask_audio"] = mask_audio
    label_tensor = torch.from_numpy(labels)
    result["av_label"] = label_tensor
    return result, label_tensor


# ---------------------------------------------------------------- 模型（smoke 合成 / 真实 lazy import）


class SyntheticV05Model:
    """smoke 用微型模型：接口与 V05AlphaModel 输出契约一致，CPU 秒级。"""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        torch = _torch_module()
        if not hasattr(cls, "_built"):
            cls._built = cls._build(torch)
        return cls._built(*args, **kwargs)

    @staticmethod
    def _build(torch: Any) -> Any:
        class _Synthetic(torch.nn.Module):
            def __init__(self, hidden: int = 32, order_classes: int = ORDER_CLASSES) -> None:
                super().__init__()
                self.hidden = int(hidden)
                self.order_classes = int(order_classes)
                self.paav_proj = torch.nn.Linear(PAAV_DIM, hidden)
                self.visual_proj = torch.nn.Linear(CLIP_DIM, hidden)
                self.audio_proj = torch.nn.Linear(AUDIO_DIM, hidden)
                self.event_proj = torch.nn.Linear(VISUAL_EVENT_DIM, hidden)
                self.mask_token = torch.nn.Parameter(torch.zeros(hidden))
                self.mask_pred = torch.nn.Linear(hidden, PAAV_DIM)
                self.av_head = torch.nn.Linear(hidden * 2, 1)
                self.order_head = torch.nn.Linear(
                    hidden * DEFAULT_ORDER_K, ORDER_CLASSES
                )

            def _segment_tokens(self, batch: Mapping[str, Any]) -> Any:
                paav = batch["paav"]
                joint = torch.nn.functional.normalize(paav[:, :, 0, :], dim=-1)
                token = self.paav_proj(joint)
                mask = batch["segment_mask"].unsqueeze(-1)
                token = token * mask
                masked = batch["masked_segments"].unsqueeze(-1) & mask
                token = torch.where(masked, self.mask_token.expand_as(token), token)
                return token

            def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
                token = self._segment_tokens(batch)
                weights = batch["segment_mask"].to(token.dtype)
                denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
                segment_tokens = token
                work_vector = (token * weights.unsqueeze(-1)).sum(dim=1) / denom
                mask_pred = self.mask_pred(token)
                visual = batch["visual_clip"].mean(dim=(1, 2))
                audio = batch["audio"].mean(dim=(1, 2))
                visual_token = self.visual_proj(visual)
                audio_token = self.audio_proj(audio)
                av_logits = self.av_head(
                    torch.cat([visual_token, audio_token], dim=-1)
                ).squeeze(-1)
                order_indices = batch.get("order_indices")
                if isinstance(order_indices, torch.Tensor):
                    indices = order_indices.long().clamp(0, token.shape[1] - 1)
                    gather_index = indices.unsqueeze(-1).expand(
                        token.shape[0], DEFAULT_ORDER_K, self.hidden
                    )
                    triplet = torch.gather(token, 1, gather_index)
                else:
                    padded = torch.zeros(
                        token.shape[0],
                        DEFAULT_ORDER_K,
                        self.hidden,
                        dtype=token.dtype,
                        device=token.device,
                    )
                    take = min(token.shape[1], DEFAULT_ORDER_K)
                    if take:
                        padded[:, :take] = token[:, :take]
                    triplet = padded
                order_logits = self.order_head(triplet.reshape(token.shape[0], -1))
                return {
                    "work_vector": work_vector,
                    "segment_tokens": segment_tokens,
                    "mask_pred": mask_pred,
                    "av_logits": av_logits,
                    "order_logits": order_logits,
                }

        return _Synthetic


def _resolve_model(cfg: TrainConfig) -> tuple[Any, str, dict[str, Any]]:
    """返回 (model, model_name, model_info)；smoke 用合成小模型。"""

    if cfg.smoke:
        hidden = int(cfg.hidden or 32)
        model = SyntheticV05Model(hidden=hidden)
        params = sum(parameter.numel() for parameter in model.parameters())
        return (
            model,
            "synthetic-v05-smoke",
            {"name": "synthetic-v05-smoke", "hidden": hidden, "params": int(params)},
        )
    try:
        from . import v05_model
    except ImportError as exc:
        raise V05TrainError(
            "nebula.v05_model（task-34 模型契约）尚未就绪；"
            "当前只能加 --smoke 用合成小模型跑通链路"
        ) from exc
    model_cls = getattr(v05_model, "V05AlphaModel", None)
    config_cls = getattr(v05_model, "V05AlphaConfig", None)
    if model_cls is None or config_cls is None:
        raise V05TrainError("nebula.v05_model 缺少 V05AlphaModel/V05AlphaConfig")
    model_config = config_cls()
    overrides = {
        "hidden": cfg.hidden,
        "local_layers": cfg.local_layers,
        "temporal_layers": cfg.temporal_layers,
        "dropout": cfg.dropout,
        "max_segments": cfg.max_segments,
    }
    applied = {key: value for key, value in overrides.items() if value is not None}
    if applied:
        model_config = replace(model_config, **applied)
    model = model_cls(model_config)
    params = sum(parameter.numel() for parameter in model.parameters())
    info = {"name": model_cls.__name__, "params": int(params)}
    for key in (
        "hidden",
        "heads",
        "local_layers",
        "temporal_layers",
        "dropout",
        "max_segments",
        "order_k",
    ):
        if hasattr(model_config, key):
            info[key] = getattr(model_config, key)
    return model, model_cls.__name__, info


@dataclass
class _TorchBundle:
    torch: Any
    device: Any


def _torch_bundle(cfg: TrainConfig) -> _TorchBundle:
    torch = _torch_module()
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    return _TorchBundle(torch=torch, device=device)


def _seed_everything(torch: Any, seed: int) -> None:
    import random

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _forward(model: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    torch = _torch_module()
    if "masked_segments" not in batch:
        batch = dict(batch)
        batch["masked_segments"] = torch.zeros_like(
            batch["segment_mask"], dtype=torch.bool
        )
    outputs = model(batch)
    if not isinstance(outputs, Mapping):
        raise V05TrainError(f"模型 forward 必须返回 dict，收到 {type(outputs).__name__}")
    missing = [
        key
        for key in ("work_vector", "segment_tokens", "mask_pred", "av_logits", "order_logits")
        if key not in outputs
    ]
    if missing:
        raise V05TrainError(f"模型输出缺少字段: {missing}")
    return dict(outputs)


# ---------------------------------------------------------------- 损失与指标


def _mask_losses(outputs: Mapping[str, Any], batch: Mapping[str, Any], torch: Any) -> tuple[Any, Any, Any]:
    """返回 (loss, cosine 均值, mse)；只在被 mask 的有效段上计算。"""

    target = batch["paav"][:, :, 0, :]
    prediction = outputs["mask_pred"]
    masked = batch["masked_segments"] & batch["segment_mask"]
    if int(masked.sum().item()) == 0:
        zero = torch.zeros((), dtype=prediction.dtype, device=prediction.device)
        return zero, zero, zero
    pred = prediction[masked].reshape(-1, prediction.shape[-1])
    tgt = target[masked].reshape(-1, target.shape[-1])
    pred_norm = torch.nn.functional.normalize(pred, dim=-1)
    tgt_norm = torch.nn.functional.normalize(tgt, dim=-1)
    cosine = (pred_norm * tgt_norm).sum(dim=-1)
    mse = ((pred_norm - tgt_norm) ** 2).mean(dim=-1)
    loss = mse + (1.0 - cosine)
    return loss.mean(), cosine.mean(), mse.mean()


def mask_prediction_loss(
    outputs: Mapping[str, Any], batch: Mapping[str, Any]
) -> tuple[Any, dict[str, float]]:
    torch = _torch_module()
    loss, cosine, mse = _mask_losses(outputs, batch, torch)
    return loss, {"cosine": float(cosine.detach().cpu()), "mse": float(mse.detach().cpu())}


def av_mismatch_loss(outputs: Mapping[str, Any], labels: Any) -> Any:
    """AV mismatch 损失；兼容模型输出 (B,) logits 或 (B,2) 二分类 logits。"""

    torch = _torch_module()
    logits = outputs["av_logits"]
    target = labels.reshape(-1).long()
    if logits.dim() == 2 and int(logits.shape[-1]) == 2:
        if logits.shape[0] != target.shape[0]:
            raise V05TrainError(
                f"AV logits 数量 {logits.shape[0]} 与标签 {target.shape[0]} 不一致"
            )
        return torch.nn.functional.cross_entropy(logits, target)
    flat = logits.reshape(-1)
    if flat.shape[0] != target.shape[0]:
        raise V05TrainError(
            f"AV logits 数量 {flat.shape[0]} 与标签 {target.shape[0]} 不一致"
        )
    return torch.nn.functional.binary_cross_entropy_with_logits(
        flat, target.to(flat.dtype)
    )


def _av_probabilities(logits: Any) -> np.ndarray:
    """把 (B,) 或 (B,2) 的 AV logits 转成"匹配"概率。"""

    array = logits.detach().cpu().numpy()
    if array.ndim == 2 and array.shape[-1] == 2:
        shifted = array - array.max(axis=-1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities = probabilities / probabilities.sum(axis=-1, keepdims=True)
        return probabilities[:, 1]
    return 1.0 / (1.0 + np.exp(-array.reshape(-1)))


def order_recovery_loss(outputs: Mapping[str, Any], targets: Any) -> Any:
    torch = _torch_module()
    logits = outputs["order_logits"]
    target = targets.reshape(-1).long()
    if logits.shape[0] != target.shape[0]:
        raise V05TrainError(
            f"order logits 数量 {logits.shape[0]} 与标签 {target.shape[0]} 不一致"
        )
    return torch.nn.functional.cross_entropy(logits, target)


def _zero_scalar(reference: Any) -> Any:
    return reference.sum() * 0.0


def compute_objective_losses(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    *,
    w_mask: float = MASK_WEIGHT,
    w_av: float = AV_WEIGHT,
    w_order: float = ORDER_WEIGHT,
) -> dict[str, Any]:
    """计算总损失：优先使用 task-34 的 v05_losses，缺失/不兼容时回退本地三项损失。

    返回 {"total", "mask", "av", "order", "source", "mode"}；total 已按权重相加。
    """

    torch = _torch_module()
    try:
        from . import v05_model
    except ImportError:  # pragma: no cover - 依赖未就绪
        v05_model = None
    losses_fn = getattr(v05_model, "v05_losses", None) if v05_model is not None else None
    if callable(losses_fn):
        try:
            raw = losses_fn(
                outputs, batch, w_mask=w_mask, w_av=w_av, w_order=w_order
            )
            total = raw.get("total", raw.get("loss"))
            if not torch.is_tensor(total):
                raise V05TrainError("v05_losses 未返回 tensor total/loss")
            return {
                "total": total,
                "mask": raw.get("mask_loss", _zero_scalar(total)),
                "av": raw.get("av_loss", _zero_scalar(total)),
                "order": raw.get("order_loss", _zero_scalar(total)),
                "source": "v05_losses",
            }
        except (TypeError, ValueError, KeyError) as exc:
            LOGGER.warning("v05_losses 调用失败，回退本地损失计算: %s", exc)

    reference = outputs["mask_pred"]
    zero = _zero_scalar(reference)
    mask_loss = zero
    av_loss = zero
    order_loss = zero
    if "masked_segments" in batch and "segment_mask" in batch:
        mask_loss, _ = mask_prediction_loss(outputs, batch)
    if "av_label" in batch:
        av_loss = av_mismatch_loss(outputs, batch["av_label"])
    if "order_label" in batch:
        order_loss = order_recovery_loss(outputs, batch["order_label"])
    total = float(w_mask) * mask_loss + float(w_av) * av_loss + float(w_order) * order_loss
    return {
        "total": total,
        "mask": mask_loss,
        "av": av_loss,
        "order": order_loss,
        "source": "local",
    }


def compute_auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """秩和法 AUC（无 sklearn 依赖，处理并列取平均秩）。"""

    scores_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores_array.shape != labels_array.shape or scores_array.size == 0:
        return None
    positives = int(labels_array.sum())
    negatives = int(labels_array.size - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores_array, kind="mergesort")
    ranks = np.empty(scores_array.size, dtype=np.float64)
    sorted_scores = scores_array[order]
    index = 0
    while index < sorted_scores.size:
        end = index + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[index]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        ranks[order[index:end]] = average_rank
        index = end
    positive_rank_sum = float(ranks[labels_array == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def effective_rank(vectors: np.ndarray) -> float | None:
    """参与率有效秩 (sum s)^2 / sum s^2。"""

    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return None
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    denominator = float(np.sum(singular**2))
    if denominator <= 1e-12:
        return 0.0
    return float(np.sum(singular) ** 2 / denominator)


def pairwise_cosine_stats(vectors: np.ndarray) -> dict[str, float | None]:
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return {"mean": None, "std": None, "min": None, "max": None}
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-9)
    similarity = normalized @ normalized.T
    mask = ~np.eye(similarity.shape[0], dtype=bool)
    values = similarity[mask]
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def top_k_neighbors(matrix: np.ndarray, k: int = 5) -> list[set[int]]:
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-9)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    neighbors: list[set[int]] = []
    for row in similarity:
        count = min(int(k), similarity.shape[0] - 1)
        nearest = np.argsort(-row, kind="mergesort")[:count]
        neighbors.append({int(index) for index in nearest})
    return neighbors


def mean_top_k_jaccard(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    *,
    k: int = 5,
) -> float | None:
    if matrix_a.shape[0] != matrix_b.shape[0] or matrix_a.shape[0] < k + 1:
        return None
    neighbors_a = top_k_neighbors(matrix_a, k=k)
    neighbors_b = top_k_neighbors(matrix_b, k=k)
    scores = []
    for left, right in zip(neighbors_a, neighbors_b, strict=True):
        union = left | right
        if not union:
            continue
        scores.append(len(left & right) / len(union))
    return float(np.mean(scores)) if scores else None


def duration_r2(vectors: np.ndarray, durations: Sequence[float]) -> float | None:
    """线性回归（含截距 + 轻量 ridge）拟合 duration 的 R²（样本内泄漏上界）。"""

    matrix = np.asarray(vectors, dtype=np.float64)
    targets = np.asarray(durations, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != targets.size or targets.size < 3:
        return None
    variance = float(np.var(targets))
    if variance <= 1e-12:
        return None
    features = np.concatenate(
        [matrix, np.ones((matrix.shape[0], 1), dtype=np.float64)], axis=1
    )
    gram = features.T @ features
    ridge = 1e-3 * np.eye(gram.shape[0], dtype=np.float64)
    ridge[-1, -1] = 0.0
    try:
        weights = np.linalg.solve(gram + ridge, features.T @ targets)
    except np.linalg.LinAlgError:
        return None
    prediction = features @ weights
    residual = float(np.sum((targets - prediction) ** 2))
    return float(1.0 - residual / (variance * targets.size))


def author_balanced_accuracy(
    vectors: np.ndarray,
    authors: Sequence[str],
    *,
    min_support: int = 2,
) -> float | None:
    """最近邻（留一）预测作者，按作者类别平衡的平均召回。"""

    matrix = np.asarray(vectors, dtype=np.float64)
    labels = [str(label) for label in authors]
    if matrix.ndim != 2 or matrix.shape[0] != len(labels) or matrix.shape[0] < 2:
        return None
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-9)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    predictions = []
    for row in range(matrix.shape[0]):
        best = int(np.argmax(similarity[row]))
        predictions.append(labels[best])
    recalls: dict[str, list[int]] = {}
    for true_label, predicted in zip(labels, predictions, strict=True):
        recalls.setdefault(true_label, []).append(int(predicted == true_label))
    supported = [
        float(np.mean(values))
        for values in recalls.values()
        if len(values) >= int(min_support)
    ]
    if not supported:
        return None
    return float(np.mean(supported))


def temporal_transforms(
    work: Mapping[str, np.ndarray],
    *,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """返回 (段序反转, 段序随机打乱) 两份作品 token。"""

    segments = int(np.asarray(work["paav"]).shape[0])
    reversed_work: dict[str, np.ndarray] = {}
    shuffled_work: dict[str, np.ndarray] = {}
    permutation = rng.permutation(segments)
    for key in _SEGMENT_KEYS:
        array = np.asarray(work[key])
        reversed_work[key] = array[::-1].copy()
        shuffled_work[key] = array[permutation].copy()
    for key in ("duration_s", "fps"):
        if key in work:
            reversed_work[key] = np.asarray(work[key])
            shuffled_work[key] = np.asarray(work[key])
    return reversed_work, shuffled_work


def _vicreg_regularizer(work_vector: Any) -> tuple[Any, Any]:
    """VICReg 的 variance/covariance 项，防止 work_vector 坍缩成常数。"""

    torch = _torch_module()
    z = work_vector.float()
    if z.ndim != 2 or z.shape[0] < 2:
        zero = z.sum() * 0.0
        return zero, zero
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-6)
    variance = torch.relu(1.0 - std).mean()
    centered = z - z.mean(dim=0, keepdim=True)
    covariance = (centered.T @ centered) / max(1, z.shape[0] - 1)
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    covariance_loss = (off_diagonal**2).sum() / max(1, z.shape[1])
    return variance, covariance_loss


def _set_mask_mean(model: Any, tokens: Mapping[int, Mapping[str, np.ndarray]], work_ids: Sequence[int]) -> np.ndarray | None:
    """用训练集 PE-AV joint 段向量的（L2 后）均值写入模型 mask_mean buffer。"""

    if not hasattr(model, "mask_mean"):
        return None
    segments = []
    for work_id in work_ids:
        work = tokens[int(work_id)]
        paav = np.asarray(work["paav"], dtype=np.float32)
        mask = np.asarray(work["segment_mask"], dtype=bool)
        if paav.ndim != 3 or paav.shape[1] < 1:
            continue
        joint = paav[:, 0, :][mask]
        if joint.size:
            segments.append(joint)
    if not segments:
        return None
    matrix = np.concatenate(segments, axis=0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-9)
    mean = normalized.mean(axis=0)
    mean = mean / max(float(np.linalg.norm(mean)), 1e-9)
    torch = _torch_module()
    with torch.no_grad():
        model.mask_mean.copy_(torch.from_numpy(mean.astype(np.float32)).to(model.mask_mean.device))
    return mean


# ---------------------------------------------------------------- 训练 / 验证循环


def _evaluate_mask(
    torch: Any,
    device: Any,
    model: Any,
    tokens: Mapping[int, Mapping[str, np.ndarray]],
    work_ids: Sequence[int],
    cfg: TrainConfig,
    *,
    rng: np.random.Generator,
) -> dict[str, float | None]:
    model.eval()
    cosines: list[float] = []
    mses: list[float] = []
    baseline_cosines: list[float] = []
    baseline_mses: list[float] = []
    all_targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch_ids in _batched_works(tokens, work_ids, batch_size=cfg.batch_size):
            batch = _tensor_batch(tokens, batch_ids, cfg)
            mask = make_mask(
                batch["segment_mask"].cpu().numpy(), rng=rng, ratio=cfg.mask_ratio
            )
            batch["masked_segments"] = torch.from_numpy(mask)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = _forward(model, batch)
            target = batch["paav"][:, :, 0, :]
            masked = batch["masked_segments"] & batch["segment_mask"]
            pred = outputs["mask_pred"][masked].reshape(-1, PAAV_DIM)
            tgt = target[masked].reshape(-1, PAAV_DIM)
            pred_norm = torch.nn.functional.normalize(pred, dim=-1)
            tgt_norm = torch.nn.functional.normalize(tgt, dim=-1)
            cosines.append(float((pred_norm * tgt_norm).sum(dim=-1).mean().cpu()))
            mses.append(
                float(((pred_norm - tgt_norm) ** 2).mean(dim=-1).mean().cpu())
            )
            all_targets.append(tgt_norm.cpu().numpy())
    target_matrix = (
        np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, PAAV_DIM))
    )
    if target_matrix.shape[0]:
        mean_vector = target_matrix.mean(axis=0)
        norm = np.linalg.norm(mean_vector)
        mean_vector = mean_vector / norm if norm > 1e-9 else mean_vector
        similarity = target_matrix @ mean_vector
        baseline_cosines.append(float(similarity.mean()))
        baseline_mses.append(float(((target_matrix - mean_vector) ** 2).mean()))
    return {
        "cosine": float(np.mean(cosines)) if cosines else None,
        "mse": float(np.mean(mses)) if mses else None,
        "full_mean_baseline_cosine": (
            float(np.mean(baseline_cosines)) if baseline_cosines else None
        ),
        "full_mean_baseline_mse": (
            float(np.mean(baseline_mses)) if baseline_mses else None
        ),
    }


def _evaluate_av(
    torch: Any,
    device: Any,
    model: Any,
    tokens: Mapping[int, Mapping[str, np.ndarray]],
    work_ids: Sequence[int],
    cfg: TrainConfig,
    *,
    rng: np.random.Generator,
) -> dict[str, float | None]:
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch_ids in _batched_works(tokens, work_ids, batch_size=cfg.batch_size):
            batch = _tensor_batch(tokens, batch_ids, cfg)
            av_batch, av_labels = make_av_batch(batch, rng=rng)
            av_batch = {key: value.to(device) for key, value in av_batch.items()}
            outputs = _forward(model, av_batch)
            probabilities = _av_probabilities(outputs["av_logits"])
            scores.extend(float(value) for value in probabilities)
            labels.extend(int(value) for value in av_labels.detach().cpu().numpy())
    if not scores:
        return {"accuracy": None, "auc": None, "negative_accuracy": None}
    scores_array = np.asarray(scores)
    labels_array = np.asarray(labels)
    predicted = (scores_array >= 0.5).astype(np.int64)
    negative_mask = labels_array == 0
    return {
        "accuracy": float((predicted == labels_array).mean()),
        "auc": compute_auc(scores_array, labels_array),
        "negative_accuracy": (
            float((predicted[negative_mask] == 0).mean())
            if negative_mask.any()
            else None
        ),
    }


def _evaluate_order(
    torch: Any,
    device: Any,
    model: Any,
    tokens: Mapping[int, Mapping[str, np.ndarray]],
    work_ids: Sequence[int],
    cfg: TrainConfig,
    *,
    rng: np.random.Generator,
) -> dict[str, float | None]:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_ids in _batched_works(tokens, work_ids, batch_size=cfg.batch_size):
            batch, targets = make_order_batch(
                tokens, batch_ids, rng=rng, max_segments=cfg.max_segments
            )
            if batch is None or targets is None:
                continue
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = _forward(model, batch)
            prediction = outputs["order_logits"].argmax(dim=-1).detach().cpu().numpy()
            correct += int((prediction == targets.numpy()).sum())
            total += int(targets.numel())
    return {
        "accuracy": float(correct / total) if total else None,
        "chance": 1.0 / ORDER_CLASSES,
        "n_samples": total,
    }


def _collect_work_vectors(
    torch: Any,
    device: Any,
    model: Any,
    tokens: Mapping[int, Mapping[str, np.ndarray]],
    work_ids: Sequence[int],
    cfg: TrainConfig,
) -> tuple[list[int], np.ndarray]:
    model.eval()
    ids: list[int] = []
    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for batch_ids in _batched_works(tokens, work_ids, batch_size=cfg.batch_size):
            batch = _tensor_batch(tokens, batch_ids, cfg)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = _forward(model, batch)
            work_vector = outputs["work_vector"].detach().cpu().numpy()
            for row, work_id in enumerate(batch_ids):
                ids.append(int(work_id))
                vectors.append(work_vector[row].astype(np.float32))
    return ids, np.stack(vectors) if vectors else np.zeros((0, 1), dtype=np.float32)


def _full_mean_jaccard(
    catalog: Catalog,
    config: Config,
    ids: Sequence[int],
    matrix: np.ndarray,
    *,
    k: int = 5,
) -> float | None:
    from . import representations

    rep = catalog.get_representation(
        name="full_mean_v1", sequence_version=config.full_sequence_version
    )
    if rep is None:
        rep = catalog.get_representation(name="full_mean_v1")
    if rep is None:
        return None
    try:
        full_ids, full_matrix = representations.load_representation_matrix(
            catalog, config, rep
        )
    except NebulaError as exc:
        LOGGER.warning("读取 full_mean_v1 失败，跳过 Top-5 Jaccard: %s", exc)
        return None
    index_full = {work_id: row for row, work_id in enumerate(full_ids)}
    rows = [(row, index_full[work_id]) for row, work_id in enumerate(ids) if work_id in index_full]
    if len(rows) < k + 1:
        return None
    ours = np.stack([matrix[row] for row, _ in rows])
    theirs = np.stack([full_matrix[index] for _, index in rows])
    return mean_top_k_jaccard(ours, theirs, k=k)


@dataclass
class TrainResult:
    metrics: dict[str, Any]
    checkpoint_path: Path
    metrics_path: Path
    vectors_path: Path
    work_ids_path: Path
    summary_text: str


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file_handle:
            json.dump(dict(payload), file_handle, ensure_ascii=False, indent=2, sort_keys=True)
            file_handle.write("\n")
        os.replace(tmp_path, target)
    finally:
        tmp_path.unlink(missing_ok=True)
    return target


def _atomic_save_npy(path: Path, array: np.ndarray) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "wb") as file_handle:
            np.save(file_handle, np.asarray(array), allow_pickle=False)
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
        with os.fdopen(handle, "wb") as file_handle:
            torch.save(dict(payload), file_handle)
        os.replace(tmp_path, target)
    finally:
        tmp_path.unlink(missing_ok=True)
    return target


def resolve_artifact_version(cfg: TrainConfig) -> str:
    """smoke 默认版本自动加 _smoke 后缀，避免覆盖真实 v05 产物。"""

    if cfg.smoke and cfg.version == DEFAULT_VERSION:
        return f"{cfg.version}_smoke"
    return cfg.version


def run_training(
    config: Config,
    catalog: Catalog,
    cfg: TrainConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> TrainResult:
    """执行一次 v0.5-alpha 训练 + 验证 + 产物落盘（研究产物，不写 DB）。"""

    cfg = cfg.validate()
    artifact_version = resolve_artifact_version(cfg)
    if artifact_version != cfg.version:
        LOGGER.warning(
            "smoke 运行使用产物版本 %s（避免覆盖真实 %s 产物）",
            artifact_version,
            cfg.version,
        )
        cfg = replace(cfg, version=artifact_version)
    started = time.perf_counter()
    bundle = _torch_bundle(cfg)
    torch = bundle.torch
    device = bundle.device
    _seed_everything(torch, cfg.seed)

    tokens, dataset_info = load_dataset(config, catalog, cfg, progress=progress)
    work_ids = sorted(tokens)
    split = split_work_ids(work_ids, seed=cfg.seed)
    train_tokens = split.train_ids
    val_tokens = split.val_ids
    if not val_tokens:
        raise V05TrainError("验证集为空；请增加作品数量或调整 split seed")

    model, model_name, model_info = _resolve_model(cfg)
    model = model.to(device)
    mask_mean = _set_mask_mean(model, tokens, train_tokens)
    if mask_mean is not None:
        LOGGER.info("mask_mean 基线已写入模型（%d 维）", mask_mean.size)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )

    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_val = -math.inf
    patience_counter = 0
    early_stopped = False

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        rng = np.random.default_rng(int(cfg.seed) + epoch)
        train_order = list(train_tokens)
        rng.shuffle(train_order)
        totals = {"total": 0.0, "mask": 0.0, "av": 0.0, "order": 0.0}
        batch_count = 0
        for batch_ids in _batched_works(tokens, train_order, batch_size=cfg.batch_size):
            batch = _tensor_batch(tokens, batch_ids, cfg)
            batch = {key: value.to(device) for key, value in batch.items()}
            # 1) masked segment prediction
            mask = make_mask(
                batch["segment_mask"].detach().cpu().numpy(),
                rng=rng,
                ratio=cfg.mask_ratio,
            )
            masked_batch = dict(batch)
            masked_batch["masked_segments"] = torch.from_numpy(mask).to(device)
            mask_outputs = _forward(model, masked_batch)
            mask_losses = compute_objective_losses(
                mask_outputs,
                masked_batch,
                w_mask=cfg.w_mask,
                w_av=0.0,
                w_order=0.0,
            )
            # 2) AV mismatch
            av_batch, _ = make_av_batch(
                {key: value for key, value in batch.items() if key != "masked_segments"},
                rng=rng,
            )
            av_batch = {key: value.to(device) for key, value in av_batch.items()}
            av_losses = compute_objective_losses(
                _forward(model, av_batch),
                av_batch,
                w_mask=0.0,
                w_av=cfg.w_av,
                w_order=0.0,
            )
            # 3) order recovery（order_indices 指向 3 个连续有效段，模型做 6 分类）
            order_loss = torch.zeros((), device=device)
            order_batch, order_targets = make_order_batch(
                tokens, batch_ids, rng=rng, max_segments=cfg.max_segments
            )
            if order_batch is not None and order_targets is not None:
                order_batch = {
                    key: value.to(device) for key, value in order_batch.items()
                }
                order_batch["order_label"] = order_targets.to(device)
                order_losses = compute_objective_losses(
                    _forward(model, order_batch),
                    order_batch,
                    w_mask=0.0,
                    w_av=0.0,
                    w_order=cfg.w_order,
                )
                order_loss = order_losses["order"]
                total_order = order_losses["total"]
            else:
                total_order = order_loss
            # 4) VICReg 正则：work_vector 不能坍缩成常数
            variance_loss, covariance_loss = _vicreg_regularizer(
                mask_outputs["work_vector"]
            )
            total_loss = (
                mask_losses["total"]
                + av_losses["total"]
                + total_order
                + 0.5 * variance_loss
                + 0.05 * covariance_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            totals["total"] += float(total_loss.detach().cpu())
            totals["mask"] += float(mask_losses["mask"].detach().cpu())
            totals["av"] += float(av_losses["av"].detach().cpu())
            totals["order"] += float(order_loss.detach().cpu())
            batch_count += 1

        val_rng = np.random.default_rng(int(cfg.seed) + 10_000 + epoch)
        mask_metrics = _evaluate_mask(
            torch, device, model, tokens, val_tokens, cfg, rng=val_rng
        )
        av_metrics = _evaluate_av(
            torch, device, model, tokens, val_tokens, cfg, rng=val_rng
        )
        record = {
            "epoch": epoch,
            "train_loss": totals["total"] / max(batch_count, 1),
            "train_mask_loss": totals["mask"] / max(batch_count, 1),
            "train_av_loss": totals["av"] / max(batch_count, 1),
            "train_order_loss": totals["order"] / max(batch_count, 1),
            "val_mask_cosine": mask_metrics["cosine"],
            "val_mask_mse": mask_metrics["mse"],
            "val_av_accuracy": av_metrics["accuracy"],
        }
        history.append(record)
        if progress:
            progress(
                f"epoch {epoch}/{cfg.epochs}: loss={record['train_loss']:.4f} "
                f"mask_cos={record['val_mask_cosine']} av_acc={record['val_av_accuracy']}"
            )
        current = record["val_mask_cosine"]
        improved = current is not None and current > best_val + 1e-4
        if improved:
            best_val = float(current)
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if cfg.patience and patience_counter >= int(cfg.patience):
                early_stopped = True
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    eval_rng = np.random.default_rng(int(cfg.seed) + 99_999)
    normal_mask = _evaluate_mask(
        torch, device, model, tokens, val_tokens, cfg, rng=eval_rng
    )
    av_metrics = _evaluate_av(
        torch, device, model, tokens, val_tokens, cfg, rng=eval_rng
    )
    order_metrics = _evaluate_order(
        torch, device, model, tokens, val_tokens, cfg, rng=eval_rng
    )
    normal_av_negative = av_metrics["negative_accuracy"]

    # 时间敏感性：反转 / 随机打乱段序后 mask cosine 变化
    reversed_tokens = {
        work_id: temporal_transforms(tokens[work_id], rng=eval_rng)[0]
        for work_id in val_tokens
    }
    shuffled_tokens = {
        work_id: temporal_transforms(tokens[work_id], rng=eval_rng)[1]
        for work_id in val_tokens
    }
    reversed_mask = _evaluate_mask(
        torch, device, model, reversed_tokens, list(val_tokens), cfg, rng=eval_rng
    )
    shuffled_mask = _evaluate_mask(
        torch, device, model, shuffled_tokens, list(val_tokens), cfg, rng=eval_rng
    )
    normal_cosine = normal_mask["cosine"]
    reversed_cosine = reversed_mask["cosine"]
    shuffled_cosine = shuffled_mask["cosine"]
    reverse_drop = (
        float(normal_cosine - reversed_cosine)
        if normal_cosine is not None and reversed_cosine is not None
        else None
    )
    shuffle_drop = (
        float(normal_cosine - shuffled_cosine)
        if normal_cosine is not None and shuffled_cosine is not None
        else None
    )

    # work vector 质量 + 泄漏检查（全部可用作品）
    all_ids, all_vectors = _collect_work_vectors(
        torch, device, model, tokens, work_ids, cfg
    )
    durations = [float(dataset_info["durations"].get(work_id, 0.0)) for work_id in all_ids]
    authors = [str(dataset_info["authors"].get(work_id, "")) for work_id in all_ids]
    collapse = pairwise_cosine_stats(all_vectors)
    # 时间敏感性（work vector 层）：正常 vs 反转/打乱段序的输出余弦
    reverse_vec_ids, reverse_vectors = _collect_work_vectors(
        torch, device, model, reversed_tokens, list(val_tokens), cfg
    )
    shuffle_vec_ids, shuffle_vectors = _collect_work_vectors(
        torch, device, model, shuffled_tokens, list(val_tokens), cfg
    )
    normal_vec_ids, normal_vectors = _collect_work_vectors(
        torch, device, model, {wid: tokens[wid] for wid in val_tokens}, list(val_tokens), cfg
    )
    order_sensitivity: dict[str, float | None] = {
        "normal_reversed_work_cosine": None,
        "normal_shuffled_work_cosine": None,
    }
    if normal_vec_ids == reverse_vec_ids and len(normal_vectors) == len(reverse_vectors):
        order_sensitivity["normal_reversed_work_cosine"] = float(
            np.mean(
                np.sum(normal_vectors * reverse_vectors, axis=1)
                / np.maximum(
                    np.linalg.norm(normal_vectors, axis=1)
                    * np.linalg.norm(reverse_vectors, axis=1),
                    1e-9,
                )
            )
        )
    if normal_vec_ids == shuffle_vec_ids and len(normal_vectors) == len(shuffle_vectors):
        order_sensitivity["normal_shuffled_work_cosine"] = float(
            np.mean(
                np.sum(normal_vectors * shuffle_vectors, axis=1)
                / np.maximum(
                    np.linalg.norm(normal_vectors, axis=1)
                    * np.linalg.norm(shuffle_vectors, axis=1),
                    1e-9,
                )
            )
        )
    representation_metrics = {
        "pairwise_cosine_mean": collapse["mean"],
        "pairwise_cosine_std": collapse["std"],
        "pairwise_cosine_min": collapse["min"],
        "pairwise_cosine_max": collapse["max"],
        "effective_rank": effective_rank(all_vectors),
        "top5_jaccard_full_mean": _full_mean_jaccard(
            catalog, config, all_ids, all_vectors, k=5
        ),
        "duration_r2": duration_r2(all_vectors, durations),
        "author_balanced_accuracy": author_balanced_accuracy(all_vectors, authors),
        "n_works": len(all_ids),
        "temporal_order_sensitivity": order_sensitivity,
    }

    # 研究产物（不注册 DB、不建地图）
    sequence_version = str(dataset_info["sequence_version"])
    version_dir = config.data_path("models") / cfg.version
    checkpoint_path = version_dir / "checkpoint.pt"
    metrics_path = config.paths.logs / f"{cfg.version}_metrics.json"
    split_path = config.paths.logs / f"{cfg.version}_split.json"
    _atomic_json(
        split_path,
        {
            "version": cfg.version,
            "seed": cfg.seed,
            "dataset_version": cfg.dataset_version,
            **split.to_dict(),
            "created_at": now_utc_iso(),
        },
    )
    vectors_dir = config.paths.representations / REPRESENTATION_NAME / sequence_version
    vectors_path = vectors_dir / "vectors.npy"
    work_ids_path = vectors_dir / "work_ids.npy"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    _atomic_save_npy(vectors_path, all_vectors.astype(np.float32))
    _atomic_save_npy(work_ids_path, np.asarray(all_ids, dtype=np.int64))
    _atomic_json(
        vectors_dir / "manifest.json",
        {
            "name": REPRESENTATION_NAME,
            "version": cfg.version,
            "sequence_version": sequence_version,
            "dataset_version": cfg.dataset_version,
            "model": model_name,
            "hidden": int(all_vectors.shape[1]) if all_vectors.size else 0,
            "n_works": len(all_ids),
            "synthetic": bool(dataset_info["synthetic"]),
            "created_at": now_utc_iso(),
            "note": "研究产物：仅落盘，不注册 representations 表、不建 map",
        },
    )

    checkpoint = {
        "version": cfg.version,
        "config": cfg.to_dict(),
        "model": model_info,
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "epoch": best_epoch,
        "best_val": None if best_val == -math.inf else float(best_val),
        "split": split.to_dict(),
        "dataset": {
            "dataset_version": cfg.dataset_version,
            "sequence_version": sequence_version,
            "synthetic": bool(dataset_info["synthetic"]),
            "n_works": dataset_info["n_works"],
        },
        "created_at": now_utc_iso(),
    }
    _atomic_torch_save(torch, checkpoint, checkpoint_path)

    elapsed = time.perf_counter() - started
    metrics = {
        "version": cfg.version,
        "created_at": now_utc_iso(),
        "elapsed_s": elapsed,
        "config": cfg.to_dict(),
        "dataset": {
            "dataset_version": cfg.dataset_version,
            "sequence_version": sequence_version,
            "source_manifest": dataset_info.get("source_manifest"),
            "synthetic": bool(dataset_info["synthetic"]),
            "n_works": dataset_info["n_works"],
            "n_missing_tokens": dataset_info["n_missing_tokens"],
            "missing_work_ids": dataset_info["missing_work_ids"],
            "truncated_works": dataset_info["truncated_works"],
            "split": split.to_dict(),
        },
        "model": model_info,
        "training": {
            "epochs_requested": int(cfg.epochs),
            "epochs_run": len(history),
            "early_stopped": early_stopped,
            "best_epoch": best_epoch,
            "best_val_mask_cosine": (None if best_val == -math.inf else float(best_val)),
            "history": history,
        },
        "val": {
            "mask_cosine": normal_mask["cosine"],
            "mask_mse": normal_mask["mse"],
            "full_mean_baseline_cosine": normal_mask["full_mean_baseline_cosine"],
            "full_mean_baseline_mse": normal_mask["full_mean_baseline_mse"],
            "av_accuracy": av_metrics["accuracy"],
            "av_auc": av_metrics["auc"],
            "av_negative_accuracy": normal_av_negative,
            "order_accuracy": order_metrics["accuracy"],
            "order_chance": order_metrics["chance"],
            "order_samples": order_metrics["n_samples"],
        },
        "temporal_sensitivity": {
            "normal_mask_cosine": normal_cosine,
            "reversed_mask_cosine": reversed_cosine,
            "shuffled_mask_cosine": shuffled_cosine,
            "reverse_drop": reverse_drop,
            "shuffle_drop": shuffle_drop,
            "av_negative_accuracy": normal_av_negative,
        },
        "representation": representation_metrics,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "metrics": str(metrics_path),
            "split": str(split_path),
            "vectors": str(vectors_path),
            "work_ids": str(work_ids_path),
        },
    }
    _atomic_json(metrics_path, metrics)
    summary = (
        f"v0.5-alpha 训练完成（{cfg.version}）："
        f"{len(history)} epoch，best_mask_cosine={best_val:.4f}，"
        f"mask_cos={normal_cosine}, av_acc={av_metrics['accuracy']}, "
        f"order_acc={order_metrics['accuracy']}，"
        f"耗时 {elapsed:.2f}s；checkpoint={checkpoint_path}"
    )
    return TrainResult(
        metrics=metrics,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        vectors_path=vectors_path,
        work_ids_path=work_ids_path,
        summary_text=summary,
    )
