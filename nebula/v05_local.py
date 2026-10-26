"""v0.5-beta 局部帧级时序模型（temporal-only / cross-attn，单一 InfoNCE）。

定位（最小实验，不训练全曲 Transformer）：
- 直接复用已落盘帧级特征，不重跑 CLIP/DINOv2/MERT：
  * 视觉: data/visual_dense/<CLIP 8fps frame_version>/<work_id>/dense_visual.npz
    per_frame (N,512) + manifest (fps/duration_s)；
  * 音频: data/audio_dense/<audio_version>/<work_id>/audio.npz
    frame_features (Nf,768)、frame_times、duration_s；
- 候选/切分完全读取 t3 的 data/v05_protocol/v2 manifest（work_id/anchor_bin/
  candidate_bins/labels/split/periodic），本模块不自造划分；协议缺失时只允许
  --smoke 合成契约测试。
- 上下文默认 3s（可配 2-4s）：以 anchor/candidate bin 中心为窗口中心，视觉
  8fps token、音频下采样到 20-30Hz（默认 25Hz），加相对时间编码与左右边界 mask。
- 模型 128d、4 heads：
  * temporal-only：视觉/音频各自 self-attn（默认 1 层）+ masked pooling，余弦打分；
  * cross-attn：各自 self-attn 后做 视觉->音频 / 音频->视觉 双向 cross-attention，
    再 pooling 与融合，余弦打分；
- 仅 InfoNCE 一个目标；输出固定 K 候选的 logits/embedding；指标 Top1/Recall@1、
  K=32 retrieval、按 work bootstrap 95% CI、坍缩/消融（audio_zero/visual_zero/
  time_shift）；旧 MLP/simple_sync 数字只引用协议 manifest 的 baselines 字段，
  不在本模块重造协议。
- 产物只落 data/models/v05_local_<arch>/ 与 logs/，不注册 DB/地图。

协议 manifest 期望结构（若 t3 最终字段略有差异，只需改 load_protocol 适配）：

{
  "protocol_version": "v2",
  "bin_seconds": 0.5,
  "context_seconds": 3.0,
  "baselines": {"mlp_top1": 0.0, "simple_sync_top1": 0.0, ...},
  "splits": {"train": [...], "val": [...], "test": [...]},
  "records": [
    {"work_id": 12, "anchor_bin": 40, "candidate_bins": [40, 41, 45],
     "labels": [1, 0, 0], "split": "train", "periodic": false}
  ]
}

records 也可以放在 records.jsonl、train/val/test.jsonl 或按 split 分文件；
labels 支持与 candidate_bins 等长的 0/1 列表（恰好一个正例），或单个正例下标。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .errors import NebulaError

try:  # 模型/训练需要 torch；纯协议解析尽量保持可 import
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - 无 torch 环境
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

LOCAL_VERSION = "v05_local"
DEFAULT_CONTEXT_SECONDS = 3.0
DEFAULT_AUDIO_HZ = 25.0
DEFAULT_HIDDEN = 128
DEFAULT_HEADS = 4
DEFAULT_SELF_LAYERS = 1
DEFAULT_CROSS_LAYERS = 1
DEFAULT_DROPOUT = 0.1
DEFAULT_TEMPERATURE = 0.07
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_PATIENCE = 4
DEFAULT_BOOTSTRAP = 200
DEFAULT_K_RETRIEVAL = 32
DEFAULT_REL_TIME_FREQS = 8
ARCHS = ("temporal", "cross")
SPLITS = ("train", "val", "test")
_PROTOCOL_NAMES = ("manifest.json", "protocol.json")
_RECORD_NAMES = ("records.jsonl", "records.json")


class V05LocalError(NebulaError):
    """v0.5-beta 局部时序数据/训练的可预期错误。"""


def _require_torch() -> Any:
    if torch is None:
        raise V05LocalError("需要 torch（uv pip install -e '.[ml]'）")
    return torch


# ============================================================================
# 配置
# ============================================================================


@dataclass(frozen=True)
class LocalModelConfig:
    """局部时序模型结构配置（默认 128d / 4 heads / 单层）。"""

    hidden: int = DEFAULT_HIDDEN
    heads: int = DEFAULT_HEADS
    self_layers: int = DEFAULT_SELF_LAYERS
    cross_layers: int = DEFAULT_CROSS_LAYERS
    dropout: float = DEFAULT_DROPOUT
    context_seconds: float = DEFAULT_CONTEXT_SECONDS
    audio_hz: float = DEFAULT_AUDIO_HZ
    visual_dim: int = 512
    audio_dim: int = 768
    temperature: float = DEFAULT_TEMPERATURE
    rel_time_mode: str = "sinusoidal"
    rel_time_freqs: int = DEFAULT_REL_TIME_FREQS
    max_visual_tokens: int = 64
    max_audio_tokens: int = 128

    def __post_init__(self) -> None:
        if int(self.hidden) <= 0 or int(self.hidden) % int(self.heads) != 0:
            raise V05LocalError(
                f"hidden({self.hidden}) 必须 >0 且能被 heads({self.heads}) 整除"
            )
        if int(self.self_layers) < 1:
            raise V05LocalError("self_layers 必须 >= 1")
        if int(self.cross_layers) < 1:
            raise V05LocalError("cross_layers 必须 >= 1")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise V05LocalError(f"dropout 必须在 [0,1)，收到 {self.dropout}")
        if not 1.0 <= float(self.context_seconds) <= 10.0:
            raise V05LocalError(f"context_seconds 必须在 [1,10]，收到 {self.context_seconds}")
        if not 10.0 <= float(self.audio_hz) <= 60.0:
            raise V05LocalError(f"audio_hz 必须在 [10,60]，收到 {self.audio_hz}")
        if int(self.visual_dim) < 1 or int(self.audio_dim) < 1:
            raise V05LocalError("visual_dim/audio_dim 必须 >= 1")
        if not 0.0 < float(self.temperature) <= 1.0:
            raise V05LocalError(f"temperature 必须在 (0,1]，收到 {self.temperature}")
        if self.rel_time_mode not in ("sinusoidal", "mlp"):
            raise V05LocalError("rel_time_mode 必须是 sinusoidal 或 mlp")
        if int(self.rel_time_freqs) < 1:
            raise V05LocalError("rel_time_freqs 必须 >= 1")
        if int(self.max_visual_tokens) < 2 or int(self.max_audio_tokens) < 2:
            raise V05LocalError("max_visual_tokens/max_audio_tokens 必须 >= 2")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalTrainConfig:
    """训练/评估配置（真实 GPU 训练由 Lead 跑；smoke 用 cpu 小数据）。"""

    arch: str = "temporal"
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    patience: int = DEFAULT_PATIENCE
    bootstrap: int = DEFAULT_BOOTSTRAP
    k_retrieval: int = DEFAULT_K_RETRIEVAL
    device: str = "auto"
    num_workers: int = 0
    model: LocalModelConfig = field(default_factory=LocalModelConfig)

    def validate(self) -> "LocalTrainConfig":
        if self.arch not in ARCHS:
            raise V05LocalError(f"arch 必须是 {ARCHS}，收到 {self.arch!r}")
        if not self.seeds or len(set(int(v) for v in self.seeds)) != len(self.seeds):
            raise V05LocalError("seeds 必须非空且不重复")
        if int(self.epochs) < 1 or int(self.batch_size) < 1:
            raise V05LocalError("epochs/batch_size 必须 >= 1")
        if float(self.lr) <= 0 or float(self.weight_decay) < 0:
            raise V05LocalError("lr 必须 >0 且 weight_decay >=0")
        if int(self.patience) < 0 or int(self.bootstrap) < 0:
            raise V05LocalError("patience/bootstrap 必须 >=0")
        if int(self.k_retrieval) < 1:
            raise V05LocalError("k_retrieval 必须 >=1")
        if self.device not in ("auto", "cpu", "cuda"):
            raise V05LocalError("device 必须是 auto/cpu/cuda")
        return self

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["seeds"] = [int(value) for value in self.seeds]
        payload["model"] = self.model.to_dict()
        return payload


# ============================================================================
# 协议读取（紧贴 t3 的 data/v05_protocol/v2）
# ============================================================================


@dataclass(frozen=True)
class ProtocolRecord:
    work_id: int
    anchor_bin: int
    candidate_bins: tuple[int, ...]
    positive_index: int
    split: str
    periodic: bool = False

    @property
    def n_candidates(self) -> int:
        return len(self.candidate_bins)


@dataclass(frozen=True)
class LocalProtocol:
    records: tuple[ProtocolRecord, ...]
    metadata: dict[str, Any]
    bin_seconds: float = 0.5
    retrieval_records: tuple[ProtocolRecord, ...] = ()
    periodic_records: tuple[ProtocolRecord, ...] = ()

    def split_records(self) -> dict[str, tuple[ProtocolRecord, ...]]:
        result: dict[str, list[ProtocolRecord]] = {name: [] for name in SPLITS}
        for record in self.records:
            result[record.split].append(record)
        return {name: tuple(items) for name, items in result.items()}

    def split_work_ids(self) -> dict[str, tuple[int, ...]]:
        splits = self.split_records()
        return {
            name: tuple(sorted({record.work_id for record in items}))
            for name, items in splits.items()
        }

    def validate_disjoint(self) -> None:
        works = self.split_work_ids()
        for left in SPLITS:
            for right in SPLITS:
                if left >= right:
                    continue
                overlap = set(works[left]) & set(works[right])
                if overlap:
                    raise V05LocalError(
                        f"协议 split 重叠: {left}/{right} 共享 work_id {sorted(overlap)[:10]}"
                    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V05LocalError(f"读取协议 JSON 失败（{path}）：{exc}") from exc
    if not isinstance(payload, dict):
        raise V05LocalError(f"协议 JSON 顶层必须是对象: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    try:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise V05LocalError(f"{path}:{line_no} JSONL 解析失败：{exc}") from exc
            if not isinstance(payload, dict):
                raise V05LocalError(f"{path}:{line_no} 必须是对象")
            items.append(payload)
    except OSError as exc:
        raise V05LocalError(f"读取协议 JSONL 失败（{path}）：{exc}") from exc
    return items


def _first_int(payload: Mapping[str, Any], names: Sequence[str]) -> int | None:
    for name in names:
        if name in payload and payload[name] is not None:
            try:
                return int(payload[name])
            except (TypeError, ValueError) as exc:
                raise V05LocalError(f"协议字段 {name} 必须是整数：{payload[name]!r}") from exc
    return None


def _parse_labels(labels: Any, n_candidates: int) -> int:
    if labels is None:
        raise V05LocalError("协议记录缺少 labels")
    if isinstance(labels, (int, np.integer)) and not isinstance(labels, bool):
        index = int(labels)
        if 0 <= index < n_candidates:
            return index
        raise V05LocalError(f"labels 正例下标越界: {labels}（K={n_candidates}）")
    if isinstance(labels, Mapping):
        raise V05LocalError("labels 不支持映射类型")
    if not isinstance(labels, (list, tuple, np.ndarray)):
        raise V05LocalError(f"labels 类型不支持: {type(labels).__name__}")
    values = list(labels)
    if len(values) != n_candidates:
        raise V05LocalError(
            f"labels 长度 {len(values)} 与 candidate_bins 长度 {n_candidates} 不一致"
        )
    positive: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool):
            flag = 1 if value else 0
        else:
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise V05LocalError(f"labels[{index}] 不是数字: {value!r}") from exc
            if not math.isfinite(number) or number not in (0.0, 1.0):
                raise V05LocalError(f"labels[{index}] 必须是 0/1，收到 {value!r}")
            flag = int(number)
        if flag:
            positive.append(index)
    if len(positive) != 1:
        raise V05LocalError(
            f"labels 必须恰好一个正例，收到 {len(positive)} 个（K={n_candidates}）"
        )
    return positive[0]


def _parse_record(payload: Mapping[str, Any], *, default_split: str | None = None) -> ProtocolRecord:
    work_id = _first_int(payload, ("work_id", "work", "width_id", "work_idx"))
    if work_id is None:
        raise V05LocalError("协议记录缺少 work_id")
    anchor_bin = _first_int(payload, ("anchor_bin", "anchor", "anchor_index", "bin"))
    if anchor_bin is None or anchor_bin < 0:
        raise V05LocalError(f"work_id={work_id} 缺少合法 anchor_bin")
    raw_candidates = payload.get("candidate_bins")
    if raw_candidates is None:
        raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, (list, tuple)):
        raise V05LocalError(f"work_id={work_id} 缺少 candidate_bins 列表")
    candidates: list[int] = []
    for value in raw_candidates:
        try:
            candidate = int(value)
        except (TypeError, ValueError) as exc:
            raise V05LocalError(f"candidate_bins 含非整数: {value!r}") from exc
        if candidate < 0:
            raise V05LocalError(f"candidate_bins 不能为负: {candidate}")
        candidates.append(candidate)
    if not candidates:
        raise V05LocalError(f"work_id={work_id} candidate_bins 为空")
    labels = payload.get("labels")
    if labels is None:
        labels = payload.get("label")
    if labels is None and payload.get("positive_index") is not None:
        labels = int(payload["positive_index"])
    positive_index = _parse_labels(labels, len(candidates))
    split = str(payload.get("split") or default_split or "").strip()
    if split not in SPLITS:
        raise V05LocalError(
            f"work_id={work_id} split 非法: {split!r}（必须是 {SPLITS}）"
        )
    periodic_raw = payload.get("periodic", False)
    periodic = bool(periodic_raw)
    return ProtocolRecord(
        work_id=int(work_id),
        anchor_bin=int(anchor_bin),
        candidate_bins=tuple(candidates),
        positive_index=int(positive_index),
        split=split,
        periodic=periodic,
    )


def _load_records_from_dir(directory: Path) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    metadata: dict[str, Any] = {}
    records: list[Mapping[str, Any]] = []
    manifest_path = next(
        (directory / name for name in _PROTOCOL_NAMES if (directory / name).is_file()),
        None,
    )
    if manifest_path is not None:
        metadata = _read_json(manifest_path)
        raw = metadata.get("records")
        if isinstance(raw, list):
            records.extend(item for item in raw if isinstance(item, Mapping))
        elif raw is not None:
            raise V05LocalError("manifest.records 必须是列表")
    if not records:
        for name in _RECORD_NAMES:
            path = directory / name
            if path.is_file():
                if path.suffix == ".jsonl":
                    records.extend(_read_jsonl(path))
                else:
                    payload = _read_json(path)
                    raw = payload.get("records")
                    if isinstance(raw, list):
                        records.extend(item for item in raw if isinstance(item, Mapping))
                break
    if not records:
        for split in SPLITS:
            path = directory / f"{split}.jsonl"
            if path.is_file():
                records.extend(_read_jsonl(path))
    if not records:
        for split in SPLITS:
            path = directory / f"{split}.json"
            if path.is_file():
                payload = _read_json(path)
                raw = payload.get("records", payload) if isinstance(payload, dict) else payload
                if isinstance(raw, list):
                    records.extend(item for item in raw if isinstance(item, Mapping))
    return records, metadata


_SPLIT_CODES = {0: "train", 1: "val", 2: "test"}


def _resolve_anchors_path(manifest_path: Path, anchors_file: str) -> Path:
    """解析 manifest.anchors_file（t3 写的是 data 根相对路径）。"""

    raw = Path(anchors_file)
    if raw.is_absolute() and raw.is_file():
        return raw
    candidates = [
        manifest_path.parent / raw.name,
        manifest_path.parent.parent / raw,
        manifest_path.parent / raw,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise V05LocalError(
        f"找不到 anchors 文件: {anchors_file}（manifest={manifest_path}）"
    )


def _anchor_candidates(
    bins_flat: np.ndarray, ptr: np.ndarray, row: int
) -> tuple[int, ...]:
    start = int(ptr[row])
    end = int(ptr[row + 1])
    if not (0 <= start <= end <= len(bins_flat)):
        raise V05LocalError(f"anchors candidate ptr 非法: row={row} start={start} end={end}")
    return tuple(int(value) for value in bins_flat[start:end])


def _load_anchors_protocol(
    manifest: Mapping[str, Any], manifest_path: Path
) -> LocalProtocol:
    """读取 t3 anchors.npz 协议（main/retrieval/periodic）。"""

    anchors_file = manifest.get("anchors_file")
    if not isinstance(anchors_file, str) or not anchors_file.strip():
        raise V05LocalError("协议 manifest 缺少 anchors_file")
    anchors_path = _resolve_anchors_path(manifest_path, anchors_file)
    try:
        data = np.load(anchors_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise V05LocalError(f"读取 anchors.npz 失败（{anchors_path}）：{exc}") from exc
    try:
        expected_sha = manifest.get("anchors_sha256")
        if isinstance(expected_sha, str) and expected_sha.strip():
            digest = hashlib.sha256(anchors_path.read_bytes()).hexdigest()
            if digest != expected_sha.strip():
                raise V05LocalError(
                    f"anchors.npz sha256 与 manifest 不一致（{anchors_path}）"
                )
        bin_seconds = float(manifest.get("bin_seconds", 0.5))
        if not math.isfinite(bin_seconds) or bin_seconds <= 0:
            raise V05LocalError(f"协议 bin_seconds 非法: {bin_seconds}")

        def _record(
            work_id: int,
            split_code: int,
            anchor_bin: int,
            label: int,
            candidate_bins: tuple[int, ...],
            *,
            periodic: bool = False,
        ) -> ProtocolRecord:
            split = _SPLIT_CODES.get(int(split_code))
            if split is None:
                raise V05LocalError(f"未知 split_code: {split_code}")
            n_candidates = len(candidate_bins)
            if n_candidates < 2:
                raise V05LocalError(
                    f"work_id={work_id} anchor_bin={anchor_bin} 候选数 < 2"
                )
            positive = int(label)
            if not 0 <= positive < n_candidates:
                raise V05LocalError(
                    f"work_id={work_id} anchor_bin={anchor_bin} label 越界: "
                    f"{positive}（K={n_candidates}）"
                )
            return ProtocolRecord(
                work_id=int(work_id),
                anchor_bin=int(anchor_bin),
                candidate_bins=candidate_bins,
                positive_index=positive,
                split=split,
                periodic=bool(periodic),
            )

        # NpzFile 每次 __getitem__ 都会重新解压数组；循环前一次性取出全部所需数组
        main_work = data["main_work_id"]
        main_split = data["main_split_code"]
        main_anchor = data["main_anchor_bin"]
        main_label = data["main_label"]
        main_bins = data["main_candidate_bins_flat"]
        main_ptr = data["main_candidate_ptr"]
        records: list[ProtocolRecord] = []
        for row in range(len(main_anchor)):
            candidate_bins = _anchor_candidates(main_bins, main_ptr, row)
            records.append(
                _record(
                    int(main_work[row]),
                    int(main_split[row]),
                    int(main_anchor[row]),
                    int(main_label[row]),
                    candidate_bins,
                )
            )
        retrieval_records: list[ProtocolRecord] = []
        if "retrieval_anchor_bin" in data:
            retrieval_work = data["retrieval_work_id"]
            retrieval_split = data["retrieval_split_code"]
            retrieval_anchor = data["retrieval_anchor_bin"]
            retrieval_label = data["retrieval_label"]
            retrieval_bins = data["retrieval_candidate_bins_flat"]
            retrieval_ptr = data["retrieval_candidate_ptr"]
            for row in range(len(retrieval_anchor)):
                candidate_bins = _anchor_candidates(retrieval_bins, retrieval_ptr, row)
                retrieval_records.append(
                    _record(
                        int(retrieval_work[row]),
                        int(retrieval_split[row]),
                        int(retrieval_anchor[row]),
                        int(retrieval_label[row]),
                        candidate_bins,
                    )
                )
        periodic_records: list[ProtocolRecord] = []
        if "periodic_anchor_bin" in data:
            periodic_work = data["periodic_work_id"]
            periodic_split = data["periodic_split_code"]
            periodic_anchor = data["periodic_anchor_bin"]
            periodic_label = data["periodic_label"]
            periodic_bins = data["periodic_candidate_bins_flat"]
            periodic_ptr = data["periodic_candidate_ptr"]
            for row in range(len(periodic_anchor)):
                candidate_bins = _anchor_candidates(periodic_bins, periodic_ptr, row)
                try:
                    periodic_records.append(
                        _record(
                            int(periodic_work[row]),
                            int(periodic_split[row]),
                            int(periodic_anchor[row]),
                            int(periodic_label[row]),
                            candidate_bins,
                            periodic=True,
                        )
                    )
                except V05LocalError:
                    continue
        if not records:
            raise V05LocalError("anchors.npz 没有 main 协议记录")
        protocol = LocalProtocol(
            records=tuple(records),
            metadata=dict(manifest),
            bin_seconds=bin_seconds,
            retrieval_records=tuple(retrieval_records),
            periodic_records=tuple(periodic_records),
        )
        protocol.validate_disjoint()
        LOGGER.info(
            "anchors 协议读取完成: main=%d retrieval=%d periodic=%d split works=%s",
            len(protocol.records),
            len(protocol.retrieval_records),
            len(protocol.periodic_records),
            {name: len(ids) for name, ids in protocol.split_work_ids().items()},
        )
        return protocol
    finally:
        data.close()


def load_protocol(path: str | Path) -> LocalProtocol:
    """读取 t3 的 data/v05_protocol/v2（anchors.npz 或 records JSON/JSONL）。"""

    target = Path(path)
    if target.exists() and target.is_file() and target.suffix == ".npz":
        raise V05LocalError("请传入协议目录或 manifest.json（anchors.npz 需要 manifest）")
    if not target.exists():
        raise V05LocalError(f"协议不存在: {target}（t3 尚未落盘时请用 --smoke）")
    if target.is_dir():
        raw_records, metadata = _load_records_from_dir(target)
        manifest_path = next(
            (target / name for name in _PROTOCOL_NAMES if (target / name).is_file()),
            target,
        )
    elif target.suffix == ".jsonl":
        raw_records = _read_jsonl(target)
        metadata = {}
        manifest_path = target
    else:
        payload = _read_json(target)
        metadata = payload
        manifest_path = target
        raw = payload.get("records")
        if raw is None:
            raw = payload.get("sets", payload.get("candidates"))
        raw_records = (
            [item for item in raw if isinstance(item, Mapping)]
            if isinstance(raw, list)
            else []
        )
    if isinstance(metadata, Mapping) and metadata.get("anchors_file"):
        # t3 的权威 schema 是 anchors.npz（含 main/retrieval/periodic）；
        # records.jsonl 只是兼容文件，优先 anchors 避免丢失 retrieval/periodic。
        return _load_anchors_protocol(metadata, manifest_path)
    if not raw_records:
        raise V05LocalError(f"协议中没有 records: {target}")
    bin_seconds = float(metadata.get("bin_seconds", 0.5))
    if not math.isfinite(bin_seconds) or bin_seconds <= 0:
        raise V05LocalError(f"协议 bin_seconds 非法: {bin_seconds}")
    # manifest.splits 给出 work 级划分时，record 可省略 split
    split_works: dict[int, str] = {}
    splits_meta = metadata.get("splits") if isinstance(metadata, Mapping) else None
    if isinstance(splits_meta, Mapping):
        for split in SPLITS:
            for work_id in splits_meta.get(split, []) or []:
                split_works[int(work_id)] = split
    records: list[ProtocolRecord] = []
    for index, raw in enumerate(raw_records):
        payload = dict(raw)
        if not payload.get("split") and isinstance(splits_meta, Mapping):
            work_id = _first_int(payload, ("work_id", "work", "width_id", "work_idx"))
            if work_id is not None and work_id in split_works:
                payload["split"] = split_works[work_id]
        try:
            records.append(_parse_record(payload))
        except V05LocalError as exc:
            raise V05LocalError(f"协议第 {index} 条记录非法：{exc}") from exc
    protocol = LocalProtocol(
        records=tuple(records), metadata=dict(metadata), bin_seconds=bin_seconds
    )
    protocol.validate_disjoint()
    LOGGER.info(
        "协议读取完成：%d 条候选集，split works=%s",
        len(protocol.records),
        {name: len(ids) for name, ids in protocol.split_work_ids().items()},
    )
    return protocol


# ============================================================================
# 帧特征提取（纯函数，便于测试）
# ============================================================================


def bin_center_seconds(bin_index: int, bin_seconds: float) -> float:
    if int(bin_index) < 0:
        raise V05LocalError(f"bin index 不能为负: {bin_index}")
    seconds = float(bin_seconds)
    if not math.isfinite(seconds) or seconds <= 0:
        raise V05LocalError(f"bin_seconds 非法: {bin_seconds}")
    return (float(bin_index) + 0.5) * seconds


def extract_visual_tokens(
    per_frame: np.ndarray,
    *,
    fps: float,
    duration_s: float,
    center_s: float,
    context_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从 (N,D) 帧向量截取以 center_s 为中心、context_s 长的视觉 token。

    返回 (features (Nv,D) float32, rel_times (Nv,) in [-1,1], mask (Nv,) bool)；
    帧 i 覆盖 [i/fps, (i+1)/fps)，token 时间取帧中心 (i+0.5)/fps。
    """

    frames = np.asarray(per_frame)
    if frames.ndim != 2 or frames.shape[0] < 1:
        raise V05LocalError(f"per_frame 形状非法: {frames.shape}")
    rate = float(fps)
    duration = float(duration_s)
    window = float(context_s)
    if not math.isfinite(rate) or rate <= 0:
        raise V05LocalError(f"fps 非法: {fps}")
    if not math.isfinite(duration) or duration <= 0:
        raise V05LocalError(f"duration_s 非法: {duration_s}")
    if not math.isfinite(window) or window <= 0:
        raise V05LocalError(f"context_s 非法: {context_s}")
    frame_times = (np.arange(frames.shape[0], dtype=np.float64) + 0.5) / rate
    start = float(center_s) - window / 2.0
    end = float(center_s) + window / 2.0
    valid = (frame_times >= start) & (frame_times <= end) & (frame_times >= 0.0) & (
        frame_times <= duration
    )
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        empty = np.zeros((1, frames.shape[1]), dtype=np.float32)
        return empty, np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=bool)
    features = frames[indices].astype(np.float32)
    times = ((frame_times[indices] - float(center_s)) / (window / 2.0)).astype(
        np.float32
    )
    return features, times, valid[indices]


def downsample_audio_tokens(
    frame_features: np.ndarray,
    frame_times: np.ndarray,
    *,
    center_s: float,
    context_s: float,
    target_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 75Hz 音频帧特征按 target_hz 时间 bin 求均值，返回局部 token。"""

    features = np.asarray(frame_features, dtype=np.float32)
    times = np.asarray(frame_times, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] < 1:
        raise V05LocalError(f"frame_features 形状非法: {features.shape}")
    if times.shape != (features.shape[0],):
        raise V05LocalError("frame_times 与 frame_features 长度不一致")
    if int(times.size) != int(round(len(times))):
        raise V05LocalError("frame_times 未排序或长度异常")
    rate = float(target_hz)
    window = float(context_s)
    if not math.isfinite(rate) or rate <= 0:
        raise V05LocalError(f"target_hz 非法: {target_hz}")
    if not math.isfinite(window) or window <= 0:
        raise V05LocalError(f"context_s 非法: {context_s}")
    start = float(center_s) - window / 2.0
    end = float(center_s) + window / 2.0
    n_tokens = max(1, int(round(window * rate)))
    edges = start + (np.arange(n_tokens + 1, dtype=np.float64) * window / n_tokens)
    order = np.argsort(times, kind="stable")
    sorted_times = times[order].astype(np.float64)
    sorted_features = features[order]
    token_features = np.zeros((n_tokens, features.shape[1]), dtype=np.float32)
    token_mask = np.zeros((n_tokens,), dtype=bool)
    left = np.searchsorted(sorted_times, edges[:-1], side="left")
    right = np.searchsorted(sorted_times, edges[1:], side="left")
    for index in range(n_tokens):
        lo, hi = int(left[index]), int(right[index])
        if hi <= lo:
            continue
        # 只用窗口范围内的帧
        segment_times = sorted_times[lo:hi]
        if segment_times[-1] < start or segment_times[0] > end:
            continue
        token_features[index] = sorted_features[lo:hi].mean(axis=0)
        token_mask[index] = True
    token_times = ((edges[:-1] + edges[1:]) / 2.0 - float(center_s)) / (window / 2.0)
    return token_features, token_times.astype(np.float32), token_mask


# ============================================================================
# 帧数据存储 / 数据集 / collate
# ============================================================================


def _discover_version(root: Path, pattern: Sequence[str], label: str) -> Path:
    if not root.is_dir():
        raise V05LocalError(f"{label} 目录不存在: {root}")
    directories = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and all(token in path.name for token in pattern)
    )
    if not directories:
        raise V05LocalError(f"{label} 下找不到匹配 {pattern} 的版本目录: {root}")
    return directories[0]


class FrameDataStore:
    """按 work 读取视觉/音频帧特征；LRU 缓存整部作品，避免重复读 npz。"""

    def __init__(
        self,
        *,
        visual_root: str | Path,
        audio_root: str | Path,
        visual_version: str | Path | None = None,
        audio_version: str | Path | None = None,
        context_seconds: float = DEFAULT_CONTEXT_SECONDS,
        audio_hz: float = DEFAULT_AUDIO_HZ,
        bin_seconds: float = 0.5,
        cache_size: int = 8,
    ) -> None:
        self.visual_root = Path(visual_root)
        self.audio_root = Path(audio_root)
        self.visual_version_dir = (
            Path(visual_version)
            if visual_version is not None
            else _discover_version(self.visual_root, ("clip", "fps8"), "visual_dense")
        )
        self.audio_version_dir = (
            Path(audio_version)
            if audio_version is not None
            else _discover_version(self.audio_root, (), "audio_dense")
        )
        if not self.visual_version_dir.is_absolute():
            self.visual_version_dir = self.visual_root / self.visual_version_dir
        if not self.audio_version_dir.is_absolute():
            self.audio_version_dir = self.audio_root / self.audio_version_dir
        self.context_seconds = float(context_seconds)
        self.audio_hz = float(audio_hz)
        self.bin_seconds = float(bin_seconds)
        self.cache_size = max(1, int(cache_size))
        self._cache: dict[int, dict[str, Any]] = {}
        self._order: list[int] = []

    def _load_work(self, work_id: int) -> dict[str, Any]:
        key = int(work_id)
        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]
        visual_dir = self.visual_version_dir / f"{key:06d}"
        audio_dir = self.audio_version_dir / f"{key:06d}"
        visual_npz = visual_dir / "dense_visual.npz"
        visual_manifest = visual_dir / "manifest.json"
        audio_npz = audio_dir / "audio.npz"
        audio_manifest = audio_dir / "manifest.json"
        for path in (visual_npz, visual_manifest, audio_npz, audio_manifest):
            if not path.is_file():
                raise V05LocalError(f"work_id={work_id} 帧数据缺失: {path}")
        try:
            with np.load(visual_npz, allow_pickle=False) as data:
                per_frame = np.asarray(data["per_frame"], dtype=np.float32)
            visual_meta = json.loads(visual_manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError) as exc:
            raise V05LocalError(f"读取视觉帧特征失败（{visual_npz}）：{exc}") from exc
        try:
            with np.load(audio_npz, allow_pickle=False) as data:
                audio_features = np.asarray(data["frame_features"], dtype=np.float32)
                audio_times = np.asarray(data["frame_times"], dtype=np.float32)
            audio_meta = json.loads(audio_manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError) as exc:
            raise V05LocalError(f"读取音频帧特征失败（{audio_npz}）：{exc}") from exc
        fps = float(visual_meta.get("fps", 0.0) or 0.0)
        visual_duration = float(visual_meta.get("duration_s", 0.0) or 0.0)
        audio_duration = float(audio_meta.get("duration_s", 0.0) or 0.0)
        if not math.isfinite(fps) or fps <= 0:
            raise V05LocalError(f"work_id={work_id} 视觉 manifest fps 非法: {fps}")
        if per_frame.ndim != 2 or per_frame.shape[0] < 1:
            raise V05LocalError(f"work_id={work_id} per_frame 形状非法: {per_frame.shape}")
        payload = {
            "per_frame": per_frame,
            "fps": fps,
            "visual_duration": visual_duration,
            "audio_features": audio_features,
            "audio_times": audio_times,
            "audio_duration": audio_duration,
            "visual_manifest": visual_meta,
            "audio_manifest": audio_meta,
        }
        self._cache[key] = payload
        self._order.append(key)
        while len(self._order) > self.cache_size:
            evicted = self._order.pop(0)
            self._cache.pop(evicted, None)
        return payload

    @property
    def visual_dim(self) -> int:
        # 读取一个作品以确定维度（真实数据固定 512）
        for work_dir in sorted(
            path for path in self.visual_version_dir.iterdir() if path.is_dir()
        ):
            npz = work_dir / "dense_visual.npz"
            if npz.is_file():
                with np.load(npz, allow_pickle=False) as data:
                    return int(np.asarray(data["per_frame"]).shape[1])
        raise V05LocalError("视觉版本目录中没有可读作品")

    @property
    def audio_dim(self) -> int:
        for work_dir in sorted(
            path for path in self.audio_version_dir.iterdir() if path.is_dir()
        ):
            npz = work_dir / "audio.npz"
            if npz.is_file():
                with np.load(npz, allow_pickle=False) as data:
                    return int(np.asarray(data["frame_features"]).shape[1])
        raise V05LocalError("音频版本目录中没有可读作品")

    def visual_tokens(
        self, work_id: int, center_s: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        work = self._load_work(work_id)
        return extract_visual_tokens(
            work["per_frame"],
            fps=work["fps"],
            duration_s=work["visual_duration"],
            center_s=center_s,
            context_s=self.context_seconds,
        )

    def audio_tokens(
        self, work_id: int, center_s: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        work = self._load_work(work_id)
        return downsample_audio_tokens(
            work["audio_features"],
            work["audio_times"],
            center_s=center_s,
            context_s=self.context_seconds,
            target_hz=self.audio_hz,
        )


@dataclass(frozen=True)
class LocalCandidateSet:
    """一条协议记录展开后的候选集（供 dataset/调试/合成测试使用）。"""

    work_id: int
    split: str
    anchor_bin: int
    periodic: bool
    visual_features: np.ndarray
    visual_times: np.ndarray
    visual_mask: np.ndarray
    audio_features: tuple[np.ndarray, ...]
    audio_times: tuple[np.ndarray, ...]
    audio_masks: tuple[np.ndarray, ...]
    positive_index: int


class LocalPairDataset:
    """协议记录 -> 视觉 anchor / K 个音频候选；按需提取帧上下文。"""

    def __init__(
        self,
        records: Sequence[ProtocolRecord],
        store: "FrameDataStore | Any",
        *,
        bin_seconds: float = 0.5,
    ) -> None:
        if not records:
            raise V05LocalError("LocalPairDataset 需要至少一条协议记录")
        self.records = tuple(records)
        self.store = store
        self.bin_seconds = float(bin_seconds)
        # 小协议子集可整条缓存（帧上下文提取是主要 CPU 成本）；全量协议不缓存，
        # 避免 4 万条记录把内存撑爆。
        self._item_cache: dict[int, LocalCandidateSet] = {}
        self._cache_items = len(self.records) <= 5000

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> LocalCandidateSet:
        if self._cache_items:
            cached = self._item_cache.get(int(index))
            if cached is not None:
                return cached
        record = self.records[int(index)]
        anchor_center = bin_center_seconds(record.anchor_bin, self.bin_seconds)
        visual_features, visual_times, visual_mask = self.store.visual_tokens(
            record.work_id, anchor_center
        )
        audio_features: list[np.ndarray] = []
        audio_times: list[np.ndarray] = []
        audio_masks: list[np.ndarray] = []
        for candidate_bin in record.candidate_bins:
            center = bin_center_seconds(candidate_bin, self.bin_seconds)
            features, times, mask = self.store.audio_tokens(record.work_id, center)
            audio_features.append(features)
            audio_times.append(times)
            audio_masks.append(mask)
        result = LocalCandidateSet(
            work_id=record.work_id,
            split=record.split,
            anchor_bin=record.anchor_bin,
            periodic=record.periodic,
            visual_features=visual_features,
            visual_times=visual_times,
            visual_mask=visual_mask,
            audio_features=tuple(audio_features),
            audio_times=tuple(audio_times),
            audio_masks=tuple(audio_masks),
            positive_index=record.positive_index,
        )
        if self._cache_items:
            self._item_cache[int(index)] = result
        return result


def _pad_2d(arrays: Sequence[np.ndarray], pad_value: float = 0.0) -> np.ndarray:
    if not arrays:
        raise V05LocalError("_pad_2d 需要非空序列")
    max_len = max(int(array.shape[0]) for array in arrays)
    width = int(arrays[0].shape[1])
    output = np.full((len(arrays), max_len, width), pad_value, dtype=np.float32)
    for index, array in enumerate(arrays):
        output[index, : array.shape[0], :] = array
    return output


def _pad_1d(arrays: Sequence[np.ndarray], pad_value: float = 0.0) -> np.ndarray:
    max_len = max(int(array.shape[0]) for array in arrays)
    output = np.full((len(arrays), max_len), pad_value, dtype=np.float32)
    for index, array in enumerate(arrays):
        output[index, : array.shape[0]] = array
    return output


def _pad_mask(arrays: Sequence[np.ndarray]) -> np.ndarray:
    max_len = max(int(array.shape[0]) for array in arrays)
    output = np.zeros((len(arrays), max_len), dtype=bool)
    for index, array in enumerate(arrays):
        output[index, : array.shape[0]] = np.asarray(array, dtype=bool)
    return output


def collate_local_sets(items: Sequence[LocalCandidateSet]) -> dict[str, Any]:
    """把变长候选集 padding 成模型 batch；labels 保留正例下标。"""

    if not items:
        raise V05LocalError("collate_local_sets 收到空 batch")
    batch_size = len(items)
    max_candidates = max(len(item.audio_features) for item in items)
    max_audio_len = max(
        int(features.shape[0])
        for item in items
        for features in item.audio_features
    )
    audio_dim = int(items[0].audio_features[0].shape[1])
    audio_features = np.zeros(
        (batch_size, max_candidates, max_audio_len, audio_dim), dtype=np.float32
    )
    audio_times = np.zeros((batch_size, max_candidates, max_audio_len), dtype=np.float32)
    audio_mask = np.zeros((batch_size, max_candidates, max_audio_len), dtype=bool)
    candidate_mask = np.zeros((batch_size, max_candidates), dtype=bool)
    labels = np.zeros((batch_size,), dtype=np.int64)
    for row, item in enumerate(items):
        for column, features in enumerate(item.audio_features):
            length = int(features.shape[0])
            if features.shape[1] != audio_dim:
                raise V05LocalError("候选项音频维度不一致")
            audio_features[row, column, :length] = features
            audio_times[row, column, :length] = item.audio_times[column]
            audio_mask[row, column, :length] = item.audio_masks[column]
            candidate_mask[row, column] = True
        if not 0 <= item.positive_index < len(item.audio_features):
            raise V05LocalError("positive_index 越界")
        labels[row] = int(item.positive_index)
    visual_features = _pad_2d([item.visual_features for item in items])
    visual_times = _pad_1d([item.visual_times for item in items])
    visual_mask = _pad_mask([item.visual_mask for item in items])
    if visual_features.shape[1] > 0:
        # 完全空的 sequence 给一个 mask=True 的零 token，避免 attention 全 mask
        empty = ~visual_mask.any(axis=1)
        visual_mask[empty, 0] = True
    if audio_mask.shape[2] > 0:
        empty_audio = ~audio_mask.any(axis=2)
        audio_mask[empty_audio, 0] = True
    return {
        "visual": visual_features,
        "visual_time": visual_times,
        "visual_mask": visual_mask,
        "audio": audio_features,
        "audio_time": audio_times,
        "audio_mask": audio_mask,
        "candidate_mask": candidate_mask,
        "labels": labels,
        "work_ids": [int(item.work_id) for item in items],
        "splits": [item.split for item in items],
        "positive_indices": [int(item.positive_index) for item in items],
    }


def collate_records(
    records: Sequence[ProtocolRecord],
    store: "FrameDataStore | Any",
    *,
    bin_seconds: float = 0.5,
) -> dict[str, Any]:
    dataset = LocalPairDataset(records, store, bin_seconds=bin_seconds)
    return collate_local_sets([dataset[index] for index in range(len(dataset))])

# ============================================================================
# 相对时间编码 / token 序列编码器 / 两种模型
# ============================================================================


def _need_torch_modules() -> tuple[Any, Any, Any]:
    torch_mod = _require_torch()
    if nn is None or F is None:  # pragma: no cover
        raise V05LocalError("torch.nn 不可用")
    return torch_mod, nn, F


def _masked_pool(tokens: Any, mask: Any) -> Any:
    """(B,T,H) + (B,T) bool -> (B,H) masked mean。"""
    torch_mod, _nn, _F = _need_torch_modules()
    if tokens.ndim != 3 or mask.shape != tokens.shape[:2]:
        raise V05LocalError("masked_pool 输入形状非法")
    weights = mask.to(tokens.dtype)
    denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    pooled = (tokens * weights.unsqueeze(-1)).sum(dim=1) / denominator
    return torch_mod.where(weights.sum(dim=1, keepdim=True) > 0, pooled, torch_mod.zeros_like(pooled))


class RelativeTimeEncoding(nn.Module if nn is not None else object):  # type: ignore[misc]
    """相对时间编码：正弦特征或 MLP，输出 (…, hidden)。"""

    def __init__(
        self, hidden: int, *, mode: str = "sinusoidal", freqs: int = DEFAULT_REL_TIME_FREQS
    ) -> None:
        _torch, nn_mod, _F = _need_torch_modules()
        super().__init__()
        self.mode = str(mode)
        self.freqs = int(freqs)
        if self.mode == "sinusoidal":
            # 额外拼接原始相对时间（线性项），避免 ±1 端点处 sin(n*pi)=0、
            # cos 相同导致左右边界不可区分
            self.proj = nn_mod.Linear(1 + 2 * self.freqs, int(hidden))
        elif self.mode == "mlp":
            self.proj = nn_mod.Sequential(
                nn_mod.Linear(1, int(hidden)),
                nn_mod.GELU(),
                nn_mod.Linear(int(hidden), int(hidden)),
            )
        else:
            raise V05LocalError(f"未知 rel_time_mode: {mode!r}")

    def forward(self, times: Any) -> Any:
        torch_mod, _nn, _F = _need_torch_modules()
        if times.ndim < 1:
            raise V05LocalError("times 至少 1 维")
        if self.mode == "mlp":
            return self.proj(times.unsqueeze(-1))
        frequencies = torch_mod.pow(
            torch_mod.tensor(2.0, device=times.device, dtype=times.dtype),
            torch_mod.arange(self.freqs, device=times.device, dtype=times.dtype),
        ) * math.pi
        angles = times.unsqueeze(-1) * frequencies
        features = torch_mod.cat(
            [
                times.unsqueeze(-1),
                torch_mod.sin(angles),
                torch_mod.cos(angles),
            ],
            dim=-1,
        )
        return self.proj(features)


class TokenSequenceEncoder(nn.Module if nn is not None else object):  # type: ignore[misc]
    """模态内 self-attn + masked pooling。"""

    def __init__(self, input_dim: int, config: LocalModelConfig) -> None:
        _torch, nn_mod, _F = _need_torch_modules()
        super().__init__()
        hidden = int(config.hidden)
        self.input_proj = nn_mod.Linear(int(input_dim), hidden)
        self.time_encoder = RelativeTimeEncoding(
            hidden, mode=config.rel_time_mode, freqs=int(config.rel_time_freqs)
        )
        self.modality = nn_mod.Parameter(torch.zeros(hidden))
        layer = nn_mod.TransformerEncoderLayer(
            d_model=hidden,
            nhead=int(config.heads),
            dim_feedforward=4 * hidden,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn_mod.TransformerEncoder(
            layer, num_layers=int(config.self_layers), enable_nested_tensor=False
        )
        self.norm = nn_mod.LayerNorm(hidden)

    def forward(self, features: Any, times: Any, mask: Any) -> tuple[Any, Any]:
        torch_mod, _nn, _F = _need_torch_modules()
        if features.ndim != 3 or times.shape != features.shape[:2] or mask.shape != features.shape[:2]:
            raise V05LocalError("TokenSequenceEncoder 输入形状非法")
        mask = mask.bool()
        batch_size, length = mask.shape
        if length < 1:
            raise V05LocalError("token 序列不能为空")
        # 全空行至少保留一个零 token，避免 attention 全 mask 产生 NaN
        empty = ~mask.any(dim=1)
        mask = mask.clone()
        mask[empty, 0] = True
        tokens = self.input_proj(features) + self.time_encoder(times) + self.modality
        tokens = torch_mod.where(mask.unsqueeze(-1), tokens, torch_mod.zeros_like(tokens))
        encoded = self.encoder(tokens, src_key_padding_mask=~mask)
        encoded = self.norm(encoded)
        return encoded, _masked_pool(encoded, mask)


class TemporalOnlyModel(nn.Module if nn is not None else object):  # type: ignore[misc]
    """视觉/音频各自 self-attn + masked pooling，余弦打分。"""

    def __init__(self, config: LocalModelConfig) -> None:
        _torch, nn_mod, _F = _need_torch_modules()
        super().__init__()
        self.config = config
        self.visual_encoder = TokenSequenceEncoder(config.visual_dim, config)
        self.audio_encoder = TokenSequenceEncoder(config.audio_dim, config)
        self.scale = 1.0 / float(config.temperature)

    @property
    def arch(self) -> str:
        return "temporal"

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        _torch, _nn, F_mod = _need_torch_modules()
        q_tokens, q_pool = self.visual_encoder(
            batch["visual"], batch["visual_time"], batch["visual_mask"]
        )
        audio = batch["audio"]
        if audio.ndim != 4:
            raise V05LocalError(f"audio batch 形状应为 (B,K,T,D)，收到 {tuple(audio.shape)}")
        batch_size, candidates, _, _ = audio.shape
        audio_flat = audio.reshape(
            batch_size * candidates, audio.shape[2], audio.shape[3]
        )
        time_flat = batch["audio_time"].reshape(batch_size * candidates, -1)
        mask_flat = batch["audio_mask"].reshape(batch_size * candidates, -1)
        _a_tokens, a_pool = self.audio_encoder(audio_flat, time_flat, mask_flat)
        query = F_mod.normalize(q_pool, dim=-1)
        candidate_emb = F_mod.normalize(a_pool, dim=-1).view(
            batch_size, candidates, -1
        )
        logits = (query.unsqueeze(1) * candidate_emb).sum(dim=-1) * self.scale
        candidate_mask = batch.get("candidate_mask")
        if candidate_mask is not None:
            logits = logits.masked_fill(
                ~candidate_mask.bool(), torch.finfo(logits.dtype).min
            )
        return {
            "query_emb": query,
            "candidate_emb": candidate_emb,
            "logits": logits,
            "visual_tokens": q_tokens,
        }


class _CrossFusionBlock(nn.Module if nn is not None else object):  # type: ignore[misc]
    """双向 cross-attention：视觉 query -> 音频 key；音频 query -> 视觉 key。"""

    def __init__(self, hidden: int, heads: int, dropout: float) -> None:
        _torch, nn_mod, _F = _need_torch_modules()
        super().__init__()
        self.visual_to_audio = nn_mod.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.audio_to_visual = nn_mod.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.visual_norm = nn_mod.LayerNorm(hidden)
        self.audio_norm = nn_mod.LayerNorm(hidden)
        self.dropout = nn_mod.Dropout(dropout)

    def forward(
        self, visual: Any, visual_mask: Any, audio: Any, audio_mask: Any
    ) -> tuple[Any, Any]:
        v_attend, _ = self.visual_to_audio(
            visual,
            audio,
            audio,
            key_padding_mask=~audio_mask,
            need_weights=False,
        )
        visual_new = self.visual_norm(visual + self.dropout(v_attend))
        a_attend, _ = self.audio_to_visual(
            audio,
            visual_new,
            visual_new,
            key_padding_mask=~visual_mask,
            need_weights=False,
        )
        audio_new = self.audio_norm(audio + self.dropout(a_attend))
        return visual_new, audio_new


class CrossAttentionModel(nn.Module if nn is not None else object):  # type: ignore[misc]
    """self-attn 后做双向 cross-attention，再池化融合的候选内配对模型。"""

    def __init__(self, config: LocalModelConfig) -> None:
        _torch, nn_mod, _F = _need_torch_modules()
        super().__init__()
        self.config = config
        self.visual_encoder = TokenSequenceEncoder(config.visual_dim, config)
        self.audio_encoder = TokenSequenceEncoder(config.audio_dim, config)
        self.cross_blocks = nn_mod.ModuleList(
            [
                _CrossFusionBlock(int(config.hidden), int(config.heads), float(config.dropout))
                for _ in range(int(config.cross_layers))
            ]
        )
        self.merge = nn_mod.Sequential(
            nn_mod.Linear(4 * int(config.hidden), int(config.hidden)),
            nn_mod.GELU(),
            nn_mod.LayerNorm(int(config.hidden)),
        )
        self.scale = 1.0 / float(config.temperature)

    @property
    def arch(self) -> str:
        return "cross"

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        _torch, _nn, F_mod = _need_torch_modules()
        visual_tokens, visual_pool = self.visual_encoder(
            batch["visual"], batch["visual_time"], batch["visual_mask"]
        )
        audio = batch["audio"]
        if audio.ndim != 4:
            raise V05LocalError(f"audio batch 形状应为 (B,K,T,D)，收到 {tuple(audio.shape)}")
        batch_size, candidates, audio_len, audio_dim = audio.shape
        visual_len = int(visual_tokens.shape[1])
        hidden = int(visual_tokens.shape[2])
        audio_flat = audio.reshape(batch_size * candidates, audio_len, audio_dim)
        time_flat = batch["audio_time"].reshape(batch_size * candidates, -1)
        mask_flat = batch["audio_mask"].reshape(batch_size * candidates, -1)
        audio_tokens, audio_pool = self.audio_encoder(audio_flat, time_flat, mask_flat)
        visual_mask = batch["visual_mask"].bool()
        visual_mask_expanded = (
            visual_mask.unsqueeze(1).expand(batch_size, candidates, visual_len)
            .reshape(batch_size * candidates, visual_len)
        )
        visual_pair = (
            visual_tokens.unsqueeze(1).expand(batch_size, candidates, visual_len, hidden)
            .reshape(batch_size * candidates, visual_len, hidden)
        )
        audio_pair = audio_tokens
        for block in self.cross_blocks:
            visual_pair, audio_pair = block(
                visual_pair,
                visual_mask_expanded,
                audio_pair,
                mask_flat,
            )
        visual_attended = _masked_pool(visual_pair, visual_mask_expanded)
        audio_attended = _masked_pool(audio_pair, mask_flat)
        visual_base = (
            visual_pool.unsqueeze(1).expand(batch_size, candidates, hidden)
            .reshape(batch_size * candidates, hidden)
        )
        merged = self.merge(
            _torch.cat(
                [visual_base, audio_pool, visual_attended, audio_attended], dim=-1
            )
        )
        query = F_mod.normalize(visual_pool, dim=-1)
        candidate_emb = F_mod.normalize(merged, dim=-1).view(
            batch_size, candidates, -1
        )
        logits = (query.unsqueeze(1) * candidate_emb).sum(dim=-1) * self.scale
        candidate_mask = batch.get("candidate_mask")
        if candidate_mask is not None:
            logits = logits.masked_fill(
                ~candidate_mask.bool(), torch.finfo(logits.dtype).min
            )
        return {
            "query_emb": query,
            "candidate_emb": candidate_emb,
            "logits": logits,
            "visual_tokens": visual_tokens,
        }


def build_local_model(arch: str, config: LocalModelConfig) -> Any:
    _require_torch()
    if arch == "temporal":
        return TemporalOnlyModel(config)
    if arch == "cross":
        return CrossAttentionModel(config)
    raise V05LocalError(f"arch 必须是 {ARCHS}，收到 {arch!r}")


def info_nce_loss(
    logits: Any, labels: Any, *, candidate_mask: Any | None = None
) -> Any:
    """标准 InfoNCE（每个 anchor 在 K 个候选中选正例）。"""
    _torch, _nn, F_mod = _need_torch_modules()
    if logits.ndim != 2:
        raise V05LocalError(f"logits 必须是 (B,K)，收到 {tuple(logits.shape)}")
    labels = labels.long().reshape(-1)
    if int(labels.shape[0]) != int(logits.shape[0]):
        raise V05LocalError("labels 数量与 logits batch 不一致")
    if candidate_mask is not None:
        candidate_mask = candidate_mask.bool()
        if tuple(candidate_mask.shape) != tuple(logits.shape):
            raise V05LocalError("candidate_mask 形状与 logits 不一致")
        logits = logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)
        if bool((~candidate_mask.gather(1, labels.unsqueeze(1))).any()):
            raise V05LocalError("正例落在被 mask 的候选上")
    return F_mod.cross_entropy(logits, labels)


# ============================================================================
# 评估 / bootstrap / 坍缩 / 消融
# ============================================================================


def _bootstrap_ci_by_work(
    correct_by_work: Mapping[int, tuple[int, int]],
    *,
    bootstrap: int,
    seed: int = 12345,
) -> dict[str, float | None]:
    if int(bootstrap) <= 0 or not correct_by_work:
        return {"mean": None, "low": None, "high": None}
    rng = np.random.default_rng(int(seed))
    works = sorted(int(work_id) for work_id in correct_by_work)
    values: list[float] = []
    for _ in range(int(bootstrap)):
        sample = rng.choice(len(works), size=len(works), replace=True)
        correct = 0
        total = 0
        for index in sample:
            c, t = correct_by_work[works[int(index)]]
            correct += int(c)
            total += int(t)
        if total > 0:
            values.append(float(correct) / float(total))
    if not values:
        return {"mean": None, "low": None, "high": None}
    return {
        "mean": float(np.mean(values)),
        "low": float(np.percentile(values, 2.5)),
        "high": float(np.percentile(values, 97.5)),
    }


def _collapse_metrics(embeddings: np.ndarray) -> dict[str, float | None]:
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return {"mean_pairwise_cosine": None, "effective_rank": None}
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-12)
    similarity = normalized @ normalized.T
    count = matrix.shape[0]
    off_diagonal = (similarity.sum() - float(np.trace(similarity))) / (
        count * (count - 1)
    )
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    total = float(singular.sum())
    if total <= 0:
        effective_rank = 0.0
    else:
        probabilities = singular / total
        probabilities = probabilities[probabilities > 0]
        effective_rank = float(np.exp(-float((probabilities * np.log(probabilities)).sum())))
    return {
        "mean_pairwise_cosine": float(off_diagonal),
        "effective_rank": effective_rank,
    }


def _batch_to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    _torch, _nn, _F = _need_torch_modules()
    output: dict[str, Any] = {}
    for key, value in batch.items():
        if _torch.is_tensor(value):
            output[key] = value.to(device)
        elif isinstance(value, np.ndarray):
            output[key] = _torch.from_numpy(value).to(device)
        else:
            output[key] = value
    return output


def _evaluate_batches(
    model: Any,
    batches: Iterable[Mapping[str, Any]],
    *,
    device: Any,
    k_retrieval: int = DEFAULT_K_RETRIEVAL,
    bootstrap: int = 0,
    bootstrap_seed: int = 12345,
) -> dict[str, Any]:
    _torch, _nn, F_mod = _need_torch_modules()
    model.eval()
    correct_by_work: dict[int, list[int]] = {}
    per_work_correct: dict[int, int] = {}
    per_work_total: dict[int, int] = {}
    embedding_rows: list[np.ndarray] = []
    total = 0
    correct = 0
    correct_k32 = 0
    total_k32 = 0
    with _torch.no_grad():
        for raw_batch in batches:
            batch = _batch_to_device(raw_batch, device)
            outputs = model(batch)
            logits = outputs["logits"]
            labels = batch["labels"].long().reshape(-1)
            predictions = logits.argmax(dim=-1)
            hits = (predictions == labels).detach().cpu().numpy()
            work_ids = [int(value) for value in batch["work_ids"]]
            for work_id, hit in zip(work_ids, hits, strict=False):
                per_work_total[work_id] = per_work_total.get(work_id, 0) + 1
                per_work_correct[work_id] = per_work_correct.get(work_id, 0) + int(hit)
                correct_by_work.setdefault(work_id, [0, 0])[0] += int(hit)
                correct_by_work.setdefault(work_id, [0, 0])[1] += 1
            total += int(hits.size)
            correct += int(hits.sum())
            candidates = int(logits.shape[1])
            k = min(int(k_retrieval), candidates)
            if k >= 1:
                subset = logits[:, :k]
                subset_labels = labels
                eligible = subset_labels < k
                if bool(eligible.any()):
                    subset_hits = (
                        subset[eligible].argmax(dim=-1)
                        == subset_labels[eligible]
                    )
                    correct_k32 += int(subset_hits.sum())
                    total_k32 += int(eligible.sum())
            query = outputs["query_emb"]
            embedding_rows.append(query.detach().cpu().numpy())
    top1 = float(correct) / float(total) if total else None
    k32 = float(correct_k32) / float(total_k32) if total_k32 else None
    embeddings = (
        np.concatenate(embedding_rows, axis=0) if embedding_rows else np.zeros((0, 0))
    )
    by_work = {
        int(work_id): (int(correct_by_work[work_id][0]), int(correct_by_work[work_id][1]))
        for work_id in correct_by_work
    }
    return {
        "top1": top1,
        "recall_at_1": top1,
        "top1_k32": k32,
        "k_retrieval": int(k_retrieval),
        "n_sets": total,
        "per_work": {
            str(work_id): (
                float(per_work_correct[work_id]) / float(per_work_total[work_id])
                if per_work_total[work_id]
                else None
            )
            for work_id in per_work_total
        },
        "bootstrap_ci": _bootstrap_ci_by_work(by_work, bootstrap=bootstrap, seed=bootstrap_seed),
        "collapse": _collapse_metrics(embeddings),
    }


def _ablate_batch(batch: Mapping[str, Any], mode: str) -> dict[str, Any]:
    output = dict(batch)
    if mode == "audio_zero":
        value = batch["audio"]
        output["audio"] = (
            np.zeros_like(value) if isinstance(value, np.ndarray) else torch.zeros_like(value)
        )
    elif mode == "visual_zero":
        value = batch["visual"]
        output["visual"] = (
            np.zeros_like(value) if isinstance(value, np.ndarray) else torch.zeros_like(value)
        )
    elif mode == "time_shift":
        times = batch["audio_time"]
        if isinstance(times, np.ndarray):
            output["audio_time"] = np.clip(times + 1.0, -1.0, 1.0)
        else:
            output["audio_time"] = (times + 1.0).clamp(-1.0, 1.0)
    elif mode != "normal":
        raise V05LocalError(f"未知消融模式: {mode!r}")
    return output


def evaluate_local(
    model: Any,
    batches: Iterable[Mapping[str, Any]],
    *,
    device: Any = "cpu",
    k_retrieval: int = DEFAULT_K_RETRIEVAL,
    bootstrap: int = 0,
    bootstrap_seed: int = 12345,
    ablations: Sequence[str] = ("normal", "audio_zero", "visual_zero", "time_shift"),
) -> dict[str, Any]:
    """固定 K 候选评估：Top1/Recall@1、K=32、bootstrap CI、坍缩与消融。"""
    normal = _evaluate_batches(
        model,
        batches,
        device=device,
        k_retrieval=k_retrieval,
        bootstrap=bootstrap,
        bootstrap_seed=bootstrap_seed,
    )
    result: dict[str, Any] = {
        "normal": normal,
        "top1": normal["top1"],
        "recall_at_1": normal["recall_at_1"],
        "top1_k32": normal["top1_k32"],
        "bootstrap_ci": normal["bootstrap_ci"],
        "collapse": normal["collapse"],
        "per_work": normal["per_work"],
        "ablations": {},
    }
    for mode in ablations:
        if mode == "normal":
            continue
        modified = (_ablate_batch(batch, mode) for batch in batches)
        result["ablations"][mode] = _evaluate_batches(
            model,
            modified,
            device=device,
            k_retrieval=k_retrieval,
            bootstrap=0,
        )
    return result


# ============================================================================
# 训练 / checkpoint
# ============================================================================


def train_local_model(
    model: Any,
    train_batches: Iterable[Mapping[str, Any]],
    val_batches: Iterable[Mapping[str, Any]],
    train_config: LocalTrainConfig,
    *,
    device: Any = "cpu",
    seed: int = 42,
) -> dict[str, Any]:
    """单 seed 训练（InfoNCE），按 val Top1 选最优；CPU smoke 也可跑。"""
    _torch, _nn, _F = _need_torch_modules()
    train_config = train_config.validate()
    if not train_batches:
        raise V05LocalError("train_batches 为空")
    if not val_batches:
        raise V05LocalError("val_batches 为空")
    torch_mod, _, _ = _need_torch_modules()
    torch_mod.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = model.to(device)
    model.train()
    optimizer = torch_mod.optim.AdamW(
        model.parameters(),
        lr=float(train_config.lr),
        weight_decay=float(train_config.weight_decay),
    )
    best_state: dict[str, Any] | None = None
    best_val: float = -math.inf
    wait = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(train_config.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_sets = 0
        for raw_batch in train_batches:
            batch = _batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss = info_nce_loss(
                outputs["logits"],
                batch["labels"],
                candidate_mask=batch.get("candidate_mask"),
            )
            if not bool(torch_mod.isfinite(loss)):
                raise V05LocalError("训练出现非有限 loss")
            loss.backward()
            torch_mod.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(batch["labels"].shape[0])
            total_loss += float(loss.detach().cpu()) * count
            total_sets += count
        val_metrics = _evaluate_batches(
            model,
            val_batches,
            device=device,
            k_retrieval=train_config.k_retrieval,
            bootstrap=0,
        )
        val_top1 = val_metrics["top1"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(1, total_sets),
                "val_top1": val_top1,
                "val_top1_k32": val_metrics["top1_k32"],
            }
        )
        if val_top1 is not None and val_top1 > best_val + 1e-5:
            best_val = float(val_top1)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if int(train_config.patience) > 0 and wait >= int(train_config.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "model": model,
        "history": history,
        "best_val_top1": (None if best_val == -math.inf else float(best_val)),
        "epochs_run": len(history),
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: Any,
    arch: str,
    model_config: LocalModelConfig,
    train_config: LocalTrainConfig,
    seed: int,
    metrics: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    torch_mod = _require_torch()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": arch,
        "seed": int(seed),
        "model_config": model_config.to_dict(),
        "train_config": train_config.to_dict(),
        "state_dict": model.state_dict(),
        "metrics": dict(metrics),
        "extra": dict(extra or {}),
    }
    torch_mod.save(payload, target)
    return target


def load_checkpoint(path: str | Path, *, device: Any = "cpu") -> tuple[Any, dict[str, Any]]:
    torch_mod = _require_torch()
    payload = torch_mod.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise V05LocalError(f"checkpoint 格式非法: {path}")
    arch = str(payload.get("arch", "temporal"))
    model_config = LocalModelConfig(**payload.get("model_config", {}))
    model = build_local_model(arch, model_config)
    model.load_state_dict(payload["state_dict"])
    model = model.to(device)
    return model, payload


# ============================================================================
# 合成数据（CPU smoke / 契约测试）
# ============================================================================


class SyntheticFrameStore:
    """与 FrameDataStore 同接口的合成帧存储（确定性随机特征）。"""

    def __init__(
        self,
        *,
        visual_dim: int,
        audio_dim: int,
        context_seconds: float = DEFAULT_CONTEXT_SECONDS,
        audio_hz: float = DEFAULT_AUDIO_HZ,
        bin_seconds: float = 0.5,
        visual_fps: float = 8.0,
        seed: int = 7,
    ) -> None:
        self.visual_dim = int(visual_dim)
        self.audio_dim = int(audio_dim)
        self.context_seconds = float(context_seconds)
        self.audio_hz = float(audio_hz)
        self.bin_seconds = float(bin_seconds)
        self.visual_fps = float(visual_fps)
        self.seed = int(seed)

    def _rng(self, work_id: int, center_s: float, salt: int) -> np.random.Generator:
        key = (
            self.seed * 1_000_003
            + int(work_id) * 9_176
            + int(round(float(center_s) * 1000.0))
            + salt * 7919
        )
        return np.random.default_rng(abs(key))

    def visual_tokens(
        self, work_id: int, center_s: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = max(2, int(round(self.context_seconds * self.visual_fps)))
        times = np.linspace(-0.95, 0.95, count, dtype=np.float32)
        features = self._rng(work_id, center_s, 1).normal(
            size=(count, self.visual_dim)
        ).astype(np.float32)
        return features, times, np.ones((count,), dtype=bool)

    def audio_tokens(
        self, work_id: int, center_s: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = max(2, int(round(self.context_seconds * self.audio_hz)))
        times = np.linspace(-0.95, 0.95, count, dtype=np.float32)
        features = self._rng(work_id, center_s, 2).normal(
            size=(count, self.audio_dim)
        ).astype(np.float32)
        return features, times, np.ones((count,), dtype=bool)


def make_synthetic_protocol(
    *,
    n_works: int = 6,
    bins_per_work: int = 20,
    anchors_per_work: int = 2,
    candidates: int = 5,
    seed: int = 7,
) -> LocalProtocol:
    """构造 train/val/test 不重叠的合成协议（仅测试/CPU smoke）。"""

    if n_works < 3:
        raise V05LocalError("合成协议至少需要 3 部作品")
    if bins_per_work < candidates + 2:
        raise V05LocalError("bins_per_work 太小")
    rng = np.random.default_rng(int(seed))
    work_ids = list(range(1, n_works + 1))
    train_count = max(1, int(round(n_works * 0.6)))
    val_count = max(1, int(round(n_works * 0.2)))
    if train_count + val_count >= n_works:
        train_count = n_works - 2
    split_by_work = {}
    for position, work_id in enumerate(work_ids):
        if position < train_count:
            split_by_work[work_id] = "train"
        elif position < train_count + val_count:
            split_by_work[work_id] = "val"
        else:
            split_by_work[work_id] = "test"
    records: list[dict[str, Any]] = []
    for work_id in work_ids:
        anchors = rng.choice(
            np.arange(2, bins_per_work - 2),
            size=min(anchors_per_work, max(1, bins_per_work - 4)),
            replace=False,
        )
        for anchor in anchors:
            anchor_bin = int(anchor)
            choices = [value for value in range(bins_per_work) if value != anchor_bin]
            negatives = rng.choice(
                np.asarray(choices), size=int(candidates) - 1, replace=False
            ).astype(int).tolist()
            candidate_bins = [anchor_bin] + [int(value) for value in negatives]
            order = rng.permutation(len(candidate_bins)).tolist()
            shuffled = [candidate_bins[index] for index in order]
            positive_position = order.index(0)
            labels = [0] * len(shuffled)
            labels[positive_position] = 1
            records.append(
                {
                    "work_id": int(work_id),
                    "anchor_bin": anchor_bin,
                    "candidate_bins": [int(value) for value in shuffled],
                    "labels": labels,
                    "split": split_by_work[int(work_id)],
                    "periodic": bool(rng.random() < 0.5),
                }
            )
    metadata = {
        "protocol_version": "v2",
        "bin_seconds": 0.5,
        "context_seconds": DEFAULT_CONTEXT_SECONDS,
        "synthetic": True,
    }
    protocol = LocalProtocol(
        records=tuple(_parse_record(record) for record in records),
        metadata=metadata,
        bin_seconds=0.5,
    )
    protocol.validate_disjoint()
    return protocol


def build_synthetic_batches(
    *,
    protocol: LocalProtocol,
    visual_dim: int,
    audio_dim: int,
    context_seconds: float = DEFAULT_CONTEXT_SECONDS,
    audio_hz: float = DEFAULT_AUDIO_HZ,
    batch_size: int = 4,
    seed: int = 7,
) -> dict[str, list[dict[str, Any]]]:
    """按协议生成 train/val/test 合成 batch 列表（CPU smoke 用）。"""

    store = SyntheticFrameStore(
        visual_dim=visual_dim,
        audio_dim=audio_dim,
        context_seconds=context_seconds,
        audio_hz=audio_hz,
        bin_seconds=protocol.bin_seconds,
        seed=seed,
    )
    dataset = LocalPairDataset(protocol.records, store, bin_seconds=protocol.bin_seconds)
    result: dict[str, list[dict[str, Any]]] = {}
    for name in SPLITS:
        indices = [
            index
            for index, record in enumerate(protocol.records)
            if record.split == name
        ]
        batches: list[dict[str, Any]] = []
        for start in range(0, len(indices), max(1, int(batch_size))):
            chunk = indices[start : start + max(1, int(batch_size))]
            items = [dataset[index] for index in chunk]
            batches.append(collate_local_sets(items))
        result[name] = batches
    return result

