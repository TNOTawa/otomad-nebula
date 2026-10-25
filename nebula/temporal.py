"""v0.4.2 固定时间结构表征（Full Mean / Temporal Stats）。

本模块只消费序列层已经保存好的片段向量（nebula.representations.ClipSequence），
纯 numpy 实现，不加载 PE-AV、不重新解码、不访问网络；同一输入的输出完全确定。

两个聚合器（对应设计文档 §7.2）：

assemble_full_mean(seq)
    对每个模态：逐行 L2 归一化 -> 不加权算术平均 -> L2 归一化。与旧版
    embed.aggregate_work_vector 完全一致，作为“完整覆盖”单一变量基线。

assemble_temporal_stats(seq, stages=4)
    每个模态独立构造 5 个语义组，joint 在有事件特征时追加第 6 组：
    每组先 L2 归一化（全零组保持全零）再乘组权重，拼接后整体 L2 归一化。

Temporal Stats 分组与固定统计量顺序（dim 契约，保证可复核）：

- G1 semantic_global（dim=clip_dim，权重 1.0）
    按片段 weight 加权的 normalized 片段平均 -> L2。
- G2 stages（dim=stages*clip_dim，权重 1.0）
    每个相对时间阶段 [k/stages, (k+1)/stages] * duration 内，按“窗口与阶段区间的
    重叠时长”加权求平均 -> 每阶段 L2 -> 按阶段顺序拼接 -> 整组 L2。
- G3 drift（dim=clip_dim，权重 0.8）
    normalize(mean(后半片段) - mean(前半片段))；n<2 时为零向量。
    对片段顺序反转敏感（符号翻转）。
- G4 dynamics（dim=12，权重 0.6）
    相邻片段 cos 统计量，固定顺序：
    [cos_mean, cos_std, cos_min, cos_max, cos_q10, cos_q90,
     frac_cos_lt_0.5, lag2_cos_mean, lag4_cos_mean,
     diff_norm_mean, diff_norm_std, diff_norm_max]
    设计文档列出 q50 但明确写 dim=12；median 与 mean 高度相关，故省略 q50。
- G5 self_similarity（dim=8，权重 0.5）
    lag 属于 (1, 2, 3, 4, 6, 8, 12, 16) 的平均 cos；无有效片段对填 0。
- G6 events（dim=(2+stages)*32=192，权重 0.8，仅 joint 且事件特征存在）
    32 维窗口特征的整片 mean(32) + std(32) + 各阶段重叠加权均值(32*stages)。

无事件特征时使用 semantic-only 变体（同样 5 组），由构建工具在 representation
params 里记录 has_events=false；两种变体的向量维度不同，禁止混用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .embed import aggregate_work_vector
from .errors import EmbeddingError
from .representations import ClipSequence, ClipTime

#: 聚合器版本标签（写入 representation 元数据）
FULL_MEAN_AGGREGATION = "full_mean_v1"
TEMPORAL_STATS_AGGREGATION = "temporal_stats_v1"

#: 支持的模态（与 DB representation_vectors.kind 一致）
VECTOR_MODALITIES: tuple[str, ...] = ("joint", "audio", "video")

#: 阶段数（与 config.temporal.stages 的合法范围一致）
STAGE_COUNT_DEFAULT = 4
MIN_STAGES = 2
MAX_STAGES = 8

#: 事件窗口特征维度（nebula/events.py 的 WINDOW_FEATURE_NAMES）
EVENT_FEATURE_DIM = 32

#: 自相似度使用的 lag 集合（顺序固定）
SELF_SIMILARITY_LAGS: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16)
DYNAMICS_DIM = 12
SELF_SIMILARITY_DIM = len(SELF_SIMILARITY_LAGS)

_EPS = 1e-12


@dataclass(frozen=True)
class TemporalGroup:
    """Temporal Stats 的一个组：名称、权重、适用模态与说明。"""

    name: str
    weight: float
    modalities: tuple[str, ...]
    description: str


#: 组顺序即最终拼接顺序；权重按设计文档 §7.2 固定。
TEMPORAL_GROUPS: tuple[TemporalGroup, ...] = (
    TemporalGroup(
        name="semantic_global",
        weight=1.0,
        modalities=VECTOR_MODALITIES,
        description="按片段 weight 加权的语义均值",
    ),
    TemporalGroup(
        name="stages",
        weight=1.0,
        modalities=VECTOR_MODALITIES,
        description="相对时间阶段内按重叠时长加权的均值",
    ),
    TemporalGroup(
        name="drift",
        weight=0.8,
        modalities=VECTOR_MODALITIES,
        description="后半段与前半段均值之差（顺序敏感）",
    ),
    TemporalGroup(
        name="dynamics",
        weight=0.6,
        modalities=VECTOR_MODALITIES,
        description="相邻片段的 cos/差分统计量（12 维）",
    ),
    TemporalGroup(
        name="self_similarity",
        weight=0.5,
        modalities=VECTOR_MODALITIES,
        description="多 lag 自相似度（8 维）",
    ),
    TemporalGroup(
        name="events",
        weight=0.8,
        modalities=("joint",),
        description="高频事件窗口特征的 mean/std/阶段均值（仅 joint）",
    ),
)

GROUP_ORDER: tuple[str, ...] = tuple(group.name for group in TEMPORAL_GROUPS)
GROUP_WEIGHTS: dict[str, float] = {group.name: group.weight for group in TEMPORAL_GROUPS}


# --------------------------------------------------------------------------- 元数据
def _validated_stages(stages: int) -> int:
    if isinstance(stages, bool) or not isinstance(stages, int):
        raise EmbeddingError(f"stages 必须是整数，收到: {stages!r}")
    if not MIN_STAGES <= stages <= MAX_STAGES:
        raise EmbeddingError(f"stages 必须在 {MIN_STAGES}..{MAX_STAGES} 之间，收到: {stages}")
    return stages


def _validated_clip_dim(clip_dim: int) -> int:
    if isinstance(clip_dim, bool) or not isinstance(clip_dim, int) or clip_dim < 1:
        raise EmbeddingError(f"clip_dim 必须是正整数，收到: {clip_dim!r}")
    return clip_dim


def group_dims(
    *,
    clip_dim: int = 1024,
    stages: int = STAGE_COUNT_DEFAULT,
    has_events: bool = False,
) -> dict[str, int]:
    """返回每个组在给定参数下的维度（events 组在无事件时为 0）。"""

    dim = _validated_clip_dim(clip_dim)
    stage_count = _validated_stages(stages)
    return {
        "semantic_global": dim,
        "stages": stage_count * dim,
        "drift": dim,
        "dynamics": DYNAMICS_DIM,
        "self_similarity": SELF_SIMILARITY_DIM,
        "events": (2 + stage_count) * EVENT_FEATURE_DIM if has_events else 0,
    }


def vector_dim(
    kind: str = "joint",
    *,
    clip_dim: int = 1024,
    stages: int = STAGE_COUNT_DEFAULT,
    has_events: bool = False,
) -> int:
    """返回某模态 Temporal Stats 向量的维度。"""

    if kind not in VECTOR_MODALITIES:
        raise EmbeddingError(f"未知表征向量种类: {kind}")
    dims = group_dims(clip_dim=clip_dim, stages=stages, has_events=has_events)
    total = (
        dims["semantic_global"]
        + dims["stages"]
        + dims["drift"]
        + dims["dynamics"]
        + dims["self_similarity"]
    )
    if kind == "joint" and has_events:
        total += dims["events"]
    return total


def group_metadata(
    *,
    clip_dim: int = 1024,
    stages: int = STAGE_COUNT_DEFAULT,
    has_events: bool = False,
) -> dict[str, dict[str, Any]]:
    """返回写入 representation params 的组元数据（顺序与权重固定）。"""

    dims = group_dims(clip_dim=clip_dim, stages=stages, has_events=has_events)
    return {
        group.name: {
            "weight": group.weight,
            "modalities": list(group.modalities),
            "dim": dims[group.name],
            "description": group.description,
        }
        for group in TEMPORAL_GROUPS
    }


def full_mean_params(
    *, clip_dim: int = 1024, sequence_version: str | None = None
) -> dict[str, Any]:
    """Full Mean 表征的 DB params（重建时可据此复现）。"""

    dim = _validated_clip_dim(clip_dim)
    return {
        "aggregation": FULL_MEAN_AGGREGATION,
        "sequence_version": sequence_version,
        "clip_dim": dim,
        "modalities": list(VECTOR_MODALITIES),
        "weighting": "uniform",
        "normalization": "l2-normalize-mean-l2-normalize",
        "vector_dims": {kind: dim for kind in VECTOR_MODALITIES},
    }


def temporal_stats_params(
    *,
    stages: int = STAGE_COUNT_DEFAULT,
    has_events: bool = False,
    event_version: str | None = None,
    clip_dim: int = 1024,
    sequence_version: str | None = None,
) -> dict[str, Any]:
    """Temporal Stats 表征的 DB params（包含变体、分组权重与维度）。"""

    stage_count = _validated_stages(stages)
    dim = _validated_clip_dim(clip_dim)
    event_dim = (2 + stage_count) * EVENT_FEATURE_DIM if has_events else 0
    vector_dims = {
        kind: vector_dim(kind, clip_dim=dim, stages=stage_count, has_events=has_events)
        for kind in VECTOR_MODALITIES
    }
    return {
        "aggregation": TEMPORAL_STATS_AGGREGATION,
        "sequence_version": sequence_version,
        "clip_dim": dim,
        "stages": stage_count,
        "stage_edges_relative": [round(index / stage_count, 9) for index in range(stage_count + 1)],
        "has_events": bool(has_events),
        "event_version": event_version if has_events else None,
        "event_feature_dim": event_dim,
        "group_order": list(GROUP_ORDER),
        "group_weights": dict(GROUP_WEIGHTS),
        "groups": group_metadata(clip_dim=dim, stages=stage_count, has_events=has_events),
        "vector_dims": vector_dims,
        "normalization": "per-group-l2-then-weight-then-final-l2",
    }


# --------------------------------------------------------------------------- 聚合
def assemble_full_mean(seq: ClipSequence) -> dict[str, np.ndarray]:
    """Full Mean：逐行归一化 -> 不加权平均 -> L2 归一化（逐模态）。

    与旧版 embed.aggregate_work_vector 公式一致；缺少的模态不生成向量，
    joint 必须存在。
    """

    arrays = _validated_clip_arrays(seq)
    return {
        kind: np.asarray(aggregate_work_vector(array), dtype=np.float32)
        for kind, array in arrays.items()
    }


def assemble_temporal_stats(
    seq: ClipSequence,
    *,
    stages: int = STAGE_COUNT_DEFAULT,
) -> dict[str, np.ndarray]:
    """Temporal Stats：固定时间结构分组聚合（逐模态），返回 L2 范数为 1 的向量。

    有事件特征的 joint 变体维度为 vector_dim("joint", has_events=True)，
    其余为无事件维度；两种变体禁止混用。
    """

    stage_count = _validated_stages(stages)
    arrays = _validated_clip_arrays(seq)
    duration = _sequence_duration(seq)
    weights = _clip_weights(seq)

    result: dict[str, np.ndarray] = {}
    for kind in VECTOR_MODALITIES:
        matrix = arrays.get(kind)
        if matrix is None:
            continue
        normalized = _normalize_rows(matrix)
        clip_dim = normalized.shape[1]
        groups: list[np.ndarray] = [
            _semantic_global_group(normalized, weights),
            _stages_group(normalized, seq.times, duration, stage_count),
            _drift_group(normalized),
            _dynamics_group(normalized),
            _self_similarity_group(normalized),
        ]
        has_events = kind == "joint" and seq.event_features is not None
        if has_events:
            events = _event_matrix(seq.event_features, n_clips=seq.n_clips)
            groups.append(_events_group(events, seq.times, duration, stage_count))
        combined = _combine_groups(groups)
        expected = vector_dim(
            kind, clip_dim=clip_dim, stages=stage_count, has_events=has_events
        )
        if combined.size != expected:  # pragma: no cover - 防御性检查
            raise EmbeddingError(
                f"{kind} Temporal Stats 维度异常: {combined.size} != {expected}"
            )
        result[kind] = combined.astype(np.float32)
    return result


# --------------------------------------------------------------------------- 校验
def _validated_clip_arrays(seq: ClipSequence) -> dict[str, np.ndarray]:
    """取出存在的模态矩阵并校验形状/数值；joint 必须存在。"""

    n_clips = seq.n_clips
    if n_clips < 1:
        raise EmbeddingError(f"作品 {seq.work_id} 没有片段记录，无法生成表征")
    if seq.joint is None:
        raise EmbeddingError(f"作品 {seq.work_id} 缺少 joint 片段向量，无法生成表征")
    arrays: dict[str, np.ndarray] = {}
    for kind in VECTOR_MODALITIES:
        value = seq.clip_array(kind)
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 2:
            raise EmbeddingError(f"作品 {seq.work_id} 的 {kind} 片段矩阵必须是二维，收到 {array.shape}")
        if array.shape[0] != n_clips:
            raise EmbeddingError(
                f"作品 {seq.work_id} 的 {kind} 片段行数 {array.shape[0]} 与片段记录数 {n_clips} 不一致"
            )
        if array.shape[1] < 1:
            raise EmbeddingError(f"作品 {seq.work_id} 的 {kind} 片段维度为空")
        if not np.all(np.isfinite(array)):
            raise EmbeddingError(f"作品 {seq.work_id} 的 {kind} 片段向量包含 NaN/Inf")
        arrays[kind] = array
    dims = {kind: array.shape[1] for kind, array in arrays.items()}
    if len(set(dims.values())) > 1:  # pragma: no cover - PE-AV 三模态同维
        raise EmbeddingError(f"作品 {seq.work_id} 的多模态片段维度不一致: {dims}")
    return arrays


def _clip_weights(seq: ClipSequence) -> np.ndarray:
    """片段权重（归一化到和为 1；旧序列缺失 weight 时退化为等权）。"""

    weights = np.asarray(seq.weights(), dtype=np.float64)
    if weights.shape != (seq.n_clips,):
        raise EmbeddingError(
            f"作品 {seq.work_id} 的片段权重形状异常: {weights.shape} != ({seq.n_clips},)"
        )
    total = float(weights.sum())
    if not np.all(np.isfinite(weights)) or total <= _EPS:
        raise EmbeddingError(f"作品 {seq.work_id} 的片段权重非法")
    return weights / total


def _sequence_duration(seq: ClipSequence) -> float:
    """作品时长（秒）：优先 DB/manifest 记录，缺失时取最后一个窗口的结束时间。"""

    duration = seq.duration_s
    if duration is not None:
        value = float(duration)
        if math.isfinite(value) and value > 0:
            return value
    if seq.times:
        last_end = max(float(clip_time.end_s) for clip_time in seq.times)
        if math.isfinite(last_end) and last_end > 0:
            return last_end
    raise EmbeddingError(f"作品 {seq.work_id} 缺少有效时长，无法划分时间阶段")


def _event_matrix(value: Any, *, n_clips: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != n_clips or array.shape[1] != EVENT_FEATURE_DIM:
        raise EmbeddingError(
            f"事件特征形状异常: {array.shape}（期望 ({n_clips}, {EVENT_FEATURE_DIM})）"
        )
    if not np.all(np.isfinite(array)):
        raise EmbeddingError("事件特征包含 NaN/Inf")
    return array


# --------------------------------------------------------------------------- 组计算
def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1)
    bad = (~np.isfinite(norms)) | (norms <= _EPS)
    if np.any(bad):
        row = int(np.flatnonzero(bad)[0])
        raise EmbeddingError(f"片段向量第 {row} 行范数为 0 或非有限，无法归一化")
    return matrix / norms[:, None]


def _unit(vector: np.ndarray, *, what: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= _EPS:
        raise EmbeddingError(f"{what}向量范数为 0，无法归一化")
    return vector / norm


def _unit_or_zero(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= _EPS:
        return np.zeros_like(vector)
    return vector / norm


def _semantic_global_group(normalized: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = normalized.T @ weights
    return _unit(mean, what="G1 semantic_global ")


def _stage_overlaps(
    times: Sequence[ClipTime], duration: float, stages: int
) -> np.ndarray:
    """(stages, n_clips) 的重叠时长矩阵（秒）；阶段边界按相对时长均分。"""

    starts = np.asarray([float(clip_time.start_s) for clip_time in times], dtype=np.float64)
    ends = starts + np.asarray(
        [float(clip_time.duration_s) for clip_time in times], dtype=np.float64
    )
    edges = np.linspace(0.0, duration, stages + 1)
    overlaps = np.zeros((stages, starts.size), dtype=np.float64)
    for index in range(stages):
        low, high = float(edges[index]), float(edges[index + 1])
        overlaps[index] = np.clip(
            np.minimum(ends, high) - np.maximum(starts, low), 0.0, None
        )
    return overlaps


def _stages_group(
    normalized: np.ndarray,
    times: Sequence[ClipTime],
    duration: float,
    stages: int,
) -> np.ndarray:
    overlaps = _stage_overlaps(times, duration, stages)
    width = normalized.shape[1]
    parts: list[np.ndarray] = []
    for index in range(stages):
        weights = overlaps[index]
        total = float(weights.sum())
        if total <= _EPS:  # 该阶段没有任何窗口覆盖：保持零块（确定性）
            parts.append(np.zeros(width, dtype=np.float64))
            continue
        parts.append(_unit_or_zero(normalized.T @ (weights / total)))
    return _unit_or_zero(np.concatenate(parts))


def _drift_group(normalized: np.ndarray) -> np.ndarray:
    count = normalized.shape[0]
    if count < 2:
        return np.zeros(normalized.shape[1], dtype=np.float64)
    half = count // 2
    first = normalized[:half].mean(axis=0)
    second = normalized[half:].mean(axis=0)
    return _unit_or_zero(second - first)


def _cosine_rows(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.clip(np.sum(first * second, axis=1), -1.0, 1.0)


def _lag_cosine_mean(normalized: np.ndarray, lag: int) -> float:
    count = normalized.shape[0]
    if lag < 1 or lag >= count:
        return 0.0
    values = _cosine_rows(normalized[:-lag], normalized[lag:])
    if values.size == 0:  # pragma: no cover - 上面的条件已排除
        return 0.0
    return float(values.mean())


def _dynamics_group(normalized: np.ndarray) -> np.ndarray:
    """G4：12 维，固定顺序见模块 docstring。"""

    count = normalized.shape[0]
    if count < 2:
        return np.zeros(DYNAMICS_DIM, dtype=np.float64)
    cosine = _cosine_rows(normalized[:-1], normalized[1:])
    q10, q90 = np.percentile(cosine, [10.0, 90.0])
    differences = normalized[1:] - normalized[:-1]
    diff_norms = np.linalg.norm(differences, axis=1)
    return np.asarray(
        [
            float(cosine.mean()),
            float(cosine.std()),
            float(cosine.min()),
            float(cosine.max()),
            float(q10),
            float(q90),
            float((cosine < 0.5).mean()),
            _lag_cosine_mean(normalized, 2),
            _lag_cosine_mean(normalized, 4),
            float(diff_norms.mean()),
            float(diff_norms.std()),
            float(diff_norms.max()),
        ],
        dtype=np.float64,
    )


def _self_similarity_group(normalized: np.ndarray) -> np.ndarray:
    return np.asarray(
        [_lag_cosine_mean(normalized, lag) for lag in SELF_SIMILARITY_LAGS],
        dtype=np.float64,
    )


def _events_group(
    events: np.ndarray,
    times: Sequence[ClipTime],
    duration: float,
    stages: int,
) -> np.ndarray:
    """G6：整片 mean/std + 各阶段重叠加权均值（与 G2 的阶段口径一致）。"""

    overlaps = _stage_overlaps(times, duration, stages)
    parts: list[np.ndarray] = [events.mean(axis=0), events.std(axis=0)]
    for index in range(stages):
        weights = overlaps[index]
        total = float(weights.sum())
        if total <= _EPS:
            parts.append(np.zeros(EVENT_FEATURE_DIM, dtype=np.float64))
            continue
        parts.append(events.T @ (weights / total))
    return _unit_or_zero(np.concatenate(parts))


def _combine_groups(groups: list[np.ndarray]) -> np.ndarray:
    specs = TEMPORAL_GROUPS[: len(groups)]
    parts: list[np.ndarray] = []
    for spec, group in zip(specs, groups, strict=True):
        parts.append(_unit_or_zero(group) * spec.weight)
    return _unit(np.concatenate(parts), what="Temporal Stats 拼接")
