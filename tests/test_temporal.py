"""nebula.temporal 的公式、维度契约、顺序敏感性与确定性测试。

覆盖设计文档 §7.2 / §9：
- Full Mean 与旧 aggregate_work_vector 公式一致、对片段顺序/权重不变；
- Temporal Stats 组顺序、12 维 dynamics / 8 维 self-similarity 固定顺序；
- 组先 L2 再乘权重、整体 L2 的拼接公式（手工可复核小例子）；
- 片段顺序反转时 Full Mean 近似不变、Temporal Stats 显著变化；
- 缺失模态 / 缺失事件特征 / 非法输入；
- 确定性（同一输入重复调用逐元素相等）与维度契约（6164 / 6356）。
"""

from __future__ import annotations

import numpy as np
import pytest

from nebula.embed import aggregate_work_vector
from nebula.errors import EmbeddingError
from nebula.representations import ClipSequence, ClipTime
from nebula.temporal import (
    DYNAMICS_DIM,
    EVENT_FEATURE_DIM,
    GROUP_ORDER,
    GROUP_WEIGHTS,
    SELF_SIMILARITY_DIM,
    SELF_SIMILARITY_LAGS,
    STAGE_COUNT_DEFAULT,
    TEMPORAL_STATS_AGGREGATION,
    assemble_full_mean,
    assemble_temporal_stats,
    full_mean_params,
    group_dims,
    group_metadata,
    temporal_stats_params,
    vector_dim,
)


# --------------------------------------------------------------------------- 测试工具
def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix / np.linalg.norm(matrix, axis=1)[:, None]


def _make_sequence(
    joint: np.ndarray,
    *,
    audio: np.ndarray | None = None,
    video: np.ndarray | None = None,
    times: tuple[ClipTime, ...] | None = None,
    duration_s: float | None = None,
    weights: list[float] | None = None,
    event_features: np.ndarray | None = None,
    work_id: int = 1,
) -> ClipSequence:
    joint = np.asarray(joint, dtype=np.float32)
    n_clips = joint.shape[0]
    if times is None:
        if weights is None:
            weights = [1.0 / n_clips] * n_clips
        times = tuple(
            ClipTime(
                index=index,
                start_s=float(index * 8.0),
                duration_s=8.0,
                valid_duration_s=8.0,
                weight=float(weights[index]),
            )
            for index in range(n_clips)
        )
    if duration_s is None:
        duration_s = max(time.end_s for time in times)
    return ClipSequence(
        work_id=work_id,
        version="test-sequence-v1",
        duration_s=duration_s,
        times=tuple(times),
        joint=joint,
        audio=None if audio is None else np.asarray(audio, dtype=np.float32),
        video=None if video is None else np.asarray(video, dtype=np.float32),
        event_features=(
            None if event_features is None else np.asarray(event_features, dtype=np.float32)
        ),
        event_version="events_v1" if event_features is not None else None,
    )


def _combine_expected(groups: list[np.ndarray]) -> np.ndarray:
    """测试侧按契约独立实现：组先 L2，再乘权重，拼接后整体 L2。"""

    weights = [GROUP_WEIGHTS[name] for name in GROUP_ORDER[: len(groups)]]
    parts = []
    for group, weight in zip(groups, weights, strict=True):
        norm = float(np.linalg.norm(group))
        unit = group / norm if norm > 1e-12 else np.zeros_like(group)
        parts.append(unit * weight)
    combined = np.concatenate(parts)
    return combined / np.linalg.norm(combined)


def _g4_expected(normalized: np.ndarray) -> np.ndarray:
    """固定 12 维顺序的独立实现。"""

    cosine = np.clip(np.sum(normalized[:-1] * normalized[1:], axis=1), -1.0, 1.0)
    q10, q90 = np.percentile(cosine, [10.0, 90.0])
    diff_norms = np.linalg.norm(normalized[1:] - normalized[:-1], axis=1)
    return np.asarray(
        [
            cosine.mean(),
            cosine.std(),
            cosine.min(),
            cosine.max(),
            q10,
            q90,
            (cosine < 0.5).mean(),
            _lag_mean(normalized, 2),
            _lag_mean(normalized, 4),
            diff_norms.mean(),
            diff_norms.std(),
            diff_norms.max(),
        ]
    )


def _lag_mean(normalized: np.ndarray, lag: int) -> float:
    if lag < 1 or lag >= normalized.shape[0]:
        return 0.0
    return float(np.sum(normalized[:-lag] * normalized[lag:], axis=1).mean())


def _g5_expected(normalized: np.ndarray) -> np.ndarray:
    return np.asarray([_lag_mean(normalized, lag) for lag in SELF_SIMILARITY_LAGS])


# --------------------------------------------------------------------------- 元数据契约
def test_vector_dim_contract():
    assert STAGE_COUNT_DEFAULT == 4
    assert SELF_SIMILARITY_DIM == 8
    assert DYNAMICS_DIM == 12
    assert EVENT_FEATURE_DIM == 32
    # 1024 维 PE-AV 片段：1024 + 4*1024 + 1024 + 12 + 8 = 6164
    assert vector_dim("joint", has_events=False) == 6164
    assert vector_dim("audio", has_events=False) == 6164
    assert vector_dim("video", has_events=False) == 6164
    # 仅 joint 追加 events 组：(2+4)*32 = 192 -> 6356
    assert vector_dim("joint", has_events=True) == 6356
    assert vector_dim("audio", has_events=True) == 6164
    # stages 变化时 stages 组维度随之变化
    assert vector_dim("joint", stages=2, has_events=False) == 1024 + 2 * 1024 + 1024 + 12 + 8
    assert vector_dim("joint", clip_dim=4, stages=4, has_events=False) == 4 * 6 + 20
    assert vector_dim("joint", clip_dim=4, stages=4, has_events=True) == 4 * 6 + 20 + 192
    with pytest.raises(EmbeddingError):
        vector_dim("bad-kind")


def test_group_metadata_contract():
    assert GROUP_ORDER == (
        "semantic_global",
        "stages",
        "drift",
        "dynamics",
        "self_similarity",
        "events",
    )
    assert GROUP_WEIGHTS == {
        "semantic_global": 1.0,
        "stages": 1.0,
        "drift": 0.8,
        "dynamics": 0.6,
        "self_similarity": 0.5,
        "events": 0.8,
    }
    dims = group_dims(clip_dim=8, stages=4, has_events=False)
    assert dims["semantic_global"] == 8
    assert dims["stages"] == 32
    assert dims["drift"] == 8
    assert dims["dynamics"] == 12
    assert dims["self_similarity"] == 8
    assert dims["events"] == 0
    assert group_dims(clip_dim=8, stages=4, has_events=True)["events"] == 192

    metadata = group_metadata(clip_dim=8, stages=4, has_events=True)
    assert set(metadata) == set(GROUP_ORDER)
    assert metadata["events"]["modalities"] == ["joint"]
    assert metadata["semantic_global"]["modalities"] == ["joint", "audio", "video"]
    assert metadata["events"]["dim"] == 192
    assert metadata["drift"]["weight"] == 0.8


def test_representation_params_record_formula():
    full = full_mean_params(clip_dim=8, sequence_version="seq-v1")
    assert full["aggregation"] == "full_mean_v1"
    assert full["sequence_version"] == "seq-v1"
    assert full["clip_dim"] == 8
    assert full["weighting"] == "uniform"
    assert full["vector_dims"] == {"joint": 8, "audio": 8, "video": 8}

    semantic = temporal_stats_params(
        stages=4, has_events=False, clip_dim=8, sequence_version="seq-v1"
    )
    assert semantic["aggregation"] == TEMPORAL_STATS_AGGREGATION
    assert semantic["sequence_version"] == "seq-v1"
    assert semantic["stages"] == 4
    assert semantic["stage_edges_relative"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert semantic["has_events"] is False
    assert semantic["event_version"] is None
    assert semantic["event_feature_dim"] == 0
    assert semantic["group_order"] == list(GROUP_ORDER)
    assert semantic["vector_dims"] == {"joint": 68, "audio": 68, "video": 68}
    assert semantic["groups"]["events"]["dim"] == 0

    with_events = temporal_stats_params(
        stages=2,
        has_events=True,
        event_version="events_v1",
        clip_dim=8,
        sequence_version="seq-v1",
    )
    assert with_events["has_events"] is True
    assert with_events["event_version"] == "events_v1"
    assert with_events["event_feature_dim"] == (2 + 2) * EVENT_FEATURE_DIM
    assert with_events["vector_dims"]["joint"] == vector_dim(
        "joint", clip_dim=8, stages=2, has_events=True
    )
    assert with_events["vector_dims"]["audio"] == vector_dim(
        "audio", clip_dim=8, stages=2, has_events=False
    )

    with pytest.raises(EmbeddingError):
        temporal_stats_params(stages=1)
    with pytest.raises(EmbeddingError):
        temporal_stats_params(stages=9)
    with pytest.raises(EmbeddingError):
        full_mean_params(clip_dim=0)


# --------------------------------------------------------------------------- Full Mean
def test_full_mean_matches_legacy_aggregation():
    rng = np.random.default_rng(7)
    joint = rng.normal(size=(5, 8)).astype(np.float32)
    audio = rng.normal(size=(5, 8)).astype(np.float32)
    video = rng.normal(size=(5, 8)).astype(np.float32)
    seq = _make_sequence(joint, audio=audio, video=video)
    assembled = assemble_full_mean(seq)
    assert set(assembled) == {"joint", "audio", "video"}
    for kind, matrix in (("joint", joint), ("audio", audio), ("video", video)):
        expected = aggregate_work_vector(matrix)
        np.testing.assert_allclose(assembled[kind], expected, atol=1e-7)
        assert assembled[kind].dtype == np.float32
        assert assembled[kind].shape == (8,)
        assert np.isclose(np.linalg.norm(assembled[kind]), 1.0, atol=1e-6)


def test_full_mean_is_order_and_weight_invariant():
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(6, 8))
    seq = _make_sequence(matrix)
    permuted = _make_sequence(matrix[[3, 0, 5, 1, 4, 2]])
    base = assemble_full_mean(seq)["joint"]
    shuffled = assemble_full_mean(permuted)["joint"]
    assert float(base @ shuffled) > 0.999999

    # Full Mean 明确不加权：权重改变不影响结果
    weighted = _make_sequence(matrix, weights=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    np.testing.assert_array_equal(assemble_full_mean(seq)["joint"], assemble_full_mean(weighted)["joint"])


def test_full_mean_missing_modalities_and_invalid_input():
    matrix = np.eye(4, dtype=np.float32)
    only_joint = assemble_full_mean(_make_sequence(matrix))
    assert set(only_joint) == {"joint"}
    no_video = assemble_full_mean(_make_sequence(matrix, audio=matrix))
    assert set(no_video) == {"joint", "audio"}

    with pytest.raises(EmbeddingError):
        assemble_full_mean(_make_sequence(matrix, video=matrix[[0, 1]]))
    with pytest.raises(EmbeddingError):
        assemble_full_mean(_make_sequence(np.zeros((2, 4), dtype=np.float32)))
    with pytest.raises(EmbeddingError):
        assemble_full_mean(_empty_sequence())


def _empty_sequence() -> ClipSequence:
    return ClipSequence(
        work_id=99,
        version="empty",
        duration_s=8.0,
        times=(),
        joint=np.zeros((0, 4), dtype=np.float32),
        audio=None,
        video=None,
    )


# --------------------------------------------------------------------------- Temporal Stats
def test_temporal_stats_contract_and_determinism():
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(9, 8)).astype(np.float32)
    seq = _make_sequence(matrix, audio=matrix, video=matrix, duration_s=72.0)
    result = assemble_temporal_stats(seq)
    assert set(result) == {"joint", "audio", "video"}
    for kind in ("joint", "audio", "video"):
        vector = result[kind]
        assert vector.dtype == np.float32
        assert vector.shape == (vector_dim(kind, clip_dim=8, stages=4, has_events=False),)
        assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)
    repeat = assemble_temporal_stats(seq)
    for kind in result:
        np.testing.assert_array_equal(result[kind], repeat[kind])


def test_temporal_stats_hand_computed_formula_and_group_weights():
    # 3 个 2 秒窗口：e0, e1, e0；duration=6；2 个阶段
    e0 = np.array([1.0, 0.0])
    e1 = np.array([0.0, 1.0])
    matrix = np.stack([e0, e1, e0])
    times = tuple(
        ClipTime(index=index, start_s=index * 2.0, duration_s=2.0, valid_duration_s=2.0, weight=1 / 3)
        for index in range(3)
    )
    seq = _make_sequence(matrix, times=times, duration_s=6.0)
    result = assemble_temporal_stats(seq, stages=2)
    normalized = _row_normalize(matrix)

    # G1: 均匀权重均值 -> L2
    g1 = _unit((e0 + e1 + e0) / 3.0)
    # G2: stage0 覆盖 e0*2s + e1*1s；stage1 覆盖 e1*1s + e0*2s -> 两阶段都同向
    g2 = _unit(np.concatenate([_unit((2 * e0 + e1) / 3.0), _unit((2 * e0 + e1) / 3.0)]))
    # G3: mean(e1, e0) - e0
    g3 = _unit((e1 + e0) / 2.0 - e0)
    # G4: 手工 12 维顺序
    expected_g4 = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, np.sqrt(2), 0.0, np.sqrt(2)]
    )
    g4 = expected_g4 / np.linalg.norm(expected_g4)
    # G5: lag1=0, lag2=1, 其余 0
    expected_g5 = np.zeros(8)
    expected_g5[0] = 0.0
    expected_g5[1] = 1.0
    g5 = expected_g5

    expected = _combine_expected([g1, g2, g3, g4, g5])
    np.testing.assert_allclose(result["joint"], expected, atol=1e-5)
    # 组件维度和 6164-like 的契约
    assert result["joint"].size == vector_dim("joint", clip_dim=2, stages=2, has_events=False)

    # 独立 helper 也覆盖固定 12/8 维顺序
    np.testing.assert_allclose(_g4_expected(normalized), expected_g4, atol=1e-12)
    np.testing.assert_allclose(_g5_expected(normalized), expected_g5, atol=1e-12)


def test_temporal_stats_uses_clip_weights_in_semantic_global():
    e0 = np.array([1.0, 0.0])
    e1 = np.array([0.0, 1.0])
    matrix = np.stack([e0, e1, e0])
    weights = [0.6, 0.3, 0.1]
    times = tuple(
        ClipTime(
            index=index,
            start_s=index * 2.0,
            duration_s=2.0,
            valid_duration_s=2.0,
            weight=weights[index],
        )
        for index in range(3)
    )
    seq = _make_sequence(matrix, times=times, duration_s=6.0)
    result = assemble_temporal_stats(seq, stages=2)

    # 只有 G1 使用 clip weight；G2 仍按重叠时长，G3/G4/G5 与均匀权重时相同
    g1 = _unit(0.6 * e0 + 0.3 * e1 + 0.1 * e0)
    g2 = _unit(np.concatenate([_unit((2 * e0 + e1) / 3.0), _unit((2 * e0 + e1) / 3.0)]))
    g3 = _unit((e1 + e0) / 2.0 - e0)
    g4 = _g4_expected(_row_normalize(matrix))
    g4_unit = g4 / np.linalg.norm(g4)
    g5 = _g5_expected(_row_normalize(matrix))
    expected = _combine_expected([g1, g2, g3, g4_unit, g5])
    np.testing.assert_allclose(result["joint"], expected, atol=1e-5)


def test_temporal_stats_reversal_sensitivity_versus_full_mean():
    rng = np.random.default_rng(21)
    direction_a = _unit(rng.normal(size=8))
    direction_b = _unit(rng.normal(size=8))
    vectors = [direction_a] * 20 + [direction_b] * 20

    forward = _make_sequence(np.stack(vectors), duration_s=320.0)
    backward = _make_sequence(np.stack(list(reversed(vectors))), duration_s=320.0)

    full_forward = assemble_full_mean(forward)["joint"]
    full_backward = assemble_full_mean(backward)["joint"]
    temporal_forward = assemble_temporal_stats(forward)["joint"]
    temporal_backward = assemble_temporal_stats(backward)["joint"]

    full_cosine = float(full_forward @ full_backward)
    temporal_cosine = float(temporal_forward @ temporal_backward)
    assert full_cosine > 0.9999, full_cosine
    assert temporal_cosine < 0.9, temporal_cosine
    assert (1.0 - temporal_cosine) > 0.2
    assert np.isclose(np.linalg.norm(temporal_forward), 1.0, atol=1e-5)


def test_temporal_stats_missing_modalities():
    matrix = np.eye(4, dtype=np.float32)
    only_joint = assemble_temporal_stats(_make_sequence(matrix))
    assert set(only_joint) == {"joint"}
    with_audio = assemble_temporal_stats(_make_sequence(matrix, audio=matrix))
    assert set(with_audio) == {"joint", "audio"}

    with pytest.raises(EmbeddingError):
        assemble_temporal_stats(_make_sequence(matrix, audio=matrix[[0, 1]]))
    with pytest.raises(EmbeddingError):
        assemble_temporal_stats(
            ClipSequence(
                work_id=5,
                version="no-joint",
                duration_s=8.0,
                times=(ClipTime(0, 0.0, 8.0, 8.0, 1.0),),
                joint=None,
                audio=matrix,
                video=None,
            )
        )


def test_temporal_stats_single_clip_zero_groups():
    # 单片段：drift / self-similarity 为全零组，不参与最终方向
    x = np.array([1.0, 0.0], dtype=np.float32)
    seq = _make_sequence(np.stack([x]), duration_s=8.0)
    result = assemble_temporal_stats(seq, stages=2)["joint"]
    g1 = _unit(x.astype(np.float64))
    g2_parts = [_unit(x.astype(np.float64)), _unit(x.astype(np.float64))]
    g2 = _unit(np.concatenate(g2_parts))
    expected = _combine_expected([g1, g2, np.zeros(2), np.zeros(12), np.zeros(8)])
    np.testing.assert_allclose(result, expected, atol=1e-6)
    assert result.size == vector_dim("joint", clip_dim=2, stages=2, has_events=False)


# --------------------------------------------------------------------------- 事件变体
def test_temporal_stats_event_variant_formula():
    e0 = np.array([1.0, 0.0])
    e1 = np.array([0.0, 1.0])
    matrix = np.stack([e0, e1])
    times = (
        ClipTime(0, 0.0, 8.0, 8.0, 0.5),
        ClipTime(1, 8.0, 8.0, 8.0, 0.5),
    )
    events = np.stack([np.arange(32, dtype=np.float64), np.arange(32, dtype=np.float64) + 10.0])
    seq = _make_sequence(
        matrix, audio=matrix, video=matrix, times=times, duration_s=16.0, event_features=events
    )
    result = assemble_temporal_stats(seq, stages=2)
    assert result["joint"].size == vector_dim("joint", clip_dim=2, stages=2, has_events=True)
    assert result["audio"].size == vector_dim("audio", clip_dim=2, stages=2, has_events=False)
    assert np.isclose(np.linalg.norm(result["joint"]), 1.0, atol=1e-5)

    normalized = _row_normalize(matrix)
    g1 = _unit(np.mean(normalized, axis=0))
    g2 = _unit(np.concatenate([_unit(e0), _unit(e1)]))
    g3 = _unit(e1 - e0)
    expected_g4 = _g4_expected(normalized)
    g4 = expected_g4 / np.linalg.norm(expected_g4)
    g5 = _g5_expected(normalized)
    # G6: mean(32) + std(32) + stage0均值(32=F0) + stage1均值(32=F1)
    g6_raw = np.concatenate(
        [events.mean(axis=0), events.std(axis=0), events[0], events[1]]
    )
    g6 = g6_raw / np.linalg.norm(g6_raw)
    expected = _combine_expected([g1, g2, g3, g4, g5, g6])
    np.testing.assert_allclose(result["joint"], expected, atol=1e-5)

    # 缺失事件特征 -> semantic-only 变体（维度更小、params 记录 has_events=false）
    semantic = assemble_temporal_stats(_make_sequence(matrix, times=times, duration_s=16.0), stages=2)
    assert semantic["joint"].size == vector_dim("joint", clip_dim=2, stages=2, has_events=False)
    assert semantic["joint"].size < result["joint"].size


def test_temporal_stats_event_shape_errors():
    matrix = np.eye(4, dtype=np.float32)
    seq = _make_sequence(matrix, event_features=np.zeros((4, 31), dtype=np.float32))
    with pytest.raises(EmbeddingError):
        assemble_temporal_stats(seq)
    seq = _make_sequence(matrix, event_features=np.zeros((3, 32), dtype=np.float32))
    with pytest.raises(EmbeddingError):
        assemble_temporal_stats(seq)
    seq = _make_sequence(matrix, event_features=np.full((4, 32), np.nan, dtype=np.float32))
    with pytest.raises(EmbeddingError):
        assemble_temporal_stats(seq)


# --------------------------------------------------------------------------- 边界输入
def test_temporal_stats_duration_fallback():
    matrix = np.eye(4, dtype=np.float32)
    times = tuple(ClipTime(index, index * 8.0, 8.0, 8.0, 0.25) for index in range(4))
    seq = ClipSequence(
        work_id=1,
        version="no-duration",
        duration_s=None,
        times=times,
        joint=matrix,
        audio=None,
        video=None,
    )
    # duration_s=None 时由最后一个窗口 end_s 推导
    result = assemble_temporal_stats(seq)
    assert result["joint"].size == vector_dim("joint", clip_dim=4, stages=4, has_events=False)
    assert np.isclose(np.linalg.norm(result["joint"]), 1.0, atol=1e-5)
    np.testing.assert_array_equal(result["joint"], assemble_temporal_stats(seq)["joint"])


def test_temporal_stats_invalid_stages_and_nan():
    matrix = np.eye(4, dtype=np.float32)
    seq = _make_sequence(matrix)
    for invalid in (1, 9, 0, True, 4.0):
        with pytest.raises(EmbeddingError):
            assemble_temporal_stats(seq, stages=invalid)
    bad = matrix.copy()
    bad[0, 0] = np.nan
    with pytest.raises(EmbeddingError):
        assemble_temporal_stats(_make_sequence(bad))
    with pytest.raises(EmbeddingError):
        assemble_temporal_stats(_empty_sequence())


def test_self_similarity_lags_and_dynamics_dim_are_fixed():
    assert SELF_SIMILARITY_LAGS == (1, 2, 3, 4, 6, 8, 12, 16)
    assert DYNAMICS_DIM == 12
    rng = np.random.default_rng(5)
    normalized = _row_normalize(rng.normal(size=(20, 8)))
    assert _g4_expected(normalized).size == DYNAMICS_DIM
    assert _g5_expected(normalized).size == SELF_SIMILARITY_DIM
