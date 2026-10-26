"""v0.5-beta（nebula/v05_beta.py + tools/train_v05beta.py + v05_data 扩展）的 CPU 测试。

不加载 PE-AV/CLIP/DINOv2/MERT、不联网、不需要 GPU：
覆盖 max_segments=None 全量 token 扩展、音频适配层、0.5s bin 对齐、候选构造与近重复/
时间歧义跳过、划分无泄漏、双塔前向/InfoNCE、合成数据训练、指标/基线/消融/产物 schema，
以及"不写 DB / 不建 map / 不写 representations"边界。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nebula import v05_beta as vb
from nebula import v05_data
from nebula.config import Config
from nebula.db import Catalog
from nebula.v05_beta import (
    BETA_BIN_SECONDS,
    BETA_MODEL_VERSION,
    BetaConfig,
    CandidateSet,
    V05BetaError,
    WorkPair,
    _artifact_version,
    _bootstrap_accuracy_ci,
    _random_baseline,
    _temporal_sync_baseline,
    build_model,
    collate_anchors,
    construct_candidates,
    evaluate_ablations,
    evaluate_candidates,
    event_change_scalars,
    fit_standardizer,
    flatten_visual_bins,
    infonce_loss,
    load_audio_bins,
    run_beta_training,
    simple_sync_baseline,
    split_work_ids,
)
from tools import build_v05_tokens as build_tokens_tool
from tools import train_v05beta as train_tool

# ---------------------------------------------------------------- 合成 token / 音频


def _token_dict(
    segments: int = 4,
    *,
    bins: int = 16,
    visual_dim: int = 512,
    dino_dim: int = 384,
    mask_value: int = 1,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(segments)
    visual = rng.normal(size=(segments, bins, visual_dim)).astype(np.float16)
    dino = rng.normal(size=(segments, bins, dino_dim)).astype(np.float16)
    audio = rng.normal(size=(segments, bins, 24)).astype(np.float16)
    event = rng.normal(size=(segments, bins, 12)).astype(np.float16)
    masks = np.full((segments, bins), int(mask_value), dtype=np.uint8)
    times = np.zeros((segments, 4), dtype=np.float32)
    times[:, 0] = np.arange(segments, dtype=np.float32) * 8.0
    times[:, 1] = 8.0
    times[:, 2] = 8.0
    times[:, 3] = 1.0 / segments
    return {
        "paav": rng.normal(size=(segments, 3, 1024)).astype(np.float16),
        "visual_clip": visual,
        "visual_dino": dino,
        "audio": audio,
        "visual_event": event,
        "mask_visual": masks.copy(),
        "mask_audio": masks.copy(),
        "mask_visual_event": masks.copy(),
        "segment_times": times,
        "segment_mask": np.ones(segments, dtype=np.uint8),
        "duration_s": np.asarray(segments * 8.0, dtype=np.float64),
        "fps": np.asarray(8.0, dtype=np.float64),
    }


def _make_pair(
    work_id: int = 1,
    *,
    bins: int = 20,
    visual_dim: int = 4,
    audio_dim: int = 3,
    times_step: float = BETA_BIN_SECONDS,
) -> WorkPair:
    rng = np.random.default_rng(work_id)
    visual = rng.normal(size=(bins, visual_dim)).astype(np.float32)
    audio = rng.normal(size=(bins, audio_dim)).astype(np.float32)
    times = np.arange(bins, dtype=np.float64) * float(times_step)
    return WorkPair(
        work_id=work_id,
        visual=visual,
        audio=audio,
        times=times,
        duration_s=float(bins) * times_step,
        visual_dim=visual_dim,
        audio_dim=audio_dim,
        coverage={"matched_ratio": 1.0},
    )


def _make_candidate_set(work_id: int = 1, anchors: int = 3) -> CandidateSet:
    rng = np.random.default_rng(work_id)
    return CandidateSet(
        work_id=work_id,
        visual=rng.normal(size=(anchors, 4)).astype(np.float32),
        candidates=tuple(
            rng.normal(size=(3, 3)).astype(np.float32) for _ in range(anchors)
        ),
        labels=np.zeros(anchors, dtype=np.int64),
        offsets=tuple(
            np.asarray([0, 1, 2], dtype=np.int64) for _ in range(anchors)
        ),
        anchor_times=np.arange(anchors, dtype=np.float64) * BETA_BIN_SECONDS,
    )


# ---------------------------------------------------------------- v05_data 扩展


def test_max_segments_none_supports_full_tokens(
    config: Config, catalog: Catalog
) -> None:
    segments = 130
    tokens = _token_dict(segments=segments)
    with pytest.raises(v05_data.V05DataError):
        v05_data._validated_tokens(tokens)
    validated = v05_data._validated_tokens(tokens, max_segments=None)
    assert validated["paav"].shape[0] == segments

    work_id, _ = catalog.upsert_local_work("full_token_work.mp4")
    path = v05_data.save_work_tokens(
        config, "v05_beta_test_full", work_id, tokens, max_segments=None
    )
    assert path.is_file()
    loaded = v05_data.load_work_tokens(config, "v05_beta_test_full", work_id)
    assert loaded is not None and loaded["paav"].shape[0] == segments
    assert int(np.asarray(loaded["truncated"]).item()) == 0

    parser = build_tokens_tool.build_parser()
    assert parser.parse_args(["--max-segments", "none"]).max_segments is None
    assert parser.parse_args(["--max-segments", "0"]).max_segments is None
    assert parser.parse_args(["--max-segments", "128"]).max_segments == 128


def test_split_work_ids_deterministic_and_no_leakage() -> None:
    ids = list(range(1, 24))
    split = split_work_ids(ids, seed=42)
    assert (len(split.train_ids), len(split.val_ids), len(split.test_ids)) == (16, 4, 3)
    assert set(split.train_ids).isdisjoint(split.val_ids)
    assert set(split.train_ids).isdisjoint(split.test_ids)
    assert set(split.val_ids).isdisjoint(split.test_ids)
    assert sorted(set(split.train_ids) | set(split.val_ids) | set(split.test_ids)) == ids
    assert split_work_ids(ids, seed=42) == split
    assert split_work_ids(ids, seed=43) != split

    explicit = split_work_ids(ids, seed=42, split_counts=[12, 6, 5])
    assert len(explicit.train_ids) == 12
    assert len(explicit.test_ids) == 5
    with pytest.raises(V05BetaError):
        split_work_ids(ids, split_counts=[10, 10, 10])

    big = split_work_ids(list(range(1, 111)), seed=42)
    assert (len(big.train_ids), len(big.val_ids), len(big.test_ids)) == (77, 22, 11)


# ---------------------------------------------------------------- 音频适配层


def _write_audio(
    config: Config,
    version: str,
    work_id: int,
    *,
    features: np.ndarray,
    mask: np.ndarray | None = None,
    times: np.ndarray | None = None,
    manifest: dict | None = None,
    flat: bool = False,
) -> Path:
    version_dir = config.data_path("audio_dense") / version
    version_dir.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        (version_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )
    payload: dict[str, np.ndarray] = {"bin_features": features.astype(np.float32)}
    if mask is not None:
        payload["bin_mask"] = mask.astype(np.uint8)
    if times is not None:
        payload["bin_times"] = times.astype(np.float64)
    if flat:
        target = version_dir / f"{work_id:06d}.npz"
    else:
        work_dir = version_dir / f"{work_id:06d}"
        work_dir.mkdir(parents=True, exist_ok=True)
        target = work_dir / "audio.npz"
    np.savez(target, **payload)
    return target


def test_audio_adapter_reads_contract_and_fallbacks(config: Config) -> None:
    features = np.arange(20, dtype=np.float32).reshape(10, 2)
    mask = np.ones(10, dtype=bool)
    mask[3] = False
    times = np.arange(10, dtype=np.float64) * 0.5
    _write_audio(
        config,
        "audio_test_v1",
        7,
        features=features,
        mask=mask,
        times=times,
        manifest={"version": "audio_test_v1", "bin_seconds": 0.5, "hop": 0.5},
    )
    loaded = load_audio_bins(config, "audio_test_v1", 7)
    assert loaded is not None
    assert loaded["features"].shape == (10, 2)
    assert loaded["mask"].tolist() == mask.tolist()
    assert np.allclose(loaded["times"], times)
    assert loaded["hop_seconds"] == pytest.approx(0.5)
    assert load_audio_bins(config, "audio_test_v1", 999) is None

    # 扁平 <work_id>.npz + 仅 features/mask，无 times -> 按 manifest hop 推断
    _write_audio(
        config,
        "audio_test_flat",
        8,
        features=features,
        mask=None,
        times=None,
        manifest={"hop_seconds": 0.25},
        flat=True,
    )
    loaded_flat = load_audio_bins(config, "audio_test_flat", 8)
    assert loaded_flat is not None
    assert loaded_flat["hop_seconds"] == pytest.approx(0.25)
    assert np.allclose(loaded_flat["times"], np.arange(10) * 0.25)
    assert bool(loaded_flat["mask"].all())


def test_audio_adapter_rejects_bad_shapes(config: Config) -> None:
    _write_audio(config, "audio_bad", 1, features=np.zeros((4, 3, 2), dtype=np.float32))
    with pytest.raises(V05BetaError):
        load_audio_bins(config, "audio_bad", 1)


# ---------------------------------------------------------------- 视觉 / 对齐 / 候选


def test_flatten_visual_bins() -> None:
    tokens = _token_dict(segments=2, bins=16)
    visual, mask, times = flatten_visual_bins(tokens, "clip")
    assert visual.shape == (32, 512)
    assert mask.shape == (32,)
    assert times.shape == (32,)
    assert np.allclose(times[:16], np.arange(16) * BETA_BIN_SECONDS)
    assert np.allclose(times[16:], 8.0 + np.arange(16) * BETA_BIN_SECONDS)
    dino, _, _ = flatten_visual_bins(tokens, "dino")
    assert dino.shape == (32, 384)


def test_build_work_pair_alignment() -> None:
    tokens = _token_dict(segments=2, bins=16)
    audio = {
        "features": np.random.default_rng(0).normal(size=(32, 3)).astype(np.float32),
        "mask": np.ones(32, dtype=bool),
        "times": np.arange(32, dtype=np.float64) * 0.5,
        "hop_seconds": 0.5,
    }
    pair, details = vb.build_work_pair(tokens, audio, work_id=5, visual_encoder="clip")
    assert pair is not None
    assert pair.n_bins == 32
    assert details["matched"] == 32
    assert pair.coverage["matched_ratio"] == pytest.approx(1.0)
    # 音频只有 1s hop：仍能对齐，但只能匹配到偶数 bin 附近的一半
    audio_coarse = dict(audio)
    audio_coarse["features"] = audio["features"][::2]
    audio_coarse["mask"] = audio["mask"][::2]
    audio_coarse["times"] = audio["times"][::2]
    audio_coarse["hop_seconds"] = 1.0
    coarse_pair, coarse_details = vb.build_work_pair(
        tokens, audio_coarse, work_id=5, visual_encoder="clip"
    )
    assert coarse_pair is not None
    assert coarse_pair.n_bins < 32
    assert coarse_details["matched"] >= coarse_pair.n_bins


def test_construct_candidates_offsets_and_skips() -> None:
    pair = _make_pair(bins=20)
    candidate_set, stats = construct_candidates(
        pair, offsets=(1, 2, 4), near_dup=0.95, rng=np.random.default_rng(0)
    )
    assert candidate_set is not None
    assert candidate_set.n_anchors == stats["kept"]
    assert stats["kept"] > 0
    for anchor in range(candidate_set.n_anchors):
        offsets = candidate_set.offsets[anchor]
        label_index = int(candidate_set.labels[anchor])
        assert int(offsets[label_index]) == 0  # labels 指向正样本位置
        assert (offsets == 0).sum() == 1
        assert set(offsets.tolist()) <= {0, 1, 2, 4}

    # 近重复：让前两个 bin 的视觉完全一致 -> Δ=1 负样本被跳过
    duplicate_pair = _make_pair(bins=10)
    duplicate_pair.visual = duplicate_pair.visual.copy()
    duplicate_pair.visual[1] = duplicate_pair.visual[0]
    _, duplicate_stats = construct_candidates(
        duplicate_pair,
        offsets=(1,),
        near_dup=0.95,
        rng=np.random.default_rng(1),
    )
    assert duplicate_stats["skipped_near_dup"] >= 1

    # 时间位置歧义：时间戳间隔与 Δ 不符 -> 跳过
    irregular = _make_pair(bins=10)
    irregular.times = irregular.times.copy()
    irregular.times[1] = 4.0  # Δ=1 期望 0.5s，实际 4.0s
    _, irregular_stats = construct_candidates(
        irregular,
        offsets=(1,),
        near_dup=0.95,
        rng=np.random.default_rng(2),
    )
    assert irregular_stats["skipped_ambiguous_time"] >= 1


def test_collate_anchors_padding() -> None:
    sets = [_make_candidate_set(1, 3), _make_candidate_set(2, 2)]
    batch = collate_anchors(sets, [(0, 0), (1, 0), (0, 1)])
    assert batch["visual"].shape == (3, 4)
    assert batch["audio"].shape == (3, 3, 3)
    assert batch["candidate_mask"].shape == (3, 3)
    assert batch["candidate_mask"].dtype == torch.bool
    assert bool(batch["candidate_mask"].all())
    assert batch["labels"].shape == (3,)
    assert batch["offsets"].shape == (3, 3)
    # anchor 2 的候选数不同 -> padding 位置 mask=False
    mixed = collate_anchors(
        [
            CandidateSet(
                work_id=1,
                visual=np.ones((2, 4), dtype=np.float32),
                candidates=(
                    np.ones((2, 3), dtype=np.float32),
                    np.ones((3, 3), dtype=np.float32),
                ),
                labels=np.asarray([0, 1]),
                offsets=(np.asarray([0, 1]), np.asarray([0, 1, 2])),
                anchor_times=np.asarray([0.0, 0.5]),
            )
        ],
        [(0, 0), (0, 1)],
    )
    assert mixed["audio"].shape == (2, 3, 3)
    assert bool(mixed["candidate_mask"][0].all()) is False
    assert float(mixed["audio"][0, 2].abs().sum()) == 0.0


# ---------------------------------------------------------------- 模型 / 损失 / 指标


def test_model_forward_and_infonce_loss() -> None:
    pair = _make_pair(bins=12)
    standardizer = fit_standardizer([pair])
    model = build_model(
        visual_dim=4,
        audio_dim=3,
        standardizer=standardizer,
        hidden=8,
        embed_dim=6,
        dropout=0.0,
    )
    batch = collate_anchors([_make_candidate_set(1, 4)], [(0, 0), (0, 1), (0, 2), (0, 3)])
    visual_embedding, audio_embedding = model(batch["visual"], batch["audio"])
    assert visual_embedding.shape == (4, 6)
    assert audio_embedding.shape == (4, 3, 6)
    norms = visual_embedding.detach().norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    loss = infonce_loss(
        visual_embedding,
        audio_embedding,
        batch["candidate_mask"],
        batch["labels"],
        temperature=0.07,
    )
    assert torch.isfinite(loss)
    loss.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads and any(float(grad.abs().sum()) > 0 for grad in grads)


def test_standardizer_train_only_statistics() -> None:
    first = _make_pair(bins=10, visual_dim=2, audio_dim=2)
    second = _make_pair(bins=10, visual_dim=2, audio_dim=2)
    first.visual = np.full_like(first.visual, 3.0)
    first.audio = np.full_like(first.audio, -2.0)
    standardizer = fit_standardizer([first])
    assert np.allclose(standardizer["visual_mean"], np.full(2, 3.0))
    assert np.allclose(standardizer["audio_mean"], np.full(2, -2.0))
    assert np.allclose(standardizer["visual_std"], np.full(2, 1e-6))
    del second


def test_evaluate_candidates_metrics_and_ablations() -> None:
    pair = _make_pair(bins=16)
    candidate_set, _ = construct_candidates(
        pair, offsets=(1, 2), near_dup=0.95, rng=np.random.default_rng(0)
    )
    assert candidate_set is not None
    standardizer = fit_standardizer([pair])
    model = build_model(
        visual_dim=4,
        audio_dim=3,
        standardizer=standardizer,
        hidden=8,
        embed_dim=6,
        dropout=0.0,
    )
    metrics = evaluate_candidates(
        model,
        [candidate_set],
        device=torch.device("cpu"),
        temperature=0.07,
        batch_size=8,
    )
    for key in (
        "top1",
        "recall_at_1",
        "recall_at_3",
        "positive_negative_gap",
        "offset_curve",
        "collapse",
        "per_work_accuracy",
    ):
        assert key in metrics
    assert set(metrics["offset_curve"]) <= {"1", "2"}
    ablations = evaluate_ablations(
        model,
        [candidate_set],
        device=torch.device("cpu"),
        temperature=0.07,
        batch_size=8,
    )
    assert {"normal", "audio_zero", "visual_zero", "time_shift", "audio_replacement"} <= set(ablations)
    assert 0.0 <= ablations["normal"]["top1"] <= 1.0


def test_baselines_and_bootstrap() -> None:
    sets = [_make_candidate_set(1, 4)]
    random_top1 = _random_baseline(sets)
    assert random_top1 == pytest.approx(1.0 / 3.0)
    sync = _temporal_sync_baseline(sets)
    assert sync["top1"] == pytest.approx(1.0)

    per_work = {1: 0.2, 2: 0.8, 3: 0.5, 4: 0.9}
    first = _bootstrap_accuracy_ci(per_work, seed=42, samples=50)
    second = _bootstrap_accuracy_ci(per_work, seed=42, samples=50)
    assert first == second
    assert first is not None
    assert first["ci95"][0] <= first["mean"] + 1e-9
    assert first["ci95"][1] + 1e-9 >= first["mean"]
    assert first["n_works"] == 4
    assert _bootstrap_accuracy_ci({}, seed=1, samples=10) is None


def test_simple_sync_baseline_manual_and_deterministic() -> None:
    """手算负绝对差评分：2 anchors、每 anchor 2 候选，正样本均可被匹配。"""

    import inspect

    def make_set() -> CandidateSet:
        return CandidateSet(
            work_id=1,
            visual=np.zeros((2, 4), dtype=np.float32),
            candidates=(
                np.zeros((2, 3), dtype=np.float32),
                np.zeros((2, 3), dtype=np.float32),
            ),
            labels=np.asarray([0, 0], dtype=np.int64),
            offsets=(np.asarray([0, 1]), np.asarray([0, 1])),
            anchor_times=np.asarray([0.0, 0.5]),
            visual_change=np.asarray([0.0, 2.0], dtype=np.float32),
            candidate_audio_change=(
                np.asarray([0.0, 2.0], dtype=np.float32),
                np.asarray([2.0, 0.0], dtype=np.float32),
            ),
            candidate_audio_valid=(
                np.asarray([True, True]),
                np.asarray([True, True]),
            ),
            anchor_valid=np.asarray([True, True]),
        )

    sets = [make_set()]
    # work 内候选 change 池 = [0,2,2,0] -> mean=1, std=1；
    # anchor0 z_visual=-1，候选 z=[-1,1] -> score=[0,-2]；anchor1 z_visual=1 -> score=[0,-2]
    result = simple_sync_baseline(sets)
    assert result["top1"] == pytest.approx(1.0)
    assert result["recall_at_1"] == pytest.approx(1.0)
    assert result["recall_at_3"] == pytest.approx(1.0)
    assert result["n_anchors"] == 2
    assert result["n_works"] == 1
    assert result["uses_model"] is False
    assert result["uses_work_id"] is False
    assert "negative absolute difference" in result["definition"]
    assert result["per_work_accuracy"]["1"] == pytest.approx(1.0)
    assert simple_sync_baseline(sets) == result
    assert simple_sync_baseline([make_set()]) == result
    parameters = inspect.signature(simple_sync_baseline).parameters
    assert list(parameters) == ["sets"]
    assert "model" not in str(parameters)


def test_simple_sync_unavailable_without_event_features() -> None:
    plain = _make_candidate_set(1, 3)  # 未提供事件统计字段
    result = simple_sync_baseline([plain])
    assert result["top1"] is None
    assert result["n_anchors"] == 0
    assert "不可用" in result["note"]
    assert simple_sync_baseline([])["top1"] is None


def test_event_change_scalars_from_tokens() -> None:
    tokens = _token_dict(segments=2, bins=16)
    audio = {
        "features": np.zeros((32, 3), dtype=np.float32),
        "mask": np.ones(32, dtype=bool),
        "times": np.arange(32, dtype=np.float64) * BETA_BIN_SECONDS,
        "hop_seconds": BETA_BIN_SECONDS,
    }
    visual_event = np.zeros((2, 16, 12), dtype=np.float32)
    visual_event[..., 6] = 0.1
    visual_event[..., 7] = 0.2
    visual_event[..., 8] = 0.3
    audio_event = np.zeros((2, 16, 24), dtype=np.float32)
    ramp = np.arange(16, dtype=np.float32)
    audio_event[0, :, 0] = ramp * 0.05
    audio_event[1, :, 0] = (ramp + 16) * 0.05
    audio_event[0, :, 2] = ramp * 0.02
    audio_event[1, :, 2] = (ramp + 16) * 0.02
    audio_event[0, :, 3] = ramp * 0.10
    audio_event[1, :, 3] = (ramp + 16) * 0.10
    audio_event[..., 6] = 0.01
    tokens["visual_event"] = visual_event
    tokens["audio"] = audio_event
    tokens["mask_visual_event"] = np.ones((2, 16), dtype=np.uint8)
    tokens["mask_audio"] = np.ones((2, 16), dtype=np.uint8)

    pair, details = vb.build_work_pair(
        tokens, audio, work_id=1, visual_encoder="clip"
    )
    assert pair is not None
    assert pair.visual_change is not None and pair.audio_change is not None
    assert pair.event_valid is not None and bool(pair.event_valid.all())
    expected_std = float(np.sqrt(0.1**2 + 0.2**2 + 0.3**2))
    assert pair.visual_change[0] == pytest.approx(expected_std, abs=1e-5)
    assert pair.visual_change[1] == pytest.approx(expected_std, abs=1e-5)
    assert pair.audio_change[0] == pytest.approx(0.01, abs=1e-5)
    assert pair.audio_change[1] == pytest.approx(0.01 + 0.05 + 0.02 + 0.10, abs=1e-5)
    assert details["event_valid_ratio"] == pytest.approx(1.0)

    visual_change, audio_change, valid = event_change_scalars(
        tokens, np.asarray([0, 1])
    )
    assert visual_change is not None and audio_change is not None and valid is not None
    assert audio_change[0] == pytest.approx(0.01, abs=1e-5)
    assert audio_change[1] == pytest.approx(0.01 + 0.05 + 0.02 + 0.10, abs=1e-5)
    four_change, four_audio, four_valid = event_change_scalars(
        tokens, np.asarray([0, 1, 16, 17])
    )
    assert four_change is not None and four_audio is not None and four_valid is not None
    assert four_change.shape == (4,)
    tokens["mask_audio"][0, 1] = 0
    pair_invalid, _ = vb.build_work_pair(
        tokens, audio, work_id=1, visual_encoder="clip"
    )
    assert pair_invalid is not None
    assert pair_invalid.event_valid is not None
    assert bool(pair_invalid.event_valid[1]) is False
    assert float(pair_invalid.audio_change[1]) == 0.0


# ---------------------------------------------------------------- 合成数据 / 端到端


def test_build_synthetic_dataset_deterministic() -> None:
    cfg = BetaConfig(smoke=True, smoke_works=12, seeds=(42,))
    dataset = vb._build_synthetic_dataset(cfg)
    assert dataset.info["synthetic"] is True
    assert len(dataset.candidate_sets) == 12
    split = dataset.split
    assert (len(split.train_ids), len(split.val_ids), len(split.test_ids)) == (8, 2, 2)
    assert set(dataset.standardizer) == {
        "visual_mean",
        "visual_std",
        "audio_mean",
        "audio_std",
    }
    again = vb._build_synthetic_dataset(cfg)
    first_set = dataset.candidate_sets[1]
    second_set = again.candidate_sets[1]
    assert np.allclose(first_set.visual, second_set.visual)
    assert np.array_equal(first_set.labels, second_set.labels)
    assert first_set.visual_change is not None
    assert first_set.candidate_audio_change is not None
    assert first_set.candidate_audio_valid is not None
    assert first_set.anchor_valid is not None
    assert "smoke" in " ".join(dataset.info["limitations"]).lower()


def test_artifact_version_smoke_isolated() -> None:
    assert _artifact_version(BetaConfig(smoke=True)) == f"{BETA_MODEL_VERSION}_smoke"
    assert _artifact_version(BetaConfig(smoke=True, version="v05_beta_custom")) == "v05_beta_custom"
    assert _artifact_version(BetaConfig()) == BETA_MODEL_VERSION


def test_run_beta_training_smoke_artifacts(
    config: Config, catalog: Catalog
) -> None:
    cfg = BetaConfig(
        version="v05_beta_test",
        smoke=True,
        smoke_works=12,
        epochs=2,
        batch_size=32,
        hidden=16,
        embed_dim=8,
        seeds=(42, 43),
        bootstrap=20,
        patience=0,
        device="cpu",
    )
    result = run_beta_training(config, catalog, cfg)
    summary = result.summary
    assert len(result.results) == 2
    assert summary["aggregate"]["n_seeds"] == 2
    assert summary["aggregate"]["test_top1_mean"] is not None
    assert len(summary["per_seed"]) == 2
    assert summary["dataset"]["synthetic"] is True
    assert summary["dataset"]["split"]["counts"] == [8, 2, 2]
    assert summary["aggregate"]["simple_sync_top1_mean"] is not None
    assert summary["aggregate"]["delta_vs_simple_sync"] is not None
    for seed_info in summary["per_seed"]:
        assert seed_info["simple_sync_top1"] is not None
    assert (config.paths.logs / "v05_beta_test_summary.json").is_file()
    candidates_manifest = json.loads(
        (config.paths.logs / "v05_beta_test_candidates_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert candidates_manifest["offsets"] == list(cfg.offsets)
    assert candidates_manifest["split"]["counts"] == [8, 2, 2]
    assert len(candidates_manifest["works"]) == 12

    for seed in (42, 43):
        checkpoint_path = (
            config.data_path("models") / f"v05_beta_test_{seed}" / "checkpoint.pt"
        )
        metrics_path = config.paths.logs / f"v05_beta_test_{seed}_metrics.json"
        assert checkpoint_path.is_file()
        assert metrics_path.is_file()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert set(("version", "seed", "config", "state_dict", "model_config")) <= set(
            checkpoint
        )
        assert checkpoint["seed"] == seed
        assert checkpoint["standardizer"]["visual_mean"].shape == (64,)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert metrics["seed"] == seed
        assert metrics["dataset"]["synthetic"] is True
        assert metrics["training"]["epochs_run"] == 2
        assert metrics["test"]["top1"] is not None
        assert metrics["test"]["bootstrap"]["n_works"] >= 1
        assert set(metrics["baselines"]) >= {
            "random_top1",
            "frozen_projection_top1",
            "temporal_sync_top1",
            "simple_sync_top1",
            "simple_sync_recall_at_3",
            "simple_sync_recall_at_5",
            "simple_sync_uses_model",
        }
        assert metrics["baselines"]["simple_sync_uses_model"] is False
        assert metrics["baselines"]["simple_sync_top1"] is not None
        assert metrics["baselines"]["delta_vs_simple_sync"] is not None
        assert metrics["gates"]["top1_above_simple_sync"] is not None
        assert set(metrics["ablations"]) >= {
            "normal",
            "audio_zero",
            "visual_zero",
            "audio_replacement",
        }
        assert set(metrics["test"]["collapse"]) >= {
            "visual_effective_rank",
            "audio_effective_rank",
        }

    # 研究产物不得写 DB / map / representations
    assert catalog.list_representations() == []
    assert catalog.conn.execute("SELECT COUNT(*) AS n FROM map_runs").fetchone()["n"] == 0
    representations_dir = config.paths.representations
    assert not any(representations_dir.iterdir())


def test_missing_audio_data_reports_error(
    config: Config, catalog: Catalog
) -> None:
    work_id, _ = catalog.upsert_local_work("beta_missing.mp4")
    catalog.update_work_metadata(work_id, duration_s=32.0)
    catalog.begin_attempt(work_id, "encoding")
    catalog.complete(work_id)
    with pytest.raises(V05BetaError) as exc_info:
        run_beta_training(config, catalog, BetaConfig(epochs=1, seeds=(42,), device="cpu"))
    assert "audio" in str(exc_info.value).lower() or "音频" in str(exc_info.value)


def test_cli_parser_smoke() -> None:
    parser = train_tool._build_parser()
    args = parser.parse_args(["--smoke", "--seeds", "2", "--offsets", "1,4", "--max-segments", "none"])
    assert args.smoke is True
    assert args.offsets == (1, 4)
    assert args.max_segments is None
    assert train_tool._config_from_args(args).seeds == (42, 43)
    split_args = parser.parse_args(["--split-counts", "16,4,3"])
    assert split_args.split_counts == (16, 4, 3)



def test_collapse_metrics_subsample_large_matrices():
    from nebula.v05_beta import (  # type: ignore[import-not-found]
        COLLAPSE_SAMPLE_MAX_ROWS,
        _effective_rank,
        _pairwise_cosine,
    )

    matrix = np.random.default_rng(0).normal(size=(COLLAPSE_SAMPLE_MAX_ROWS + 123, 16))
    cosine = _pairwise_cosine(matrix)
    assert cosine["n_rows"] == matrix.shape[0]
    assert cosine["sampled"] is True
    assert np.isfinite(cosine["mean"])
    assert np.isfinite(cosine["std"])
    rank = _effective_rank(matrix)
    assert rank is not None and np.isfinite(rank) and rank > 0
    small = _pairwise_cosine(matrix[:10])
    assert small["sampled"] is False

