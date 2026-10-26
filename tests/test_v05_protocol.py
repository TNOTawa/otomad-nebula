"""v0.5-beta 协议 v2（nebula/v05_protocol.py + tools/build_v05_protocol.py）CPU 测试。

不写真实 data / 不训练 / 不联网：
覆盖固定 K=6、双向偏移、periodic 独立集、K=32 retrieval、split 校验、指标手算、
anchor 记录/manifest 原子落盘、旧 checkpoint adapter、simple_sync/random/frozen 基线、
以及端到端 evaluate_protocol（合成小数据，tmp config）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nebula import v05_beta, v05_protocol
from nebula.config import Config
from nebula.v05_protocol import (
    ProtocolConfig,
    V05ProtocolError,
    _bootstrap_ci,
    _rank_metrics,
    _simple_sync_metrics,
    build_anchor_records,
    build_protocol_bundle,
    candidate_sets_from_records,
    evaluate_protocol,
    load_beta_split,
    load_old_checkpoint,
    save_protocol,
)
from tools import build_v05_protocol as protocol_tool

# ---------------------------------------------------------------- 合成 pair / split


def _make_pair(
    work_id: int,
    *,
    bins: int = 60,
    visual_dim: int = 8,
    audio_dim: int = 6,
    identical: bool = False,
) -> v05_beta.WorkPair:
    rng = np.random.default_rng(work_id)
    if identical:
        visual = np.tile(np.linspace(0.1, 1.0, visual_dim, dtype=np.float32), (bins, 1))
        audio = np.tile(np.linspace(0.2, 0.8, audio_dim, dtype=np.float32), (bins, 1))
    else:
        visual = rng.normal(size=(bins, visual_dim)).astype(np.float32)
        audio = rng.normal(size=(bins, audio_dim)).astype(np.float32)
    times = np.arange(bins, dtype=np.float64) * 0.5
    visual_change = np.zeros(bins, dtype=np.float32)
    audio_change = np.zeros(bins, dtype=np.float32)
    if bins > 1:
        visual_change[1:] = np.linalg.norm(np.diff(visual, axis=0), axis=1).astype(np.float32)
        audio_change[1:] = np.linalg.norm(np.diff(audio, axis=0), axis=1).astype(np.float32)
    return v05_beta.WorkPair(
        work_id=int(work_id),
        visual=visual,
        audio=audio,
        times=times,
        duration_s=float(bins) * 0.5,
        visual_dim=visual_dim,
        audio_dim=audio_dim,
        coverage={"matched_ratio": 1.0},
        visual_change=visual_change,
        audio_change=audio_change,
        event_valid=np.ones(bins, dtype=bool),
    )


def _small_split() -> dict[str, tuple[int, ...]]:
    return {
        "train": (1, 2, 3),
        "val": (4, 5),
        "test": (6,),
    }


def _small_bundle(*, test_bins: int = 80) -> v05_protocol.ProtocolBundle:
    pairs = {work_id: _make_pair(work_id) for work_id in (1, 2, 3, 4, 5)}
    pairs[6] = _make_pair(6, bins=test_bins)
    cfg = ProtocolConfig()
    data_info = {
        "tokens_version": "v05_beta_v1",
        "audio_version": "test_audio",
        "visual_encoder": "clip",
        "visual_dim": 8,
        "audio_dim": 6,
        "bin_seconds": 0.5,
        "n_works": len(pairs),
        "model_config": {"hidden": 8, "embed_dim": 8, "temperature": 0.07},
    }
    return build_protocol_bundle(pairs, _small_split(), config=cfg, data_info=data_info)


def _write_beta_summary(config: Config, counts=(74, 21, 12)) -> Path:
    ids = list(range(1, sum(counts) + 1))
    train = ids[: counts[0]]
    val = ids[counts[0] : counts[0] + counts[1]]
    test = ids[counts[0] + counts[1] :]
    summary = {
        "dataset": {
            "audio_version": "test_audio",
            "split": {
                "train_work_ids": train,
                "val_work_ids": val,
                "test_work_ids": test,
            },
        }
    }
    path = config.paths.logs / "v05_beta_summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


# ---------------------------------------------------------------- 协议配置 / split


def test_protocol_config_validation() -> None:
    with pytest.raises(V05ProtocolError):
        ProtocolConfig(main_k=1).validate()
    with pytest.raises(V05ProtocolError):
        ProtocolConfig(main_k=8, retrieval_k=8).validate()
    with pytest.raises(V05ProtocolError):
        ProtocolConfig(visual_encoder="other").validate()
    assert ProtocolConfig().validate().main_k == 6


def test_load_beta_split_requires_74_21_12(config: Config) -> None:
    _write_beta_summary(config)
    split = load_beta_split(config)
    assert [len(split[name]) for name in ("train", "val", "test")] == [74, 21, 12]
    assert set(split["train"]).isdisjoint(split["val"])
    assert set(split["train"]).isdisjoint(split["test"])
    # 错误数量必须拒绝
    _write_beta_summary(config, counts=(73, 21, 12))
    with pytest.raises(V05ProtocolError):
        load_beta_split(config)


# ---------------------------------------------------------------- anchor 记录


def test_main_anchors_fixed_k6_bidirectional_deterministic() -> None:
    pair = _make_pair(1, bins=80)
    first = build_anchor_records(pair, split="test", seed=42, near_dup=0.95)
    second = build_anchor_records(pair, split="test", seed=42, near_dup=0.95)
    main, periodic, periodic_candidates, retrieval, stats = first
    assert retrieval == []
    assert periodic == []
    assert periodic_candidates == []
    assert main and not stats["periodic_anchors"]
    positive_offsets = set()
    negative_offsets = set()
    for record in main:
        assert record.candidate_bins.size == 6
        assert int(record.candidate_offsets[record.label]) == 0
        assert (record.candidate_offsets == 0).sum() == 1
        for offset in record.candidate_offsets:
            if offset > 0:
                positive_offsets.add(int(offset))
            elif offset < 0:
                negative_offsets.add(int(offset))
        assert record.available_negatives >= 5
    # 双向偏移都出现（同一作品不同 anchor）
    assert positive_offsets and negative_offsets
    assert set(positive_offsets) <= {1, 2, 4, 8, 16}
    assert set(negative_offsets) <= {-1, -2, -4, -8, -16}
    # 统计与确定性
    assert stats["main_kept"] == len(main)
    assert int(sum(stats["available_negatives_hist"].values())) == stats["anchors"]
    again_main = second[0]
    assert len(main) == len(again_main)
    for left, right in zip(main, again_main, strict=True):
        assert np.array_equal(left.candidate_bins, right.candidate_bins)
        assert left.label == right.label


def test_insufficient_anchors_skipped_and_periodic_separated() -> None:
    short_pair = _make_pair(2, bins=4)
    main, periodic, periodic_candidates, _retrieval, stats = build_anchor_records(
        short_pair, split="test", seed=42, near_dup=0.95
    )
    assert main == []
    assert periodic == []
    assert periodic_candidates == []
    assert stats["main_skipped_insufficient"] > 0

    identical_pair = _make_pair(3, bins=30, identical=True)
    main_two, periodic_two, periodic_candidates_two, _retrieval_two, stats_two = build_anchor_records(
        identical_pair, split="test", seed=42, near_dup=0.95
    )
    assert main_two == []
    assert periodic_two
    assert periodic_candidates_two
    assert stats_two["periodic_anchors"] == len(periodic_two)
    assert stats_two["main_skipped_insufficient"] == 0
    # periodic 候选数可变，且 label 指向正样本
    for record in periodic_two:
        assert record.candidate_bins.size >= 2
        assert int(record.candidate_offsets[record.label]) == 0


def test_retrieval_k32() -> None:
    pair = _make_pair(4, bins=80)
    main, _periodic, _periodic_candidates, retrieval, stats = build_anchor_records(
        pair, split="test", seed=42, near_dup=0.95, build_retrieval=True
    )
    assert main
    assert retrieval
    assert stats["retrieval_kept"] == len(retrieval)
    for record in retrieval:
        assert record.candidate_bins.size == 32
        assert int(record.candidate_offsets[record.label]) == 0
        assert (record.candidate_offsets == 0).sum() == 1
        assert record.available_negatives >= 31
    # 确定性
    _, _, _, retrieval_again, _ = build_anchor_records(
        pair, split="test", seed=42, near_dup=0.95, build_retrieval=True
    )
    assert len(retrieval) == len(retrieval_again)
    for left, right in zip(retrieval, retrieval_again, strict=True):
        assert np.array_equal(left.candidate_bins, right.candidate_bins)


def test_protocol_bundle_manifest_and_candidate_distribution() -> None:
    bundle = _small_bundle()
    manifest = bundle.manifest
    assert manifest["protocol_version"] == "v2"
    assert manifest["split"]["train"]["count"] == 3
    assert manifest["split"]["test"]["count"] == 1
    distribution = manifest["candidate_distribution"]
    assert distribution["main"]["k"] == 6
    assert distribution["main"]["kept"] == sum(
        len(bundle.main_anchors[name]) for name in ("train", "val", "test")
    )
    assert distribution["retrieval"]["k"] == 32
    assert distribution["retrieval"]["anchors"] == len(bundle.retrieval_anchors)
    # retrieval 只对 evaluation_split(test) 构建
    assert all(record.split == "test" for record in bundle.retrieval_anchors)
    # periodic 与 main 不重叠
    main_keys = {
        (record.work_id, record.anchor_bin)
        for name in ("train", "val", "test")
        for record in bundle.main_anchors[name]
    }
    periodic_keys = {(record.work_id, record.anchor_bin) for record in bundle.periodic_anchors}
    assert main_keys.isdisjoint(periodic_keys)


def test_save_protocol_atomic_files(config: Config) -> None:
    bundle = _small_bundle()
    saved = save_protocol(bundle, config)
    manifest_path = Path(saved["manifest_path"])
    anchors_path = Path(saved["anchors_path"])
    assert manifest_path.is_file() and anchors_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["anchors_sha256"] == saved["anchors_sha256"]
    assert manifest["anchors_file"].startswith("v05_protocol/")
    with np.load(anchors_path, allow_pickle=False) as payload:
        assert "main_work_id" in payload
        assert payload["main_candidate_bins_flat"].size == 6 * manifest["anchors_counts"]["main"]
        assert "periodic_candidate_work_id" in payload
        assert payload["periodic_candidate_work_id"].size == manifest["anchors_counts"]["periodic_candidates"]
        assert "retrieval_candidate_ptr" in payload


# ---------------------------------------------------------------- 指标 / 候选集


def test_rank_metrics_hand_calculation() -> None:
    scores = np.asarray(
        [
            [4.0, 1.0, 2.0, 3.0],
            [3.0, 4.0, 1.0, 2.0],
            [1.0, 2.0, 3.0, 4.0],
        ]
    )
    labels = np.asarray([0, 2, 3])
    mask = np.ones_like(scores, dtype=bool)
    metrics = _rank_metrics(scores, labels, mask, k_values=(1, 3))
    assert metrics["top1"] == pytest.approx(2.0 / 3.0)
    assert metrics["recall"]["1"] == pytest.approx(2.0 / 3.0)
    assert metrics["recall"]["3"] == pytest.approx(2.0 / 3.0)
    assert metrics["n_anchors"] == 3
    empty = _rank_metrics(
        np.zeros((0, 4)), np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=bool)
    )
    assert empty["top1"] is None


def test_random_and_simple_sync_metrics() -> None:
    bundle = _small_bundle()
    main_records = bundle.main_anchors["test"]
    sets = candidate_sets_from_records(bundle.pairs, main_records)
    simple = _simple_sync_metrics(sets, bootstrap=20, seed=42)
    assert simple["n_anchors"] == sum(item.n_anchors for item in sets)
    assert simple["top1"] is not None
    assert set(simple["recall"]) >= {"1", "5", "10"}
    assert simple["bootstrap"] is not None
    random_metrics = v05_protocol._random_baseline(6, n_anchors=10, n_works=2)
    assert random_metrics["top1"] == pytest.approx(1.0 / 6.0)
    assert random_metrics["recall"]["5"] == pytest.approx(5.0 / 6.0)


def test_bootstrap_ci_deterministic() -> None:
    per_work = {"1": 0.1, "2": 0.3, "3": 0.8, "4": 0.6}
    first = _bootstrap_ci(per_work, seed=42, samples=100)
    second = _bootstrap_ci(per_work, seed=42, samples=100)
    assert first == second
    assert first is not None
    assert first["ci95"][0] <= first["mean"] + 1e-9
    assert first["ci95"][1] + 1e-9 >= first["mean"]
    assert _bootstrap_ci({}, seed=1, samples=10) is None


# ---------------------------------------------------------------- checkpoint adapter / 端到端


def _save_fake_checkpoint(
    path: Path,
    *,
    pair: v05_beta.WorkPair,
    standardizer: dict[str, np.ndarray],
    visual_encoder: str = "clip",
    audio_version: str = "test_audio",
) -> Path:
    model = v05_beta.build_model(
        visual_dim=pair.visual_dim,
        audio_dim=pair.audio_dim,
        standardizer=standardizer,
        hidden=8,
        embed_dim=8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v05_beta_test",
        "seed": 42,
        "config": {},
        "dataset": {
            "dataset_version": "v05_beta_v1",
            "audio_version": audio_version,
            "visual_encoder": visual_encoder,
            "n_works_available": 6,
        },
        "model_config": {
            "visual_dim": pair.visual_dim,
            "audio_dim": pair.audio_dim,
            "hidden": 8,
            "embed_dim": 8,
            "temperature": 0.07,
        },
        "standardizer": standardizer,
        "state_dict": model.state_dict(),
        "optimizer": {},
        "epoch": 1,
        "best_val_top1": 0.3,
        "test_top1": 0.3,
    }
    torch.save(payload, path)
    return path


def test_old_checkpoint_adapter_roundtrip(tmp_path: Path) -> None:
    pair = _make_pair(1)
    standardizer = v05_beta.fit_standardizer([pair])
    checkpoint_path = _save_fake_checkpoint(
        tmp_path / "v05_beta_fake" / "checkpoint.pt",
        pair=pair,
        standardizer=standardizer,
    )
    protocol_info = {
        "visual_encoder": "clip",
        "audio_version": "test_audio",
        "visual_dim": pair.visual_dim,
        "audio_dim": pair.audio_dim,
    }
    loaded = load_old_checkpoint(
        checkpoint_path, protocol_info=protocol_info, device=torch.device("cpu")
    )
    assert loaded.name == "v05_beta_fake"
    assert loaded.info["model_config"]["hidden"] == 8
    original_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    for key, value in loaded.model.state_dict().items():
        assert torch.allclose(value, original_state[key])
    with pytest.raises(V05ProtocolError):
        load_old_checkpoint(
            checkpoint_path,
            protocol_info={**protocol_info, "visual_encoder": "dino"},
            device=torch.device("cpu"),
        )


def test_evaluate_protocol_end_to_end(config: Config, tmp_path: Path) -> None:
    bundle = _small_bundle()
    standardizer = v05_beta.fit_standardizer(
        [bundle.pairs[work_id] for work_id in bundle.split["train"]]
    )
    checkpoint_path = _save_fake_checkpoint(
        tmp_path / "v05_beta_42" / "checkpoint.pt",
        pair=bundle.pairs[1],
        standardizer=standardizer,
    )
    cfg = ProtocolConfig(
        batch_size=32,
        bootstrap=20,
        device="cpu",
        checkpoint_paths=(str(checkpoint_path),),
    )
    metrics = evaluate_protocol(config, bundle, cfg)
    assert Path(metrics["artifacts"]["metrics"]).is_file()
    assert Path(metrics["artifacts"]["csv"]).is_file()
    assert Path(metrics["artifacts"]["protocol_manifest"]).is_file()
    variants = {row["variant"] for row in metrics["rows"]}
    assert {"main_k6", "retrieval_k32", "periodic", "periodic_candidate"} <= variants
    models = {row["model"] for row in metrics["rows"]}
    assert {"v05_beta_42", "random", "frozen_projection", "simple_sync"} <= models
    checkpoint_rows = [
        row for row in metrics["rows"] if row["model"] == "v05_beta_42"
    ]
    main_row = next(row for row in checkpoint_rows if row["variant"] == "main_k6")
    assert main_row["k"] == 6
    assert main_row["top1"] is not None
    assert main_row["chance_top1"] == pytest.approx(1.0 / 6.0)
    retrieval_row = next(
        row for row in checkpoint_rows if row["variant"] == "retrieval_k32"
    )
    assert retrieval_row["k"] == 32
    assert retrieval_row["top1"] is not None
    assert retrieval_row["chance_top1"] == pytest.approx(1.0 / 32.0)
    assert metrics["candidate_distribution"]["main"]["k"] == 6
    assert "旧 beta Recall@5" in " ".join(metrics["notes"])


def test_cli_parser_smoke() -> None:
    parser = protocol_tool._build_parser()
    args = parser.parse_args(
        ["--main-k", "6", "--retrieval-k", "32", "--checkpoints", "a.pt", "b.pt"]
    )
    cfg = protocol_tool._config_from_args(args)
    assert cfg.main_k == 6
    assert cfg.retrieval_k == 32
    assert cfg.checkpoint_paths == ("a.pt", "b.pt")
    assert protocol_tool._format_value(None) == "—"
