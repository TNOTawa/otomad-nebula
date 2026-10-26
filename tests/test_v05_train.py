"""v0.5-alpha 训练/验证（nebula.v05_train + tools/train_v05）的 CPU 离线测试。

全部使用合成 tokens / 微型合成模型（--smoke），不加载 PE-AV、不联网、不需要 GPU；
覆盖确定性 split、变长 collate+mask、三目标训练、验证指标、时间敏感性、产物落盘与
"不写 DB / 不建 map" 边界，以及依赖未就绪时的明确报错。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nebula import v05_train as vt
from nebula.config import Config
from nebula.db import Catalog
from nebula.v05_train import (
    DEFAULT_ORDER_K,
    ORDER_CLASSES,
    ORDER_PERMUTATIONS,
    TrainConfig,
    V05TrainError,
    author_balanced_accuracy,
    collate_works,
    compute_auc,
    duration_r2,
    effective_rank,
    make_av_batch,
    make_mask,
    make_order_batch,
    mean_top_k_jaccard,
    pairwise_cosine_stats,
    run_training,
    split_work_ids,
    synthetic_tokens,
    temporal_transforms,
)
from tools import train_v05 as train_tool

# ---------------------------------------------------------------- 夹具/工具


def _make_work(
    catalog: Catalog,
    index: int,
    *,
    duration_s: float | None = None,
    author: str | None = None,
) -> int:
    work_id, _ = catalog.upsert_local_work(f"v05_work_{index:03d}.mp4")
    catalog.update_work_metadata(
        work_id,
        duration_s=float(8 * (4 + index % 5)) if duration_s is None else duration_s,
        author_name=author if author is not None else f"author-{index % 3}",
    )
    catalog.begin_attempt(work_id, "encoding")
    catalog.complete(work_id)
    return work_id


def _seed_works(catalog: Catalog, count: int) -> list[int]:
    return [_make_work(catalog, index) for index in range(count)]


def _json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- split / 数据


def test_smoke_default_version_is_isolated() -> None:
    from nebula.v05_train import resolve_artifact_version

    assert (
        resolve_artifact_version(TrainConfig(smoke=True)) == "v05_alpha_v1_smoke"
    )
    assert (
        resolve_artifact_version(TrainConfig(smoke=True, version="v05_custom"))
        == "v05_custom"
    )
    assert resolve_artifact_version(TrainConfig()) == "v05_alpha_v1"


def test_split_deterministic_and_sizes() -> None:
    ids = list(range(1, 111))
    split = split_work_ids(ids, seed=42)
    assert len(split.train_ids) == 90
    assert len(split.val_ids) == 20
    assert set(split.train_ids).isdisjoint(split.val_ids)
    assert sorted(set(split.train_ids) | set(split.val_ids)) == ids
    assert split_work_ids(ids, seed=42) == split
    assert split_work_ids(ids, seed=43) != split
    assert split.to_dict()["train_work_ids"] == list(split.train_ids)

    # 小数据集也至少保留 1 个验证作品且训练集非空
    small = split_work_ids([5, 3, 9], seed=7)
    assert len(small.val_ids) == 1
    assert len(small.train_ids) == 2
    with pytest.raises(V05TrainError):
        split_work_ids([])


def test_synthetic_tokens_contract_determinism() -> None:
    first = synthetic_tokens(7, seed=42)
    second = synthetic_tokens(7, seed=42)
    other = synthetic_tokens(8, seed=42)
    segments = int(first["paav"].shape[0])
    assert first["paav"].shape == (segments, 3, 1024)
    assert first["visual_clip"].shape == (segments, 16, 512)
    assert first["visual_dino"].shape == (segments, 16, 384)
    assert first["audio"].shape == (segments, 16, 24)
    assert first["visual_event"].shape == (segments, 16, 12)
    assert first["mask_visual"].shape == (segments, 16)
    assert first["segment_times"].shape == (segments, 4)
    assert first["segment_mask"].shape == (segments,)
    for key in ("paav", "visual_clip", "audio", "segment_times"):
        assert np.array_equal(first[key], second[key])
    assert first["paav"].shape != other["paav"].shape or not np.array_equal(
        first["paav"], other["paav"]
    )
    assert 4 <= segments <= 12
    assert float(first["duration_s"]) == pytest.approx(segments * 8.0)


def test_collate_padding_and_truncation() -> None:
    works = [synthetic_tokens(1), synthetic_tokens(2)]
    max_w = max(int(work["paav"].shape[0]) for work in works)
    batch = collate_works(works, max_segments=128)
    assert batch["segment_mask"].shape == (2, max_w)
    assert batch["segment_mask"][1].all()
    assert batch["masked_segments"].shape == (2, max_w)
    assert not batch["masked_segments"].any()
    assert batch["visual_clip"].shape == (2, max_w, 16, 512)
    assert batch["mask_visual"].dtype == torch.bool

    # 人为造一个短作品：padding 区域必须全零且 segment_mask=False
    short = synthetic_tokens(3)
    short = {key: (value[:3] if value.ndim and value.shape[0] > 3 else value) for key, value in short.items()}
    padded = collate_works([short, works[0]], max_segments=128)
    short_len = int(short["paav"].shape[0])
    assert short_len == 3
    other_len = int(works[0]["paav"].shape[0])
    assert other_len > short_len
    assert padded["segment_mask"][0, short_len:].sum().item() == 0
    assert float(padded["paav"][0, short_len:].abs().sum()) == 0.0
    assert padded["segment_mask"][0, :short_len].all()

    truncated = collate_works(works, max_segments=5)
    assert truncated["segment_mask"].shape == (2, 5)
    assert truncated["paav"].shape == (2, 5, 3, 1024)


def test_make_mask_only_valid_and_min_one() -> None:
    segment_mask = np.array([[True] * 10, [True, True, False, False, False, False, False, False, False, False]])
    rng = np.random.default_rng(0)
    mask = make_mask(segment_mask, rng=rng, ratio=0.2)
    assert mask.shape == segment_mask.shape
    assert not (mask & ~segment_mask).any()
    assert mask[0].sum() == 2
    assert mask[1].sum() == 1
    assert not mask[0].all()


def test_av_batch_corruption_and_labels() -> None:
    works = [synthetic_tokens(index, seed=3) for index in (1, 2, 3)]
    batch = collate_works(works)
    rng = np.random.default_rng(7)
    av_batch, labels = make_av_batch(batch, rng=rng)
    labels_array = labels.numpy()
    assert labels_array.shape == (3,)
    assert set(np.unique(labels_array)) == {0.0, 1.0}
    for row in range(3):
        if labels_array[row] == 0.0:
            source = (row + 1) % 3
            assert torch.equal(av_batch["audio"][row], batch["audio"][source])
            assert torch.equal(av_batch["mask_audio"][row], batch["mask_audio"][source])
        else:
            assert torch.equal(av_batch["audio"][row], batch["audio"][row])


def test_order_batch_indices_and_targets() -> None:
    works = {index: synthetic_tokens(index, seed=11) for index in (1, 2, 3)}
    rng = np.random.default_rng(5)
    batch, targets = make_order_batch(works, [1, 2, 3], rng=rng)
    assert batch is not None and targets is not None
    order_indices = batch["order_indices"]
    assert order_indices.shape == (3, DEFAULT_ORDER_K)
    assert targets.shape == (3,)
    for row in range(3):
        indices = [int(value) for value in order_indices[row]]
        offsets = tuple(index - min(indices) for index in indices)
        assert offsets == ORDER_PERMUTATIONS[int(targets[row])]
    rng_again = np.random.default_rng(5)
    batch_again, targets_again = make_order_batch(works, [1, 2, 3], rng=rng_again)
    assert torch.equal(targets, targets_again)
    assert torch.equal(order_indices, batch_again["order_indices"])
    assert len(ORDER_PERMUTATIONS) == ORDER_CLASSES == 6


def test_real_model_contract_with_synthetic_tokens() -> None:
    try:
        from nebula.v05_model import V05AlphaConfig, V05AlphaModel
    except ImportError:  # pragma: no cover - task-34 未就绪
        pytest.skip("nebula.v05_model 尚未就绪")
    model_config = V05AlphaConfig(
        hidden=16,
        heads=2,
        local_layers=1,
        temporal_layers=1,
        dropout=0.0,
        max_segments=16,
    )
    model = V05AlphaModel(model_config)
    model.eval()
    works = {index: synthetic_tokens(index) for index in (1, 2, 3)}
    batch = collate_works([works[1], works[2]], max_segments=16)
    batch["masked_segments"] = torch.from_numpy(
        make_mask(batch["segment_mask"].numpy(), rng=np.random.default_rng(1))
    )
    outputs = vt._forward(model, batch)
    width = int(batch["paav"].shape[1])
    assert tuple(outputs["work_vector"].shape) == (2, 16)
    assert tuple(outputs["mask_pred"].shape) == (2, width, 1024)
    assert tuple(outputs["av_logits"].shape) == (2,)
    mask_losses = vt.compute_objective_losses(
        outputs, batch, w_mask=1.0, w_av=0.0, w_order=0.0
    )
    assert torch.isfinite(mask_losses["total"])
    assert mask_losses["source"] == "v05_losses"

    av_batch, _ = make_av_batch(
        {key: value for key, value in batch.items() if key != "masked_segments"},
        rng=np.random.default_rng(2),
    )
    av_outputs = vt._forward(model, av_batch)
    av_losses = vt.compute_objective_losses(
        av_outputs, av_batch, w_mask=0.0, w_av=0.5, w_order=0.0
    )
    assert torch.isfinite(av_losses["total"])

    order_batch, order_targets = make_order_batch(
        works, [1, 2], rng=np.random.default_rng(3), max_segments=16
    )
    assert order_batch is not None and order_targets is not None
    order_batch["order_label"] = order_targets
    order_outputs = vt._forward(model, order_batch)
    assert tuple(order_outputs["order_logits"].shape) == (2, 6)
    order_losses = vt.compute_objective_losses(
        order_outputs, order_batch, w_mask=0.0, w_av=0.0, w_order=0.5
    )
    assert torch.isfinite(order_losses["total"])


def test_temporal_transforms_reverse_and_shuffle() -> None:
    work = synthetic_tokens(1, seed=2)
    rng = np.random.default_rng(0)
    reversed_work, shuffled_work = temporal_transforms(work, rng=rng)
    assert np.array_equal(reversed_work["paav"], work["paav"][::-1])
    assert np.array_equal(
        reversed_work["segment_mask"], work["segment_mask"][::-1]
    )
    assert shuffled_work["paav"].shape == work["paav"].shape
    assert np.array_equal(
        np.sort(shuffled_work["segment_times"][:, 0]), np.sort(work["segment_times"][:, 0])
    )
    for key in ("duration_s", "fps"):
        assert np.array_equal(reversed_work[key], work[key])


# ---------------------------------------------------------------- 指标函数


def test_auc_effective_rank_and_pairwise() -> None:
    assert compute_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert compute_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == pytest.approx(0.0)
    assert compute_auc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == pytest.approx(0.5)
    assert compute_auc([0.9], [1]) is None
    assert compute_auc([], []) is None

    vectors = np.array(
        [[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]], dtype=np.float64
    )
    rank = effective_rank(vectors)
    assert rank is not None and 1.5 < rank < 2.5
    assert effective_rank(np.ones((4, 3))) == pytest.approx(0.0)

    stats = pairwise_cosine_stats(np.array([[1, 0], [1, 0], [0, 1]], dtype=float))
    assert stats["mean"] == pytest.approx((1 + 0 + 0) / 3)
    assert stats["min"] == pytest.approx(0.0)
    assert pairwise_cosine_stats(np.zeros((1, 3)))["mean"] is None


def test_av_loss_supports_one_and_two_logit_heads() -> None:
    from nebula.v05_train import av_mismatch_loss

    labels = torch.tensor([1.0, 0.0])
    binary = {"av_logits": torch.tensor([0.2, -0.3], dtype=torch.float32)}
    binary_loss = av_mismatch_loss(binary, labels)
    assert torch.isfinite(binary_loss)
    two_class = {
        "av_logits": torch.tensor([[0.0, 2.0], [2.0, 0.0]], dtype=torch.float32)
    }
    two_class_loss = av_mismatch_loss(two_class, labels)
    assert torch.isfinite(two_class_loss)
    probabilities = vt._av_probabilities(two_class["av_logits"])
    assert probabilities.shape == (2,)
    assert probabilities[0] > 0.8 and probabilities[1] < 0.2


def test_duration_r2_and_author_leak_metrics() -> None:
    durations = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    vectors = np.stack(
        [np.array([value, value * 0.5]) for value in durations]
    )
    assert duration_r2(vectors, durations) == pytest.approx(1.0, abs=1e-6)
    assert duration_r2(vectors[:2], durations[:2]) is None

    cluster_vectors = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float64
    )
    authors = ["a", "a", "b", "b"]
    accuracy = author_balanced_accuracy(cluster_vectors, authors)
    assert accuracy is not None and accuracy >= 0.99
    assert author_balanced_accuracy(cluster_vectors, ["a"] * 4) == pytest.approx(1.0)
    assert author_balanced_accuracy(cluster_vectors[:1], ["a"]) is None


def test_top_k_jaccard_known_neighborhoods() -> None:
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.99, 0.01],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    assert mean_top_k_jaccard(matrix, matrix, k=2) == pytest.approx(1.0)
    rotated = matrix[[1, 0, 3, 2, 4]]
    value = mean_top_k_jaccard(matrix, rotated, k=2)
    assert value is not None and 0.0 <= value <= 1.0
    assert mean_top_k_jaccard(matrix[:2], matrix[:2], k=2) is None


# ---------------------------------------------------------------- smoke 训练产物


def test_smoke_cli_creates_artifacts_and_no_db_writes(
    config: Config, catalog: Catalog
) -> None:
    work_ids = _seed_works(catalog, 8)
    representations_before = catalog.list_representations()
    maps_before = catalog.conn.execute("SELECT COUNT(*) AS n FROM map_runs").fetchone()["n"]

    exit_code = train_tool.main(
        [
            "--config",
            str(config.path),
            "--smoke",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--limit",
            "8",
            "--device",
            "cpu",
            "--version",
            "v05_smoke_test",
            "--quiet",
        ]
    )
    assert exit_code == 0

    checkpoint_path = config.data_path("models") / "v05_smoke_test" / "checkpoint.pt"
    metrics_path = config.paths.logs / "v05_smoke_test_metrics.json"
    vectors_path = (
        config.paths.representations
        / "transformer_alpha_v1"
        / config.full_sequence_version
        / "vectors.npy"
    )
    work_ids_path = vectors_path.parent / "work_ids.npy"
    manifest_path = vectors_path.parent / "manifest.json"
    assert checkpoint_path.is_file()
    assert metrics_path.is_file()
    assert vectors_path.is_file()
    assert work_ids_path.is_file()
    assert manifest_path.is_file()
    split_path = config.paths.logs / "v05_smoke_test_split.json"
    assert split_path.is_file()
    split_payload = _json_load(split_path)
    assert split_payload["seed"] == 42
    assert split_payload["val_work_ids"]

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert set(("config", "state_dict", "optimizer", "epoch", "best_val", "version")) <= set(checkpoint)
    assert checkpoint["version"] == "v05_smoke_test"
    assert checkpoint["state_dict"]
    assert isinstance(checkpoint["epoch"], int) and checkpoint["epoch"] >= 1
    assert checkpoint["split"]["val_work_ids"]

    metrics = _json_load(metrics_path)
    assert metrics["version"] == "v05_smoke_test"
    assert metrics["dataset"]["synthetic"] is True
    assert metrics["dataset"]["n_works"] == 8
    split = metrics["dataset"]["split"]
    assert sorted(split["train_work_ids"] + split["val_work_ids"]) == sorted(work_ids)
    assert split["val_work_ids"]
    assert metrics["training"]["epochs_run"] == 1
    assert len(metrics["training"]["history"]) == 1
    assert isinstance(metrics["training"]["early_stopped"], bool)

    val = metrics["val"]
    for key in (
        "mask_cosine",
        "mask_mse",
        "full_mean_baseline_cosine",
        "full_mean_baseline_mse",
        "av_accuracy",
        "av_auc",
        "av_negative_accuracy",
        "order_accuracy",
        "order_chance",
    ):
        assert key in val
    assert val["order_chance"] == pytest.approx(1.0 / 6.0)
    assert val["mask_cosine"] is not None
    assert val["full_mean_baseline_cosine"] is not None

    sensitivity = metrics["temporal_sensitivity"]
    assert sensitivity["normal_mask_cosine"] is not None
    assert sensitivity["reverse_drop"] == pytest.approx(
        sensitivity["normal_mask_cosine"] - sensitivity["reversed_mask_cosine"]
    )
    assert sensitivity["shuffle_drop"] == pytest.approx(
        sensitivity["normal_mask_cosine"] - sensitivity["shuffled_mask_cosine"]
    )

    representation = metrics["representation"]
    assert representation["n_works"] == 8
    assert representation["effective_rank"] is not None
    assert representation["pairwise_cosine_mean"] is not None
    assert representation["top5_jaccard_full_mean"] is None  # smoke 无 DB 表征
    assert representation["duration_r2"] is not None

    vectors = np.load(vectors_path)
    ids = np.load(work_ids_path)
    hidden = int(metrics["model"]["hidden"])
    assert vectors.shape == (8, hidden)
    assert vectors.dtype == np.float32
    assert ids.shape == (8,)
    assert sorted(int(value) for value in ids) == sorted(work_ids)
    manifest = _json_load(manifest_path)
    assert manifest["synthetic"] is True
    assert manifest["n_works"] == 8

    # 研究产物不得写 DB：无 representations 行、无 map_runs 行
    assert catalog.list_representations() == representations_before
    maps_after = catalog.conn.execute("SELECT COUNT(*) AS n FROM map_runs").fetchone()["n"]
    assert maps_after == maps_before

    # checkpoint 可恢复进同结构小模型
    hidden = int(metrics["model"]["hidden"])
    restored = vt.SyntheticV05Model(hidden=hidden)
    restored.load_state_dict(checkpoint["state_dict"])
    restored.eval()
    batch = collate_works([synthetic_tokens(work_ids[0])])
    with torch.no_grad():
        first = restored(batch)
        second = restored(batch)
    assert first["work_vector"].shape == (1, hidden)
    assert torch.allclose(first["work_vector"], second["work_vector"])


def test_non_smoke_without_dataset_reports_error(
    config: Config, catalog: Catalog, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_works(catalog, 2)
    exit_code = train_tool.main(
        [
            "--config",
            str(config.path),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--device",
            "cpu",
            "--dataset-version",
            "no_such_dataset",
            "--quiet",
        ]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "错误" in captured.err
    with pytest.raises(V05TrainError):
        run_training(
            config,
            catalog,
            TrainConfig(epochs=1, batch_size=1, device="cpu", dataset_version="no_such_dataset"),
        )
