"""v0.5-beta 局部帧级时序模型（nebula.v05_local + CLI）的 CPU 单元测试。

不读真实数据、不联网、不训练大模型：合成协议/合成帧 + 小 config 验证
协议契约、候选 collate、temporal/cross 模型、InfoNCE、评估指标、bootstrap、
消融、checkpoint 与 --smoke CLI。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nebula.v05_local import (  # noqa: E402
    ARCHS,
    FrameDataStore,
    LocalModelConfig,
    LocalPairDataset,
    LocalProtocol,
    LocalTrainConfig,
    RelativeTimeEncoding,
    TemporalOnlyModel,
    V05LocalError,
    _batch_to_device,
    _parse_record,
    bin_center_seconds,
    build_local_model,
    build_synthetic_batches,
    collate_local_sets,
    downsample_audio_tokens,
    evaluate_local,
    extract_visual_tokens,
    info_nce_loss,
    load_checkpoint,
    load_protocol,
    make_synthetic_protocol,
    save_checkpoint,
    train_local_model,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "tools" / "train_v05_local.py"


def _tiny_config(**overrides) -> LocalModelConfig:
    base = {
        "hidden": 32,
        "heads": 4,
        "visual_dim": 16,
        "audio_dim": 12,
        "context_seconds": 1.0,
        "audio_hz": 25.0,
        "self_layers": 1,
        "cross_layers": 1,
        "dropout": 0.0,
    }
    base.update(overrides)
    return LocalModelConfig(**base)  # type: ignore[arg-type]


def _tiny_batches() -> dict[str, list[dict[str, np.ndarray]]]:
    protocol = make_synthetic_protocol(
        n_works=9, bins_per_work=16, anchors_per_work=2, candidates=4, seed=11
    )
    return build_synthetic_batches(
        protocol=protocol,
        visual_dim=16,
        audio_dim=12,
        context_seconds=1.0,
        audio_hz=25.0,
        batch_size=3,
        seed=11,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 配置 / 协议契约
# ──────────────────────────────────────────────────────────────────────────────


def test_model_config_validation():
    with pytest.raises(V05LocalError, match="整除"):
        LocalModelConfig(hidden=30, heads=4)
    with pytest.raises(V05LocalError, match="context_seconds"):
        LocalModelConfig(context_seconds=0.5)
    with pytest.raises(V05LocalError, match="audio_hz"):
        LocalModelConfig(audio_hz=5.0)
    with pytest.raises(V05LocalError, match="temperature"):
        LocalModelConfig(temperature=2.0)
    with pytest.raises(V05LocalError, match="rel_time_mode"):
        LocalModelConfig(rel_time_mode="time2vec")


def _write_protocol(tmp_path: Path) -> Path:
    manifest = {
        "protocol_version": "v2",
        "bin_seconds": 0.5,
        "baselines": {"mlp_top1": 0.25, "simple_sync_top1": 0.2},
        "splits": {"train": [1, 2], "val": [3], "test": [4]},
        "records": [
            {
                "work_id": 1,
                "anchor_bin": 4,
                "candidate_bins": [4, 5, 6],
                "labels": [1, 0, 0],
                "periodic": False,
            },
            {
                "work_id": 2,
                "anchor_bin": 8,
                "candidate_bins": [8, 9, 10],
                "labels": 1,
                "split": "train",
            },
            {
                "work_id": 3,
                "anchor_bin": 2,
                "candidate_bins": [2, 3, 4],
                "labels": [0, 0, 1],
                "split": "val",
            },
            {
                "work_id": 4,
                "anchor_bin": 6,
                "candidate_bins": [6, 7, 8, 9],
                "labels": [0, 1, 0, 0],
                "split": "test",
                "periodic": True,
            },
        ],
    }
    path = tmp_path / "protocol"
    path.mkdir()
    (path / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return path


def test_load_protocol_manifest(tmp_path: Path):
    protocol = load_protocol(_write_protocol(tmp_path))
    assert isinstance(protocol, LocalProtocol)
    assert protocol.bin_seconds == 0.5
    assert len(protocol.records) == 4
    splits = protocol.split_records()
    assert {name: len(items) for name, items in splits.items()} == {
        "train": 2,
        "val": 1,
        "test": 1,
    }
    assert protocol.split_work_ids()["train"] == (1, 2)
    assert protocol.records[0].positive_index == 0
    assert protocol.records[1].positive_index == 1
    assert protocol.records[3].periodic is True
    assert protocol.metadata["baselines"]["mlp_top1"] == 0.25
    protocol.validate_disjoint()


def test_load_protocol_jsonl(tmp_path: Path):
    path = tmp_path / "records.jsonl"
    lines = [
        {
            "work_id": index + 1,
            "anchor_bin": 3,
            "candidate_bins": [3, 4, 5],
            "labels": [1, 0, 0],
            "split": split,
        }
        for index, split in enumerate(("train", "train", "val", "test"))
    ]
    path.write_text(
        "\n".join(json.dumps(item) for item in lines) + "\n", encoding="utf-8"
    )
    protocol = load_protocol(path)
    assert len(protocol.records) == 4
    assert protocol.split_work_ids()["test"] == (4,)


def test_load_protocol_errors(tmp_path: Path):
    with pytest.raises(V05LocalError, match="不存在"):
        load_protocol(tmp_path / "missing")
    path = tmp_path / "bad"
    path.mkdir()
    (path / "manifest.json").write_text(
        json.dumps({"records": [{"work_id": 1, "anchor_bin": 1, "candidate_bins": [1, 2], "labels": [0, 0], "split": "train"}]}),
        encoding="utf-8",
    )
    with pytest.raises(V05LocalError, match="正例"):
        load_protocol(path)
    # split 重叠
    overlap = tmp_path / "overlap"
    overlap.mkdir()
    (overlap / "manifest.json").write_text(
        json.dumps(
            {
                "records": [
                    {"work_id": 1, "anchor_bin": 1, "candidate_bins": [1, 2], "labels": [1, 0], "split": "train"},
                    {"work_id": 1, "anchor_bin": 2, "candidate_bins": [2, 3], "labels": [1, 0], "split": "test"},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(V05LocalError, match="重叠"):
        load_protocol(overlap)




def test_load_protocol_anchors_npz(tmp_path: Path):
    """t3 实际磁盘 schema：manifest + anchors.npz（main/retrieval/periodic）。"""

    import hashlib

    anchors_path = tmp_path / "anchors.npz"
    np.savez_compressed(
        anchors_path,
        main_work_id=np.array([1, 2], dtype=np.int64),
        main_split_code=np.array([0, 2], dtype=np.int8),
        main_anchor_bin=np.array([4, 8], dtype=np.int64),
        main_label=np.array([1, 0], dtype=np.int64),
        main_candidate_bins_flat=np.array([4, 5, 6, 8, 9, 10], dtype=np.int64),
        main_candidate_ptr=np.array([0, 3, 6], dtype=np.int64),
        retrieval_work_id=np.array([2], dtype=np.int64),
        retrieval_split_code=np.array([2], dtype=np.int8),
        retrieval_anchor_bin=np.array([8], dtype=np.int64),
        retrieval_label=np.array([0], dtype=np.int64),
        retrieval_candidate_bins_flat=np.array([8, 9, 10], dtype=np.int64),
        retrieval_candidate_ptr=np.array([0, 3], dtype=np.int64),
        periodic_work_id=np.array([1], dtype=np.int64),
        periodic_split_code=np.array([0], dtype=np.int8),
        periodic_anchor_bin=np.array([4], dtype=np.int64),
        periodic_label=np.array([0], dtype=np.int64),
        periodic_candidate_bins_flat=np.array([4, 5], dtype=np.int64),
        periodic_candidate_ptr=np.array([0, 2], dtype=np.int64),
    )
    manifest = {
        "protocol_version": "v2",
        "bin_seconds": 0.5,
        "main_k": 3,
        "retrieval_k": 3,
        "anchors_file": "anchors.npz",
        "anchors_sha256": hashlib.sha256(anchors_path.read_bytes()).hexdigest(),
        "split": {
            "train": {"count": 1, "work_ids": [1]},
            "val": {"count": 0, "work_ids": []},
            "test": {"count": 1, "work_ids": [2]},
        },
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    # records.jsonl 只是兼容文件；anchors 存在时应优先 anchors
    (tmp_path / "records.jsonl").write_text(
        json.dumps(
            {
                "work_id": 99,
                "anchor_bin": 1,
                "candidate_bins": [1, 2],
                "labels": [1, 0],
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    protocol = load_protocol(tmp_path)
    assert len(protocol.records) == 2
    assert len(protocol.retrieval_records) == 1
    assert len(protocol.periodic_records) == 1
    assert protocol.records[0].candidate_bins == (4, 5, 6)
    assert protocol.records[0].positive_index == 1
    assert protocol.split_records()["train"][0].work_id == 1
    assert protocol.split_records()["test"][0].work_id == 2
    assert protocol.retrieval_records[0].candidate_bins == (8, 9, 10)
    assert protocol.periodic_records[0].periodic is True
    assert protocol.metadata["main_k"] == 3


# ──────────────────────────────────────────────────────────────────────────────
# 帧上下文 / 音频下采样 / 合成 dataset
# ──────────────────────────────────────────────────────────────────────────────


def test_bin_center_and_visual_tokens():
    assert bin_center_seconds(4, 0.5) == pytest.approx(2.25)
    with pytest.raises(V05LocalError):
        bin_center_seconds(-1, 0.5)
    frames = np.arange(80, dtype=np.float32).reshape(40, 2)
    features, times, mask = extract_visual_tokens(
        frames, fps=8.0, duration_s=5.0, center_s=2.5, context_s=1.0
    )
    assert features.shape[1] == 2
    assert np.all(np.abs(times) <= 1.0 + 1e-6)
    assert mask.any()
    assert np.all(np.diff(times) > 0)
    # 作品末尾：右侧 token 大量被 mask 掉
    _f, _t, edge_mask = extract_visual_tokens(
        frames, fps=8.0, duration_s=5.0, center_s=4.8, context_s=1.0
    )
    assert edge_mask.sum() <= mask.sum()


def test_downsample_audio_tokens():
    times = np.arange(750, dtype=np.float32) / 75.0
    features = np.expand_dims(np.arange(750, dtype=np.float32), axis=1)
    tokens, rel_times, mask = downsample_audio_tokens(
        features, times, center_s=5.0, context_s=1.0, target_hz=25.0
    )
    assert tokens.shape == (25, 1)
    assert rel_times.shape == (25,)
    assert mask.shape == (25,)
    assert mask.all()
    assert np.all(np.abs(rel_times) <= 1.0 + 1e-6)


def test_synthetic_protocol_and_collate():
    protocol = make_synthetic_protocol(
        n_works=9, bins_per_work=20, anchors_per_work=2, candidates=5, seed=3
    )
    assert len(protocol.records) == 18
    protocol.validate_disjoint()
    batches = build_synthetic_batches(
        protocol=protocol, visual_dim=16, audio_dim=12, batch_size=4, seed=3
    )
    for name in ("train", "val", "test"):
        assert batches[name], name
        batch = batches[name][0]
        assert batch["visual"].ndim == 3
        assert batch["audio"].ndim == 4
        assert batch["candidate_mask"].dtype == bool
        assert batch["labels"].shape == (batch["visual"].shape[0],)
        assert batch["candidate_mask"].all()
        assert np.all(batch["visual_mask"].any(axis=1))
        assert np.all(batch["audio_mask"].any(axis=2))


# ──────────────────────────────────────────────────────────────────────────────
# 模型 / InfoNCE / padding 不变性
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("arch", ["temporal", "cross"])
def test_models_forward_and_padding_invariance(arch: str):
    config = _tiny_config()
    batches = _tiny_batches()
    model = build_local_model(arch, config).eval()
    raw = batches["train"][0]
    batch = _batch_to_device(raw, "cpu")
    with torch.no_grad():
        first = model(batch)
    assert set(first) >= {"query_emb", "candidate_emb", "logits"}
    assert first["query_emb"].shape == (batch["visual"].shape[0], config.hidden)
    assert first["logits"].shape == (batch["candidate_mask"].shape[0], batch["candidate_mask"].shape[1])
    assert torch.allclose(
        first["query_emb"].norm(dim=-1), torch.ones_like(first["query_emb"].norm(dim=-1)), atol=1e-5
    )
    assert torch.isfinite(first["logits"]).all()
    # 改 padded token / 无效候选内容不影响输出
    perturbed = {key: (value.copy() if isinstance(value, np.ndarray) else value) for key, value in raw.items()}
    for key, mask_key in (("visual", "visual_mask"), ("audio", "audio_mask")):
        array = perturbed[key]
        mask = perturbed[mask_key]
        invalid = ~mask
        if invalid.any():
            array[invalid] = np.random.default_rng(0).normal(size=array[invalid].shape).astype(np.float32) * 10
    for row in range(perturbed["candidate_mask"].shape[0]):
        for column in range(perturbed["candidate_mask"].shape[1]):
            if not perturbed["candidate_mask"][row, column]:
                perturbed["audio"][row, column] = 99.0
                perturbed["audio_mask"][row, column] = False
    with torch.no_grad():
        second = model(_batch_to_device(perturbed, "cpu"))
    for key in ("query_emb", "candidate_emb", "logits"):
        assert torch.allclose(first[key], second[key], atol=1e-6), (arch, key)


@pytest.mark.parametrize("arch", ["temporal", "cross"])
def test_model_info_nce_backward(arch: str):
    config = _tiny_config()
    model = build_local_model(arch, config)
    batch = _batch_to_device(_tiny_batches()["train"][0], "cpu")
    outputs = model(batch)
    loss = info_nce_loss(outputs["logits"], batch["labels"], candidate_mask=batch["candidate_mask"])
    assert torch.isfinite(loss)
    loss.backward()
    nonzero = sum(
        1
        for parameter in model.parameters()
        if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
    )
    total = sum(1 for _ in model.parameters())
    assert nonzero >= int(total * 0.5), (nonzero, total)


def test_info_nce_known_values():
    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    labels = torch.tensor([0, 2])
    loss = info_nce_loss(logits, labels)
    manual = torch.nn.functional.cross_entropy(logits, labels)
    assert float(loss) == pytest.approx(float(manual))
    better = info_nce_loss(
        torch.tensor([[10.0, 0.0, 0.0], [0.0, 0.0, 10.0]]), labels
    )
    assert float(better) < float(loss)
    with pytest.raises(V05LocalError, match="正例"):
        info_nce_loss(
            logits, labels, candidate_mask=torch.tensor([[True, False, False], [True, True, False]])
        )


def test_relative_time_encoding():
    model = RelativeTimeEncoding(16, mode="sinusoidal", freqs=4)
    times = torch.linspace(-1.0, 1.0, 5)
    output = model(times)
    assert output.shape == (5, 16)
    assert not torch.allclose(output[0], output[-1])
    mlp = RelativeTimeEncoding(16, mode="mlp")
    assert mlp(times).shape == (5, 16)


# ──────────────────────────────────────────────────────────────────────────────
# 评估 / bootstrap / checkpoint / 训练
# ──────────────────────────────────────────────────────────────────────────────


class _OracleModel(torch.nn.Module):
    """按 labels 生成 one-hot logits 的假模型，用于指标测试。"""

    def forward(self, batch):
        labels = batch["labels"].long()
        logits = torch.zeros(
            (labels.shape[0], batch["candidate_mask"].shape[1]), dtype=torch.float32
        )
        logits[torch.arange(labels.shape[0]), labels] = 10.0
        return {
            "query_emb": torch.nn.functional.normalize(torch.randn(labels.shape[0], 4), dim=-1),
            "candidate_emb": torch.zeros(labels.shape[0], logits.shape[1], 4),
            "logits": logits,
        }


def test_evaluate_metrics_and_bootstrap():
    batches = _tiny_batches()["test"]
    model = _OracleModel()
    metrics = evaluate_local(model, batches, device="cpu", bootstrap=20)
    assert metrics["top1"] == 1.0
    assert metrics["recall_at_1"] == 1.0
    assert metrics["top1_k32"] == 1.0
    ci = metrics["bootstrap_ci"]
    assert ci["low"] is not None and ci["high"] is not None
    assert 0.0 <= ci["low"] <= ci["high"] <= 1.0
    assert metrics["collapse"]["effective_rank"] is not None
    assert set(metrics["ablations"]) == {"audio_zero", "visual_zero", "time_shift"}


def test_train_smoke_and_checkpoint(tmp_path: Path):
    batches = _tiny_batches()
    config = _tiny_config()
    train_config = LocalTrainConfig(
        arch="temporal", seeds=(42,), epochs=1, batch_size=3, patience=0, bootstrap=5, model=config
    )
    model = build_local_model("temporal", config)
    result = train_local_model(
        model, batches["train"], batches["val"], train_config, device="cpu", seed=42
    )
    assert result["epochs_run"] == 1
    assert result["history"]
    assert all(np.isfinite(item["train_loss"]) for item in result["history"])
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model=result["model"],
        arch="temporal",
        model_config=config,
        train_config=train_config,
        seed=42,
        metrics={"top1": 0.5},
    )
    loaded, payload = load_checkpoint(checkpoint)
    assert payload["arch"] == "temporal"
    assert payload["metrics"]["top1"] == 0.5
    assert isinstance(loaded, TemporalOnlyModel)


# ──────────────────────────────────────────────────────────────────────────────
# 真实数据格式适配（合成 npz 文件）+ 合成协议读取
# ──────────────────────────────────────────────────────────────────────────────


def _write_fake_work(
    visual_dir: Path, audio_dir: Path, work_id: int, *, visual_dim=16, audio_dim=12
) -> None:
    work_visual = visual_dir / f"{work_id:06d}"
    work_audio = audio_dir / f"{work_id:06d}"
    work_visual.mkdir(parents=True)
    work_audio.mkdir(parents=True)
    frames = np.random.default_rng(work_id).normal(
        size=(40, visual_dim)
    ).astype(np.float16)
    np.savez_compressed(work_visual / "dense_visual.npz", per_frame=frames)
    (work_visual / "manifest.json").write_text(
        json.dumps(
            {
                "version": "clip__fps8",
                "fps": 8.0,
                "duration_s": 5.0,
                "n_frames": 40,
            }
        ),
        encoding="utf-8",
    )
    times = np.arange(375, dtype=np.float32) / 75.0
    audio = np.random.default_rng(work_id + 100).normal(
        size=(375, audio_dim)
    ).astype(np.float16)
    np.savez_compressed(
        work_audio / "audio.npz",
        frame_features=audio,
        frame_times=times,
        duration_s=np.float64(5.0),
    )
    (work_audio / "manifest.json").write_text(
        json.dumps(
            {
                "version": "fake_audio",
                "sample_rate": 24000,
                "frame_rate": 75.0,
                "dim": audio_dim,
                "duration_s": 5.0,
            }
        ),
        encoding="utf-8",
    )


def test_frame_data_store_and_dataset(tmp_path: Path):
    visual_root = tmp_path / "visual_dense" / "clip__fps8"
    audio_root = tmp_path / "audio_dense" / "fake_audio"
    for work_id in (1, 2):
        _write_fake_work(visual_root, audio_root, work_id)
    store = FrameDataStore(
        visual_root=tmp_path / "visual_dense",
        audio_root=tmp_path / "audio_dense",
        context_seconds=1.0,
        audio_hz=25.0,
        bin_seconds=0.5,
    )
    assert store.visual_dim == 16
    assert store.audio_dim == 12
    protocol = LocalProtocol(
        records=(
            _parse_record(
                {
                    "work_id": 1,
                    "anchor_bin": 4,
                    "candidate_bins": [4, 5, 6],
                    "labels": [1, 0, 0],
                    "split": "train",
                }
            ),
        ),
        metadata={"protocol_version": "v2"},
        bin_seconds=0.5,
    )
    dataset = LocalPairDataset(protocol.records, store, bin_seconds=0.5)
    batch = collate_local_sets([dataset[0]])
    assert batch["visual"].shape[0] == 1
    assert batch["audio"].shape[1] == 3
    config = _tiny_config(context_seconds=1.0)
    model = build_local_model("temporal", config).eval()
    device_batch = _batch_to_device(batch, "cpu")
    with torch.no_grad():
        outputs = model(device_batch)
    assert outputs["logits"].shape == (1, 3)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def test_cli_help_offline():
    proc = subprocess.run(
        [sys.executable, str(CLI_PATH), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "InfoNCE" in proc.stdout
    for flag in ("--arch", "--smoke", "--seeds", "--protocol-dir", "--device"):
        assert flag in proc.stdout, flag


def test_cli_smoke_writes_checkpoint_and_metrics(tmp_path: Path):
    out_dir = tmp_path / "smoke"
    proc = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--smoke",
            "--arch",
            "both",
            "--epochs",
            "1",
            "--seeds",
            "42",
            "--batch-size",
            "3",
            "--out-dir",
            str(out_dir),
            "--json",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["synthetic"] is True
    assert set(summary["archs"]) == set(ARCHS)
    for arch in ARCHS:
        checkpoint = out_dir / f"v05_local_{arch}" / "seed_42" / "checkpoint.pt"
        metrics_path = out_dir / f"v05_local_{arch}" / "seed_42" / "metrics.json"
        assert checkpoint.is_file() and metrics_path.is_file()
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert payload["arch"] == arch
        assert payload["test"]["top1"] is not None
