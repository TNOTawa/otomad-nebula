"""v0.4.2 表征层测试：registry / loader / 版本隔离 / 构建 CLI。

覆盖：
- legacy 表征矩阵来自 embeddings 表且排除 work_links duplicate；
- 派生表征登记幂等、向量原子写入与加载、维度不一致报错；
- load_clip_sequence 读取片段顺序/权重/事件版本；
- 构建 CLI 只从已保存 clip 序列重建（不需要源视频/模型）：
  full_mean 与旧 aggregate_work_vector 一致、canonical only、增量与 --force、
  temporal_stats 的 semantic / events 变体、sequence_state 与事件缺失跳过、
  版本/表征隔离、list 输出与命令行入口。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from nebula import representations as R
from nebula.embed import aggregate_work_vector, save_vector
from nebula.errors import NebulaError
from tools import build_representation as build_cli

CLIP_DIM = 4
SEQ_VERSION = "peav-test__f16_full8s_ov0_48k"
SEQ_VERSION_B = "peav-test-h336__f16_full8s_ov0_48k"
LEGACY_VERSION = "peav-test__f16_clip8s_x5_48k"


# --------------------------------------------------------------------------- 测试数据工具
def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _register_sequence(
    catalog,
    version: str = SEQ_VERSION,
    *,
    clips_per_work: int = 0,
    params: dict | None = None,
) -> None:
    catalog.register_embedding_version(
        version,
        model_checkpoint="test-only:hash",
        model_revision=None,
        frames=16,
        clip_seconds=8.0,
        clips_per_work=clips_per_work,
        audio_sample_rate=48000,
        params=params if params is not None else {"sampling_mode": "full_coverage"},
    )


def _add_work(
    catalog,
    config,
    *,
    version: str = SEQ_VERSION,
    name: str = "w.mp4",
    n_clips: int = 5,
    clip_dim: int = CLIP_DIM,
    with_audio: bool = True,
    with_video: bool = True,
    events: np.ndarray | None = None,
    event_version: str = "events_v1",
    state: str | None = "assembled",
    seed: int = 0,
) -> tuple[int, np.ndarray, object]:
    """在序列层写入一部作品的 clip_* npy + DB 记录（不依赖真实视频）。"""

    work_id, _ = catalog.upsert_local_work(name)
    work_dir = config.paths.embeddings / version / f"{work_id:06d}"
    work_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    matrices = {"clip_joint": rng.normal(size=(n_clips, clip_dim)).astype(np.float32)}
    if with_audio:
        matrices["clip_audio"] = rng.normal(size=(n_clips, clip_dim)).astype(np.float32)
    if with_video:
        matrices["clip_video"] = rng.normal(size=(n_clips, clip_dim)).astype(np.float32)
    for kind, matrix in matrices.items():
        path = work_dir / f"{kind}.npy"
        save_vector(path, matrix)
        catalog.register_embedding(
            work_id,
            version=version,
            kind=kind,
            path=config.data_relpath(path),
            dim=int(matrix.shape[1]),
            n_clips=int(matrix.shape[0]),
        )
    weights = [1.0 / n_clips] * n_clips
    catalog.replace_clips(
        work_id,
        version=version,
        clips=[
            (index, float(index * 8.0), 8.0, 8.0, float(weights[index]))
            for index in range(n_clips)
        ],
    )
    duration_s = float(n_clips * 8.0)
    catalog.update_work_metadata(work_id, duration_s=duration_s)
    if state is not None:
        catalog.upsert_sequence_state(
            work_id,
            version=version,
            duration_s=duration_s,
            n_clips=n_clips,
            n_clips_done=n_clips,
            status=state,
        )
    if events is not None:
        save_vector(work_dir / "event_features.npy", np.asarray(events, dtype=np.float32))
        (work_dir / "event_manifest.json").write_text(
            json.dumps({"version": event_version}), encoding="utf-8"
        )
    return work_id, matrices["clip_joint"], work_dir


def _add_legacy_vectors(
    catalog,
    config,
    *,
    version: str = LEGACY_VERSION,
    name: str = "legacy.mp4",
    clip_dim: int = CLIP_DIM,
    seed: int = 0,
) -> tuple[int, dict[str, np.ndarray]]:
    """写 legacy 的 work_joint/audio/video 向量（旧版本无 clip 序列）。"""

    work_id, _ = catalog.upsert_local_work(name)
    rng = np.random.default_rng(seed)
    vectors = {
        kind: _unit(rng.normal(size=clip_dim))
        for kind in ("work_joint", "work_audio", "work_video")
    }
    work_dir = config.paths.embeddings / version / f"{work_id:06d}"
    work_dir.mkdir(parents=True, exist_ok=True)
    for kind, vector in vectors.items():
        path = work_dir / f"{kind}.npy"
        save_vector(path, vector)
        catalog.register_embedding(
            work_id, version=version, kind=kind, path=config.data_relpath(path), dim=clip_dim
        )
    return work_id, vectors


def _representation_row(catalog, name: str, version: str) -> dict:
    row = catalog.get_representation(name=name, sequence_version=version)
    assert row is not None
    return row


# --------------------------------------------------------------------------- registry / loader
def test_legacy_representation_matrix_matches_embeddings_and_excludes_duplicates(
    config, catalog
):
    _register_sequence(catalog, LEGACY_VERSION, clips_per_work=5, params={"legacy": True})
    w1, vectors1 = _add_legacy_vectors(catalog, config, name="a.mp4", seed=1)
    w2, vectors2 = _add_legacy_vectors(catalog, config, name="b.mp4", seed=2)
    duplicate, _ = _add_legacy_vectors(catalog, config, name="dup.mp4", seed=3)
    catalog.link_works(w1, duplicate, method="test", confidence=1.0)

    catalog.ensure_legacy_representation(LEGACY_VERSION)
    rep = _representation_row(catalog, R.LEGACY_NAME, LEGACY_VERSION)
    assert rep["kind"] == "legacy"
    ids, matrix = R.load_representation_matrix(catalog, config, rep, kind="joint")
    assert ids == (w1, w2)
    np.testing.assert_allclose(matrix, np.stack([vectors1["work_joint"], vectors2["work_joint"]]))
    per_work = R.load_representation_vectors(catalog, config, rep, kind="audio")
    assert set(per_work) == {w1, w2}

    # legacy 自动补登（ensure 幂等）
    rep_id = catalog.ensure_legacy_representation(LEGACY_VERSION)
    assert rep_id == int(rep["id"])


def test_derived_representation_registry_roundtrip_and_version_isolation(config, catalog):
    _register_sequence(catalog)
    _register_sequence(catalog, SEQ_VERSION_B)
    w1, _, _ = _add_work(catalog, config, name="a.mp4", seed=1)
    w2, _, _ = _add_work(catalog, config, name="b.mp4", seed=2)

    params = {"aggregation": "test", "stages": 4}
    rep_id = catalog.register_representation(
        R.FULL_MEAN_V1,
        sequence_version=SEQ_VERSION,
        kind="full_mean",
        params=params,
        path_prefix="representations/full_mean_v1/" + SEQ_VERSION,
    )
    assert (
        catalog.register_representation(
            R.FULL_MEAN_V1,
            sequence_version=SEQ_VERSION,
            kind="full_mean",
            params=params,
            path_prefix="representations/full_mean_v1/" + SEQ_VERSION,
        )
        == rep_id
    )
    with pytest.raises(NebulaError):
        catalog.register_representation(
            R.FULL_MEAN_V1,
            sequence_version=SEQ_VERSION,
            kind="full_mean",
            params={"aggregation": "different"},
        )

    rep = catalog.get_representation(rep_id)
    vector = _unit(np.arange(1, 9, dtype=np.float32))
    saved = R.save_representation_vectors(
        catalog, config, rep, w1, {"joint": vector, "audio": vector, "video": vector}
    )
    assert {kind: info["dim"] for kind, info in saved.items()} == {
        "joint": 8,
        "audio": 8,
        "video": 8,
    }
    path = R.representation_dir(config, rep) / f"{w1:06d}" / "joint.npy"
    assert path.is_file()
    ids, matrix = R.load_representation_matrix(catalog, config, rep, kind="joint")
    assert ids == (w1,)
    np.testing.assert_allclose(matrix, vector[None, :])

    # 同表征名 + 另一个 sequence version：独立登记、独立向量目录
    rep_b_id = catalog.register_representation(
        R.FULL_MEAN_V1,
        sequence_version=SEQ_VERSION_B,
        kind="full_mean",
        params=params,
    )
    assert rep_b_id != rep_id
    rep_b = catalog.get_representation(rep_b_id)
    R.save_representation_vectors(catalog, config, rep_b, w2, {"joint": vector * 2})
    assert set(R.load_representation_vectors(catalog, config, rep_b, "joint")) == {w2}
    assert R.load_representation_vectors(catalog, config, rep, "joint")[w1].shape == vector.shape
    assert R.load_representation_vectors(catalog, config, rep, "joint").keys() != R.load_representation_vectors(
        catalog, config, rep_b, "joint"
    ).keys()

    # 清空向量后 loader 不再返回旧数据
    assert catalog.clear_representation_vectors(rep_id) == 3
    assert R.load_representation_vectors(catalog, config, rep, kind="joint") == {}
    with pytest.raises(NebulaError):
        R.load_representation_matrix(catalog, config, rep, kind="joint")


def test_save_representation_vectors_validation(config, catalog):
    _register_sequence(catalog)
    w1, _, _ = _add_work(catalog, config, name="a.mp4")
    rep_id = catalog.register_representation(
        R.TEMPORAL_STATS_V1,
        sequence_version=SEQ_VERSION,
        kind="temporal_stats",
        params={"aggregation": "temporal_stats_v1"},
    )
    rep = catalog.get_representation(rep_id)
    with pytest.raises(NebulaError):
        R.save_representation_vectors(catalog, config, rep, w1, {})
    with pytest.raises(NebulaError):
        R.save_representation_vectors(catalog, config, rep, w1, {"bad_kind": np.ones(4)})
    with pytest.raises(NebulaError):
        R.save_representation_vectors(
            catalog, config, rep, w1, {"joint": np.ones((2, 4), dtype=np.float32)}
        )
    with pytest.raises(NebulaError):
        R.save_representation_vectors(
            catalog, config, rep, w1, {"joint": np.array([1.0, np.nan], dtype=np.float32)}
        )


def test_load_representation_matrix_rejects_dim_mismatch(config, catalog):
    _register_sequence(catalog)
    w1, _, _ = _add_work(catalog, config, name="a.mp4")
    w2, _, _ = _add_work(catalog, config, name="b.mp4")
    rep_id = catalog.register_representation(
        R.FULL_MEAN_V1, sequence_version=SEQ_VERSION, kind="full_mean", params={}
    )
    rep = catalog.get_representation(rep_id)
    for work_id in (w1, w2):
        R.save_representation_vectors(
            catalog, config, rep, work_id, {"joint": np.ones(CLIP_DIM, dtype=np.float32)}
        )
    # 直接往 DB 登记一条不同维度的记录，矩阵加载必须报错而不是静默 stack 失败
    catalog.register_representation_vector(
        rep_id,
        w2,
        kind="audio",
        path="representations/full_mean_v1/x/audio.npy",
        dim=3,
    )
    catalog.register_representation_vector(
        rep_id,
        w1,
        kind="audio",
        path="representations/full_mean_v1/y/audio.npy",
        dim=5,
    )
    with pytest.raises(NebulaError):
        R.load_representation_matrix(catalog, config, rep, kind="audio")


def test_load_clip_sequence_reads_times_weights_and_events(config, catalog):
    _register_sequence(catalog)
    events = np.arange(5 * 32, dtype=np.float32).reshape(5, 32)
    work_id, joint, work_dir = _add_work(
        catalog, config, name="a.mp4", events=events, seed=3
    )
    seq = R.load_clip_sequence(catalog, config, SEQ_VERSION, work_id)
    assert seq.work_id == work_id
    assert seq.version == SEQ_VERSION
    assert seq.n_clips == 5
    assert seq.duration_s == pytest.approx(40.0)
    assert [time.index for time in seq.times] == [0, 1, 2, 3, 4]
    assert [time.start_s for time in seq.times] == [0.0, 8.0, 16.0, 24.0, 32.0]
    assert all(time.valid_duration_s == pytest.approx(8.0) for time in seq.times)
    assert all(time.weight == pytest.approx(0.2) for time in seq.times)
    np.testing.assert_allclose(seq.joint, joint)
    assert seq.audio is not None and seq.video is not None
    np.testing.assert_allclose(seq.event_features, events)
    assert seq.event_version == "events_v1"

    # 事件特征行数与片段数不一致时，loader 忽略事件特征而不是崩溃
    second, _, second_dir = _add_work(catalog, config, name="b.mp4", seed=4)
    save_vector(second_dir / "event_features.npy", np.zeros((2, 32), dtype=np.float32))
    (second_dir / "event_manifest.json").write_text(
        json.dumps({"version": "events_v1"}), encoding="utf-8"
    )
    seq2 = R.load_clip_sequence(catalog, config, SEQ_VERSION, second)
    assert seq2.event_features is None
    assert seq2.event_version == "events_v1"


def test_resolve_representation_and_list_available(config, catalog):
    _register_sequence(catalog)
    w1, _, _ = _add_work(catalog, config, name="a.mp4")
    rep_id = catalog.register_representation(
        R.FULL_MEAN_V1,
        sequence_version=SEQ_VERSION,
        kind="full_mean",
        params={"aggregation": "full_mean_v1"},
    )
    rep = catalog.get_representation(rep_id)
    R.save_representation_vectors(
        catalog, config, rep, w1, {"joint": np.ones(CLIP_DIM, dtype=np.float32)}
    )
    catalog.set_representation_status(rep_id, "ready", dim=CLIP_DIM)

    resolved = R.resolve_representation(
        catalog, config, name="full_mean", sequence_version=SEQ_VERSION
    )
    assert int(resolved["id"]) == rep_id
    assert int(
        R.resolve_representation(
            catalog, config, name="full-mean", sequence_version=SEQ_VERSION
        )["id"]
    ) == rep_id
    with pytest.raises(NebulaError):
        R.resolve_representation(
            catalog, config, name=R.FULL_MEAN_V1, sequence_version="missing-version"
        )
    with pytest.raises(NebulaError):
        R.resolve_representation(catalog, config, name="no_such_representation")

    rows = R.list_available(catalog, config)
    row = next(item for item in rows if item["name"] == R.FULL_MEAN_V1)
    assert row["label"] == R.representation_label(R.FULL_MEAN_V1)
    assert row["maps"] == 0
    assert row["status"] == "ready"

    catalog.create_map_run(
        "map_v001",
        embedding_version=SEQ_VERSION,
        representation_id=rep_id,
        params={"n_neighbors": 2},
        coords=[(w1, 0.0, 0.0, 0.0, 0.0, 0.0)],
        coords_path="maps/map_v001/coords.parquet",
    )
    rows = R.list_available(catalog, config)
    row = next(item for item in rows if item["name"] == R.FULL_MEAN_V1)
    assert row["maps"] == 1


def test_resolve_legacy_by_default_sampling_mode(config, catalog):
    _register_sequence(catalog, config.legacy_embedding_version, clips_per_work=5)
    _add_legacy_vectors(catalog, config, version=config.legacy_embedding_version)
    rep = R.resolve_representation(catalog, config)
    assert rep["name"] == R.LEGACY_NAME
    assert rep["sequence_version"] == config.legacy_embedding_version


# --------------------------------------------------------------------------- 构建 CLI
def test_build_full_mean_from_saved_sequence(config, catalog):
    _register_sequence(catalog)
    w1, joint1, _ = _add_work(catalog, config, name="a.mp4", clip_dim=1024, seed=1)
    w2, joint2, _ = _add_work(catalog, config, name="b.mp4", clip_dim=1024, seed=2)
    duplicate, _, _ = _add_work(catalog, config, name="dup.mp4", clip_dim=1024, seed=3)
    catalog.link_works(w1, duplicate, method="test", confidence=1.0)

    summary = build_cli.build_representation(
        catalog, config, name="full_mean", sequence=SEQ_VERSION
    )
    assert summary["status"] == "ready"
    assert summary["n_built"] == 2
    assert summary["n_excluded_duplicates"] == 1
    assert summary["dim"] == 1024

    rep = _representation_row(catalog, R.FULL_MEAN_V1, SEQ_VERSION)
    assert rep["kind"] == "full_mean"
    assert rep["status"] == "ready"
    assert rep["dim"] == 1024
    assert json.loads(rep["params_json"])["aggregation"] == "full_mean_v1"
    ids, matrix = R.load_representation_matrix(catalog, config, rep, kind="joint")
    assert ids == (w1, w2)
    np.testing.assert_allclose(matrix[0], aggregate_work_vector(joint1), atol=1e-7)
    np.testing.assert_allclose(matrix[1], aggregate_work_vector(joint2), atol=1e-7)
    # 派生向量确实落在 data/representations/<name>/<sequence>/<work_id>/
    for work_id in (w1, w2):
        assert (
            R.representation_dir(config, rep) / f"{work_id:06d}" / "joint.npy"
        ).is_file()
    assert not (R.representation_dir(config, rep) / f"{duplicate:06d}" / "joint.npy").exists()
    # 三部作品都只有 joint/audio/video 向量（clip_dim=1024）
    audio_vectors = R.load_representation_vectors(catalog, config, rep, kind="audio")
    assert set(audio_vectors) == {w1, w2}
    assert audio_vectors[w1].shape == (1024,)


def test_build_full_mean_incremental_and_force(config, catalog):
    _register_sequence(catalog)
    _add_work(catalog, config, name="a.mp4", seed=1)
    _add_work(catalog, config, name="b.mp4", seed=2)
    first = build_cli.build_representation(catalog, config, name="full_mean", sequence=SEQ_VERSION)
    assert first["n_built"] == 2 and first["n_existing"] == 0

    build_cli.build_representation(catalog, config, name="full_mean", sequence=SEQ_VERSION)
    catalog.get_representation(name=R.FULL_MEAN_V1, sequence_version=SEQ_VERSION)

    _add_work(catalog, config, name="c.mp4", seed=3)
    incremental = build_cli.build_representation(
        catalog, config, name="full_mean", sequence=SEQ_VERSION
    )
    assert incremental["n_built"] == 1
    assert incremental["n_existing"] == 2
    assert incremental["status"] == "ready"

    noop = build_cli.build_representation(
        catalog, config, name="full_mean", sequence=SEQ_VERSION
    )
    assert noop["n_built"] == 0
    assert noop["n_existing"] == 3

    forced = build_cli.build_representation(
        catalog, config, name="full_mean", sequence=SEQ_VERSION, force=True
    )
    assert forced["n_built"] == 3
    assert forced["n_existing"] == 0
    rep = _representation_row(catalog, R.FULL_MEAN_V1, SEQ_VERSION)
    assert len(catalog.list_representation_vectors(int(rep["id"]), kind="joint")) == 3


def test_build_temporal_stats_semantic_variant(config, catalog):
    _register_sequence(catalog)
    _add_work(catalog, config, name="a.mp4", clip_dim=1024, seed=1)
    _add_work(catalog, config, name="b.mp4", clip_dim=1024, seed=2)

    summary = build_cli.build_representation(
        catalog, config, name="temporal_stats", sequence=SEQ_VERSION
    )
    assert summary["status"] == "ready"
    assert summary["has_events"] is False
    assert summary["event_version"] is None
    assert summary["dim"] == 6164
    rep = _representation_row(catalog, R.TEMPORAL_STATS_V1, SEQ_VERSION)
    params = json.loads(rep["params_json"])
    assert params["aggregation"] == "temporal_stats_v1"
    assert params["stages"] == config.temporal.stages
    assert params["has_events"] is False
    assert params["event_version"] is None
    assert params["group_weights"]["drift"] == 0.8
    assert params["vector_dims"] == {"joint": 6164, "audio": 6164, "video": 6164}
    ids, matrix = R.load_representation_matrix(catalog, config, rep, kind="joint")
    assert len(ids) == 2 and matrix.shape == (2, 6164)
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), np.ones(2), atol=1e-5)


def test_build_temporal_stats_events_variant_skips_missing_events(config, catalog):
    _register_sequence(catalog)
    events_a = np.random.default_rng(11).normal(size=(5, 32)).astype(np.float32)
    events_b = np.random.default_rng(12).normal(size=(5, 32)).astype(np.float32)
    with_a, _, _ = _add_work(catalog, config, name="a.mp4", clip_dim=1024, events=events_a, seed=1)
    with_b, _, _ = _add_work(catalog, config, name="b.mp4", clip_dim=1024, events=events_b, seed=2)
    missing, _, _ = _add_work(catalog, config, name="c.mp4", clip_dim=1024, seed=3)

    summary = build_cli.build_representation(
        catalog, config, name="temporal_stats", sequence=SEQ_VERSION
    )
    assert summary["has_events"] is True
    assert summary["event_version"] == "events_v1"
    assert summary["dim"] == 6356
    assert summary["n_built"] == 2
    reasons = {item["work_id"]: item["reason"] for item in summary["skipped"]}
    assert missing in reasons and "事件" in reasons[missing]

    rep = _representation_row(catalog, R.TEMPORAL_STATS_V1, SEQ_VERSION)
    params = json.loads(rep["params_json"])
    assert params["has_events"] is True
    assert params["event_version"] == "events_v1"
    assert params["vector_dims"]["joint"] == 6356
    assert params["vector_dims"]["audio"] == 6164
    ids, joint_matrix = R.load_representation_matrix(catalog, config, rep, kind="joint")
    assert ids == (with_a, with_b)
    assert joint_matrix.shape == (2, 6356)
    ids_audio, audio_matrix = R.load_representation_matrix(catalog, config, rep, kind="audio")
    assert ids_audio == (with_a, with_b)
    assert audio_matrix.shape == (2, 6164)


def test_build_skips_incomplete_and_non_assembled_sequences(config, catalog):
    _register_sequence(catalog)
    good, _, _ = _add_work(catalog, config, name="good.mp4", seed=1)
    _add_work(catalog, config, name="in-progress.mp4", seed=2, state="in_progress")
    _add_work(catalog, config, name="failed.mp4", seed=3, state="failed")
    mismatched, _, _ = _add_work(catalog, config, name="mismatch.mp4", seed=4)
    catalog.replace_clips(
        mismatched, version=SEQ_VERSION, clips=[(0, 0.0, 8.0, 8.0, 0.5), (1, 8.0, 8.0, 8.0, 0.5)]
    )

    summary = build_cli.build_representation(
        catalog, config, name="full_mean", sequence=SEQ_VERSION
    )
    assert summary["n_built"] == 1
    skipped_ids = {item["work_id"] for item in summary["skipped"]}
    assert len(skipped_ids) == 3
    assert good not in skipped_ids
    reasons = {item["work_id"]: item["reason"] for item in summary["skipped"]}
    assert any("sequence_state" in reason for reason in reasons.values())
    assert any("不完整" in reason for reason in reasons.values())
    rep = _representation_row(catalog, R.FULL_MEAN_V1, SEQ_VERSION)
    assert set(R.load_representation_vectors(catalog, config, rep, "joint")) == {good}


def test_build_sequence_resolution_and_errors(config, catalog):
    _register_sequence(catalog)
    _register_sequence(
        catalog, config.legacy_embedding_version, clips_per_work=5, params={"legacy": True}
    )
    _add_work(catalog, config, name="a.mp4", seed=1)
    assert build_cli.resolve_sequence_version(catalog, config, "auto") == config.embedding_version
    assert build_cli.resolve_sequence_version(catalog, config, "legacy") == config.legacy_embedding_version
    with pytest.raises(NebulaError):
        build_cli.resolve_sequence_version(catalog, config, "missing-version")
    with pytest.raises(NebulaError):
        build_cli.build_representation(catalog, config, name="transformer", sequence=SEQ_VERSION)
    with pytest.raises(NebulaError):
        build_cli.build_representation(catalog, config, name="full_mean", sequence=SEQ_VERSION, limit=0)
    with pytest.raises(NebulaError):
        build_cli.build_representation(
            catalog, config, name="full_mean", sequence=SEQ_VERSION, force=True, limit=1
        )


def test_build_sequence_auto_resolves_legacy_version(config, catalog):
    legacy = config.legacy_embedding_version
    _register_sequence(catalog, legacy, clips_per_work=5, params={"legacy": True})
    work_id, joint, _ = _add_work(catalog, config, version=legacy, name="legacy.mp4", state=None, seed=1)
    summary = build_cli.build_representation(catalog, config, name="full_mean", sequence="auto")
    assert summary["sequence_version"] == legacy
    assert summary["n_built"] == 1
    rep = _representation_row(catalog, R.FULL_MEAN_V1, legacy)
    ids, matrix = R.load_representation_matrix(catalog, config, rep, kind="joint")
    assert ids == (work_id,)
    np.testing.assert_allclose(matrix[0], aggregate_work_vector(joint), atol=1e-7)


def test_build_force_rebuilds_when_event_variant_changes(config, catalog):
    _register_sequence(catalog)
    w1, _, dir1 = _add_work(catalog, config, name="a.mp4", clip_dim=1024, seed=1)
    w2, _, dir2 = _add_work(catalog, config, name="b.mp4", clip_dim=1024, seed=2)
    semantic = build_cli.build_representation(
        catalog, config, name="temporal_stats", sequence=SEQ_VERSION
    )
    assert semantic["has_events"] is False and semantic["dim"] == 6164

    for _work_id, work_dir, seed in ((w1, dir1, 31), (w2, dir2, 32)):
        rng = np.random.default_rng(seed)
        save_vector(
            work_dir / "event_features.npy",
            rng.normal(size=(5, 32)).astype(np.float32),
        )
        (work_dir / "event_manifest.json").write_text(
            json.dumps({"version": "events_v1"}), encoding="utf-8"
        )

    with pytest.raises(NebulaError, match="force"):
        build_cli.build_representation(
            catalog, config, name="temporal_stats", sequence=SEQ_VERSION
        )
    rebuilt = build_cli.build_representation(
        catalog, config, name="temporal_stats", sequence=SEQ_VERSION, force=True
    )
    assert rebuilt["has_events"] is True
    assert rebuilt["dim"] == 6356
    assert rebuilt["n_built"] == 2
    rep = _representation_row(catalog, R.TEMPORAL_STATS_V1, SEQ_VERSION)
    assert rep["status"] == "ready" and rep["dim"] == 6356
    ids, matrix = R.load_representation_matrix(catalog, config, rep, kind="joint")
    assert ids == (w1, w2) and matrix.shape == (2, 6356)


def test_build_cross_sequence_isolation(config, catalog):
    _register_sequence(catalog)
    _register_sequence(catalog, SEQ_VERSION_B)
    work_a, joint_a, _ = _add_work(catalog, config, name="a.mp4", seed=1)
    work_b, joint_b, _ = _add_work(catalog, config, version=SEQ_VERSION_B, name="b.mp4", seed=2)
    build_cli.build_representation(catalog, config, name="full_mean", sequence=SEQ_VERSION)
    build_cli.build_representation(catalog, config, name="full_mean", sequence=SEQ_VERSION_B)

    rep_a = _representation_row(catalog, R.FULL_MEAN_V1, SEQ_VERSION)
    rep_b = _representation_row(catalog, R.FULL_MEAN_V1, SEQ_VERSION_B)
    assert int(rep_a["id"]) != int(rep_b["id"])
    ids_a, matrix_a = R.load_representation_matrix(catalog, config, rep_a, kind="joint")
    ids_b, matrix_b = R.load_representation_matrix(catalog, config, rep_b, kind="joint")
    assert ids_a == (work_a,) and ids_b == (work_b,)
    np.testing.assert_allclose(matrix_a[0], aggregate_work_vector(joint_a), atol=1e-7)
    np.testing.assert_allclose(matrix_b[0], aggregate_work_vector(joint_b), atol=1e-7)
    assert R.representation_dir(config, rep_a) != R.representation_dir(config, rep_b)


def test_list_rows_and_cli_main(config, catalog, capsys):
    _register_sequence(catalog)
    _add_work(catalog, config, name="a.mp4", seed=1)
    assert build_cli.main(["--config", str(config.path), "list"]) == 0
    out = capsys.readouterr().out
    assert "没有任何已登记的表征" in out

    assert (
        build_cli.main(
            [
                "--config",
                str(config.path),
                "full_mean",
                "--sequence",
                SEQ_VERSION,
            ]
        )
        == 0
    )
    summary_out = capsys.readouterr().out
    assert "status=ready" in summary_out
    assert "新建=1" in summary_out

    rep = _representation_row(catalog, R.FULL_MEAN_V1, SEQ_VERSION)
    rows = build_cli.list_representation_rows(catalog)
    row = next(item for item in rows if int(item["id"]) == int(rep["id"]))
    assert row["label"] == R.representation_label(R.FULL_MEAN_V1)
    assert row["n_joint"] == 1
    assert row["n_audio"] == 1
    assert row["n_video"] == 1
    assert row["n_maps_ready"] == 0

    assert build_cli.main(["--config", str(config.path), "list"]) == 0
    listed = capsys.readouterr().out
    assert "full_mean_v1" in listed
def test_build_marks_failed_on_corrupt_sequence_and_recovers(config, catalog):
    _register_sequence(catalog)
    good, _, good_dir = _add_work(catalog, config, name="good.mp4", seed=1)
    bad, joint_bad, bad_dir = _add_work(catalog, config, name="bad.mp4", seed=2)
    # 破坏序列层的 npy：构建应记该部失败、表征置 failed，而不是写半成品后置 ready
    (bad_dir / "clip_joint.npy").write_bytes(b"not a numpy file")

    failed = build_cli.build_representation(
        catalog, config, name="full_mean", sequence=SEQ_VERSION
    )
    assert failed["status"] == "failed"
    assert failed["n_errors"] == 1
    assert failed["errors"][0]["work_id"] == bad
    rep = _representation_row(catalog, R.FULL_MEAN_V1, SEQ_VERSION)
    assert rep["status"] == "failed"
    assert rep["dim"] == 0

    # 修复损坏文件后，无 --force 的增量构建只补缺失作品并置 ready
    save_vector(bad_dir / "clip_joint.npy", joint_bad)
    recovered = build_cli.build_representation(
        catalog, config, name="full_mean", sequence=SEQ_VERSION
    )
    assert recovered["status"] == "ready"
    assert recovered["n_built"] == 1
    assert recovered["n_existing"] == 1
    rep = _representation_row(catalog, R.FULL_MEAN_V1, SEQ_VERSION)
    assert rep["status"] == "ready" and rep["dim"] == CLIP_DIM
    ids, matrix = R.load_representation_matrix(catalog, config, rep, kind="joint")
    assert ids == (good, bad)
    assert matrix.shape == (2, CLIP_DIM)
