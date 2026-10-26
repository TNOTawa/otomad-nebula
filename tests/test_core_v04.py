"""v0.4/0.4.3 核心层（配置版本、schema v3/v4、表征/序列/视觉登记）的单元测试。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from nebula.config import load_config
from nebula.db import (
    LEGACY_REPRESENTATION_NAME,
    SCHEMA_VERSION,
    Catalog,
)
from nebula.errors import ConfigError, DatabaseError
from nebula.representations import DENSE_VISUAL_CLIP_V1, representation_label


# --------------------------------------------------------------------- config
def _repo_config_data(repo_root: Path) -> dict:
    return yaml.safe_load((repo_root / "config.yaml").read_text(encoding="utf-8"))


def test_repo_config_has_v04_sections(repo_root: Path):
    cfg = load_config(repo_root / "config.yaml")
    assert cfg.sampling.mode in ("legacy", "full_coverage")
    assert cfg.sampling.overlap_seconds == 0.0
    assert cfg.events.enabled is True
    assert cfg.events.audio_rate_hz == 100
    assert cfg.events.video_fps == 30
    assert cfg.events.version == "events_v1"
    assert cfg.temporal.stages == 4
    assert cfg.media.clip_height is None
    assert cfg.media.long_video_seconds == 3600.0
    assert cfg.paths.representations == (cfg.paths.data / "representations").resolve()
    cfg.ensure_directories()
    assert cfg.paths.representations.is_dir()


def test_legacy_and_full_versions_are_distinct(config):
    legacy = config.legacy_embedding_version
    full = config.full_sequence_version
    assert legacy.startswith("peav-small-16f__f16_clip8s_x5_48k")
    assert full.startswith("peav-small-16f__f16_full8s_ov0_48k")
    full_cfg = replace(config, sampling=replace(config.sampling, mode="full_coverage"))
    legacy_cfg = replace(config, sampling=replace(config.sampling, mode="legacy"))
    assert full_cfg.embedding_version == full
    assert legacy_cfg.embedding_version == legacy
    # v0.4 起仓库默认 = full_coverage
    assert config.embedding_version == full


def test_clip_height_changes_version(config):
    low = replace(config, media=replace(config.media, clip_height=336))
    assert low.full_sequence_version.endswith("_h336")
    assert low.legacy_embedding_version.endswith("_h336")
    assert low.full_sequence_version != config.full_sequence_version


def test_invalid_sampling_overlap_rejected(tmp_path: Path, repo_root: Path):
    data = _repo_config_data(repo_root)
    data["sampling"]["mode"] = "full_coverage"
    data["sampling"]["overlap_seconds"] = data["media"]["clip_seconds"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError, match="overlap_seconds"):
        load_config(path)


def test_invalid_events_config_rejected(tmp_path: Path, repo_root: Path):
    data = _repo_config_data(repo_root)
    data["events"]["audio_rate_hz"] = 5
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError, match="audio_rate_hz"):
        load_config(path)


# ------------------------------------------------------------------ schema v3
def _register_version(catalog: Catalog, version: str = "v1") -> None:
    catalog.register_embedding_version(
        version,
        model_checkpoint="c",
        model_revision=None,
        frames=16,
        clip_seconds=8.0,
        clips_per_work=5,
        audio_sample_rate=48000,
        params={},
    )


def test_schema_v3_migration_adds_columns_tables_and_backfill(tmp_path: Path):
    db_path = tmp_path / "migrate_v2.sqlite"
    first = Catalog(db_path)
    first.initialize()
    _register_version(first)
    work_id, _ = first.upsert_local_work("a.mp4")
    first.create_map_run(
        "map_v001", embedding_version="v1", params={}, coords=[(work_id, 0.0, 0.0, 0.0, 0.0, 0.0)]
    )
    # 模拟 v2 数据库：删除 v3 新增的列/表，并把版本号改回 2
    first.conn.execute("DROP TABLE representation_vectors")
    first.conn.execute("DROP TABLE representations")
    first.conn.execute("DROP TABLE sequence_states")
    first.conn.execute("ALTER TABLE map_runs DROP COLUMN representation_id")
    first.conn.execute("ALTER TABLE clips DROP COLUMN weight")
    first.conn.execute("ALTER TABLE clips DROP COLUMN valid_duration_s")
    first.conn.execute("ALTER TABLE work_timings DROP COLUMN events_s")
    first.conn.execute("UPDATE schema_meta SET value = '2' WHERE key = 'schema_version'")
    first.close()

    migrated = Catalog(db_path)
    migrated.initialize()
    try:
        assert (
            migrated.conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()["value"]
            == SCHEMA_VERSION
        )
        clip_cols = {row["name"] for row in migrated.conn.execute("PRAGMA table_info(clips)")}
        assert {"valid_duration_s", "weight"} <= clip_cols
        timing_cols = {
            row["name"] for row in migrated.conn.execute("PRAGMA table_info(work_timings)")
        }
        assert "events_s" in timing_cols
        tables = {
            row[0]
            for row in migrated.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"representations", "representation_vectors", "sequence_states"} <= tables
        legacy = migrated.get_representation(
            name=LEGACY_REPRESENTATION_NAME, sequence_version="v1"
        )
        assert legacy is not None and legacy["kind"] == "legacy" and legacy["status"] == "ready"
        run = migrated.get_map("map_v001")
        assert run["representation_id"] == legacy["id"]
        assert migrated.latest_map(representation_id=legacy["id"])["version"] == "map_v001"
        # 幂等
        migrated.initialize()
        assert len(migrated.list_representations(name=LEGACY_REPRESENTATION_NAME)) == 1
    finally:
        migrated.close()


# ------------------------------------------------------------------ registries
def test_representation_registry_and_vectors(catalog: Catalog):
    _register_version(catalog)
    rep_id = catalog.register_representation(
        "full_mean_v1",
        sequence_version="v1",
        kind="full_mean",
        params={"algo": "mean-v1"},
        note="test",
    )
    assert rep_id > 0
    # 幂等
    assert (
        catalog.register_representation(
            "full_mean_v1", sequence_version="v1", kind="full_mean", params={"algo": "mean-v1"}
        )
        == rep_id
    )
    with pytest.raises(DatabaseError):
        catalog.register_representation(
            "full_mean_v1", sequence_version="v1", kind="full_mean", params={"algo": "other"}
        )
    with pytest.raises(DatabaseError):
        catalog.register_representation(
            "missing_seq", sequence_version="nope", kind="full_mean", params={}
        )
    assert catalog.get_representation(name="full_mean_v1")["id"] == rep_id
    assert catalog.get_representation(rep_id)["name"] == "full_mean_v1"

    w1, _ = catalog.upsert_local_work("a.mp4")
    w2, _ = catalog.upsert_local_work("b.mp4")
    for work_id in (w1, w2):
        catalog.register_representation_vector(
            rep_id, work_id, kind="joint", path=f"representations/full_mean_v1/v1/{work_id:06d}/joint.npy", dim=8
        )
    vectors = catalog.list_representation_vectors(rep_id)
    assert len(vectors) == 2
    assert catalog.list_works_with_representation(rep_id)[0]["representation_dim"] == 8
    assert catalog.clear_representation_vectors(rep_id) == 2
    assert catalog.list_representation_vectors(rep_id) == []

    catalog.set_representation_status(rep_id, "ready", dim=8, note="done")
    ready = catalog.get_representation(rep_id)
    assert ready["status"] == "ready" and ready["dim"] == 8
    assert [row["id"] for row in catalog.list_representations(name="full_mean_v1")] == [rep_id]
    with pytest.raises(DatabaseError):
        catalog.set_representation_status(rep_id, "unknown")


def test_ensure_legacy_representation(catalog: Catalog):
    _register_version(catalog, "legacy_v")
    rep_id = catalog.ensure_legacy_representation("legacy_v")
    assert catalog.ensure_legacy_representation("legacy_v") == rep_id
    assert catalog.get_representation(rep_id)["kind"] == "legacy"
    with pytest.raises(DatabaseError):
        catalog.ensure_legacy_representation("missing")


def test_sequence_states_and_clip_weights(catalog: Catalog):
    _register_version(catalog)
    work_id, _ = catalog.upsert_local_work("a.mp4")
    catalog.upsert_sequence_state(
        work_id,
        version="v1",
        duration_s=180.0,
        n_clips=23,
        n_clips_done=5,
        partial_relpath="embeddings/v1/000001/.partial",
    )
    state = catalog.get_sequence_state(work_id, version="v1")
    assert state["n_clips"] == 23 and state["n_clips_done"] == 5
    catalog.upsert_sequence_state(
        work_id, version="v1", duration_s=180.0, n_clips=23, n_clips_done=23, status="assembled"
    )
    assert catalog.get_sequence_state(work_id, version="v1")["status"] == "assembled"
    assert len(catalog.list_sequence_states(version="v1", status="assembled")) == 1
    with pytest.raises(DatabaseError):
        catalog.upsert_sequence_state(
            work_id, version="v1", duration_s=1.0, n_clips=2, n_clips_done=3
        )
    with pytest.raises(DatabaseError):
        catalog.upsert_sequence_state(
            work_id, version="v1", duration_s=1.0, n_clips=2, status="nope"
        )

    catalog.replace_clips(
        work_id,
        version="v1",
        clips=[(0, 0.0, 8.0, 8.0, 0.6), (1, 7.0, 8.0, 8.0, 0.4)],
    )
    clips = catalog.get_clips(work_id, version="v1")
    assert clips[0]["valid_duration_s"] == 8.0 and clips[0]["weight"] == pytest.approx(0.6)
    catalog.replace_clips(work_id, version="v1", clips=[(0, 0.0, 8.0)])
    clips = catalog.get_clips(work_id, version="v1")
    assert clips[0]["valid_duration_s"] is None and clips[0]["weight"] is None
    catalog.delete_sequence_state(work_id, version="v1")
    assert catalog.get_sequence_state(work_id, version="v1") is None


def test_map_bound_to_representation(catalog: Catalog):
    _register_version(catalog)
    rep_id = catalog.register_representation(
        "full_mean_v1", sequence_version="v1", kind="full_mean", params={}
    )
    w, _ = catalog.upsert_local_work("a.mp4")
    catalog.create_map_run(
        "map_v001",
        embedding_version="v1",
        representation_id=rep_id,
        params={},
        coords=[(w, 0.0, 0.0, 0.0, 0.0, 0.0)],
    )
    run = catalog.get_map("map_v001")
    assert run["representation_id"] == rep_id
    assert catalog.latest_map(representation_id=rep_id)["version"] == "map_v001"
    assert catalog.list_maps(representation_id=rep_id)[0]["version"] == "map_v001"
    with pytest.raises(DatabaseError):
        catalog.create_map_run(
            "map_v002",
            embedding_version="other",
            representation_id=rep_id,
            params={},
            coords=[],
        )
    with pytest.raises(DatabaseError):
        catalog.create_map_run(
            "map_v003", embedding_version="v1", representation_id=9999, params={}, coords=[]
        )

# ---------------------------------------------------------------- v0.4.3 visual
VISUAL_MODEL = "openai/clip-vit-base-patch16"
VISUAL_REVISION = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
VISUAL_VERSION = f"{VISUAL_MODEL}@{VISUAL_REVISION}__fps8_s224_letterbox"


def _register_visual_version(catalog: Catalog, version: str = VISUAL_VERSION) -> None:
    catalog.register_visual_version(
        version,
        model_id=VISUAL_MODEL,
        revision=VISUAL_REVISION,
        preprocessing="letterbox",
        fps=8.0,
        size=224,
        dim=512,
        params={"preprocess": "letterbox", "frame_source": "ffmpeg-stream"},
    )


def test_visual_versions_and_sequences_registry(catalog: Catalog):
    _register_visual_version(catalog)
    # 幂等
    _register_visual_version(catalog)
    row = catalog.get_visual_version(VISUAL_VERSION)
    assert row is not None
    assert row["model_id"] == VISUAL_MODEL
    assert row["revision"] == VISUAL_REVISION
    assert row["preprocessing"] == "letterbox"
    assert row["fps"] == pytest.approx(8.0)
    assert row["size"] == 224
    assert row["dim"] == 512
    versions = catalog.list_visual_versions()
    assert [entry["version"] for entry in versions] == [VISUAL_VERSION]

    # 同版本参数冲突必须报错（不同 revision/preprocess/fps 都应被拒绝）
    with pytest.raises(DatabaseError):
        catalog.register_visual_version(
            VISUAL_VERSION,
            model_id=VISUAL_MODEL,
            revision="other-revision",
            preprocessing="letterbox",
            fps=8.0,
            size=224,
            dim=512,
            params={"preprocess": "letterbox", "frame_source": "ffmpeg-stream"},
        )
    with pytest.raises(DatabaseError):
        catalog.register_visual_version(
            VISUAL_VERSION,
            model_id=VISUAL_MODEL,
            revision=VISUAL_REVISION,
            preprocessing="stretch",
            fps=8.0,
            size=224,
            dim=512,
            params={"preprocess": "letterbox", "frame_source": "ffmpeg-stream"},
        )

    work_id, _ = catalog.upsert_local_work("visual.mp4")
    catalog.upsert_visual_sequence(
        work_id,
        version=VISUAL_VERSION,
        status="in_progress",
        n_frames=100,
        n_windows=3,
    )
    assert catalog.get_visual_sequence(work_id, version=VISUAL_VERSION)["status"] == "in_progress"
    catalog.upsert_visual_sequence(
        work_id,
        version=VISUAL_VERSION,
        status="completed",
        frames_path="visual_dense/x/frames.npz",
        windows_path="visual_dense/x/windows.npy",
        n_frames=100,
        n_windows=3,
        dim=512,
        decode_s=1.5,
        encode_s=2.5,
        model_load_s=0.5,
        total_s=4.5,
    )
    saved = catalog.get_visual_sequence(work_id, version=VISUAL_VERSION)
    assert saved["status"] == "completed"
    assert saved["frames_path"] == "visual_dense/x/frames.npz"
    assert saved["n_windows"] == 3
    assert catalog.visual_status_summary(VISUAL_VERSION)["completed"] == 1
    assert catalog.list_visual_sequences(version=VISUAL_VERSION, status="completed")[0]["work_id"] == work_id

    with pytest.raises(DatabaseError):
        catalog.upsert_visual_sequence(work_id, version=VISUAL_VERSION, status="done")
    with pytest.raises(DatabaseError):
        catalog.upsert_visual_sequence(work_id, version="missing-version", status="completed")
    with pytest.raises(DatabaseError):
        catalog.upsert_visual_sequence(
            work_id, version=VISUAL_VERSION, status="completed", n_frames=-1
        )


def test_visual_representation_kind_and_map_binding(catalog: Catalog):
    _register_version(catalog)
    _register_visual_version(catalog)
    rep_id = catalog.register_representation(
        DENSE_VISUAL_CLIP_V1,
        sequence_version=VISUAL_VERSION,
        kind="visual",
        source_kind="visual",
        params={"aggregation": "l2-normalize-mean-l2-normalize"},
    )
    saved = catalog.get_representation(rep_id)
    assert saved["kind"] == "visual"
    assert saved["source_kind"] == "visual"
    assert representation_label(DENSE_VISUAL_CLIP_V1).startswith("Dense Visual")
    work_id, _ = catalog.upsert_local_work("visual.mp4")
    catalog.register_representation_vector(
        rep_id, work_id, kind="joint", path="representations/visual/v1/000001/joint.npy", dim=512
    )
    catalog.create_map_run(
        "map_v020",
        embedding_version=VISUAL_VERSION,
        representation_id=rep_id,
        params={},
        coords=[(work_id, 0.0, 0.0, 0.0, 0.0, 0.0)],
    )
    run = catalog.get_map("map_v020")
    assert run["representation_id"] == rep_id
    assert run["source_kind"] == "visual"
    assert run["embedding_version"] == VISUAL_VERSION
    assert catalog.latest_map(representation_id=rep_id)["version"] == "map_v020"
    with pytest.raises(DatabaseError):
        catalog.register_representation(
            "bad_kind", sequence_version="v1", kind="unknown", params={}
        )
    with pytest.raises(DatabaseError):
        catalog.register_representation(
            "missing_visual_version",
            sequence_version="not-registered-visual",
            kind="visual",
            source_kind="visual",
            params={},
        )
    with pytest.raises(DatabaseError):
        catalog.create_map_run(
            "map_v021",
            embedding_version="not-registered-visual",
            representation_id=rep_id,
            params={},
            coords=[],
        )


def test_schema_v5_visual_tables_and_migration(tmp_path: Path):
    db_path = tmp_path / "v5.sqlite"
    first = Catalog(db_path)
    first.initialize()
    tables = {
        row[0] for row in first.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {
        "visual_versions",
        "visual_sequences",
        "visual_window_plans",
        "visual_window_sequences",
    } <= tables
    version = first.conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()["value"]
    assert version == SCHEMA_VERSION == "5"
    # 模拟 v4 库重新迁移：v4 -> v5 只加窗口计划表并更新版本
    first.conn.execute("UPDATE schema_meta SET value = '4' WHERE key = 'schema_version'")
    first.close()
    migrated = Catalog(db_path)
    migrated.initialize()
    try:
        assert (
            migrated.conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()["value"]
            == SCHEMA_VERSION
        )
        tables = {
            row[0]
            for row in migrated.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"visual_window_plans", "visual_window_sequences"} <= tables
        _register_visual_version(migrated)
        assert migrated.get_visual_version(VISUAL_VERSION) is not None
    finally:
        migrated.close()


def test_visual_window_plans_and_sequences(catalog: Catalog):
    _register_visual_version(catalog)
    plan = "win8s_ov0__coverage-v1"
    catalog.register_visual_window_plan(
        plan,
        clip_seconds=8.0,
        overlap_seconds=0.0,
        weight_scheme="coverage-v1",
        params={"note": "test"},
    )
    catalog.register_visual_window_plan(
        plan,
        clip_seconds=8.0,
        overlap_seconds=0.0,
        weight_scheme="coverage-v1",
        params={"note": "test"},
    )
    saved_plan = catalog.get_visual_window_plan(plan)
    assert saved_plan is not None and saved_plan["clip_seconds"] == 8.0
    assert [row["version"] for row in catalog.list_visual_window_plans()] == [plan]
    with pytest.raises(DatabaseError):
        catalog.register_visual_window_plan(
            plan, clip_seconds=4.0, overlap_seconds=0.0, weight_scheme="coverage-v1"
        )
    with pytest.raises(DatabaseError):
        catalog.register_visual_window_plan(
            "bad", clip_seconds=0.0, overlap_seconds=0.0, weight_scheme="coverage-v1"
        )

    work_id, _ = catalog.upsert_local_work("window.mp4")
    catalog.upsert_visual_window_sequence(
        work_id,
        frame_version=VISUAL_VERSION,
        window_plan_version=plan,
        status="in_progress",
        n_windows=2,
    )
    catalog.upsert_visual_window_sequence(
        work_id,
        frame_version=VISUAL_VERSION,
        window_plan_version=plan,
        status="completed",
        windows_path="visual_windows/plan/frame/000001/windows.npy",
        window_std_path="visual_windows/plan/frame/000001/window_std.npy",
        window_weights_path="visual_windows/plan/frame/000001/window_weights.npy",
        n_windows=2,
        dim=512,
        total_s=1.5,
    )
    row = catalog.get_visual_window_sequence(
        work_id, frame_version=VISUAL_VERSION, window_plan_version=plan
    )
    assert row is not None
    assert row["status"] == "completed" and row["n_windows"] == 2 and row["dim"] == 512
    assert row["windows_path"].startswith("visual_windows/")
    assert catalog.visual_window_status_summary(plan)["completed"] == 1
    assert catalog.list_visual_window_sequences(window_plan_version=plan)[0]["work_id"] == work_id

    # 不同窗口计划不复用同一窗口记录
    other_plan = "win4s_ov0__coverage-v1"
    catalog.register_visual_window_plan(
        other_plan, clip_seconds=4.0, overlap_seconds=0.0, weight_scheme="coverage-v1"
    )
    assert (
        catalog.get_visual_window_sequence(
            work_id, frame_version=VISUAL_VERSION, window_plan_version=other_plan
        )
        is None
    )
    with pytest.raises(DatabaseError):
        catalog.upsert_visual_window_sequence(
            work_id,
            frame_version=VISUAL_VERSION,
            window_plan_version="missing-plan",
            status="completed",
        )
    with pytest.raises(DatabaseError):
        catalog.upsert_visual_window_sequence(
            work_id,
            frame_version=VISUAL_VERSION,
            window_plan_version=plan,
            status="done",
        )
    with pytest.raises(DatabaseError):
        catalog.upsert_visual_window_sequence(
            work_id,
            frame_version=VISUAL_VERSION,
            window_plan_version=plan,
            status="completed",
            n_windows=-1,
        )

