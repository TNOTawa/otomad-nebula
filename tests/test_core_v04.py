"""v0.4 核心层（配置版本、schema v3、表征/序列登记）的单元测试。"""

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
