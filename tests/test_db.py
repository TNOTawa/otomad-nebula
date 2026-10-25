from __future__ import annotations

import pytest

from nebula.db import (
    TRANSITIONS,
    Catalog,
    work_key_for_bilibili,
    work_key_for_local,
)
from nebula.errors import DatabaseError


def test_initialize_idempotent(catalog: Catalog):
    catalog.initialize()
    tables = {
        row[0]
        for row in catalog.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    for name in ("works", "work_events", "lists", "appearances", "embeddings", "clips", "map_runs"):
        assert name in tables


def test_local_work_upsert_is_idempotent(catalog: Catalog):
    work_id, created = catalog.upsert_local_work("a/b.mp4")
    assert created is True
    again_id, created_again = catalog.upsert_local_work("a/b.mp4")
    assert created_again is False
    assert again_id == work_id
    assert work_key_for_local("a\\b.mp4") == "local:a/b.mp4"
    work = catalog.get_work(work_id)
    assert work["work_key"] == "local:a/b.mp4"
    assert work["source"] == "local"
    assert work["status"] == "pending"
    assert work["attempts"] == 0


def test_local_work_requires_relative_path(catalog: Catalog):
    with pytest.raises(ValueError):
        work_key_for_local("C:/abs/path.mp4")
    with pytest.raises(ValueError):
        work_key_for_local("../escape.mp4")


def test_bilibili_dedupe_and_part(catalog: Catalog):
    id1, created = catalog.upsert_bilibili_work("BV1xx411c7mD", title="t1")
    id2, created2 = catalog.upsert_bilibili_work("BV1xx411c7mD", title="t2")
    assert created is True and created2 is False
    assert id1 == id2
    assert catalog.get_work(id1)["title"] == "t2"  # COALESCE 补全/更新
    idp2, _ = catalog.upsert_bilibili_work("BV1xx411c7mD", part=2)
    assert idp2 != id1
    assert work_key_for_bilibili("BV1xx411c7mD", 2) == "bilibili:BV1xx411c7mD:p2"
    with pytest.raises(DatabaseError):
        work_key_for_bilibili("", 1)
    with pytest.raises(DatabaseError):
        catalog.upsert_bilibili_work("BV1xx411c7mD", part=0)


def test_upsert_preserves_status(catalog: Catalog):
    work_id, _ = catalog.upsert_local_work("x.mp4")
    catalog.set_status(work_id, "encoding")
    catalog.upsert_local_work("x.mp4", title="new")
    assert catalog.get_work(work_id)["status"] == "encoding"


def test_status_machine(catalog: Catalog):
    work_id, _ = catalog.upsert_local_work("v.mp4")
    assert catalog.begin_attempt(work_id, "encoding") == "pending"
    work = catalog.get_work(work_id)
    assert work["status"] == "encoding"
    assert work["attempts"] == 1
    catalog.set_status(work_id, "completed")
    assert catalog.get_work(work_id)["status"] == "completed"
    with pytest.raises(DatabaseError):
        catalog.set_status(work_id, "encoding")
    catalog.set_status(work_id, "completed")  # 同状态幂等
    assert catalog.get_work(work_id)["status"] == "completed"


def test_failed_work_can_retry(catalog: Catalog):
    work_id, _ = catalog.upsert_local_work("bad.mp4")
    catalog.begin_attempt(work_id, "encoding")
    catalog.fail(work_id, "boom")
    work = catalog.get_work(work_id)
    assert work["status"] == "failed"
    assert work["error"] == "boom"
    assert work["attempts"] == 1
    catalog.set_status(work_id, "pending")
    catalog.begin_attempt(work_id, "encoding")
    assert catalog.get_work(work_id)["attempts"] == 2


def test_invalid_status_values(catalog: Catalog):
    work_id, _ = catalog.upsert_local_work("v.mp4")
    with pytest.raises(DatabaseError):
        catalog.set_status(work_id, "done")
    with pytest.raises(DatabaseError):
        catalog.set_status(9999, "failed")
    with pytest.raises(DatabaseError):
        catalog.begin_attempt(work_id, "encoding_extra")


def test_transitions_cover_all_statuses():
    from nebula.db import STATUSES

    assert set(TRANSITIONS) == set(STATUSES)


def test_update_metadata_and_events(catalog: Catalog):
    work_id, _ = catalog.upsert_local_work("v.mp4")
    catalog.update_work_metadata(work_id, duration_s=12.5, author_uid="42", has_audio=1)
    work = catalog.get_work(work_id)
    assert work["duration_s"] == 12.5
    assert work["author_uid"] == "42"
    with pytest.raises(DatabaseError):
        catalog.update_work_metadata(work_id, status="completed")
    with pytest.raises(DatabaseError):
        catalog.update_work_metadata(9999, title="x")
    events = catalog.get_events(work_id)
    assert events and events[0]["event"] == "created"


def test_embedding_version_and_embeddings(catalog: Catalog):
    params = {"ablation": None, "preprocess": "v1"}
    catalog.register_embedding_version(
        "v1",
        model_checkpoint="facebook/pe-av-small-16-frame",
        model_revision="abc",
        frames=16,
        clip_seconds=8.0,
        clips_per_work=5,
        audio_sample_rate=48000,
        params=params,
    )
    catalog.register_embedding_version(
        "v1",
        model_checkpoint="facebook/pe-av-small-16-frame",
        model_revision="abc",
        frames=16,
        clip_seconds=8.0,
        clips_per_work=5,
        audio_sample_rate=48000,
        params=params,
    )
    with pytest.raises(DatabaseError):
        catalog.register_embedding_version(
            "v1",
            model_checkpoint="facebook/pe-av-small-16-frame",
            model_revision="abc",
            frames=16,
            clip_seconds=8.0,
            clips_per_work=5,
            audio_sample_rate=48000,
            params={"ablation": "mute"},
        )

    work_id, _ = catalog.upsert_local_work("v.mp4")
    with pytest.raises(DatabaseError):
        catalog.register_embedding(work_id, version="missing", kind="work_joint", path="x.npy", dim=4)
    catalog.register_embedding(
        work_id, version="v1", kind="work_joint", path="embeddings/v1/work.npy", dim=4
    )
    catalog.register_embedding(
        work_id, version="v1", kind="clip_joint", path="embeddings/v1/clip.npy", dim=4, n_clips=3
    )
    assert catalog.has_embedding(work_id, version="v1")
    assert not catalog.has_embedding(work_id, version="v1", kind="work_audio")
    rows = catalog.list_works_with_embedding("v1")
    assert len(rows) == 1
    assert rows[0]["embedding_path"] == "embeddings/v1/work.npy"
    assert rows[0]["embedding_dim"] == 4
    assert len(catalog.list_embeddings("v1")) == 2
    assert len(catalog.list_embeddings("v1", kind="clip_joint")) == 1
    with pytest.raises(DatabaseError):
        catalog.register_embedding(work_id, version="v1", kind="bad_kind", path="x", dim=4)


def test_clips(catalog: Catalog):
    work_id, _ = catalog.upsert_local_work("v.mp4")
    catalog.replace_clips(work_id, version="v1", clips=[(0, 0.0, 8.0), (1, 8.0, 8.0)])
    clips = catalog.get_clips(work_id, version="v1")
    assert [(c["idx"], c["start_s"], c["duration_s"]) for c in clips] == [
        (0, 0.0, 8.0),
        (1, 8.0, 8.0),
    ]
    catalog.replace_clips(work_id, version="v1", clips=[(0, 1.0, 5.0)])
    assert len(catalog.get_clips(work_id, version="v1")) == 1


def test_lists_and_appearances(catalog: Catalog):
    list_id = catalog.ensure_list("votes_2025", name="2025音MAD选票", kind="votes", vote_year=2025)
    assert catalog.ensure_list(
        "votes_2025", name="重复导入", kind="votes", vote_year=2025
    ) == list_id
    work_id, _ = catalog.upsert_bilibili_work("BV1xx411c7mD")
    catalog.add_appearance(work_id, list_id, rank=3, votes=100)
    catalog.add_appearance(work_id, list_id, votes=120)
    apps = catalog.get_appearances(work_id)
    assert len(apps) == 1
    assert apps[0]["rank"] == 3 and apps[0]["votes"] == 120
    assert apps[0]["list_name"] == "2025音MAD选票"
    rows = catalog.appearance_rows()
    assert rows[0]["vote_year"] == 2025
    assert len(catalog.list_lists()) == 1
    with pytest.raises(DatabaseError):
        catalog.ensure_list("bad", name="x", kind="unknown")
    with pytest.raises(DatabaseError):
        catalog.add_appearance(work_id, list_id, rank=-1)


def test_map_runs_and_coords(catalog: Catalog):
    catalog.register_embedding_version(
        "v1",
        model_checkpoint="c",
        model_revision=None,
        frames=16,
        clip_seconds=8.0,
        clips_per_work=5,
        audio_sample_rate=48000,
        params={},
    )
    w1, _ = catalog.upsert_local_work("a.mp4")
    w2, _ = catalog.upsert_local_work("b.mp4")
    assert catalog.next_map_version() == "map_v001"
    catalog.create_map_run(
        "map_v001",
        embedding_version="v1",
        params={"n_neighbors": 15},
        coords=[(w1, 0.1, 0.2, 0.3, 0.4, 0.5), (w2, 1.0, 1.1, 1.2, 1.3, 1.4)],
        coords_path="maps/map_v001/coords.parquet",
    )
    assert catalog.next_map_version() == "map_v002"
    run = catalog.get_map("map_v001")
    assert run["n_works"] == 2 and run["status"] == "ready"
    assert catalog.latest_map()["version"] == "map_v001"
    rows = catalog.load_map_rows("map_v001")
    assert len(rows) == 2
    assert rows[0]["x2"] == 0.1 and rows[0]["z3"] == 0.5
    catalog.replace_map_coords("map_v001", [(w1, 2.0, 2.1, 2.2, 2.3, 2.4)])
    assert len(catalog.load_map_rows("map_v001")) == 1
    with pytest.raises(DatabaseError):
        catalog.create_map_run(
            "map_v001", embedding_version="v1", params={}, coords=[]
        )
    with pytest.raises(DatabaseError):
        catalog.create_map_run(
            "nebula_1", embedding_version="v1", params={}, coords=[]
        )
    with pytest.raises(DatabaseError):
        catalog.replace_map_coords("missing", [])


def test_stats(catalog: Catalog):
    w, _ = catalog.upsert_local_work("a.mp4")
    catalog.begin_attempt(w, "encoding")
    catalog.fail(w, "x")
    stats = catalog.stats()
    assert stats["works_total"] == 1
    assert stats["status"]["failed"] == 1
    assert stats["maps"] == 0

def test_schema_v2_migration_adds_filter_json(tmp_path):
    from nebula.db import SCHEMA_VERSION

    db_path = tmp_path / "migrate.sqlite"
    first = Catalog(db_path)
    first.initialize()
    first.conn.execute("ALTER TABLE map_runs DROP COLUMN filter_json")
    first.conn.execute("UPDATE schema_meta SET value = '1' WHERE key = 'schema_version'")
    first.close()

    migrated = Catalog(db_path)
    migrated.initialize()
    columns = {row["name"] for row in migrated.conn.execute("PRAGMA table_info(map_runs)")}
    assert "filter_json" in columns
    version = migrated.conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()["value"]
    assert version == SCHEMA_VERSION
    tables = {
        row[0]
        for row in migrated.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    for name in ("work_links", "pipeline_runs", "work_timings"):
        assert name in tables
    migrated.close()


def test_work_links(catalog):
    local_id, _ = catalog.upsert_local_work("x/a.mp4")
    bili_id, _ = catalog.upsert_bilibili_work("BV1xx411c7mD")
    link_id = catalog.link_works(bili_id, local_id, method="title+duration", confidence=0.98)
    assert link_id > 0
    assert catalog.get_duplicate_work_ids() == {local_id}
    assert catalog.get_canonical_work_id(local_id) == bili_id
    assert catalog.get_canonical_work_id(bili_id) == bili_id
    link = catalog.get_work_link(local_id)
    assert link["canonical_work_id"] == bili_id
    duplicates = catalog.get_linked_duplicates(bili_id)
    assert [row["id"] for row in duplicates] == [local_id]
    assert duplicates[0]["link_method"] == "title+duration"
    assert len(catalog.get_work_links()) == 1
    with pytest.raises(DatabaseError):
        catalog.link_works(bili_id, local_id, method="again")
    with pytest.raises(DatabaseError):
        catalog.link_works(local_id, local_id, method="self")
    with pytest.raises(DatabaseError):
        catalog.link_works(9999, bili_id, method="missing")
    with pytest.raises(DatabaseError):
        catalog.link_works(local_id, bili_id, method="canonical_is_duplicate")
    assert catalog.unlink_work(local_id) == bili_id
    assert catalog.get_duplicate_work_ids() == set()
    with pytest.raises(DatabaseError):
        catalog.unlink_work(local_id)


def test_pipeline_runs_and_timings(catalog):
    run_id = catalog.start_pipeline_run(embedding_version="v1", params={"note": "test"})
    work_id, _ = catalog.upsert_local_work("v.mp4")
    catalog.record_work_timing(
        run_id,
        work_id,
        embedding_version="v1",
        probe_s=0.1,
        extract_s=1.2,
        preprocess_s=2.0,
        forward_s=3.0,
        save_s=0.2,
        total_s=6.5,
    )
    catalog.record_work_timing(
        run_id,
        work_id,
        embedding_version="v1",
        extract_s=1.0,
        total_s=6.3,
    )
    timings = catalog.list_work_timings(run_id=run_id)
    assert len(timings) == 1
    assert timings[0]["total_s"] == 6.3
    with pytest.raises(DatabaseError):
        catalog.record_work_timing(run_id, work_id, embedding_version="v1", total_s=-1)
    catalog.finish_pipeline_run(
        run_id, n_selected=1, n_completed=1, n_failed=0, total_s=6.5, model_load_s=25.0
    )
    runs = catalog.list_pipeline_runs()
    assert runs[0]["model_load_s"] == 25.0
    assert runs[0]["finished_at"]
    assert runs[0]["n_completed"] == 1
    with pytest.raises(DatabaseError):
        catalog.finish_pipeline_run(
            9999, n_selected=0, n_completed=0, n_failed=0, total_s=0
        )
    stats = catalog.stats()
    assert stats["pipeline_runs"] == 1
    assert stats["timings"] == 1


def test_map_run_filter_spec(catalog):
    catalog.register_embedding_version(
        "v1",
        model_checkpoint="c",
        model_revision=None,
        frames=16,
        clip_seconds=8.0,
        clips_per_work=5,
        audio_sample_rate=48000,
        params={},
    )
    work_id, _ = catalog.upsert_local_work("a.mp4")
    catalog.create_map_run(
        "map_v001",
        embedding_version="v1",
        params={"n_neighbors": 8},
        coords=[(work_id, 0.0, 0.0, 0.0, 0.0, 0.0)],
        filter_spec={"list_key": "votes_2025", "n_neighbors": 8},
    )
    run = catalog.get_map("map_v001")
    assert "votes_2025" in run["filter_json"]
    catalog.create_map_run(
        "map_v002",
        embedding_version="v1",
        params={"n_neighbors": 15},
        coords=[(work_id, 0.0, 0.0, 0.0, 0.0, 0.0)],
    )
    assert catalog.get_map("map_v002")["filter_json"] is None
