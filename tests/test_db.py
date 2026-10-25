from __future__ import annotations

import pytest

from nebula.db import (
    Catalog,
    TRANSITIONS,
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
