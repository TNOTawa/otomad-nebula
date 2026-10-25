from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from nebula.config import Config
from nebula.db import Catalog
from nebula.embed import HashTestEmbedder, load_vector
from nebula.errors import PipelineError
from nebula.ingest import register_local_videos
from nebula.pipeline import Pipeline


def _copy_video(config: Config, source: Path, rel_path: str) -> Path:
    target = config.paths.input_local / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _make_synth_video(path: Path, *, duration: int = 40, freq: int = 330) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("缺少 ffmpeg")
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=160x120:rate=10:duration={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={freq}:sample_rate=48000:duration={duration}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _open_catalog(config: Config) -> Catalog:
    catalog = Catalog(config.paths.db_path)
    catalog.initialize()
    return catalog


def _pipeline(config: Config, catalog: Catalog) -> Pipeline:
    return Pipeline(
        config,
        catalog=catalog,
        embedder=HashTestEmbedder(),
        allow_test_embedder=True,
    )


def test_process_local_completes_and_is_idempotent(config, sample_video):
    rel = "album/sample.mp4"
    _copy_video(config, sample_video, rel)
    catalog = _open_catalog(config)
    try:
        pipeline = _pipeline(config, catalog)
        summary = register_local_videos(catalog, config)
        assert summary.files == 1
        assert summary.created == 1
        stats = pipeline.process_pending(source="local", verify_sync=1)
        assert stats.completed == 1 and stats.failed == 0
        work = catalog.list_works()[0]
        assert work["status"] == "completed"
        assert work["attempts"] == 1
        assert work["duration_s"] == pytest.approx(12.0, abs=0.2)
        assert work["has_audio"] == 1
        version = pipeline.embedding_version
        kind_rows = {row["kind"]: row for row in catalog.list_embeddings(version)}
        assert set(kind_rows) == {
            "work_joint",
            "work_audio",
            "work_video",
            "clip_joint",
            "clip_audio",
            "clip_video",
        }
        joint = load_vector(config.data_path(kind_rows["work_joint"]["path"]))
        assert joint.shape == (32,)
        assert np.linalg.norm(joint) == pytest.approx(1.0, abs=1e-5)
        clip_joint = load_vector(config.data_path(kind_rows["clip_joint"]["path"]))
        assert clip_joint.shape == (1, 32)  # 12s 视频只有 1 个 8s 片段
        clips = catalog.get_clips(work["id"], version=version)
        assert len(clips) == 1
        assert clips[0]["start_s"] == pytest.approx(2.0)
        assert clips[0]["duration_s"] == pytest.approx(8.0)
        assert not (config.paths.temp / f"work_{work['id']:06d}").exists()
        sync = json.loads((config.paths.logs / "sync_report.json").read_text(encoding="utf-8"))
        assert sync["all_ok"] is True
        assert len(sync["checks"]) == 1
        # 第二次运行：已有当前版本向量 -> 跳过，不重新编码
        stats2 = pipeline.process_pending(source="local")
        assert stats2.selected == 0
        assert stats2.outcomes == []
        assert catalog.get_work(work["id"])["attempts"] == 1
        pipeline.close()
    finally:
        catalog.close()


def test_multiple_clips_for_long_video(config, tmp_path):
    source = _make_synth_video(tmp_path / "long.mp4", duration=40, freq=220)
    _copy_video(config, source, "long.mp4")
    catalog = _open_catalog(config)
    try:
        pipeline = _pipeline(config, catalog)
        register_local_videos(catalog, config)
        stats = pipeline.process_pending(source="local")
        assert stats.completed == 1
        work = catalog.list_works()[0]
        version = pipeline.embedding_version
        clips = catalog.get_clips(work["id"], version=version)
        assert [round(c["start_s"], 2) for c in clips] == [0.0, 8.0, 16.0, 24.0, 32.0]
        joint = load_vector(
            config.data_path(
                catalog.get_embedding(work["id"], version=version, kind="clip_joint")["path"]
            )
        )
        assert joint.shape == (5, 32)
        pipeline.close()
    finally:
        catalog.close()


def test_failed_work_does_not_block_others(config, sample_video):
    _copy_video(config, sample_video, "good.mp4")
    bad = config.paths.input_local / "bad.mp4"
    bad.write_bytes(b"not a real video")
    catalog = _open_catalog(config)
    try:
        pipeline = _pipeline(config, catalog)
        register_local_videos(catalog, config)
        stats = pipeline.process_pending(source="local", verify_sync=0)
        assert stats.completed == 1
        assert stats.failed == 1
        failed = [o for o in stats.outcomes if o.status == "failed"][0]
        assert "MediaError" in (failed.message or "") or "失败" in (failed.message or "")
        bad_work = catalog.get_work_by_key("local:bad.mp4")
        good_work = catalog.get_work_by_key("local:good.mp4")
        assert bad_work["status"] == "failed" and bad_work["error"]
        assert good_work["status"] == "completed"
        # probe 阶段失败时还没创建临时目录，不应残留任何临时文件
        assert not (config.paths.temp / f"work_{bad_work['id']:06d}").exists()
        # 重试仍失败，attempts 增加
        stats_retry = pipeline.process_pending(source="local", retry_failed=True)
        assert stats_retry.failed == 1
        assert catalog.get_work(bad_work["id"])["attempts"] >= 2
        pipeline.close()
    finally:
        catalog.close()


def test_retry_works_reencodes_same_vector(config, sample_video):
    _copy_video(config, sample_video, "v.mp4")
    catalog = _open_catalog(config)
    try:
        pipeline = _pipeline(config, catalog)
        register_local_videos(catalog, config)
        pipeline.process_pending(source="local", verify_sync=0)
        work = catalog.list_works()[0]
        version = pipeline.embedding_version
        before = load_vector(
            config.data_path(catalog.get_embedding(work["id"], version=version)["path"])
        )
        stats = pipeline.retry_works([work["id"]])
        assert stats.completed == 1
        after = load_vector(
            config.data_path(catalog.get_embedding(work["id"], version=version)["path"])
        )
        assert np.allclose(before, after, atol=1e-6)
        pipeline.close()
    finally:
        catalog.close()


def test_limit(config, tmp_path):
    for index in range(3):
        source = _make_synth_video(tmp_path / f"v{index}.mp4", duration=12, freq=200 + index * 50)
        _copy_video(config, source, f"v{index}.mp4")
    catalog = _open_catalog(config)
    try:
        pipeline = _pipeline(config, catalog)
        register_local_videos(catalog, config)
        stats = pipeline.process_pending(source="local", limit=2)
        assert stats.selected == 2
        assert stats.completed == 2
        assert len(catalog.list_works(status="pending")) == 1
        pipeline.close()
    finally:
        catalog.close()


def test_status_and_clean_temp(config, sample_video):
    _copy_video(config, sample_video, "v.mp4")
    catalog = _open_catalog(config)
    try:
        pipeline = _pipeline(config, catalog)
        register_local_videos(catalog, config)
        pipeline.process_pending(source="local")
        status = pipeline.status()
        assert status["works_total"] == 1
        assert status["status"]["completed"] == 1
        assert status["temp_dirs"] == 0
        (config.paths.temp / "leftover").mkdir(parents=True, exist_ok=True)
        assert pipeline.clean_temp() == 1
        pipeline.close()
    finally:
        catalog.close()


def test_test_embedder_requires_explicit_opt_in(config):
    catalog = _open_catalog(config)
    try:
        pipeline = Pipeline(config, catalog=catalog, embedder=HashTestEmbedder())
        with pytest.raises(PipelineError, match="测试用嵌入器"):
            pipeline.process_pending()
        pipeline.close()
    finally:
        catalog.close()


def test_ablation_runs_and_writes_report(config, sample_video):
    _copy_video(config, sample_video, "v.mp4")
    catalog = _open_catalog(config)
    try:
        pipeline = _pipeline(config, catalog)
        register_local_videos(catalog, config)
        pipeline.process_pending(source="local")
        results = pipeline.ablate(limit=1)
        assert len(results) == 1
        entry = results[0]
        assert "error" not in entry
        assert entry["n_clips"] == 1
        assert -1.0 <= entry["joint_vs_mute_cosine"] <= 1.0
        report = json.loads(
            (config.paths.logs / "ablation_report.json").read_text(encoding="utf-8")
        )
        assert report["embedding_version"] == pipeline.embedding_version
        assert len(report["results"]) == 1
        pipeline.close()
    finally:
        catalog.close()

def test_interrupted_encoding_with_vectors_self_heals(config, sample_video):
    """模拟“向量已登记、片段已登记，但状态卡在 encoding”的崩溃窗口。"""

    _copy_video(config, sample_video, "v.mp4")
    catalog = _open_catalog(config)
    try:
        pipeline = _pipeline(config, catalog)
        register_local_videos(catalog, config)
        pipeline.process_pending(source="local", verify_sync=0)
        work = catalog.list_works()[0]
        version = pipeline.embedding_version
        before = load_vector(
            config.data_path(catalog.get_embedding(work["id"], version=version)["path"])
        )
        catalog.reopen(work["id"], reason="模拟处理中断")
        catalog.begin_attempt(work["id"], "encoding")
        assert catalog.get_work(work["id"])["status"] == "encoding"
        attempts = catalog.get_work(work["id"])["attempts"]
        stats = pipeline.process_pending(source="local")
        assert stats.selected == 0
        healed = catalog.get_work(work["id"])
        assert healed["status"] == "completed"
        assert healed["attempts"] == attempts
        after = load_vector(
            config.data_path(catalog.get_embedding(work["id"], version=version)["path"])
        )
        assert np.allclose(before, after, atol=0)
        events = [e["detail"] for e in catalog.get_events(work["id"], limit=10) if e["event"] == "status"]
        assert any("encoding->completed" in (detail or "") for detail in events)
        pipeline.close()
    finally:
        catalog.close()

def test_interrupted_downloading_with_vectors_self_heals_and_batch_continues(config, tmp_path):
    """模拟 Bilibili 作品在 downloading 阶段被杀，且向量/片段齐全时的自愈。

    修复点：downloading -> completed 不是合法迁移，必须先补 downloading -> encoding；
    且自愈失败不能阻塞整批其他作品。
    """

    from nebula.pipeline import PipelineStats  # noqa: F401  (确认类型可用)

    source_a = _make_synth_video(tmp_path / "a.mp4", duration=12, freq=300)
    source_b = _make_synth_video(tmp_path / "b.mp4", duration=12, freq=400)
    _copy_video(config, source_a, "a.mp4")
    _copy_video(config, source_b, "b.mp4")
    catalog = _open_catalog(config)
    try:
        pipeline = _pipeline(config, catalog)
        register_local_videos(catalog, config)
        pipeline.process_pending(source="local", verify_sync=0)
        a_work = catalog.get_work_by_key("local:a.mp4")
        version = pipeline.embedding_version
        assert catalog.has_embedding(a_work["id"], version=version)
        # 模拟 A 在 downloading 阶段崩溃
        catalog.reopen(a_work["id"], reason="模拟下载中断")
        catalog.begin_attempt(a_work["id"], "downloading")
        assert catalog.get_work(a_work["id"])["status"] == "downloading"
        # 新增第三个作品，验证整批不被自愈异常阻塞
        source_c = _make_synth_video(tmp_path / "c.mp4", duration=12, freq=500)
        _copy_video(config, source_c, "c.mp4")
        register_local_videos(catalog, config)
        stats = pipeline.process_pending(source="local")
        assert stats.completed == 1  # 只处理新作品 C
        assert catalog.get_work(a_work["id"])["status"] == "completed"
        c_work = catalog.get_work_by_key("local:c.mp4")
        assert c_work["status"] == "completed"
        pipeline.close()
    finally:
        catalog.close()

def test_pipeline_records_stage_timings_and_perf_csv(config, sample_video, tmp_path):
    _copy_video(config, sample_video, "timed.mp4")
    catalog = _open_catalog(config)
    try:
        pipeline = _pipeline(config, catalog)
        register_local_videos(catalog, config)
        stats = pipeline.process_pending(source="local", verify_sync=0)
        assert stats.completed == 1
        runs = catalog.list_pipeline_runs()
        assert len(runs) == 1
        run = runs[0]
        assert run["embedding_version"] == pipeline.embedding_version
        assert run["n_selected"] == 1 and run["n_completed"] == 1 and run["n_failed"] == 0
        assert run["finished_at"]
        assert run["total_s"] is not None and run["total_s"] >= 0
        assert run["model_load_s"] == 0.0  # HashTestEmbedder 不加载模型
        timings = catalog.list_work_timings(run_id=run["id"])
        assert len(timings) == 1
        row = timings[0]
        for name in ("download_s", "probe_s", "extract_s", "preprocess_s",
                     "forward_s", "save_s", "total_s"):
            assert row[name] >= 0.0
        assert row["total_s"] == pytest.approx(stats.outcomes[0].elapsed_s, abs=0.5)
        assert row["download_s"] == 0.0
        perf = config.paths.logs / f"perf_{run['id']:03d}.csv"
        assert perf.is_file()
        header = perf.read_text(encoding="utf-8").splitlines()[0]
        assert "extract_s" in header and "forward_s" in header
        pipeline.close()
    finally:
        catalog.close()


def test_process_pending_list_key_orders_and_limits(config, tmp_path):
    ids = {}
    for index in range(3):
        source = _make_synth_video(tmp_path / f"l{index}.mp4", duration=12, freq=250 + index * 50)
        _copy_video(config, source, f"l{index}.mp4")
    catalog = _open_catalog(config)
    try:
        pipeline = _pipeline(config, catalog)
        register_local_videos(catalog, config)
        for work in catalog.list_works():
            ids[work["work_key"]] = work["id"]
        list_id = catalog.ensure_list("votes_test", name="测试榜单", kind="votes", vote_year=2025)
        # 排名故意打乱：l1 第 1、l2 第 2、l0 第 3
        catalog.add_appearance(ids["local:l1.mp4"], list_id, rank=1)
        catalog.add_appearance(ids["local:l2.mp4"], list_id, rank=2)
        catalog.add_appearance(ids["local:l0.mp4"], list_id, rank=3)
        stats = pipeline.process_pending(list_key="votes_test", list_limit=2)
        assert stats.selected == 2
        assert [o.work_id for o in stats.outcomes] == [
            ids["local:l1.mp4"],
            ids["local:l2.mp4"],
        ]
        assert catalog.get_work(ids["local:l0.mp4"])["status"] == "pending"
        pipeline.close()
    finally:
        catalog.close()
