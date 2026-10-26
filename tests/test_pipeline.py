from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
import types
from dataclasses import replace
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


def _make_synth_video(
    path: Path, *, duration: int = 40, freq: int = 330, has_audio: bool = True
) -> Path:
    """用 ffmpeg 合成测试视频；has_audio=False 时生成无音轨视频。"""

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("缺少 ffmpeg")
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=160x120:rate=10:duration={duration}",
    ]
    if has_audio:
        command += [
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={freq}:sample_rate=48000:duration={duration}",
        ]
    command += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if has_audio:
        command += ["-c:a", "aac", "-ar", "48000", "-ac", "2"]
    command += [str(path)]
    subprocess.run(command, check=True, capture_output=True)
    return path


def _full_config(
    config: Config,
    *,
    events: bool = False,
    clip_height: int | None = None,
    long_video_seconds: float | None = None,
    batch_size: int | None = None,
) -> Config:
    """生成 full_coverage 测试配置：默认关闭 events（避免依赖 t3 的 nebula.events）。"""

    media_fields: dict[str, object] = {}
    if clip_height is not None:
        media_fields["clip_height"] = clip_height
    if long_video_seconds is not None:
        media_fields["long_video_seconds"] = long_video_seconds
    model = replace(config.model, batch_size=batch_size) if batch_size is not None else config.model
    return replace(
        config,
        sampling=replace(config.sampling, mode="full_coverage"),
        events=replace(config.events, enabled=events),
        media=replace(config.media, **media_fields),
        model=model,
    )


def _full_pipeline(
    config: Config, catalog: Catalog, *, allow_long: bool = False, embedder=None
) -> Pipeline:
    return Pipeline(
        config,
        catalog=catalog,
        embedder=embedder if embedder is not None else HashTestEmbedder(),
        allow_test_embedder=True,
        allow_long=allow_long,
    )


def _work_dir(config: Config, version: str, work_id: int) -> Path:
    return config.paths.embeddings / version / f"{work_id:06d}"


class _RecordingEmbedder(HashTestEmbedder):
    """可注入失败、并统计编码片段对数量的测试嵌入器。

    同一个类既用于“注入第 2 批失败”，也用于恢复阶段的计数，避免因
    type(embedder).__name__ 不同导致同一 sequence version 的 params 冲突。
    """

    def __init__(self, *, fail_on_call: int | None = None) -> None:
        super().__init__()
        self.fail_on_call = fail_on_call
        self.calls = 0
        self.encoded_pairs = 0

    def encode_pairs(self, pairs):
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise RuntimeError(f"注入：第 {self.calls} 批编码失败")
        self.encoded_pairs += len(pairs)
        return super().encode_pairs(pairs)


def _open_catalog(config: Config) -> Catalog:
    catalog = Catalog(config.paths.db_path)
    catalog.initialize()
    return catalog


def _pipeline(config: Config, catalog: Catalog) -> Pipeline:
    """legacy 专用流水线：显式切回 legacy 模式，不依赖 config.yaml 的默认值。"""

    return Pipeline(
        config,
        catalog=catalog,
        embedder=HashTestEmbedder(),
        allow_test_embedder=True,
        sampling_mode="legacy",
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


# ──────────────────────────────────────────────────────────────────────────────
# v0.4 full_coverage 模式
# ──────────────────────────────────────────────────────────────────────────────


def _manifest(config: Config, version: str, work_id: int) -> dict:
    path = _work_dir(config, version, work_id) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_full_mode_12s_sequence_layout(config, sample_video):
    _copy_video(config, sample_video, "full12.mp4")
    cfg = _full_config(config)
    catalog = _open_catalog(cfg)
    try:
        pipeline = _full_pipeline(cfg, catalog)
        register_local_videos(catalog, cfg)
        stats = pipeline.process_pending(source="local")
        assert stats.completed == 1 and stats.failed == 0, [
            o.message for o in stats.outcomes
        ]
        work = catalog.list_works()[0]
        work_id = int(work["id"])
        version = pipeline.embedding_version

        # 版本命名：full 序列版本，clips_per_work=0，params 不混入事件字段
        assert version == "test-hash-v1__f16_full8s_ov0_48k"
        version_row = catalog.get_embedding_version(version)
        assert version_row["clips_per_work"] == 0
        params = json.loads(version_row["params_json"])
        assert params["sampling_mode"] == "full_coverage"
        assert params["full_coverage"] is True
        assert params["overlap_seconds"] == 0.0
        assert params["clip_height"] is None
        assert "event_version" not in params
        assert "events_enabled" not in params

        # 片段：12s -> 2 个 8s 窗口，起点 0/4，末尾锚定 12
        clips = catalog.get_clips(work_id, version=version)
        assert len(clips) == 2
        assert [c["idx"] for c in clips] == [0, 1]
        assert [round(c["start_s"], 6) for c in clips] == [0.0, 4.0]
        assert [round(c["duration_s"], 6) for c in clips] == [8.0, 8.0]
        assert [round(c["valid_duration_s"], 6) for c in clips] == [8.0, 8.0]
        assert sum(c["weight"] for c in clips) == pytest.approx(1.0, abs=1e-6)
        assert all(c["weight"] == pytest.approx(0.5, abs=1e-6) for c in clips)

        # 六种向量 kind 都登记；clip_* n_clips=2，work_* 为 None
        kinds = {row["kind"]: row for row in catalog.list_embeddings(version)}
        assert set(kinds) == {
            "work_joint",
            "work_audio",
            "work_video",
            "clip_joint",
            "clip_audio",
            "clip_video",
        }
        for kind in ("clip_joint", "clip_audio", "clip_video"):
            assert kinds[kind]["n_clips"] == 2
            array = load_vector(cfg.data_path(kinds[kind]["path"]))
            assert array.shape == (2, 32)
        for kind in ("work_joint", "work_audio", "work_video"):
            assert kinds[kind]["n_clips"] is None
            array = load_vector(cfg.data_path(kinds[kind]["path"]))
            assert array.shape == (32,)
            assert np.linalg.norm(array) == pytest.approx(1.0, abs=1e-5)

        # clip_times.npy: (T,4) = start, duration, valid, weight
        clip_times = load_vector(_work_dir(cfg, version, work_id) / "clip_times.npy")
        assert clip_times.shape == (2, 4)
        assert clip_times.dtype == np.float32
        assert np.allclose(clip_times[:, 0], [0.0, 4.0])
        assert np.allclose(clip_times[:, 1], [8.0, 8.0])
        assert np.allclose(clip_times[:, 2], [8.0, 8.0])
        assert float(clip_times[:, 3].sum()) == pytest.approx(1.0, abs=1e-6)

        # manifest 与序列状态
        manifest = _manifest(cfg, version, work_id)
        assert manifest["n_clips"] == 2
        assert manifest["duration_s"] == pytest.approx(12.0, abs=0.05)
        assert manifest["sampling"]["mode"] == "full_coverage"
        assert manifest["event_version"] == cfg.events.version
        assert manifest["events_scanned"] is False
        state = catalog.get_sequence_state(work_id, version=version)
        assert state is not None
        assert state["status"] == "assembled"
        assert state["n_clips"] == 2 and state["n_clips_done"] == 2

        # 完成后清理 .partial 与 temp
        assert not (_work_dir(cfg, version, work_id) / ".partial").exists()
        assert not (cfg.paths.temp / f"work_{work_id:06d}").exists()

        # 幂等：已有向量 -> 跳过
        stats2 = pipeline.process_pending(source="local")
        assert stats2.selected == 0
        pipeline.close()
    finally:
        catalog.close()


def test_full_mode_40s_five_windows(config, tmp_path):
    source = _make_synth_video(tmp_path / "full40.mp4", duration=40, freq=220)
    _copy_video(config, source, "full40.mp4")
    cfg = _full_config(config)
    catalog = _open_catalog(cfg)
    try:
        pipeline = _full_pipeline(cfg, catalog)
        register_local_videos(catalog, cfg)
        stats = pipeline.process_pending(source="local")
        assert stats.completed == 1, [o.message for o in stats.outcomes]
        work = catalog.list_works()[0]
        version = pipeline.embedding_version
        clips = catalog.get_clips(int(work["id"]), version=version)
        assert len(clips) == 5
        assert [round(c["start_s"], 6) for c in clips] == [0.0, 8.0, 16.0, 24.0, 32.0]
        assert all(c["duration_s"] == 8.0 for c in clips)
        assert clips[-1]["start_s"] + clips[-1]["duration_s"] == pytest.approx(40.0, abs=1e-6)
        assert sum(c["weight"] for c in clips) == pytest.approx(1.0, abs=1e-6)
        clip_joint = load_vector(
            cfg.data_path(
                catalog.get_embedding(int(work["id"]), version=version, kind="clip_joint")["path"]
            )
        )
        assert clip_joint.shape == (5, 32)
        pipeline.close()
    finally:
        catalog.close()


def test_full_mode_180s_23_windows(config, tmp_path):
    source = _make_synth_video(tmp_path / "full180.mp4", duration=180, freq=330)
    _copy_video(config, source, "full180.mp4")
    cfg = _full_config(config)
    catalog = _open_catalog(cfg)
    try:
        pipeline = _full_pipeline(cfg, catalog)
        register_local_videos(catalog, cfg)
        stats = pipeline.process_pending(source="local")
        assert stats.completed == 1, [o.message for o in stats.outcomes]
        work = catalog.list_works()[0]
        work_id = int(work["id"])
        version = pipeline.embedding_version
        clips = catalog.get_clips(work_id, version=version)
        assert len(clips) == 23
        assert clips[-1]["start_s"] == pytest.approx(172.0, abs=1e-6)
        assert clips[-1]["start_s"] + clips[-1]["duration_s"] == pytest.approx(180.0, abs=1e-3)
        assert sum(c["weight"] for c in clips) == pytest.approx(1.0, abs=1e-6)
        clip_joint = load_vector(
            cfg.data_path(
                catalog.get_embedding(work_id, version=version, kind="clip_joint")["path"]
            )
        )
        assert clip_joint.shape == (23, 32)
        clip_times = load_vector(_work_dir(cfg, version, work_id) / "clip_times.npy")
        assert clip_times.shape == (23, 4)
        assert float(clip_times[:, 3].sum()) == pytest.approx(1.0, abs=1e-6)
        state = catalog.get_sequence_state(work_id, version=version)
        assert state["status"] == "assembled" and state["n_clips_done"] == 23
        assert not (_work_dir(cfg, version, work_id) / ".partial").exists()
        pipeline.close()
    finally:
        catalog.close()


def test_full_mode_no_audio_video(config, tmp_path):
    source = _make_synth_video(tmp_path / "silent12.mp4", duration=12, has_audio=False)
    _copy_video(config, source, "silent12.mp4")
    cfg = _full_config(config)
    catalog = _open_catalog(cfg)
    try:
        pipeline = _full_pipeline(cfg, catalog)
        register_local_videos(catalog, cfg)
        stats = pipeline.process_pending(source="local")
        assert stats.completed == 1, [o.message for o in stats.outcomes]
        work = catalog.list_works()[0]
        assert work["has_audio"] == 0
        version = pipeline.embedding_version
        assert len(catalog.get_clips(int(work["id"]), version=version)) == 2
        kinds = {row["kind"] for row in catalog.list_embeddings(version)}
        assert "clip_audio" in kinds and "work_audio" in kinds
        pipeline.close()
    finally:
        catalog.close()


def test_full_mode_clip_height_changes_version_and_manifest(config, sample_video):
    _copy_video(config, sample_video, "h336.mp4")
    cfg = _full_config(config, clip_height=336)
    catalog = _open_catalog(cfg)
    try:
        pipeline = _full_pipeline(cfg, catalog)
        register_local_videos(catalog, cfg)
        stats = pipeline.process_pending(source="local")
        assert stats.completed == 1, [o.message for o in stats.outcomes]
        assert pipeline.embedding_version.endswith("__f16_full8s_ov0_48k_h336")
        work = catalog.list_works()[0]
        manifest = _manifest(cfg, pipeline.embedding_version, int(work["id"]))
        assert manifest["clip_height"] == 336
        assert cfg.full_sequence_version.endswith("_h336")
        pipeline.close()
    finally:
        catalog.close()


def test_full_mode_rejects_long_video_without_allow_long(config, sample_video):
    _copy_video(config, sample_video, "longish.mp4")
    cfg = _full_config(config, long_video_seconds=10.0)
    catalog = _open_catalog(cfg)
    try:
        pipeline = _full_pipeline(cfg, catalog)
        register_local_videos(catalog, cfg)
        stats = pipeline.process_pending(source="local")
        assert stats.failed == 1 and stats.completed == 0
        failure = stats.outcomes[0]
        assert "long_video_seconds" in (failure.message or "")
        assert "--allow-long" in (failure.message or "")
        work = catalog.list_works()[0]
        assert work["status"] == "failed"
        # 显式允许后可以处理
        pipeline_allowed = _full_pipeline(cfg, catalog, allow_long=True)
        stats_allowed = pipeline_allowed.process_pending(source="local")
        assert stats_allowed.completed == 1, [o.message for o in stats_allowed.outcomes]
        version = pipeline_allowed.embedding_version
        assert len(catalog.get_clips(int(work["id"]), version=version)) == 2
        pipeline_allowed.close()
    finally:
        catalog.close()


def test_full_mode_resumes_missing_batches_after_failure(config, tmp_path):
    source = _make_synth_video(tmp_path / "resume40.mp4", duration=40, freq=300)
    _copy_video(config, source, "resume40.mp4")
    cfg = _full_config(config, batch_size=3)
    catalog = _open_catalog(cfg)
    try:
        pipeline = _full_pipeline(cfg, catalog, embedder=_RecordingEmbedder(fail_on_call=2))
        register_local_videos(catalog, cfg)
        stats = pipeline.process_pending(source="local")
        assert stats.failed == 1, [o.message for o in stats.outcomes]
        work = catalog.list_works()[0]
        work_id = int(work["id"])
        version = pipeline.embedding_version
        partial = _work_dir(cfg, version, work_id) / ".partial"
        assert partial.is_dir()
        assert (partial / "plan.json").is_file()
        assert (partial / "batch_0000.npz").is_file()
        assert not (partial / "batch_0001.npz").exists()
        assert not (_work_dir(cfg, version, work_id) / "clip_joint.npy").exists()
        assert not catalog.has_embedding(work_id, version=version)
        assert (cfg.paths.temp / f"work_{work_id:06d}").is_dir()
        state = catalog.get_sequence_state(work_id, version=version)
        assert state["status"] == "failed"
        assert state["n_clips"] == 5 and state["n_clips_done"] == 3

        # 恢复：只编码缺失的第 2 批（2 个片段）
        counting = _RecordingEmbedder()
        pipeline_resume = _full_pipeline(cfg, catalog, embedder=counting)
        stats2 = pipeline_resume.process_pending(source="local")
        assert stats2.completed == 1, [o.message for o in stats2.outcomes]
        assert counting.encoded_pairs == 2
        clips = catalog.get_clips(work_id, version=version)
        assert len(clips) == 5
        assert not partial.exists()
        assert not (cfg.paths.temp / f"work_{work_id:06d}").exists()
        state = catalog.get_sequence_state(work_id, version=version)
        assert state["status"] == "assembled" and state["n_clips_done"] == 5
        clip_times = load_vector(_work_dir(cfg, version, work_id) / "clip_times.npy")
        assert clip_times.shape == (5, 4)
        pipeline_resume.close()
    finally:
        catalog.close()


def test_full_mode_discards_partial_when_plan_changes(config, tmp_path):
    source = _make_synth_video(tmp_path / "mismatch40.mp4", duration=40, freq=360)
    _copy_video(config, source, "mismatch40.mp4")
    cfg = _full_config(config, batch_size=3)
    catalog = _open_catalog(cfg)
    try:
        pipeline = _full_pipeline(cfg, catalog, embedder=_RecordingEmbedder(fail_on_call=2))
        register_local_videos(catalog, cfg)
        stats = pipeline.process_pending(source="local")
        assert stats.failed == 1
        work = catalog.list_works()[0]
        work_id = int(work["id"])
        version = pipeline.embedding_version
        plan_path = _work_dir(cfg, version, work_id) / ".partial" / "plan.json"
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        payload["plans"][0]["start_s"] = 0.5  # 伪造与当前采样不一致的旧计划
        plan_path.write_text(json.dumps(payload), encoding="utf-8")

        counting = _RecordingEmbedder()
        pipeline_retry = _full_pipeline(cfg, catalog, embedder=counting)
        stats2 = pipeline_retry.process_pending(source="local")
        assert stats2.completed == 1, [o.message for o in stats2.outcomes]
        # partial 被丢弃 -> 全部 5 个片段的重新编码
        assert counting.encoded_pairs == 5
        assert not (_work_dir(cfg, version, work_id) / ".partial").exists()
        pipeline_retry.close()
    finally:
        catalog.close()


def test_full_mode_events_integration_records_events_s(config, sample_video, monkeypatch):
    calls: dict[str, list] = {"scan": [], "save": []}
    fake_module = types.ModuleType("nebula.events")

    class _FakeFeatures:
        version = "events_v1"

    def fake_scan(source, plans, *, ffmpeg="ffmpeg", audio_rate_hz=100, video_fps=30, timeout_s=600.0):
        calls["scan"].append(
            {
                "source": Path(source),
                "plans": list(plans),
                "ffmpeg": ffmpeg,
                "audio_rate_hz": audio_rate_hz,
                "video_fps": video_fps,
                "timeout_s": timeout_s,
            }
        )
        return _FakeFeatures()

    def fake_save(work_dir, features):
        calls["save"].append({"work_dir": Path(work_dir), "features": features})
        target = Path(work_dir) / "event_scan.npz"
        target.write_bytes(b"fake-event-scan")
        (Path(work_dir) / "event_features.npy").write_bytes(b"fake-event-features")

    fake_module.scan_events = fake_scan
    fake_module.save_events_bundle = fake_save
    monkeypatch.setitem(sys.modules, "nebula.events", fake_module)

    _copy_video(config, sample_video, "events12.mp4")
    cfg = _full_config(config, events=True)
    catalog = _open_catalog(cfg)
    try:
        pipeline = _full_pipeline(cfg, catalog)
        register_local_videos(catalog, cfg)
        stats = pipeline.process_pending(source="local")
        assert stats.completed == 1, [o.message for o in stats.outcomes]
        work = catalog.list_works()[0]
        work_id = int(work["id"])
        version = pipeline.embedding_version
        assert len(calls["scan"]) == 1
        scan_call = calls["scan"][0]
        assert len(scan_call["plans"]) == 2
        assert scan_call["audio_rate_hz"] == cfg.events.audio_rate_hz
        assert scan_call["video_fps"] == cfg.events.video_fps
        assert scan_call["ffmpeg"] == cfg.media.ffmpeg
        assert sum(p.weight for p in scan_call["plans"]) == pytest.approx(1.0, abs=1e-6)
        assert len(calls["save"]) == 1
        assert calls["save"][0]["work_dir"] == _work_dir(cfg, version, work_id)
        work_dir = _work_dir(cfg, version, work_id)
        assert (work_dir / "event_scan.npz").is_file()
        manifest = _manifest(cfg, version, work_id)
        assert manifest["events_scanned"] is True
        runs = catalog.list_pipeline_runs()
        timings = catalog.list_work_timings(run_id=runs[0]["id"])
        assert len(timings) == 1
        assert timings[0]["events_s"] >= 0.0
        pipeline.close()
    finally:
        catalog.close()


def test_full_mode_bilibili_reuses_persistent_source_cache(
    config, sample_video, monkeypatch
):
    """sources 持久缓存命中时：不创建 BilibiliClient、不下载、处理完成后缓存保留。"""

    cfg = _full_config(config)
    catalog = _open_catalog(cfg)
    try:
        bvid = "BV1xx411c7m1"
        work_id, _created = catalog.upsert_bilibili_work(
            bvid, title="复用下载测试", duration_s=12.0
        )
        sources_dir = cfg.paths.sources / bvid
        sources_dir.mkdir(parents=True, exist_ok=True)
        cached_source = sources_dir / f"{bvid}_p1.mp4"
        shutil.copy2(sample_video, cached_source)

        def _no_network(*args, **kwargs):
            raise AssertionError("已有 sources 持久缓存时不应创建 BilibiliClient")

        monkeypatch.setattr("nebula.download.BilibiliClient", _no_network)
        pipeline = _full_pipeline(cfg, catalog)
        stats = pipeline.process_pending(source="bilibili")
        assert stats.completed == 1, [o.message for o in stats.outcomes]
        assert catalog.get_work(work_id)["status"] == "completed"
        version = pipeline.embedding_version
        assert len(catalog.get_clips(work_id, version=version)) == 2
        # temp 清理不影响 sources 持久缓存
        assert not (cfg.paths.temp / f"work_{work_id:06d}").exists()
        assert cached_source.is_file()
        pipeline.close()
    finally:
        catalog.close()


def _patch_fake_bilibili_client(monkeypatch) -> dict[str, int]:
    """替换 BilibiliClient 为离线 fake，返回统计（创建次数）。"""

    stats = {"created": 0}

    class FakeBilibiliClient:
        def __init__(self, *args, **kwargs):
            stats["created"] += 1

        def fetch_meta(self, bvid, *, part=1):
            return types.SimpleNamespace(
                title="缓存测试",
                author_uid="1",
                author_name="测试UP",
                duration_s=12.0,
                publish_date="2024-01-01",
            )

    monkeypatch.setattr("nebula.download.BilibiliClient", FakeBilibiliClient)
    return stats


def test_bilibili_source_downloaded_once_and_persisted(
    config, sample_video, monkeypatch
):
    """两次处理（两个采样版本）只下载一次；下载写入 sources 持久缓存。"""

    cfg = _full_config(config)
    catalog = _open_catalog(cfg)
    try:
        bvid = "BV1xx411c7m2"
        work_id, _created = catalog.upsert_bilibili_work(
            bvid, title="源缓存复用测试", duration_s=12.0
        )
        calls: list[tuple[str, int, Path]] = []

        def fake_resolve(
            config_, client_, bvid_, part_, temp_dir_, download_, **kwargs
        ):
            calls.append((bvid_, int(part_), Path(temp_dir_)))
            target_dir = Path(config_.paths.sources) / bvid_
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{bvid_}_p{int(part_)}.mp4"
            shutil.copy2(sample_video, target)
            return target

        monkeypatch.setattr(
            "nebula.download.resolve_bilibili_source", fake_resolve, raising=False
        )
        client_stats = _patch_fake_bilibili_client(monkeypatch)
        cached_source = cfg.paths.sources / bvid / f"{bvid}_p1.mp4"

        # 第一次处理（legacy 版本）：缓存未命中 -> 取元数据 + 通过 helper 下载
        legacy = _pipeline(cfg, catalog)
        stats = legacy.process_pending(source="bilibili")
        assert stats.completed == 1, [o.message for o in stats.outcomes]
        legacy.close()
        assert len(calls) == 1
        assert client_stats["created"] == 1
        assert cached_source.is_file()
        assert not (cfg.paths.temp / f"work_{work_id:06d}").exists()

        # 第二次处理（full_coverage 版本）：sources 缓存命中 -> 不再下载/请求元数据
        full = _full_pipeline(cfg, catalog)
        stats2 = full.process_pending(source="bilibili")
        assert stats2.completed == 1, [o.message for o in stats2.outcomes]
        full.close()
        assert len(calls) == 1, "第二次处理必须复用 sources 缓存，不能重复下载"
        assert client_stats["created"] == 1, "缓存命中不应创建 BilibiliClient"
        assert cached_source.is_file()
        assert not (cfg.paths.temp / f"work_{work_id:06d}").exists()
    finally:
        catalog.close()


def test_bilibili_source_fallback_downloads_once_and_persists(
    config, sample_video, monkeypatch
):
    """helper 尚未交付时的回退：temp 暂存下载 -> 原子移入 sources，第二次复用。"""

    cfg = _full_config(config)
    catalog = _open_catalog(cfg)
    try:
        bvid = "BV1xx411c7m3"
        work_id, _created = catalog.upsert_bilibili_work(
            bvid, title="回退缓存测试", duration_s=12.0
        )
        monkeypatch.setattr(
            "nebula.pipeline._load_resolve_bilibili_source", lambda: None
        )
        downloads: list[Path] = []

        def fake_download(bvid_, *, part, out_dir, download):
            out_path = Path(out_dir)
            downloads.append(out_path)
            target = out_path / f"{bvid_}_p{int(part)}.mp4"
            shutil.copy2(sample_video, target)
            return target

        monkeypatch.setattr("nebula.download.download_bilibili", fake_download)
        client_stats = _patch_fake_bilibili_client(monkeypatch)
        cached_source = cfg.paths.sources / bvid / f"{bvid}_p1.mp4"

        legacy = _pipeline(cfg, catalog)
        stats = legacy.process_pending(source="bilibili")
        assert stats.completed == 1, [o.message for o in stats.outcomes]
        legacy.close()
        assert len(downloads) == 1
        assert client_stats["created"] == 1
        assert cached_source.is_file()
        # 下载暂存目录应随 temp 清理，不残留
        assert not (cfg.paths.temp / f"work_{work_id:06d}").exists()
        assert list(cfg.paths.sources.rglob("*.part")) == []

        full = _full_pipeline(cfg, catalog)
        stats2 = full.process_pending(source="bilibili")
        assert stats2.completed == 1, [o.message for o in stats2.outcomes]
        full.close()
        assert len(downloads) == 1, "第二次处理必须复用 sources 缓存，不能重复下载"
        assert client_stats["created"] == 1
        assert cached_source.is_file()
    finally:
        catalog.close()


def test_bilibili_source_helper_outside_sources_rejected(
    config, sample_video, monkeypatch
):
    """helper 返回非 sources 路径时应失败，避免缓存丢失/写入 temp。"""

    cfg = _full_config(config)
    catalog = _open_catalog(cfg)
    try:
        bvid = "BV1xx411c7m4"
        catalog.upsert_bilibili_work(bvid, title="越界缓存测试", duration_s=12.0)

        def fake_resolve(
            config_, client_, bvid_, part_, temp_dir_, download_, **kwargs
        ):
            return sample_video  # 不在 config.paths.sources 下

        monkeypatch.setattr(
            "nebula.download.resolve_bilibili_source", fake_resolve, raising=False
        )
        _patch_fake_bilibili_client(monkeypatch)
        pipeline = _full_pipeline(cfg, catalog)
        stats = pipeline.process_pending(source="bilibili")
        pipeline.close()
        assert stats.failed == 1
        assert "sources" in (stats.outcomes[0].message or "")
    finally:
        catalog.close()


def test_legacy_mode_regression_columns_and_version(config, sample_video):
    _copy_video(config, sample_video, "legacy12.mp4")
    catalog = _open_catalog(config)
    try:
        pipeline = _pipeline(config, catalog)
        # legacy 测试必须显式切回 legacy，不能依赖 config.yaml 默认（已切到 full_coverage）
        assert pipeline.sampling_mode == "legacy"
        register_local_videos(catalog, config)
        stats = pipeline.process_pending(source="local")
        assert stats.completed == 1
        work = catalog.list_works()[0]
        version = pipeline.embedding_version
        assert version.endswith("__f16_clip8s_x5_48k")
        version_row = catalog.get_embedding_version(version)
        assert version_row["clips_per_work"] == 5
        clips = catalog.get_clips(int(work["id"]), version=version)
        assert len(clips) == 1
        # legacy 行为：valid_duration_s / weight 保持 NULL
        assert clips[0]["valid_duration_s"] is None
        assert clips[0]["weight"] is None
        assert not (_work_dir(config, version, int(work["id"])) / ".partial").exists()
        pipeline.close()
    finally:
        catalog.close()


def test_default_sampling_mode_follows_config(config):
    """config.yaml 默认已是 full_coverage：不传 sampling_mode 时 Pipeline 跟随配置。

    同时验证显式 sampling_mode="legacy" 覆盖配置，避免旧测试依赖默认值。
    """

    assert config.sampling.mode == "full_coverage"
    catalog = _open_catalog(config)
    try:
        default_pipeline = Pipeline(
            config, catalog=catalog, embedder=HashTestEmbedder(), allow_test_embedder=True
        )
        try:
            assert default_pipeline.sampling_mode == "full_coverage"
            assert default_pipeline.embedding_version == "test-hash-v1__f16_full8s_ov0_48k"
        finally:
            default_pipeline.close()

        explicit_legacy = Pipeline(
            config,
            catalog=catalog,
            embedder=HashTestEmbedder(),
            allow_test_embedder=True,
            sampling_mode="legacy",
        )
        try:
            assert explicit_legacy.sampling_mode == "legacy"
            assert explicit_legacy.embedding_version.endswith("__f16_clip8s_x5_48k")
        finally:
            explicit_legacy.close()
    finally:
        catalog.close()


def test_invalid_sampling_mode_raises(config):
    catalog = _open_catalog(config)
    try:
        with pytest.raises(PipelineError, match="采样模式"):
            Pipeline(
                config,
                catalog=catalog,
                embedder=HashTestEmbedder(),
                allow_test_embedder=True,
                sampling_mode="nope",
            )
    finally:
        catalog.close()


def test_full_mode_retry_is_deterministic(config, sample_video):
    """full 模式重新编码同一作品：向量与 clip_times 必须逐元素一致。"""

    _copy_video(config, sample_video, "det12.mp4")
    cfg = _full_config(config)
    catalog = _open_catalog(cfg)
    try:
        pipeline = _full_pipeline(cfg, catalog)
        register_local_videos(catalog, cfg)
        stats = pipeline.process_pending(source="local")
        assert stats.completed == 1
        work = catalog.list_works()[0]
        work_id = int(work["id"])
        version = pipeline.embedding_version
        before = load_vector(
            cfg.data_path(
                catalog.get_embedding(work_id, version=version, kind="clip_joint")["path"]
            )
        )
        times_before = load_vector(_work_dir(cfg, version, work_id) / "clip_times.npy")

        retried = pipeline.retry_works([work_id])
        assert retried.completed == 1, [o.message for o in retried.outcomes]
        after = load_vector(
            cfg.data_path(
                catalog.get_embedding(work_id, version=version, kind="clip_joint")["path"]
            )
        )
        times_after = load_vector(_work_dir(cfg, version, work_id) / "clip_times.npy")

        assert np.array_equal(before, after)
        assert np.array_equal(times_before, times_after)
        state = catalog.get_sequence_state(work_id, version=version)
        assert state["status"] == "assembled" and state["n_clips_done"] == 2
        pipeline.close()
    finally:
        catalog.close()


def test_cli_run_options_parse():
    cli = importlib.import_module("pipeline")
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "process-local",
            "--sampling",
            "full_coverage",
            "--clip-height",
            "336",
            "--allow-long",
        ]
    )
    assert args.sampling == "full_coverage"
    assert args.clip_height == 336
    assert args.allow_long is True
    default_args = parser.parse_args(["run-pending"])
    assert default_args.sampling is None
    assert default_args.clip_height is None
    assert default_args.allow_long is False

