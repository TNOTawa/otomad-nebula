"""独立端到端验收（E2E）：合成视频 -> 编目 -> 流水线 -> 向量 -> 星云 -> 界面。

设计原则（与实现方测试区分）：

- 使用 tests/fixtures/video_factory.py 现场合成的视频（不同时长/音高/画面/无音轨/坏文件），
  不依赖任何私有素材，也不复用实现方测试的辅助函数或断言逻辑；
- 使用临时 config + 临时 SQLite + 临时 data/temp 目录，不触碰仓库里的真实数据；
- 只注入 HashTestEmbedder（离线、确定性），通过公开接口驱动整条链路；
- 断言的是“可观察的验收行为”：作品状态、失败隔离、向量形状/范数、采样段数、
  重复运行不重编码、星云坐标、appdata 数据帧、Streamlit 页面可渲染，
  而不是实现细节。

默认离线运行（需要本机 ffmpeg/ffprobe），标记为 integration；qa.py 默认会运行它。
"""

from __future__ import annotations

import hashlib
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nebula.appdata import (
    apply_color,
    available_color_fields,
    external_url,
    hover_text,
    load_nebula_frame,
    search_frame,
)
from nebula.config import Config
from nebula.db import Catalog
from nebula.embed import HashTestEmbedder, load_vector
from nebula.errors import MapError
from nebula.ingest import register_local_videos
from nebula.mapplot import build_map
from nebula.pipeline import Pipeline
from tests.fixtures.video_factory import (
    build_video_library,
    create_temp_config,
    media_tools_available,
    probe_duration_s,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[1]

#: 参与 E2E 的文件：5 个应成功 + 2 个应失败（采样下限 / 无法探测）。
_INSTALL_LAYOUT: dict[str, str] = {
    "short_4s.mp4": "short_4s.mp4",
    "mid_12s.mp4": "mid_12s.mp4",
    "long_40s.mp4": "long_40s.mp4",
    "odd_17s.mp4": "series/ep02/odd_17s.mp4",  # 验证递归扫描与子目录相对路径
    "silent_6s.mp4": "silent_6s.mp4",
    "tiny_05s.mp4": "broken/tiny_05s.mp4",  # 0.5s < min_duration，应失败
    "corrupt.mp4": "broken/corrupt.mp4",  # 不是视频，应失败
}

_EXPECTED_OK = {
    "short_4s.mp4",
    "mid_12s.mp4",
    "long_40s.mp4",
    "odd_17s.mp4",
    "silent_6s.mp4",
}
_EXPECTED_FAIL = {"tiny_05s.mp4", "corrupt.mp4"}
_EXPECTED_FAILURES = {
    "tiny_05s.mp4": "SamplingError",
    "corrupt.mp4": "MediaError",
}

_VECTOR_KINDS = {
    "work_joint",
    "work_audio",
    "work_video",
    "clip_joint",
    "clip_audio",
    "clip_video",
}


@dataclass
class E2EState:
    """一次 E2E 运行共享的状态。"""

    root: Path
    cfg: Config
    catalog: Catalog
    pipeline: Pipeline
    summary: object
    stats: object
    version: str
    installed: dict[str, Path] = field(default_factory=dict)
    good_durations: dict[str, float] = field(default_factory=dict)


def _expected_clip_count(duration_s: float) -> int:
    """按方案文档的采样规则独立推导片段数：8 秒片段、最多 5 段、短视频 1 段。"""

    if duration_s <= 8.0 + 1e-9:
        return 1
    fits = int(math.floor(duration_s / 8.0 + 1e-9))
    return min(5, max(1, fits))


def _sha1_file(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _work_by_rel(state: E2EState, rel_path: str) -> dict:
    for work in state.catalog.list_works():
        if work["local_path"] == rel_path:
            return work
    raise AssertionError(f"数据库中没有登记作品: {rel_path}")


def _rel_path_of(state: E2EState, name: str) -> str:
    return state.installed[name].relative_to(state.cfg.paths.input_local).as_posix()


@pytest.fixture(scope="module")
def e2e(tmp_path_factory: pytest.TempPathFactory) -> E2EState:
    if not media_tools_available():
        pytest.skip("缺少 ffmpeg/ffprobe，跳过 E2E")

    root = tmp_path_factory.mktemp("e2e")
    cfg = create_temp_config(root)
    library = build_video_library(root / "video_cache")

    installed: dict[str, Path] = {}
    for name, rel_path in _INSTALL_LAYOUT.items():
        destination = cfg.paths.input_local / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(library[name], destination)
        installed[name] = destination

    catalog = Catalog(cfg.paths.db_path)
    catalog.initialize()
    pipeline = Pipeline(
        cfg,
        catalog=catalog,
        embedder=HashTestEmbedder(),
        allow_test_embedder=True,
    )
    try:
        summary = register_local_videos(catalog, cfg)
        version = pipeline.embedding_version
        stats = pipeline.process_pending(source="local", verify_sync=2)
        good_durations = {
            name: probe_duration_s(installed[name]) for name in _EXPECTED_OK
        }
        state = E2EState(
            root=root,
            cfg=cfg,
            catalog=catalog,
            pipeline=pipeline,
            summary=summary,
            stats=stats,
            version=version,
            installed=installed,
            good_durations=good_durations,
        )
        yield state
    finally:
        pipeline.close()
        catalog.close()


# ──────────────────────────────────────────────────────────────────────────────
# 1. 扫描与编目
# ──────────────────────────────────────────────────────────────────────────────


def test_ingest_registers_every_installed_file(e2e: E2EState) -> None:
    assert e2e.summary.files == len(_INSTALL_LAYOUT)
    assert e2e.summary.created == len(_INSTALL_LAYOUT)
    assert e2e.summary.existing == 0

    works = e2e.catalog.list_works()
    assert len(works) == len(_INSTALL_LAYOUT)
    assert {work["local_path"] for work in works} == set(_INSTALL_LAYOUT.values())
    assert all(work["source"] == "local" for work in works)
    # 本地作品去重键 = local:<相对 input_local 的 POSIX 路径>
    assert {work["work_key"] for work in works} == {
        "local:" + rel_path for rel_path in _INSTALL_LAYOUT.values()
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. 处理结果：成功作品全部 completed，失败作品隔离且可读
# ──────────────────────────────────────────────────────────────────────────────


def test_good_works_completed_and_bad_works_failed(e2e: E2EState) -> None:
    works = e2e.catalog.list_works()
    completed = {work["local_path"]: work for work in works if work["status"] == "completed"}
    failed = {work["local_path"]: work for work in works if work["status"] == "failed"}

    expected_completed = {_rel_path_of(e2e, name) for name in _EXPECTED_OK}
    expected_failed = {_rel_path_of(e2e, name) for name in _EXPECTED_FAIL}
    assert set(completed) == expected_completed, "成功作品集合与预期不一致"
    assert set(failed) == expected_failed, "失败作品集合与预期不一致"
    assert e2e.stats.completed == len(_EXPECTED_OK)
    assert e2e.stats.failed == len(_EXPECTED_FAIL)
    assert e2e.stats.skipped == 0

    for work in works:
        assert work["attempts"] == 1, f"每部作品应只尝试一次: {work['local_path']}"
    for rel_path, work in completed.items():
        assert work["error"] is None, f"成功作品不应有错误信息: {rel_path}"

    # 元数据由 ffprobe 写入；无音轨视频要正确记录 has_audio=0，且不阻塞编码
    for name in _EXPECTED_OK:
        work = completed[_rel_path_of(e2e, name)]
        duration = float(work["duration_s"])
        assert duration == pytest.approx(e2e.good_durations[name], abs=0.2)
        expected_audio = 0 if name == "silent_6s.mp4" else 1
        assert int(work["has_audio"]) == expected_audio
        assert int(work["has_video"]) == 1


def test_failure_messages_are_readable_and_isolated(e2e: E2EState) -> None:
    completed_before = {
        work["local_path"]
        for work in e2e.catalog.list_works()
        if work["status"] == "completed"
    }
    for name, expected_type in _EXPECTED_FAILURES.items():
        work = _work_by_rel(e2e, _rel_path_of(e2e, name))
        assert work["status"] == "failed", f"{name} 应记录为 failed"
        message = str(work["error"] or "")
        assert message.strip(), f"{name} 的失败原因不能为空"
        assert expected_type in message, f"{name} 的失败类型应可读: {message}"
        # 失败作品不能阻塞其他作品
        assert completed_before == {
            _rel_path_of(e2e, item) for item in _EXPECTED_OK
        }
    events = e2e.catalog.get_events(_work_by_rel(e2e, _rel_path_of(e2e, "corrupt.mp4"))["id"])
    assert any("failed" in str(event["detail"]) for event in events), "失败状态应写入事件日志"


# ──────────────────────────────────────────────────────────────────────────────
# 3. 向量文件与采样规则
# ──────────────────────────────────────────────────────────────────────────────


def test_vectors_match_shape_and_sampling_rule(e2e: E2EState) -> None:
    dims: set[int] = set()
    completed_ids = {
        work["id"]
        for work in e2e.catalog.list_works()
        if work["status"] == "completed"
    }
    all_rows = [
        row
        for row in e2e.catalog.list_embeddings(e2e.version)
        if row["work_id"] in completed_ids
    ]
    assert all_rows, "应登记向量"
    assert len(all_rows) == len(_EXPECTED_OK) * len(_VECTOR_KINDS)

    for name in _EXPECTED_OK:
        work = _work_by_rel(e2e, _rel_path_of(e2e, name))
        rows = [row for row in all_rows if row["work_id"] == work["id"]]
        kinds = {row["kind"]: row for row in rows}
        assert set(kinds) == _VECTOR_KINDS

        expected_clips = _expected_clip_count(
            float(work["duration_s"]) if work["duration_s"] else e2e.good_durations[name]
        )
        # 片段计划落库且符合采样规则（不越界、不重叠、索引连续）
        clips = e2e.catalog.get_clips(work["id"], version=e2e.version)
        assert len(clips) == expected_clips, f"{name} 片段数应为 {expected_clips}"
        assert [clip["idx"] for clip in clips] == list(range(expected_clips))
        previous_end = 0.0
        for clip in clips:
            start = float(clip["start_s"])
            duration = float(clip["duration_s"])
            assert start >= 0.0 and duration > 0.0
            assert start + duration <= float(work["duration_s"]) + 0.01
            assert start >= previous_end - 1e-6
            previous_end = start + duration

        for row in kinds.values():
            rel = str(row["path"])
            assert not Path(rel).is_absolute(), f"向量路径必须是相对 data 的路径: {rel}"
            assert "\\" not in rel, f"向量路径应使用 POSIX 分隔符: {rel}"
            vector_path = e2e.cfg.data_path(rel)
            assert vector_path.is_file(), f"登记的向量文件不存在: {vector_path}"
            assert vector_path.stat().st_size > 0

        joint = load_vector(e2e.cfg.data_path(kinds["work_joint"]["path"]))
        assert joint.dtype == np.float32
        assert joint.ndim == 1
        dim = int(joint.shape[0])
        assert int(kinds["work_joint"]["dim"]) == dim
        assert np.linalg.norm(joint) == pytest.approx(1.0, abs=1e-4), f"{name} 作品向量未归一化"
        dims.add(dim)

        clip_joint = load_vector(e2e.cfg.data_path(kinds["clip_joint"]["path"]))
        assert clip_joint.shape == (expected_clips, dim)
        assert int(kinds["clip_joint"]["n_clips"]) == expected_clips
        norms = np.linalg.norm(clip_joint, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), f"{name} 片段向量未逐行归一化"
        for kind in ("clip_audio", "clip_video"):
            array = load_vector(e2e.cfg.data_path(kinds[kind]["path"]))
            assert array.shape == (expected_clips, dim)
        for kind in ("work_audio", "work_video"):
            assert load_vector(e2e.cfg.data_path(kinds[kind]["path"])).shape == (dim,)

    assert len(dims) == 1, f"同一向量版本内维度必须一致，实际 {dims}"


# ──────────────────────────────────────────────────────────────────────────────
# 4. 重复运行：跳过已完成作品，不重新编码
# ──────────────────────────────────────────────────────────────────────────────


def _vector_snapshot(state: E2EState) -> dict[str, tuple[int, str]]:
    snapshot: dict[str, tuple[int, str]] = {}
    for path in sorted(state.cfg.paths.embeddings.rglob("*.npy")):
        rel = path.relative_to(state.cfg.paths.embeddings).as_posix()
        snapshot[rel] = (path.stat().st_mtime_ns, _sha1_file(path))
    return snapshot


def test_rerun_skips_completed_works_without_reencoding(e2e: E2EState) -> None:
    statuses_before = {
        work["id"]: (work["status"], int(work["attempts"])) for work in e2e.catalog.list_works()
    }
    vectors_before = _vector_snapshot(e2e)

    stats = e2e.pipeline.process_pending(source="local", retry_failed=False)
    assert stats.selected == 0, "重复运行不应再选中任何已完成作品"
    assert stats.outcomes == []
    assert stats.completed == 0 and stats.failed == 0 and stats.skipped == 0

    assert _vector_snapshot(e2e) == vectors_before, "重复运行不应重新写入向量文件"
    statuses_after = {
        work["id"]: (work["status"], int(work["attempts"])) for work in e2e.catalog.list_works()
    }
    assert statuses_after == statuses_before, "重复运行不应改变状态与尝试次数"


# ──────────────────────────────────────────────────────────────────────────────
# 5. 星云生成与 appdata 数据准备
# ──────────────────────────────────────────────────────────────────────────────


def test_map_build_produces_versioned_coordinates(e2e: E2EState) -> None:
    result = build_map(e2e.catalog, e2e.cfg, version=e2e.version)
    assert result.version == "map_v001"
    assert result.embedding_version == e2e.version
    assert result.n_works == len(_EXPECTED_OK)

    for path in (result.coords_path, result.model_2d_path, result.model_3d_path):
        assert path.is_file(), f"星云产物缺失: {path}"
        assert not Path(e2e.cfg.data_relpath(path)).is_absolute()
    assert (result.coords_path.parent / "meta.json").is_file()

    coords = pd.read_parquet(result.coords_path)
    assert len(coords) == len(_EXPECTED_OK)
    for column in ("work_id", "x2", "y2", "x3", "y3", "z3"):
        assert column in coords.columns, f"坐标文件缺少列: {column}"
    values = coords[["x2", "y2", "x3", "y3", "z3"]].to_numpy(dtype=float)
    assert np.isfinite(values).all(), "星云坐标必须是有限数值"
    assert len(set(coords["work_id"].tolist())) == len(_EXPECTED_OK)

    run = e2e.catalog.get_map(result.version)
    assert run is not None and run["status"] == "ready"
    assert int(run["n_works"]) == len(_EXPECTED_OK)
    assert e2e.catalog.latest_map()["version"] == result.version
    assert e2e.catalog.next_map_version() == "map_v002", "星云版本必须单调递增"
    assert len(e2e.catalog.load_map_rows(result.version)) == len(_EXPECTED_OK)


def test_appdata_frame_has_real_coordinates(e2e: E2EState) -> None:
    map_info, frame = load_nebula_frame(e2e.catalog, e2e.cfg)
    assert map_info["version"] == "map_v001"
    assert len(frame) == len(_EXPECTED_OK)
    required_columns = {
        "work_id",
        "work_key",
        "source",
        "file_name",
        "status",
        "duration_s",
        "x2",
        "y2",
        "x3",
        "y3",
        "z3",
        "x",
        "y",
        "z",
        "rank",
        "votes",
        "vote_year",
        "vote_years",
        "list_names",
        "search_text",
        "hover_text",
    }
    assert required_columns <= set(frame.columns)
    assert np.isfinite(frame[["x2", "y2", "x3", "y3", "z3"]].to_numpy(dtype=float)).all()
    completed_ids = {
        work["id"]
        for work in e2e.catalog.list_works()
        if work["status"] == "completed"
    }
    assert set(frame["work_id"].tolist()) == completed_ids
    assert frame["status"].eq("completed").all()

    # 搜索：按文件名子串精确过滤；空查询返回全部
    short_work = _work_by_rel(e2e, _rel_path_of(e2e, "short_4s.mp4"))
    found = search_frame(frame, "SHORT_4S")
    assert short_work["id"] in set(found["work_id"].tolist())
    assert len(found) == 1, "按唯一文件名搜索应只命中一个作品"
    assert len(search_frame(frame, "")) == len(frame)

    # hover 文本与外部链接
    texts = hover_text(frame)
    assert len(texts) == len(frame)
    assert all(isinstance(text, str) and text.strip() for text in texts)
    local_row = frame.iloc[0]
    assert external_url(local_row) is None, "本地作品没有 Bilibili 链接"

    # 染色字段：时长可连续染色；不染色时按空间密度
    field_ids = [field_id for field_id, _label in available_color_fields(frame)]
    assert "duration_s" in field_ids
    duration_spec = apply_color(frame, "duration_s")
    assert len(duration_spec.values) == len(frame)
    assert all(value is not None for value in duration_spec.values)
    density_spec = apply_color(frame, "none")
    assert len(density_spec.values) == len(frame)

    # 无星云时必须给出中文 MapError，而不是空 DataFrame
    empty_catalog = Catalog(e2e.root / "empty" / "catalog.sqlite")
    empty_catalog.initialize()
    try:
        with pytest.raises(MapError):
            load_nebula_frame(empty_catalog, e2e.cfg)
    finally:
        empty_catalog.close()


# ──────────────────────────────────────────────────────────────────────────────
# 6. Streamlit 页面渲染 + 临时目录清理
# ──────────────────────────────────────────────────────────────────────────────


def test_streamlit_app_renders_real_nebula(e2e: E2EState, monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("NEBULA_CONFIG", str(e2e.cfg.path))
    app_test = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=120)
    app_test.run()

    assert len(app_test.exception) == 0, f"页面出现异常: {list(app_test.exception)}"
    assert len(app_test.error) == 0, f"页面显示错误: {list(app_test.error)}"
    charts = app_test.get("plotly_chart")
    assert len(charts) == 1, "主图应渲染且只有一张星云图"
    spec = charts[0].proto.spec
    assert "scatter" in str(spec).lower(), "主图应是 Plotly 散点图"


def test_temp_cleanup_and_data_artifacts(e2e: E2EState) -> None:
    for name in _EXPECTED_OK:
        work = _work_by_rel(e2e, _rel_path_of(e2e, name))
        temp_dir = e2e.cfg.paths.temp / f"work_{int(work['id']):06d}"
        assert not temp_dir.exists(), f"成功作品的临时目录未清理: {temp_dir}"

    status = e2e.pipeline.status()
    assert status["status"]["completed"] == len(_EXPECTED_OK)
    assert status["status"]["failed"] == len(_EXPECTED_FAIL)
    assert status["temp_dirs"] == 0
    assert e2e.pipeline.clean_temp() == 0, "成功路径不应留下临时文件"

    # 数据目录内不应出现写到一半的临时文件
    for pattern in ("*.partial", "*.tmp"):
        assert not list(e2e.cfg.paths.data.rglob(pattern))
