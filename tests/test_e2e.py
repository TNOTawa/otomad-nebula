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
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nebula.appdata import (
    apply_color,
    available_color_fields,
    clip_heatmap_data,
    clip_pair_rows,
    external_url,
    filter_frame,
    hover_text,
    hub_rows,
    load_nebula_frame,
    neighbor_rows,
    search_frame,
)
from nebula.config import Config
from nebula.db import Catalog
from nebula.download import BilibiliMeta
from nebula.embed import HashTestEmbedder, load_vector, save_vector
from nebula.errors import IngestError, MapError, NebulaError
from nebula.identity import (
    candidates_from_manual_links,
    find_candidates,
    link_candidates,
    load_manual_links,
    normalize_title,
)
from nebula.ingest import register_local_videos
from nebula.mapplot import build_map, load_embedding_matrix
from nebula.pipeline import Pipeline
from nebula.similarity import load_similarity_index
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

# ══════════════════════════════════════════════════════════════════════════════
# 7. v0.2 独立验收：临时 DB + 合成向量（不依赖 ffmpeg/模型）
#    覆盖 A1 高维近邻、A2 近邻密度/筛选星云、B1 身份关联与去重。
#    断言由本文件独立实现（直接 np.load 读向量、手算余弦），不复用实现方测试的
#    断言逻辑；所有数据在临时目录，不触碰仓库真实数据。
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class VectorEnv:
    """A1/A2 独立验收所用的合成向量环境。"""

    root: Path
    cfg: Config
    catalog: Catalog
    version: str
    work_ids: dict[str, int]


def _unit64(values) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    assert float(np.linalg.norm(vector)) > 0
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def _direct_work_vectors(env: VectorEnv, kind: str = "work_joint") -> dict[int, np.ndarray]:
    """独立读取一个模态的全部作品向量（直接 np.load，不经过实现方 API）。"""

    vectors: dict[int, np.ndarray] = {}
    for row in env.catalog.list_embeddings(env.version, kind=kind):
        array = np.asarray(
            np.load(env.cfg.data_path(str(row["path"])), allow_pickle=False), dtype=np.float64
        )
        if array.ndim == 2 and array.shape[0] == 1:
            array = array.reshape(-1)
        assert array.ndim == 1, f"{kind} 作品向量形状异常: {array.shape}"
        vectors[int(row["work_id"])] = array / np.linalg.norm(array)
    return vectors


def _expected_ranked(vectors: dict[int, np.ndarray], work_id: int) -> list[tuple[int, float]]:
    """手算余弦排序：相似度降序、work_id 升序（并列时稳定），不含自身。"""

    pairs = [
        (other_id, float(np.dot(vectors[work_id], vector)))
        for other_id, vector in vectors.items()
        if other_id != work_id
    ]
    pairs.sort(key=lambda pair: (-pair[1], pair[0]))
    return pairs


def _expected_kth_distances(
    vectors: dict[int, np.ndarray], k: int
) -> dict[int, float]:
    distances: dict[int, float] = {}
    for work_id in vectors:
        ranked = _expected_ranked(vectors, work_id)
        take = min(int(k), len(ranked))
        distances[work_id] = 1.0 - ranked[take - 1][1]
    return distances


@pytest.fixture(scope="module")
def vector_env(tmp_path_factory: pytest.TempPathFactory) -> VectorEnv:
    root = tmp_path_factory.mktemp("accept-vectors")
    cfg = create_temp_config(root)
    catalog = Catalog(cfg.paths.db_path)
    catalog.initialize()
    version = "v-accept"
    catalog.register_embedding_version(
        version,
        model_checkpoint="synthetic",
        model_revision=None,
        frames=16,
        clip_seconds=8.0,
        clips_per_work=5,
        audio_sample_rate=48000,
        params={"test": "e2e-acceptance"},
    )
    lists = {
        "votes_2025": catalog.ensure_list(
            "votes_2025", name="2025年音MAD选票", kind="votes", vote_year=2025
        ),
        "votes_2026": catalog.ensure_list(
            "votes_2026", name="2026年音MAD选票", kind="votes", vote_year=2026
        ),
    }
    specs = [
        {
            "name": "w1",
            "source": "bilibili",
            "bvid": "BV1xx411c7m1",
            "title": "合成作品一",
            "duration_s": 100.0,
            "joint": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "audio": [1.0, 0.0, 0.0, 0.0],
            "video": [1.0, 0.0, 0.0, 0.0],
            "clip_joint": [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            "clip_audio": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]],
            "lists": [("votes_2025", 1, 500)],
        },
        {
            "name": "w2",
            "source": "bilibili",
            "bvid": "BV1xx411c7m2",
            "title": "合成作品二",
            "duration_s": 120.0,
            "joint": [0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "audio": [0.8, 0.6, 0.0, 0.0],
            "video": [0.6, 0.8, 0.0, 0.0],
            "clip_joint": [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.9, 0.435889894, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            "clip_audio": [
                [0.0, 1.0, 0.0, 0.0],
                [0.9, 0.435889894, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            "lists": [("votes_2025", 2, 400)],
        },
        {
            "name": "w3",
            "source": "bilibili",
            "bvid": "BV1xx411c7m3",
            "title": "合成作品三",
            "duration_s": 80.0,
            "joint": [0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "audio": [0.0, 1.0, 0.0, 0.0],
            "video": None,
            "clip_joint": [
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            "clip_audio": [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            "lists": [("votes_2025", 3, 300)],
        },
        {
            "name": "w4",
            "source": "bilibili",
            "bvid": "BV1xx411c7m4",
            "title": "合成作品四",
            "duration_s": 60.0,
            "joint": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "audio": None,
            "video": [1.0, 0.0, 0.0, 0.0],
            "lists": [("votes_2026", 1, 100)],
        },
        {
            "name": "w5",
            "source": "local",
            "local_path": "accept/local_a.mp4",
            "title": "本地作品五",
            "duration_s": 90.0,
            "joint": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "audio": None,
            "video": None,
            "lists": [("votes_2025", 4, 200)],
        },
        {
            "name": "w6",
            "source": "local",
            "local_path": "accept/local_b.mp4",
            "title": "本地作品六",
            "duration_s": 45.0,
            "joint": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "audio": None,
            "video": None,
            "lists": [("votes_2026", 2, 50)],
        },
    ]
    work_ids: dict[str, int] = {}
    try:
        for spec in specs:
            if spec["source"] == "local":
                work_id, _created = catalog.upsert_local_work(
                    str(spec["local_path"]), title=str(spec["title"])
                )
            else:
                work_id, _created = catalog.upsert_bilibili_work(
                    str(spec["bvid"]), title=str(spec["title"])
                )
            catalog.begin_attempt(work_id, "encoding")
            catalog.update_work_metadata(work_id, duration_s=float(spec["duration_s"]))
            work_dir = cfg.paths.embeddings / version / f"{work_id:06d}"
            for kind, values in (
                ("work_joint", spec["joint"]),
                ("work_audio", spec["audio"]),
                ("work_video", spec["video"]),
            ):
                if values is None:
                    continue
                array = _unit64(values)
                save_vector(work_dir / f"{kind}.npy", array)
                catalog.register_embedding(
                    work_id,
                    version=version,
                    kind=kind,
                    path=cfg.data_relpath(work_dir / f"{kind}.npy"),
                    dim=int(array.shape[0]),
                )
            for clip_kind in ("clip_joint", "clip_audio"):
                clip_rows = spec.get(clip_kind)
                if clip_rows is None:
                    continue
                clip_matrix = np.stack([_unit64(row) for row in clip_rows]).astype(np.float32)
                save_vector(work_dir / f"{clip_kind}.npy", clip_matrix)
                catalog.register_embedding(
                    work_id,
                    version=version,
                    kind=clip_kind,
                    path=cfg.data_relpath(work_dir / f"{clip_kind}.npy"),
                    dim=int(clip_matrix.shape[1]),
                    n_clips=int(clip_matrix.shape[0]),
                )
                catalog.replace_clips(
                    work_id,
                    version=version,
                    clips=[(index, index * 8.0, 8.0) for index in range(clip_matrix.shape[0])],
                )
            catalog.complete(work_id)
            for list_key, rank, votes in spec["lists"]:
                catalog.add_appearance(work_id, lists[list_key], rank=int(rank), votes=int(votes))
            work_ids[str(spec["name"])] = int(work_id)
        yield VectorEnv(root=root, cfg=cfg, catalog=catalog, version=version, work_ids=work_ids)
    finally:
        catalog.close()


# ---------------------------------------------------------------- A1 近邻

def test_a1_neighbors_match_hand_computed_cosine(vector_env: VectorEnv) -> None:
    """A1：Top-k 近邻顺序/相似度必须与手算高维余弦一致，不含自身。"""

    index = load_similarity_index(vector_env.catalog, vector_env.cfg, version=vector_env.version)
    try:
        assert list(index.work_ids) == sorted(vector_env.work_ids.values())
        vectors = _direct_work_vectors(vector_env)
        for work_id in index.work_ids:
            expected = _expected_ranked(vectors, work_id)[:3]
            neighbors = index.neighbors(work_id, k=3)
            assert [neighbor.work_id for neighbor in neighbors] == [
                pair[0] for pair in expected
            ]
            assert [neighbor.rank for neighbor in neighbors] == [1, 2, 3]
            for neighbor, (expected_id, expected_sim) in zip(neighbors, expected, strict=True):
                assert neighbor.work_id == expected_id
                assert neighbor.similarity == pytest.approx(expected_sim, abs=1e-6)
                assert neighbor.work_id != work_id
            # k 超过可用作品数时截断，而不是报错
            assert len(index.neighbors(work_id, k=100)) == len(index.work_ids) - 1
    finally:
        index.close()


def test_a1_similarity_and_kth_distance_match_hand_computed(vector_env: VectorEnv) -> None:
    """A1：similarity() 与 kth_neighbor_distance() 必须等于手算余弦值。"""

    index = load_similarity_index(vector_env.catalog, vector_env.cfg, version=vector_env.version)
    try:
        vectors = _direct_work_vectors(vector_env)
        ids = list(vectors)
        for first in ids:
            for second in ids:
                expected = 1.0 if first == second else float(
                    np.dot(vectors[first], vectors[second])
                )
                assert index.similarity(first, second) == pytest.approx(expected, abs=1e-6)
        expected_kth = _expected_kth_distances(vectors, 3)
        actual_kth = index.kth_neighbor_distance(k=3)
        assert set(actual_kth) == set(expected_kth)
        for work_id, expected in expected_kth.items():
            assert actual_kth[work_id] == pytest.approx(expected, abs=1e-6)
    finally:
        index.close()


def test_a1_modalities_cache_and_close(vector_env: VectorEnv) -> None:
    """A1：三模态可用性、缺失模态报中文错误、矩阵缓存内容正确、close 后拒绝查询。"""

    index = load_similarity_index(vector_env.catalog, vector_env.cfg, version=vector_env.version)
    try:
        assert index.available_kinds == ("work_joint", "work_audio", "work_video")
        vectors = _direct_work_vectors(vector_env)
        matrix = index.matrix("work_joint")
        assert matrix.dtype == np.float32
        assert matrix.shape == (len(vectors), 8)
        for row_index, work_id in enumerate(index.work_ids):
            assert matrix[row_index] == pytest.approx(vectors[work_id], abs=1e-6)
            assert float(np.linalg.norm(matrix[row_index])) == pytest.approx(1.0, abs=1e-6)
        # w6 没有音频向量
        with pytest.raises(NebulaError):
            index.neighbors(vector_env.work_ids["w6"], kind="work_audio")
        # 未知模态
        with pytest.raises(NebulaError):
            index.matrix("work_unknown")
    finally:
        index.close()
    with pytest.raises(NebulaError):
        index.neighbors(vector_env.work_ids["w1"], k=1)


# ---------------------------------------------------------------- A2 数据准备/密度

def test_a2_frame_filters_and_neighbor_rows(vector_env: VectorEnv) -> None:
    """A2：load_nebula_frame 的 v0.2 列、filter_frame 语义、neighbor_rows 展示行。"""

    result = build_map(
        vector_env.catalog, vector_env.cfg, version=vector_env.version, note="acceptance-full"
    )
    assert result.version == "map_v001"
    assert result.n_works == len(vector_env.work_ids)
    _map_info, frame = load_nebula_frame(vector_env.catalog, vector_env.cfg)

    assert len(frame) == len(vector_env.work_ids)
    for column in (
        "work_id",
        "title",
        "list_keys",
        "list_names",
        "vote_years",
        "rank",
        "votes",
        "is_canonical",
        "search_text",
        "hover_text",
    ):
        assert column in frame.columns, f"缺少 v0.2 展示列: {column}"
    assert frame["is_canonical"].all()

    ids = vector_env.work_ids
    bilibili_ids = {ids["w1"], ids["w2"], ids["w3"], ids["w4"]}
    local_ids = {ids["w5"], ids["w6"]}
    assert set(filter_frame(frame, sources=["bilibili"])["work_id"]) == bilibili_ids
    assert set(filter_frame(frame, sources=["local"])["work_id"]) == local_ids
    assert set(filter_frame(frame, vote_years=[2025])["work_id"]) == {
        ids["w1"],
        ids["w2"],
        ids["w3"],
        ids["w5"],
    }
    assert set(filter_frame(frame, list_keys=["votes_2026"])["work_id"]) == {
        ids["w4"],
        ids["w6"],
    }
    assert set(
        filter_frame(frame, sources=["local"], vote_years=[2026])["work_id"]
    ) == {ids["w6"]}
    assert len(filter_frame(frame)) == len(frame)
    # 空选择 = 不过滤（契约语义）
    assert len(filter_frame(frame, sources=[], list_keys=[], vote_years=[])) == len(frame)

    index = load_similarity_index(vector_env.catalog, vector_env.cfg, version=vector_env.version)
    try:
        rows = neighbor_rows(frame, index, ids["w1"], kind="work_joint", k=3)
        vectors = _direct_work_vectors(vector_env)
        expected = _expected_ranked(vectors, ids["w1"])[:3]
        assert [row["work_id"] for row in rows] == [pair[0] for pair in expected]
        assert [row["rank"] for row in rows] == [1, 2, 3]
        for row, (expected_id, expected_sim) in zip(rows, expected, strict=True):
            assert row["work_id"] == expected_id
            assert row["similarity"] == pytest.approx(expected_sim, abs=1e-6)
            assert row["in_map"] is True
            assert row["title"]
        for kind in index.available_kinds:
            assert neighbor_rows(frame, index, ids["w1"], kind=kind, k=2)
    finally:
        index.close()


def test_a2_density_knn_matches_hand_computed(vector_env: VectorEnv) -> None:
    """A2：高维近邻密度 = 1 - 第 5 近邻高维余弦相似度（不是平面距离）。"""

    _map_info, frame = load_nebula_frame(vector_env.catalog, vector_env.cfg)
    index = load_similarity_index(vector_env.catalog, vector_env.cfg, version=vector_env.version)
    try:
        spec = apply_color(frame, "density_knn", similarity_index=index)
        assert len(spec.values) == len(frame)
        assert spec.categorical is False
        vectors = _direct_work_vectors(vector_env)
        expected = _expected_kth_distances(vectors, 5)
        for work_id, value in zip(frame["work_id"].tolist(), spec.values, strict=True):
            assert value is not None
            assert float(value) == pytest.approx(float(expected[int(work_id)]), abs=1e-5)
        with pytest.raises(MapError):
            apply_color(frame, "density_knn")
    finally:
        index.close()

    with_similarity = {field_id for field_id, _label in available_color_fields(frame, has_similarity=True)}
    without_similarity = {
        field_id for field_id, _label in available_color_fields(frame, has_similarity=False)
    }
    assert "density_knn" in with_similarity
    assert "density_knn" not in without_similarity

# ---------------------------------------------------------------- P1 近邻解释

def _hand_topk_sets(vectors: dict[int, np.ndarray], k: int) -> dict[int, set[int]]:
    """手算每部作品的 Top-k work_id 集合（相似度降序、work_id 升序并列）。"""

    return {
        work_id: {neighbor_id for neighbor_id, _sim in _expected_ranked(vectors, work_id)[:k]}
        for work_id in vectors
    }


def test_p1_pair_percentile_and_mutual_hand_computed(vector_env: VectorEnv) -> None:
    """P1：全库分位与互为 Top-k 标记必须与手算全库成对相似度分布一致。"""

    index = load_similarity_index(vector_env.catalog, vector_env.cfg, version=vector_env.version)
    try:
        vectors = _direct_work_vectors(vector_env)
        ids = sorted(vectors)
        pair_sims = sorted(
            float(np.dot(vectors[first], vectors[second]))
            for position, first in enumerate(ids)
            for second in ids[position + 1 :]
        )
        assert pair_sims

        def hand_percentile_range(similarity: float) -> tuple[float, float]:
            """分位定义用严格小于；合成向量存在精确并列，允许 ±1e-6 并列边界区间。"""

            epsilon = 1e-6
            lower = sum(1 for value in pair_sims if value < similarity - epsilon)
            upper = sum(1 for value in pair_sims if value <= similarity + epsilon)
            return (
                100.0 * lower / len(pair_sims),
                100.0 * upper / len(pair_sims),
            )

        for work_id in ids:
            ranked = _expected_ranked(vectors, work_id)
            neighbors = index.neighbors(work_id, k=3)
            assert [neighbor.work_id for neighbor in neighbors] == [
                pair[0] for pair in ranked[:3]
            ]
            for neighbor in neighbors:
                assert 0.0 <= neighbor.pair_percentile <= 100.0
                lower, upper = hand_percentile_range(neighbor.similarity)
                assert lower - 1e-6 <= neighbor.pair_percentile <= upper + 1e-6, (
                    neighbor,
                    lower,
                    upper,
                )
                target_topk = {
                    item[0] for item in _expected_ranked(vectors, neighbor.work_id)[:3]
                }
                assert neighbor.is_mutual == (work_id in target_topk)
                # 独立方法（pair_percentile / similarity）与近邻行字段一致
                assert index.pair_percentile(work_id, neighbor.work_id) == pytest.approx(
                    neighbor.pair_percentile, abs=1e-6
                )
                assert index.similarity(work_id, neighbor.work_id) == pytest.approx(
                    neighbor.similarity, abs=1e-6
                )
            # 分位随相似度单调不减，且相似度按 rank 降序
            similarities = [neighbor.similarity for neighbor in neighbors]
            percentiles = [neighbor.pair_percentile for neighbor in neighbors]
            assert similarities == sorted(similarities, reverse=True)
            assert percentiles == sorted(percentiles, reverse=True)
        # 成对分位对称
        for first in ids:
            for second in ids:
                if first == second:
                    continue
                assert index.pair_percentile(first, second) == pytest.approx(
                    index.pair_percentile(second, first), abs=1e-9
                )
    finally:
        index.close()


def test_p1_hubness_matches_hand_computed_in_degree(vector_env: VectorEnv) -> None:
    """P1：hubness 应等于“被多少部作品列入 Top-k”的手算入度。"""

    index = load_similarity_index(vector_env.catalog, vector_env.cfg, version=vector_env.version)
    try:
        vectors = _direct_work_vectors(vector_env)
        ids = sorted(vectors)
        k = 3
        hand_topk = _hand_topk_sets(vectors, k)
        expected_counts = {
            work_id: sum(1 for targets in hand_topk.values() if work_id in targets)
            for work_id in ids
        }
        actual = index.hubness(k=k)
        assert actual == expected_counts
        assert sum(actual.values()) == len(ids) * min(k, len(ids) - 1)
        assert all(count >= 0 for count in actual.values())

        expected_top = sorted(expected_counts.items(), key=lambda item: (-item[1], item[0]))[:3]
        assert index.hub_top(k=k, n=3) == [(int(work_id), int(count)) for work_id, count in expected_top]

        frame = pd.DataFrame(
            [{"work_id": work_id, "title": f"作品{work_id}"} for work_id in ids]
        )
        rows = hub_rows(frame, index, kind="work_joint", k=k, n=3)
        assert [row["work_id"] for row in rows] == [item[0] for item in expected_top]
        assert [row["count"] for row in rows] == [item[1] for item in expected_top]
        assert all(row["in_map"] is True for row in rows)
        assert [row["rank"] for row in rows] == [1, 2, 3]
    finally:
        index.close()


# ---------------------------------------------------------------- P2 片段级相似探索

def _direct_clip_rows(env: VectorEnv, work_id: int, kind: str) -> np.ndarray:
    row = env.catalog.get_embedding(work_id, version=env.version, kind=kind)
    assert row is not None, f"work {work_id} 缺少 {kind} 登记"
    array = np.asarray(
        np.load(env.cfg.data_path(str(row["path"])), allow_pickle=False), dtype=np.float64
    )
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / norms


def test_p2_clip_similarity_heatmap_and_pairing_schema(vector_env: VectorEnv) -> None:
    """P2：片段热图/最高配对/解释字段与手算片段余弦一致，schema 稳定。"""

    index = load_similarity_index(vector_env.catalog, vector_env.cfg, version=vector_env.version)
    try:
        a = vector_env.work_ids["w1"]
        b = vector_env.work_ids["w2"]
        for clip_kind in ("clip_joint", "clip_audio"):
            clips_a = _direct_clip_rows(vector_env, a, clip_kind)
            clips_b = _direct_clip_rows(vector_env, b, clip_kind)
            manual = np.clip(clips_a @ clips_b.T, -1.0, 1.0)
            actual = index.clip_similarity(a, b, kind=clip_kind)
            assert actual.shape == manual.shape
            assert np.allclose(actual, manual, atol=1e-6)

        kind = "clip_joint"
        clips_a = _direct_clip_rows(vector_env, a, kind)
        clips_b = _direct_clip_rows(vector_env, b, kind)
        manual = np.clip(clips_a @ clips_b.T, -1.0, 1.0)
        rows = manual.shape[0]
        columns = manual.shape[1]
        flattened = [
            (float(manual[row, column]), row * columns + column)
            for row in range(rows)
            for column in range(columns)
        ]
        flattened.sort(key=lambda item: (-item[0], item[1]))
        expected_pairs = [
            (int(flat // columns), int(flat % columns), similarity)
            for similarity, flat in flattened[:3]
        ]
        best = index.best_clip_pairs(a, b, kind=kind, k=3)
        assert [(item["a_idx"], item["b_idx"]) for item in best] == [
            (pair[0], pair[1]) for pair in expected_pairs
        ]
        for item, expected in zip(best, expected_pairs, strict=True):
            assert item["similarity"] == pytest.approx(expected[2], abs=1e-6)

        frame = pd.DataFrame(
            [
                {"work_id": a, "title": "作品A"},
                {"work_id": b, "title": "作品B"},
            ]
        )
        heatmap = clip_heatmap_data(index, frame, a, b, kind=kind)
        assert heatmap["kind"] == kind
        assert (heatmap["a_work_id"], heatmap["b_work_id"]) == (a, b)
        assert heatmap["n_a"] == rows and heatmap["n_b"] == columns
        assert np.allclose(np.asarray(heatmap["matrix"]), manual, atol=1e-6)
        assert len(heatmap["row_labels"]) == rows
        assert len(heatmap["col_labels"]) == columns
        # 片段标签包含起始秒（w1/w2 的 clips 为 idx*8.0 秒）
        assert any("8.0s" in label for label in heatmap["row_labels"]), heatmap["row_labels"]

        pair_rows = clip_pair_rows(index, frame, a, b, kind=kind, k=3)
        assert [row["rank"] for row in pair_rows] == [1, 2, 3]
        for row, expected in zip(pair_rows, expected_pairs, strict=True):
            assert (row["a_idx"], row["b_idx"]) == (expected[0], expected[1])
            assert row["a_start_s"] == pytest.approx(expected[0] * 8.0)
            assert row["b_start_s"] == pytest.approx(expected[1] * 8.0)
            assert row["similarity"] == pytest.approx(expected[2], abs=1e-6)
            assert row["a_title"] and row["b_title"]

        explanation = index.explain_pair(a, b, kind=kind, threshold=0.5)
        assert explanation["n_a"] == rows and explanation["n_b"] == columns
        assert explanation["max"] == pytest.approx(float(np.max(manual)), abs=1e-6)
        assert explanation["mean"] == pytest.approx(float(np.mean(manual)), abs=1e-6)
        assert explanation["count_above_threshold"] == int(np.count_nonzero(manual >= 0.5))
        assert explanation["threshold"] == pytest.approx(0.5)

        # 片段级接口的边界：无片段向量的作品应报中文 NebulaError；阈值非法同样拒绝
        with pytest.raises(NebulaError):
            index.clip_similarity(a, vector_env.work_ids["w4"], kind=kind)
        with pytest.raises(NebulaError):
            index.explain_pair(a, b, kind=kind, threshold=2.0)
        assert "clip_joint" in index.available_clip_kinds
        assert index.has_clip_vector(a, "clip_joint") is True
    finally:
        index.close()



# ---------------------------------------------------------------- A2 筛选星云

def test_a2_filtered_maps_persist_filter_and_incomparable_coords(
    vector_env: VectorEnv,
) -> None:
    """A2：筛选星云独立版本、filter_json 落库、坐标与全量版不可比。"""

    full = build_map(
        vector_env.catalog, vector_env.cfg, version=vector_env.version, note="acceptance-full-2"
    )
    assert full.n_works == len(vector_env.work_ids)

    bilibili_ids = {
        vector_env.work_ids[name] for name in ("w1", "w2", "w3", "w4")
    }
    filtered = build_map(
        vector_env.catalog,
        vector_env.cfg,
        version=vector_env.version,
        filter_spec={"dataset": "bilibili"},
        note="acceptance-bilibili",
    )
    assert filtered.version != full.version
    assert filtered.n_works == len(bilibili_ids)

    run = vector_env.catalog.get_map(filtered.version)
    assert json.loads(str(run["filter_json"])) == {"dataset": "bilibili"}
    meta = json.loads((filtered.coords_path.parent / "meta.json").read_text(encoding="utf-8"))
    assert meta["filter_spec"] == {"dataset": "bilibili"}

    all_ids, all_matrix = load_embedding_matrix(
        vector_env.catalog, vector_env.cfg, version=vector_env.version
    )
    filtered_ids, filtered_matrix = load_embedding_matrix(
        vector_env.catalog,
        vector_env.cfg,
        version=vector_env.version,
        filter_spec={"dataset": "bilibili"},
    )
    assert set(all_ids) == set(vector_env.work_ids.values())
    assert set(filtered_ids) == bilibili_ids
    assert all_matrix.shape[0] == len(all_ids)
    assert filtered_matrix.shape[0] == len(bilibili_ids)

    full_coords = pd.read_parquet(full.coords_path).set_index("work_id")
    filtered_coords = pd.read_parquet(filtered.coords_path).set_index("work_id")
    common = sorted(set(full_coords.index) & set(filtered_coords.index))
    assert common == sorted(bilibili_ids)
    diffs = np.abs(
        full_coords.loc[common, ["x2", "y2", "x3", "y3", "z3"]].to_numpy(dtype=float)
        - filtered_coords.loc[common, ["x2", "y2", "x3", "y3", "z3"]].to_numpy(dtype=float)
    )
    assert float(diffs.max()) > 1e-6, "不同筛选星云的坐标不应相同（坐标不可跨版本比较）"

    year_map = build_map(
        vector_env.catalog,
        vector_env.cfg,
        version=vector_env.version,
        filter_spec={"vote_year": 2025},
    )
    assert year_map.n_works == 4
    expected_2025 = {
        vector_env.work_ids[name] for name in ("w1", "w2", "w3", "w5")
    }
    assert set(load_embedding_matrix(
        vector_env.catalog, vector_env.cfg, version=vector_env.version,
        filter_spec={"vote_year": 2025},
    )[0]) == expected_2025

    # 筛选后作品不足 3 部时必须中文报错，而不是生成空星云
    with pytest.raises(MapError, match="至少"):
        build_map(
            vector_env.catalog,
            vector_env.cfg,
            version=vector_env.version,
            filter_spec={"dataset": "local"},
        )

# ---------------------------------------------------------------- B1 身份关联

class _FakeBilibiliClient:
    """离线假 Bilibili 客户端：只返回预置元数据，不触网。"""

    def __init__(self, metas: dict[str, BilibiliMeta]) -> None:
        self.metas = metas
        self.closed = False

    def fetch_meta(self, bvid: str, *, part: int = 1) -> BilibiliMeta:
        return self.metas[bvid]

    def close(self) -> None:
        self.closed = True


@dataclass
class IdentityEnv:
    """B1 独立验收环境。"""

    root: Path
    cfg: Config
    catalog: Catalog
    version: str
    ids: dict[str, int]
    metas: dict[str, BilibiliMeta]


@pytest.fixture(scope="module")
def identity_env(tmp_path_factory: pytest.TempPathFactory) -> IdentityEnv:
    root = tmp_path_factory.mktemp("accept-identity")
    cfg = create_temp_config(root)
    catalog = Catalog(cfg.paths.db_path)
    catalog.initialize()
    version = "v-ident"
    catalog.register_embedding_version(
        version,
        model_checkpoint="synthetic",
        model_revision=None,
        frames=16,
        clip_seconds=8.0,
        clips_per_work=5,
        audio_sample_rate=48000,
        params={"test": "e2e-identity"},
    )
    list_id = catalog.ensure_list(
        "votes_2025", name="2025年音MAD选票", kind="votes", vote_year=2025
    )
    ids: dict[str, int] = {}

    def create_work(
        name: str,
        *,
        source: str,
        title: str,
        duration_s: float | None,
        bvid: str | None = None,
        local_path: str | None = None,
        joint=None,
        clip=None,
        audio=None,
        ranks: tuple[int, int] | None = None,
    ) -> None:
        if source == "local":
            assert local_path is not None
            target = cfg.paths.input_local / local_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"acceptance-local-video-bytes")
            work_id, _created = catalog.upsert_local_work(local_path, title=title)
        else:
            assert bvid is not None
            work_id, _created = catalog.upsert_bilibili_work(bvid, title=title)
        catalog.begin_attempt(work_id, "encoding")
        if duration_s is not None:
            catalog.update_work_metadata(work_id, duration_s=duration_s)
        work_dir = cfg.paths.embeddings / version / f"{work_id:06d}"
        if joint is not None:
            save_vector(work_dir / "work_joint.npy", _unit64(joint))
            catalog.register_embedding(
                work_id,
                version=version,
                kind="work_joint",
                path=cfg.data_relpath(work_dir / "work_joint.npy"),
                dim=8,
            )
        if audio is not None:
            save_vector(work_dir / "work_audio.npy", _unit64(audio))
            catalog.register_embedding(
                work_id,
                version=version,
                kind="work_audio",
                path=cfg.data_relpath(work_dir / "work_audio.npy"),
                dim=8,
            )
        if clip is not None:
            matrix = np.stack([_unit64(row) for row in clip])
            save_vector(work_dir / "clip_joint.npy", matrix)
            catalog.register_embedding(
                work_id,
                version=version,
                kind="clip_joint",
                path=cfg.data_relpath(work_dir / "clip_joint.npy"),
                dim=8,
                n_clips=int(matrix.shape[0]),
            )
            catalog.replace_clips(
                work_id,
                version=version,
                clips=[(index, index * 8.0, 8.0) for index in range(matrix.shape[0])],
            )
        catalog.complete(work_id)
        if ranks is not None:
            catalog.add_appearance(work_id, list_id, rank=ranks[0], votes=ranks[1])
        ids[name] = int(work_id)

    metas = {
        "BV1xx411c7n1": BilibiliMeta(
            bvid="BV1xx411c7n1",
            title="作品甲 LIVE",
            author_uid="1001",
            author_name="UP甲",
            duration_s=60.5,
            publish_date="2025-01-02",
            cover_url=None,
            page_count=1,
        ),
        "BV1xx411c7n2": BilibiliMeta(
            bvid="BV1xx411c7n2",
            title="作品乙",
            author_uid="1002",
            author_name="UP乙",
            duration_s=70.0,
            publish_date="2025-02-03",
            cover_url=None,
            page_count=1,
        ),
        "BV1xx411c7n3": BilibiliMeta(
            bvid="BV1xx411c7n3",
            title="完全不同的作品",
            author_uid="1003",
            author_name="UP丙",
            duration_s=100.0,
            publish_date="2025-03-04",
            cover_url=None,
            page_count=1,
        ),
    }
    try:
        create_work(
            "local_a",
            source="local",
            local_path="same/作品甲.mp4",
            title="作品甲 Live",
            duration_s=61.0,
            joint=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            clip=[
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            audio=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        create_work(
            "local_b",
            source="local",
            local_path="same/作品乙.mp4",
            title="作品乙",
            duration_s=50.0,
            joint=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        create_work(
            "local_c",
            source="local",
            local_path="same/作品丙.mp4",
            title="作品丙",
            duration_s=40.0,
            joint=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        create_work(
            "bili_a",
            source="bilibili",
            bvid="BV1xx411c7n1",
            title="作品甲   LIVE",
            duration_s=None,
            ranks=(1, 500),
        )
        create_work(
            "bili_b",
            source="bilibili",
            bvid="BV1xx411c7n2",
            title="作品乙",
            duration_s=None,
            ranks=(2, 400),
        )
        create_work(
            "bili_c",
            source="bilibili",
            bvid="BV1xx411c7n3",
            title="完全不同的作品",
            duration_s=None,
            ranks=(3, 300),
        )
        yield IdentityEnv(
            root=root, cfg=cfg, catalog=catalog, version=version, ids=ids, metas=metas
        )
    finally:
        catalog.close()


def test_b1_normalize_title_boundaries() -> None:
    """B1：标题规范化处理空白/换行/全半角/引号/零宽字符，空标题报中文错误。"""

    equal_pairs = [
        ("作品 甲\nLive", "作品甲live"),
        ("【音MAD】Foo：Live！", "[音MAD]Foo:Live!"),
        ("ＡＢＣ　１２３", "abc123"),
        ("作\u200b品甲", "作品甲"),
        ("「测试」作品", '"测试"作品'),
    ]
    for first, second in equal_pairs:
        assert normalize_title(first) == normalize_title(second), (first, second)
    assert normalize_title("作品甲") != normalize_title("作品乙")
    for bad in ("", " \n\t ", "\u200b"):
        with pytest.raises(IngestError):
            normalize_title(bad)
    with pytest.raises(IngestError):
        normalize_title(123)


def test_b1_find_candidates_double_validation(identity_env: IdentityEnv) -> None:
    """B1：只有“标题规范化完全相等 + 时长差 <= 3s”才接受，其余保留拒绝原因。"""

    client = _FakeBilibiliClient(identity_env.metas)
    candidates = find_candidates(
        identity_env.catalog,
        identity_env.cfg,
        list_key="votes_2025",
        client=client,
    )
    accepted = [candidate for candidate in candidates if candidate.accepted]
    assert len(accepted) == 1
    winner = accepted[0]
    assert winner.local_work_id == identity_env.ids["local_a"]
    assert winner.bilibili_work_id == identity_env.ids["bili_a"]
    assert winner.method == "title+duration"
    assert winner.local_duration_s == pytest.approx(61.0)
    assert winner.bilibili_duration_s == pytest.approx(60.5)
    assert winner.confidence > 0.9
    assert "时长差" in winner.reason

    rejected = [candidate for candidate in candidates if not candidate.accepted]
    target = [
        candidate
        for candidate in rejected
        if candidate.local_work_id == identity_env.ids["local_b"]
        and candidate.bilibili_work_id == identity_env.ids["bili_b"]
    ]
    assert target, "本地作品乙与 B 站作品乙应产生一条被拒绝的候选"
    assert "容差" in target[0].reason
    assert target[0].bilibili_duration_s == pytest.approx(70.0)

    # 标题完全不同的作品不应产生候选
    assert not any(
        candidate.bilibili_work_id == identity_env.ids["bili_c"] for candidate in candidates
    )

    # 拉取元数据时应把真实标题/时长补全到作品记录
    bili_a = identity_env.catalog.get_work(identity_env.ids["bili_a"])
    assert bili_a["title"] == "作品甲 LIVE"
    assert float(bili_a["duration_s"]) == pytest.approx(60.5)


def test_b1_link_copies_vectors_and_excludes_duplicate(identity_env: IdentityEnv) -> None:
    """B1：关联后复制向量给 canonical、duplicate 默认不进星云/近邻、原始数据保留。"""

    catalog = identity_env.catalog
    cfg = identity_env.cfg
    version = identity_env.version
    ids = identity_env.ids

    candidate = find_candidates(
        catalog, cfg, list_key="votes_2025", client=_FakeBilibiliClient(identity_env.metas)
    )
    accepted = [item for item in candidate if item.accepted]
    assert len(accepted) == 1
    duplicate = ids["local_a"]
    canonical = ids["bili_a"]

    local_dir = cfg.paths.embeddings / version / f"{duplicate:06d}"
    local_before = {
        path.name: hashlib.sha1(path.read_bytes()).hexdigest()
        for path in sorted(local_dir.rglob("*.npy"))
    }
    local_video = cfg.paths.input_local / "same" / "作品甲.mp4"
    video_bytes_before = local_video.read_bytes()

    result = link_candidates(catalog, cfg, candidates=candidate, copy_vectors=True, version=version)
    assert result["linked"] == 1
    assert result["copied"] == 1
    assert catalog.get_duplicate_work_ids() == {duplicate}
    assert catalog.get_canonical_work_id(duplicate) == canonical
    assert catalog.get_work(canonical)["status"] == "completed"
    assert catalog.has_embedding(canonical, version=version, kind="work_joint")
    assert catalog.has_embedding(canonical, version=version, kind="clip_joint")
    assert catalog.has_embedding(canonical, version=version, kind="work_audio")
    assert len(catalog.get_clips(canonical, version=version)) == 2

    canonical_dir = cfg.paths.embeddings / version / f"{canonical:06d}"
    for kind in ("work_joint", "clip_joint", "work_audio"):
        local_array = np.load(local_dir / f"{kind}.npy", allow_pickle=False)
        copied_array = np.load(canonical_dir / f"{kind}.npy", allow_pickle=False)
        assert np.array_equal(local_array, copied_array), f"{kind} 复制内容不一致"

    # 本地原始数据不删除、不改写
    local_after = {
        path.name: hashlib.sha1(path.read_bytes()).hexdigest()
        for path in sorted(local_dir.rglob("*.npy"))
    }
    assert local_after == local_before
    assert local_video.read_bytes() == video_bytes_before
    assert catalog.has_embedding(duplicate, version=version, kind="work_joint")

    # duplicate 默认不进入星云
    default_map = build_map(catalog, cfg, version=version, note="identity-default")
    assert default_map.n_works == 3  # canonical + local_b + local_c
    assert duplicate not in {
        int(row["id"]) for row in catalog.load_map_rows(default_map.version)
    }
    duplicated_map = build_map(
        catalog,
        cfg,
        version=version,
        note="identity-with-duplicates",
        filter_spec={"include_duplicates": True},
    )
    assert duplicated_map.n_works == 4
    assert duplicate in {
        int(row["id"]) for row in catalog.load_map_rows(duplicated_map.version)
    }
    assert json.loads(str(catalog.get_map(duplicated_map.version)["filter_json"])) == {
        "include_duplicates": True
    }

    # duplicate 默认不进入高维近邻索引
    index = load_similarity_index(catalog, cfg, version=version)
    try:
        assert duplicate not in index.work_ids
        for work_id in index.work_ids:
            assert duplicate not in {
                neighbor.work_id for neighbor in index.neighbors(work_id, k=5)
            }
    finally:
        index.close()

    # unlink 后 duplicate 恢复为普通作品
    restored = catalog.unlink_work(duplicate, reason="独立验收-撤销")
    assert restored == canonical
    assert catalog.get_duplicate_work_ids() == set()


def test_b1_manual_links_override(identity_env: IdentityEnv) -> None:
    """B1：手动关联清单可覆盖自动规则，非法清单给出带原因的 IngestError。"""

    catalog = identity_env.catalog
    cfg = identity_env.cfg
    ids = identity_env.ids
    manual_path = cfg.paths.input_lists / "local_bv_links.csv"
    manual_path.write_text(
        "local_path,bvid\nsame/作品乙.mp4,BV1xx411c7n2\n", encoding="utf-8"
    )
    links = load_manual_links(catalog, cfg)
    assert links == [(ids["local_b"], ids["bili_b"])]

    manual_candidates = candidates_from_manual_links(catalog, links)
    assert len(manual_candidates) == 1
    assert manual_candidates[0].accepted is True
    assert manual_candidates[0].method == "manual"
    result = link_candidates(
        catalog, cfg, candidates=manual_candidates, copy_vectors=False, version=identity_env.version
    )
    assert result["linked"] == 1
    assert result["copied"] == 0
    # 上一测试末尾已 unlink local_a，这里只应新增 local_b
    assert catalog.get_duplicate_work_ids() == {ids["local_b"]}
    assert catalog.get_canonical_work_id(ids["local_b"]) == ids["bili_b"]

    # 非法 BV 号：带行号的中文错误
    manual_path.write_text("local_path,bvid\nsame/作品丙.mp4,NOT_A_BVID\n", encoding="utf-8")
    with pytest.raises(IngestError, match="BV"):
        load_manual_links(catalog, cfg)
    # 未登记的本地路径：带行号提示先 scan-local
    manual_path.write_text(
        "local_path,bvid\nnot/registered.mp4,BV1xx411c7n3\n", encoding="utf-8"
    )
    with pytest.raises(IngestError, match="未登记"):
        load_manual_links(catalog, cfg)

# ---------------------------------------------------------------- A2 界面（AppTest）

@dataclass
class AppEnv:
    """A2 界面独立验收环境。"""

    root: Path
    cfg: Config
    catalog: Catalog
    version: str
    map_version: str
    work_ids: dict[str, int]


@pytest.fixture(scope="module")
def app_env(tmp_path_factory: pytest.TempPathFactory) -> AppEnv:
    root = tmp_path_factory.mktemp("accept-app")
    cfg = create_temp_config(root)
    catalog = Catalog(cfg.paths.db_path)
    catalog.initialize()
    version = "v-app"
    catalog.register_embedding_version(
        version,
        model_checkpoint="synthetic",
        model_revision=None,
        frames=16,
        clip_seconds=8.0,
        clips_per_work=5,
        audio_sample_rate=48000,
        params={"test": "e2e-app"},
    )
    list_2025 = catalog.ensure_list(
        "votes_2025", name="2025年音MAD选票", kind="votes", vote_year=2025
    )
    list_2026 = catalog.ensure_list(
        "votes_2026", name="2026年音MAD选票", kind="votes", vote_year=2026
    )
    specs = [
        ("app1", "bilibili", "BV1xx411c7p1", "界面作品一", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], list_2025, 1),
        ("app2", "bilibili", "BV1xx411c7p2", "界面作品二", [0.9, 0.435889894, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], list_2025, 2),
        ("app3", "bilibili", "BV1xx411c7p3", "界面作品三", [0.7, 0.714142842, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], list_2026, 1),
        ("app4", "local", None, "本地界面作品四", [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], list_2026, 2),
        ("app5", "local", None, "本地界面作品五", [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], list_2025, 3),
    ]
    work_ids: dict[str, int] = {}
    try:
        for name, source, bvid, title, vector, target_list, rank in specs:
            if source == "local":
                work_id, _created = catalog.upsert_local_work(
                    f"app/{name}.mp4", title=title
                )
            else:
                work_id, _created = catalog.upsert_bilibili_work(str(bvid), title=title)
            catalog.begin_attempt(work_id, "encoding")
            catalog.update_work_metadata(work_id, duration_s=60.0 + rank)
            work_dir = cfg.paths.embeddings / version / f"{work_id:06d}"
            has_audio = name in {"app1", "app2", "app3"}
            has_video = name in {"app1", "app2", "app3", "app4"}
            for kind, values in (
                ("work_joint", vector),
                ("work_audio", [1.0, 0.0, 0.0, 0.0] if has_audio else None),
                ("work_video", [0.0, 1.0, 0.0, 0.0] if has_video else None),
            ):
                if values is None:
                    continue
                array = _unit64(values)
                save_vector(work_dir / f"{kind}.npy", array)
                catalog.register_embedding(
                    work_id,
                    version=version,
                    kind=kind,
                    path=cfg.data_relpath(work_dir / f"{kind}.npy"),
                    dim=int(array.shape[0]),
                )
            if name in {"app1", "app2"}:
                clip_matrices = {
                    "clip_joint": np.stack(
                        [
                            _unit64([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                            _unit64([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                        ]
                    ),
                    "clip_audio": np.stack(
                        [_unit64([1.0, 0.0, 0.0, 0.0]), _unit64([0.0, 1.0, 0.0, 0.0])]
                    ),
                }
                for clip_kind, matrix in clip_matrices.items():
                    save_vector(work_dir / f"{clip_kind}.npy", matrix)
                    catalog.register_embedding(
                        work_id,
                        version=version,
                        kind=clip_kind,
                        path=cfg.data_relpath(work_dir / f"{clip_kind}.npy"),
                        dim=int(matrix.shape[1]),
                        n_clips=int(matrix.shape[0]),
                    )
                    catalog.replace_clips(
                        work_id,
                        version=version,
                        clips=[(index, index * 8.0, 8.0) for index in range(matrix.shape[0])],
                    )
            catalog.complete(work_id)
            catalog.add_appearance(work_id, target_list, rank=rank, votes=100 - rank)
            work_ids[name] = int(work_id)
        built = build_map(catalog, cfg, version=version, note="acceptance-app")
        yield AppEnv(
            root=root,
            cfg=cfg,
            catalog=catalog,
            version=version,
            map_version=built.version,
            work_ids=work_ids,
        )
    finally:
        catalog.close()


def test_v02_app_neighbor_panel_density_filters_and_3d(
    app_env: AppEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A2 独立验收：AppTest 渲染近邻面板/高亮、近邻跳转、密度染色、筛选与 3D。"""

    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("NEBULA_CONFIG", str(app_env.cfg.path))
    at = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=120)
    at.run()
    assert len(at.exception) == 0, f"页面出现异常: {list(at.exception)}"
    assert len(at.error) == 0, f"页面显示错误: {list(at.error)}"

    # 基础控件与主图
    assert at.selectbox(key="map_version").value == app_env.map_version
    assert at.radio(key="projection").value == "2d"
    assert at.selectbox(key="color_field").value is not None
    assert at.selectbox(key="filter_source").value == "all"
    charts = at.get("plotly_chart")
    assert len(charts) == 1
    spec = json.loads(charts[0].proto.spec)
    assert "scatter" in str(spec).lower()

    # 默认选中第一部作品：详情 + 三个模态的近邻表 + 高亮/连线 trace
    tables = [element.value for element in at.dataframe]
    assert len(tables) == 3, f"应有 joint/audio/video 三张近邻表，实际 {len(tables)}"
    trace_names = [str(trace.get("name") or "") for trace in spec["data"]]
    assert any("选中作品" in name for name in trace_names)
    assert any("Top5 近邻" in name for name in trace_names)
    assert len(spec["data"]) >= 3, "2D 默认应画出选中点、近邻点与连线"

    # 独立手算 app1 的联合 Top1 近邻，然后通过“跳到近邻作品”验证跳转
    vectors: dict[int, np.ndarray] = {}
    for row in app_env.catalog.list_embeddings(app_env.version, kind="work_joint"):
        array = np.asarray(
            np.load(app_env.cfg.data_path(str(row["path"])), allow_pickle=False),
            dtype=np.float64,
        )
        vectors[int(row["work_id"])] = array / np.linalg.norm(array)
    expected_top1 = _expected_ranked(vectors, app_env.work_ids["app1"])[0]
    jump_work_id, jump_similarity = expected_top1
    jump_titles = {
        int(row["id"]): str(row["title"])
        for row in app_env.catalog.list_works()
    }
    assert at.selectbox(key="manual_work").value == app_env.work_ids["app1"]
    at.selectbox(key="neighbor_jump").set_value(jump_work_id).run()
    assert len(at.exception) == 0
    assert at.selectbox(key="manual_work").value == jump_work_id
    headers = [str(element.value) for element in at.subheader]
    assert jump_titles[jump_work_id] in headers

    # 高维近邻密度染色（需要 SimilarityIndex，不能退化成平面距离）
    at.selectbox(key="color_field").set_value("density_knn").run()
    assert len(at.exception) == 0
    assert len(at.error) == 0
    assert len(at.get("plotly_chart")) == 1

    # 筛选：来源 local -> 2 部；年份 2026 -> 2 部；搜索唯一标题 -> 1 部
    at.selectbox(key="filter_source").set_value("local").run()
    captions = [str(element.value) for element in at.caption]
    assert any("当前显示 2 / 5" in caption for caption in captions), captions
    at.selectbox(key="filter_source").set_value("all").run()
    at.multiselect(key="filter_years").set_value([2026]).run()
    captions = [str(element.value) for element in at.caption]
    assert any("当前显示 2 / 5" in caption for caption in captions), captions
    at.multiselect(key="filter_years").set_value([]).run()
    at.text_input(key="search").set_value("界面作品三").run()
    captions = [str(element.value) for element in at.caption]
    assert any("当前显示 1 / 5" in caption for caption in captions), captions
    at.text_input(key="search").set_value("").run()

    # 选中没有音频/画面向量的作品：显式提示而不是报错
    at.selectbox(key="manual_work").set_value(app_env.work_ids["app5"]).run()
    assert len(at.exception) == 0
    captions = [str(element.value) for element in at.caption]
    assert any("无音频模态向量" in caption for caption in captions), captions
    assert any("无画面模态向量" in caption for caption in captions), captions

    # 3D 投影 + 连线开关
    at.radio(key="projection").set_value("3d").run()
    assert len(at.exception) == 0
    assert len(at.error) == 0
    spec_3d = json.loads(at.get("plotly_chart")[0].proto.spec)
    assert any("scatter3d" in str(trace.get("type", "")).lower() for trace in spec_3d["data"])
    assert at.checkbox(key="show_lines_3d").value is False
    at.checkbox(key="show_lines_3d").set_value(True).run()
    assert len(at.exception) == 0
    spec_3d_lines = json.loads(at.get("plotly_chart")[0].proto.spec)
    assert len(spec_3d_lines["data"]) >= 3, "3D 打开连线后应有选中点、近邻点与连线"

    # 近邻跳到可比较作品后，Top1 邻居余弦值应与手算一致（展示精度 4 位）
    assert jump_similarity == pytest.approx(max(
        sim for work_id, sim in _expected_ranked(vectors, app_env.work_ids["app1"])
    ))

def test_v02_app_hubness_and_clip_explorer(
    app_env: AppEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1/P2 界面独立验收：近邻解释列、枢纽作品表、片段热图与配对表。"""

    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("NEBULA_CONFIG", str(app_env.cfg.path))
    at = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=120)
    at.run()
    assert len(at.exception) == 0, f"页面出现异常: {list(at.exception)}"

    first = app_env.work_ids["app1"]
    compare = app_env.work_ids["app2"]
    at.selectbox(key="manual_work").set_value(first).run()
    assert len(at.exception) == 0

    # 近邻表必须带“全库分位/双向”解释列
    tables = [element.value for element in at.dataframe]
    assert tables, "近邻表应渲染"
    columns = set(str(column) for column in tables[0].columns)
    assert {"全库分位", "双向"} <= columns, columns

    # 枢纽作品表（st.table）
    hub_tables = at.get("table")
    assert len(hub_tables) >= 1, "枢纽作品表应渲染"
    assert len(hub_tables[0].value) >= 1

    # 片段级对应：默认关闭；勾选后出现对比下拉、热图与配对表
    checkbox = at.checkbox(key="show_clip_panel")
    assert checkbox.value is False
    at.checkbox(key="show_clip_panel").set_value(True).run()
    assert len(at.exception) == 0
    at.selectbox(key="clip_compare_work").set_value(compare).run()
    assert len(at.exception) == 0

    charts = at.get("plotly_chart")
    assert len(charts) == 2, f"应有星云图 + 片段热图两张图，实际 {len(charts)}"
    specs = [json.loads(chart.proto.spec) for chart in charts]
    assert any(
        str(trace.get("type", "")).lower() == "heatmap"
        for spec in specs
        for trace in spec.get("data", [])
    ), "片段级对应应渲染 Plotly heatmap"
    captions = [str(element.value) for element in at.caption]
    assert any("片段级统计" in caption for caption in captions), captions
    # 3 个模态近邻表 + 1 个最高片段配对表
    assert len([element for element in at.dataframe]) >= 4

    # 片段模态可切换（clip_joint / clip_audio 都有数据）
    clip_kind = at.selectbox(key="clip_kind")
    kind_labels = [str(option) for option in clip_kind.options]
    assert {"联合片段", "音频片段"} <= set(kind_labels), kind_labels
    at.selectbox(key="clip_kind").set_value("clip_audio").run()
    assert len(at.exception) == 0
    assert len(at.get("plotly_chart")) == 2

    # 关闭热图后回到单图，且不残留异常
    at.checkbox(key="show_clip_panel").set_value(False).run()
    assert len(at.exception) == 0
    assert len(at.get("plotly_chart")) == 1


# ──────────────────────────────────────────────────────────────────────────────
# v0.4 全片覆盖 -> Full Mean -> 表征地图 -> 空间对比（离线全链路）
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class V04E2EState:
    """v0.4 全链路验收环境。"""

    root: Path
    cfg: Config
    catalog: Catalog
    version: str
    legacy_version: str
    work_ids: dict[str, int]
    installed: dict[str, Path]
    rep_id: int
    map_version: str
    failed_work_id: int
    reused_batches: int
    report_path: Path


def _v04_run_cli(args: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )


def _expected_full_window_count(duration_s: float, clip_seconds: float = 8.0) -> int:
    """独立推导全片覆盖窗口数：短视频 1 段，否则 ceil((duration-clip)/clip)+1。"""

    if duration_s <= clip_seconds + 1e-9:
        return 1
    return int(math.ceil((duration_s - clip_seconds) / clip_seconds - 1e-12)) + 1


class _ScriptedHashEmbedder(HashTestEmbedder):
    """可控中断的离线嵌入器：在第 N 次 encode_pairs 注入失败，并记录调用批次。"""

    def __init__(self, *, fail_on_call: int | None = None) -> None:
        super().__init__()
        self.fail_on_call = fail_on_call
        self.calls = 0
        self.batch_sizes: list[int] = []

    def encode_pairs(self, pairs):  # noqa: ANN001 - 与 Embedder 协议一致
        self.calls += 1
        self.batch_sizes.append(len(pairs))
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise RuntimeError("注入的全片编码中断（验收用）")
        return super().encode_pairs(pairs)


@pytest.fixture(scope="module")
def v04_e2e(tmp_path_factory: pytest.TempPathFactory) -> V04E2EState:
    """构建一次 v0.4 全链路环境：4 部合成视频 + 中断恢复 + Full Mean + 表征地图 + 空间对比。"""

    if not media_tools_available():
        pytest.skip("缺少 ffmpeg/ffprobe，跳过 v0.4 全链路 E2E")

    root = tmp_path_factory.mktemp("v04-e2e")
    cfg = create_temp_config(
        root,
        overrides={
            "sampling": {"mode": "full_coverage", "overlap_seconds": 0.0},
            "events": {"enabled": False},
            "model": {"batch_size": 2},
            "umap": {"n_jobs": 1, "random_state": 42},
        },
    )
    library = build_video_library(root / "video_cache")
    names = ["short_4s.mp4", "mid_12s.mp4", "odd_17s.mp4", "long_40s.mp4"]
    installed: dict[str, Path] = {}
    for name in names:
        destination = cfg.paths.input_local / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(library[name], destination)
        installed[name] = destination

    catalog = Catalog(cfg.paths.db_path)
    catalog.initialize()
    try:
        summary = register_local_videos(catalog, cfg)
        assert summary.files == len(names)
        works = {str(work["local_path"]): int(work["id"]) for work in catalog.list_works()}
        work_ids = {name: works[name] for name in names}

        # 1) 首次全片编码：第 3 次 encode_pairs 注入失败，留下 .partial 批次
        interrupted = _ScriptedHashEmbedder(fail_on_call=3)
        first_pipeline = Pipeline(
            cfg,
            embedder=interrupted,
            allow_test_embedder=True,
        )
        try:
            version = first_pipeline.embedding_version
            expected_full_version = cfg.full_sequence_version.replace(
                str(cfg.model.version), "test-hash-v1"
            )
            assert version == expected_full_version
            assert "full" in version and "_ov0" in version
            first_stats = first_pipeline.process_pending(source="local")
        finally:
            first_pipeline.close()
        batch_size = max(1, int(cfg.model.batch_size))
        expected_total_batches = sum(
            math.ceil(
                _expected_full_window_count(probe_duration_s(video_path)) / batch_size
            )
            for video_path in installed.values()
        )
        assert interrupted.calls == expected_total_batches, interrupted.batch_sizes
        assert first_stats.failed == 1, first_stats.summary()
        failed_outcome = next(item for item in first_stats.outcomes if item.status == "failed")
        failed_work_id = int(failed_outcome.work_id)
        state = catalog.get_sequence_state(failed_work_id, version=version)
        assert state is not None and state["status"] == "failed"
        failed_partial = cfg.paths.embeddings / version / f"{failed_work_id:06d}" / ".partial"
        reused = sorted(failed_partial.glob("batch_*.npz"))
        assert reused, "中断后必须保留已完成的批次文件"

        # 2) 重启恢复：只补缺失批次，不重复编码已存在批次
        resumed = _ScriptedHashEmbedder()
        resume_pipeline = Pipeline(cfg, embedder=resumed, allow_test_embedder=True)
        try:
            assert resume_pipeline.embedding_version == version
            second_stats = resume_pipeline.process_pending(source="local")
        finally:
            resume_pipeline.close()
        assert second_stats.failed == 0 and second_stats.completed == 1
        failed_name = next(name for name, work_id in work_ids.items() if work_id == failed_work_id)
        failed_duration = probe_duration_s(installed[failed_name])
        expected_failed_windows = _expected_full_window_count(failed_duration)
        expected_failed_batches = math.ceil(expected_failed_windows / batch_size)
        assert len(resumed.batch_sizes) == expected_failed_batches - len(reused)
        assert not failed_partial.exists()

        # 3) 覆盖率/版本隔离/片段序列落库
        for name, work_id in work_ids.items():
            work = catalog.get_work(work_id)
            assert work is not None and work["status"] == "completed"
            duration_s = probe_duration_s(installed[name])
            clips = catalog.get_clips(work_id, version=version)
            expected_windows = _expected_full_window_count(duration_s)
            assert len(clips) == expected_windows, (name, len(clips), expected_windows)
            assert [int(row["idx"]) for row in clips] == list(range(expected_windows))
            ordered = sorted(clips, key=lambda row: int(row["idx"]))
            assert ordered[0]["start_s"] == pytest.approx(0.0, abs=1e-6)
            assert ordered[-1]["start_s"] + ordered[-1]["duration_s"] == pytest.approx(
                duration_s, abs=1e-2
            )
            assert sum(float(row["weight"]) for row in clips) == pytest.approx(1.0, abs=1e-6)
            for row in ordered:
                assert row["valid_duration_s"] == pytest.approx(row["duration_s"], abs=1e-6)
                assert row["weight"] is not None and float(row["weight"]) > 0
            for previous, following in zip(ordered, ordered[1:], strict=False):
                assert following["start_s"] <= previous["start_s"] + previous["duration_s"] + 1e-9
            for probe_t in np.linspace(0.0, duration_s - 1e-6, 13):
                assert any(
                    float(row["start_s"]) - 1e-9
                    <= float(probe_t)
                    <= float(row["start_s"]) + float(row["duration_s"]) + 1e-9
                    for row in ordered
                ), (name, probe_t)
            for kind in ("clip_joint", "clip_audio", "clip_video"):
                embedding = catalog.get_embedding(work_id, version=version, kind=kind)
                assert embedding is not None, (name, kind)
                vector = load_vector(cfg.data_path(str(embedding["path"])))
                assert vector.shape[0] == expected_windows
            times = np.load(
                cfg.paths.embeddings / version / f"{work_id:06d}" / "clip_times.npy"
            )
            assert times.shape == (expected_windows, 4)
            assert float(times[:, 3].sum()) == pytest.approx(1.0, abs=1e-6)
            sequence_state = catalog.get_sequence_state(work_id, version=version)
            assert sequence_state is not None
            assert sequence_state["status"] == "assembled"
            assert int(sequence_state["n_clips_done"]) == expected_windows

        # 4) 从已保存序列构建 Full Mean 表征（CLI，不重编码）
        env = os.environ.copy()
        env.update({"PYTHONIOENCODING": "utf-8", "NEBULA_CONFIG": str(cfg.path)})
        build = _v04_run_cli(
            [
                "tools/build_representation.py",
                "--config",
                str(cfg.path),
                "full_mean",
                "--sequence",
                version,
            ],
            cwd=REPO_ROOT,
            env=env,
        )
        assert build.returncode == 0, build.stderr[-3000:]
        representation = catalog.get_representation(
            name="full_mean_v1", sequence_version=version
        )
        assert representation is not None and representation["status"] == "ready"
        rep_id = int(representation["id"])
        representation_vectors = catalog.list_representation_vectors(rep_id)
        joint_works = {
            int(row["work_id"]) for row in representation_vectors if row["kind"] == "joint"
        }
        assert joint_works == set(work_ids.values())

        # 5) 表征地图（CLI），地图必须绑定代表征与同一序列版本
        map_build = _v04_run_cli(
            [
                "map.py",
                "build",
                "--config",
                str(cfg.path),
                "--representation",
                "full_mean_v1",
                "--sequence-version",
                version,
            ],
            cwd=REPO_ROOT,
            env=env,
        )
        assert map_build.returncode == 0, map_build.stderr[-3000:]
        map_row = catalog.latest_map()
        assert map_row is not None
        assert int(map_row["representation_id"]) == rep_id
        assert map_row["embedding_version"] == version
        assert int(map_row["n_works"]) == len(work_ids)
        map_rows = catalog.load_map_rows(str(map_row["version"]))
        assert {int(row["id"]) for row in map_rows} == set(work_ids.values())

        # 6) 旧序列/legacy 表征与空间隔离（同一批作品、不同空间）
        legacy_version = cfg.legacy_embedding_version
        assert legacy_version != version
        catalog.register_embedding_version(
            legacy_version,
            model_checkpoint="synthetic",
            model_revision=None,
            frames=16,
            clip_seconds=cfg.media.clip_seconds,
            clips_per_work=cfg.media.clips_per_work,
            audio_sample_rate=cfg.media.audio_sample_rate,
            params={"test": "v04-e2e-legacy"},
        )
        for work_id in work_ids.values():
            clip_embedding = catalog.get_embedding(work_id, version=version, kind="clip_joint")
            assert clip_embedding is not None
            clip_matrix = load_vector(cfg.data_path(str(clip_embedding["path"])))
            legacy_vector = np.asarray(clip_matrix[0], dtype=np.float32)
            legacy_path = (
                cfg.paths.embeddings / legacy_version / f"{work_id:06d}" / "work_joint.npy"
            )
            save_vector(legacy_path, legacy_vector)
            catalog.register_embedding(
                work_id,
                version=legacy_version,
                kind="work_joint",
                path=cfg.data_relpath(legacy_path),
                dim=int(legacy_vector.size),
            )
        legacy_rep_id = catalog.ensure_legacy_representation(legacy_version)
        assert legacy_rep_id != rep_id
        assert catalog.get_representation(legacy_rep_id)["name"] == "legacy_5clip_mean"

        # 7) appdata 默认地图 = 最新表征地图，并带表征空间信息
        map_info, frame = load_nebula_frame(catalog, cfg)
        assert len(frame) == len(work_ids)
        assert map_info["version"] == str(map_row["version"])
        assert map_info["representation_id"] == rep_id
        assert map_info["representation_name"] == "full_mean_v1"
        assert map_info["representation_sequence_version"] == version
        assert "Full Mean" in str(map_info["representation_label"])
        if "is_canonical" in frame:
            assert bool(frame["is_canonical"].all())

        # 8) 邻域迁移对比 CLI：legacy × full_mean 输出 JSON/CSV
        compare = _v04_run_cli(
            [
                "tools/compare_spaces.py",
                "--config",
                str(cfg.path),
                "--a",
                "legacy",
                "--a-seq",
                legacy_version,
                "--b",
                "full_mean_v1",
                "--b-seq",
                version,
                "--k",
                "2",
                "--out-dir",
                str(cfg.paths.logs),
            ],
            cwd=REPO_ROOT,
            env=env,
        )
        assert compare.returncode == 0, compare.stderr[-3000:]
        report_files = sorted(cfg.paths.logs.glob("space_compare_*.json"))
        assert report_files, "compare_spaces 应输出 JSON 报告"
        report_path = report_files[-1]
        assert report_path.with_suffix(".csv").is_file()

        yield V04E2EState(
            root=root,
            cfg=cfg,
            catalog=catalog,
            version=version,
            legacy_version=legacy_version,
            work_ids=work_ids,
            installed=installed,
            rep_id=rep_id,
            map_version=str(map_row["version"]),
            failed_work_id=failed_work_id,
            reused_batches=len(reused),
            report_path=report_path,
        )
    finally:
        catalog.close()


def test_v04_full_coverage_and_interruption_recovery(v04_e2e: V04E2EState) -> None:
    """v0.4 全片覆盖：无片段上限、末尾锚定、任意时刻覆盖、中断后只补缺失批。"""

    assert "full" in v04_e2e.version
    assert v04_e2e.version != v04_e2e.legacy_version
    assert v04_e2e.reused_batches >= 1
    work = v04_e2e.catalog.get_work(v04_e2e.failed_work_id)
    assert work is not None and work["status"] == "completed"
    state = v04_e2e.catalog.get_sequence_state(
        v04_e2e.failed_work_id, version=v04_e2e.version
    )
    assert state is not None and state["status"] == "assembled"
    clips = v04_e2e.catalog.get_clips(
        v04_e2e.failed_work_id, version=v04_e2e.version
    )
    assert len(clips) == _expected_full_window_count(probe_duration_s(
        next(path for name, path in v04_e2e.installed.items()
             if v04_e2e.work_ids[name] == v04_e2e.failed_work_id)
    ))
    assert all(work_id > 0 for work_id in v04_e2e.work_ids.values())


def test_v04_full_mean_map_binding_and_appdata(v04_e2e: V04E2EState) -> None:
    """Full Mean 表征与地图：地图绑定代表征、序列一致、appdata 默认读取该空间。"""

    representation = v04_e2e.catalog.get_representation(v04_e2e.rep_id)
    assert representation is not None
    assert representation["name"] == "full_mean_v1"
    assert representation["sequence_version"] == v04_e2e.version
    assert representation["status"] == "ready"

    map_row = v04_e2e.catalog.get_map(v04_e2e.map_version)
    assert map_row is not None
    assert int(map_row["representation_id"]) == v04_e2e.rep_id
    assert map_row["embedding_version"] == v04_e2e.version

    map_info, frame = load_nebula_frame(v04_e2e.catalog, v04_e2e.cfg)
    assert map_info["representation_name"] == "full_mean_v1"
    assert len(frame) == len(v04_e2e.work_ids)

    # 不同序列/表征的向量不混入同一矩阵：legacy 表征只有手动登记的 work_joint
    from nebula.representations import load_representation_matrix, resolve_representation

    legacy_rep = resolve_representation(
        v04_e2e.catalog,
        v04_e2e.cfg,
        name="legacy",
        sequence_version=v04_e2e.legacy_version,
    )
    legacy_ids, legacy_matrix = load_representation_matrix(
        v04_e2e.catalog, v04_e2e.cfg, legacy_rep
    )
    assert set(legacy_ids) == set(v04_e2e.work_ids.values())
    assert legacy_matrix.shape[0] == len(v04_e2e.work_ids)


def test_v04_compare_spaces_legacy_vs_full(v04_e2e: V04E2EState) -> None:
    """空间对比工具：共同作品、序列版本与报告结构正确。"""

    report = json.loads(v04_e2e.report_path.read_text(encoding="utf-8"))
    assert report["scope"]["n_common"] == len(v04_e2e.work_ids)
    assert report["a"]["sequence_version"] == v04_e2e.legacy_version
    assert report["b"]["sequence_version"] == v04_e2e.version
    assert report["b"]["name"] == "full_mean_v1"
    assert report["a"]["name"] == "legacy_5clip_mean"
    assert len(report["works"]) == len(v04_e2e.work_ids)
    for key in ("topk", "pairs", "cosine", "hubness"):
        assert key in report
    assert -1.0 <= float(report["cosine"]["pearson"]) <= 1.0
    assert 0.0 <= float(report["pairs"]["stable_pair_ratio"]) <= 1.0



