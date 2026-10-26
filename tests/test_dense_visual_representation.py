"""密集视觉表征构建（tools/build_dense_visual_representation.py）的离线测试。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from nebula import representations
from nebula.db import Catalog
from nebula.errors import NebulaError
from tools.build_dense_visual_representation import (  # type: ignore[import-not-found]
    aggregate_window_vectors,
    build_representation,
)

VERSION = "clip-vit-base-patch16-test__fps8_s16_letterbox"


def _register_visual_version(catalog: Catalog) -> None:
    catalog.register_visual_version(
        VERSION,
        model_id="openai/clip-vit-base-patch16",
        revision="rev-test",
        preprocessing="letterbox",
        fps=8.0,
        size=16,
        dim=4,
        params={"test": True},
    )


def _write_windows(config, catalog: Catalog, work_id: int, windows: np.ndarray, weights=None) -> None:
    work_dir = config.paths.visual_dense / VERSION / f"{work_id:06d}"
    work_dir.mkdir(parents=True, exist_ok=True)
    np.save(work_dir / "windows.npy", windows.astype(np.float32))
    manifest = {
        "version": VERSION,
        "n_windows": int(windows.shape[0]),
        "window_times": (
            [{"start_s": i, "end_s": i + 1} for i in range(windows.shape[0])]
            if weights is None
            else [
                {"start_s": i, "end_s": i + 1, "weight": float(w)}
                for i, w in enumerate(weights)
            ]
        ),
    }
    (work_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    catalog.upsert_visual_sequence(
        work_id,
        version=VERSION,
        status="completed",
        frames_path="visual_dense/x/frames.npz",
        windows_path=config.data_relpath(work_dir / "windows.npy"),
        n_frames=int(windows.shape[0] * 8),
        n_windows=int(windows.shape[0]),
        dim=int(windows.shape[1]),
    )


def test_aggregate_window_vectors_equal_and_weighted():
    windows = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    equal = aggregate_window_vectors(windows)
    assert equal.shape == (2,)
    assert np.linalg.norm(equal) == pytest.approx(1.0)
    assert equal[0] == pytest.approx(equal[1])
    weighted = aggregate_window_vectors(windows, [3.0, 1.0])
    assert weighted[0] > weighted[1]
    with pytest.raises(NebulaError):
        aggregate_window_vectors(np.zeros((2, 2)))
    with pytest.raises(NebulaError):
        aggregate_window_vectors(windows, [1.0])


def test_build_representation_from_windows(config, catalog):
    _register_visual_version(catalog)
    w1, _ = catalog.upsert_local_work("a.mp4")
    w2, _ = catalog.upsert_local_work("b.mp4")
    _write_windows(config, catalog, w1, np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32))
    _write_windows(config, catalog, w2, np.asarray([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32))
    summary = build_representation(catalog, config, visual_version=VERSION)
    assert summary["n_built"] == 2
    assert summary["name"] == representations.DENSE_VISUAL_CLIP_V1
    rep = catalog.get_representation(int(summary["representation_id"]))
    assert rep["kind"] == "visual" and rep["source_kind"] == "visual"
    assert rep["status"] == "ready" and rep["dim"] == 4
    rows = catalog.list_representation_vectors(int(rep["id"]), kind="joint")
    assert len(rows) == 2
    vector = np.load(config.data_path(str(rows[0]["path"])))
    assert np.linalg.norm(vector) == pytest.approx(1.0)
    # 已存在时默认报错，--force 可重建
    with pytest.raises(NebulaError):
        build_representation(catalog, config, visual_version=VERSION)
    forced = build_representation(catalog, config, visual_version=VERSION, force=True)
    assert forced["n_built"] == 2


def test_build_skips_incomplete_and_duplicates(config, catalog):
    _register_visual_version(catalog)
    w1, _ = catalog.upsert_local_work("a.mp4")
    w2, _ = catalog.upsert_local_work("b.mp4")
    w3, _ = catalog.upsert_local_work("c.mp4")
    _write_windows(config, catalog, w1, np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    _write_windows(config, catalog, w2, np.asarray([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32))
    # w3 in_progress 不应进入
    catalog.upsert_visual_sequence(w3, version=VERSION, status="in_progress", windows_path="visual_dense/x/w.npy")
    # w2 视为 duplicate
    catalog.link_works(w1, w2, method="test", confidence=0.9)
    summary = build_representation(catalog, config, visual_version=VERSION, limit=10)
    assert summary["n_built"] == 1
    rep = catalog.get_representation(int(summary["representation_id"]))
    rows = catalog.list_representation_vectors(int(rep["id"]), kind="joint")
    assert [int(row["work_id"]) for row in rows] == [w1]


def test_build_representation_v2_window_plan_uses_coverage_weights(config, catalog):
    """v0.4.3.1：窗口计划层 + 显式覆盖权重，而不是等权平均。"""

    _register_visual_version(catalog)
    plan = "win8s_ov0__coverage-v1"
    catalog.register_visual_window_plan(
        plan,
        clip_seconds=8.0,
        overlap_seconds=0.0,
        weight_scheme="coverage-v1",
        params={"test": True},
    )
    w1, _ = catalog.upsert_local_work("v2-a.mp4")
    w2, _ = catalog.upsert_local_work("v2-b.mp4")
    windows1 = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    windows2 = np.asarray([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    weight_map = {
        w1: np.asarray([0.75, 0.25], dtype=np.float32),
        w2: np.asarray([1.0], dtype=np.float32),
    }
    for work_id, windows in ((w1, windows1), (w2, windows2)):
        work_dir = config.paths.visual_windows / plan / VERSION / f"{work_id:06d}"
        work_dir.mkdir(parents=True, exist_ok=True)
        np.save(work_dir / "windows.npy", windows)
        np.save(work_dir / "window_weights.npy", weight_map[work_id])
        np.save(work_dir / "window_std.npy", np.zeros_like(windows))
        catalog.upsert_visual_window_sequence(
            work_id,
            frame_version=VERSION,
            window_plan_version=plan,
            status="completed",
            windows_path=config.data_relpath(work_dir / "windows.npy"),
            window_std_path=config.data_relpath(work_dir / "window_std.npy"),
            window_weights_path=config.data_relpath(work_dir / "window_weights.npy"),
            n_windows=int(windows.shape[0]),
            dim=int(windows.shape[1]),
        )
    summary = build_representation(
        catalog, config, visual_version=VERSION, window_plan_version=plan
    )
    assert summary["name"] == representations.DENSE_VISUAL_CLIP_V2
    assert summary["n_built"] == 2
    rep = catalog.get_representation(int(summary["representation_id"]))
    assert rep["params_json"].find(plan) >= 0
    rows = catalog.list_representation_vectors(int(rep["id"]), kind="joint")
    by_work = {
        int(row["work_id"]): np.load(config.data_path(str(row["path"]))) for row in rows
    }
    expected = aggregate_window_vectors(windows1, weight_map[w1])
    assert np.allclose(by_work[w1], expected, atol=1e-6)
    equal = aggregate_window_vectors(windows1, [1.0, 1.0])
    assert not np.allclose(by_work[w1], equal, atol=1e-4)
    with pytest.raises(NebulaError):
        build_representation(catalog, config, visual_version=VERSION, window_plan_version=plan)
    forced = build_representation(
        catalog, config, visual_version=VERSION, window_plan_version=plan, force=True
    )
    assert forced["n_built"] == 2


def test_build_representation_v2_missing_weights_fails_work(config, catalog):
    _register_visual_version(catalog)
    plan = "win8s_ov0__coverage-v1"
    catalog.register_visual_window_plan(
        plan, clip_seconds=8.0, overlap_seconds=0.0, weight_scheme="coverage-v1"
    )
    work_id, _ = catalog.upsert_local_work("v2-missing.mp4")
    work_dir = config.paths.visual_windows / plan / VERSION / f"{work_id:06d}"
    work_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        work_dir / "windows.npy",
        np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    catalog.upsert_visual_window_sequence(
        work_id,
        frame_version=VERSION,
        window_plan_version=plan,
        status="completed",
        windows_path=config.data_relpath(work_dir / "windows.npy"),
        n_windows=1,
        dim=4,
    )
    with pytest.raises(NebulaError, match="没有构建出任何作品"):
        build_representation(
            catalog, config, visual_version=VERSION, window_plan_version=plan
        )

