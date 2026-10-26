"""tools/build_visual_windows.py / nebul.visual_pipeline 窗口层重建的离线测试。

覆盖：

- 从已保存帧 npz 重建窗口层（不下载/不解码/不跑模型），非整 8s 视频（17s）
  coverage-v1 权重与等权不同、和为 1；
- DB 登记（visual_window_plans + visual_window_sequences）；
- resume / --force；
- 同一帧版本不同窗口计划（clip/overlap/权重方案）不互相跳过、目录隔离；
- CLI 参数（--window-seconds/--overlap-seconds/--weight-scheme/--frame-version/
  --all/--fps/--limit/--force/--dry-run）与帧版本过滤；
- 缺帧/未登记帧版本报告；dry-run 零写入；重建不修改帧 npz（不重新编码）。

全部使用合成特征，不加载视觉模型、不联网。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from nebula import sampling as sampling_module
from nebula.config import Config
from nebula.db import Catalog
from nebula.dense_visual import DenseVisualFeatures, save_features
from nebula.sampling import plan_full_coverage
from nebula.util import now_utc_iso
from nebula.visual_pipeline import (
    WINDOW_MANIFEST_NAME,
    WINDOW_STD_NAME,
    WINDOW_WEIGHTS_NAME,
    WINDOWS_NAME,
    build_visual_windows,
    compose_visual_version,
    resolve_window_plan,
    visual_windows_dir,
)
from tools import build_visual_windows as build_tool

FRAME_DIM = 4


# ---------------------------------------------------------------- 合成数据


def _make_work(catalog: Catalog, duration_s: float, index: int = 0) -> int:
    work_id, _ = catalog.upsert_local_work(f"window_tool_{index}.mp4")
    catalog.update_work_metadata(work_id, duration_s=float(duration_s))
    catalog.begin_attempt(work_id, "encoding")
    catalog.complete(work_id)
    return work_id


def _make_frame_bundle(
    config: Config,
    catalog: Catalog,
    work_id: int,
    *,
    frame_version: str,
    duration_s: float,
    fps: float = 2.0,
    encoder: str = "fake-a",
    revision: str = "rev-a",
) -> Path:
    n_frames = max(1, int(round(duration_s * fps)))
    base = np.arange(1, FRAME_DIM + 1, dtype=np.float32)
    per_frame = np.tile(base, (n_frames, 1))
    per_frame[:, 0] += np.arange(n_frames, dtype=np.float32)
    per_frame /= np.maximum(
        np.linalg.norm(per_frame, axis=1, keepdims=True), 1e-9
    )
    features = DenseVisualFeatures(
        per_frame=per_frame.astype(np.float32),
        fps=float(fps),
        duration_s=float(duration_s),
        n_frames=n_frames,
        dim=FRAME_DIM,
        version=frame_version,
        decode_s=0.01,
        encode_s=0.02,
        model_load_s=0.0,
        encoder=encoder,
        revision=revision,
        size=32,
        preprocess="letterbox",
        created_at=now_utc_iso(),
    )
    out_dir = config.paths.visual_dense / frame_version / f"{work_id:06d}"
    saved = save_features(features, out_dir)
    if catalog.get_visual_version(frame_version) is None:
        catalog.register_visual_version(
            frame_version,
            model_id=encoder,
            revision=revision,
            preprocessing="letterbox",
            fps=float(fps),
            size=32,
            dim=FRAME_DIM,
            params={"source": "test", "fps": float(fps)},
        )
    catalog.upsert_visual_sequence(
        work_id,
        version=frame_version,
        status="completed",
        frames_path=config.data_relpath(saved["npz"]),
        n_frames=n_frames,
        dim=FRAME_DIM,
        decode_s=0.01,
        encode_s=0.02,
        model_load_s=0.0,
        total_s=0.03,
    )
    return out_dir


def _frame_version_name(fps: float = 2.0) -> str:
    return compose_visual_version("fake-a@rev-a", fps, 32, "letterbox")


def _bundle_dir(
    config: Config, plan_version: str, frame_version: str, work_id: int
) -> Path:
    return visual_windows_dir(config, plan_version, frame_version, work_id)


def _load_weights(
    config: Config, plan_version: str, frame_version: str, work_id: int
) -> np.ndarray:
    return np.load(
        _bundle_dir(config, plan_version, frame_version, work_id)
        / WINDOW_WEIGHTS_NAME
    )


def _expected_windows(duration_s: float, clip: float = 8.0, overlap: float = 0.0):
    return [
        (float(item.start_s), float(item.end_s))
        for item in plan_full_coverage(
            duration_s, clip_seconds=clip, overlap_seconds=overlap
        )
    ]


# ---------------------------------------------------------------- 权重与登记




def _make_legacy_frame_bundle(
    config: Config,
    catalog: Catalog,
    work_id: int,
    *,
    frame_version: str,
    duration_s: float,
    fps: float = 2.0,
    npz_version: str | None = None,
) -> Path:
    """构造 v0.4.3 旧版 bundle：npz version 逻辑版本，manifest version 为 slug。"""

    n_frames = max(1, int(round(duration_s * fps)))
    base = np.arange(1, FRAME_DIM + 1, dtype=np.float32)
    per_frame = np.tile(base, (n_frames, 1))
    per_frame[:, 0] += np.arange(n_frames, dtype=np.float32)
    per_frame /= np.maximum(np.linalg.norm(per_frame, axis=1, keepdims=True), 1e-9)
    logical_version = npz_version or f"fake-a@rev-a__fps{fps:g}_s32_letterbox"
    out_dir = config.paths.visual_dense / frame_version / f"{work_id:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "dense_visual.npz",
        per_frame=per_frame.astype(np.float16),
        fps=np.float32(fps),
        duration_s=np.float32(duration_s),
        n_frames=np.int64(n_frames),
        dim=np.int64(FRAME_DIM),
        version=np.array(logical_version),
    )
    manifest = {
        "version": frame_version,
        "encoder": "fake-a@rev-a",
        "revision": "rev-a",
        "fps": float(fps),
        "duration_s": float(duration_s),
        "n_frames": n_frames,
        "dim": FRAME_DIM,
        "size": 32,
        "preprocess": "letterbox",
        "decode_s": 0.01,
        "encode_s": 0.02,
        "model_load_s": 0.0,
        "created_at": "2026-01-01T00:00:00Z",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if catalog.get_visual_version(frame_version) is None:
        catalog.register_visual_version(
            frame_version,
            model_id="fake-a",
            revision="rev-a",
            preprocessing="letterbox",
            fps=float(fps),
            size=32,
            dim=FRAME_DIM,
            params={"source": "test", "fps": float(fps)},
        )
    catalog.upsert_visual_sequence(
        work_id,
        version=frame_version,
        status="completed",
        frames_path=config.data_relpath(out_dir / "dense_visual.npz"),
        n_frames=n_frames,
        dim=FRAME_DIM,
        decode_s=0.01,
        encode_s=0.02,
        model_load_s=0.0,
        total_s=0.03,
    )
    return out_dir


def test_legacy_v043_bundle_version_fields_compatible(
    config: Config, catalog: Catalog
) -> None:
    """v0.4.3 真实数据 npz 逻辑版本 / manifest slug 版本经兼容读取可重建窗口层。"""

    work_id = _make_work(catalog, 17.0, index=0)
    frame_version = _frame_version_name()
    _make_legacy_frame_bundle(
        config, catalog, work_id, frame_version=frame_version, duration_s=17.0
    )
    result = build_visual_windows(catalog, config, frame_versions=[frame_version])
    assert result["created"] == 1
    assert result["failed"] == []
    windows = _expected_windows(17.0)
    weights = _load_weights(
        config, config.visual_window_plan_version, frame_version, work_id
    )
    assert len(weights) == len(windows)
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-5)

    # 真正不相关的版本仍必须失败（不能把损坏/错版本当旧格式接受）
    bad_version = compose_visual_version("fake-a@rev-a", 3.0, 32, "letterbox")
    _make_legacy_frame_bundle(
        config,
        catalog,
        work_id,
        frame_version=bad_version,
        duration_s=17.0,
        fps=3.0,
        npz_version="unrelated/encoder@deadbeef__fps3_s32_letterbox",
    )
    bad_result = build_visual_windows(
        catalog, config, frame_versions=[bad_version]
    )
    assert bad_result["created"] == 0
    assert bad_result["failed"]


def test_rebuild_coverage_weights_from_frames(
    config: Config, catalog: Catalog
) -> None:
    work_id = _make_work(catalog, 17.0, index=0)
    frame_version = _frame_version_name()
    frame_dir = _make_frame_bundle(
        config, catalog, work_id, frame_version=frame_version, duration_s=17.0
    )
    frame_mtime = (frame_dir / "dense_visual.npz").stat().st_mtime_ns
    plan = resolve_window_plan(config)
    assert plan.version == config.visual_window_plan_version

    result = build_visual_windows(
        catalog, config, frame_versions=[frame_version]
    )

    assert result["created"] == 1
    assert result["skipped"] == 0
    assert result["missing_frames"] == 0
    assert result["failed"] == []
    assert result["window_plan_version"] == plan.version
    assert result["total_windows"] == len(_expected_windows(17.0))
    assert (frame_dir / "dense_visual.npz").stat().st_mtime_ns == frame_mtime

    windows = _expected_windows(17.0)
    weights = _load_weights(config, plan.version, frame_version, work_id)
    expected = np.asarray(
        sampling_module.coverage_weights(windows, 17.0), dtype=np.float32
    )
    expected = expected / expected.sum()
    assert np.allclose(weights, expected, atol=1e-6)
    equal_weights = np.full(len(windows), 1.0 / len(windows), dtype=np.float32)
    assert float(np.max(np.abs(weights - equal_weights))) > 0.01
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-5)

    window_row = catalog.get_visual_window_sequence(
        work_id, frame_version=frame_version, window_plan_version=plan.version
    )
    assert window_row is not None and window_row["status"] == "completed"
    assert int(window_row["n_windows"]) == len(windows)
    assert int(window_row["dim"]) == FRAME_DIM
    assert window_row["windows_path"].startswith("visual_windows/")
    assert config.data_path(window_row["windows_path"]).is_file()
    assert config.data_path(window_row["window_std_path"]).is_file()
    assert config.data_path(window_row["window_weights_path"]).is_file()
    plan_row = catalog.get_visual_window_plan(plan.version)
    assert plan_row is not None
    assert plan_row["clip_seconds"] == pytest.approx(float(config.media.clip_seconds))
    assert plan_row["overlap_seconds"] == pytest.approx(
        float(config.sampling.overlap_seconds)
    )
    assert plan_row["weight_scheme"] == config.visual_dense.weight_scheme

    manifest = json.loads(
        (
            _bundle_dir(config, plan.version, frame_version, work_id)
            / WINDOW_MANIFEST_NAME
        ).read_text(encoding="utf-8")
    )
    assert manifest["window_plan_version"] == plan.version
    assert manifest["frame_version"] == frame_version
    assert manifest["weight_scheme"] == config.visual_dense.weight_scheme
    assert int(manifest["n_windows"]) == len(windows)
    assert float(manifest["weight_sum"]) == pytest.approx(1.0, abs=1e-5)

    # 单窗口（4s < 8s）时 coverage 与 equal 权重都为 1
    work_short = _make_work(catalog, 4.0, index=1)
    short_version = _frame_version_name(4.0)
    _make_frame_bundle(
        config, catalog, work_short, frame_version=short_version, duration_s=4.0, fps=4.0
    )
    short_result = build_visual_windows(
        catalog, config, frame_versions=[short_version]
    )
    assert short_result["created"] == 1
    short_weights = _load_weights(config, plan.version, short_version, work_short)
    assert short_weights.shape == (1,)
    assert float(short_weights[0]) == pytest.approx(1.0)


# ---------------------------------------------------------------- 计划隔离 / resume / force


def test_window_plan_isolation_and_force(
    config: Config, catalog: Catalog
) -> None:
    work_id = _make_work(catalog, 13.0, index=0)
    frame_version = _frame_version_name()
    _make_frame_bundle(
        config, catalog, work_id, frame_version=frame_version, duration_s=13.0
    )
    plan_a = resolve_window_plan(config)
    first = build_visual_windows(catalog, config, frame_versions=[frame_version])
    assert first["created"] == 1
    row_a = catalog.get_visual_window_sequence(
        work_id, frame_version=frame_version, window_plan_version=plan_a.version
    )
    assert row_a is not None and row_a["status"] == "completed"
    bundle_dir_a = _bundle_dir(config, plan_a.version, frame_version, work_id)
    manifest_a = json.loads(
        (bundle_dir_a / WINDOW_MANIFEST_NAME).read_text(encoding="utf-8")
    )

    # 同帧版本、不同窗口计划（5s/1s overlap）：不跳过，独立目录/DB 行
    plan_b = resolve_window_plan(
        config, window_seconds=5.0, overlap_seconds=1.0
    )
    assert plan_b.version == "win5s_ov1__coverage-v1"
    assert plan_b.version != plan_a.version
    second = build_visual_windows(
        catalog,
        config,
        window_seconds=5.0,
        overlap_seconds=1.0,
        frame_versions=[frame_version],
    )
    assert second["created"] == 1
    assert second["skipped"] == 0
    row_b = catalog.get_visual_window_sequence(
        work_id, frame_version=frame_version, window_plan_version=plan_b.version
    )
    assert row_b is not None and row_b["status"] == "completed"
    assert _bundle_dir(config, plan_b.version, frame_version, work_id).is_dir()
    assert int(row_b["n_windows"]) == len(
        _expected_windows(13.0, clip=5.0, overlap=1.0)
    )
    # 旧计划行与产物不被覆盖
    row_a_after = catalog.get_visual_window_sequence(
        work_id, frame_version=frame_version, window_plan_version=plan_a.version
    )
    assert row_a_after is not None and row_a_after["status"] == "completed"
    assert manifest_a == json.loads(
        (bundle_dir_a / WINDOW_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert {row["version"] for row in catalog.list_visual_window_plans()} == {
        plan_a.version,
        plan_b.version,
    }

    # resume：同一计划再跑只跳过
    resume = build_visual_windows(
        catalog, config, frame_versions=[frame_version]
    )
    assert resume["created"] == 0
    assert resume["skipped"] == 1
    # force：强制重建
    before_mtime = (bundle_dir_a / WINDOW_MANIFEST_NAME).stat().st_mtime_ns
    time.sleep(0.01)
    forced = build_visual_windows(
        catalog, config, frame_versions=[frame_version], force=True
    )
    assert forced["created"] == 1
    after_mtime = (bundle_dir_a / WINDOW_MANIFEST_NAME).stat().st_mtime_ns
    assert after_mtime >= before_mtime
    assert forced["total_windows"] == len(_expected_windows(13.0))


# ---------------------------------------------------------------- CLI


def test_cli_equal_scheme_and_flags(
    config: Config, catalog: Catalog, capsys: pytest.CaptureFixture[str]
) -> None:
    work_id = _make_work(catalog, 17.0, index=0)
    frame_version = _frame_version_name()
    _make_frame_bundle(
        config, catalog, work_id, frame_version=frame_version, duration_s=17.0
    )

    exit_code = build_tool.main(
        [
            "--config",
            str(config.path),
            "--frame-version",
            frame_version,
            "--window-seconds",
            "5",
            "--overlap-seconds",
            "0",
            "--weight-scheme",
            "equal",
            "--quiet",
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "win5s_ov0__equal" in output
    assert "完成 1 个窗口序列" in output
    plan_version = "win5s_ov0__equal"
    weights = _load_weights(config, plan_version, frame_version, work_id)
    windows = _expected_windows(17.0, clip=5.0)
    assert np.allclose(weights, np.full(len(windows), 1.0 / len(windows)), atol=1e-6)
    row = catalog.get_visual_window_sequence(
        work_id, frame_version=frame_version, window_plan_version=plan_version
    )
    assert row is not None and row["status"] == "completed"
    assert catalog.get_visual_window_plan(plan_version)["weight_scheme"] == "equal"

    # dry-run：不写 DB/文件
    exit_code = build_tool.main(
        [
            "--config",
            str(config.path),
            "--frame-version",
            frame_version,
            "--window-seconds",
            "6",
            "--weight-scheme",
            "equal",
            "--dry-run",
            "--quiet",
        ]
    )
    assert exit_code == 0
    assert "dry-run" in capsys.readouterr().out
    assert catalog.get_visual_window_plan("win6s_ov0__equal") is None
    assert not _bundle_dir(
        config, "win6s_ov0__equal", frame_version, work_id
    ).exists()


def test_cli_fps_limit_and_missing_frames(
    config: Config, catalog: Catalog, capsys: pytest.CaptureFixture[str]
) -> None:
    version_2fps = _frame_version_name(2.0)
    version_4fps = _frame_version_name(4.0)
    work_a = _make_work(catalog, 13.0, index=0)
    work_b = _make_work(catalog, 13.0, index=1)
    _make_frame_bundle(
        config, catalog, work_a, frame_version=version_2fps, duration_s=13.0, fps=2.0
    )
    _make_frame_bundle(
        config, catalog, work_b, frame_version=version_4fps, duration_s=13.0, fps=4.0
    )
    # work_b 没有 2fps 帧版本 -> 缺帧；work_a 没有 4fps 帧版本 -> 缺帧
    result = build_visual_windows(
        catalog, config, fps=4.0, limit=2, dry_run=True
    )
    assert result["frame_versions"] == [version_4fps]
    assert result["works"] == 2
    assert result["would_process"] == 1

    exit_code = build_tool.main(
        [
            "--config",
            str(config.path),
            "--fps",
            "4",
            "--limit",
            "1",
            "--dry-run",
            "--quiet",
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert version_4fps in output
    assert version_2fps not in output

    # 未登记帧版本：报告并跳过（不写非法外键）
    unknown = "unregistered-frame-version"
    result = build_visual_windows(
        catalog, config, frame_versions=[unknown], dry_run=False
    )
    assert result["created"] == 0
    assert result["failed"]
    assert result["failed"][0]["frame_version"] == unknown
    assert catalog.get_visual_window_sequence(
        work_a, frame_version=unknown, window_plan_version=config.visual_window_plan_version
    ) is None


def test_all_flag_discovers_frame_dirs(
    config: Config, catalog: Catalog, capsys: pytest.CaptureFixture[str]
) -> None:
    work_id = _make_work(catalog, 13.0, index=0)
    frame_version = _frame_version_name()
    _make_frame_bundle(
        config, catalog, work_id, frame_version=frame_version, duration_s=13.0
    )
    exit_code = build_tool.main(
        ["--config", str(config.path), "--all", "--dry-run", "--quiet"]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert frame_version in output
    assert "dry-run" in output


# ---------------------------------------------------------------- 边界


def test_pending_and_duplicate_works_skipped(
    config: Config, catalog: Catalog
) -> None:
    work_id = _make_work(catalog, 13.0, index=0)
    frame_version = _frame_version_name()
    _make_frame_bundle(
        config, catalog, work_id, frame_version=frame_version, duration_s=13.0
    )
    pending, _ = catalog.upsert_local_work("tool_pending.mp4")
    catalog.update_work_metadata(pending, duration_s=13.0)
    duplicate, _ = catalog.upsert_local_work("tool_duplicate.mp4")
    catalog.update_work_metadata(duplicate, duration_s=13.0)
    catalog.begin_attempt(duplicate, "encoding")
    catalog.complete(duplicate)
    catalog.link_works(work_id, duplicate, method="test", confidence=1.0)

    result = build_visual_windows(catalog, config, frame_versions=[frame_version])
    assert result["works"] == 1
    assert result["created"] == 1
    plan_version = config.visual_window_plan_version
    assert catalog.get_visual_window_sequence(
        pending, frame_version=frame_version, window_plan_version=plan_version
    ) is None
    assert catalog.get_visual_window_sequence(
        duplicate, frame_version=frame_version, window_plan_version=plan_version
    ) is None


def test_rebuild_does_not_touch_frame_bundle_or_sources(
    config: Config, catalog: Catalog
) -> None:
    """重建只读帧 npz；不写 data/visual_dense，也不动 data/sources。"""

    work_id = _make_work(catalog, 13.0, index=0)
    frame_version = _frame_version_name()
    frame_dir = _make_frame_bundle(
        config, catalog, work_id, frame_version=frame_version, duration_s=13.0
    )
    before = {
        path.relative_to(config.paths.visual_dense): path.stat().st_mtime_ns
        for path in frame_dir.rglob("*")
        if path.is_file()
    }
    source_dir = config.paths.sources / "BV1fakeSource"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "sentinel.bin").write_bytes(b"keep")
    build_visual_windows(catalog, config, frame_versions=[frame_version])
    after = {
        path.relative_to(config.paths.visual_dense): path.stat().st_mtime_ns
        for path in frame_dir.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert (source_dir / "sentinel.bin").read_bytes() == b"keep"
    bundle = _bundle_dir(
        config, config.visual_window_plan_version, frame_version, work_id
    )
    windows = np.load(bundle / WINDOWS_NAME)
    std = np.load(bundle / WINDOW_STD_NAME)
    assert windows.shape == std.shape == (len(_expected_windows(13.0)), FRAME_DIM)
