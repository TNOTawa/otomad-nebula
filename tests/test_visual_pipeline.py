"""v0.4.3.1 密集视觉流水线（nebula.visual_pipeline）离线测试。

全部使用 fake 编码器 / fake 提取器 / fake 下载器与合成视频，不加载模型、不联网：

- 帧编码层（visual_sequences，仅 frames_path）与窗口计划层（visual_window_sequences，
  data/visual_windows/<window_plan_version>/<frame_version>/<work_id>/）分离；
- 双编码器各跑一次、canonical 去重、pending 作品不处理、Bilibili 源缓存只下载一次；
- coverage-v1 窗口权重（非整 8s 视频）与 equal 方案不同、和为 1；
- 断点续跑：帧已完成只重建缺失窗口（不重新编码）；窗口计划不同不跳过；
- 30fps 子集选择确定性、短/中/长时长桶覆盖；
- status 摘要（帧层 + 窗口层状态/存储）与 clean-temp 清理时机；dry-run 零写入。
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from nebula import sampling as sampling_module
from nebula.config import Config, VisualEncoderConfig
from nebula.db import Catalog
from nebula.dense_visual import (
    DENSE_VISUAL_MANIFEST_NAME,
    DenseVisualError,
    DenseVisualFeatures,
    compress_version,
    visual_version_slug,
)
from nebula.media import probe_media
from nebula.sampling import plan_full_coverage
from nebula.visual_pipeline import (
    WINDOW_MANIFEST_NAME,
    WINDOW_STD_NAME,
    WINDOW_WEIGHTS_NAME,
    WINDOWS_NAME,
    clean_completed_temp_dirs,
    compose_visual_version,
    run_visual_pipeline,
    select_canonical_completed_works,
    select_thirty_fps_works,
    visual_status,
    visual_version_specs,
    visual_windows_dir,
    window_plan_root,
)
from tests.fixtures.video_factory import build_video_library, media_tools_available

BVID = "BV1demo00001"
FAKE_DIMS = {"fake-a": 8, "fake-b": 6}


# ---------------------------------------------------------------- 测试替身


class FakeEncoder:
    """确定性 fake 视觉编码器（不依赖 torch）：输出按行 L2 归一化。"""

    def __init__(self, model: str, revision: str, dim: int) -> None:
        self.version = f"{model}@{revision}"
        self.revision = revision
        self.dim = dim
        self.model_load_s = 0.01
        self.closed = 0
        self.calls = 0

    def encode(self, frames: np.ndarray) -> np.ndarray:
        self.calls += 1
        array = np.asarray(frames, dtype=np.float32)
        if array.ndim != 4:
            raise DenseVisualError(f"fake encoder 收到非法帧批次: {array.shape}")
        rows = int(array.shape[0])
        means = array.reshape(rows, -1).mean(axis=1) / 255.0
        features = np.tile(np.linspace(0.1, 1.0, self.dim, dtype=np.float32), (rows, 1))
        features[:, 0] = means
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return (features / np.maximum(norms, 1e-9)).astype(np.float32)

    def close(self) -> None:
        self.closed += 1


class FakeExtractor:
    """fake 提取器：按视频 probe 时长生成确定性帧序列（可注入失败条件）。"""

    def __init__(self, config: Config, *, fail=None) -> None:
        self.config = config
        self.fail = fail
        self.calls: list[dict] = []

    def __call__(self, source, encoder, *, fps, size, batch_size, ffmpeg, preprocess, timeout_s):
        record = {
            "source": Path(source),
            "encoder": str(encoder.version),
            "fps": float(fps),
            "size": int(size),
            "preprocess": str(preprocess),
            "batch_size": int(batch_size),
            "index": len(self.calls),
        }
        self.calls.append(record)
        if self.fail is not None and self.fail(record):
            raise DenseVisualError("模拟视觉编码失败")
        info = probe_media(
            Path(source),
            ffprobe=self.config.media.ffprobe,
            timeout_s=self.config.media.probe_timeout_seconds,
        )
        n_frames = max(1, int(round(info.duration_s * float(fps))))
        dim = int(encoder.dim)
        base = np.arange(1, dim + 1, dtype=np.float32)
        per_frame = np.tile(base, (n_frames, 1))
        per_frame[:, 0] += np.arange(n_frames, dtype=np.float32)
        norms = np.linalg.norm(per_frame, axis=1, keepdims=True)
        per_frame = (per_frame / np.maximum(norms, 1e-9)).astype(np.float32)
        return DenseVisualFeatures(
            per_frame=per_frame,
            fps=float(fps),
            duration_s=float(info.duration_s),
            n_frames=n_frames,
            dim=dim,
            version=compose_visual_version(
                str(encoder.version), float(fps), int(size), str(preprocess)
            ),
            decode_s=0.1,
            encode_s=0.2,
            model_load_s=float(getattr(encoder, "model_load_s", 0.0)),
            encoder=str(encoder.version),
            revision=getattr(encoder, "revision", None),
            size=int(size),
            preprocess=str(preprocess),
            created_at="2026-01-01T00:00:00Z",
        )


def _make_factory(dims: dict[str, int] | None = None):
    created: list[FakeEncoder] = []
    dims = dict(FAKE_DIMS if dims is None else dims)

    def factory(model: str, revision: str) -> FakeEncoder:
        encoder = FakeEncoder(model, revision, dims.get(model, 8))
        created.append(encoder)
        return encoder

    factory.created = created  # type: ignore[attr-defined]
    return factory


def _make_downloader(video_path: Path):
    calls: list[dict] = []

    def downloader(bvid: str, *, part: int, out_dir: Path, download) -> Path:
        calls.append({"bvid": bvid, "part": part, "out_dir": Path(out_dir)})
        target_dir = Path(out_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{bvid}_p{part}.mp4"
        shutil.copy2(video_path, target)
        return target

    downloader.calls = calls  # type: ignore[attr-defined]
    return downloader


# ---------------------------------------------------------------- 夹具与工具


@pytest.fixture(scope="module")
def videos() -> dict[str, Path]:
    if not media_tools_available():
        pytest.skip("缺少 ffmpeg/ffprobe")
    return build_video_library(
        names=["short_4s.mp4", "mid_12s.mp4", "odd_17s.mp4", "long_40s.mp4"]
    )


def _visual_config(
    config: Config,
    *,
    subset_count: int = 0,
    fps: float = 2.0,
    subset_fps: float = 4.0,
    size: int = 32,
    encoders: tuple[tuple[str, str], ...] = (("fake-a", "rev-a"), ("fake-b", "rev-b")),
) -> Config:
    dense = dataclasses.replace(
        config.visual_dense,
        fps=fps,
        subset_fps=subset_fps,
        subset_count=subset_count,
        size=size,
        preprocess="letterbox",
        batch_size=4,
        encoders=tuple(
            VisualEncoderConfig(model=model, revision=revision)
            for model, revision in encoders
        ),
    )
    return dataclasses.replace(config, visual_dense=dense)


def _duration_of(path: Path, config: Config) -> float:
    info = probe_media(
        path,
        ffprobe=config.media.ffprobe,
        timeout_s=config.media.probe_timeout_seconds,
    )
    return float(info.duration_s)


def _make_local_work(
    catalog: Catalog,
    config: Config,
    videos: dict[str, Path],
    rel_path: str,
    video_name: str = "short_4s.mp4",
    *,
    completed: bool = True,
    duration_s: float | None = None,
) -> int:
    target = config.paths.input_local / Path(rel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(videos[video_name], target)
    work_id, _ = catalog.upsert_local_work(rel_path)
    actual_duration = (
        duration_s if duration_s is not None else _duration_of(target, config)
    )
    catalog.update_work_metadata(work_id, duration_s=float(actual_duration))
    if completed:
        catalog.begin_attempt(work_id, "encoding")
        catalog.complete(work_id)
    return work_id


def _make_bilibili_work(catalog: Catalog, duration_s: float = 4.0) -> int:
    work_id, _ = catalog.upsert_bilibili_work(BVID, part=1, title="fake bili")
    catalog.update_work_metadata(work_id, duration_s=float(duration_s))
    catalog.begin_attempt(work_id, "encoding")
    catalog.complete(work_id)
    return work_id


def _frame_sequence(catalog: Catalog, work_id: int, version: str):
    return catalog.get_visual_sequence(work_id, version=version)


def _window_sequence(catalog: Catalog, work_id: int, frame_version: str, plan_version: str):
    return catalog.get_visual_window_sequence(
        work_id, frame_version=frame_version, window_plan_version=plan_version
    )


def _expected_window_count(
    config: Config,
    duration_s: float,
    *,
    clip_seconds: float | None = None,
    overlap_seconds: float | None = None,
) -> int:
    plans = plan_full_coverage(
        duration_s,
        clip_seconds=config.media.clip_seconds if clip_seconds is None else clip_seconds,
        overlap_seconds=(
            config.sampling.overlap_seconds
            if overlap_seconds is None
            else overlap_seconds
        ),
        min_duration_seconds=config.media.min_duration_seconds,
    )
    return len(plans)


def _windows_bundle_dir(
    config: Config,
    *,
    plan_version: str,
    frame_version: str,
    work_id: int,
) -> Path:
    return visual_windows_dir(config, plan_version, frame_version, work_id)


def _load_window_bundle(config: Config, plan_version: str, frame_version: str, work_id: int):
    bundle_dir = _windows_bundle_dir(
        config, plan_version=plan_version, frame_version=frame_version, work_id=work_id
    )
    windows = np.load(bundle_dir / WINDOWS_NAME, allow_pickle=False)
    std = np.load(bundle_dir / WINDOW_STD_NAME, allow_pickle=False)
    weights = np.load(bundle_dir / WINDOW_WEIGHTS_NAME, allow_pickle=False)
    manifest = json.loads(
        (bundle_dir / WINDOW_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    return windows, std, weights, manifest


# ---------------------------------------------------------------- 版本契约


def test_visual_version_specs(config: Config) -> None:
    cfg = _visual_config(config, subset_count=16)
    specs = visual_version_specs(cfg, include_subset=True)
    assert len(specs) == 4
    main = [spec for spec in specs if not spec.subset]
    subset = [spec for spec in specs if spec.subset]
    assert [spec.version for spec in main] == [
        compose_visual_version("fake-a@rev-a", 2.0, 32, "letterbox"),
        compose_visual_version("fake-b@rev-b", 2.0, 32, "letterbox"),
    ]
    assert [spec.fps for spec in subset] == [4.0, 4.0]
    assert all(spec.preprocess == "letterbox" for spec in specs)
    assert visual_version_specs(cfg, include_subset=False) == main
    assert cfg.visual_window_plan_version == "win8s_ov0__coverage-v1"


# ---------------------------------------------------------------- 帧层 + 窗口层


def test_run_two_encoders_canonical_only_and_resume(
    config: Config, catalog: Catalog, videos: dict[str, Path]
) -> None:
    cfg = _visual_config(config, subset_count=0)
    work_a = _make_local_work(catalog, cfg, videos, "a.mp4", "mid_12s.mp4")
    work_b = _make_local_work(catalog, cfg, videos, "b.mp4", "short_4s.mp4")
    pending = catalog.upsert_local_work("pending.mp4")[0]
    catalog.update_work_metadata(pending, duration_s=4.0)
    duplicate = _make_local_work(catalog, cfg, videos, "dup.mp4", "short_4s.mp4")
    catalog.link_works(work_a, duplicate, method="test", confidence=1.0)
    factory = _make_factory()
    extractor = FakeExtractor(cfg)
    plan_version = cfg.visual_window_plan_version

    summary = run_visual_pipeline(
        catalog, cfg, encoder_factory=factory, extractor=extractor
    )

    assert summary.selected_works == 2
    assert summary.tasks_total == 4
    assert summary.tasks_completed == 4
    assert summary.tasks_failed == 0
    assert summary.tasks_skipped == 0
    assert summary.windows_total == 4
    assert summary.windows_completed == 4
    assert summary.windows_failed == 0
    assert summary.window_plan_version == plan_version
    assert len(extractor.calls) == 4
    assert len(factory.created) == 2
    assert all(encoder.closed == 1 for encoder in factory.created)

    # 帧版本 / 窗口计划都已登记
    assert [entry["version"] for entry in catalog.list_visual_versions()] == list(
        summary.versions
    )
    plan_row = catalog.get_visual_window_plan(plan_version)
    assert plan_row is not None
    assert plan_row["clip_seconds"] == pytest.approx(float(cfg.media.clip_seconds))
    assert plan_row["overlap_seconds"] == pytest.approx(float(cfg.sampling.overlap_seconds))
    assert plan_row["weight_scheme"] == cfg.visual_dense.weight_scheme

    for work_id, rel in ((work_a, "a.mp4"), (work_b, "b.mp4")):
        duration = _duration_of(cfg.paths.input_local / rel, cfg)
        expected_windows = _expected_window_count(cfg, duration)
        for frame_version in summary.versions:
            frame_row = _frame_sequence(catalog, work_id, frame_version)
            assert frame_row is not None
            assert frame_row["status"] == "completed"
            assert frame_row["frames_path"]
            assert frame_row["windows_path"] is None  # 帧层不再写窗口路径
            frame_dir = cfg.paths.visual_dense / frame_version / f"{work_id:06d}"
            assert (frame_dir / "dense_visual.npz").is_file()
            assert (frame_dir / DENSE_VISUAL_MANIFEST_NAME).is_file()
            assert not (frame_dir / WINDOWS_NAME).exists()  # 窗口不进帧目录

            window_row = _window_sequence(catalog, work_id, frame_version, plan_version)
            assert window_row is not None
            assert window_row["status"] == "completed"
            assert window_row["n_windows"] == expected_windows
            assert window_row["dim"] in FAKE_DIMS.values()
            bundle_dir = _windows_bundle_dir(
                cfg,
                plan_version=plan_version,
                frame_version=frame_version,
                work_id=work_id,
            )
            windows = np.load(bundle_dir / WINDOWS_NAME)
            std = np.load(bundle_dir / WINDOW_STD_NAME)
            weights = np.load(bundle_dir / WINDOW_WEIGHTS_NAME)
            manifest = json.loads(
                (bundle_dir / WINDOW_MANIFEST_NAME).read_text(encoding="utf-8")
            )
            assert windows.shape == (expected_windows, window_row["dim"])
            assert std.shape == windows.shape
            assert weights.shape == (expected_windows,)
            assert float(weights.sum()) == pytest.approx(1.0, abs=1e-5)
            assert np.all(weights >= 0)
            assert manifest["frame_version"] == frame_version
            assert manifest["window_plan_version"] == plan_version
            assert manifest["weight_scheme"] == cfg.visual_dense.weight_scheme
            assert int(manifest["n_windows"]) == expected_windows
            assert len(manifest["window_times"]) == expected_windows
            assert float(manifest["window_times"][-1][1]) == pytest.approx(
                duration, abs=0.05
            )
            assert config.data_path(window_row["windows_path"]).is_file()
            assert config.data_path(window_row["window_std_path"]).is_file()
            assert config.data_path(window_row["window_weights_path"]).is_file()

    assert _frame_sequence(catalog, pending, summary.versions[0]) is None
    assert _frame_sequence(catalog, duplicate, summary.versions[0]) is None

    calls_after_first = len(extractor.calls)
    second = run_visual_pipeline(
        catalog, cfg, encoder_factory=factory, extractor=extractor
    )
    assert second.tasks_completed == 0
    assert second.tasks_skipped == 4
    assert second.windows_completed == 0
    assert second.windows_skipped == 4
    assert len(extractor.calls) == calls_after_first


def test_coverage_weights_differ_from_equal_for_non_multiple_8s(
    config: Config, catalog: Catalog, videos: dict[str, Path]
) -> None:
    """17s 视频（非 8s 整倍）coverage-v1 权重必须区别于等权且和为 1。"""

    cfg = _visual_config(config, subset_count=0, encoders=(("fake-a", "rev-a"),))
    work_id = _make_local_work(catalog, cfg, videos, "odd.mp4", "odd_17s.mp4")
    run_visual_pipeline(
        catalog, cfg, encoder_factory=_make_factory(), extractor=FakeExtractor(cfg)
    )
    frame_version = visual_version_specs(cfg, include_subset=False)[0].version
    duration = _duration_of(cfg.paths.input_local / "odd.mp4", cfg)
    windows = [
        (float(item.start_s), float(item.end_s))
        for item in plan_full_coverage(
            duration,
            clip_seconds=cfg.media.clip_seconds,
            overlap_seconds=cfg.sampling.overlap_seconds,
            min_duration_seconds=cfg.media.min_duration_seconds,
        )
    ]
    _, _, weights, manifest = _load_window_bundle(
        cfg, cfg.visual_window_plan_version, frame_version, work_id
    )
    public_weights = np.asarray(
        sampling_module.coverage_weights(windows, duration), dtype=np.float32
    )
    public_weights = public_weights / public_weights.sum()
    assert np.allclose(weights, public_weights, atol=1e-6)
    equal_weights = np.full(len(windows), 1.0 / len(windows), dtype=np.float32)
    assert float(np.max(np.abs(weights - equal_weights))) > 0.01
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-5)
    assert manifest["weight_sum"] == pytest.approx(1.0, abs=1e-5)

    # 同一帧版本切换 equal 权重方案：只重建窗口层，不重新编码帧
    cfg_equal = dataclasses.replace(
        cfg,
        visual_dense=dataclasses.replace(cfg.visual_dense, weight_scheme="equal"),
    )
    factory = _make_factory()
    extractor = FakeExtractor(cfg_equal)
    extractor.calls = []
    result = run_visual_pipeline(
        catalog, cfg_equal, encoder_factory=factory, extractor=extractor
    )
    assert extractor.calls == []
    assert result.tasks_skipped == 1
    assert result.windows_completed == 1
    equal_plan_version = cfg_equal.visual_window_plan_version
    _, _, equal_weights, _ = _load_window_bundle(
        cfg_equal, equal_plan_version, frame_version, work_id
    )
    assert np.allclose(
        equal_weights, np.full(len(windows), 1.0 / len(windows)), atol=1e-6
    )
    # 旧计划行仍独立存在
    old_row = _window_sequence(
        catalog, work_id, frame_version, cfg.visual_window_plan_version
    )
    assert old_row is not None and old_row["status"] == "completed"
    new_row = _window_sequence(catalog, work_id, frame_version, equal_plan_version)
    assert new_row is not None and new_row["status"] == "completed"


def test_resume_rebuilds_missing_windows_without_reencoding(
    config: Config, catalog: Catalog, videos: dict[str, Path]
) -> None:
    cfg = _visual_config(config, subset_count=0, encoders=(("fake-a", "rev-a"),))
    work_id = _make_local_work(catalog, cfg, videos, "rebuild.mp4", "short_4s.mp4")
    factory = _make_factory()
    extractor = FakeExtractor(cfg)
    run_visual_pipeline(catalog, cfg, encoder_factory=factory, extractor=extractor)
    frame_version = visual_version_specs(cfg, include_subset=False)[0].version
    plan_version = cfg.visual_window_plan_version
    calls_after = len(extractor.calls)
    frame_dir = cfg.paths.visual_dense / frame_version / f"{work_id:06d}"
    frame_npz = frame_dir / "dense_visual.npz"
    mtime_before = frame_npz.stat().st_mtime_ns

    bundle_dir = _windows_bundle_dir(
        cfg, plan_version=plan_version, frame_version=frame_version, work_id=work_id
    )
    shutil.rmtree(bundle_dir)
    catalog.upsert_visual_window_sequence(
        work_id,
        frame_version=frame_version,
        window_plan_version=plan_version,
        status="failed",
        error="模拟窗口文件丢失",
    )

    summary = run_visual_pipeline(
        catalog, cfg, encoder_factory=factory, extractor=extractor
    )
    assert summary.tasks_skipped == 1
    assert summary.tasks_completed == 0
    assert summary.windows_completed == 1
    assert len(extractor.calls) == calls_after  # 帧不重复编码
    assert frame_npz.stat().st_mtime_ns == mtime_before
    row = _window_sequence(catalog, work_id, frame_version, plan_version)
    assert row is not None and row["status"] == "completed"
    assert config.data_path(row["windows_path"]).is_file()


def test_window_plan_isolation_in_pipeline(
    config: Config, catalog: Catalog, videos: dict[str, Path]
) -> None:
    cfg = _visual_config(config, subset_count=0, encoders=(("fake-a", "rev-a"),))
    work_id = _make_local_work(catalog, cfg, videos, "plan.mp4", "mid_12s.mp4")
    factory = _make_factory()
    extractor = FakeExtractor(cfg)
    first = run_visual_pipeline(
        catalog, cfg, encoder_factory=factory, extractor=extractor
    )
    frame_version = visual_version_specs(cfg, include_subset=False)[0].version
    plan_a = cfg.visual_window_plan_version
    calls_after_python = len(extractor.calls)
    row_a = _window_sequence(catalog, work_id, frame_version, plan_a)
    assert row_a is not None and row_a["status"] == "completed"
    n_a = int(row_a["n_windows"])

    cfg_b = dataclasses.replace(
        cfg, media=dataclasses.replace(cfg.media, clip_seconds=5.0)
    )
    assert cfg_b.visual_window_plan_version == "win5s_ov0__coverage-v1"
    plan_b = cfg_b.visual_window_plan_version
    second = run_visual_pipeline(
        catalog, cfg_b, encoder_factory=factory, extractor=extractor
    )
    assert len(extractor.calls) == calls_after_python  # 帧未重新编码
    assert second.tasks_skipped == 1
    assert second.windows_completed == 1
    row_b = _window_sequence(catalog, work_id, frame_version, plan_b)
    assert row_b is not None and row_b["status"] == "completed"
    assert int(row_b["n_windows"]) != n_a
    row_a_after = _window_sequence(catalog, work_id, frame_version, plan_a)
    assert row_a_after is not None and row_a_after["status"] == "completed"
    assert plan_a in {row["version"] for row in catalog.list_visual_window_plans()}
    assert plan_b in {row["version"] for row in catalog.list_visual_window_plans()}
    assert first.window_plan_version == plan_a


def test_bilibili_source_cache_download_once(
    config: Config, catalog: Catalog, videos: dict[str, Path]
) -> None:
    cfg = _visual_config(config, subset_count=0)
    work_id = _make_bilibili_work(catalog, duration_s=4.0)
    downloader = _make_downloader(videos["short_4s.mp4"])
    cache_dir = cfg.paths.sources / BVID
    assert not cache_dir.exists() or not any(cache_dir.iterdir())
    plan_version = cfg.visual_window_plan_version

    first = run_visual_pipeline(
        catalog,
        cfg,
        encoder_factory=_make_factory(),
        extractor=FakeExtractor(cfg),
        downloader=downloader,
    )
    assert first.tasks_completed == 2
    assert first.downloaded_works == 1
    assert len(downloader.calls) == 1
    cached_files = sorted(cache_dir.glob(f"{BVID}_p1.*"))
    assert len(cached_files) == 1
    cached = cached_files[0]
    assert cached.stat().st_size > 0
    assert not (cfg.paths.temp / f"work_{work_id:06d}").exists()

    second = run_visual_pipeline(
        catalog,
        cfg,
        encoder_factory=_make_factory(),
        extractor=FakeExtractor(cfg),
        downloader=downloader,
    )
    assert second.tasks_skipped == 2
    assert second.downloaded_works == 0
    assert len(downloader.calls) == 1  # 缓存命中，不再下载
    assert cached.is_file()  # 缓存保留
    for frame_version in visual_version_specs(cfg, include_subset=False):
        row = _window_sequence(catalog, work_id, frame_version.version, plan_version)
        assert row is not None and row["status"] == "completed"


def test_resume_after_encoder_failure_and_temp_cleanup(
    config: Config, catalog: Catalog, videos: dict[str, Path]
) -> None:
    cfg = _visual_config(config, subset_count=0)
    work_id = _make_bilibili_work(catalog, duration_s=4.0)
    downloader = _make_downloader(videos["short_4s.mp4"])
    failing = FakeExtractor(
        cfg, fail=lambda record: record["index"] == 0 and record["fps"] == 2.0
    )

    first = run_visual_pipeline(
        catalog,
        cfg,
        encoder_factory=_make_factory(),
        extractor=failing,
        downloader=downloader,
    )
    assert first.tasks_total == 2
    assert first.tasks_completed == 1
    assert first.tasks_failed == 1
    assert first.downloaded_works == 1
    assert first.cleaned_temp_dirs == 0
    assert first.windows_completed == 1  # 成功帧任务对应窗口层
    assert len(downloader.calls) == 1
    temp_root = cfg.paths.temp / f"work_{work_id:06d}"
    assert temp_root.is_dir()  # 有失败帧任务时保留 temp
    versions = visual_version_specs(cfg, include_subset=False)
    statuses = sorted(
        str(_frame_sequence(catalog, work_id, spec.version)["status"])
        for spec in versions
    )
    assert statuses == ["completed", "failed"]

    retry_extractor = FakeExtractor(cfg)
    second = run_visual_pipeline(
        catalog,
        cfg,
        encoder_factory=_make_factory(),
        extractor=retry_extractor,
        downloader=downloader,
    )
    assert second.tasks_completed == 1
    assert second.tasks_skipped == 1
    assert second.tasks_failed == 0
    assert second.downloaded_works == 0
    assert second.cleaned_temp_dirs == 1
    assert second.windows_completed == 1
    assert second.windows_skipped == 1
    assert len(downloader.calls) == 1  # 复用 data/sources 缓存
    assert not temp_root.exists()

    plan_version = cfg.visual_window_plan_version
    for spec in versions:
        frame_row = _frame_sequence(catalog, work_id, spec.version)
        assert frame_row["status"] == "completed"
        assert frame_row["decode_s"] == pytest.approx(0.1)
        assert frame_row["encode_s"] == pytest.approx(0.2)
        window_row = _window_sequence(catalog, work_id, spec.version, plan_version)
        assert window_row is not None and window_row["status"] == "completed"


# ---------------------------------------------------------------- 子集 / 选择


def test_subset_tasks_and_include_subset_flag(
    config: Config, catalog: Catalog, videos: dict[str, Path]
) -> None:
    cfg = _visual_config(config, subset_count=2, fps=2.0, subset_fps=4.0)
    names = ["short_4s.mp4", "mid_12s.mp4", "odd_17s.mp4", "long_40s.mp4"]
    work_ids = [
        _make_local_work(catalog, cfg, videos, f"subset_{index}.mp4", name)
        for index, name in enumerate(names)
    ]
    expected_subset = select_thirty_fps_works(catalog, cfg, 2)
    assert len(expected_subset) == 2
    plan_version = cfg.visual_window_plan_version

    factory = _make_factory()
    extractor = FakeExtractor(cfg)
    summary = run_visual_pipeline(
        catalog, cfg, encoder_factory=factory, extractor=extractor
    )
    main_versions = [
        spec.version for spec in visual_version_specs(cfg, include_subset=False)
    ]
    subset_versions = [
        spec.version
        for spec in visual_version_specs(cfg, include_subset=True)
        if spec.subset
    ]
    assert len(main_versions) == 2 and len(subset_versions) == 2
    frame_tasks = len(work_ids) * 2 + len(expected_subset) * 2
    assert summary.tasks_total == frame_tasks
    assert summary.tasks_completed == frame_tasks
    assert summary.windows_completed == frame_tasks
    assert summary.subset_works == 2
    for work_id in work_ids:
        for version in main_versions:
            assert _frame_sequence(catalog, work_id, version)["status"] == "completed"
            assert (
                _window_sequence(catalog, work_id, version, plan_version)["status"]
                == "completed"
            )
        for version in subset_versions:
            frame_row = _frame_sequence(catalog, work_id, version)
            window_row = _window_sequence(catalog, work_id, version, plan_version)
            if work_id in expected_subset:
                assert frame_row is not None and frame_row["status"] == "completed"
                assert window_row is not None and window_row["status"] == "completed"
            else:
                assert frame_row is None
                assert window_row is None

    calls_after_first = len(extractor.calls)
    second = run_visual_pipeline(
        catalog,
        cfg,
        include_subset=False,
        encoder_factory=_make_factory(),
        extractor=extractor,
    )
    assert second.tasks_completed == 0
    assert second.tasks_skipped == len(work_ids) * 2
    assert second.windows_skipped == len(work_ids) * 2
    assert len(extractor.calls) == calls_after_first


def _write_full_sequence_files(
    config: Config,
    catalog: Catalog,
    work_id: int,
    *,
    dynamic: bool,
    duration_s: float,
) -> None:
    """写入用于子集选择的 clip_joint 与 event_features（动态/静态两种模式）。"""

    catalog.update_work_metadata(work_id, duration_s=float(duration_s))
    version = config.full_sequence_version
    work_dir = config.paths.embeddings / version / f"{work_id:06d}"
    work_dir.mkdir(parents=True, exist_ok=True)
    if dynamic:
        clip = np.array([[1.0, 0.0], [0.0, 1.0]] * 4, dtype=np.float32)
        events = np.array([[1.0] * 32, [-1.0] * 32] * 4, dtype=np.float32)
    else:
        clip = np.tile(np.array([[0.5, 0.5]], dtype=np.float32), (8, 1))
        events = np.zeros((8, 32), dtype=np.float32)
    np.save(work_dir / "clip_joint.npy", clip)
    np.save(work_dir / "event_features.npy", events)


def test_select_thirty_fps_works_deterministic_buckets_and_dynamics(
    config: Config, catalog: Catalog
) -> None:
    cfg = _visual_config(config, subset_count=9)
    durations = [30.0, 60.0, 90.0, 120.0, 180.0, 240.0, 300.0, 420.0, 600.0]
    work_ids: list[int] = []
    for index, duration in enumerate(durations):
        work_id, _ = catalog.upsert_local_work(f"select_{index}.mp4")
        catalog.begin_attempt(work_id, "encoding")
        catalog.complete(work_id)
        dynamic = index in (0, 5, 8)
        _write_full_sequence_files(
            cfg, catalog, work_id, dynamic=dynamic, duration_s=duration
        )
        work_ids.append(work_id)

    first = select_thirty_fps_works(catalog, cfg, 3)
    second = select_thirty_fps_works(catalog, cfg, 3)
    assert first == second == sorted([work_ids[0], work_ids[5], work_ids[8]])

    assert select_thirty_fps_works(catalog, cfg, 9) == sorted(work_ids)
    assert select_thirty_fps_works(catalog, cfg, 0) == []
    assert select_thirty_fps_works(catalog, cfg, 20) == sorted(work_ids)

    selected = [durations[work_ids.index(work_id)] for work_id in first]
    ordered = sorted(durations)
    low, high = ordered[len(ordered) // 3], ordered[(2 * len(ordered)) // 3]
    buckets = [
        "short" if value <= low else ("medium" if value <= high else "long")
        for value in selected
    ]
    assert sorted(buckets) == ["long", "medium", "short"]


def test_select_canonical_completed_works_excludes_duplicates_and_pending(
    config: Config, catalog: Catalog, videos: dict[str, Path]
) -> None:
    cfg = _visual_config(config, subset_count=0)
    canonical = _make_local_work(catalog, cfg, videos, "canonical.mp4", "short_4s.mp4")
    duplicate = _make_local_work(catalog, cfg, videos, "duplicate.mp4", "short_4s.mp4")
    catalog.link_works(canonical, duplicate, method="test", confidence=1.0)
    pending = catalog.upsert_local_work("pending_only.mp4")[0]
    catalog.update_work_metadata(pending, duration_s=4.0)

    works = select_canonical_completed_works(catalog)
    ids = {int(work["id"]) for work in works}
    assert ids == {canonical}


# ---------------------------------------------------------------- dry-run / status / 清理


def test_dry_run_does_not_write_or_call_encoders(
    config: Config, catalog: Catalog, videos: dict[str, Path]
) -> None:
    cfg = _visual_config(config, subset_count=0)
    _make_local_work(catalog, cfg, videos, "dry_a.mp4", "short_4s.mp4")
    _make_local_work(catalog, cfg, videos, "dry_b.mp4", "mid_12s.mp4")
    factory = _make_factory()
    extractor = FakeExtractor(cfg)

    summary = run_visual_pipeline(
        catalog,
        cfg,
        dry_run=True,
        encoder_factory=factory,
        extractor=extractor,
    )

    assert summary.dry_run is True
    assert summary.tasks_total == 4
    assert summary.windows_total == 4
    assert summary.tasks_completed == 0
    assert summary.windows_completed == 0
    assert summary.window_plan_version == cfg.visual_window_plan_version
    assert extractor.calls == []
    assert factory.created == []
    assert catalog.list_visual_versions() == []
    assert catalog.list_visual_window_plans() == []
    assert catalog.list_visual_sequences() == []
    assert catalog.list_visual_window_sequences() == []
    assert not any(cfg.paths.visual_dense.iterdir())
    assert not any(window_plan_root(cfg).iterdir())


def test_status_and_clean_temp(
    config: Config, catalog: Catalog, videos: dict[str, Path]
) -> None:
    cfg = _visual_config(config, subset_count=0)
    work_a = _make_local_work(catalog, cfg, videos, "status_a.mp4", "short_4s.mp4")
    work_b = _make_local_work(catalog, cfg, videos, "status_b.mp4", "mid_12s.mp4")
    run_visual_pipeline(
        catalog, cfg, encoder_factory=_make_factory(), extractor=FakeExtractor(cfg)
    )

    payload = visual_status(catalog, cfg)
    assert payload["expected_works"] == 2
    assert len(payload["versions"]) == 2
    assert payload["total_storage_bytes"] > 0
    for version in payload["versions"]:
        assert version["status_counts"]["completed"] == 2
        assert version["status_counts"]["failed"] == 0
        assert version["n_registered"] == 2
        assert version["storage_bytes"] > 0
        assert version["missing_files"] == 0
    assert len(payload["window_plans"]) == 1
    window_plan = payload["window_plans"][0]
    assert window_plan["version"] == cfg.visual_window_plan_version
    assert window_plan["status_counts"]["completed"] == 4  # 2 作品 x 2 帧版本
    assert window_plan["n_registered"] == 4
    assert window_plan["storage_bytes"] > 0
    assert window_plan["missing_files"] == 0
    assert payload["window_total_storage_bytes"] > 0

    completed_temp = cfg.paths.temp / f"work_{work_a:06d}" / "download"
    completed_temp.mkdir(parents=True, exist_ok=True)
    (completed_temp / "stale.mp4").write_bytes(b"x" * 1024)
    pending_work, _ = catalog.upsert_local_work("still_pending.mp4")
    catalog.update_work_metadata(pending_work, duration_s=4.0)
    pending_temp = cfg.paths.temp / f"work_{pending_work:06d}" / "download"
    pending_temp.mkdir(parents=True, exist_ok=True)
    (pending_temp / "keep.mp4").write_bytes(b"y" * 2048)

    result = clean_completed_temp_dirs(catalog, cfg)
    assert result["removed_dirs"] == 1
    assert result["kept_dirs"] == 0
    assert result["freed_bytes"] >= 1024
    assert not completed_temp.parent.exists()
    assert pending_temp.parent.is_dir()
    assert not (cfg.paths.temp / f"work_{work_b:06d}").exists()


def test_resume_interrupted_in_progress_sequence(
    config: Config, catalog: Catalog, videos: dict[str, Path]
) -> None:
    """帧层停在 in_progress：续跑重做帧任务并补窗口层。"""

    cfg = _visual_config(config, subset_count=0, encoders=(("fake-a", "rev-a"),))
    work_id = _make_local_work(
        catalog, cfg, videos, "resume_inprogress.mp4", "short_4s.mp4"
    )
    spec = visual_version_specs(cfg, include_subset=False)[0]
    catalog.register_visual_version(
        spec.version,
        model_id=spec.model,
        revision=spec.revision,
        preprocessing=spec.preprocess,
        fps=spec.fps,
        size=spec.size,
        dim=8,
        params={
            "encoder_version": spec.encoder_version,
            "encoder_dim": 8,
            "preprocess": spec.preprocess,
            "batch_size": int(cfg.visual_dense.batch_size),
            "subset": False,
            "source": "nebula.visual_pipeline",
        },
    )
    catalog.upsert_visual_sequence(
        work_id, version=spec.version, status="in_progress", n_frames=3
    )
    extractor = FakeExtractor(cfg)

    summary = run_visual_pipeline(
        catalog, cfg, encoder_factory=_make_factory(), extractor=extractor
    )

    assert summary.tasks_completed == 1
    assert summary.windows_completed == 1
    frame_row = _frame_sequence(catalog, work_id, spec.version)
    assert frame_row["status"] == "completed"
    assert frame_row["n_frames"] == int(
        round(
            _duration_of(cfg.paths.input_local / "resume_inprogress.mp4", cfg)
            * spec.fps
        )
    )
    window_row = _window_sequence(
        catalog, work_id, spec.version, cfg.visual_window_plan_version
    )
    assert window_row is not None and window_row["status"] == "completed"
    assert len(extractor.calls) == 1


def test_compose_visual_version_is_filesystem_slug():
    version = compose_visual_version("fake/model@rev-123", 8.0, 224, "letterbox")
    assert "/" not in version and "@" not in version
    assert version == visual_version_slug(
        compress_version("fake/model@rev-123", 8.0, 224, "letterbox")
    )

