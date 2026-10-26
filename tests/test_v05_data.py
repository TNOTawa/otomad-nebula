"""nebula.v05_data 的测试（v0.5-alpha 局部音画 token 数据集）。

覆盖：
- 合成 PE-AV 窗口 / clip_times / 事件 bundle / CLIP+DINOv2 帧 bundle 的 shape、dtype、
  bin 时间映射、mean/std、mask、短段与缺失源（mask=0）、超长截断（128 段）；
- 保存/读取 npz + manifest、原子写、坏文件；
- build_dataset 的 canonical+completed 过滤、缺失源/失败/已存在/--force/--limit/--work-ids、
  manifest 增量合并；
- CLI --help 与临时库端到端（--version/--sequence-version/--clip-frame-version 等）；
- 全程只用 tmp 合成数据，不加载 torch/PE-AV，不联网、不写真实 data/。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from nebula import v05_data
from nebula.dense_visual import (
    DenseVisualFeatures,
    compress_version,
    save_bundle,
    visual_version_slug,
)
from nebula.embed import save_vector
from nebula.errors import NebulaError
from nebula.events import EventFeatures, save_events_bundle

SEQ = "v05-test-seq"
CLIP_FRAME = "clip-test-v1"
DINO_FRAME = "dino-test-v1"
AUDIO_RATE = 100.0
VIDEO_FPS = 30.0
VISUAL_FPS = 8.0


# --------------------------------------------------------------------------- 合成源数据
def _register_sequence(catalog, version: str = SEQ) -> None:
    catalog.register_embedding_version(
        version,
        model_checkpoint="test-only:hash",
        model_revision=None,
        frames=16,
        clip_seconds=8.0,
        clips_per_work=0,
        audio_sample_rate=48000,
        params={"test": "v05-data"},
    )


def _frame_matrix(duration_s: float, fps: float, dim: int, offset: int) -> np.ndarray:
    count = int(round(duration_s * fps))
    matrix = np.zeros((count, dim), dtype=np.float32)
    frames_per_bin = int(round(0.5 * fps))
    for index in range(count):
        frame_bin = index // frames_per_bin
        matrix[index, (frame_bin + offset) % dim] = 1.0
    return matrix


def _save_visual_bundle(
    config,
    frame_version: str,
    work_id: int,
    *,
    duration_s: float,
    dim: int,
    offset: int,
    fps: float = VISUAL_FPS,
) -> None:
    frames = _frame_matrix(duration_s, fps, dim, offset)
    features = DenseVisualFeatures(
        per_frame=frames,
        fps=fps,
        duration_s=duration_s,
        n_frames=int(frames.shape[0]),
        dim=dim,
        version=frame_version,
        decode_s=0.0,
        encode_s=0.0,
        model_load_s=0.0,
        encoder="test/clip" if dim == 512 else "test/dinov2",
        revision="test-revision",
        size=224,
        preprocess="letterbox",
    )
    save_bundle(config.paths.visual_dense / frame_version / f"{work_id:06d}", features)


def _make_work(
    catalog,
    config,
    *,
    name: str = "work.mp4",
    sequence_version: str = SEQ,
    n_segments: int = 2,
    duration_s: float = 16.0,
    completed: bool = True,
    with_audio: bool = True,
    with_video: bool = True,
    with_events: bool = True,
    with_visual: bool = True,
    clip_frame_version: str = CLIP_FRAME,
    dino_frame_version: str = DINO_FRAME,
) -> dict:
    _register_sequence(catalog, sequence_version)
    work_id, _ = catalog.upsert_local_work(f"v05/{name}", title=f"测试-{name}")
    if completed:
        catalog.begin_attempt(work_id, "encoding")
        catalog.complete(work_id)
    catalog.update_work_metadata(work_id, duration_s=duration_s)
    work_dir = config.paths.embeddings / sequence_version / f"{work_id:06d}"
    work_dir.mkdir(parents=True, exist_ok=True)

    rows = np.arange(1024, dtype=np.float32) / 1024.0
    joint = np.tile(rows, (n_segments, 1))
    audio = joint * 2.0
    video = joint * 3.0
    save_vector(work_dir / "clip_joint.npy", joint)
    catalog.register_embedding(
        work_id,
        version=sequence_version,
        kind="clip_joint",
        path=config.data_relpath(work_dir / "clip_joint.npy"),
        dim=1024,
        n_clips=n_segments,
    )
    for kind, matrix, enabled in (
        ("clip_audio", audio, with_audio),
        ("clip_video", video, with_video),
    ):
        if not enabled:
            continue
        save_vector(work_dir / f"{kind}.npy", matrix)
        catalog.register_embedding(
            work_id,
            version=sequence_version,
            kind=kind,
            path=config.data_relpath(work_dir / f"{kind}.npy"),
            dim=1024,
            n_clips=n_segments,
        )

    weights = 1.0 / float(n_segments)
    clip_times = np.asarray(
        [
            [
                index * 8.0,
                min(8.0, duration_s - index * 8.0),
                min(8.0, duration_s - index * 8.0),
                weights,
            ]
            for index in range(n_segments)
        ],
        dtype=np.float32,
    )
    save_vector(work_dir / "clip_times.npy", clip_times)
    (work_dir / "manifest.json").write_text(
        json.dumps({"duration_s": duration_s, "n_clips": n_segments}),
        encoding="utf-8",
    )

    if with_events:
        n_audio = int(round(duration_s * AUDIO_RATE))
        n_video = int(round(duration_s * VIDEO_FPS))
        audio_events = np.zeros((n_audio, 6), dtype=np.float32)
        audio_events[:, 0] = 1.0
        audio_events[:, 1] = np.arange(n_audio, dtype=np.float32)
        av_events = np.zeros((n_audio, 6), dtype=np.float32)
        av_events[:, 2] = 2.0
        video_events = np.zeros((n_video, 6), dtype=np.float32)
        video_events[:, 0] = np.arange(n_video, dtype=np.float32)
        features = EventFeatures(
            audio=audio_events,
            video=video_events,
            av=av_events,
            window_features=np.zeros((n_segments, 32), dtype=np.float32),
            audio_rate_hz=AUDIO_RATE,
            video_fps=VIDEO_FPS,
            video_fps_effective=VIDEO_FPS,
            duration_s=duration_s,
            version="events_v1",
        )
        save_events_bundle(work_dir, features)

    if with_visual:
        _save_visual_bundle(
            config,
            clip_frame_version,
            work_id,
            duration_s=duration_s,
            dim=512,
            offset=0,
        )
        _save_visual_bundle(
            config,
            dino_frame_version,
            work_id,
            duration_s=duration_s,
            dim=384,
            offset=100,
        )
    return {
        "work_id": int(work_id),
        "work_dir": work_dir,
        "joint": joint,
        "audio": audio,
        "video": video,
        "clip_times": clip_times,
    }


def _tokens_for(work: dict, config) -> dict[str, np.ndarray]:
    return v05_data.build_work_tokens(
        config,
        work["work_id"],
        sequence_version=SEQ,
        clip_frame_version=CLIP_FRAME,
        dino_frame_version=DINO_FRAME,
    )


def _expected_event_indices(rate: float, start_s: float, bin_index: int) -> np.ndarray:
    low = start_s + bin_index * 0.5
    high = low + 0.5
    count = int(round(high * rate))  # 足够大的上界
    indices = np.arange(count, dtype=np.float64)
    times = indices / rate
    selected = indices[(times >= low) & (times < high)]
    return selected


# --------------------------------------------------------------------------- 单作品 token
def test_build_work_tokens_shapes_bins_and_masks(config, catalog):
    work = _make_work(catalog, config, name="tokens.mp4")
    tokens = _tokens_for(work, config)

    assert tokens["paav"].shape == (2, 3, 1024)
    assert tokens["paav"].dtype == np.float16
    assert tokens["visual_clip"].shape == (2, 16, 512)
    assert tokens["visual_dino"].shape == (2, 16, 384)
    assert tokens["audio"].shape == (2, 16, 24)
    assert tokens["visual_event"].shape == (2, 16, 12)
    for key in ("visual_clip", "visual_dino", "audio", "visual_event"):
        assert tokens[key].dtype == np.float16
        assert np.all(np.isfinite(tokens[key]))
    for key in ("mask_visual", "mask_audio", "mask_visual_event"):
        assert tokens[key].shape == (2, 16)
        assert tokens[key].dtype == np.uint8
        assert int(tokens[key].sum()) == 32
    assert tokens["segment_times"].shape == (2, 4)
    assert tokens["segment_times"].dtype == np.float32
    np.testing.assert_allclose(tokens["segment_times"], work["clip_times"])
    assert tokens["segment_mask"].shape == (2,)
    assert tokens["segment_mask"].dtype == np.uint8
    assert np.array_equal(tokens["segment_mask"], np.ones(2, dtype=np.uint8))
    assert float(tokens["duration_s"]) == pytest.approx(16.0)
    assert float(tokens["fps"]) == pytest.approx(8.0)
    assert int(tokens["n_segments_original"]) == 2
    assert int(tokens["truncated"]) == 0

    # paav 三通道 = clip_joint / clip_audio / clip_video
    np.testing.assert_allclose(
        tokens["paav"][:, 0].astype(np.float32), work["joint"], atol=2e-3
    )
    np.testing.assert_allclose(
        tokens["paav"][:, 1].astype(np.float32), work["audio"], atol=2e-3
    )
    np.testing.assert_allclose(
        tokens["paav"][:, 2].astype(np.float32), work["video"], atol=2e-3
    )

    # 视觉 bin：one-hot 帧 -> one-hot token（段 s、bin k 的帧 bin = s*16+k）
    for segment in range(2):
        for bin_index in range(16):
            expected_frame_bin = segment * 16 + bin_index
            assert int(np.argmax(tokens["visual_clip"][segment, bin_index])) == expected_frame_bin
            assert (
                int(np.argmax(tokens["visual_dino"][segment, bin_index]))
                == (expected_frame_bin + 100) % 384
            )

    # 音频：audio[:,1] = 100Hz 索引；av[:,2] = 2；按 bin 时间映射 mean/std
    for segment in range(2):
        start = segment * 8.0
        for bin_index in (0, 1, 15):
            indices = _expected_event_indices(AUDIO_RATE, start, bin_index)
            token = tokens["audio"][segment, bin_index].astype(np.float64)
            assert token[0] == pytest.approx(1.0, abs=1e-3)
            assert token[1] == pytest.approx(indices.mean(), abs=1.5)
            assert token[7] == pytest.approx(indices.std(), abs=0.05)
            assert token[14] == pytest.approx(2.0, abs=1e-3)
            assert token[20] == pytest.approx(0.0, abs=1e-3)

    # 视觉事件：video[:,0] = 30fps 索引；mean/std
    for segment in range(2):
        start = segment * 8.0
        for bin_index in (0, 3, 15):
            indices = _expected_event_indices(VIDEO_FPS, start, bin_index)
            token = tokens["visual_event"][segment, bin_index].astype(np.float64)
            assert token[0] == pytest.approx(indices.mean(), abs=0.6)
            assert token[6] == pytest.approx(indices.std(), abs=0.05)

    # 确定性：同一输入重复构建逐元素相同
    repeated = _tokens_for(work, config)
    for key in tokens:
        np.testing.assert_array_equal(tokens[key], repeated[key])


def test_short_video_masks_missing_bins(config, catalog):
    work = _make_work(
        catalog,
        config,
        name="short.mp4",
        n_segments=1,
        duration_s=4.0,
    )
    tokens = _tokens_for(work, config)
    assert tokens["paav"].shape == (1, 3, 1024)
    assert int(tokens["mask_visual"].sum()) == 8  # 4s / 0.5s = 8 个有帧的 bin
    assert int(tokens["mask_audio"].sum()) == 8
    assert int(tokens["mask_visual_event"].sum()) == 8
    assert np.all(tokens["visual_clip"][0, 8:] == 0)
    assert np.all(tokens["audio"][0, 8:] == 0)
    assert np.all(tokens["visual_event"][0, 8:] == 0)
    np.testing.assert_allclose(tokens["segment_times"][0], [0.0, 4.0, 4.0, 1.0])


def test_missing_event_and_visual_sources_mask_zero(config, catalog):
    no_events = _make_work(
        catalog, config, name="no-events.mp4", with_events=False
    )
    tokens = _tokens_for(no_events, config)
    assert int(tokens["mask_audio"].sum()) == 0
    assert int(tokens["mask_visual_event"].sum()) == 0
    assert np.all(tokens["audio"] == 0)
    assert np.all(tokens["visual_event"] == 0)
    assert int(tokens["mask_visual"].sum()) == 32  # 视觉仍在

    no_visual = _make_work(
        catalog, config, name="no-visual.mp4", with_visual=False
    )
    tokens = _tokens_for(no_visual, config)
    assert int(tokens["mask_visual"].sum()) == 0
    assert np.all(tokens["visual_clip"] == 0)
    assert np.all(tokens["visual_dino"] == 0)
    assert int(tokens["mask_audio"].sum()) == 32

    only_events = _make_work(
        catalog,
        config,
        name="no-events-visual.mp4",
        with_events=False,
        with_visual=False,
    )
    tokens = _tokens_for(only_events, config)
    assert int(tokens["mask_visual"].sum()) == 0
    assert int(tokens["mask_audio"].sum()) == 0
    assert int(tokens["mask_visual_event"].sum()) == 0
    assert float(tokens["fps"]) == pytest.approx(float(config.visual_dense.fps))


def test_missing_peav_modalities_use_zero_channels(config, catalog):
    work = _make_work(
        catalog, config, name="joint-only.mp4", with_audio=False, with_video=False
    )
    tokens = _tokens_for(work, config)
    np.testing.assert_allclose(
        tokens["paav"][:, 0].astype(np.float32), work["joint"], atol=2e-3
    )
    assert np.all(tokens["paav"][:, 1] == 0)
    assert np.all(tokens["paav"][:, 2] == 0)


def test_truncation_to_max_segments(config, catalog):
    work = _make_work(
        catalog,
        config,
        name="long.mp4",
        n_segments=v05_data.V05_MAX_SEGMENTS + 2,
        duration_s=float((v05_data.V05_MAX_SEGMENTS + 2) * 8),
        with_events=False,
        with_visual=False,
    )
    tokens = _tokens_for(work, config)
    assert tokens["paav"].shape == (v05_data.V05_MAX_SEGMENTS, 3, 1024)
    assert int(tokens["n_segments_original"]) == v05_data.V05_MAX_SEGMENTS + 2
    assert int(tokens["truncated"]) == 1
    np.testing.assert_allclose(
        tokens["segment_times"], work["clip_times"][: v05_data.V05_MAX_SEGMENTS]
    )
    assert int(tokens["mask_visual"].sum()) == 0

    # 没有序列 manifest 时，截断前仍按原始 clip_times 记录完整 duration
    (work["work_dir"] / "manifest.json").unlink()
    without_manifest = _tokens_for(work, config)
    assert float(without_manifest["duration_s"]) == pytest.approx(
        float((v05_data.V05_MAX_SEGMENTS + 2) * 8)
    )


def test_invalid_source_shapes_and_missing_times(config, catalog):
    work = _make_work(catalog, config, name="bad.mp4")
    work_dir = work["work_dir"]

    # clip_times 行数不一致
    save_vector(work_dir / "clip_times.npy", work["clip_times"][:1])
    with pytest.raises(NebulaError):
        _tokens_for(work, config)
    save_vector(work_dir / "clip_times.npy", work["clip_times"])

    # 缺 clip_times
    (work_dir / "clip_times.npy").unlink()
    with pytest.raises(NebulaError):
        _tokens_for(work, config)
    save_vector(work_dir / "clip_times.npy", work["clip_times"])

    # 缺 clip_joint
    (work_dir / "clip_joint.npy").unlink()
    with pytest.raises(NebulaError):
        _tokens_for(work, config)
    save_vector(work_dir / "clip_joint.npy", work["joint"])

    # NaN
    bad = work["joint"].copy()
    bad[0, 0] = np.nan
    save_vector(work_dir / "clip_joint.npy", bad)
    with pytest.raises(NebulaError):
        _tokens_for(work, config)
    save_vector(work_dir / "clip_joint.npy", work["joint"])

    # 视觉 bundle 维度与契约不符（10 != 512）
    _save_visual_bundle(
        config, CLIP_FRAME, work["work_id"], duration_s=16.0, dim=10, offset=0
    )
    with pytest.raises(NebulaError):
        _tokens_for(work, config)


# --------------------------------------------------------------------------- 保存 / 读取
def test_save_and_load_work_tokens_roundtrip(config, catalog):
    work = _make_work(catalog, config, name="roundtrip.mp4")
    tokens = _tokens_for(work, config)
    path = v05_data.save_work_tokens(config, "roundtrip-ds", work["work_id"], tokens)
    assert path.is_file()
    assert path.name == f"{work['work_id']:06d}.npz"
    assert not list(path.parent.glob("*.tmp*"))
    loaded = v05_data.load_work_tokens(config, "roundtrip-ds", work["work_id"])
    assert loaded is not None
    for key in ("paav", "visual_clip", "audio", "mask_visual", "segment_times"):
        np.testing.assert_array_equal(loaded[key], tokens[key])
    assert v05_data.load_work_tokens(config, "roundtrip-ds", 999999) is None

    path.write_bytes(b"not a numpy file")
    with pytest.raises(NebulaError):
        v05_data.load_work_tokens(config, "roundtrip-ds", work["work_id"])

    # 形状不合法时拒绝写入
    broken = dict(tokens)
    broken["mask_visual"] = np.zeros((3, 16), dtype=np.uint8)
    with pytest.raises(NebulaError):
        v05_data.save_work_tokens(config, "roundtrip-ds", work["work_id"], broken)


def test_dataset_manifest_roundtrip(config):
    assert v05_data.load_dataset_manifest(config, "missing-ds") is None
    payload = {"dataset_version": "x", "created_at": "2026-10-26T00:00:00Z"}
    path = v05_data.dataset_manifest_path(config, "x-ds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert v05_data.load_dataset_manifest(config, "x-ds") == payload
    path.write_text("{bad json", encoding="utf-8")
    with pytest.raises(NebulaError):
        v05_data.load_dataset_manifest(config, "x-ds")
    with pytest.raises(NebulaError):
        v05_data.dataset_dir(config, "../escape")


def test_default_frame_versions_from_config(config):
    clip_version, dino_version = v05_data.resolve_default_frame_versions(config)
    assert clip_version is not None and dino_version is not None
    assert clip_version != dino_version
    expected: dict[str, str] = {}
    for encoder in config.visual_dense.encoders:
        raw = compress_version(
            f"{encoder.model}@{encoder.revision}",
            config.visual_dense.fps,
            config.visual_dense.size,
            config.visual_dense.preprocess,
        )
        model = encoder.model.lower()
        if "clip" in model:
            expected["clip"] = visual_version_slug(raw)
        elif "dinov2" in model:
            expected["dino"] = visual_version_slug(raw)
    assert clip_version == expected["clip"]
    assert dino_version == expected["dino"]
    assert "fps8" in clip_version and "s224" in clip_version


# --------------------------------------------------------------------------- 数据集
def test_build_dataset_filters_counts_and_manifest(config, catalog):
    good_a = _make_work(catalog, config, name="a.mp4")
    good_b = _make_work(catalog, config, name="b.mp4")
    missing = _make_work(catalog, config, name="missing.mp4")
    pending = _make_work(catalog, config, name="pending.mp4", completed=False)
    duplicate = _make_work(catalog, config, name="dup.mp4")
    catalog.link_works(good_a["work_id"], duplicate["work_id"], method="test", confidence=1.0)
    (missing["work_dir"] / "clip_times.npy").unlink()

    summary = v05_data.build_dataset(
        catalog,
        config,
        version="dataset-ds",
        sequence_version=SEQ,
        clip_frame_version=CLIP_FRAME,
        dino_frame_version=DINO_FRAME,
        force=True,
    )
    assert summary["n_candidates"] == 3  # duplicate 排除，pending 不是候选
    assert summary["n_built"] == 2
    assert summary["n_existing"] == 0
    assert summary["n_missing"] == 1
    assert summary["n_failed"] == 0
    assert summary["n_truncated"] == 0
    assert summary["missing"][0]["work_id"] == missing["work_id"]
    assert "clip_times" in summary["missing"][0]["reasons"][0]
    assert pending["work_id"] not in summary["built_work_ids"]
    assert duplicate["work_id"] not in summary["built_work_ids"]

    for work in (good_a, good_b):
        path = v05_data.token_path(config, "dataset-ds", work["work_id"])
        assert path.is_file()
        loaded = v05_data.load_work_tokens(config, "dataset-ds", work["work_id"])
        assert loaded is not None and int(np.asarray(loaded["n_segments_original"])) == 2

    manifest = v05_data.load_dataset_manifest(config, "dataset-ds")
    assert manifest is not None
    assert manifest["dataset_version"] == "dataset-ds"
    assert manifest["sequence_version"] == SEQ
    assert manifest["clip_frame_version"] == CLIP_FRAME
    assert manifest["dino_frame_version"] == DINO_FRAME
    assert manifest["bin_seconds"] == v05_data.V05_BIN_SECONDS
    assert manifest["bins_per_segment"] == 16
    assert manifest["max_segments"] == 128
    assert manifest["dims"]["paav"] == [3, 1024]
    assert manifest["token_layout"]["paav"] == ["joint", "audio", "video"]
    assert manifest["token_layout"]["audio"][0] == "audio_mean_6"
    assert manifest["counts"]["n_built"] == 2
    assert len(manifest["works"]) == 2
    assert manifest["created_at"]
    assert all(item["n_segments"] == 2 for item in manifest["works"])
    assert {item["work_id"] for item in manifest["works"]} == {
        good_a["work_id"],
        good_b["work_id"],
    }

    # 无 --force：已有 npz 跳过，manifest 合并保留已有作品
    again = v05_data.build_dataset(
        catalog,
        config,
        version="dataset-ds",
        sequence_version=SEQ,
        clip_frame_version=CLIP_FRAME,
        dino_frame_version=DINO_FRAME,
    )
    assert again["n_existing"] == 2
    assert again["n_built"] == 0
    manifest = v05_data.load_dataset_manifest(config, "dataset-ds")
    assert len(manifest["works"]) == 2

    forced = v05_data.build_dataset(
        catalog,
        config,
        version="dataset-ds",
        sequence_version=SEQ,
        clip_frame_version=CLIP_FRAME,
        dino_frame_version=DINO_FRAME,
        force=True,
    )
    assert forced["n_built"] == 2

    limited = v05_data.build_dataset(
        catalog,
        config,
        version="dataset-limit-ds",
        sequence_version=SEQ,
        clip_frame_version=CLIP_FRAME,
        dino_frame_version=DINO_FRAME,
        force=True,
        limit=1,
    )
    assert limited["n_candidates"] == 1
    assert limited["n_built"] == 1

    subset = v05_data.build_dataset(
        catalog,
        config,
        version="dataset-subset-ds",
        sequence_version=SEQ,
        clip_frame_version=CLIP_FRAME,
        dino_frame_version=DINO_FRAME,
        force=True,
        work_ids=[good_b["work_id"]],
    )
    assert subset["n_candidates"] == 1
    assert subset["n_built"] == 1
    assert v05_data.token_path(config, "dataset-subset-ds", good_b["work_id"]).is_file()

    explicit_problem = v05_data.build_dataset(
        catalog,
        config,
        version="dataset-problem-ds",
        sequence_version=SEQ,
        clip_frame_version=CLIP_FRAME,
        dino_frame_version=DINO_FRAME,
        force=True,
        work_ids=[pending["work_id"], 999999],
    )
    assert explicit_problem["n_candidates"] == 0
    assert explicit_problem["n_missing"] == 2
    reasons = [item["reasons"][0] for item in explicit_problem["missing"]]
    assert any("completed" in reason for reason in reasons)
    assert any("不存在" in reason for reason in reasons)


def test_build_dataset_manifest_merges_incremental_runs(config, catalog):
    first = _make_work(catalog, config, name="inc1.mp4")
    second = _make_work(catalog, config, name="inc2.mp4")
    run1 = v05_data.build_dataset(
        catalog,
        config,
        version="merge-ds",
        sequence_version=SEQ,
        clip_frame_version=CLIP_FRAME,
        dino_frame_version=DINO_FRAME,
        force=True,
        limit=1,
    )
    assert run1["n_built"] == 1
    run2 = v05_data.build_dataset(
        catalog,
        config,
        version="merge-ds",
        sequence_version=SEQ,
        clip_frame_version=CLIP_FRAME,
        dino_frame_version=DINO_FRAME,
    )
    assert run2["n_existing"] == 1
    assert run2["n_built"] == 1
    manifest = v05_data.load_dataset_manifest(config, "merge-ds")
    assert {item["work_id"] for item in manifest["works"]} == {
        first["work_id"],
        second["work_id"],
    }


def test_build_dataset_missing_sequence_versions(config, catalog):
    _make_work(catalog, config, name="seq.mp4")
    with pytest.raises(NebulaError):
        v05_data.build_dataset(
            catalog,
            config,
            version="bad-seq-ds",
            sequence_version="missing-sequence",
        )
    with pytest.raises(NebulaError):
        v05_data.build_dataset(catalog, config, version="../escape")
    with pytest.raises(NebulaError):
        v05_data.build_dataset(catalog, config, version="ok-ds", limit=0)


# --------------------------------------------------------------------------- CLI
def test_cli_help(capsys):
    from tools import build_v05_tokens

    with pytest.raises(SystemExit) as excinfo:
        build_v05_tokens.main(["--help"])
    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    for flag in (
        "--version",
        "--sequence-version",
        "--clip-frame-version",
        "--dino-frame-version",
        "--force",
        "--limit",
        "--work-ids",
    ):
        assert flag in output


def test_cli_end_to_end(config, catalog, capsys):
    from tools import build_v05_tokens

    first = _make_work(catalog, config, name="cli1.mp4")
    second = _make_work(catalog, config, name="cli2.mp4")
    code = build_v05_tokens.main(
        [
            "--config",
            str(config.path),
            "--version",
            "cli-ds",
            "--sequence-version",
            SEQ,
            "--clip-frame-version",
            CLIP_FRAME,
            "--dino-frame-version",
            DINO_FRAME,
            "--force",
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "v0.5-alpha" in output
    assert "cli-ds" in output
    assert "新建=2" in output
    for work in (first, second):
        assert v05_data.token_path(config, "cli-ds", work["work_id"]).is_file()
    assert v05_data.dataset_manifest_path(config, "cli-ds").is_file()
    manifest = v05_data.load_dataset_manifest(config, "cli-ds")
    assert manifest["counts"]["n_built"] == 2


def test_cli_reports_missing_sources(config, catalog, capsys):
    from tools import build_v05_tokens

    work = _make_work(catalog, config, name="cli-missing.mp4")
    (work["work_dir"] / "clip_times.npy").unlink()
    code = build_v05_tokens.main(
        [
            "--config",
            str(config.path),
            "--version",
            "cli-missing-ds",
            "--sequence-version",
            SEQ,
            "--clip-frame-version",
            CLIP_FRAME,
            "--dino-frame-version",
            DINO_FRAME,
        ]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert "缺失源" in captured.out or "缺失" in captured.out
    assert work["work_id"] in [
        item["work_id"] for item in v05_data.load_dataset_manifest(config, "cli-missing-ds")["missing"]
    ]
