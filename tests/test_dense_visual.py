"""v0.4.3 密集视觉语义编码（nebula.dense_visual + pilot CLI）的离线测试。

覆盖：

- iter_frame_batches：单条 ffmpeg 流式解码的批次形状/帧数/EOF 校验、无音轨视频、
  确定性、坏文件与非法参数；
- FakeVisualEncoder 注入下的 extract_dense_features：形状/归一化/版本串/耗时字段/确定性；
- pool_windows / pool_windows_std：按重叠时长加权的手算对照、边界与非法窗口；
- DenseVisualFeatures 校验、save/load roundtrip（fp16 npz + manifest）、原子写与损坏检测；
- HFImageEncoder：构造参数校验、懒加载元数据、cache_dir(hub) 解析（不加载模型/不联网）；
- 模块导入不 import torch/transformers；pilot CLI --help 与参数解析。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nebula import dense_visual as dense_module
from nebula.config import load_config
from nebula.dense_visual import (
    DENSE_VISUAL_LEGACY_MANIFEST_NAME,
    DENSE_VISUAL_MANIFEST_NAME,
    DENSE_VISUAL_NPZ_NAME,
    PREPROCESS_MODES,
    WINDOW_STD_NPY_NAME,
    WINDOWS_NPY_NAME,
    DenseVisualError,
    DenseVisualFeatures,
    HFImageEncoder,
    VisualBundle,
    VisualEncoder,
    build_video_filter,
    compress_version,
    extract_dense_features,
    iter_frame_batches,
    load_bundle,
    load_dense_visual,
    load_features,
    pool_windows,
    pool_windows_std,
    save_bundle,
    save_dense_visual,
    save_features,
    visual_version_slug,
)
from nebula.errors import NebulaError
from nebula.sampling import coverage_weights, plan_full_coverage
from tests.fixtures.video_factory import build_video_library, media_tools_available

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "tools" / "dense_visual_pilot.py"
COMPARE_CLI_PATH = REPO_ROOT / "tools" / "dense_preprocess_compare.py"


# ──────────────────────────────────────────────────────────────────────────────
# 测试替身与工具
# ──────────────────────────────────────────────────────────────────────────────


class FakeVisualEncoder:
    """确定性 fake 帧编码器（不依赖 torch）：按帧内容统计生成归一化向量。"""

    def __init__(self, dim: int = 8, version: str = "fake-visual-v1") -> None:
        self.version = version
        self.dim = dim
        self.model_load_s = 0.0
        self.batch_shapes: list[tuple[int, ...]] = []
        self.closed = 0

    def encode(self, frames: np.ndarray) -> np.ndarray:
        array = np.asarray(frames)
        if array.ndim != 4 or array.shape[0] < 1:
            raise DenseVisualError(f"fake encoder 收到非法帧批次: {array.shape}")
        arr = array.astype(np.float32) / 255.0
        self.batch_shapes.append(tuple(int(value) for value in array.shape))
        means = arr.mean(axis=(1, 2, 3))
        channel_means = [arr[..., index].mean(axis=(1, 2)) for index in range(3)]
        channel_stds = [arr[..., index].std(axis=(1, 2)) for index in range(3)]
        features = np.stack(
            [means, *channel_means, *channel_stds, arr.std(axis=(1, 2, 3))], axis=1
        ).astype(np.float32)
        if features.shape[1] >= self.dim:
            features = features[:, : self.dim]
        else:
            padding = np.repeat(features[:, -1:], self.dim - features.shape[1], axis=1)
            features = np.concatenate([features, padding], axis=1)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return (features / np.maximum(norms, 1e-9)).astype(np.float32)

    def close(self) -> None:
        self.closed += 1


def _unit(rows: list[list[float]]) -> np.ndarray:
    array = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return (array / norms).astype(np.float32)


def _make_features(
    per_frame: np.ndarray,
    *,
    fps: float = 1.0,
    duration_s: float | None = None,
    version: str = "fake__fps1_s4_stretch",
    encoder: str = "fake",
    size: int = 4,
    preprocess: str = "stretch",
) -> DenseVisualFeatures:
    matrix = np.asarray(per_frame, dtype=np.float32)
    if matrix.ndim != 2:
        raise AssertionError("测试构造的 per_frame 必须是二维")
    n_frames, dim = matrix.shape
    fallback_duration = (n_frames / fps) if fps else 1.0
    return DenseVisualFeatures(
        per_frame=matrix,
        fps=fps,
        duration_s=float(duration_s if duration_s is not None else fallback_duration),
        n_frames=n_frames,
        dim=dim,
        version=version,
        decode_s=0.0,
        encode_s=0.0,
        model_load_s=0.0,
        encoder=encoder,
        size=size,
        preprocess=preprocess,
        created_at="2026-01-01T00:00:00Z",
    )


@pytest.fixture(scope="module")
def videos() -> dict[str, Path]:
    if not media_tools_available():
        pytest.skip("缺少 ffmpeg/ffprobe")
    return build_video_library(
        names=["mid_12s.mp4", "silent_6s.mp4", "tiny_05s.mp4", "short_4s.mp4"]
    )


def _collect_batches(source: Path, **kwargs) -> list[np.ndarray]:
    return list(iter_frame_batches(source, **kwargs))


# ──────────────────────────────────────────────────────────────────────────────
# iter_frame_batches
# ──────────────────────────────────────────────────────────────────────────────


def test_iter_frame_batches_shapes_and_frame_count(videos: dict[str, Path]):
    batches = _collect_batches(
        videos["mid_12s.mp4"], fps=2, size=32, batch_size=5
    )
    assert batches, "至少产出一个批次"
    total = 0
    for batch in batches:
        assert batch.dtype == np.uint8
        assert batch.ndim == 4
        assert batch.shape[1:] == (32, 32, 3)
        assert 1 <= batch.shape[0] <= 5
        total += int(batch.shape[0])
    # 12s * 2fps = 24 帧（允许 ±2 帧容差）
    assert total == pytest.approx(24, abs=2)
    assert sum(batch.shape[0] for batch in batches[:-1]) == 20
    assert batches[-1].shape[0] == total - 20


def test_iter_frame_batches_silent_video(videos: dict[str, Path]):
    batches = _collect_batches(videos["silent_6s.mp4"], fps=3, size=16, batch_size=4)
    total = sum(int(batch.shape[0]) for batch in batches)
    assert total == pytest.approx(18, abs=2)
    assert all(batch.shape[1:] == (16, 16, 3) for batch in batches)


def test_iter_frame_batches_tiny_video(videos: dict[str, Path]):
    batches = _collect_batches(videos["tiny_05s.mp4"], fps=2, size=16, batch_size=4)
    total = sum(int(batch.shape[0]) for batch in batches)
    assert total >= 1
    assert total <= 3


def test_iter_frame_batches_deterministic(videos: dict[str, Path]):
    first = _collect_batches(videos["mid_12s.mp4"], fps=2, size=16, batch_size=7)
    second = _collect_batches(videos["mid_12s.mp4"], fps=2, size=16, batch_size=7)
    assert len(first) == len(second)
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left, right)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fps": 0},
        {"fps": -1},
        {"fps": float("nan")},
        {"fps": float("inf")},
        {"size": 1},
        {"size": 0},
        {"batch_size": 0},
        {"timeout_s": 0},
        {"timeout_s": -5},
    ],
)
def test_iter_frame_batches_invalid_params(videos: dict[str, Path], kwargs: dict):
    with pytest.raises(DenseVisualError):
        _collect_batches(videos["mid_12s.mp4"], **{"fps": 2, "size": 16, **kwargs})


def test_iter_frame_batches_missing_file(tmp_path: Path):
    with pytest.raises(DenseVisualError, match="不存在"):
        _collect_batches(tmp_path / "missing.mp4", fps=2, size=16)


def test_iter_frame_batches_corrupt_file(tmp_path: Path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"this is definitely not a video" * 64)
    with pytest.raises(DenseVisualError):
        _collect_batches(broken, fps=2, size=16)


def test_iter_frame_batches_missing_ffmpeg(videos: dict[str, Path]):
    with pytest.raises(DenseVisualError, match="ffmpeg"):
        _collect_batches(
            videos["mid_12s.mp4"], fps=2, size=16, ffmpeg="definitely-missing-ffmpeg"
        )


def test_iter_frame_batches_detects_frame_count_mismatch(
    videos: dict[str, Path], monkeypatch
):
    # 伪造 probe 时长为 30s（预期 60 帧 @2fps），实际只有 24 帧 -> 必须报错
    monkeypatch.setattr(dense_module, "_probe_duration_s", lambda *a, **k: 30.0)
    with pytest.raises(DenseVisualError, match="帧数"):
        _collect_batches(videos["mid_12s.mp4"], fps=2, size=16, batch_size=8)


# ──────────────────────────────────────────────────────────────────────────────
# extract_dense_features
# ──────────────────────────────────────────────────────────────────────────────


def test_extract_features_with_fake_encoder(videos: dict[str, Path]):
    encoder = FakeVisualEncoder(dim=8)
    features = extract_dense_features(
        videos["mid_12s.mp4"], encoder, fps=2, size=32, batch_size=5
    )
    assert isinstance(features, DenseVisualFeatures)
    assert features.dim == 8
    assert features.per_frame.dtype == np.float32
    assert features.per_frame.shape == (features.n_frames, 8)
    assert features.n_frames == int(sum(shape[0] for shape in encoder.batch_shapes))
    assert features.n_frames == pytest.approx(24, abs=2)
    norms = np.linalg.norm(features.per_frame, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert features.fps == pytest.approx(2.0)
    assert features.duration_s == pytest.approx(12.0, abs=0.1)
    assert features.version == "fake-visual-v1__fps2_s32_stretch"
    assert features.version.startswith(encoder.version)
    assert "fps2" in features.version and "s32" in features.version
    assert features.preprocess == "stretch"
    assert features.decode_s > 0.0
    assert features.encode_s >= 0.0
    assert features.model_load_s == 0.0
    assert features.encoder == encoder.version
    assert features.size == 32
    assert features.created_at


def test_extract_features_deterministic(videos: dict[str, Path]):
    first = extract_dense_features(
        videos["silent_6s.mp4"],
        FakeVisualEncoder(dim=6),
        fps=2,
        size=16,
        batch_size=4,
    )
    second = extract_dense_features(
        videos["silent_6s.mp4"],
        FakeVisualEncoder(dim=6),
        fps=2,
        size=16,
        batch_size=4,
    )
    assert np.array_equal(first.per_frame, second.per_frame)
    assert first.version == second.version


def test_extract_features_rejects_encoder_without_encode(videos: dict[str, Path]):
    class Broken:
        version = "broken"
        dim = 4

    with pytest.raises(DenseVisualError, match="encode"):
        extract_dense_features(videos["tiny_05s.mp4"], Broken())  # type: ignore[arg-type]


def test_extract_features_rejects_wrong_row_count(videos: dict[str, Path]):
    class WrongRows(FakeVisualEncoder):
        def encode(self, frames: np.ndarray) -> np.ndarray:
            return np.ones((1, 4), dtype=np.float32)

    with pytest.raises(DenseVisualError, match="行数"):
        extract_dense_features(
            videos["silent_6s.mp4"], WrongRows(dim=4), fps=2, size=16
        )


def test_extract_features_rejects_dim_mismatch(videos: dict[str, Path]):
    class WrongDim(FakeVisualEncoder):
        def __init__(self) -> None:
            super().__init__(dim=8)

        def encode(self, frames: np.ndarray) -> np.ndarray:
            return np.ones((int(frames.shape[0]), 4), dtype=np.float32)

    with pytest.raises(DenseVisualError, match="维度"):
        extract_dense_features(
            videos["silent_6s.mp4"], WrongDim(), fps=2, size=16
        )


def test_extract_features_missing_source(tmp_path: Path):
    with pytest.raises(DenseVisualError, match="不存在"):
        extract_dense_features(
            tmp_path / "missing.mp4", FakeVisualEncoder(), fps=2, size=16
        )


# ──────────────────────────────────────────────────────────────────────────────
# 窗口池化（手算对照）
# ──────────────────────────────────────────────────────────────────────────────


def test_pool_windows_hand_calculated():
    features = _make_features(
        _unit([[1, 0], [1, 0], [0, 1], [0, 1]]), fps=1.0, duration_s=4.0
    )
    pooled = pool_windows(
        features,
        [(0.0, 2.0), (2.0, 4.0), (0.0, 4.0), (0.5, 1.5), (1.5, 2.5), (0.0, 0.5)],
    )
    assert pooled.shape == (6, 2)
    assert pooled.dtype == np.float32
    assert np.allclose(pooled[0], [1.0, 0.0], atol=1e-6)
    assert np.allclose(pooled[1], [0.0, 1.0], atol=1e-6)
    assert np.allclose(pooled[2], [1.0, 1.0] / np.sqrt(2.0), atol=1e-6)
    assert np.allclose(pooled[3], [1.0, 0.0], atol=1e-6)
    assert np.allclose(pooled[4], [1.0, 1.0] / np.sqrt(2.0), atol=1e-6)
    assert np.allclose(pooled[5], [1.0, 0.0], atol=1e-6)


def test_pool_windows_weights_by_overlap_duration():
    features = _make_features(_unit([[1, 0], [0, 1]]), fps=1.0, duration_s=2.0)
    pooled = pool_windows(features, [(0.0, 1.5)])
    expected = np.array([1.0, 0.5], dtype=np.float32)
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(pooled[0], expected, atol=1e-6)


def test_pool_windows_output_rows_are_unit_norm():
    features = _make_features(_unit([[3, 4, 0], [0, 5, 12]]), fps=1.0, duration_s=2.0)
    pooled = pool_windows(features, [(0.0, 2.0)])
    assert float(np.linalg.norm(pooled[0])) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "windows",
    [
        [],
        [(0.0, 0.0)],
        [(-1.0, 2.0)],
        [(1.0, 0.5)],
        [(0.0, 99.0)],
        [(4.5, 4.6)],
        ["bad"],
    ],
)
def test_pool_windows_invalid_windows(windows):
    features = _make_features(
        _unit([[1, 0], [0, 1], [1, 1], [1, 1]]), fps=1.0, duration_s=4.0
    )
    with pytest.raises(DenseVisualError):
        pool_windows(features, windows)


def test_pool_windows_requires_features_type():
    with pytest.raises(DenseVisualError, match="DenseVisualFeatures"):
        pool_windows({"per_frame": np.ones((2, 2))}, [(0.0, 1.0)])  # type: ignore[arg-type]


def test_pool_windows_std_hand_calculated():
    features = _make_features(_unit([[1, 0], [0, 1]]), fps=1.0, duration_s=2.0)
    std = pool_windows_std(features, [(0.0, 2.0), (0.0, 1.0)])
    assert std.shape == (2, 2)
    assert std.dtype == np.float32
    assert np.allclose(std[0], [0.5, 0.5], atol=1e-6)
    assert np.allclose(std[1], [0.0, 0.0], atol=1e-6)


def test_pool_windows_on_extracted_features(videos: dict[str, Path]):
    features = extract_dense_features(
        videos["mid_12s.mp4"], FakeVisualEncoder(dim=8), fps=2, size=16, batch_size=6
    )
    windows = [(float(plan.start_s), float(plan.end_s)) for plan in plan_full_coverage(12.0)]
    assert len(windows) == 2
    pooled = pool_windows(features, windows)
    std = pool_windows_std(features, windows)
    assert pooled.shape == (2, 8)
    assert np.allclose(np.linalg.norm(pooled, axis=1), 1.0, atol=1e-5)
    assert std.shape == (2, 8)
    assert np.all(std >= 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# DenseVisualFeatures / 版本串
# ──────────────────────────────────────────────────────────────────────────────


def test_features_validation_rejects_inconsistent_shapes():
    with pytest.raises(DenseVisualError, match="n_frames"):
        DenseVisualFeatures(
            per_frame=np.ones((2, 4), dtype=np.float32),
            fps=1.0,
            duration_s=2.0,
            n_frames=3,
            dim=4,
            version="v",
            decode_s=0.0,
            encode_s=0.0,
            model_load_s=0.0,
        )
    with pytest.raises(DenseVisualError, match="dim"):
        DenseVisualFeatures(
            per_frame=np.ones((2, 4), dtype=np.float32),
            fps=1.0,
            duration_s=2.0,
            n_frames=2,
            dim=5,
            version="v",
            decode_s=0.0,
            encode_s=0.0,
            model_load_s=0.0,
        )


def test_features_validation_rejects_nan_and_bad_timing():
    with pytest.raises(DenseVisualError, match="NaN"):
        _make_features(np.asarray([[1.0, np.nan]], dtype=np.float32))
    with pytest.raises(DenseVisualError, match="fps"):
        _make_features(_unit([[1, 0]]), fps=0.0)
    with pytest.raises(DenseVisualError):
        _make_features(_unit([[1, 0]]), duration_s=-1.0)


def test_compress_version():
    assert compress_version("clip-x", 8, 224) == "clip-x__fps8_s224_stretch"
    assert (
        compress_version("dinov2", 7.5, 336, "letterbox")
        == "dinov2__fps7.5_s336_letterbox"
    )
    # CONTRACTS_V2 §10.3：版本号保留 encoder@revision 原始形式
    raw = compress_version(
        "openai/clip-vit-base-patch16@57c216476eefef5ab752ec549e440a49ae4ae5f3",
        8,
        224,
        "letterbox",
    )
    assert raw == (
        "openai/clip-vit-base-patch16@57c216476eefef5ab752ec549e440a49ae4ae5f3"
        "__fps8_s224_letterbox"
    )
    # 文件系统安全目录名另行 slug（可选辅助函数）
    assert visual_version_slug(raw) == (
        "openai-clip-vit-base-patch16-57c216476eefef5ab752ec549e440a49ae4ae5f3"
        "__fps8_s224_letterbox"
    )
    with pytest.raises(DenseVisualError):
        compress_version("", 8, 224)
    with pytest.raises(DenseVisualError, match="preprocess"):
        compress_version("clip-x", 8, 224, "crop")
    with pytest.raises(DenseVisualError, match="version"):
        visual_version_slug("")


# ──────────────────────────────────────────────────────────────────────────────
# 保存 / 读取
# ──────────────────────────────────────────────────────────────────────────────


def _extract_small_features(videos: dict[str, Path]) -> DenseVisualFeatures:
    return extract_dense_features(
        videos["silent_6s.mp4"], FakeVisualEncoder(dim=6), fps=2, size=16, batch_size=4
    )


def test_save_load_roundtrip(tmp_path: Path, videos: dict[str, Path]):
    features = _extract_small_features(videos)
    saved = save_features(features, tmp_path / "out")
    assert saved["npz"] == tmp_path / "out" / DENSE_VISUAL_NPZ_NAME
    assert saved["manifest"] == tmp_path / "out" / DENSE_VISUAL_MANIFEST_NAME
    assert saved["npz"].is_file() and saved["manifest"].is_file()

    with np.load(saved["npz"], allow_pickle=False) as data:
        assert data["per_frame"].dtype == np.float16
        assert int(np.asarray(data["n_frames"]).item()) == features.n_frames
        assert str(np.asarray(data["version"]).item()) == features.version

    manifest = json.loads(saved["manifest"].read_text(encoding="utf-8"))
    for key in (
        "version",
        "encoder",
        "revision",
        "fps",
        "size",
        "preprocess",
        "duration_s",
        "n_frames",
        "dim",
        "n_windows",
        "window_times",
        "created_at",
    ):
        assert key in manifest, key
    assert manifest["version"] == features.version
    assert manifest["encoder"] == features.encoder
    assert manifest["dtype"] == "float16"
    assert manifest["preprocess"] == "stretch"
    assert manifest["n_windows"] == 0
    assert manifest["window_times"] == []

    loaded = load_features(tmp_path / "out")
    assert loaded is not None
    assert loaded.per_frame.shape == features.per_frame.shape
    assert loaded.n_frames == features.n_frames
    assert loaded.dim == features.dim
    assert loaded.fps == pytest.approx(features.fps)
    assert loaded.duration_s == pytest.approx(features.duration_s)
    assert loaded.version == features.version
    assert loaded.encoder == features.encoder
    assert loaded.size == features.size
    assert loaded.preprocess == "stretch"
    assert np.allclose(loaded.per_frame, features.per_frame, atol=2e-3)
    bundle = load_bundle(tmp_path / "out")
    assert bundle is not None and bundle.windows is None and bundle.window_times is None
    assert np.allclose(np.linalg.norm(loaded.per_frame, axis=1), 1.0, atol=2e-3)
    leftovers = [path.name for path in saved["npz"].parent.iterdir() if ".tmp" in path.name]
    assert leftovers == []


def test_save_load_alias_roundtrip(tmp_path: Path, videos: dict[str, Path]):
    features = _extract_small_features(videos)
    saved = save_dense_visual(features, tmp_path / "alias")
    assert saved["npz"].is_file()
    loaded = load_dense_visual(tmp_path / "alias")
    assert loaded is not None and loaded.n_frames == features.n_frames


def test_load_missing_returns_none(tmp_path: Path):
    assert load_features(tmp_path / "missing") is None
    (tmp_path / "partial").mkdir()
    (tmp_path / "partial" / DENSE_VISUAL_NPZ_NAME).write_bytes(b"not-a-real-npz")
    assert load_features(tmp_path / "partial") is None


def test_load_corrupt_manifest_raises(tmp_path: Path, videos: dict[str, Path]):
    features = _extract_small_features(videos)
    saved = save_features(features, tmp_path / "out")
    saved["manifest"].write_text("{not valid json", encoding="utf-8")
    with pytest.raises(DenseVisualError, match="manifest"):
        load_features(tmp_path / "out")


def test_load_corrupt_npz_raises(tmp_path: Path, videos: dict[str, Path]):
    features = _extract_small_features(videos)
    saved = save_features(features, tmp_path / "out")
    saved["npz"].write_bytes(b"definitely not an npz")
    with pytest.raises(DenseVisualError, match="npz"):
        load_features(tmp_path / "out")


def test_load_version_mismatch_raises(tmp_path: Path, videos: dict[str, Path]):
    features = _extract_small_features(videos)
    saved = save_features(features, tmp_path / "out")
    with np.load(saved["npz"], allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    arrays["version"] = np.asarray("other-version")
    np.savez_compressed(saved["npz"], **arrays)
    with pytest.raises(DenseVisualError, match="版本不一致"):
        load_features(tmp_path / "out")


def test_save_rejects_non_features(tmp_path: Path):
    with pytest.raises(DenseVisualError, match="DenseVisualFeatures"):
        save_features({"a": 1}, tmp_path)  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────────────────
# HFImageEncoder（不加载模型/不联网）
# ──────────────────────────────────────────────────────────────────────────────


def test_hf_encoder_init_validation(tmp_path: Path):
    with pytest.raises(DenseVisualError, match="model_id"):
        HFImageEncoder("")
    with pytest.raises(DenseVisualError, match="revision"):
        HFImageEncoder("openai/clip-vit-base-patch16", revision=5)  # type: ignore[arg-type]
    with pytest.raises(DenseVisualError, match="device"):
        HFImageEncoder("openai/clip-vit-base-patch16", device="tpu")
    with pytest.raises(DenseVisualError, match="dtype"):
        HFImageEncoder("openai/clip-vit-base-patch16", dtype="float16")


def test_hf_encoder_is_lazy_and_close_is_idempotent(tmp_path: Path):
    encoder = HFImageEncoder(
        "openai/clip-vit-base-patch16",
        revision="9f888eea",
        cache_dir=tmp_path,
    )
    assert encoder.version == "openai/clip-vit-base-patch16@9f888eea"
    assert encoder.model_load_s == 0.0
    assert encoder.dim == 0
    encoder.close()
    encoder.close()
    assert encoder.model_load_s == 0.0


def test_hf_encoder_resolves_hub_cache_dir(tmp_path: Path):
    plain = HFImageEncoder("x/y", cache_dir=tmp_path)
    assert plain.cache_dir == tmp_path
    (tmp_path / "hub").mkdir()
    with_hub = HFImageEncoder("x/y", cache_dir=tmp_path)
    assert with_hub.cache_dir == tmp_path / "hub"


def test_module_import_does_not_import_torch_or_transformers():
    code = (
        "import sys; import nebula.dense_visual;"
        "print(int('torch' in sys.modules), int('transformers' in sys.modules))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "0 0", proc.stdout


def test_fake_encoder_satisfies_protocol():
    assert isinstance(FakeVisualEncoder(), VisualEncoder)


# ──────────────────────────────────────────────────────────────────────────────
# pilot CLI
# ──────────────────────────────────────────────────────────────────────────────


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("dense_visual_pilot", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_help_runs_offline():
    proc = subprocess.run(
        [sys.executable, str(CLI_PATH), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "密集视觉" in proc.stdout
    for flag in (
        "--encoder",
        "--local-relpath",
        "--fps",
        "--preprocess",
        "--limit",
        "--out-dir",
        "--save-bundle",
        "--json",
    ):
        assert flag in proc.stdout, flag


def test_cli_parser_and_encoder_aliases():
    module = _load_cli_module()
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--local-relpath",
            "a/b.mp4",
            "--encoder",
            "dinov2-small",
            "--fps",
            "8",
            "--limit",
            "3",
            "--out-dir",
            "temp/x",
            "--json",
        ]
    )
    assert args.local_relpath == "a/b.mp4"
    assert args.source is None
    assert args.fps == 8.0
    assert args.limit == 3
    assert args.out_dir == "temp/x"
    assert args.json is True
    assert args.preprocess is None
    assert args.save_bundle is False
    explicit = parser.parse_args(
        ["--source", "v.mp4", "--preprocess", "letterbox", "--save-bundle"]
    )
    assert explicit.preprocess == "letterbox"
    assert explicit.save_bundle is True

    config = load_config(REPO_ROOT / "config.yaml")
    model, revision = module.resolve_encoder_spec(config, encoder=None, revision=None)
    assert model == config.visual_dense.encoders[0].model
    assert revision == config.visual_dense.encoders[0].revision
    with pytest.raises(DenseVisualError, match="revision"):
        module.resolve_encoder_spec(config, encoder="unknown/no-revision", revision=None)
    assert module.resolve_encoder_id("dinov2-small") == "facebook/dinov2-small"
    assert module.resolve_encoder_id("openai/clip-vit-base-patch16") == (
        "openai/clip-vit-base-patch16"
    )
    with pytest.raises(DenseVisualError, match="encoder"):
        module.resolve_encoder_id("")


def test_cli_missing_source_reports_error_offline(tmp_path: Path):
    proc = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--source",
            str(tmp_path / "missing.mp4"),
            "--encoder",
            "fake/model",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 1
    assert "不存在" in proc.stderr


# ──────────────────────────────────────────────────────────────────────────────
# v0.4.3：预处理 filter / letterbox 解码 / 版本
# ──────────────────────────────────────────────────────────────────────────────


def test_build_video_filter_stretch_and_letterbox():
    assert PREPROCESS_MODES == ("stretch", "letterbox")
    assert build_video_filter(fps=8, size=224) == "fps=8,scale=224:224,format=rgb24"
    assert (
        build_video_filter(fps=8, size=224, preprocess="stretch")
        == "fps=8,scale=224:224,format=rgb24"
    )
    assert build_video_filter(
        fps=7.5, size=336, preprocess="letterbox"
    ) == (
        "fps=7.5,scale=336:336:force_original_aspect_ratio=decrease,"
        "pad=336:336:(ow-iw)/2:(oh-ih)/2:color=black,format=rgb24"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"preprocess": "crop"},
        {"preprocess": ""},
        {"fps": 0},
        {"fps": -1},
        {"size": 1},
        {"size": 0},
    ],
)
def test_build_video_filter_invalid_params(kwargs: dict):
    arguments = {"fps": 2, "size": 16, "preprocess": "stretch", **kwargs}
    with pytest.raises(DenseVisualError):
        build_video_filter(**arguments)


def test_iter_frame_batches_letterbox_black_bars(videos: dict[str, Path]):
    batch = next(
        iter(
            iter_frame_batches(
                videos["short_4s.mp4"],
                fps=2,
                size=64,
                batch_size=1,
                preprocess="letterbox",
            )
        )
    )
    frame = batch[0]
    assert frame.shape == (64, 64, 3)
    # 160x120 (4:3) -> 64x48，上下各 8 行黑边
    assert np.all(frame[:8] == 0)
    assert np.all(frame[-8:] == 0)
    assert np.any(frame[8:-8] != 0), "letterbox 中间区域应有画面内容"

    stretched = next(
        iter(
            iter_frame_batches(
                videos["short_4s.mp4"], fps=2, size=64, batch_size=1, preprocess="stretch"
            )
        )
    )
    assert stretched[0].shape == (64, 64, 3)
    assert np.any(stretched[0] != 0)


def test_extract_features_letterbox_version_and_preprocess(videos: dict[str, Path]):
    features = extract_dense_features(
        videos["short_4s.mp4"],
        FakeVisualEncoder(dim=7),
        fps=2,
        size=64,
        batch_size=2,
        preprocess="letterbox",
    )
    assert features.preprocess == "letterbox"
    assert features.version.endswith("__fps2_s64_letterbox")
    assert "fps2" in features.version and "s64" in features.version
    assert np.allclose(np.linalg.norm(features.per_frame, axis=1), 1.0, atol=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# v0.4.3：bundle 保存 / 读取（帧级 + 窗口级）
# ──────────────────────────────────────────────────────────────────────────────


def _bundle_inputs(videos: dict[str, Path]):
    features = _extract_small_features(videos)
    window_pairs = [(0.0, 3.0), (3.0, 6.0)]
    pooled = pool_windows(features, window_pairs)
    std = pool_windows_std(features, window_pairs)
    return features, pooled, std, window_pairs


def test_save_load_bundle_roundtrip(tmp_path: Path, videos: dict[str, Path]):
    features, pooled, std, window_pairs = _bundle_inputs(videos)
    target = tmp_path / "bundle"
    saved = save_bundle(
        target,
        features,
        windows=pooled,
        window_std=std,
        window_times=window_pairs,
        extra={"work_id": 12},
    )
    assert set(saved) == {"npz", "windows", "window_std", "manifest"}
    assert saved["npz"] == target / DENSE_VISUAL_NPZ_NAME
    assert saved["windows"] == target / WINDOWS_NPY_NAME
    assert saved["window_std"] == target / WINDOW_STD_NPY_NAME
    assert saved["manifest"] == target / DENSE_VISUAL_MANIFEST_NAME
    for path in saved.values():
        assert path.is_file()

    manifest = json.loads(saved["manifest"].read_text(encoding="utf-8"))
    for key in (
        "version",
        "encoder",
        "revision",
        "fps",
        "size",
        "preprocess",
        "duration_s",
        "n_frames",
        "dim",
        "n_windows",
        "window_times",
        "window_weights",
        "created_at",
    ):
        assert key in manifest, key
    assert manifest["n_windows"] == 2
    assert manifest["window_times"] == [[0.0, 3.0], [3.0, 6.0]]
    assert manifest["window_weights"] is None  # 未传权重时保持旧行为
    assert manifest["preprocess"] == "stretch"
    assert manifest["extra"] == {"work_id": 12}

    windows_array = np.load(saved["windows"], allow_pickle=False)
    assert windows_array.dtype == np.float32
    assert windows_array.shape == pooled.shape
    std_array = np.load(saved["window_std"], allow_pickle=False)
    assert std_array.dtype == np.float32
    assert np.all(std_array >= 0.0)

    bundle = load_bundle(target)
    assert isinstance(bundle, VisualBundle)
    assert bundle.features.version == features.version
    assert bundle.manifest["n_windows"] == 2
    assert bundle.windows is not None and bundle.window_std is not None
    assert bundle.windows.dtype == np.float32
    assert np.allclose(bundle.windows, pooled, atol=1e-6)
    assert np.allclose(bundle.window_std, std, atol=1e-6)
    assert bundle.window_times == ((0.0, 3.0), (3.0, 6.0))
    assert bundle.window_weights is None
    assert np.allclose(np.linalg.norm(bundle.windows, axis=1), 1.0, atol=1e-5)
    leftovers = [path.name for path in target.iterdir() if ".tmp" in path.name]
    assert leftovers == []


def test_save_bundle_feature_only(tmp_path: Path, videos: dict[str, Path]):
    features = _extract_small_features(videos)
    target = tmp_path / "features-only"
    saved = save_bundle(target, features)
    assert set(saved) == {"npz", "manifest"}
    assert not (target / WINDOWS_NPY_NAME).exists()
    assert not (target / WINDOW_STD_NPY_NAME).exists()

    bundle = load_bundle(target)
    assert isinstance(bundle, VisualBundle)
    assert bundle.windows is None
    assert bundle.window_std is None
    assert bundle.window_times is None
    assert bundle.window_weights is None
    assert bundle.manifest["n_windows"] == 0
    assert bundle.manifest["window_times"] == []
    assert bundle.manifest["window_weights"] is None


def test_save_bundle_accepts_plain_lists(tmp_path: Path, videos: dict[str, Path]):
    features, pooled, std, window_pairs = _bundle_inputs(videos)
    save_bundle(
        tmp_path / "lists",
        features,
        windows=pooled.tolist(),
        window_std=std.tolist(),
        window_times=[list(pair) for pair in window_pairs],
    )
    bundle = load_bundle(tmp_path / "lists")
    assert bundle is not None and bundle.windows is not None
    assert np.allclose(bundle.windows, pooled, atol=1e-6)


def test_save_bundle_validation_errors(tmp_path: Path, videos: dict[str, Path]):
    features, pooled, std, window_pairs = _bundle_inputs(videos)
    with pytest.raises(DenseVisualError, match="window_std"):
        save_bundle(tmp_path / "a", features, window_std=std)
    with pytest.raises(DenseVisualError, match="window_times"):
        save_bundle(tmp_path / "b", features, windows=pooled)
    with pytest.raises(DenseVisualError, match="window_times"):
        save_bundle(tmp_path / "c", features, window_times=window_pairs)
    with pytest.raises(DenseVisualError, match="维度"):
        save_bundle(
            tmp_path / "d",
            features,
            windows=pooled[:, :1],
            window_times=window_pairs,
        )
    with pytest.raises(DenseVisualError, match="长度"):
        save_bundle(
            tmp_path / "e",
            features,
            windows=pooled,
            window_times=[(0.0, 1.0)],
        )
    with pytest.raises(DenseVisualError):
        save_bundle(
            tmp_path / "f",
            features,
            windows=pooled,
            window_times=[(2.0, 1.0), (3.0, 6.0)],
        )
    negative = np.full_like(std, -1.0)
    with pytest.raises(DenseVisualError, match="负"):
        save_bundle(
            tmp_path / "g",
            features,
            windows=pooled,
            window_std=negative,
            window_times=window_pairs,
        )
    with pytest.raises(DenseVisualError, match="extra"):
        save_bundle(tmp_path / "h", features, extra={"bad": object()})  # type: ignore[dict-item]


def test_save_load_bundle_window_weights_roundtrip(
    tmp_path: Path, videos: dict[str, Path]
):
    features, pooled, std, window_pairs = _bundle_inputs(videos)
    weights = coverage_weights(window_pairs, features.duration_s)
    assert len(weights) == pooled.shape[0]
    assert sum(weights) == pytest.approx(1.0, abs=1e-6)
    target = tmp_path / "weights"
    saved = save_bundle(
        target,
        features,
        windows=pooled,
        window_std=std,
        window_times=window_pairs,
        window_weights=weights,
    )
    manifest = json.loads(saved["manifest"].read_text(encoding="utf-8"))
    assert manifest["window_weights"] == pytest.approx(list(weights), abs=1e-12)

    bundle = load_bundle(target)
    assert isinstance(bundle, VisualBundle)
    assert bundle.window_weights is not None
    assert bundle.window_weights.dtype == np.float64
    assert bundle.window_weights.shape == (pooled.shape[0],)
    assert np.allclose(bundle.window_weights, weights, atol=1e-12)
    assert float(bundle.window_weights.sum()) == pytest.approx(1.0, abs=1e-6)

    # 允许 list 输入且与窗口一一对应
    save_bundle(
        tmp_path / "weights-list",
        features,
        windows=pooled.tolist(),
        window_times=[list(pair) for pair in window_pairs],
        window_weights=list(weights),
    )
    listed = load_bundle(tmp_path / "weights-list")
    assert listed is not None and listed.window_weights is not None
    assert np.allclose(listed.window_weights, weights, atol=1e-12)


def test_save_bundle_window_weights_validation(
    tmp_path: Path, videos: dict[str, Path]
):
    features, pooled, _std, window_pairs = _bundle_inputs(videos)
    with pytest.raises(DenseVisualError, match="window_weights"):
        save_bundle(
            tmp_path / "a", features, window_weights=[0.5, 0.5]
        )
    with pytest.raises(DenseVisualError, match="长度"):
        save_bundle(
            tmp_path / "b",
            features,
            windows=pooled,
            window_times=window_pairs,
            window_weights=[1.0],
        )
    with pytest.raises(DenseVisualError, match="负"):
        save_bundle(
            tmp_path / "c",
            features,
            windows=pooled,
            window_times=window_pairs,
            window_weights=[1.5, -0.5],
        )
    with pytest.raises(DenseVisualError, match="NaN"):
        save_bundle(
            tmp_path / "d",
            features,
            windows=pooled,
            window_times=window_pairs,
            window_weights=[0.5, float("nan")],
        )
    with pytest.raises(DenseVisualError, match="之和"):
        save_bundle(
            tmp_path / "e",
            features,
            windows=pooled,
            window_times=window_pairs,
            window_weights=[0.0, 0.0],
        )
    with pytest.raises(DenseVisualError, match="一维"):
        save_bundle(
            tmp_path / "f",
            features,
            windows=pooled,
            window_times=window_pairs,
            window_weights=np.asarray([[0.5, 0.5]], dtype=np.float64),
        )


def test_load_bundle_legacy_manifest_without_weights(
    tmp_path: Path, videos: dict[str, Path]
):
    features, pooled, _std, window_pairs = _bundle_inputs(videos)
    weights = coverage_weights(window_pairs, features.duration_s)
    target = tmp_path / "legacy-no-weights"
    saved = save_bundle(
        target,
        features,
        windows=pooled,
        window_times=window_pairs,
        window_weights=weights,
    )
    manifest = json.loads(saved["manifest"].read_text(encoding="utf-8"))
    manifest.pop("window_weights")  # 模拟 v0.4.3 旧 bundle（无该字段）
    saved["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    bundle = load_bundle(target)
    assert bundle is not None
    assert bundle.window_weights is None
    assert bundle.windows is not None


def test_load_bundle_accepts_legacy_logical_vs_slug_version(
    tmp_path: Path, videos: dict[str, Path]
):
    """v0.4.3 旧数据：npz 存逻辑版本、manifest 存 slug 版本时应可读取。"""

    logical_version = (
        "openai/clip-vit-base-patch16@57c216476eefef5ab752ec549e440a49ae4ae5f3"
        "__fps2_s16_letterbox"
    )
    features = _make_features(
        _unit([[1, 0], [0, 1]]),
        fps=1.0,
        duration_s=2.0,
        version=logical_version,
        preprocess="letterbox",
    )
    windows = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    window_pairs = [(0.0, 1.0), (1.0, 2.0)]
    target = tmp_path / "legacy-version"
    saved = save_bundle(
        target,
        features,
        windows=windows,
        window_times=window_pairs,
    )
    manifest = json.loads(saved["manifest"].read_text(encoding="utf-8"))
    slug_version = visual_version_slug(logical_version)
    manifest["version"] = slug_version
    saved["manifest"].write_text(json.dumps(manifest), encoding="utf-8")

    bundle = load_bundle(target)
    assert bundle is not None
    # 兼容读取时采用 manifest（DB/目录）版本身份
    assert bundle.features.version == slug_version
    assert bundle.manifest["version"] == slug_version
    assert bundle.windows is not None
    loaded = load_features(target)
    assert loaded is not None and loaded.version == slug_version

    # 非 slug 等价的版本仍然报错
    manifest["version"] = "totally-different-version"
    saved["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DenseVisualError, match="版本不一致"):
        load_bundle(target)


def test_load_bundle_corrupt_window_weights_raises(
    tmp_path: Path, videos: dict[str, Path]
):
    features, pooled, _std, window_pairs = _bundle_inputs(videos)
    target = tmp_path / "bad-weights"
    saved = save_bundle(
        target,
        features,
        windows=pooled,
        window_times=window_pairs,
        window_weights=[0.5, 0.5],
    )
    manifest = json.loads(saved["manifest"].read_text(encoding="utf-8"))
    manifest["window_weights"] = [1.0]  # 长度与 n_windows 不一致
    saved["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DenseVisualError, match="window_weights"):
        load_bundle(target)


def test_load_bundle_legacy_manifest_fallback(tmp_path: Path, videos: dict[str, Path]):
    features = _extract_small_features(videos)
    target = tmp_path / "legacy"
    saved = save_features(features, target)
    manifest = json.loads(saved["manifest"].read_text(encoding="utf-8"))
    # 模拟旧版仅帧级 bundle：去掉 v0.4.3 新增字段，manifest 文件名回退
    for key in ("preprocess", "n_windows", "window_times"):
        manifest.pop(key, None)
    manifest["files"] = {
        "per_frame": DENSE_VISUAL_NPZ_NAME,
        "manifest": DENSE_VISUAL_LEGACY_MANIFEST_NAME,
    }
    legacy_path = target / DENSE_VISUAL_LEGACY_MANIFEST_NAME
    legacy_path.write_text(json.dumps(manifest), encoding="utf-8")
    saved["manifest"].unlink()

    bundle = load_bundle(target)
    assert bundle is not None
    assert bundle.features.version == features.version
    assert bundle.features.preprocess == "stretch"
    assert bundle.windows is None and bundle.window_times is None


def test_load_bundle_missing_windows_raises(tmp_path: Path, videos: dict[str, Path]):
    features, pooled, _std, window_pairs = _bundle_inputs(videos)
    target = tmp_path / "missing-windows"
    saved = save_bundle(
        target, features, windows=pooled, window_times=window_pairs
    )
    saved["windows"].unlink()
    with pytest.raises(DenseVisualError, match="缺少窗口"):
        load_bundle(target)


def test_load_bundle_corrupt_windows_raises(tmp_path: Path, videos: dict[str, Path]):
    features, pooled, _std, window_pairs = _bundle_inputs(videos)
    target = tmp_path / "corrupt-windows"
    saved = save_bundle(
        target, features, windows=pooled, window_times=window_pairs
    )
    np.save(saved["windows"], np.ones((1, 3), dtype=np.float32))
    with pytest.raises(DenseVisualError, match="窗口"):
        load_bundle(target)


def test_visual_bundle_validation():
    features = _make_features(_unit([[1, 0], [0, 1]]), fps=1.0, duration_s=2.0)
    windows = np.asarray([[1.0, 0.0]], dtype=np.float32)
    with pytest.raises(DenseVisualError, match="window_times"):
        VisualBundle(features=features, windows=windows)
    with pytest.raises(DenseVisualError, match="window_times"):
        VisualBundle(features=features, window_times=((0.0, 1.0),))
    with pytest.raises(DenseVisualError, match="window_std"):
        VisualBundle(features=features, window_std=windows)
    with pytest.raises(DenseVisualError, match="window_weights"):
        VisualBundle(features=features, window_weights=np.asarray([1.0]))
    with pytest.raises(DenseVisualError, match="长度"):
        VisualBundle(
            features=features,
            windows=windows,
            window_times=[(0.0, 1.0)],
            window_weights=np.asarray([1.0, 0.0]),
        )


# ──────────────────────────────────────────────────────────────────────────────
# v0.4.3：预处理对照工具（只读 DB / 纯函数部分）
# ──────────────────────────────────────────────────────────────────────────────


def _load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_cli_help_runs_offline():
    proc = subprocess.run(
        [sys.executable, str(COMPARE_CLI_PATH), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "预处理对照" in proc.stdout
    for flag in ("--work-ids", "--encoder", "--max-windows", "--out", "--json"):
        assert flag in proc.stdout, flag


def test_compare_cli_parser_and_encoder_spec():
    module = _load_module_from_path(COMPARE_CLI_PATH, "dense_preprocess_compare")
    parser = module.build_parser()
    args = parser.parse_args(
        ["--work-ids", "1", "2", "--encoder", "dinov2-small", "--max-windows", "3", "--json"]
    )
    assert args.work_ids == [1, 2]
    assert args.max_windows == 3
    assert args.json is True
    assert module.resolve_encoder_id("dinov2-small") == "facebook/dinov2-small"

    config = load_config(REPO_ROOT / "config.yaml")
    default_model, default_revision = module.resolve_encoder_spec(
        config, encoder=None, revision=None
    )
    assert default_model == config.visual_dense.encoders[0].model
    assert default_revision == config.visual_dense.encoders[0].revision
    with pytest.raises(DenseVisualError, match="revision"):
        module.resolve_encoder_spec(
            config, encoder="unknown/model-without-revision", revision=None
        )


def test_compare_tool_reads_catalog_read_only(config):
    module = _load_module_from_path(COMPARE_CLI_PATH, "dense_preprocess_compare")
    from nebula.db import Catalog

    catalog = Catalog(config.paths.db_path)
    catalog.initialize()
    try:
        first, _ = catalog.upsert_local_work("a.mp4")
        second, _ = catalog.upsert_local_work("b.mp4")
        for work_id in (first, second):
            catalog.begin_attempt(work_id, "encoding")
            catalog.complete(work_id)
    finally:
        catalog.close()

    works = module.load_local_completed_works(config.paths.db_path, limit=1)
    assert len(works) == 1
    assert int(works[0]["id"]) == first
    ordered = module.load_local_completed_works(
        config.paths.db_path, work_ids=[second, first]
    )
    assert [int(item["id"]) for item in ordered] == [second, first]
    with pytest.raises(NebulaError, match="work_id"):
        module.load_local_completed_works(config.paths.db_path, work_ids=[999])


def test_compare_cosine_stats_and_summarize():
    module = _load_module_from_path(COMPARE_CLI_PATH, "dense_preprocess_compare")
    left = _unit([[1, 0], [0, 1]])
    same = module._cosine_stats(left, left.copy())
    assert same["mean"] == pytest.approx(1.0)
    assert same["min"] == pytest.approx(1.0)
    flipped = _unit([[-1, 0], [0, 1]])
    mixed = module._cosine_stats(left, flipped)
    assert mixed["mean"] == pytest.approx(0.0)
    assert mixed["min"] == pytest.approx(-1.0)
    assert mixed["max"] == pytest.approx(1.0)

    results = [
        {
            "frames_equal": True,
            "cosine": {"mean": 0.9, "min": 0.8},
            "timing_ratio": {
                "letterbox_over_stretch_decode": 1.1,
                "letterbox_over_stretch_encode": 1.05,
            },
            "modes": {
                "stretch": {"version": "v-stretch"},
                "letterbox": {"version": "v-letterbox"},
            },
        },
        {
            "frames_equal": True,
            "cosine": {"mean": 0.95, "min": 0.9},
            "timing_ratio": {
                "letterbox_over_stretch_decode": 1.2,
                "letterbox_over_stretch_encode": 1.1,
            },
            "modes": {
                "stretch": {"version": "v-stretch"},
                "letterbox": {"version": "v-letterbox"},
            },
        },
    ]
    summary = module.summarize(results)
    assert summary["n_works"] == 2
    assert summary["window_cosine_mean"] == pytest.approx(0.925)
    assert summary["window_cosine_min"] == pytest.approx(0.8)
    assert summary["frames_equal_all"] is True
    assert summary["mean_decode_ratio_letterbox_over_stretch"] == pytest.approx(1.15)
    assert summary["versions"]["stretch"] == ["v-stretch"]

