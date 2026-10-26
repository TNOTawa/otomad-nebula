"""v0.5-beta 密集音频模块（nebula.audio_dense + CLI）的离线单元测试。

默认测试不加载真实 MERT、不联网：用 fake 帧编码器 + ffmpeg 合成正弦视频验证
ffmpeg 解码/窗口/时间映射/bin 聚合/断点复用/版本隔离；真实模型 pilot 由
tools/audio_dense.py 单独运行并落 logs/audio_dense_pilot.json。
"""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nebula.audio_dense import (
    AUDIO_MANIFEST_NAME,
    AUDIO_NPZ_NAME,
    AudioDenseError,
    AudioDenseFeatures,
    HFAudioEncoder,
    compose_audio_version,
    extract_audio_features,
    is_audio_bundle_complete,
    iter_audio_windows,
    load_audio_bundle,
    save_audio_bundle,
)
from nebula.db import Catalog

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "tools" / "audio_dense.py"


class FakeAudioEncoder:
    """确定性 fake 音频帧编码器（不依赖 torch）：frame_rate=50Hz、dim=8。"""

    version = "fake-audio-v1"
    sample_rate = 16000
    frame_rate = 50.0
    dim = 8
    model_load_s = 0.0
    frame_stride_samples = 320
    frame_receptive_samples = 400
    layer = 3

    def __init__(self, *, bad_shape: bool = False, nan: bool = False) -> None:
        self.bad_shape = bad_shape
        self.nan = nan
        self.calls = 0
        self.closed = 0

    def encode(
        self, waveforms: np.ndarray, valid_samples: list[int] | tuple[int, ...]
    ) -> list[np.ndarray]:
        self.calls += 1
        result: list[np.ndarray] = []
        for row, length in enumerate(valid_samples):
            count = max(1, (int(length) - 400) // 320 + 1)
            base = (
                np.arange(count, dtype=np.float32)[:, None] * 0.01
                + np.arange(self.dim, dtype=np.float32)[None, :] * 0.1
                + np.float32(row) * 0.01
            )
            result.append(base.astype(np.float32))
        if self.bad_shape:
            return [item[:, :4] for item in result]
        if self.nan:
            result[0][0, 0] = float("nan")
        return result

    def close(self) -> None:
        self.closed += 1


@pytest.fixture(scope="module")
def sine_video(tmp_path_factory) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("缺少 ffmpeg")
    target = tmp_path_factory.mktemp("audio_dense") / "sine6.mp4"
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
            "sine=frequency=440:sample_rate=48000:duration=6",
            "-c:a",
            "aac",
            str(target),
        ],
        check=True,
        capture_output=True,
    )
    return target


def _extract(
    source: Path, encoder: FakeAudioEncoder | None = None, **kwargs
) -> AudioDenseFeatures:
    encoder = encoder if encoder is not None else FakeAudioEncoder()
    arguments = {
        "context_s": 2.0,
        "hop_s": 2.0,
        "bin_seconds": 0.5,
        "batch_size": 2,
    }
    arguments.update(kwargs)
    return extract_audio_features(source, encoder, **arguments)  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────────────────
# 版本与参数
# ──────────────────────────────────────────────────────────────────────────────


def test_compose_audio_version_contains_all_fields():
    version = compose_audio_version(
        "m-a-p/MERT-v1-95M@abcdef",
        sample_rate=24000,
        context_s=5.0,
        hop_s=5.0,
        layer=9,
        pool="mean_std",
        bin_seconds=0.5,
        normalize="l2",
    )
    for token in (
        "m-a-p-MERT-v1-95M-abcdef",
        "sr24000",
        "ctx5",
        "hop5",
        "l9",
        "poolmean_std",
        "bin0.5",
        "norml2",
    ):
        assert token in version, token
    assert "/" not in version and "@" not in version
    assert version == compose_audio_version(
        "m-a-p/MERT-v1-95M@abcdef",
        sample_rate=24000,
        context_s=5.0,
        hop_s=5.0,
        layer=9,
        pool="mean_std",
        bin_seconds=0.5,
        normalize="l2",
    )


def test_compose_audio_version_none_layer():
    version = compose_audio_version(
        "x",
        sample_rate=16000,
        context_s=1.0,
        hop_s=1.0,
        layer=None,
        pool="mean",
        bin_seconds=0.5,
        normalize="none",
    )
    assert "lnone" in version and "poolmean" in version and "normnone" in version


def test_compose_audio_version_invalid():
    with pytest.raises(AudioDenseError):
        compose_audio_version(
            "",
            sample_rate=16000,
            context_s=1.0,
            hop_s=1.0,
            layer=1,
            pool="mean",
            bin_seconds=0.5,
            normalize="l2",
        )
    with pytest.raises(AudioDenseError, match="hop_s"):
        compose_audio_version(
            "x",
            sample_rate=16000,
            context_s=1.0,
            hop_s=2.0,
            layer=1,
            pool="mean",
            bin_seconds=0.5,
            normalize="l2",
        )
    with pytest.raises(AudioDenseError, match="pool"):
        compose_audio_version(
            "x",
            sample_rate=16000,
            context_s=1.0,
            hop_s=1.0,
            layer=1,
            pool="max",
            bin_seconds=0.5,
            normalize="l2",
        )
    with pytest.raises(AudioDenseError, match="normalize"):
        compose_audio_version(
            "x",
            sample_rate=16000,
            context_s=1.0,
            hop_s=1.0,
            layer=1,
            pool="mean",
            bin_seconds=0.5,
            normalize="zscore",
        )
    with pytest.raises(AudioDenseError, match="sample_rate"):
        compose_audio_version(
            "x",
            sample_rate=0,
            context_s=1.0,
            hop_s=1.0,
            layer=1,
            pool="mean",
            bin_seconds=0.5,
            normalize="l2",
        )


# ──────────────────────────────────────────────────────────────────────────────
# ffmpeg 解码 / 时间映射 / bin 聚合
# ──────────────────────────────────────────────────────────────────────────────


def test_iter_audio_windows(sine_video: Path):
    stats: dict[str, float] = {}
    windows = list(
        iter_audio_windows(
            sine_video,
            sample_rate=16000,
            context_s=2.0,
            hop_s=2.0,
            stats=stats,
        )
    )
    assert len(windows) >= 3
    for index, (start_s, waveform, actual) in enumerate(windows):
        assert start_s == pytest.approx(index * 2.0, abs=1e-9)
        assert waveform.shape == (32000,)
        assert waveform.dtype == np.float32
        assert 1 <= actual <= 32000
    assert stats["decoded_samples"] > 0
    assert stats["decoded_duration_s"] == pytest.approx(6.0, abs=0.1)
    # 最后一个窗口允许不足 2s（AAC 编码延迟）
    assert windows[-1][2] <= 32000


def test_iter_audio_windows_max_duration(sine_video: Path):
    stats: dict[str, float] = {}
    windows = list(
        iter_audio_windows(
            sine_video,
            sample_rate=16000,
            context_s=2.0,
            hop_s=2.0,
            max_duration_s=1.0,
            stats=stats,
        )
    )
    assert windows
    assert stats["decoded_samples"] <= 16000
    assert stats["decoded_duration_s"] <= 1.0 + 1e-9


def test_extract_audio_features_duration_drift(sine_video: Path):
    features = _extract(sine_video)
    assert abs(features.decoded_duration_s - features.duration_s) <= 0.5


def test_iter_audio_windows_invalid(sine_video: Path):
    with pytest.raises(AudioDenseError):
        list(iter_audio_windows(sine_video, sample_rate=0, context_s=1.0, hop_s=1.0))
    with pytest.raises(AudioDenseError, match="hop_s"):
        list(iter_audio_windows(sine_video, sample_rate=16000, context_s=1.0, hop_s=2.0))
    with pytest.raises(AudioDenseError):
        list(
            iter_audio_windows(
                sine_video, sample_rate=16000, context_s=1.0, hop_s=1.0, timeout_s=0.0
            )
        )


def test_aggregate_bins_hand_calculated():
    from nebula.audio_dense import _aggregate_bins

    frames = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    times = np.asarray([0.1, 0.4, 0.6], dtype=np.float32)
    mean, std, mask, bin_times = _aggregate_bins(
        frames, times, duration_s=1.0, bin_seconds=0.5
    )
    assert mean.shape == (2, 2)
    assert mask.tolist() == [True, True]
    assert mean[0] == pytest.approx([0.5, 0.5])
    assert std[0] == pytest.approx([0.5, 0.5])
    assert mean[1] == pytest.approx([1.0, 1.0])
    assert std[1] == pytest.approx([0.0, 0.0])
    assert bin_times[0] == pytest.approx([0.0, 0.5])
    assert bin_times[1] == pytest.approx([0.5, 1.0])


def test_extract_audio_features(sine_video: Path):
    encoder = FakeAudioEncoder()
    features = _extract(sine_video, encoder)
    assert features.version.startswith("fake-audio-v1__sr16000_ctx2_hop2")
    assert features.dim == 8
    assert features.frame_features.ndim == 2 and features.frame_features.shape[1] == 8
    assert features.frame_features.shape[0] > 0
    assert features.frame_times.shape == (features.frame_features.shape[0],)
    assert np.all(np.diff(features.frame_times) >= -1e-6)
    # 帧特征在内存/落盘用 fp16 保存，L2 归一化后有 ~1e-3 量化误差
    norms = np.linalg.norm(features.frame_features, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-2)
    expected_bins = max(1, int(math.ceil(features.duration_s / 0.5 - 1e-9)))
    assert features.bin_features.shape == (expected_bins, 8)
    assert features.bin_std.shape == (expected_bins, 8)
    assert features.bin_mask.shape == (expected_bins,)
    assert bool(features.bin_mask.all())
    assert np.all(features.bin_std >= 0.0)
    assert features.decoded_duration_s > 0.0
    assert abs(features.decoded_duration_s - features.duration_s) <= 0.5
    assert features.sample_rate == 16000
    assert features.context_s == 2.0 and features.hop_s == 2.0
    assert features.bin_seconds == 0.5
    assert features.pool == "mean_std"
    assert features.normalize == "l2"
    assert features.layer == 3
    assert encoder.calls >= 1
    assert features.created_at


def test_extract_audio_features_version_override(sine_video: Path):
    features = _extract(sine_video, version="my_audio_v1")
    assert features.version == "my_audio_v1"


def test_extract_audio_features_normalize_none(sine_video: Path):
    features = _extract(sine_video, normalize="none")
    norms = np.linalg.norm(features.frame_features, axis=1)
    assert not np.allclose(norms, 1.0)


def test_extract_audio_features_errors(sine_video: Path, tmp_path: Path):
    with pytest.raises(AudioDenseError, match="不存在"):
        _extract(tmp_path / "missing.mp4")
    with pytest.raises(AudioDenseError, match="bin_seconds"):
        _extract(sine_video, bin_seconds=0.0)
    with pytest.raises(AudioDenseError, match="normalize"):
        _extract(sine_video, normalize="zscore")
    with pytest.raises(AudioDenseError, match="pool"):
        _extract(sine_video, pool="max")
    with pytest.raises(AudioDenseError, match="batch_size"):
        _extract(sine_video, batch_size=0)
    with pytest.raises(AudioDenseError, match="形状"):
        _extract(sine_video, encoder=FakeAudioEncoder(bad_shape=True))
    with pytest.raises(AudioDenseError, match="NaN"):
        _extract(sine_video, encoder=FakeAudioEncoder(nan=True))


# ──────────────────────────────────────────────────────────────────────────────
# 保存 / 读取 / 断点复用 / 版本隔离
# ──────────────────────────────────────────────────────────────────────────────


def _features(sine_video: Path) -> AudioDenseFeatures:
    return _extract(sine_video)


def test_save_load_roundtrip(tmp_path: Path, sine_video: Path):
    features = _features(sine_video)
    target = tmp_path / "audio_dense" / features.version / "000001"
    saved = save_audio_bundle(target, features)
    assert saved["npz"] == target / AUDIO_NPZ_NAME
    assert saved["manifest"] == target / AUDIO_MANIFEST_NAME
    assert saved["npz"].is_file() and saved["manifest"].is_file()
    assert "skipped" not in saved
    manifest = json.loads(saved["manifest"].read_text(encoding="utf-8"))
    for key in (
        "version",
        "encoder",
        "revision",
        "sample_rate",
        "context_s",
        "hop_s",
        "layer",
        "pool",
        "bin_seconds",
        "normalize",
        "frame_rate",
        "dim",
        "n_frames",
        "n_bins",
        "duration_s",
        "created_at",
    ):
        assert key in manifest, key
    assert manifest["n_frames"] == features.frame_features.shape[0]
    assert manifest["n_bins"] == features.bin_features.shape[0]
    with np.load(saved["npz"], allow_pickle=False) as data:
        assert data["frame_features"].dtype == np.float16
        assert data["bin_features"].dtype == np.float32
        assert data["bin_mask"].dtype == bool
        assert data["frame_times"].shape == data["frame_features"].shape[:1]

    loaded = load_audio_bundle(target)
    assert loaded is not None
    assert loaded.version == features.version
    assert loaded.frame_features.shape == features.frame_features.shape
    assert np.allclose(
        loaded.frame_features.astype(np.float32),
        features.frame_features.astype(np.float32),
        atol=2e-3,
    )
    assert np.allclose(loaded.bin_features, features.bin_features, atol=1e-6)
    assert loaded.bin_mask.tolist() == features.bin_mask.tolist()
    assert loaded.work_id == features.work_id
    assert is_audio_bundle_complete(target, version=features.version)
    assert load_audio_bundle(tmp_path / "missing") is None


def test_resume_skip_and_force(tmp_path: Path, sine_video: Path):
    features = _features(sine_video)
    target = tmp_path / "bundle"
    first = save_audio_bundle(target, features)
    assert "skipped" not in first
    second = save_audio_bundle(target, features)
    assert second.get("skipped") is True
    forced = save_audio_bundle(target, features, force=True)
    assert "skipped" not in forced


def test_version_isolation(tmp_path: Path, sine_video: Path):
    features = _extract(sine_video, version="version_a")
    target = tmp_path / "audio_dense" / "version_a" / "000001"
    save_audio_bundle(target, features)
    assert is_audio_bundle_complete(target, version="version_a")
    assert not is_audio_bundle_complete(target, version="version_b")
    assert load_audio_bundle(tmp_path / "audio_dense" / "version_b" / "000001") is None


def test_corrupt_bundle_raises(tmp_path: Path, sine_video: Path):
    features = _features(sine_video)
    target = tmp_path / "bundle"
    saved = save_audio_bundle(target, features)
    saved["manifest"].write_text("{bad json", encoding="utf-8")
    with pytest.raises(AudioDenseError, match="manifest"):
        load_audio_bundle(target)
    assert not is_audio_bundle_complete(target)


# ──────────────────────────────────────────────────────────────────────────────
# HF 编码器构造校验（不加载/不下载）
# ──────────────────────────────────────────────────────────────────────────────


def test_hf_encoder_constructor_validation(tmp_path: Path):
    with pytest.raises(AudioDenseError, match="model_id"):
        HFAudioEncoder("")
    with pytest.raises(AudioDenseError, match="device"):
        HFAudioEncoder("m-a-p/MERT-v1-95M", device="tpu")
    with pytest.raises(AudioDenseError, match="layer"):
        HFAudioEncoder("m-a-p/MERT-v1-95M", layer=0)
    with pytest.raises(AudioDenseError, match="revision"):
        HFAudioEncoder("m-a-p/MERT-v1-95M", revision=5)  # type: ignore[arg-type]
    encoder = HFAudioEncoder(
        "m-a-p/MERT-v1-95M", revision="abc123", cache_dir=tmp_path
    )
    assert encoder.version == "m-a-p/MERT-v1-95M@abc123"
    assert encoder.model_load_s == 0.0
    assert encoder.sample_rate == 0
    encoder.close()
    encoder.close()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("audio_dense_cli", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_help_offline():
    proc = subprocess.run(
        [sys.executable, str(CLI_PATH), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "MERT" in proc.stdout
    for flag in (
        "--pilot",
        "--limit",
        "--work-ids",
        "--force",
        "--version",
        "--device",
        "--batch-size",
    ):
        assert flag in proc.stdout, flag


def test_cli_parser_defaults():
    module = _load_cli_module()
    parser = module.build_parser()
    args = parser.parse_args(["--pilot", "--limit", "20", "--device", "cpu"])
    assert args.pilot is True
    assert args.limit == 20
    assert args.device == "cpu"
    assert args.context == 5.0
    assert args.hop == 5.0
    assert args.bin_seconds == 0.5
    assert args.normalize == "l2"
    assert args.pool == "mean_std"
    explicit = parser.parse_args(
        ["--work-ids", "1", "2", "--force", "--version", "audio_v1"]
    )
    assert explicit.work_ids == [1, 2]
    assert explicit.force is True
    assert explicit.version == "audio_v1"


def test_cli_bilibili_selection_and_source_resolution(
    config, tmp_path: Path, monkeypatch
):
    module = _load_cli_module()
    catalog = Catalog(config.paths.db_path)
    catalog.initialize()
    try:
        local_file = config.paths.input_local / "local.mp4"
        local_file.parent.mkdir(parents=True, exist_ok=True)
        local_file.write_bytes(b"fake")
        local_id, _created = catalog.upsert_local_work("local.mp4")
        catalog.update_work_metadata(local_id, duration_s=10.0)
        catalog.begin_attempt(local_id, "encoding")
        catalog.complete(local_id)
        bilibili_id, _created = catalog.upsert_bilibili_work(
            "BV1xx411c7m1", title="b站作品", duration_s=30.0
        )
        catalog.begin_attempt(bilibili_id, "encoding")
        catalog.complete(bilibili_id)
    finally:
        catalog.close()

    works = module._load_works(config, work_ids=None, limit=10)
    ids = {int(item["id"]) for item in works}
    assert local_id in ids and bilibili_id in ids
    local_only = module._load_works(
        config, work_ids=None, limit=10, local_only=True
    )
    assert {int(item["id"]) for item in local_only} == {local_id}

    cached = tmp_path / "cached_source.mp4"
    cached.write_bytes(b"fake")
    calls: dict[str, object] = {}

    def fake_resolve(
        config_, client, bvid, part, temp_dir, download, **kwargs
    ):
        calls.update(
            {
                "config": config_,
                "client": client,
                "bvid": bvid,
                "part": part,
                "temp_dir": temp_dir,
                "download": download,
            }
        )
        return cached

    monkeypatch.setattr("nebula.download.resolve_bilibili_source", fake_resolve)
    bilibili_work = next(
        item for item in works if int(item["id"]) == bilibili_id
    )
    path = module._resolve_work_source(
        config, bilibili_work, temp_root=config.paths.temp
    )
    assert path == cached
    assert calls["client"] is None
    assert calls["bvid"] == "BV1xx411c7m1"
    assert int(calls["part"]) == 1  # type: ignore[arg-type]
    assert Path(calls["temp_dir"]) == config.paths.temp / f"work_{bilibili_id:06d}"  # type: ignore[arg-type]

    local_work = next(item for item in works if int(item["id"]) == local_id)
    assert (
        module._resolve_work_source(
            config, local_work, temp_root=config.paths.temp
        )
        == local_file
    )

def test_cli_load_works_read_only(config):
    module = _load_cli_module()
    catalog = Catalog(config.paths.db_path)
    catalog.initialize()
    try:
        work_ids: list[int] = []
        for rel, duration in (("a.mp4", 10.0), ("b.mp4", 20.0), ("c.mp4", 30.0)):
            target = config.paths.input_local / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fake")
            work_id, _created = catalog.upsert_local_work(rel)
            catalog.update_work_metadata(work_id, duration_s=duration)
            catalog.begin_attempt(work_id, "encoding")
            catalog.complete(work_id)
            work_ids.append(work_id)
        duplicate, _ = catalog.upsert_local_work("dup.mp4")
        (config.paths.input_local / "dup.mp4").write_bytes(b"fake")
        catalog.update_work_metadata(duplicate, duration_s=15.0)
        catalog.begin_attempt(duplicate, "encoding")
        catalog.complete(duplicate)
        catalog.link_works(work_ids[0], duplicate, method="test", confidence=1.0)
    finally:
        catalog.close()
    selected = module._load_works(config, work_ids=None, limit=2)
    assert [item["id"] for item in selected] == [work_ids[0], work_ids[2]]
    ordered = module._load_works(config, work_ids=[work_ids[2], work_ids[1]], limit=2)
    assert [item["id"] for item in ordered] == [work_ids[2], work_ids[1]]
    with pytest.raises(AudioDenseError, match="work_id"):
        module._load_works(config, work_ids=[99999], limit=2)
