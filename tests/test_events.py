"""nebula.events（v0.4.1 高频音画事件扫描）的集成测试。

全部使用本机 ffmpeg 现场合成的受控视频（不联网、不加载模型），覆盖：
- 固定特征名/版本契约；
- 形状、采样率、确定性；
- 已知闪帧（视频 cut）与音脉冲（音频 onset）被检出；
- 窗口特征与窗口计划一一对应、窗口统计正确；
- 无音轨补静音、无视频流报错；
- 全片只解码两次（一次音频、一次视频）；
- 事件 bundle 原子保存与加载 roundtrip / 版本不兼容处理。

窗口计划对象只要求具备 CoverageClip 契约的 index/start_s/duration_s/valid_duration_s/weight
属性（本测试用本地 _Plan 模拟，t1 的 CoverageClip 可直接传入）。
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from nebula.errors import MediaError
from nebula.events import (
    AUDIO_FEATURE_NAMES,
    AV_FEATURE_NAMES,
    EVENT_BUNDLE_FILES,
    EVENT_VERSION,
    VIDEO_FEATURE_NAMES,
    WINDOW_FEATURE_NAMES,
    EventFeatures,
    EventScanError,
    load_events_bundle,
    save_events_bundle,
    scan_events,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="缺少 ffmpeg/ffprobe 可执行文件",
    ),
]

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

# 受控视频：12s / 30fps / 96x54；闪帧在 2.0s、5.0s、9.0s（帧号 60/150/270）；
# 音频脉冲（1kHz、100ms）在 3.0s、6.0s、10.0s —— 故意与闪帧错开，避免“假同步”。
FLASH_FRAMES = (60, 150, 270)
PULSE_TIMES = (3.0, 6.0, 10.0)


@dataclass(frozen=True)
class _Plan:
    """模拟 t1 的 CoverageClip 契约字段（duck typing，测试无需 import sampling）。"""

    index: int
    start_s: float
    duration_s: float
    valid_duration_s: float
    weight: float

    @property
    def end_s(self) -> float:
        return round(self.start_s + self.duration_s, 6)


def _plans(lengths: list[float]) -> tuple[_Plan, ...]:
    total = float(sum(lengths))
    plans: list[_Plan] = []
    start = 0.0
    for index, length in enumerate(lengths):
        plans.append(
            _Plan(
                index=index,
                start_s=round(start, 3),
                duration_s=round(length, 3),
                valid_duration_s=round(length, 3),
                weight=round(length / total, 9),
            )
        )
        start += length
    return tuple(plans)


def _build_controlled_video(
    target: Path,
    *,
    duration_s: float = 12.0,
    fps: int = 30,
    flash_frames: tuple[int, ...] = FLASH_FRAMES,
    pulses: tuple[tuple[float, float], ...] = tuple((time_s, 0.1) for time_s in PULSE_TIMES),
    with_audio: bool = True,
) -> Path:
    """用 Python 逐帧构造画面/音频，再用 ffmpeg 编码成 mp4（受控、确定性）。"""

    width, height = 96, 54
    frame_count = int(round(duration_s * fps))
    frames = np.zeros((frame_count, height, width), dtype=np.uint8)
    for frame_index in flash_frames:
        if 0 <= frame_index < frame_count:
            frames[frame_index, :, :] = 255
    raw_path = target.with_name(target.stem + ".raw")
    raw_path.write_bytes(frames.tobytes())

    command = [
        FFMPEG,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        str(raw_path),
    ]
    audio_path = target.with_name(target.stem + ".wav")
    if with_audio:
        samples = np.zeros(int(round(duration_s * 48000)), dtype=np.float32)
        for start_s, length_s in pulses:
            first = int(round(start_s * 48000))
            last = min(samples.size, first + int(round(length_s * 48000)))
            times = np.arange(last - first, dtype=np.float32) / 48000.0
            samples[first:last] += 0.6 * np.sin(2.0 * np.pi * 1000.0 * times)
        with wave.open(str(audio_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(48000)
            handle.writeframes(
                np.clip(samples * 32767.0, -32768, 32767).astype(np.int16).tobytes()
            )
        command += ["-i", str(audio_path), "-c:a", "aac", "-ar", "48000", "-ac", "2"]
    command += [
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target),
    ]
    subprocess.run(command, check=True, capture_output=True)
    raw_path.unlink(missing_ok=True)
    if audio_path.is_file():
        audio_path.unlink()
    assert target.is_file() and target.stat().st_size > 0
    return target


# ---------------------------------------------------------------- 夹具


@pytest.fixture(scope="module")
def controlled_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("events_media")
    return _build_controlled_video(directory / "controlled_12s.mp4")


@pytest.fixture(scope="module")
def silent_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("events_media")
    return _build_controlled_video(directory / "silent_12s.mp4", with_audio=False)


@pytest.fixture(scope="module")
def short_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("events_media")
    return _build_controlled_video(
        directory / "short_15s.mp4",
        duration_s=1.5,
        flash_frames=(20,),
        pulses=((0.5, 0.1),),
    )


@pytest.fixture(scope="module")
def audio_only(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("events_media")
    target = directory / "audio_only.m4a"
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-c:a",
            "aac",
            str(target),
        ],
        check=True,
        capture_output=True,
    )
    return target


@pytest.fixture(scope="module")
def features(controlled_video: Path) -> EventFeatures:
    return scan_events(controlled_video, _plans([8.0, 4.0]))


def _name_index(name: str) -> int:
    return WINDOW_FEATURE_NAMES.index(name)


# ---------------------------------------------------------------- 契约与形状


def test_feature_names_and_version_contract() -> None:
    assert EVENT_VERSION == "events_v1"
    assert len(AUDIO_FEATURE_NAMES) == 6
    assert len(VIDEO_FEATURE_NAMES) == 6
    assert len(AV_FEATURE_NAMES) == 6
    assert len(WINDOW_FEATURE_NAMES) == 32
    assert len(set(AUDIO_FEATURE_NAMES)) == 6
    assert len(set(VIDEO_FEATURE_NAMES)) == 6
    assert len(set(AV_FEATURE_NAMES)) == 6
    assert len(set(WINDOW_FEATURE_NAMES)) == 32
    # 固定分组顺序：音频 12 -> 视频 10 -> 音画 6 -> 窗口统计 4
    assert WINDOW_FEATURE_NAMES[:12] == tuple(
        name for name in WINDOW_FEATURE_NAMES if name.startswith("audio_")
    )[:12]
    assert WINDOW_FEATURE_NAMES[12:22] == tuple(
        name for name in WINDOW_FEATURE_NAMES if name.startswith("video_")
    )[:10]
    assert WINDOW_FEATURE_NAMES[22:28] == (
        "av_max_correlation",
        "av_best_lag_s",
        "av_corr_lag0",
        "av_coincidence_rate",
        "av_envelope_correlation",
        "av_event_rate_diff",
    )
    assert WINDOW_FEATURE_NAMES[28:] == (
        "window_duration_s",
        "window_valid_ratio",
        "window_start_ratio",
        "window_weight",
    )


def test_scan_shapes_rates_and_version(features: EventFeatures) -> None:
    assert features.version == EVENT_VERSION
    assert features.audio_rate_hz == pytest.approx(100.0)
    assert features.video_fps == pytest.approx(30.0)
    assert 28.0 <= features.video_fps_effective <= 32.0
    assert features.duration_s == pytest.approx(12.0, abs=0.2)

    assert features.audio.ndim == 2 and features.audio.shape[1] == 6
    assert features.video.ndim == 2 and features.video.shape[1] == 6
    assert features.av.shape == features.audio.shape
    assert features.window_features.shape == (2, 32)
    assert features.audio.shape[0] == round(features.duration_s * features.audio_rate_hz)
    assert abs(features.video.shape[0] - round(features.duration_s * 30)) <= 3
    for array in (features.audio, features.video, features.av, features.window_features):
        assert array.dtype == np.float32
        assert np.isfinite(array).all()


# ---------------------------------------------------------------- 已知事件检出


def test_video_flash_frames_detected(features: EventFeatures) -> None:
    diff = features.video[:, VIDEO_FEATURE_NAMES.index("frame_diff_mean")]
    strong = np.flatnonzero(diff > 0.5)
    # 每个闪帧出现/消失各产生一次强帧差；受编码影响允许 ±1 帧偏差
    expected = {
        index
        for frame in FLASH_FRAMES
        for index in (frame, frame + 1)
    }
    assert strong.size >= 4
    for index in strong:
        assert min(abs(int(index) - candidate) for candidate in expected) <= 1, index
    for frame in FLASH_FRAMES:
        window = diff[max(0, frame - 1) : frame + 2]
        assert float(window.max()) > 0.5, frame

    cut = features.video[:, VIDEO_FEATURE_NAMES.index("cut")]
    cut_frames = np.flatnonzero(cut > 0.5)
    assert cut_frames.size >= 4
    for frame in FLASH_FRAMES:
        assert any(abs(int(index) - frame) <= 1 for index in cut_frames), frame


def test_audio_pulses_detected(features: EventFeatures) -> None:
    onset = features.audio[:, AUDIO_FEATURE_NAMES.index("onset_strength")]
    times = (np.arange(onset.size) + 0.5) / features.audio_rate_hz
    assert float(onset.max()) > 0.5
    strong_times = times[onset > 0.5]
    # 只应检出 3 次脉冲起点（脉冲结束时是负向 flux，不产生 onset）
    assert strong_times.size == len(PULSE_TIMES)
    for expected, detected in zip(sorted(PULSE_TIMES), strong_times, strict=True):
        assert abs(float(detected) - expected) <= 0.05


# ---------------------------------------------------------------- 窗口对齐


def test_window_features_follow_plan_order(controlled_video: Path) -> None:
    plans = _plans([2.5, 9.5])
    features = scan_events(controlled_video, plans)
    assert features.window_features.shape == (len(plans), 32)

    duration = features.window_features[:, _name_index("window_duration_s")]
    starts = features.window_features[:, _name_index("window_start_ratio")]
    valid = features.window_features[:, _name_index("window_valid_ratio")]
    weights = features.window_features[:, _name_index("window_weight")]
    assert duration.tolist() == pytest.approx([2.5, 9.5], abs=1e-5)
    assert starts.tolist() == pytest.approx([0.0, 2.5 / 12.0], abs=1e-5)
    assert valid.tolist() == pytest.approx([1.0, 1.0], abs=1e-6)
    assert weights.tolist() == pytest.approx(
        [2.5 / 12.0, 9.5 / 12.0], abs=1e-6
    )

    # 3.0/6.0/10.0 的脉冲都在第二个窗口；第一个窗口不应有 onset 事件
    rates = features.window_features[:, _name_index("audio_onset_rate")]
    assert float(rates[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(rates[1]) > 0.1
    # 两个窗口都包含闪帧（2.0s 与 5.0s/9.0s），cut 率应为正
    cut_rates = features.window_features[:, _name_index("video_cut_rate")]
    assert float(cut_rates[0]) > 0.0
    assert float(cut_rates[1]) > 0.0


# ---------------------------------------------------------------- 确定性


def test_scan_is_deterministic(controlled_video: Path, features: EventFeatures) -> None:
    again = scan_events(controlled_video, _plans([8.0, 4.0]))
    np.testing.assert_array_equal(again.audio, features.audio)
    np.testing.assert_array_equal(again.video, features.video)
    np.testing.assert_array_equal(again.av, features.av)
    np.testing.assert_array_equal(again.window_features, features.window_features)
    assert again.audio_rate_hz == features.audio_rate_hz
    assert again.video_fps_effective == features.video_fps_effective
    assert again.version == features.version


def test_scan_decodes_only_twice(
    controlled_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """全片只允许一条音频命令 + 一条视频命令（禁止逐片段重复解码）。"""

    commands: list[list[str]] = []
    real_popen = subprocess.Popen

    def recording_popen(args, *positional, **keywords):
        commands.append([str(item) for item in args])
        return real_popen(args, *positional, **keywords)

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    scan_events(controlled_video, _plans([8.0, 4.0]))

    decode_commands = [command for command in commands if "pipe:1" in command]
    assert len(decode_commands) == 2
    audio_commands = [command for command in decode_commands if "-vn" in command]
    video_commands = [command for command in decode_commands if "-an" in command]
    assert len(audio_commands) == 1
    assert len(video_commands) == 1
    # 不允许出现按片段 seek/裁剪的额外解码
    for command in decode_commands:
        assert "-ss" not in command


# ---------------------------------------------------------------- 无音轨 / 无视频流


def test_no_audio_source_uses_silence(silent_video: Path) -> None:
    features = scan_events(silent_video, _plans([6.0, 6.0]))
    assert features.version == EVENT_VERSION
    assert features.audio.shape[0] == round(features.duration_s * features.audio_rate_hz)
    np.testing.assert_allclose(
        features.audio[:, AUDIO_FEATURE_NAMES.index("rms")], 0.0, atol=1e-9
    )
    np.testing.assert_allclose(
        features.audio[:, AUDIO_FEATURE_NAMES.index("onset_strength")], 0.0, atol=1e-9
    )
    np.testing.assert_allclose(
        features.audio[:, AUDIO_FEATURE_NAMES.index("log_energy")],
        np.log10(1e-10),
        atol=1e-9,
    )
    assert np.isfinite(features.window_features).all()
    assert np.isfinite(features.av).all()


def test_no_video_stream_raises(audio_only: Path) -> None:
    with pytest.raises(EventScanError, match="视频流") as info:
        scan_events(audio_only, _plans([2.0]))
    assert isinstance(info.value, MediaError)
    assert isinstance(info.value, EventScanError)


# ---------------------------------------------------------------- 短窗口 / 参数校验


def test_short_video_single_window(short_video: Path) -> None:
    features = scan_events(short_video, _plans([1.5]))
    assert features.window_features.shape == (1, 32)
    assert features.duration_s == pytest.approx(1.5, abs=0.2)
    assert features.audio.shape[0] == round(features.duration_s * features.audio_rate_hz)
    assert abs(features.video.shape[0] - round(features.duration_s * 30)) <= 3
    window_duration = features.window_features[0, _name_index("window_duration_s")]
    assert float(window_duration) == pytest.approx(1.5, abs=0.2)
    assert np.isfinite(features.window_features).all()


def test_empty_plans_raise(controlled_video: Path) -> None:
    with pytest.raises(EventScanError, match="至少需要一个窗口"):
        scan_events(controlled_video, [])


def test_plan_beyond_duration_raises(controlled_video: Path) -> None:
    bad = _Plan(index=0, start_s=11.5, duration_s=8.0, valid_duration_s=8.0, weight=1.0)
    with pytest.raises(EventScanError, match="超出视频时长"):
        scan_events(controlled_video, [bad])


def test_plan_missing_attributes_raises(controlled_video: Path) -> None:
    with pytest.raises(EventScanError, match="缺少 start_s/duration_s"):
        scan_events(controlled_video, [object()])  # type: ignore[list-item]


def test_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(EventScanError, match="不存在"):
        scan_events(tmp_path / "missing.mp4", _plans([1.0]))


def test_invalid_rates_raise(controlled_video: Path) -> None:
    with pytest.raises(EventScanError):
        scan_events(controlled_video, _plans([1.0]), audio_rate_hz=0)
    with pytest.raises(EventScanError):
        scan_events(controlled_video, _plans([1.0]), video_fps=0)
    with pytest.raises(EventScanError):
        scan_events(controlled_video, _plans([1.0]), timeout_s=0)


# ---------------------------------------------------------------- bundle 保存/加载


def test_save_and_load_roundtrip(tmp_path: Path, features: EventFeatures) -> None:
    work_dir = tmp_path / "work_000001"
    save_events_bundle(work_dir, features)

    for name in EVENT_BUNDLE_FILES:
        assert (work_dir / name).is_file(), name
    manifest = json.loads((work_dir / "event_manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == EVENT_VERSION
    assert manifest["n_windows"] == features.window_features.shape[0]
    assert manifest["feature_names"]["window"] == list(WINDOW_FEATURE_NAMES)
    assert manifest["audio_rate_hz"] == pytest.approx(features.audio_rate_hz)
    assert manifest["video_fps"] == pytest.approx(features.video_fps)
    assert manifest["shapes"]["window_features"] == [2, 32]
    assert (work_dir / "event_features.npy").stat().st_size > 0

    loaded = load_events_bundle(work_dir)
    assert loaded is not None
    assert loaded.version == features.version
    assert loaded.audio_rate_hz == pytest.approx(features.audio_rate_hz)
    assert loaded.video_fps_effective == pytest.approx(features.video_fps_effective)
    assert loaded.duration_s == pytest.approx(features.duration_s)
    np.testing.assert_array_equal(loaded.audio, features.audio)
    np.testing.assert_array_equal(loaded.video, features.video)
    np.testing.assert_array_equal(loaded.av, features.av)
    np.testing.assert_array_equal(loaded.window_features, features.window_features)


def test_load_missing_bundle_returns_none(tmp_path: Path) -> None:
    assert load_events_bundle(tmp_path / "no_such_dir") is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert load_events_bundle(empty) is None


def test_load_version_mismatch_returns_none(
    tmp_path: Path, features: EventFeatures
) -> None:
    work_dir = tmp_path / "work_mismatch"
    save_events_bundle(work_dir, features)
    manifest_path = work_dir / "event_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = "events_v0"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    assert load_events_bundle(work_dir) is None


def test_save_rejects_non_finite(tmp_path: Path, features: EventFeatures) -> None:
    broken = dataclasses.replace(
        features, audio=np.full_like(features.audio, np.nan)
    )
    with pytest.raises(EventScanError, match="非有限"):
        save_events_bundle(tmp_path / "broken", broken)


def test_load_missing_one_file_returns_none(
    tmp_path: Path, features: EventFeatures
) -> None:
    work_dir = tmp_path / "work_partial"
    save_events_bundle(work_dir, features)
    (work_dir / "event_features.npy").unlink()
    assert load_events_bundle(work_dir) is None
