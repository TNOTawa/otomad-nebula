"""nebula.media 的 ffmpeg / ffprobe 集成测试。

这些测试需要本机安装 ffmpeg / ffprobe（见 config.yaml 的 media 段），
标记为 integration；缺少 ffmpeg 时会自动 skip 而不是失败。
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nebula import media as media_module
from nebula.errors import MediaError
from nebula.media import (
    ExtractedClip,
    MediaInfo,
    black_video_clip,
    clip_duration_report,
    extract_clip,
    mute_audio_clip,
    probe_media,
)
from nebula.sampling import ClipPlan, plan_clips

pytestmark = pytest.mark.integration


def _ffmpeg() -> str:
    """返回本机 ffmpeg 路径；缺失时跳过测试。"""

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("缺少 ffmpeg")
    return ffmpeg


def _read_wav_samples(path: Path) -> tuple[np.ndarray, int, int]:
    """读取 16-bit PCM wav，返回（单声道 float32 采样, 采样率, 声道数）。"""

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    assert sample_width == 2, f"预期 16-bit PCM，实际 sample width={sample_width}"
    raw = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        raw = raw.reshape(-1, channels).mean(axis=1)
    return raw, sample_rate, channels


@pytest.fixture
def video_only(tmp_path: Path) -> Path:
    """生成一段 4 秒、无音轨的测试视频（testsrc2，160x120@10fps）。"""

    ffmpeg = _ffmpeg()
    target = tmp_path / "video_only.mp4"
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
            "testsrc2=size=160x120:rate=10:duration=4",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(target),
        ],
        check=True,
        capture_output=True,
    )
    return target


@pytest.fixture
def audio_only(tmp_path: Path) -> Path:
    """生成一段 3 秒、无视频轨的测试音频（AAC / m4a）。"""

    ffmpeg = _ffmpeg()
    target = tmp_path / "audio_only.m4a"
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
            "sine=frequency=660:sample_rate=48000:duration=3",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(target),
        ],
        check=True,
        capture_output=True,
    )
    return target


@pytest.fixture
def tone_switch_video(tmp_path: Path) -> Path:
    """4 秒测试视频：音频前 2 秒 440Hz、后 2 秒 880Hz，用于验证 -ss 偏移。"""

    ffmpeg = _ffmpeg()
    target = tmp_path / "tone_switch.mp4"
    tone_graph = (
        "sine=frequency=440:sample_rate=48000:duration=2[a];"
        "sine=frequency=880:sample_rate=48000:duration=2[b];"
        "[a][b]concat=n=2:v=0:a=1"
    )
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
            "testsrc2=size=160x120:rate=10:duration=4",
            "-f",
            "lavfi",
            "-i",
            tone_graph,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-shortest",
            str(target),
        ],
        check=True,
        capture_output=True,
    )
    return target


def _dominant_frequency(samples: np.ndarray, sample_rate: int) -> float:
    """用 FFT 估计主频（仅用于测试音调切换素材）。"""

    window = samples * np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(window))
    frequencies = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    return float(frequencies[int(np.argmax(spectrum))])


# ---------------------------------------------------------------------------
# probe_media
# ---------------------------------------------------------------------------


def test_probe_sample_video(sample_video: Path, config):
    info = probe_media(sample_video, ffprobe=config.media.ffprobe)

    assert isinstance(info, MediaInfo)
    assert info.path == sample_video.resolve()
    assert info.duration_s == pytest.approx(12.0, abs=0.2)
    assert info.has_video is True
    assert info.has_audio is True
    assert info.width == 160
    assert info.height == 120
    assert info.fps == pytest.approx(10.0, abs=0.01)
    assert info.video_codec == "h264"
    assert info.audio_codec == "aac"


def test_probe_video_only(video_only: Path, config):
    info = probe_media(video_only, ffprobe=config.media.ffprobe)

    assert info.has_video is True
    assert info.has_audio is False
    assert info.audio_codec is None
    assert info.duration_s == pytest.approx(4.0, abs=0.2)
    assert (info.width, info.height) == (160, 120)


def test_probe_audio_only(audio_only: Path, config):
    info = probe_media(audio_only, ffprobe=config.media.ffprobe)

    assert info.has_audio is True
    assert info.has_video is False
    assert info.audio_codec == "aac"
    assert info.width is None
    assert info.height is None
    assert info.fps is None
    assert info.video_codec is None
    assert info.duration_s == pytest.approx(3.0, abs=0.2)


def test_probe_missing_file_raises(config, tmp_path: Path):
    with pytest.raises(MediaError, match="不存在"):
        probe_media(tmp_path / "missing.mp4", ffprobe=config.media.ffprobe)


def test_probe_corrupt_file_raises(config, tmp_path: Path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"this is definitely not a video file" * 50)
    with pytest.raises(MediaError):
        probe_media(broken, ffprobe=config.media.ffprobe)


def test_probe_missing_ffprobe_raises(sample_video: Path):
    with pytest.raises(MediaError, match="找不到 ffprobe"):
        probe_media(sample_video, ffprobe="definitely-missing-ffprobe-binary")


def test_probe_invalid_timeout_raises(sample_video: Path, config):
    with pytest.raises(MediaError, match="超时"):
        probe_media(sample_video, ffprobe=config.media.ffprobe, timeout_s=0)


# ---------------------------------------------------------------------------
# extract_clip
# ---------------------------------------------------------------------------


def test_extract_two_clips_with_matching_durations(sample_video: Path, config, tmp_path: Path):
    info = probe_media(sample_video, ffprobe=config.media.ffprobe)
    plans = plan_clips(info.duration_s, clip_seconds=4.0, clips_per_work=2)
    assert len(plans) == 2

    clips = [
        extract_clip(sample_video, plan, tmp_path / "clips", media=config.media) for plan in plans
    ]
    assert len(clips) == 2
    for plan, clip in zip(plans, clips, strict=True):
        assert isinstance(clip, ExtractedClip)
        assert clip.index == plan.index
        assert clip.start_s == pytest.approx(plan.start_s)
        assert clip.duration_s == pytest.approx(plan.duration_s)
        assert clip.video_path.name == f"clip_{plan.index:02d}.mp4"
        assert clip.audio_path.name == f"clip_{plan.index:02d}.wav"
        assert clip.video_path.is_file() and clip.video_path.stat().st_size > 0
        assert clip.audio_path.is_file() and clip.audio_path.stat().st_size > 0

        video_info = probe_media(clip.video_path, ffprobe=config.media.ffprobe)
        assert video_info.has_video is True
        assert video_info.video_codec == "h264"
        assert (video_info.width, video_info.height) == (160, 120)
        # A5：临时 mp4 只保留视频轨，音频单独走 wav，避免多一次 AAC 编码
        assert video_info.has_audio is False, "临时 mp4 不应带 AAC 音轨"

        _, sample_rate, channels = _read_wav_samples(clip.audio_path)
        assert sample_rate == 48000
        assert channels == 2

        report = clip_duration_report(
            clip.video_path, clip.audio_path, ffprobe=config.media.ffprobe
        )
        assert report["ok"] is True
        assert report["video_duration_s"] == pytest.approx(plan.duration_s, abs=0.15)
        assert report["audio_duration_s"] == pytest.approx(plan.duration_s, abs=0.15)
        assert report["delta_s"] <= 0.15


def test_extract_audio_uses_plan_offset(tone_switch_video: Path, config, tmp_path: Path):
    """-ss 必须同时作用于音频：第 0 段是 440Hz，第 2 秒起的片段应为 880Hz。"""

    first = extract_clip(
        tone_switch_video,
        ClipPlan(index=0, start_s=0.0, duration_s=2.0),
        tmp_path / "first",
        media=config.media,
    )
    second = extract_clip(
        tone_switch_video,
        ClipPlan(index=1, start_s=2.0, duration_s=2.0),
        tmp_path / "second",
        media=config.media,
    )

    for clip, expected_hz in ((first, 440.0), (second, 880.0)):
        samples, sample_rate, _ = _read_wav_samples(clip.audio_path)
        middle = samples[int(0.2 * sample_rate) : int(1.8 * sample_rate)]
        dominant = _dominant_frequency(middle, sample_rate)
        assert dominant == pytest.approx(expected_hz, abs=20.0), (
            f"片段 {clip.index} 主频 {dominant:.1f}Hz，预期 {expected_hz}Hz"
        )


def test_extract_video_without_audio_pads_silence(video_only: Path, config, tmp_path: Path):
    plans = plan_clips(4.0, clip_seconds=config.media.clip_seconds)
    assert len(plans) == 1
    clip = extract_clip(video_only, plans[0], tmp_path, media=config.media)

    assert clip.audio_path.is_file() and clip.audio_path.stat().st_size > 0
    samples, sample_rate, channels = _read_wav_samples(clip.audio_path)
    assert sample_rate == 48000
    assert channels == 2
    assert samples.size > 0
    assert float(np.max(np.abs(samples))) == 0.0

    report = clip_duration_report(clip.video_path, clip.audio_path, ffprobe=config.media.ffprobe)
    assert report["ok"] is True
    assert report["video_duration_s"] == pytest.approx(4.0, abs=0.15)
    assert report["audio_duration_s"] == pytest.approx(4.0, abs=0.15)
    assert probe_media(clip.video_path, ffprobe=config.media.ffprobe).has_audio is False


def test_extract_with_info_skips_ffprobe(sample_video: Path, config, tmp_path: Path, monkeypatch):
    """A4：传入已探测的 MediaInfo 时必须完全跳过 ffprobe。"""

    info = probe_media(sample_video, ffprobe=config.media.ffprobe)
    plan = plan_clips(info.duration_s, clip_seconds=4.0, clips_per_work=2)[0]

    def _fail(*args, **kwargs):
        raise AssertionError("传入 info 时 extract_clip 不应再调用 probe_media/ffprobe")

    monkeypatch.setattr(media_module, "probe_media", _fail)
    clip = extract_clip(sample_video, plan, tmp_path / "clips", media=config.media, info=info)

    assert clip.index == plan.index
    assert clip.video_path.is_file() and clip.audio_path.is_file()
    assert clip.duration_s == plan.duration_s


def test_extract_with_info_uses_provided_duration(
    sample_video: Path, config, tmp_path: Path, monkeypatch
):
    """A4：越界判断使用传入的 MediaInfo，不重新 probe。"""

    info = probe_media(sample_video, ffprobe=config.media.ffprobe)
    short_info = replace(info, duration_s=3.0)

    def _fail(*args, **kwargs):
        raise AssertionError("传入 info 时不应再调用 probe_media/ffprobe")

    monkeypatch.setattr(media_module, "probe_media", _fail)
    with pytest.raises(MediaError, match="超出"):
        extract_clip(
            sample_video,
            ClipPlan(index=0, start_s=2.0, duration_s=2.0),
            tmp_path,
            media=config.media,
            info=short_info,
        )


def test_extract_rejects_invalid_info_type(sample_video: Path, config, tmp_path: Path):
    with pytest.raises(MediaError, match="info 参数类型非法"):
        extract_clip(
            sample_video,
            ClipPlan(index=0, start_s=0.0, duration_s=1.0),
            tmp_path,
            media=config.media,
            info="not-media-info",  # type: ignore[arg-type]
        )


def test_extract_audio_only_source_raises(audio_only: Path, config, tmp_path: Path):
    with pytest.raises(MediaError, match="没有视频轨"):
        extract_clip(
            audio_only,
            ClipPlan(index=0, start_s=0.0, duration_s=1.0),
            tmp_path,
            media=config.media,
        )


def test_extract_out_of_range_plan_raises(sample_video: Path, config, tmp_path: Path):
    with pytest.raises(MediaError, match="超出"):
        extract_clip(
            sample_video,
            ClipPlan(index=0, start_s=10.0, duration_s=8.0),
            tmp_path,
            media=config.media,
        )


def test_extract_invalid_duration_raises(sample_video: Path, config, tmp_path: Path):
    with pytest.raises(MediaError, match="必须为正数"):
        extract_clip(
            sample_video,
            ClipPlan(index=0, start_s=0.0, duration_s=0.0),
            tmp_path,
            media=config.media,
        )


def test_extract_missing_ffmpeg_raises(sample_video: Path, config, tmp_path: Path):
    broken_media = replace(config.media, ffmpeg="definitely-missing-ffmpeg-binary")
    with pytest.raises(MediaError, match="找不到 ffmpeg"):
        extract_clip(
            sample_video,
            ClipPlan(index=0, start_s=0.0, duration_s=2.0),
            tmp_path,
            media=broken_media,
        )


# ---------------------------------------------------------------------------
# 消融素材与时长报告
# ---------------------------------------------------------------------------


def test_mute_audio_clip_is_silent_and_same_length(config, tmp_path: Path):
    target = mute_audio_clip(2.5, tmp_path / "mute.wav", media=config.media)
    assert target.is_file() and target.stat().st_size > 0

    info = probe_media(target, ffprobe=config.media.ffprobe)
    assert info.has_audio is True
    assert info.has_video is False
    assert info.duration_s == pytest.approx(2.5, abs=0.05)

    samples, sample_rate, channels = _read_wav_samples(target)
    assert sample_rate == 48000
    assert channels == 2
    assert float(np.max(np.abs(samples))) == 0.0


def test_black_video_clip_is_black_and_same_length(config, tmp_path: Path):
    target = black_video_clip(2.0, tmp_path / "black.mp4", media=config.media)
    assert target.is_file() and target.stat().st_size > 0

    info = probe_media(target, ffprobe=config.media.ffprobe)
    assert info.has_video is True
    assert info.has_audio is False
    assert (info.width, info.height) == (336, 336)
    assert info.fps == pytest.approx(25.0, abs=0.01)
    assert info.duration_s == pytest.approx(2.0, abs=0.1)

    ffmpeg = _ffmpeg()
    proc = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(target),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    assert proc.stdout, "应当能解码出至少一帧画面"
    assert max(proc.stdout) == 0, "纯黑视频解码后的像素应全为 0"


def test_black_video_clip_custom_size(config, tmp_path: Path):
    target = black_video_clip(
        1.5, tmp_path / "small_black.mp4", media=config.media, width=64, height=48, fps=10
    )
    info = probe_media(target, ffprobe=config.media.ffprobe)
    assert (info.width, info.height) == (64, 48)
    assert info.fps == pytest.approx(10.0, abs=0.01)


def test_clip_duration_report_detects_mismatch(config, tmp_path: Path):
    video = black_video_clip(2.0, tmp_path / "video.mp4", media=config.media)
    audio = mute_audio_clip(3.0, tmp_path / "audio.wav", media=config.media)

    report = clip_duration_report(video, audio, ffprobe=config.media.ffprobe, tolerance_s=0.15)
    assert report["ok"] is False
    assert report["video_duration_s"] == pytest.approx(2.0, abs=0.1)
    assert report["audio_duration_s"] == pytest.approx(3.0, abs=0.05)
    assert report["delta_s"] == pytest.approx(1.0, abs=0.15)


def test_clip_duration_report_invalid_tolerance_raises(config, tmp_path: Path):
    video = black_video_clip(1.0, tmp_path / "video.mp4", media=config.media)
    audio = mute_audio_clip(1.0, tmp_path / "audio.wav", media=config.media)
    with pytest.raises(MediaError, match="容差"):
        clip_duration_report(video, audio, ffprobe=config.media.ffprobe, tolerance_s=-1.0)
