"""密集视觉在音频轨长于视频轨时的帧数校验回归测试（v0.4.3 全量批次发现的真实问题）。"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from nebula.dense_visual import (
    DenseVisualError,
    DenseVisualFeatures,
    extract_dense_features,
    iter_frame_batches,
)


class _FakeEncoder:
    version = "fake-duration-v1"
    dim = 4

    def encode(self, frames: np.ndarray) -> np.ndarray:
        matrix = np.asarray(frames, dtype=np.float32)
        return np.stack(
            [
                matrix[:, :, :, 0].mean(axis=(1, 2)),
                matrix[:, :, :, 1].mean(axis=(1, 2)),
                matrix[:, :, :, 2].mean(axis=(1, 2)),
                matrix.mean(axis=(1, 2, 3)),
            ],
            axis=1,
        )


def _make_video_with_long_audio(path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("缺少 ffmpeg")
    command = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10:duration=5",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=8",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "48000", "-ac", "2", str(path),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return path


def test_frames_expected_from_video_stream_not_container(tmp_path: Path):
    source = _make_video_with_long_audio(tmp_path / "av_mismatch.mp4")
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration:stream=codec_type,duration", "-of", "json", str(source),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(probe.stdout)
    container = float(payload["format"]["duration"])
    video = next(
        float(stream["duration"])
        for stream in payload["streams"]
        if stream["codec_type"] == "video"
    )
    assert container > video + 1.0  # 音频明显更长

    # 旧行为（按容器时长估算 64 帧）应失败；按视频流时长应通过
    with pytest.raises(DenseVisualError, match="解码帧数与时长不匹配"):
        list(iter_frame_batches(source, fps=8, size=16, batch_size=8))
    frames = sum(batch.shape[0] for batch in iter_frame_batches(
        source, fps=8, size=16, batch_size=8, expected_duration_s=video
    ))
    assert 32 <= frames <= 48

    features = extract_dense_features(source, _FakeEncoder(), fps=8, size=16, batch_size=8)
    assert isinstance(features, DenseVisualFeatures)
    # 视频 5s @8fps ≈ 40 帧；容器 8s 若按容器估算会是 64 帧
    assert 32 <= features.n_frames <= 48
    # duration_s 仍为容器时长，用于与 PE-AV 窗口对齐
    assert features.duration_s == pytest.approx(container, abs=0.5)
    assert features.per_frame.shape == (features.n_frames, 4)
    assert np.all(np.isfinite(features.per_frame))
