"""A6 音画错位实验（nebula.avsync + tools/desync_experiment.py）的离线测试。

覆盖：

- 音频 wav 读写 / 循环移位 / 交换（np.roll 语义、格式校验、同路径保护）；
- Δ = 1 - cos 的数学性质与非法输入报错；
- 变体构建：reference / quality / shift_0.5s / shift_1.0s / swap 的文件与格式，
  单片段作品跳过 swap；
- 用 HashTestEmbedder 注入跑通 evaluate_variants + build_report + write_report，
  校验 JSON / CSV schema 与中文摘要（不加载模型、不写 data/、不联网）。
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nebula.avsync import (
    CSV_COLUMNS,
    DEFAULT_SHIFTS_S,
    MODALITIES,
    PERIODICITY_CAVEAT,
    VARIANT_QUALITY,
    VARIANT_REFERENCE,
    VARIANT_SWAP,
    ClipVariants,
    VariantBuild,
    VariantDeltas,
    build_report,
    build_work_variants,
    compute_delta,
    evaluate_variants,
    format_summary,
    make_shifted_wav,
    make_swap_wav,
    read_wav_pcm,
    roll_audio_samples,
    shift_variant_name,
    summarize_deltas,
    write_report,
    write_wav_pcm,
)
from nebula.embed import EmbeddingBatch, HashTestEmbedder
from nebula.errors import NebulaError
from nebula.media import probe_media
from nebula.sampling import plan_clips

REPO_ROOT = Path(__file__).resolve().parents[1]


# ──────────────────────────────────────────────────────────────────────────────
# 合成 wav
# ──────────────────────────────────────────────────────────────────────────────


def _tone_samples(
    *, freq: float = 440.0, seconds: float = 1.0, sample_rate: int = 48000, channels: int = 2
) -> np.ndarray:
    """生成 16-bit 正弦 PCM 矩阵 (n_samples, channels)。"""

    frames = int(round(sample_rate * seconds))
    timeline = np.arange(frames, dtype=np.float64) / float(sample_rate)
    mono = (0.5 * np.sin(2.0 * np.pi * freq * timeline) * 32767.0).astype(np.int16)
    return np.repeat(mono[:, None], channels, axis=1)


def _write_tone_wav(
    path: Path,
    *,
    freq: float = 440.0,
    seconds: float = 1.0,
    sample_rate: int = 48000,
    channels: int = 2,
) -> Path:
    return write_wav_pcm(
        path,
        _tone_samples(
            freq=freq, seconds=seconds, sample_rate=sample_rate, channels=channels
        ),
        sample_rate,
    )


# ──────────────────────────────────────────────────────────────────────────────
# wav 读写
# ──────────────────────────────────────────────────────────────────────────────


def test_write_read_wav_roundtrip(tmp_path: Path):
    samples = _tone_samples(seconds=0.2)
    target = write_wav_pcm(tmp_path / "a.wav", samples, 48000)

    loaded, rate, channels = read_wav_pcm(target)
    assert rate == 48000
    assert channels == 2
    assert np.array_equal(loaded, samples)
    assert loaded.dtype == np.int16


def test_write_wav_rejects_non_int16(tmp_path: Path):
    with pytest.raises(NebulaError, match="int16"):
        write_wav_pcm(tmp_path / "a.wav", np.zeros((10, 2), dtype=np.float32))


def test_write_wav_rejects_bad_shape(tmp_path: Path):
    with pytest.raises(NebulaError, match="形状"):
        write_wav_pcm(tmp_path / "a.wav", np.zeros(10, dtype=np.int16))


def test_write_wav_rejects_bad_sample_rate(tmp_path: Path):
    with pytest.raises(NebulaError, match="采样率"):
        write_wav_pcm(tmp_path / "a.wav", np.zeros((10, 2), dtype=np.int16), 0)


def test_read_wav_rejects_8bit(tmp_path: Path):
    import wave

    target = tmp_path / "eight_bit.wav"
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)
        handle.setframerate(8000)
        handle.writeframes(b"\x80" * 100)
    with pytest.raises(NebulaError, match="16-bit"):
        read_wav_pcm(target)


def test_read_wav_missing_file_raises(tmp_path: Path):
    with pytest.raises(NebulaError, match="不存在"):
        read_wav_pcm(tmp_path / "missing.wav")


# ──────────────────────────────────────────────────────────────────────────────
# 循环移位
# ──────────────────────────────────────────────────────────────────────────────


def test_shift_variant_name():
    assert shift_variant_name(0.5) == "shift_0.5s"
    assert shift_variant_name(1.0) == "shift_1.0s"


@pytest.mark.parametrize("bad", [0, 0.0, float("nan"), float("inf"), True])
def test_shift_variant_name_rejects_invalid(bad: object):
    with pytest.raises(NebulaError, match="移位量"):
        shift_variant_name(bad)  # type: ignore[arg-type]


def test_roll_audio_zero_shift_returns_equal_copy():
    samples = np.arange(100, dtype=np.int16).reshape(50, 2)
    rolled = roll_audio_samples(samples, 0.0, sample_rate=10)
    assert np.array_equal(rolled, samples)
    assert rolled is not samples


def test_roll_audio_half_second_uses_np_roll():
    samples = np.arange(40, dtype=np.int16).reshape(20, 2)
    rolled = roll_audio_samples(samples, 0.5, sample_rate=10)
    # 0.5s * 10Hz = 5 帧
    assert np.array_equal(rolled, np.roll(samples, shift=5, axis=0))
    assert rolled.shape == samples.shape


def test_roll_audio_wraps_large_shift():
    samples = np.arange(40, dtype=np.int16).reshape(20, 2)
    rolled = roll_audio_samples(samples, 3.0, sample_rate=10)
    # 3s * 10Hz = 30 帧，对 20 帧循环等价于 10 帧
    assert np.array_equal(rolled, np.roll(samples, shift=10, axis=0))


@pytest.mark.parametrize("shift", [float("nan"), float("inf")])
def test_roll_audio_invalid_shift_raises(shift: float):
    samples = np.arange(40, dtype=np.int16).reshape(20, 2)
    with pytest.raises(NebulaError, match="移位量"):
        roll_audio_samples(samples, shift, sample_rate=10)


def test_make_shifted_wav_preserves_format_and_length(tmp_path: Path):
    # 437Hz 的 0.5s 是 218.5 个周期，移位后波形必然变化（440Hz 会是整数周期）
    source = _write_tone_wav(tmp_path / "ref.wav", seconds=1.0, freq=437.0)
    target = make_shifted_wav(source, tmp_path / "shift.wav", 0.5)

    shifted, rate, channels = read_wav_pcm(target)
    reference, ref_rate, ref_channels = read_wav_pcm(source)
    assert rate == ref_rate == 48000
    assert channels == ref_channels == 2
    assert shifted.shape == reference.shape
    assert np.array_equal(shifted, np.roll(reference, shift=24000, axis=0))
    assert not np.array_equal(shifted, reference)


def test_make_shifted_wav_rejects_mono(tmp_path: Path):
    source = _write_tone_wav(tmp_path / "mono.wav", channels=1)
    with pytest.raises(NebulaError, match="48000Hz / 2ch"):
        make_shifted_wav(source, tmp_path / "out.wav", 0.5)


def test_make_shifted_wav_rejects_same_path(tmp_path: Path):
    source = _write_tone_wav(tmp_path / "ref.wav")
    with pytest.raises(NebulaError, match="不能与输入相同"):
        make_shifted_wav(source, source, 0.5)


# ──────────────────────────────────────────────────────────────────────────────
# 音频交换
# ──────────────────────────────────────────────────────────────────────────────


def test_make_swap_wav_copies_other_audio(tmp_path: Path):
    other = _write_tone_wav(tmp_path / "other.wav", freq=880.0)
    target = make_swap_wav(other, tmp_path / "swap.wav")

    swapped, rate, channels = read_wav_pcm(target)
    expected, _, _ = read_wav_pcm(other)
    assert rate == 48000
    assert channels == 2
    assert np.array_equal(swapped, expected)


def test_make_swap_wav_rejects_wrong_format(tmp_path: Path):
    source = _write_tone_wav(tmp_path / "mono.wav", channels=1, sample_rate=44100)
    with pytest.raises(NebulaError, match="48000Hz / 2ch"):
        make_swap_wav(source, tmp_path / "swap.wav")


def test_make_swap_wav_rejects_same_path(tmp_path: Path):
    source = _write_tone_wav(tmp_path / "ref.wav")
    with pytest.raises(NebulaError, match="不能与输入相同"):
        make_swap_wav(source, source)


# ──────────────────────────────────────────────────────────────────────────────
# Δ = 1 - cos
# ──────────────────────────────────────────────────────────────────────────────


def _basis(dim: int, index: int) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    vector[index] = 1.0
    return vector


def test_compute_delta_identical_is_zero():
    vector = np.array([0.3, -0.4, 0.5, 0.7], dtype=np.float32)
    assert compute_delta(vector, vector.copy()) == pytest.approx(0.0, abs=1e-6)


def test_compute_delta_orthogonal_is_one():
    assert compute_delta(_basis(4, 0), _basis(4, 1)) == pytest.approx(1.0, abs=1e-6)


def test_compute_delta_opposite_is_two():
    assert compute_delta(_basis(4, 0), -_basis(4, 0)) == pytest.approx(2.0, abs=1e-6)


def test_compute_delta_ignores_scale():
    vector = np.array([0.2, 0.4, -0.6], dtype=np.float32)
    assert compute_delta(vector, vector * 5.0) == pytest.approx(0.0, abs=1e-6)


def test_compute_delta_symmetry():
    left = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    right = np.array([-0.3, 0.5, 0.8], dtype=np.float32)
    assert compute_delta(left, right) == pytest.approx(compute_delta(right, left), abs=1e-6)


def test_compute_delta_dimension_mismatch_raises():
    with pytest.raises(NebulaError, match="维度不一致"):
        compute_delta(np.ones(3, dtype=np.float32), np.ones(4, dtype=np.float32))


def test_compute_delta_nan_raises():
    with pytest.raises(NebulaError, match="NaN/Inf"):
        compute_delta(np.array([1.0, np.nan]), np.array([1.0, 0.0]))


def test_compute_delta_zero_vector_raises():
    with pytest.raises(NebulaError, match="零向量"):
        compute_delta(np.zeros(4), np.ones(4))


def test_summarize_deltas_statistics():
    records = [
        VariantDeltas(
            work_id=1,
            clip_index=0,
            start_s=0.0,
            duration_s=8.0,
            deltas={
                VARIANT_REFERENCE: {"joint": 0.0, "audio": 0.0, "video": 0.0},
                VARIANT_QUALITY: {"joint": 0.1, "audio": 0.2, "video": 0.3},
            },
        ),
        VariantDeltas(
            work_id=1,
            clip_index=1,
            start_s=8.0,
            duration_s=8.0,
            deltas={
                VARIANT_REFERENCE: {"joint": 0.0, "audio": 0.0, "video": 0.0},
                VARIANT_QUALITY: {"joint": 0.3, "audio": 0.4, "video": 0.5},
            },
        ),
    ]
    stats = summarize_deltas(records)
    quality = stats[VARIANT_QUALITY]["joint"]
    assert quality["count"] == 2
    assert quality["mean"] == pytest.approx(0.2, abs=1e-6)
    assert quality["median"] == pytest.approx(0.2, abs=1e-6)
    assert quality["min"] == pytest.approx(0.1, abs=1e-6)
    assert quality["max"] == pytest.approx(0.3, abs=1e-6)
    assert set(stats) == {VARIANT_REFERENCE, VARIANT_QUALITY}
    assert set(MODALITIES) <= set(stats[VARIANT_QUALITY])


# ──────────────────────────────────────────────────────────────────────────────
# 变体构建（需要 ffmpeg）
# ──────────────────────────────────────────────────────────────────────────────


def _two_clip_plans(sample_video: Path, config) -> tuple:
    info = probe_media(sample_video, ffprobe=config.media.ffprobe)
    plans = plan_clips(info.duration_s, clip_seconds=4.0, clips_per_work=2)
    assert len(plans) == 2
    return info, plans


@pytest.mark.integration
def test_build_work_variants_files_and_formats(sample_video: Path, config, tmp_path: Path):
    info, plans = _two_clip_plans(sample_video, config)
    build = build_work_variants(
        sample_video,
        info,
        plans,
        work_id=7,
        out_dir=tmp_path / "variants",
        media=config.media,
    )

    assert isinstance(build, VariantBuild)
    assert build.work_id == 7
    assert len(build.clips) == 2
    assert build.skipped == ()
    expected_variants = {
        VARIANT_REFERENCE,
        VARIANT_QUALITY,
        shift_variant_name(DEFAULT_SHIFTS_S[0]),
        shift_variant_name(DEFAULT_SHIFTS_S[1]),
        VARIANT_SWAP,
    }
    for position, clip in enumerate(build.clips):
        assert isinstance(clip, ClipVariants)
        assert set(clip.pairs) == expected_variants
        ref_video, ref_audio = clip.reference
        assert ref_video.is_file() and ref_audio.is_file()

        # quality：只换画面压缩，音频复用 reference
        quality_video, quality_audio = clip.pairs[VARIANT_QUALITY]
        assert quality_video.is_file() and quality_video.stat().st_size > 0
        assert quality_audio == ref_audio
        assert quality_video.read_bytes() != ref_video.read_bytes(), "crf28 再编码应与原片段不同"
        assert probe_media(quality_video, ffprobe=config.media.ffprobe).has_audio is False

        # shift：画面文件与 reference 完全相同，仅音频循环移位
        ref_samples, ref_rate, ref_channels = read_wav_pcm(ref_audio)
        for shift in DEFAULT_SHIFTS_S:
            name = shift_variant_name(shift)
            shifted_video, shifted_audio = clip.pairs[name]
            assert shifted_video == ref_video
            samples, rate, channels = read_wav_pcm(shifted_audio)
            assert (rate, channels) == (48000, 2) == (ref_rate, ref_channels)
            assert samples.shape == ref_samples.shape
            assert np.array_equal(samples, np.roll(ref_samples, int(shift * 48000), axis=0))
            # 注意：测试素材是 440Hz 正弦，0.5s / 1.0s 恰好是整数个周期，
            # 循环移位后可能与原音频逐样本相同（这正是实验报告里周期性注意点的来源）。

        # swap：音频等于下一片段的 reference 音频，画面不变
        next_clip = build.clips[(position + 1) % len(build.clips)]
        swap_video, swap_audio = clip.pairs[VARIANT_SWAP]
        assert swap_video == ref_video
        swap_samples, swap_rate, _swap_channels = read_wav_pcm(swap_audio)
        other_samples, other_rate, _other_channels = read_wav_pcm(next_clip.reference[1])
        assert swap_rate == other_rate == 48000
        assert np.array_equal(swap_samples, other_samples)


@pytest.mark.integration
def test_build_work_variants_single_clip_skips_swap(sample_video: Path, config, tmp_path: Path):
    info = probe_media(sample_video, ffprobe=config.media.ffprobe)
    plans = plan_clips(info.duration_s, clip_seconds=8.0, clips_per_work=1)
    assert len(plans) == 1
    build = build_work_variants(
        sample_video,
        info,
        plans,
        work_id=8,
        out_dir=tmp_path / "variants",
        media=config.media,
    )

    assert len(build.clips) == 1
    assert VARIANT_SWAP not in build.clips[0].pairs
    assert any("swap" in reason for reason in build.skipped)
    assert any("1 个片段" in reason for reason in build.skipped)


# ──────────────────────────────────────────────────────────────────────────────
# 评估 + 报告 schema（HashTestEmbedder 注入，不加载模型）
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_evaluate_and_report_schema_with_hash_embedder(
    sample_video: Path, config, tmp_path: Path
):
    info, plans = _two_clip_plans(sample_video, config)
    build = build_work_variants(
        sample_video,
        info,
        plans,
        work_id=7,
        out_dir=tmp_path / "variants",
        media=config.media,
    )

    embedder = HashTestEmbedder()
    results = evaluate_variants([build], embedder=embedder)
    assert len(results) == 2
    for record in results:
        assert isinstance(record, VariantDeltas)
        assert record.work_id == 7
        assert set(record.deltas) == set(build.clips[0].pairs)
        # reference 与自身比较，Δ 应约等于 0
        for modality in MODALITIES:
            value = record.deltas[VARIANT_REFERENCE][modality]
            assert value is not None and value == pytest.approx(0.0, abs=1e-6)
        for values in record.deltas.values():
            for modality in MODALITIES:
                value = values[modality]
                assert value is not None
                assert 0.0 <= value <= 2.0

    report = build_report(
        embedding_version="test-hash-v1",
        model_info={
            "checkpoint": "none",
            "embedder": "HashTestEmbedder",
            "device": "cpu",
            "autocast_bfloat16": False,
            "model_load_s": 0.0,
            "batch_size": 5,
        },
        config_info={
            "clip_seconds": 4.0,
            "clips_per_work": 2,
            "quality_crf": 28,
            "shifts_s": [0.5, 1.0],
            "batch_size": 5,
        },
        work_meta=[
            {
                "work_id": 7,
                "title": "测试作品",
                "local_path": "input/local/test.mp4",
                "duration_s": info.duration_s,
            }
        ],
        results=results,
        skipped=[{"work_id": 8, "reason": "测试跳过"}],
    )

    assert report["experiment"] == "av_desync"
    assert report["embedding_version"] == "test-hash-v1"
    assert report["n_works"] == 1
    assert report["n_clips"] == 2
    assert report["n_records"] == 2
    assert report["model"]["embedder"] == "HashTestEmbedder"
    assert report["skipped"] == [{"work_id": 8, "reason": "测试跳过"}]
    assert PERIODICITY_CAVEAT in report["notes"]

    stats = report["stats"]
    assert set(stats) == set(build.clips[0].pairs)
    for modalities in stats.values():
        assert set(modalities) == set(MODALITIES)
        for modality_values in modalities.values():
            assert set(modality_values) == {"count", "mean", "median", "min", "max", "std"}
            assert modality_values["count"] == 2
    assert stats[VARIANT_REFERENCE]["joint"]["mean"] == pytest.approx(0.0, abs=1e-6)

    conclusion = report["conclusion"]
    assert "delta_joint_means" in conclusion
    assert "ratio_vs_quality" in conclusion
    assert "periodicity_caveat" in conclusion
    assert conclusion["periodicity_caveat"] == PERIODICITY_CAVEAT
    assert isinstance(conclusion["interpretation"], str) and conclusion["interpretation"]

    work = report["works"][0]
    assert work["work_id"] == 7
    assert work["n_clips"] == 2
    assert len(work["clips"]) == 2
    assert set(work["clips"][0]["deltas"]) == set(build.clips[0].pairs)

    json_path = tmp_path / "out" / "av_desync_experiment.json"
    csv_path = tmp_path / "out" / "av_desync_experiment.csv"
    write_report(report, json_path=json_path, csv_path=csv_path)
    assert json_path.is_file() and csv_path.is_file()

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["experiment"] == report["experiment"]
    assert loaded["n_clips"] == report["n_clips"]

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == report["n_clips"] * len(build.clips[0].pairs)
    assert tuple(rows[0].keys()) == CSV_COLUMNS
    variants = {row["variant"] for row in rows}
    assert variants == set(build.clips[0].pairs)
    leftovers = [path.name for path in json_path.parent.iterdir() if ".tmp" in path.name]
    assert leftovers == [], leftovers

    summary = format_summary(report)
    assert "音画错位实验" in summary
    assert VARIANT_QUALITY in summary
    assert "结论" in summary
    assert "周期性音乐" in summary


class _BadBatchEmbedder:
    """返回行数错误的假嵌入器，用于校验 evaluate_variants 的形状防护。"""

    version = "bad"
    is_test_only = True

    def encode_pairs(self, pairs):  # noqa: ANN001 - 测试替身
        return EmbeddingBatch(joint=np.zeros((1, 4), dtype=np.float32))

    def close(self) -> None:
        return None


def test_evaluate_variants_rejects_bad_embedder_output():
    clip = ClipVariants(
        clip_index=0,
        start_s=0.0,
        duration_s=8.0,
        pairs={
            VARIANT_REFERENCE: (Path("a.mp4"), Path("a.wav")),
            VARIANT_QUALITY: (Path("b.mp4"), Path("a.wav")),
        },
    )
    build = VariantBuild(work_id=1, clips=(clip,), skipped=())
    with pytest.raises(NebulaError, match="形状异常"):
        evaluate_variants([build], embedder=_BadBatchEmbedder())  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def test_cli_help_runs_offline():
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "desync_experiment.py"), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "音画错位实验" in proc.stdout
