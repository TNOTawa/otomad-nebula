"""v0.4.1 高频音画事件扫描（events_v1）。

本模块把整条视频时间轴压缩成三层高频序列（见 docs/plans/v0.4-full-coverage-design.md 第 6 节）：

- 音频事件序列：48k 单声道解码 -> numpy STFT（win 25ms / hop 10ms，100Hz）-> 6 维特征；
- 视频事件序列：ffmpeg fps=30 / scale=96:54 / gray 单次解码 -> 6 维帧级特征；
- 音画关系序列：视频运动包络重采样到 100Hz 后与音频 onset 包络做 ±0.3s 局部互相关 -> 6 维；
- 窗口特征：对每个 CoverageClip 窗口压缩成固定 32 维（音频 12 + 视频 10 + 音画 6 + 窗口统计 4），
  与 plans 顺序严格一一对应，供 v0.4.2 Temporal Stats 表征使用。

约束（全部为契约要求）：

- 全片只解码两次（音频一条、视频一条 ffmpeg 命令），禁止逐片段重复扫描；
- 解码采用 stdout 流式读取，内存占用与视频时长基本无关（超长片也不会一次性载入全部音频）；
- 无音轨时用静音特征补齐（不调用音频解码）；无视频流 / 解码失败 / 数据异常一律抛
  EventScanError（MediaError 子类），绝不静默写出零向量；
- 所有计算确定性、无随机数；nan/inf 只会在单个统计量上归零，不会整窗伪造数据。

事件 bundle 布局（写在作品目录 work_dir 下，原子写）：

    event_scan.npz        audio/video/av 原始序列 + rates + version + 特征名
    event_features.npy    (T,32) 窗口特征便捷副本（representations 读取）
    event_manifest.json   version/rates/shapes/feature names
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from .errors import MediaError

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查
    from .sampling import CoverageClip

LOGGER = logging.getLogger(__name__)

EVENT_VERSION = "events_v1"

# 6 维音频事件特征（固定顺序，写入 bundle 与表征参数）
AUDIO_FEATURE_NAMES: tuple[str, ...] = (
    "rms",
    "log_energy",
    "spectral_flux",
    "onset_strength",
    "spectral_centroid_norm",
    "spectral_flatness",
)
# 6 维视频事件特征
VIDEO_FEATURE_NAMES: tuple[str, ...] = (
    "frame_diff_mean",
    "motion_energy",
    "edge_change",
    "luma_mean",
    "luma_std",
    "cut",
)
# 6 维音画关系特征（时间分辨率 = audio_rate_hz）
AV_FEATURE_NAMES: tuple[str, ...] = (
    "max_correlation",
    "best_lag_s",
    "corr_lag0",
    "coincidence_rate",
    "envelope_correlation",
    "event_rate_diff",
)
# 窗口压缩后的 32 维固定顺序：音频 12 + 视频 10 + 音画 6 + 窗口统计 4
WINDOW_FEATURE_NAMES: tuple[str, ...] = (
    # 音频 12
    "audio_onset_mean",
    "audio_onset_std",
    "audio_onset_p90",
    "audio_onset_rate",
    "audio_rms_mean",
    "audio_rms_std",
    "audio_log_energy_mean",
    "audio_spectral_flux_mean",
    "audio_spectral_flux_std",
    "audio_spectral_centroid_mean",
    "audio_spectral_flatness_mean",
    "audio_onset_autocorr",
    # 视频 10
    "video_frame_diff_mean",
    "video_frame_diff_std",
    "video_frame_diff_p90",
    "video_motion_mean",
    "video_motion_std",
    "video_edge_change_mean",
    "video_edge_change_std",
    "video_luma_mean",
    "video_cut_rate",
    "video_flicker_rate",
    # 音画 6
    "av_max_correlation",
    "av_best_lag_s",
    "av_corr_lag0",
    "av_coincidence_rate",
    "av_envelope_correlation",
    "av_event_rate_diff",
    # 窗口统计 4
    "window_duration_s",
    "window_valid_ratio",
    "window_start_ratio",
    "window_weight",
)

EVENT_BUNDLE_FILES: tuple[str, ...] = (
    "event_scan.npz",
    "event_features.npy",
    "event_manifest.json",
)

_DECODE_SAMPLE_RATE = 48000
_STFT_WINDOW_SECONDS = 0.025
_VIDEO_WIDTH = 96
_VIDEO_HEIGHT = 54
_VIDEO_FRAME_BYTES = _VIDEO_WIDTH * _VIDEO_HEIGHT
_AUDIO_CHUNK_SAMPLES = _DECODE_SAMPLE_RATE * 4  # 每次从管道读 4 秒
_AV_MAX_LAG_SECONDS = 0.3
_AV_LOCAL_WINDOW_SECONDS = 0.6
_AV_SMOOTH_SECONDS = 0.05
_COINCIDENCE_RADIUS_SECONDS = 0.03
_AV_EVENT_Z_THRESHOLD = 1.0
_CUT_MIN_THRESHOLD = 0.15
_CUT_MEDIAN_FACTOR = 3.0
_FLICKER_MIN_AMPLITUDE = 0.02
_PLAN_END_TOLERANCE_S = 0.25
_STDERR_TAIL_CHARS = 1200
_MAX_DECODE_PROGRESS_LOG = 20


class EventScanError(MediaError):
    """高频音画事件扫描失败（参数非法、解码失败、数据异常）。"""


@dataclass(frozen=True)
class EventFeatures:
    """一次全片事件扫描的结果。

    字段顺序与 docs/plans/v0.4-full-coverage-design.md 第 6 节一致：
    audio/video/av 为原始高频序列，window_features 为与 CoverageClip 一一对应的 32 维压缩特征。
    """

    audio: np.ndarray  # (n_a, 6) float32 @ audio_rate_hz
    video: np.ndarray  # (n_v, 6) float32 @ video_fps_effective
    av: np.ndarray  # (n_a, 6) float32 @ audio_rate_hz
    window_features: np.ndarray  # (T, 32) float32，与 plans 顺序一一对应
    audio_rate_hz: float
    video_fps: float
    video_fps_effective: float
    duration_s: float
    version: str


# ---------------------------------------------------------------- 通用小工具


def _stderr_tail(stderr: bytes | str | None, limit: int = _STDERR_TAIL_CHARS) -> str:
    if not stderr:
        return ""
    text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else str(stderr)
    text = text.strip()
    if len(text) > limit:
        return "..." + text[-limit:]
    return text


def _require_positive_number(value: Any, *, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EventScanError(f"{what}必须是数字，收到 {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise EventScanError(f"{what}必须大于 0，收到 {value!r}")
    return number


def _require_positive_int(value: Any, *, what: str) -> int:
    number = _require_positive_number(value, what=what)
    integer = int(round(number))
    if abs(number - integer) > 1e-9:
        raise EventScanError(f"{what}必须是整数，收到 {value!r}")
    return integer


def _run_capture(command: Sequence[str], *, what: str, timeout_s: float) -> subprocess.CompletedProcess:
    """运行完整捕获型子进程并把启动/超时错误统一转成 MediaError。"""

    try:
        return subprocess.run(
            list(command),
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        raise MediaError(f"找不到可执行文件（{command[0]}）：请检查 PATH 或 config.yaml 中的 media 配置") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"{what}超时（>{timeout_s:g} 秒）：{command[0]}") from exc
    except OSError as exc:
        raise MediaError(f"{what}无法启动（{command[0]}）：{exc}") from exc


def _resolve_ffprobe(ffmpeg: str) -> str:
    """尽量找到与 ffmpeg 同目录的 ffprobe，否则退回 PATH 中的 ffprobe。"""

    resolved = shutil.which(str(ffmpeg))
    if resolved is not None:
        binary = Path(resolved)
        candidates = (
            binary.with_name("ffprobe" + binary.suffix),
            binary.with_name("ffprobe"),
        )
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
    found = shutil.which("ffprobe")
    return found if found is not None else "ffprobe"


# ---------------------------------------------------------------- 探测与校验


@dataclass(frozen=True)
class _ProbeInfo:
    duration_s: float
    has_audio: bool
    has_video: bool


def _probe_source(source: Path, ffprobe: str, timeout_s: float) -> _ProbeInfo:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(source),
    ]
    proc = _run_capture(command, what="ffprobe 探测", timeout_s=timeout_s)
    if proc.returncode != 0:
        tail = _stderr_tail(proc.stderr)
        detail = f"；stderr 摘要: {tail}" if tail else ""
        raise MediaError(f"ffprobe 无法解析媒体文件: {source}{detail}")
    try:
        payload = json.loads((proc.stdout or b"").decode("utf-8", "replace") or "{}")
    except json.JSONDecodeError as exc:
        raise MediaError(f"ffprobe 输出不是合法 JSON: {source}") from exc
    streams = payload.get("streams") or []
    if not isinstance(streams, list) or not streams:
        raise MediaError(f"ffprobe 未解析到任何音视频流: {source}")
    has_video = any(
        stream.get("codec_type") == "video"
        and not bool((stream.get("disposition") or {}).get("attached_pic"))
        for stream in streams
        if isinstance(stream, dict)
    )
    has_audio = any(
        isinstance(stream, dict) and stream.get("codec_type") == "audio" for stream in streams
    )
    duration_raw = None
    payload_format = payload.get("format")
    if isinstance(payload_format, dict):
        duration_raw = payload_format.get("duration")
    if duration_raw is None:
        candidates = [
            stream.get("duration")
            for stream in streams
            if isinstance(stream, dict) and stream.get("duration") is not None
        ]
        if candidates:
            duration_raw = max(candidates, key=lambda value: float(value))
    if duration_raw is None:
        raise MediaError(f"无法从 ffprobe 结果读取媒体时长: {source}")
    duration_s = float(duration_raw)
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise MediaError(f"媒体时长非法（{duration_raw!r}）: {source}")
    return _ProbeInfo(duration_s=duration_s, has_audio=has_audio, has_video=has_video)


def _plan_values(
    plans: Sequence["CoverageClip"], duration_s: float
) -> list[tuple[float, float, float, float, float | None]]:
    """校验窗口计划，返回 [(start, end, valid_duration, raw_duration, weight)]。

    计划对象只需具备 index/start_s/duration_s 以及可选的 valid_duration_s/weight
    （CoverageClip 契约字段）；本模块不 import sampling，避免与 t1 并行开发耦合。
    """

    if not plans:
        raise EventScanError("事件扫描至少需要一个窗口计划（CoverageClip）")
    values: list[tuple[float, float, float, float, float | None]] = []
    for position, plan in enumerate(plans):
        start_raw = getattr(plan, "start_s", None)
        duration_raw = getattr(plan, "duration_s", None)
        if start_raw is None or duration_raw is None:
            raise EventScanError(
                f"第 {position} 个窗口计划缺少 start_s/duration_s 属性: {plan!r}"
            )
        start = float(start_raw)
        raw_duration = float(duration_raw)
        if not math.isfinite(start) or start < 0:
            raise EventScanError(f"第 {position} 个窗口 start_s 非法: {start_raw!r}")
        if not math.isfinite(raw_duration) or raw_duration <= 0:
            raise EventScanError(f"第 {position} 个窗口 duration_s 非法: {duration_raw!r}")
        end = start + raw_duration
        if start >= duration_s + _PLAN_END_TOLERANCE_S:
            raise EventScanError(
                f"第 {position} 个窗口起点 {start:.3f}s 超出视频时长 {duration_s:.3f}s"
            )
        if end > duration_s + _PLAN_END_TOLERANCE_S:
            raise EventScanError(
                f"第 {position} 个窗口结束 {end:.3f}s 超出视频时长 {duration_s:.3f}s"
            )
        valid_raw = getattr(plan, "valid_duration_s", None)
        if valid_raw is None:
            valid = raw_duration
        else:
            valid = float(valid_raw)
            if not math.isfinite(valid) or valid < 0:
                raise EventScanError(f"第 {position} 个窗口 valid_duration_s 非法: {valid_raw!r}")
            valid = min(valid, raw_duration)
        weight_raw = getattr(plan, "weight", None)
        weight = None
        if weight_raw is not None:
            weight = float(weight_raw)
            if not math.isfinite(weight) or weight < 0:
                raise EventScanError(f"第 {position} 个窗口 weight 非法: {weight_raw!r}")
        values.append((start, min(end, duration_s), valid, raw_duration, weight))
    return values


# ---------------------------------------------------------------- 数值工具


def _rolling_mean(values: np.ndarray, size: int) -> np.ndarray:
    """边缘补全（edge padding）的滑动均值，输出长度与输入一致。"""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return np.zeros(0, dtype=np.float32)
    if size <= 1:
        return array.astype(np.float32)
    pad_left = size // 2
    pad_right = size - 1 - pad_left
    padded = np.pad(array, (pad_left, pad_right), mode="edge")
    cumulative = np.concatenate(([0.0], np.cumsum(padded, dtype=np.float64)))
    out = (cumulative[size:] - cumulative[:-size]) / float(size)
    return out.astype(np.float32)


def _rolling_max(values: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return np.zeros(0, dtype=np.float32)
    if size <= 1:
        return array.copy()
    pad_left = size // 2
    pad_right = size - 1 - pad_left
    padded = np.pad(array, (pad_left, pad_right), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)
    return windows.max(axis=1).astype(np.float32)


def _zscore(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return np.zeros(0, dtype=np.float32)
    std = float(array.std())
    if not math.isfinite(std) or std <= 1e-12:
        return np.zeros(array.shape, dtype=np.float32)
    return ((array - float(array.mean())) / std).astype(np.float32)


def _shift(values: np.ndarray, lag: int) -> np.ndarray:
    """返回 shifted[t] = values[t + lag]，越界补 0。"""

    array = np.asarray(values, dtype=np.float32)
    out = np.zeros_like(array)
    count = array.size
    if lag == 0:
        out[:] = array
    elif lag > 0:
        if lag < count:
            out[: count - lag] = array[lag:]
    else:
        offset = -lag
        if offset < count:
            out[offset:] = array[: count - offset]
    return out


def _mean_std_p90(values: np.ndarray) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0, 0.0, 0.0
    return (
        float(array.mean()),
        float(array.std()),
        float(np.percentile(array, 90)),
    )


def _window_slice(start: float, end: float, rate: float, size: int) -> slice:
    """把时间窗口 [start, end) 映射到网格下标切片，保证至少覆盖 1 个采样。"""

    if size <= 0:
        return slice(0, 0)
    left = min(max(0, int(round(start * rate))), size - 1)
    right = min(max(left + 1, int(round(end * rate))), size)
    return slice(left, right)


def _rhythm_strength(onset: np.ndarray, audio_rate_hz: float) -> float:
    """窗口内 onset 包络在 0.2s..1.0s 延迟上的最大归一化自相关（节奏强度）。"""

    values = np.asarray(onset, dtype=np.float64)
    if values.size < 4:
        return 0.0
    energy = float(np.sum(values * values))
    if not math.isfinite(energy) or energy <= 1e-12:
        return 0.0
    low = max(1, int(round(0.2 * audio_rate_hz)))
    high = min(values.size - 1, int(round(1.0 * audio_rate_hz)))
    best = 0.0
    for lag in range(low, high + 1):
        numerator = float(np.sum(values[:-lag] * values[lag:]))
        best = max(best, numerator / energy)
    return max(0.0, best)


# ---------------------------------------------------------------- 音频事件序列


def _silent_audio_features(duration_s: float, audio_rate_hz: float) -> tuple[np.ndarray, float]:
    """无音轨时按静音解析式生成 (n,6) 特征，避免为超长视频分配巨大零数组。"""

    hop = max(1, int(round(_DECODE_SAMPLE_RATE / audio_rate_hz)))
    effective_rate = _DECODE_SAMPLE_RATE / hop
    count = max(1, int(round(duration_s * effective_rate)))
    features = np.zeros((count, 6), dtype=np.float32)
    features[:, 1] = math.log10(1e-10)  # log_energy：能量 0 时的固定下限
    features[:, 5] = 1.0  # spectral_flatness：全零幅度谱的几何/算术均值比
    return features, effective_rate


def _audio_frame_features(segment: np.ndarray, prev_magnitude: np.ndarray, hann: np.ndarray,
                          frequencies: np.ndarray, nyquist: float) -> tuple[np.ndarray, np.ndarray]:
    """计算单个 25ms 音频帧的 6 维特征；segment 需已补零到窗长。"""

    signal = np.asarray(segment, dtype=np.float64)
    energy = float(np.sum(signal * signal))
    rms = math.sqrt(energy / max(1, signal.size))
    magnitude = np.abs(np.fft.rfft(signal * hann)).astype(np.float64)
    difference = magnitude - prev_magnitude
    total_flux = float(np.sum(np.abs(difference)))
    positive_flux = float(np.sum(np.maximum(difference, 0.0)))
    bin_count = max(1, magnitude.size)
    magnitude_sum = float(np.sum(magnitude))
    if magnitude_sum > 1e-12:
        centroid = float(np.dot(magnitude, frequencies)) / magnitude_sum / nyquist
    else:
        centroid = 0.0
    geometric = float(np.exp(np.mean(np.log(magnitude + 1e-10))))
    arithmetic = float(np.mean(magnitude)) + 1e-10
    flatness = geometric / arithmetic
    feature = np.array(
        [
            rms,
            math.log10(energy + 1e-10),
            total_flux / bin_count,
            positive_flux / bin_count,
            min(max(centroid, 0.0), 1.0),
            min(max(flatness, 0.0), 1.0),
        ],
        dtype=np.float32,
    )
    return feature, magnitude


def _decode_audio_features(
    source: Path,
    *,
    ffmpeg: str,
    audio_rate_hz: int,
    duration_s: float,
    timeout_s: float,
) -> tuple[np.ndarray, float]:
    """单次 ffmpeg 音频解码（流式读取）并计算 100Hz 音频事件序列。"""

    hop = max(1, int(round(_DECODE_SAMPLE_RATE / float(audio_rate_hz))))
    effective_rate = _DECODE_SAMPLE_RATE / hop
    window = max(1, int(round(_STFT_WINDOW_SECONDS * _DECODE_SAMPLE_RATE)))
    n_expected = max(1, int(round(duration_s * effective_rate)))
    hann = np.hanning(window).astype(np.float64)
    frequencies = np.fft.rfftfreq(window, d=1.0 / _DECODE_SAMPLE_RATE).astype(np.float64)
    nyquist = _DECODE_SAMPLE_RATE / 2.0
    zero_magnitude = np.zeros(frequencies.size, dtype=np.float64)

    features = np.zeros((n_expected, 6), dtype=np.float32)
    buffer = np.zeros(0, dtype=np.float32)
    buffer_start = 0
    frame_index = 0
    prev_magnitude = zero_magnitude

    command = [
        ffmpeg,
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(_DECODE_SAMPLE_RATE),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    deadline = time.monotonic() + timeout_s
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise MediaError(f"找不到 ffmpeg 可执行文件（{ffmpeg}）：请检查 PATH 或 config.yaml") from exc
    except OSError as exc:
        raise MediaError(f"无法启动 ffmpeg 音频解码（{ffmpeg}）：{exc}") from exc

    try:
        assert process.stdout is not None and process.stderr is not None
        while True:
            if time.monotonic() > deadline:
                process.kill()
                raise MediaError(f"音频解码超时（>{timeout_s:g} 秒）: {source}")
            raw = process.stdout.read(_AUDIO_CHUNK_SAMPLES * 4)
            if not raw:
                break
            samples = np.frombuffer(raw, dtype=np.float32)
            if samples.size:
                buffer = (
                    np.concatenate((buffer, samples))
                    if buffer.size
                    else samples.astype(np.float32, copy=True)
                )
            while frame_index < n_expected:
                offset = frame_index * hop - buffer_start
                if offset + window > buffer.size:
                    break
                segment = buffer[offset : offset + window]
                feature, prev_magnitude = _audio_frame_features(
                    segment, prev_magnitude, hann, frequencies, nyquist
                )
                features[frame_index] = feature
                frame_index += 1
            if frame_index < n_expected:
                next_start = frame_index * hop
                trim = next_start - buffer_start
                if trim > 0:
                    buffer = buffer[trim:]
                    buffer_start = next_start
        stderr = process.stderr.read()
        returncode = process.wait(timeout=max(1.0, deadline - time.monotonic()))
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    if returncode != 0:
        tail = _stderr_tail(stderr)
        detail = f"；stderr 摘要: {tail}" if tail else ""
        raise MediaError(f"音频解码失败（退出码 {returncode}）: {source}{detail}")

    # 解码样本可能比容器时长略短/略长：用零补齐剩余帧，保证网格与 duration_s 对齐
    while frame_index < n_expected:
        offset = frame_index * hop - buffer_start
        segment = np.zeros(window, dtype=np.float32)
        if 0 <= offset < buffer.size:
            available = buffer[offset : offset + window]
            segment[: available.size] = available
        feature, prev_magnitude = _audio_frame_features(
            segment, prev_magnitude, hann, frequencies, nyquist
        )
        features[frame_index] = feature
        frame_index += 1

    if not np.isfinite(features).all():
        raise EventScanError(f"音频事件特征出现非有限值: {source}")
    return features, effective_rate


# ---------------------------------------------------------------- 视频事件序列


def _read_exact(stream: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        data = stream.read(remaining)
        if not data:
            break
        chunks.append(data)
        remaining -= len(data)
    if not chunks:
        return b""
    if remaining > 0:
        raise MediaError("视频解码输出被截断（帧数据不完整）")
    return b"".join(chunks)


def _edge_energy(gray: np.ndarray) -> float:
    dx = float(np.mean(np.abs(np.diff(gray, axis=1)))) if gray.shape[1] > 1 else 0.0
    dy = float(np.mean(np.abs(np.diff(gray, axis=0)))) if gray.shape[0] > 1 else 0.0
    return 0.5 * (dx + dy)


def _decode_video_features(
    source: Path,
    *,
    ffmpeg: str,
    video_fps: int,
    timeout_s: float,
) -> np.ndarray:
    """单次 ffmpeg 视频解码（流式逐帧读取）并计算帧级 6 维事件序列。"""

    command = [
        ffmpeg,
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(source),
        "-an",
        "-sn",
        "-vf",
        f"fps={int(video_fps)},scale={_VIDEO_WIDTH}:{_VIDEO_HEIGHT},format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    deadline = time.monotonic() + timeout_s
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise MediaError(f"找不到 ffmpeg 可执行文件（{ffmpeg}）：请检查 PATH 或 config.yaml") from exc
    except OSError as exc:
        raise MediaError(f"无法启动 ffmpeg 视频解码（{ffmpeg}）：{exc}") from exc

    frame_diffs: list[float] = []
    motion: list[float] = []
    edge_changes: list[float] = []
    luma_means: list[float] = []
    luma_stds: list[float] = []
    previous: np.ndarray | None = None
    previous_edge = 0.0
    stderr = b""
    returncode = -1
    try:
        assert process.stdout is not None and process.stderr is not None
        frame_index = 0
        while True:
            if time.monotonic() > deadline:
                process.kill()
                raise MediaError(f"视频解码超时（>{timeout_s:g} 秒）: {source}")
            raw = _read_exact(process.stdout, _VIDEO_FRAME_BYTES)
            if not raw:
                break
            gray = (
                np.frombuffer(raw, dtype=np.uint8)
                .reshape(_VIDEO_HEIGHT, _VIDEO_WIDTH)
                .astype(np.float32)
                / 255.0
            )
            edge = _edge_energy(gray)
            if previous is None:
                frame_diff = 0.0
                motion_energy = 0.0
                edge_change = 0.0
            else:
                difference = np.abs(gray - previous)
                frame_diff = float(difference.mean())
                motion_energy = float(np.mean(difference * difference))
                edge_change = abs(edge - previous_edge)
            frame_diffs.append(frame_diff)
            motion.append(motion_energy)
            edge_changes.append(edge_change)
            luma_means.append(float(gray.mean()))
            luma_stds.append(float(gray.std()))
            previous = gray
            previous_edge = edge
            frame_index += 1
        stderr = process.stderr.read()
        returncode = process.wait(timeout=max(1.0, deadline - time.monotonic()))
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    if returncode != 0:
        tail = _stderr_tail(stderr)
        detail = f"；stderr 摘要: {tail}" if tail else ""
        raise MediaError(f"视频解码失败（退出码 {returncode}）: {source}{detail}")
    if not frame_diffs:
        raise EventScanError(f"视频解码结果为空（没有可用帧）: {source}")

    diff_array = np.asarray(frame_diffs, dtype=np.float32)
    median = float(np.median(diff_array))
    threshold = max(_CUT_MIN_THRESHOLD, _CUT_MEDIAN_FACTOR * median)
    cuts = (diff_array > threshold).astype(np.float32)
    features = np.column_stack(
        [
            diff_array,
            np.asarray(motion, dtype=np.float32),
            np.asarray(edge_changes, dtype=np.float32),
            np.asarray(luma_means, dtype=np.float32),
            np.asarray(luma_stds, dtype=np.float32),
            cuts,
        ]
    ).astype(np.float32)
    if not np.isfinite(features).all():
        raise EventScanError(f"视频事件特征出现非有限值: {source}")
    return features


# ---------------------------------------------------------------- 音画关系序列


def _av_relation_features(
    audio_features: np.ndarray,
    video_features: np.ndarray,
    *,
    audio_rate_hz: float,
    video_fps_effective: float,
) -> np.ndarray:
    """把视频运动包络重采样到音频网格，计算 6 维局部音画关系序列。"""

    n_audio = audio_features.shape[0]
    n_video = video_features.shape[0]
    audio_onset = audio_features[:, 3].astype(np.float64)
    audio_time = (np.arange(n_audio, dtype=np.float64) + 0.5) / audio_rate_hz
    if n_video > 0 and video_fps_effective > 0:
        video_time = (np.arange(n_video, dtype=np.float64) + 0.5) / video_fps_effective
        motion = np.interp(audio_time, video_time, video_features[:, 1].astype(np.float64))
    else:
        motion = np.zeros(n_audio, dtype=np.float64)

    audio_norm = _zscore(audio_onset)
    visual_norm = _zscore(motion)
    local = max(3, int(round(_AV_LOCAL_WINDOW_SECONDS * audio_rate_hz)))
    max_lag = max(1, int(round(_AV_MAX_LAG_SECONDS * audio_rate_hz)))
    lags = np.arange(-max_lag, max_lag + 1, dtype=np.int64)

    correlation = np.empty((lags.size, n_audio), dtype=np.float32)
    for index, lag in enumerate(lags):
        correlation[index] = _rolling_mean(audio_norm * _shift(visual_norm, int(lag)), local)
    tiny_penalty = np.abs(lags).astype(np.float32)[:, None] * np.float32(1e-6)
    best_index = np.argmax(correlation - tiny_penalty, axis=0)
    columns = np.arange(n_audio)
    max_correlation = correlation[best_index, columns]
    best_lag = lags[best_index].astype(np.float32) / float(audio_rate_hz)

    zero_index = int(np.argmin(np.abs(lags)))
    corr_lag0 = correlation[zero_index]

    audio_event = (audio_norm > _AV_EVENT_Z_THRESHOLD).astype(np.float32)
    visual_event = (visual_norm > _AV_EVENT_Z_THRESHOLD).astype(np.float32)
    radius = max(1, int(round(_COINCIDENCE_RADIUS_SECONDS * audio_rate_hz)))
    visual_near = _rolling_max(visual_event, 2 * radius + 1)
    coincidence = _rolling_mean(audio_event * visual_near, local)

    smooth = max(3, int(round(_AV_SMOOTH_SECONDS * audio_rate_hz)))
    audio_smooth = _rolling_mean(audio_norm, smooth)
    visual_smooth = _rolling_mean(visual_norm, smooth)
    mean_a = _rolling_mean(audio_smooth, local)
    mean_v = _rolling_mean(visual_smooth, local)
    mean_cross = _rolling_mean(audio_smooth * visual_smooth, local)
    var_a = np.maximum(_rolling_mean(audio_smooth * audio_smooth, local) - mean_a * mean_a, 0.0)
    var_v = np.maximum(_rolling_mean(visual_smooth * visual_smooth, local) - mean_v * mean_v, 0.0)
    covariance = mean_cross - mean_a * mean_v
    envelope_correlation = np.clip(
        covariance / np.sqrt(var_a * var_v + 1e-8), -1.0, 1.0
    ).astype(np.float32)

    event_rate_diff = _rolling_mean(visual_event - audio_event, local) * float(audio_rate_hz)

    matrix = np.column_stack(
        [
            max_correlation,
            best_lag,
            corr_lag0,
            coincidence,
            envelope_correlation,
            event_rate_diff,
        ]
    ).astype(np.float32)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------- 窗口压缩


def _window_features(
    plan_values: list[tuple[float, float, float, float, float | None]],
    *,
    audio_features: np.ndarray,
    video_features: np.ndarray,
    av_features: np.ndarray,
    audio_rate_hz: float,
    video_fps_effective: float,
    duration_s: float,
) -> np.ndarray:
    """把高频序列按窗口压成 (T,32) 固定顺序特征。"""

    n_audio = audio_features.shape[0]
    n_video = video_features.shape[0]
    global_onset = audio_features[:, 3].astype(np.float64)
    onset_threshold = float(global_onset.mean() + global_onset.std())

    luma = video_features[:, 3].astype(np.float64)
    flicker = np.zeros(n_video, dtype=np.float32)
    if n_video >= 3:
        diff = np.diff(luma)
        previous = diff[:-1]
        current = diff[1:]
        sign_flip = np.sign(previous) != np.sign(current)
        amplitude = np.minimum(np.abs(previous), np.abs(current))
        flicker[2:] = (sign_flip & (amplitude > _FLICKER_MIN_AMPLITUDE)).astype(np.float32)

    default_weight = 1.0 / float(len(plan_values))
    rows: list[np.ndarray] = []
    for start, end, valid, raw_duration, weight in plan_values:
        duration = max(1e-6, float(end) - float(start))
        audio_slice = _window_slice(start, end, audio_rate_hz, n_audio)
        video_slice = _window_slice(start, end, video_fps_effective, n_video)
        av_slice = _window_slice(start, end, audio_rate_hz, av_features.shape[0])

        audio_segment = audio_features[audio_slice]
        onset = audio_segment[:, 3].astype(np.float64)
        rms = audio_segment[:, 0].astype(np.float64)
        log_energy = audio_segment[:, 1].astype(np.float64)
        flux = audio_segment[:, 2].astype(np.float64)
        centroid = audio_segment[:, 4].astype(np.float64)
        flatness = audio_segment[:, 5].astype(np.float64)
        onset_mean, onset_std, onset_p90 = _mean_std_p90(onset)
        onset_rate = float(np.count_nonzero(onset > onset_threshold)) / duration
        rms_mean, rms_std, _ = _mean_std_p90(rms)
        flux_mean, flux_std, _ = _mean_std_p90(flux)
        log_energy_mean = float(log_energy.mean()) if log_energy.size else 0.0
        centroid_mean = float(centroid.mean()) if centroid.size else 0.0
        flatness_mean = float(flatness.mean()) if flatness.size else 0.0
        rhythm = _rhythm_strength(onset, audio_rate_hz)

        video_segment = video_features[video_slice]
        frame_diff = video_segment[:, 0].astype(np.float64)
        motion = video_segment[:, 1].astype(np.float64)
        edge_change = video_segment[:, 2].astype(np.float64)
        luma_mean = video_segment[:, 3].astype(np.float64)
        cut = video_segment[:, 5].astype(np.float64)
        frame_diff_mean, frame_diff_std, frame_diff_p90 = _mean_std_p90(frame_diff)
        motion_mean, motion_std, _ = _mean_std_p90(motion)
        edge_mean, edge_std, _ = _mean_std_p90(edge_change)
        cut_rate = float(cut.sum()) / duration
        flicker_rate = float(flicker[video_slice].sum()) / duration
        video_luma_mean = float(luma_mean.mean()) if luma_mean.size else 0.0

        av_segment = av_features[av_slice]
        av_means = (
            av_segment.astype(np.float64).mean(axis=0)
            if av_segment.size
            else np.zeros(len(AV_FEATURE_NAMES), dtype=np.float64)
        )

        window_vector = np.concatenate(
            [
                np.array(
                    [
                        onset_mean,
                        onset_std,
                        onset_p90,
                        onset_rate,
                        rms_mean,
                        rms_std,
                        log_energy_mean,
                        flux_mean,
                        flux_std,
                        centroid_mean,
                        flatness_mean,
                        rhythm,
                    ],
                    dtype=np.float64,
                ),
                np.array(
                    [
                        frame_diff_mean,
                        frame_diff_std,
                        frame_diff_p90,
                        motion_mean,
                        motion_std,
                        edge_mean,
                        edge_std,
                        video_luma_mean,
                        cut_rate,
                        flicker_rate,
                    ],
                    dtype=np.float64,
                ),
                av_means,
                np.array(
                    [
                        duration,
                        (valid / raw_duration) if raw_duration > 0 else 1.0,
                        (start / duration_s) if duration_s > 0 else 0.0,
                        float(default_weight if weight is None else weight),
                    ],
                    dtype=np.float64,
                ),
            ]
        )
        rows.append(window_vector.astype(np.float32))

    matrix = np.vstack(rows).astype(np.float32)
    if matrix.shape[1] != len(WINDOW_FEATURE_NAMES):  # pragma: no cover - 防御性检查
        raise EventScanError(
            f"窗口特征维度 {matrix.shape[1]} 与特征名数量 {len(WINDOW_FEATURE_NAMES)} 不一致"
        )
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------- 主入口


def scan_events(
    source: Path,
    plans: Sequence["CoverageClip"],
    *,
    ffmpeg: str = "ffmpeg",
    audio_rate_hz: int = 100,
    video_fps: int = 30,
    timeout_s: float = 600.0,
) -> EventFeatures:
    """对整部视频做一次高频事件扫描（全片只解码两次）。

    参数
    ----
    source:
        视频文件路径。
    plans:
        窗口计划序列（CoverageClip 或具备 index/start_s/duration_s 的同类对象），
        window_features 的行与它严格一一对应。
    ffmpeg:
        ffmpeg 可执行文件；同时用于推导同目录 ffprobe。
    audio_rate_hz:
        音频事件序列时间分辨率（默认 100Hz，STFT hop 10ms）。
    video_fps:
        视频事件扫描帧率上限（默认 30fps；源帧率更低时由 ffmpeg 复制帧）。
    timeout_s:
        单条解码/探测命令的超时秒数。

    异常
    ----
    EventScanError / MediaError:
        窗口非法、源文件缺失、没有视频流、音视频解码失败、数据非有限等。
    """

    source_path = Path(source)
    if not source_path.is_file():
        raise EventScanError(f"事件扫描源文件不存在: {source_path}")
    rate = _require_positive_int(audio_rate_hz, what="audio_rate_hz")
    fps = _require_positive_int(video_fps, what="video_fps")
    timeout = _require_positive_number(timeout_s, what="timeout_s")
    ffprobe = _resolve_ffprobe(str(ffmpeg))

    probe = _probe_source(source_path, ffprobe, timeout)
    if not probe.has_video:
        raise EventScanError(f"源文件没有视频流，无法做事件扫描: {source_path}")
    plan_values = _plan_values(plans, probe.duration_s)

    if probe.has_audio:
        audio_features, audio_rate_effective = _decode_audio_features(
            source_path,
            ffmpeg=str(ffmpeg),
            audio_rate_hz=rate,
            duration_s=probe.duration_s,
            timeout_s=timeout,
        )
    else:
        audio_features, audio_rate_effective = _silent_audio_features(
            probe.duration_s, float(rate)
        )
        LOGGER.warning("源文件没有音轨，事件扫描使用静音特征: %s", source_path)

    video_features = _decode_video_features(
        source_path,
        ffmpeg=str(ffmpeg),
        video_fps=fps,
        timeout_s=timeout,
    )
    video_fps_effective = float(video_features.shape[0]) / probe.duration_s

    av_features = _av_relation_features(
        audio_features,
        video_features,
        audio_rate_hz=audio_rate_effective,
        video_fps_effective=video_fps_effective,
    )
    window = _window_features(
        plan_values,
        audio_features=audio_features,
        video_features=video_features,
        av_features=av_features,
        audio_rate_hz=audio_rate_effective,
        video_fps_effective=video_fps_effective,
        duration_s=probe.duration_s,
    )

    features = EventFeatures(
        audio=np.ascontiguousarray(audio_features, dtype=np.float32),
        video=np.ascontiguousarray(video_features, dtype=np.float32),
        av=np.ascontiguousarray(av_features, dtype=np.float32),
        window_features=np.ascontiguousarray(window, dtype=np.float32),
        audio_rate_hz=float(audio_rate_effective),
        video_fps=float(fps),
        video_fps_effective=float(video_fps_effective),
        duration_s=float(probe.duration_s),
        version=EVENT_VERSION,
    )
    LOGGER.info(
        "事件扫描完成: %s, %.2fs, audio=%d@%.1fHz, video=%d@%.2ffps, windows=%d",
        source_path,
        probe.duration_s,
        features.audio.shape[0],
        features.audio_rate_hz,
        features.video.shape[0],
        features.video_fps_effective,
        features.window_features.shape[0],
    )
    return features


# ---------------------------------------------------------------- bundle 读写


def _float32_matrix(array: Any, *, name: str, columns: int) -> np.ndarray:
    matrix = np.asarray(array, dtype=np.float32)
    if matrix.ndim != 2:
        raise EventScanError(f"事件序列 {name} 必须是二维矩阵，收到 {matrix.shape}")
    if matrix.shape[1] != columns:
        raise EventScanError(
            f"事件序列 {name} 列数应为 {columns}，收到 {matrix.shape[1]}"
        )
    if matrix.shape[0] < 1:
        raise EventScanError(f"事件序列 {name} 为空")
    if not np.isfinite(matrix).all():
        raise EventScanError(f"事件序列 {name} 含非有限值，拒绝写入")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _validated_features(features: EventFeatures) -> dict[str, np.ndarray]:
    if not isinstance(features, EventFeatures):
        raise EventScanError(f"save_events_bundle 需要 EventFeatures，收到 {type(features).__name__}")
    audio = _float32_matrix(features.audio, name="audio", columns=len(AUDIO_FEATURE_NAMES))
    video = _float32_matrix(features.video, name="video", columns=len(VIDEO_FEATURE_NAMES))
    av = _float32_matrix(features.av, name="av", columns=len(AV_FEATURE_NAMES))
    window = _float32_matrix(
        features.window_features, name="window_features", columns=len(WINDOW_FEATURE_NAMES)
    )
    if av.shape[0] != audio.shape[0]:
        raise EventScanError(
            f"音画序列行数 {av.shape[0]} 与音频序列行数 {audio.shape[0]} 不一致"
        )
    rates = {
        "audio_rate_hz": _require_positive_number(features.audio_rate_hz, what="audio_rate_hz"),
        "video_fps": _require_positive_number(features.video_fps, what="video_fps"),
        "video_fps_effective": _require_positive_number(
            features.video_fps_effective, what="video_fps_effective"
        ),
        "duration_s": _require_positive_number(features.duration_s, what="duration_s"),
    }
    version = str(features.version or "").strip()
    if not version:
        raise EventScanError("EventFeatures.version 不能为空")
    arrays = {
        "audio": audio,
        "video": video,
        "av": av,
        "window_features": window,
        "audio_rate_hz": np.float64(rates["audio_rate_hz"]),
        "video_fps": np.float64(rates["video_fps"]),
        "video_fps_effective": np.float64(rates["video_fps_effective"]),
        "duration_s": np.float64(rates["duration_s"]),
        "version": np.asarray(version),
        "audio_feature_names": np.asarray(AUDIO_FEATURE_NAMES, dtype=str),
        "video_feature_names": np.asarray(VIDEO_FEATURE_NAMES, dtype=str),
        "av_feature_names": np.asarray(AV_FEATURE_NAMES, dtype=str),
        "window_feature_names": np.asarray(WINDOW_FEATURE_NAMES, dtype=str),
    }
    return arrays


def _manifest_payload(features: EventFeatures) -> dict[str, Any]:
    return {
        "version": str(features.version),
        "audio_rate_hz": float(features.audio_rate_hz),
        "video_fps": float(features.video_fps),
        "video_fps_effective": float(features.video_fps_effective),
        "duration_s": float(features.duration_s),
        "n_windows": int(features.window_features.shape[0]),
        "shapes": {
            "audio": [int(value) for value in features.audio.shape],
            "video": [int(value) for value in features.video.shape],
            "av": [int(value) for value in features.av.shape],
            "window_features": [int(value) for value in features.window_features.shape],
        },
        "feature_names": {
            "audio": list(AUDIO_FEATURE_NAMES),
            "video": list(VIDEO_FEATURE_NAMES),
            "av": list(AV_FEATURE_NAMES),
            "window": list(WINDOW_FEATURE_NAMES),
        },
    }


def _temp_file(directory: Path, *, prefix: str, suffix: str) -> Path:
    handle, name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=str(directory))
    os.close(handle)
    return Path(name)


def save_events_bundle(work_dir: Path, features: EventFeatures) -> None:
    """把事件扫描结果原子写入作品目录（npz + npy + manifest.json）。

    三个文件都先写同目录临时文件再 os.replace，避免中断留下半个 bundle。
    """

    arrays = _validated_features(features)
    target = Path(work_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest = _manifest_payload(features)

    scan_path = target / "event_scan.npz"
    features_path = target / "event_features.npy"
    manifest_path = target / "event_manifest.json"
    temporary: list[Path] = []
    try:
        scan_tmp = _temp_file(target, prefix=".event_scan.", suffix=".npz")
        temporary.append(scan_tmp)
        np.savez_compressed(scan_tmp, **arrays)
        os.replace(scan_tmp, scan_path)

        features_tmp = _temp_file(target, prefix=".event_features.", suffix=".npy")
        temporary.append(features_tmp)
        np.save(features_tmp, arrays["window_features"], allow_pickle=False)
        os.replace(features_tmp, features_path)

        manifest_tmp = _temp_file(target, prefix=".event_manifest.", suffix=".json")
        temporary.append(manifest_tmp)
        manifest_tmp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(manifest_tmp, manifest_path)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    LOGGER.info(
        "事件 bundle 已保存: %s（%d 窗口, 版本 %s）",
        target,
        manifest["n_windows"],
        manifest["version"],
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EventScanError(f"事件 manifest 无法读取: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EventScanError(f"事件 manifest 顶层必须是对象: {path}")
    return payload


def load_events_bundle(work_dir: Path) -> EventFeatures | None:
    """读取事件 bundle；三个文件任一缺失返回 None，损坏或不兼容则抛异常。

    版本与当前 EVENT_VERSION 不一致时返回 None（旧算法结果不参与当前表征），
    但结构损坏会抛 EventScanError，避免把坏数据静默当成“没有事件”。
    """

    target = Path(work_dir)
    if not target.is_dir():
        return None
    paths = {name: target / name for name in EVENT_BUNDLE_FILES}
    if not all(path.is_file() for path in paths.values()):
        return None
    manifest = _load_manifest(paths["event_manifest.json"])
    try:
        with np.load(paths["event_scan.npz"], allow_pickle=False) as data:
            audio = np.asarray(data["audio"], dtype=np.float32)
            video = np.asarray(data["video"], dtype=np.float32)
            av = np.asarray(data["av"], dtype=np.float32)
            scan_window = np.asarray(data["window_features"], dtype=np.float32)
            audio_rate = float(np.asarray(data["audio_rate_hz"]).item())
            video_fps = float(np.asarray(data["video_fps"]).item())
            video_fps_effective = float(np.asarray(data["video_fps_effective"]).item())
            duration_s = float(np.asarray(data["duration_s"]).item())
            stored_version = str(np.asarray(data["version"]).item())
        window = np.asarray(np.load(paths["event_features.npy"], allow_pickle=False), dtype=np.float32)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise EventScanError(f"事件 bundle 读取失败: {target}: {exc}") from exc

    version = str(manifest.get("version") or stored_version)
    if version != EVENT_VERSION:
        LOGGER.warning(
            "事件 bundle 版本 %s 与当前 %s 不一致，忽略: %s",
            version,
            EVENT_VERSION,
            target,
        )
        return None
    try:
        names = manifest.get("feature_names") or {}
        window_names = names.get("window") if isinstance(names, dict) else None
        if list(window_names or []) != list(WINDOW_FEATURE_NAMES):
            raise EventScanError(
                f"事件窗口特征顺序与当前 EVENT_VERSION 不兼容: {target}"
            )
        audio = _float32_matrix(audio, name="audio", columns=len(AUDIO_FEATURE_NAMES))
        video = _float32_matrix(video, name="video", columns=len(VIDEO_FEATURE_NAMES))
        av = _float32_matrix(av, name="av", columns=len(AV_FEATURE_NAMES))
        window = _float32_matrix(window, name="window_features", columns=len(WINDOW_FEATURE_NAMES))
    except EventScanError as exc:
        raise EventScanError(f"事件 bundle 数据结构非法: {target}: {exc}") from exc
    if window.shape != scan_window.shape or not np.array_equal(window, scan_window):
        raise EventScanError(f"event_features.npy 与 event_scan.npz 中的窗口特征不一致: {target}")
    if av.shape[0] != audio.shape[0]:
        raise EventScanError(f"音画序列行数与音频序列不一致: {target}")
    if not (
        math.isfinite(audio_rate)
        and math.isfinite(video_fps)
        and math.isfinite(video_fps_effective)
        and math.isfinite(duration_s)
        and audio_rate > 0
        and video_fps > 0
        and video_fps_effective > 0
        and duration_s > 0
    ):
        raise EventScanError(f"事件 bundle 的速率/时长非法: {target}")

    return EventFeatures(
        audio=audio,
        video=video,
        av=av,
        window_features=window,
        audio_rate_hz=audio_rate,
        video_fps=video_fps,
        video_fps_effective=video_fps_effective,
        duration_s=duration_s,
        version=version,
    )
