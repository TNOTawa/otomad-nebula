"""A6 音画错位实验：PE-AV 联合向量对“音画同步关系”的敏感度。

要回答的问题：联合向量到底对音画同步关系敏感，还是只对音画各自内容敏感？

做法：对每个生产片段（默认 8s）构造只改一个因素的变体：

- reference：原始音画（作为每个片段的参照，Δ 恒为 0）；
- quality：同一视频用更差编码参数（默认 crf 28）再编码、音频不变，
  用来测量“普通再编码 / 轻微画质退化”造成的 Δ 基线；
- shift_0.5s / shift_1.0s：画面不变，音频在片段内循环移位（np.roll）；
- swap：画面不变，音频换成同作品其他片段的音频（单片段作品跳过并记录）。

用 PE-AV 编码后计算 Δ = 1 - cos(向量)（joint / audio / video 三个模态）。
若错位变体的 Δ(joint) 显著大于 quality 基线，说明联合向量确实编码了同步关系；
同时可以看机制：错位只改音频时 video 模态应几乎不变，而 audio 模态变化明显。

注意（报告必须保留的注意点）：周期性强的音乐（固定 BPM）在 0.5s / 1.0s
接近整拍的偏移下，移位后的音频内容仍可能与原音频相似，导致 Δ 偏小；
单点 Δ 不宜直接解读为“模型对同步不敏感”，应结合分布与 swap（不同内容）对照。

边界：本模块只读 input/local 的原始视频与只读打开 data/catalog.sqlite；
产物只写 logs/ 与 temp/。纯逻辑部分不加载模型，默认测试可用
HashTestEmbedder 注入验证；真实模型运行由 tools/desync_experiment.py 完成。
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .config import MediaConfig
from .embed import Embedder
from .errors import NebulaError
from .media import MediaInfo, extract_clip
from .sampling import ClipPlan
from .util import now_utc_iso

logger = logging.getLogger(__name__)

VARIANT_REFERENCE = "reference"
"""原始音画参照变体。"""

VARIANT_QUALITY = "quality"
"""普通再编码基线变体（视频 crf 更大，音频不变）。"""

VARIANT_SWAP = "swap"
"""同作品其他片段音频交换变体。"""

DEFAULT_SHIFTS_S: tuple[float, ...] = (0.5, 1.0)
"""默认的音频循环移位量（秒）。"""

QUALITY_DEFAULT_CRF = 28
"""quality 基线变体的 libx264 CRF（比生产默认更差）。"""

WAV_SAMPLE_RATE = 48000
"""实验统一音频采样率（与 media.extract_clip 一致）。"""

WAV_CHANNELS = 2
"""实验统一音频声道数。"""

WAV_SAMPLE_WIDTH = 2
"""PCM 采样字节数（16-bit）。"""

MODALITIES: tuple[str, ...] = ("joint", "audio", "video")
"""报告的三个向量模态。"""

_EPS = 1e-12

CSV_COLUMNS: tuple[str, ...] = (
    "work_id",
    "title",
    "clip_index",
    "start_s",
    "duration_s",
    "variant",
    "joint_delta",
    "audio_delta",
    "video_delta",
)
"""CSV 明细列（每个片段 × 每个变体一行）。"""

PERIODICITY_CAVEAT = (
    "周期性音乐注意点：固定 BPM / 节拍强烈的音乐在 0.5s / 1.0s 接近整拍的偏移下，"
    "循环移位后的音频与原始音频仍可能高度相似，Δ 会偏小；单点 Δ 不应直接解读为"
    "“模型对同步不敏感”，请结合聚合分布与 swap（不同音频内容）对照。"
)

REPORT_NOTES: tuple[str, ...] = (
    "Δ = 1 - cos(向量)；reference 的 Δ 恒为 0，仅用于校验流程。",
    "quality 变体对同一画面用更大 CRF 再编码、音频完全不变，作为普通编码噪声基线。",
    "shift 变体画面文件与 reference 完全相同，只对 48k/2ch PCM 音频做 np.roll 循环移位。",
    "swap 变体画面不变、音频换成同作品下一片段音频；只有 1 个片段的作品跳过 swap。",
    PERIODICITY_CAVEAT,
    "只读 input/local 与 data/catalog.sqlite，结果只写 logs/ 与 temp/。",
)


# ──────────────────────────────────────────────────────────────────────────────
# 变体命名与顺序
# ──────────────────────────────────────────────────────────────────────────────


def shift_variant_name(shift_s: float) -> str:
    """把移位量转成稳定的变体名，例如 0.5 -> shift_0.5s、1.0 -> shift_1.0s。"""

    value = _validate_shift(shift_s)
    return f"shift_{value:.1f}s"


def _as_shift_float(shift_s: float) -> float:
    """把移位量转成有限 float（允许 0，0 表示不变）。"""

    if isinstance(shift_s, bool) or not isinstance(shift_s, (int, float)):
        raise NebulaError(f"音频移位量非法: {shift_s!r}")
    value = float(shift_s)
    if not math.isfinite(value):
        raise NebulaError(f"音频移位量必须是有限数，实际 {shift_s!r}")
    return value


def _validate_shift(shift_s: float) -> float:
    """校验变体移位量为有限非零数（0 没有构造变体的意义）。"""

    value = _as_shift_float(shift_s)
    if value == 0:
        raise NebulaError(f"音频移位量必须是非零有限数，实际 {shift_s!r}")
    return value


def _variant_order(names: Iterable[str]) -> list[str]:
    """按 reference -> quality -> shift（数值升序）-> swap -> 其他 的稳定顺序排序。"""

    unique = set(names)
    ordered: list[str] = []
    if VARIANT_REFERENCE in unique:
        ordered.append(VARIANT_REFERENCE)
    if VARIANT_QUALITY in unique:
        ordered.append(VARIANT_QUALITY)
    shifts: list[tuple[float, str]] = []
    for name in unique - set(ordered) - {VARIANT_SWAP}:
        if name.startswith("shift_") and name.endswith("s"):
            try:
                shifts.append((float(name[len("shift_") : -1]), name))
            except ValueError:
                continue
    ordered.extend(name for _value, name in sorted(shifts))
    if VARIANT_SWAP in unique:
        ordered.append(VARIANT_SWAP)
    ordered.extend(sorted(unique - set(ordered)))
    return ordered


# ──────────────────────────────────────────────────────────────────────────────
# 音频：wav 读写 / 循环移位 / 交换
# ──────────────────────────────────────────────────────────────────────────────


def read_wav_pcm(path: str | Path) -> tuple[np.ndarray, int, int]:
    """读取 16-bit PCM wav，返回 (int16 矩阵 (n_samples, channels), 采样率, 声道数)。

    异常
    ----
    NebulaError:
        文件不存在、不是 wav、不是 16-bit PCM 或数据长度不合法时抛出。
    """

    source = Path(path)
    if not source.is_file():
        raise NebulaError(f"wav 文件不存在: {source}")
    try:
        with wave.open(str(source), "rb") as handle:
            channels = int(handle.getnchannels())
            sample_rate = int(handle.getframerate())
            sample_width = int(handle.getsampwidth())
            frames = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError, OSError) as exc:
        raise NebulaError(f"读取 wav 失败（{source}）：{exc}") from exc
    if sample_width != WAV_SAMPLE_WIDTH:
        raise NebulaError(
            f"只支持 16-bit PCM wav（{source} 的 sample width={sample_width}）"
        )
    if channels < 1:
        raise NebulaError(f"wav 声道数非法（{source} channels={channels}）")
    data = np.frombuffer(frames, dtype="<i2")
    if data.size % channels != 0:
        raise NebulaError(f"wav 数据长度与声道数不匹配（{source}）")
    return data.reshape(-1, channels).copy(), sample_rate, channels


def write_wav_pcm(
    path: str | Path, samples: np.ndarray, sample_rate: int = WAV_SAMPLE_RATE
) -> Path:
    """原子写入 16-bit PCM wav（先写同目录临时文件再 os.replace）。"""

    array = np.asarray(samples)
    if array.dtype != np.int16:
        raise NebulaError(f"wav 采样必须是 int16，实际 dtype={array.dtype}")
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 1:
        raise NebulaError(f"wav 采样矩阵形状非法: {array.shape}")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate < 1:
        raise NebulaError(f"采样率必须为正整数，实际 {sample_rate!r}")
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise NebulaError(f"无法创建 wav 输出目录 {target.parent}：{exc}") from exc
    tmp_path: Path | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp.wav", dir=str(target.parent)
        )
        tmp_path = Path(tmp_name)
        with os.fdopen(fd, "wb") as raw, wave.open(raw, "wb") as handle:
            handle.setnchannels(int(array.shape[1]))
            handle.setsampwidth(WAV_SAMPLE_WIDTH)
            handle.setframerate(int(sample_rate))
            handle.writeframes(np.ascontiguousarray(array, dtype="<i2").tobytes())
        os.replace(tmp_path, target)
    except (OSError, wave.Error) as exc:
        raise NebulaError(f"写入 wav 失败（{target}）：{exc}") from exc
    finally:
        if tmp_path is not None:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:  # pragma: no cover - 清理失败不改变主结果
                logger.warning("无法删除 wav 临时文件: %s", tmp_path)
    return target


def roll_audio_samples(
    samples: np.ndarray, shift_s: float, *, sample_rate: int = WAV_SAMPLE_RATE
) -> np.ndarray:
    """在片段内部循环移位音频（np.roll），返回等长 int16 矩阵（不修改输入）。"""

    array = np.asarray(samples)
    if array.ndim != 2 or array.dtype != np.int16:
        raise NebulaError(
            f"音频采样必须是二维 int16 矩阵，实际 dtype={array.dtype} shape={array.shape}"
        )
    if array.shape[0] == 0:
        raise NebulaError("音频采样为空，无法移位")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate < 1:
        raise NebulaError(f"采样率必须为正整数，实际 {sample_rate!r}")
    value = _as_shift_float(shift_s)
    shift_frames = int(round(value * sample_rate))
    shift_frames %= array.shape[0]
    if shift_frames == 0:
        return array.copy()
    return np.roll(array, shift=shift_frames, axis=0).copy()


def _require_extracted_wav_format(path: Path, *, rate: int, channels: int, what: str) -> None:
    """校验 wav 是实验要求的 48k / 双声道格式。"""

    if rate != WAV_SAMPLE_RATE or channels != WAV_CHANNELS:
        raise NebulaError(
            f"{what}必须是 {WAV_SAMPLE_RATE}Hz / {WAV_CHANNELS}ch PCM wav，"
            f"实际 {rate}Hz / {channels}ch（{path}）"
        )


def make_shifted_wav(
    src_wav: str | Path,
    dst_wav: str | Path,
    shift_s: float,
    *,
    sample_rate: int = WAV_SAMPLE_RATE,
) -> Path:
    """把片段音频循环移位 shift_s 秒并写成新的 48k/2ch PCM wav。"""

    source = Path(src_wav)
    target = Path(dst_wav)
    if source.resolve() == target.resolve():
        raise NebulaError("移位 wav 的输出路径不能与输入相同")
    samples, rate, channels = read_wav_pcm(source)
    _require_extracted_wav_format(source, rate=rate, channels=channels, what="移位输入")
    shifted = roll_audio_samples(samples, shift_s, sample_rate=rate)
    return write_wav_pcm(target, shifted, rate)


def make_swap_wav(src_wav: str | Path, dst_wav: str | Path) -> Path:
    """把其他片段的音频复制成新的 48k/2ch PCM wav（swap 变体用）。"""

    source = Path(src_wav)
    target = Path(dst_wav)
    if source.resolve() == target.resolve():
        raise NebulaError("交换 wav 的输出路径不能与输入相同")
    samples, rate, channels = read_wav_pcm(source)
    _require_extracted_wav_format(source, rate=rate, channels=channels, what="交换输入")
    return write_wav_pcm(target, samples, rate)


# ──────────────────────────────────────────────────────────────────────────────
# 视频：普通再编码基线
# ──────────────────────────────────────────────────────────────────────────────


def _run_ffmpeg(command: list[str], *, what: str, timeout_s: float) -> None:
    """执行 ffmpeg 命令；失败时抛带 stderr 摘要的 NebulaError。"""

    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        raise NebulaError(f"找不到 ffmpeg 可执行文件: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise NebulaError(f"{what}超时（>{timeout_s:g}s）") from exc
    except OSError as exc:
        raise NebulaError(f"{what}无法启动 ffmpeg: {exc}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-1000:]
        raise NebulaError(f"{what}失败（退出码 {proc.returncode}）：{tail}")


def make_quality_video(
    source_video: str | Path,
    dst_video: str | Path,
    *,
    media: MediaConfig,
    crf: int = QUALITY_DEFAULT_CRF,
) -> Path:
    """对视频片段用更差 CRF 再编码（画面轻微退化，音频不变），作为 Δ 基线。

    只重编码画面：reference 的 wav 直接复用，因此 quality 变体与 reference
    的唯一差异是画面压缩质量。
    """

    source = Path(source_video)
    target = Path(dst_video)
    if not source.is_file():
        raise NebulaError(f"画质基线输入视频不存在: {source}")
    if isinstance(crf, bool) or not isinstance(crf, int) or not 0 <= crf <= 51:
        raise NebulaError(f"quality 变体 CRF 必须在 0..51，实际 {crf!r}")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise NebulaError(f"无法创建画质基线输出目录 {target.parent}：{exc}") from exc
    command = [
        str(media.ffmpeg),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(source),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        str(media.video_preset),
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target),
    ]
    _run_ffmpeg(command, what="画质基线再编码", timeout_s=media.extract_timeout_seconds)
    if not target.is_file() or target.stat().st_size <= 0:
        raise NebulaError(f"画质基线再编码未生成有效文件: {target}")
    return target


# ──────────────────────────────────────────────────────────────────────────────
# 变体构建
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClipVariants:
    """一个片段的所有变体（pairs[variant] = (视频文件, 音频文件)）。"""

    clip_index: int
    start_s: float
    duration_s: float
    pairs: dict[str, tuple[Path, Path]]

    @property
    def reference(self) -> tuple[Path, Path]:
        return self.pairs[VARIANT_REFERENCE]


@dataclass(frozen=True)
class VariantBuild:
    """一部作品全部片段的变体构建结果。"""

    work_id: int
    clips: tuple[ClipVariants, ...]
    skipped: tuple[str, ...]


@dataclass(frozen=True)
class VariantDeltas:
    """一个片段相对 reference 的各变体 Δ（deltas[variant][modality]）。"""

    work_id: int
    clip_index: int
    start_s: float
    duration_s: float
    deltas: dict[str, dict[str, float | None]]


def build_work_variants(
    source: str | Path,
    info: MediaInfo,
    plans: Sequence[ClipPlan],
    *,
    work_id: int,
    out_dir: str | Path,
    media: MediaConfig,
    shifts_s: Sequence[float] = DEFAULT_SHIFTS_S,
    quality_crf: int = QUALITY_DEFAULT_CRF,
    allow_swap: bool = True,
) -> VariantBuild:
    """截取一部作品的片段并生成 reference / quality / shift / swap 变体。

    参数
    ----
    source:
        原视频路径。
    info:
        已探测的 MediaInfo（传给 extract_clip，跳过重复 ffprobe）。
    plans:
        采样计划（来自 nebula.sampling.plan_clips）。
    work_id:
        作品 id，仅用于错误信息与结果标记。
    out_dir:
        变体输出目录（建议放在 temp/ 下）。
    media:
        MediaConfig（ffmpeg 路径、编码参数、超时等）。
    shifts_s:
        音频循环移位量列表（秒），默认 0.5 / 1.0。
    quality_crf:
        quality 基线 CRF，默认 28。
    allow_swap:
        是否生成 swap 变体（作品只有 1 个片段时无论如何都会跳过并记录）。

    返回
    ----
    VariantBuild:
        clips 内每个 ClipVariants 都包含 reference 与至少一个其他变体；
        skipped 记录被跳过的变体及中文原因。
    """

    src = Path(source)
    if not src.is_file():
        raise NebulaError(f"源视频不存在: {src}")
    if not isinstance(info, MediaInfo):
        raise NebulaError(f"info 参数类型非法: {type(info).__name__}（需要 MediaInfo）")
    if not info.has_video:
        raise NebulaError(f"源视频没有视频轨，无法做音画错位实验: {src}")
    if isinstance(work_id, bool) or not isinstance(work_id, int) or work_id < 1:
        raise NebulaError(f"work_id 必须是正整数，实际 {work_id!r}")
    plan_list = list(plans)
    if not plan_list:
        raise NebulaError("采样计划为空，无法构建变体")
    shifts = tuple(_validate_shift(value) for value in shifts_s)
    if len(set(shifts)) != len(shifts):
        raise NebulaError(f"音频移位量存在重复: {shifts!r}")

    root = Path(out_dir)
    reference_dir = root / "reference"
    extracted = [
        extract_clip(src, plan, reference_dir, media=media, info=info) for plan in plan_list
    ]

    skipped: list[str] = []
    if not allow_swap:
        skipped.append("swap: 按参数关闭")
    elif len(extracted) < 2:
        skipped.append("swap: 作品只有 1 个片段，跳过音频交换变体")

    clip_variants: list[ClipVariants] = []
    for position, clip in enumerate(extracted):
        pairs: dict[str, tuple[Path, Path]] = {
            VARIANT_REFERENCE: (clip.video_path, clip.audio_path)
        }
        quality_video = root / "quality" / f"clip_{clip.index:02d}.mp4"
        make_quality_video(
            clip.video_path, quality_video, media=media, crf=quality_crf
        )
        pairs[VARIANT_QUALITY] = (quality_video, clip.audio_path)
        for shift in shifts:
            name = shift_variant_name(shift)
            shifted_path = root / name / f"clip_{clip.index:02d}.wav"
            make_shifted_wav(clip.audio_path, shifted_path, shift)
            pairs[name] = (clip.video_path, shifted_path)
        if allow_swap and len(extracted) >= 2:
            other = extracted[(position + 1) % len(extracted)]
            swap_path = root / VARIANT_SWAP / f"clip_{clip.index:02d}.wav"
            make_swap_wav(other.audio_path, swap_path)
            pairs[VARIANT_SWAP] = (clip.video_path, swap_path)
        clip_variants.append(
            ClipVariants(
                clip_index=int(clip.index),
                start_s=float(clip.start_s),
                duration_s=float(clip.duration_s),
                pairs=pairs,
            )
        )
    return VariantBuild(
        work_id=int(work_id), clips=tuple(clip_variants), skipped=tuple(skipped)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Δ 计算与评估
# ──────────────────────────────────────────────────────────────────────────────


def _as_unit_vector(value: Any, *, what: str) -> np.ndarray:
    """把一维向量转成 float64 单位向量；零范数 / NaN 直接报错。"""

    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise NebulaError(f"{what} 不是合法数值向量: {exc}") from exc
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise NebulaError(f"{what} 必须是一维向量，实际形状 {array.shape}")
    if not np.all(np.isfinite(array)):
        raise NebulaError(f"{what} 含 NaN/Inf，无法计算 Δ")
    norm = float(np.linalg.norm(array))
    if norm <= _EPS:
        raise NebulaError(f"{what} 是零向量，无法计算 Δ")
    return array / norm


def compute_delta(reference: Any, variant: Any) -> float:
    """计算 Δ = 1 - cos(reference, variant)（输入无需预先归一化）。"""

    left = _as_unit_vector(reference, what="reference 向量")
    right = _as_unit_vector(variant, what="variant 向量")
    if left.shape != right.shape:
        raise NebulaError(
            f"reference 与 variant 维度不一致: {left.shape} != {right.shape}"
        )
    cosine = float(np.dot(left, right))
    return max(0.0, 1.0 - cosine)


def evaluate_variants(
    builds: Sequence[VariantBuild], *, embedder: Embedder
) -> list[VariantDeltas]:
    """用给定的 Embedder 编码全部变体，计算每个片段各变体的 Δ。

    实现上把所有 (视频, 音频) 对按“作品 -> 片段 -> 变体”顺序拼成一次
    encode_pairs 调用（内部仍按 embedder 的 batch_size 分批），因此同一个
    片段的 reference 与变体通常落在同一个 batch 内，避免分批带来的数值噪声。
    """

    if not builds:
        raise NebulaError("没有可评估的变体")
    flat_pairs: list[tuple[Path, Path]] = []
    plan: list[tuple[int, int, list[str]]] = []
    for build_position, build in enumerate(builds):
        if not build.clips:
            raise NebulaError(f"作品 {build.work_id} 没有片段，无法评估")
        for clip_position, clip in enumerate(build.clips):
            if VARIANT_REFERENCE not in clip.pairs:
                raise NebulaError(
                    f"作品 {build.work_id} 片段 {clip.clip_index} 缺少 reference 变体"
                )
            names = _variant_order(clip.pairs)
            plan.append((build_position, clip_position, names))
            for name in names:
                flat_pairs.append(clip.pairs[name])
    if not flat_pairs:
        raise NebulaError("没有可编码的变体对")

    batch = embedder.encode_pairs(flat_pairs)
    joint = np.asarray(batch.joint)
    if joint.ndim != 2 or joint.shape[0] != len(flat_pairs):
        raise NebulaError(
            f"嵌入器返回的 joint 形状异常: {joint.shape}（期望行数 {len(flat_pairs)}）"
        )
    audio = None if batch.audio is None else np.asarray(batch.audio)
    video = None if batch.video is None else np.asarray(batch.video)
    for name, matrix in (("audio", audio), ("video", video)):
        if matrix is not None and (matrix.ndim != 2 or matrix.shape[0] != len(flat_pairs)):
            raise NebulaError(
                f"嵌入器返回的 {name} 形状异常: {matrix.shape}（期望行数 {len(flat_pairs)}）"
            )

    results: list[list[VariantDeltas]] = [[] for _ in builds]
    row = 0
    for build_position, clip_position, names in plan:
        build = builds[build_position]
        clip = build.clips[clip_position]
        reference_row = row
        deltas: dict[str, dict[str, float | None]] = {}
        for offset, name in enumerate(names):
            current = row + offset
            entry: dict[str, float | None] = {}
            for modality, matrix in (("joint", joint), ("audio", audio), ("video", video)):
                if matrix is None:
                    entry[modality] = None
                else:
                    entry[modality] = compute_delta(matrix[reference_row], matrix[current])
            deltas[name] = entry
        results[build_position].append(
            VariantDeltas(
                work_id=int(build.work_id),
                clip_index=int(clip.clip_index),
                start_s=float(clip.start_s),
                duration_s=float(clip.duration_s),
                deltas=deltas,
            )
        )
        row += len(names)
    return [record for records in results for record in records]


# ──────────────────────────────────────────────────────────────────────────────
# 聚合统计与结论
# ──────────────────────────────────────────────────────────────────────────────


def _describe(values: Sequence[float]) -> dict[str, Any]:
    """返回一组 Δ 的 count/mean/median/min/max/std（空列表时数值为 None）。"""

    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
        }
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise NebulaError("Δ 统计输入含 NaN/Inf")
    return {
        "count": int(array.size),
        "mean": round(float(array.mean()), 6),
        "median": round(float(np.median(array)), 6),
        "min": round(float(array.min()), 6),
        "max": round(float(array.max()), 6),
        "std": round(float(array.std(ddof=0)), 6),
    }


def summarize_deltas(records: Sequence[VariantDeltas]) -> dict[str, dict[str, dict[str, Any]]]:
    """按变体 × 模态汇总 Δ 的 count/mean/median/min/max/std。"""

    if not records:
        raise NebulaError("没有 Δ 记录可汇总")
    all_names = {name for record in records for name in record.deltas}
    stats: dict[str, dict[str, dict[str, Any]]] = {}
    for name in _variant_order(all_names):
        stats[name] = {}
        for modality in MODALITIES:
            values = [
                float(record.deltas[name][modality])
                for record in records
                if name in record.deltas and record.deltas[name].get(modality) is not None
            ]
            stats[name][modality] = _describe(values)
    return stats


def _mean_of(stats: Mapping[str, Any], variant: str, modality: str) -> float | None:
    entry = stats.get(variant, {}).get(modality, {})
    value = entry.get("mean") if isinstance(entry, Mapping) else None
    return float(value) if value is not None else None


def build_conclusion(stats: Mapping[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """根据聚合 Δ 生成“同步敏感度”结论（含周期性注意点与机制解释）。"""

    joint_means: dict[str, float] = {}
    for name, modalities in stats.items():
        mean = modalities.get("joint", {}).get("mean")
        if mean is not None:
            joint_means[name] = float(mean)
    quality_mean = joint_means.get(VARIANT_QUALITY)
    sync_names = [
        name
        for name in joint_means
        if name not in (VARIANT_REFERENCE, VARIANT_QUALITY)
    ]
    ratios: dict[str, float | None] = {}
    for name in sync_names:
        if quality_mean is not None and quality_mean > _EPS:
            ratios[name] = round(joint_means[name] / quality_mean, 4)
        else:
            ratios[name] = None
    finite_ratios = [value for value in ratios.values() if value is not None]
    max_ratio = max(finite_ratios, default=0.0)

    if quality_mean is None:
        magnitude = "缺少 quality 基线，无法判断 Δ 的相对量级"
    elif quality_mean <= _EPS:
        magnitude = "普通再编码噪声几乎为 0，错位 Δ 全部显著大于基线"
    elif max_ratio >= 5.0:
        magnitude = f"错位 Δ 显著大于普通再编码噪声（最大 {max_ratio:.2f}x）"
    elif max_ratio >= 2.0:
        magnitude = f"错位 Δ 明显大于普通再编码噪声（最大 {max_ratio:.2f}x）"
    elif max_ratio >= 1.2:
        magnitude = f"错位 Δ 略大于普通再编码噪声（最大 {max_ratio:.2f}x）"
    else:
        magnitude = (
            "错位 Δ 与普通再编码噪声同量级（最大 "
            f"{max_ratio:.2f}x）；若样本量小或音乐节拍周期性强，"
            "可能无法区分“对同步不敏感”与“偏移后音频内容仍相似”"
        )

    audio_shift_means = [
        _mean_of(stats, name, "audio")
        for name in sync_names
        if _mean_of(stats, name, "audio") is not None
    ]
    video_shift_means = [
        _mean_of(stats, name, "video")
        for name in sync_names
        if _mean_of(stats, name, "video") is not None
    ]
    audio_mean = (
        round(float(np.mean(audio_shift_means)), 6) if audio_shift_means else None
    )
    video_mean = (
        round(float(np.mean(video_shift_means)), 6) if video_shift_means else None
    )
    if audio_mean is not None and video_mean is not None:
        mechanism = (
            f"机制检查：错位变体的 audio Δ 均值 {audio_mean:.6f}，"
            f"video Δ 均值 {video_mean:.6f}（画面未变，理论上应接近 0）。"
            "若 audio 明显变化而 video 接近 0、joint 随之变化，说明联合编码确实融合了"
            "音频/视频两路信息，而不是只依赖单一模态。"
        )
    else:
        mechanism = "机制检查：缺少 audio 或 video 单模态向量，无法做双模态对照。"

    interpretation = (
        f"Δ_quality(joint) 均值 = "
        f"{'N/A' if quality_mean is None else format(quality_mean, '.6f')}；"
        + "，".join(
            f"Δ_{name}(joint) 均值 = {joint_means[name]:.6f}"
            + (
                f"（{ratios[name]:.2f}x quality）"
                if ratios.get(name) is not None
                else ""
            )
            for name in sync_names
        )
        + f"。结论：{magnitude}。"
        + mechanism
    )
    return {
        "delta_joint_means": {
            name: round(value, 6) for name, value in joint_means.items()
        },
        "ratio_vs_quality": ratios,
        "max_ratio_vs_quality": round(float(max_ratio), 4),
        "verdict": magnitude,
        "mechanism": mechanism,
        "interpretation": interpretation,
        "periodicity_caveat": PERIODICITY_CAVEAT,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 报告生成与落盘
# ──────────────────────────────────────────────────────────────────────────────


def _round_delta(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def build_report(
    *,
    embedding_version: str,
    model_info: Mapping[str, Any],
    config_info: Mapping[str, Any],
    work_meta: Sequence[Mapping[str, Any]],
    results: Sequence[VariantDeltas],
    skipped: Sequence[Mapping[str, Any]] = (),
    extra_notes: Sequence[str] = (),
) -> dict[str, Any]:
    """把变体 Δ 记录组装成 JSON 可序列化的实验报告。

    work_meta 元素至少包含 work_id / title / duration_s / local_path；
    results 为 evaluate_variants 的输出；skipped 为跳过的工作或变体说明。
    """

    if not embedding_version:
        raise NebulaError("embedding_version 不能为空")
    if not isinstance(model_info, Mapping) or not isinstance(config_info, Mapping):
        raise NebulaError("model_info / config_info 必须是映射")
    records_by_work: dict[int, list[VariantDeltas]] = {}
    for record in results:
        records_by_work.setdefault(int(record.work_id), []).append(record)

    works: list[dict[str, Any]] = []
    total_clips = 0
    total_records = 0
    for meta in work_meta:
        work_id = int(meta["work_id"])
        records = sorted(records_by_work.get(work_id, []), key=lambda item: item.clip_index)
        clips: list[dict[str, Any]] = []
        for record in records:
            clips.append(
                {
                    "clip_index": int(record.clip_index),
                    "start_s": round(float(record.start_s), 3),
                    "duration_s": round(float(record.duration_s), 3),
                    "deltas": {
                        name: {
                            modality: _round_delta(values.get(modality))
                            for modality in MODALITIES
                        }
                        for name, values in record.deltas.items()
                    },
                }
            )
        work_means: dict[str, float | None] = {}
        for name in _variant_order({name for record in records for name in record.deltas}):
            values = [
                float(record.deltas[name]["joint"])
                for record in records
                if name in record.deltas and record.deltas[name].get("joint") is not None
            ]
            work_means[name] = (
                round(float(np.mean(values)), 6) if values else None
            )
        works.append(
            {
                "work_id": work_id,
                "title": meta.get("title"),
                "local_path": meta.get("local_path"),
                "duration_s": (
                    round(float(meta["duration_s"]), 3)
                    if meta.get("duration_s") is not None
                    else None
                ),
                "n_clips": len(clips),
                "mean_joint_delta": work_means,
                "clips": clips,
            }
        )
        total_clips += len(clips)
        total_records += len(clips)

    stats = summarize_deltas(results) if results else {}
    conclusion = build_conclusion(stats) if stats else {}
    report: dict[str, Any] = {
        "experiment": "av_desync",
        "generated_at": now_utc_iso(),
        "embedding_version": str(embedding_version),
        "model": dict(model_info),
        "config": dict(config_info),
        "n_works": len(works),
        "n_clips": total_clips,
        "n_records": total_records,
        "stats": stats,
        "conclusion": conclusion,
        "works": works,
        "skipped": [dict(item) for item in skipped],
        "notes": list(REPORT_NOTES) + list(extra_notes),
    }
    return report


def _atomic_write_text(path: Path, text: str) -> None:
    """以“同目录临时文件 + os.replace”原子写入文本。"""

    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise NebulaError(f"无法创建报告目录 {target.parent}：{exc}") from exc
    tmp_path: Path | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        tmp_path = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(tmp_path, target)
    except OSError as exc:
        raise NebulaError(f"写入报告失败（{target}）：{exc}") from exc
    finally:
        if tmp_path is not None:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:  # pragma: no cover - 清理失败不改变主结果
                logger.warning("无法删除报告临时文件: %s", tmp_path)


def _report_csv(report: Mapping[str, Any]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    for work in report.get("works", []):
        for clip in work.get("clips", []):
            deltas = clip.get("deltas", {})
            for variant in _variant_order(deltas):
                values = deltas.get(variant, {})
                writer.writerow(
                    {
                        "work_id": work.get("work_id"),
                        "title": work.get("title"),
                        "clip_index": clip.get("clip_index"),
                        "start_s": clip.get("start_s"),
                        "duration_s": clip.get("duration_s"),
                        "variant": variant,
                        "joint_delta": values.get("joint"),
                        "audio_delta": values.get("audio"),
                        "video_delta": values.get("video"),
                    }
                )
    return buffer.getvalue()


def write_report(
    report: Mapping[str, Any], *, json_path: str | Path, csv_path: str | Path
) -> None:
    """把实验报告写成 JSON（完整明细）与 CSV（每个片段×变体一行）。"""

    if not isinstance(report, Mapping):
        raise NebulaError("音画错位实验报告必须是 dict")
    json_target = Path(json_path)
    csv_target = Path(csv_path)
    json_text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    csv_text = _report_csv(report)
    _atomic_write_text(json_target, json_text)
    _atomic_write_text(csv_target, csv_text)


def format_summary(report: Mapping[str, Any]) -> str:
    """把报告渲染成中文控制台摘要（多行字符串）。"""

    model = report.get("model") or {}
    lines = [
        "音画错位实验（Δ = 1 - cos；越小越相似）",
        f"  模型: {model.get('checkpoint')}（embedder={model.get('embedder')}，"
        f"version={report.get('embedding_version')}）",
        f"  设备: {model.get('device')}"
        f"{'，bf16 autocast' if model.get('autocast_bfloat16') else '，float32'}"
        f"；模型加载 {model.get('model_load_s')}s",
        f"  作品/片段/记录: {report.get('n_works')} 部 / "
        f"{report.get('n_clips')} 段 / {report.get('n_records')} 条",
        "  变体                joint(mean/median/min/max)              "
        "audio(mean)   video(mean)",
    ]
    stats = report.get("stats") or {}
    for name in _variant_order(stats):
        joint = stats[name].get("joint", {})
        audio = stats[name].get("audio", {})
        video = stats[name].get("video", {})
        lines.append(
            f"  {name:<18} "
            f"{_fmt(joint.get('mean'))}/{_fmt(joint.get('median'))}/"
            f"{_fmt(joint.get('min'))}/{_fmt(joint.get('max'))}    "
            f"{_fmt(audio.get('mean'))}        {_fmt(video.get('mean'))}"
        )
    conclusion = report.get("conclusion") or {}
    lines.append(f"  结论: {conclusion.get('verdict')}")
    interpretation = conclusion.get("interpretation")
    if interpretation:
        lines.append(f"  解读: {interpretation}")
    lines.append("  注意: 周期性音乐在整拍偏移下仍可能相似，单点 Δ 不宜过度解读。")
    skipped = report.get("skipped") or []
    if skipped:
        lines.append(f"  跳过记录: {len(skipped)} 条")
        for item in skipped[:5]:
            lines.append(
                f"    work_id={item.get('work_id')}: {item.get('reason')}"
            )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    """把统计值格式化为 6 位小数或 N/A。"""

    if value is None:
        return "   N/A  "
    return f"{float(value):.6f}"
