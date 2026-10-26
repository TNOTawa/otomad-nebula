"""v0.5-beta 密集音频编码：冻结 MERT 帧特征 -> 0.5s 全局 bin。

事实核查（2026-10-26，transformers 5.20.2）：
- transformers 顶层没有 MERTModel；m-a-p/MERT-v1-95M 仓库自带
  configuration_MERT.py / modeling_MERT.py（custom code），必须
  trust_remote_code=True 才能加载。
- 该仓库在 5.20.2 下可正常 AutoModel.from_pretrained + GPU 前向（MERTModel
  继承 HubertModel，94.37M 参数，24kHz 输入，conv stride 320 -> 75Hz 帧）。
- 因此本模块唯一使用 MERT-v1-95M，不引入 wav2vec2/CLAP 备选，避免多模型混用。
  局限（写进 manifest 与 pilot 报告）：MERT 是音乐预训练模型，本编码器只用其
  冻结帧特征、不做语义标签；层选择（默认第 9 层）未调优。

产物：data/audio_dense/<audio_version>/<work_id>/audio.npz + manifest.json。
audio.npz 数组：frame_features (N,D) float16、frame_times (N,) float32、
bin_features (n_bins,D) float32（均值）、bin_std (n_bins,D) float32、
bin_times (n_bins,2) float32、bin_mask (n_bins,) bool、duration_s 标量。

版本号包含 model/revision/sample_rate/context/hop/layer/pool/bin/normalize；
参数变化生成新版本目录，不覆盖旧版本。写入原子（临时文件+os.replace），支持
断点复用。只读本地源文件，不下载网络视频、不写 DB、不删除源文件。
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import math
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence, runtime_checkable

import numpy as np

from .errors import NebulaError
from .media import MediaError, probe_media
from .util import now_utc_iso, slugify

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "m-a-p/MERT-v1-95M"
"""默认音频编码器（MERT-v1-95M，必须 trust_remote_code）。"""

DEFAULT_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
"""MERT-v1-95M 固定 revision（2026-10-26 核对）。"""

DEFAULT_SAMPLE_RATE = 24000
DEFAULT_CONTEXT_S = 5.0
DEFAULT_HOP_S = 5.0
DEFAULT_BIN_SECONDS = 0.5
DEFAULT_LAYER = 9
DEFAULT_POOL = "mean_std"
DEFAULT_NORMALIZE = "l2"
DEFAULT_BATCH_SIZE = 8

AUDIO_NPZ_NAME = "audio.npz"
AUDIO_MANIFEST_NAME = "manifest.json"
NORMALIZE_MODES = ("none", "l2")
POOL_MODES = ("mean", "mean_std")


class AudioDenseError(NebulaError):
    """密集音频解码/编码/存取失败。"""


def _positive_number(value: Any, *, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AudioDenseError(f"{what} 必须是数字，收到 {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise AudioDenseError(f"{what} 必须大于 0，收到 {value!r}")
    return number


def _positive_int(value: Any, *, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise AudioDenseError(f"{what} 必须是整数，收到 {value!r}")
    number = int(value)
    if number < 1:
        raise AudioDenseError(f"{what} 必须 >= 1，收到 {value!r}")
    return number


def _validate_layer(layer: int | None) -> int | None:
    return None if layer is None else _positive_int(layer, what="layer")


def _validate_pool(pool: str) -> str:
    if pool not in POOL_MODES:
        raise AudioDenseError(f"pool 必须是 {POOL_MODES}，收到 {pool!r}")
    return pool


def _validate_normalize(normalize: str) -> str:
    if normalize not in NORMALIZE_MODES:
        raise AudioDenseError(f"normalize 必须是 {NORMALIZE_MODES}，收到 {normalize!r}")
    return normalize


def compose_audio_version(
    encoder_version: str,
    *,
    sample_rate: int,
    context_s: float,
    hop_s: float,
    layer: int | None,
    pool: str,
    bin_seconds: float,
    normalize: str,
) -> str:
    """组合 audio dataset 版本号（model/revision/sr/ctx/hop/layer/pool/bin/norm）。"""

    if not isinstance(encoder_version, str) or not encoder_version.strip():
        raise AudioDenseError(f"encoder_version 不能为空，收到 {encoder_version!r}")
    rate = _positive_int(sample_rate, what="sample_rate")
    context = _positive_number(context_s, what="context_s")
    hop = _positive_number(hop_s, what="hop_s")
    if hop > context + 1e-9:
        raise AudioDenseError(f"hop_s({hop:g}) 不能大于 context_s({context:g})")
    pool_value = _validate_pool(pool)
    bin_value = _positive_number(bin_seconds, what="bin_seconds")
    norm_value = _validate_normalize(normalize)
    layer_value = _validate_layer(layer)
    layer_tag = "none" if layer_value is None else str(layer_value)
    raw = (
        f"{encoder_version.strip()}__sr{rate}_ctx{context:g}_hop{hop:g}"
        f"_l{layer_tag}_pool{pool_value}_bin{bin_value:g}_norm{norm_value}"
    )
    return slugify(raw)


def _resolve_hub_cache_dir(cache_dir: str | Path | None) -> Path | None:
    """把 HF_HOME 布局的缓存根解析成 from_pretrained(cache_dir=...) 目录。"""

    if cache_dir is None:
        return None
    root = Path(cache_dir)
    hub = root / "hub"
    if hub.is_dir():
        return hub
    return root


def _which(name: str) -> str | None:
    import shutil

    return shutil.which(name)


def _resolve_ffprobe(ffmpeg: str) -> str:
    resolved = _which(str(ffmpeg))
    if resolved is not None:
        binary = Path(resolved)
        for candidate in (
            binary.with_name("ffprobe" + binary.suffix),
            binary.with_name("ffprobe"),
        ):
            if candidate.is_file():
                return str(candidate)
    found = _which("ffprobe")
    return found if found is not None else "ffprobe"


def _stderr_tail(stderr: bytes | str | None, limit: int = 800) -> str:
    if not stderr:
        return ""
    text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else str(stderr)
    text = text.strip()
    return text[-limit:] if len(text) > limit else text


def probe_audio_duration(
    source: str | Path, *, ffmpeg: str = "ffmpeg", timeout_s: float = 120.0
) -> float:
    """用项目 media.probe_media 读取时长，并要求有音频流。"""

    src = Path(source)
    if not src.is_file():
        raise AudioDenseError(f"源文件不存在: {src}")
    ffprobe = _resolve_ffprobe(ffmpeg)
    try:
        info = probe_media(src, ffprobe=ffprobe, timeout_s=float(timeout_s))
    except MediaError as exc:
        raise AudioDenseError(f"无法 probe 源文件（{src}）：{exc}") from exc
    if not info.has_audio:
        raise AudioDenseError(f"源文件没有音频流: {src}")
    duration = float(info.duration_s)
    if not math.isfinite(duration) or duration <= 0:
        raise AudioDenseError(f"源文件时长非法: {duration!r}（{src}）")
    return duration


def iter_audio_windows(
    source: str | Path,
    *,
    sample_rate: int,
    context_s: float,
    hop_s: float,
    ffmpeg: str = "ffmpeg",
    timeout_s: float = 600.0,
    max_duration_s: float | None = None,
    stats: dict[str, float] | None = None,
) -> Iterator[tuple[float, np.ndarray, int]]:
    """流式解码音轨，逐窗产出 (start_s, float32 context 波形, 有效采样数)。

    窗口长度 = context_s*sample_rate，按 hop_s 前移；最后一窗 0 补齐并返回实际
    有效采样数。max_duration_s 给定时最多解码该时长（用于把音频裁到 ffprobe 容器
    时长，避免个别 Bilibili 源音频流比视频长 1-2s 造成 bin 时间轴漂移）。
    stats 若提供会写入 decoded_samples / decoded_duration_s。
    """

    src = Path(source)
    if not src.is_file():
        raise AudioDenseError(f"源文件不存在: {src}")
    rate = _positive_int(sample_rate, what="sample_rate")
    context = _positive_number(context_s, what="context_s")
    hop = _positive_number(hop_s, what="hop_s")
    if hop > context + 1e-9:
        raise AudioDenseError(f"hop_s({hop:g}) 不能大于 context_s({context:g})")
    timeout = _positive_number(timeout_s, what="timeout_s")
    window_samples = max(1, int(round(context * rate)))
    hop_samples = max(1, int(round(hop * rate)))
    cap_samples: int | None = None
    if max_duration_s is not None:
        cap_value = _positive_number(max_duration_s, what="max_duration_s")
        cap_samples = max(1, int(math.ceil(cap_value * rate)))
    command = [
        str(ffmpeg),
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(src),
        "-vn",
        "-sn",
        "-ac",
        "1",
        "-ar",
        str(rate),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except FileNotFoundError as exc:
        raise AudioDenseError(f"找不到 ffmpeg 可执行文件（{ffmpeg}）") from exc
    except OSError as exc:
        raise AudioDenseError(f"无法启动 ffmpeg 音频解码：{exc}") from exc
    assert process.stdout is not None and process.stderr is not None
    buffer = np.empty(0, dtype=np.float32)
    pending = b""
    total_samples = 0
    window_index = 0
    eof = False
    returncode = -1
    stderr = b""
    deadline = time.monotonic() + timeout

    def _append(raw: bytes) -> None:
        nonlocal buffer, pending
        pending += raw
        usable = (len(pending) // 4) * 4
        if usable == 0:
            return
        chunk = np.frombuffer(pending[:usable], dtype="<f4").astype(np.float32)
        pending = pending[usable:]
        buffer = np.concatenate([buffer, chunk]) if buffer.size else chunk

    try:
        while True:
            if time.monotonic() > deadline:
                process.kill()
                raise AudioDenseError(f"音频解码超时（>{timeout:g}s）：{src}")
            if not eof and buffer.size < window_samples:
                if cap_samples is not None and total_samples >= cap_samples:
                    eof = True
                    continue
                read_samples = hop_samples
                if cap_samples is not None:
                    read_samples = min(read_samples, cap_samples - total_samples)
                raw = process.stdout.read(read_samples * 4)
                if not raw:
                    eof = True
                else:
                    if cap_samples is not None:
                        allowed = cap_samples - total_samples
                        if len(raw) // 4 > allowed:
                            raw = raw[: allowed * 4]
                    total_samples += len(raw) // 4
                    _append(raw)
                continue
            if buffer.size >= window_samples:
                start_s = window_index * hop
                window = buffer[:window_samples].copy()
                window_index += 1
                buffer = buffer[hop_samples:]
                yield start_s, window, window_samples
                continue
            break
        while buffer.size > 0:
            actual = int(buffer.size)
            start_s = window_index * hop
            window = np.zeros(window_samples, dtype=np.float32)
            window[:actual] = buffer[:actual]
            window_index += 1
            buffer = np.empty(0, dtype=np.float32)
            yield start_s, window, actual
        # 达到 max_duration_s 后不再读取 stdout；必须把剩余管道排空，
        # 否则 ffmpeg 会因管道写满阻塞，导致 wait/stderr.read 死锁。
        if cap_samples is not None and process.poll() is None:
            try:
                while process.stdout.read(1 << 20):
                    pass
            except OSError:  # pragma: no cover - 管道已关闭
                pass
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
        raise AudioDenseError(f"ffmpeg 音频解码失败（退出码 {returncode}）：{src}{detail}")
    if total_samples <= 0:
        raise AudioDenseError(f"没有解码出任何音频采样: {src}")
    if stats is not None:
        stats["decoded_samples"] = float(total_samples)
        stats["decoded_duration_s"] = float(total_samples) / float(rate)


# ──────────────────────────────────────────────────────────────────────────────
# 编码器协议 / MERT 冻结编码器
# ──────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class AudioFrameEncoder(Protocol):
    """音频帧编码器统一接口（测试可注入 fake 实现）。"""

    version: str
    sample_rate: int
    frame_rate: float
    dim: int
    model_load_s: float
    frame_stride_samples: int
    frame_receptive_samples: int

    def encode(
        self, waveforms: np.ndarray, valid_samples: Sequence[int]
    ) -> list[np.ndarray]:
        """(B, n_samples) float32 -> 每个样本的 (T_i, D) float32 帧特征。"""
        ...

    def close(self) -> None:
        ...


def _conv_output_stats(config: Any) -> tuple[int, int]:
    """返回 (conv stride 总步长, 感受野采样数)。"""

    strides = list(getattr(config, "conv_stride", None) or [5, 2, 2, 2, 2, 2, 2])
    kernels = list(getattr(config, "conv_kernel", None) or [10, 3, 3, 3, 3, 2, 2])
    if len(strides) != len(kernels) or not strides:
        raise AudioDenseError("模型 conv_stride/conv_kernel 配置非法")
    stride_acc = 1
    receptive = 1
    for kernel, stride in zip(kernels, strides, strict=False):
        receptive += (int(kernel) - 1) * stride_acc
        stride_acc *= int(stride)
    if stride_acc < 1 or receptive < 1:
        raise AudioDenseError("模型卷积参数非法")
    return int(stride_acc), int(receptive)


def _valid_frame_count(samples: int, *, receptive: int, stride: int) -> int:
    if samples < 1:
        return 0
    return max(0, (int(samples) - int(receptive)) // int(stride) + 1)


class HFAudioEncoder:
    """冻结 MERT/HuBERT 风格 transformers 音频编码器（默认 MERT-v1-95M）。

    懒加载；MERT 的 custom code 在 transformers 5.20.2 下 output_hidden_states
    不返回 hidden_states，所以用 forward hook 捕获第 layer 层（1-based，默认 9）
    encoder 层输出；layer=None 时取 last_hidden_state。
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        revision: str | None = DEFAULT_REVISION,
        cache_dir: str | Path | None = None,
        device: str = "auto",
        layer: int | None = DEFAULT_LAYER,
        trust_remote_code: bool = True,
        local_files_only: bool = False,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise AudioDenseError(f"model_id 不能为空，收到 {model_id!r}")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise AudioDenseError(f"revision 必须是字符串或 None，收到 {revision!r}")
        if device not in ("auto", "cpu", "cuda"):
            raise AudioDenseError(f"device 必须是 auto/cpu/cuda，收到 {device!r}")
        self.model_id = model_id.strip()
        self.revision = revision.strip() if isinstance(revision, str) else None
        self.cache_dir = _resolve_hub_cache_dir(cache_dir)
        self.device = device
        self.layer = _validate_layer(layer)
        self.trust_remote_code = bool(trust_remote_code)
        self.local_files_only = bool(local_files_only)
        self.version = (
            self.model_id if self.revision is None else f"{self.model_id}@{self.revision}"
        )
        self.sample_rate = 0
        self.frame_rate = 0.0
        self.dim = 0
        self.frame_stride_samples = 0
        self.frame_receptive_samples = 0
        self.model_load_s = 0.0
        self._model: Any = None
        self._torch: Any = None
        self._loaded = False
        self._resolved_device = "cpu"
        self._encoder_layers: list[Any] = []

    def _load_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        if self.revision is not None:
            kwargs["revision"] = self.revision
        if self.cache_dir is not None:
            kwargs["cache_dir"] = str(self.cache_dir)
        return kwargs

    def _select_device(self, torch: Any) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise AudioDenseError("device=cuda 但当前环境没有可用 CUDA 设备")
        return self.device

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        started = time.perf_counter()
        try:
            import torch
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:
            raise AudioDenseError(f"加载音频编码器需要 torch/transformers：{exc}") from exc
        kwargs = self._load_kwargs()
        captured = io.StringIO()
        try:
            # MERT 的 custom modeling 在 import 时会 print 提示（例如缺少 nnAudio），
            # 这里重定向 stdout，避免污染 CLI --json 的机器可读输出。
            with contextlib.redirect_stdout(captured):
                config = AutoConfig.from_pretrained(self.model_id, **kwargs)
                model = AutoModel.from_pretrained(self.model_id, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise AudioDenseError(f"加载音频编码器失败（{self.model_id}）：{exc}") from exc
        console_output = captured.getvalue().strip()
        if console_output:
            logger.debug("音频编码器加载时的 stdout: %s", console_output)
        device = self._select_device(torch)
        try:
            model.eval()
            model = model.to(device)
        except (RuntimeError, ValueError) as exc:
            raise AudioDenseError(f"音频编码器无法移动到设备 {device}：{exc}") from exc
        sample_rate = int(getattr(config, "sample_rate", 0) or 0) or DEFAULT_SAMPLE_RATE
        stride, receptive = _conv_output_stats(config)
        hidden_size = int(getattr(config, "hidden_size", 0) or 0)
        if hidden_size <= 0:
            raise AudioDenseError("模型 config 缺少 hidden_size")
        layers = getattr(getattr(model, "encoder", None), "layers", None)
        encoder_layers = list(layers) if layers is not None else []
        if self.layer is not None:
            if not encoder_layers:
                raise AudioDenseError("模型没有 encoder.layers，无法按 layer 取帧特征")
            if self.layer > len(encoder_layers):
                raise AudioDenseError(
                    f"layer={self.layer} 超过模型层数 {len(encoder_layers)}"
                )
        self.sample_rate = sample_rate
        self.frame_stride_samples = stride
        self.frame_receptive_samples = receptive
        self.frame_rate = float(sample_rate) / float(stride)
        self.dim = hidden_size
        self._model = model
        self._torch = torch
        self._resolved_device = device
        self._encoder_layers = encoder_layers
        self._loaded = True
        self.model_load_s = time.perf_counter() - started
        logger.info(
            "音频编码器加载完成：%s（device=%s sr=%d frame_rate=%.2fHz layer=%s dim=%d 耗时 %.1fs）",
            self.version,
            device,
            sample_rate,
            self.frame_rate,
            self.layer,
            self.dim,
            self.model_load_s,
        )

    def encode(
        self, waveforms: np.ndarray, valid_samples: Sequence[int]
    ) -> list[np.ndarray]:
        """编码 (B, n_samples) float32 波形；每个样本返回 (valid_frames, D)。"""

        array = np.asarray(waveforms)
        if array.ndim != 2:
            raise AudioDenseError(f"waveforms 必须是 (B,n) 二维数组，收到 {array.shape}")
        if array.shape[0] < 1:
            raise AudioDenseError("waveforms 批次为空")
        if array.dtype != np.float32:
            array = array.astype(np.float32, copy=False)
        lengths = [int(value) for value in valid_samples]
        if len(lengths) != int(array.shape[0]):
            raise AudioDenseError("valid_samples 数量必须与 batch 一致")
        if any(value < 1 or value > array.shape[1] for value in lengths):
            raise AudioDenseError(f"valid_samples 非法: {lengths}（n={array.shape[1]}）")
        self._ensure_loaded()
        assert self._torch is not None and self._model is not None
        torch = self._torch
        tensor = torch.from_numpy(np.ascontiguousarray(array)).to(self._resolved_device)
        attention_mask = torch.zeros(
            (tensor.shape[0], tensor.shape[1]),
            dtype=torch.long,
            device=self._resolved_device,
        )
        for row, length in enumerate(lengths):
            attention_mask[row, :length] = 1
        captured: list[Any] = []
        handle = None
        if self.layer is not None:
            module = self._encoder_layers[self.layer - 1]

            def _hook(_module: Any, _inputs: Any, output: Any) -> None:
                captured.append(output[0] if isinstance(output, (tuple, list)) else output)

            handle = module.register_forward_hook(_hook)
        cuda = self._resolved_device == "cuda"
        if cuda:
            torch.cuda.synchronize()
        try:
            with torch.inference_mode():
                outputs = self._model(tensor, attention_mask=attention_mask)
        except Exception as exc:  # noqa: BLE001
            raise AudioDenseError(f"音频编码前向失败：{exc}") from exc
        finally:
            if handle is not None:
                handle.remove()
        if cuda:
            torch.cuda.synchronize()
        if self.layer is not None:
            if not captured:
                raise AudioDenseError("未捕获到指定层输出（forward hook 失败）")
            hidden = captured[0]
        else:
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                raise AudioDenseError("模型输出缺少 last_hidden_state")
        matrix = hidden.detach().to(device="cpu", dtype=torch.float32).numpy()
        result: list[np.ndarray] = []
        for row, length in enumerate(lengths):
            count = min(
                int(matrix.shape[1]),
                _valid_frame_count(
                    length,
                    receptive=self.frame_receptive_samples,
                    stride=self.frame_stride_samples,
                ),
            )
            count = max(1, count) if matrix.shape[1] >= 1 else 0
            result.append(np.ascontiguousarray(matrix[row, :count]))
        return result

    def close(self) -> None:
        was_loaded = self._loaded
        model = self._model
        self._model = None
        self._torch = None
        self._loaded = False
        self.model_load_s = 0.0
        if not was_loaded:
            return
        del model
        try:
            import torch
        except ImportError:  # pragma: no cover
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (RuntimeError, AssertionError):  # pragma: no cover
            logger.debug("清理音频编码器显存失败", exc_info=True)


# ──────────────────────────────────────────────────────────────────────────────
# 特征结构 / bin 聚合 / 编码流程
# ──────────────────────────────────────────────────────────────────────────────


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norms, 1e-9)


@dataclass(frozen=True)
class AudioDenseFeatures:
    """一部作品的密集音频特征（帧级 + 0.5s bin 级）。"""

    frame_features: np.ndarray
    frame_times: np.ndarray
    bin_features: np.ndarray
    bin_std: np.ndarray
    bin_times: np.ndarray
    bin_mask: np.ndarray
    duration_s: float
    version: str
    encoder: str
    revision: str | None
    sample_rate: int
    context_s: float
    hop_s: float
    layer: int | None
    pool: str
    bin_seconds: float
    normalize: str
    frame_rate: float
    dim: int
    decode_s: float = 0.0
    encode_s: float = 0.0
    model_load_s: float = 0.0
    wall_s: float = 0.0
    decoded_duration_s: float = 0.0
    created_at: str = ""
    work_id: int | None = None

    def __post_init__(self) -> None:
        frames = np.asarray(self.frame_features, dtype=np.float32)
        if frames.ndim != 2 or frames.shape[0] < 1 or frames.shape[1] < 1:
            raise AudioDenseError(f"frame_features 形状非法: {frames.shape}")
        n_frames, dim = frames.shape
        times = np.asarray(self.frame_times, dtype=np.float32)
        if times.shape != (n_frames,):
            raise AudioDenseError("frame_times 长度与 frame_features 不一致")
        bin_features = np.asarray(self.bin_features, dtype=np.float32)
        if bin_features.ndim != 2 or bin_features.shape[1] != dim:
            raise AudioDenseError("bin_features 形状非法")
        bins = int(bin_features.shape[0])
        if np.asarray(self.bin_std).shape != (bins, dim):
            raise AudioDenseError("bin_std 形状非法")
        if np.asarray(self.bin_mask).shape != (bins,):
            raise AudioDenseError("bin_mask 形状非法")
        duration = _positive_number(self.duration_s, what="duration_s")
        _positive_number(self.bin_seconds, what="bin_seconds")
        if not np.all(np.isfinite(frames)) or not np.all(np.isfinite(times)):
            raise AudioDenseError("帧特征/时间含 NaN/Inf")
        if not np.all(np.isfinite(bin_features)) or not np.all(
            np.isfinite(np.asarray(self.bin_std, dtype=np.float64))
        ):
            raise AudioDenseError("bin 特征含 NaN/Inf")
        if np.any(times < -1e-6) or np.any(times > duration + self.bin_seconds + 1e-6):
            raise AudioDenseError("frame_times 超出作品时长")
        object.__setattr__(self, "frame_features", frames)
        object.__setattr__(self, "frame_times", times)
        object.__setattr__(self, "bin_features", bin_features)
        object.__setattr__(self, "bin_mask", np.asarray(self.bin_mask, dtype=bool))


def _aggregate_bins(
    frame_features: np.ndarray,
    frame_times: np.ndarray,
    *,
    duration_s: float,
    bin_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把帧特征按全局 0.5s bin 聚合为 mean/std/mask/bin_times。"""

    n_bins = max(1, int(math.ceil(duration_s / bin_seconds - 1e-9)))
    dim = int(frame_features.shape[1])
    indices = np.floor(np.asarray(frame_times, dtype=np.float64) / bin_seconds).astype(
        np.int64
    )
    indices = np.clip(indices, 0, n_bins - 1)
    sums = np.zeros((n_bins, dim), dtype=np.float64)
    squares = np.zeros((n_bins, dim), dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    values = frame_features.astype(np.float64)
    np.add.at(sums, indices, values)
    np.add.at(squares, indices, values**2)
    np.add.at(counts, indices, 1.0)
    safe = np.maximum(counts, 1.0)[:, None]
    mean = sums / safe
    variance = np.maximum(squares / safe - mean**2, 0.0)
    std = np.sqrt(variance)
    mask = counts > 0
    mean[~mask] = 0.0
    std[~mask] = 0.0
    starts = np.arange(n_bins, dtype=np.float64) * bin_seconds
    ends = np.minimum(starts + bin_seconds, duration_s)
    bin_times = np.stack([starts, ends], axis=1).astype(np.float32)
    return mean.astype(np.float32), std.astype(np.float32), mask, bin_times


def extract_audio_features(
    source: str | Path,
    encoder: AudioFrameEncoder,
    *,
    version: str | None = None,
    sample_rate: int | None = None,
    context_s: float = DEFAULT_CONTEXT_S,
    hop_s: float = DEFAULT_HOP_S,
    bin_seconds: float = DEFAULT_BIN_SECONDS,
    normalize: str = DEFAULT_NORMALIZE,
    pool: str = DEFAULT_POOL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    ffmpeg: str = "ffmpeg",
    timeout_s: float = 600.0,
    work_id: int | None = None,
) -> AudioDenseFeatures:
    """解码整条音轨 -> 逐帧特征 -> 0.5s bin 聚合（流式 context windows）。"""

    src = Path(source)
    if not src.is_file():
        raise AudioDenseError(f"源文件不存在: {src}")
    if not callable(getattr(encoder, "encode", None)):
        raise AudioDenseError("encoder 缺少可调用的 encode(waveforms, valid_samples)")
    if not isinstance(getattr(encoder, "version", None), str):
        raise AudioDenseError("encoder 缺少字符串 version")
    bin_value = _positive_number(bin_seconds, what="bin_seconds")
    context_value = _positive_number(context_s, what="context_s")
    hop_value = _positive_number(hop_s, what="hop_s")
    if hop_value > context_value + 1e-9:
        raise AudioDenseError(f"hop_s({hop_value:g}) 不能大于 context_s({context_value:g})")
    norm_value = _validate_normalize(normalize)
    pool_value = _validate_pool(pool)
    batch = _positive_int(batch_size, what="batch_size")
    timeout = _positive_number(timeout_s, what="timeout_s")
    rate = _positive_int(
        sample_rate if sample_rate is not None else getattr(encoder, "sample_rate", 0),
        what="sample_rate",
    )
    dim = _positive_int(getattr(encoder, "dim", 0), what="encoder.dim")
    frame_rate = _positive_number(getattr(encoder, "frame_rate", 0.0), what="encoder.frame_rate")
    stride = int(getattr(encoder, "frame_stride_samples", 0) or 0)
    receptive = int(getattr(encoder, "frame_receptive_samples", 0) or 0)

    started = time.perf_counter()
    duration_s = probe_audio_duration(src, ffmpeg=ffmpeg, timeout_s=timeout)
    n_bins = max(1, int(math.ceil(duration_s / bin_value - 1e-9)))
    sums = np.zeros((n_bins, dim), dtype=np.float64)
    squares = np.zeros((n_bins, dim), dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    stats: dict[str, float] = {}
    windows = iter_audio_windows(
        src,
        sample_rate=rate,
        context_s=context_value,
        hop_s=hop_value,
        ffmpeg=ffmpeg,
        timeout_s=timeout,
        max_duration_s=duration_s,
        stats=stats,
    )
    frames: list[np.ndarray] = []
    times: list[np.ndarray] = []
    decode_s = 0.0
    encode_s = 0.0
    pending: list[tuple[float, np.ndarray, int]] = []

    def _flush() -> None:
        nonlocal encode_s
        if not pending:
            return
        waveforms = np.stack([item[1] for item in pending], axis=0)
        valid = [item[2] for item in pending]
        encode_started = time.perf_counter()
        try:
            encoded = encoder.encode(waveforms, valid)
        except AudioDenseError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AudioDenseError(
                f"音频编码失败（{type(encoder).__name__}）：{exc}"
            ) from exc
        encode_s += time.perf_counter() - encode_started
        if len(encoded) != len(pending):
            raise AudioDenseError(
                f"编码器返回 {len(encoded)} 个样本，期望 {len(pending)}"
            )
        for (window_start, _waveform, _actual), matrix in zip(pending, encoded, strict=False):
            array = np.asarray(matrix, dtype=np.float32)
            if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != dim:
                raise AudioDenseError(f"编码器输出形状异常: {array.shape}")
            if not np.all(np.isfinite(array)):
                raise AudioDenseError("编码器输出含 NaN/Inf")
            if norm_value == "l2":
                array = _normalize_rows(array)
            count = int(array.shape[0])
            if stride > 0 and receptive > 0:
                offsets = np.arange(count, dtype=np.float64) * stride + receptive / 2.0
                local_times = window_start + offsets / float(rate)
            else:
                local_times = window_start + (
                    np.arange(count, dtype=np.float64) + 0.5
                ) / frame_rate
            frames.append(array.astype(np.float16))
            times.append(local_times.astype(np.float32))
            bin_indices = np.clip(
                np.floor(local_times / bin_value).astype(np.int64), 0, n_bins - 1
            )
            values = array.astype(np.float64)
            np.add.at(sums, bin_indices, values)
            np.add.at(squares, bin_indices, values**2)
            np.add.at(counts, bin_indices, 1.0)
        pending.clear()

    iterator = iter(windows)
    try:
        while True:
            read_started = time.perf_counter()
            try:
                item = next(iterator)
            except StopIteration:
                decode_s += time.perf_counter() - read_started
                break
            decode_s += time.perf_counter() - read_started
            pending.append(item)
            if len(pending) >= batch:
                _flush()
        _flush()
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
    if not frames:
        raise AudioDenseError(f"没有可用的音频帧特征: {src}")
    frame_features = np.concatenate(frames, axis=0)
    frame_times = np.concatenate(times, axis=0)
    order = np.argsort(frame_times, kind="stable")
    frame_features = frame_features[order]
    frame_times = frame_times[order]
    safe = np.maximum(counts, 1.0)[:, None]
    bin_features = (sums / safe).astype(np.float32)
    variance = np.maximum(squares / safe - (sums / safe) ** 2, 0.0)
    bin_std = np.sqrt(variance).astype(np.float32)
    bin_mask = counts > 0
    bin_features[~bin_mask] = 0.0
    bin_std[~bin_mask] = 0.0
    starts = np.arange(n_bins, dtype=np.float64) * bin_value
    ends = np.minimum(starts + bin_value, duration_s)
    bin_times = np.stack([starts, ends], axis=1).astype(np.float32)
    if version is None:
        version = compose_audio_version(
            str(encoder.version),
            sample_rate=rate,
            context_s=context_value,
            hop_s=hop_value,
            layer=getattr(encoder, "layer", None),
            pool=pool_value,
            bin_seconds=bin_value,
            normalize=norm_value,
        )
    wall_s = time.perf_counter() - started
    return AudioDenseFeatures(
        frame_features=frame_features,
        frame_times=frame_times,
        bin_features=bin_features,
        bin_std=bin_std,
        bin_times=bin_times,
        bin_mask=bin_mask,
        duration_s=duration_s,
        version=version,
        encoder=str(encoder.version),
        revision=getattr(encoder, "revision", None),
        sample_rate=rate,
        context_s=context_value,
        hop_s=hop_value,
        layer=getattr(encoder, "layer", None),
        pool=pool_value,
        bin_seconds=bin_value,
        normalize=norm_value,
        frame_rate=frame_rate,
        dim=dim,
        decode_s=decode_s,
        encode_s=encode_s,
        model_load_s=float(getattr(encoder, "model_load_s", 0.0) or 0.0),
        wall_s=wall_s,
        decoded_duration_s=float(stats.get("decoded_duration_s", 0.0)),
        created_at=now_utc_iso(),
        work_id=work_id,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 保存 / 读取（原子写）
# ──────────────────────────────────────────────────────────────────────────────


def _atomic_write_bytes(path: Path, writer: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
        # Windows 上若目标文件正被其它进程短暂读取，os.replace 可能报
        # WinError 5；重试几次避免并发读取导致整部作品失败。
        last_error: OSError | None = None
        for attempt in range(3):
            try:
                os.replace(tmp_path, target)
                last_error = None
                break
            except OSError as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.2 * (attempt + 1))
        if last_error is not None:
            raise AudioDenseError(f"写入文件失败（{target}）：{last_error}") from last_error
    except OSError as exc:
        raise AudioDenseError(f"写入文件失败（{target}）：{exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


def _manifest_payload(features: AudioDenseFeatures) -> dict[str, Any]:
    return {
        "version": features.version,
        "encoder": features.encoder,
        "revision": features.revision,
        "sample_rate": int(features.sample_rate),
        "context_s": float(features.context_s),
        "hop_s": float(features.hop_s),
        "layer": features.layer,
        "pool": features.pool,
        "bin_seconds": float(features.bin_seconds),
        "normalize": features.normalize,
        "frame_rate": float(features.frame_rate),
        "dim": int(features.dim),
        "n_frames": int(features.frame_features.shape[0]),
        "n_bins": int(features.bin_features.shape[0]),
        "duration_s": float(features.duration_s),
        "decoded_duration_s": float(features.decoded_duration_s),
        "decode_s": float(features.decode_s),
        "encode_s": float(features.encode_s),
        "model_load_s": float(features.model_load_s),
        "wall_s": float(features.wall_s),
        "created_at": features.created_at or now_utc_iso(),
        "work_id": features.work_id,
        "files": {"audio": AUDIO_NPZ_NAME, "manifest": AUDIO_MANIFEST_NAME},
    }


def save_audio_bundle(
    out_dir: str | Path, features: AudioDenseFeatures, *, force: bool = False
) -> dict[str, Path]:
    """原子写 audio.npz + manifest.json；已存在完整产物且 force=False 时跳过。"""

    if not isinstance(features, AudioDenseFeatures):
        raise AudioDenseError(f"需要 AudioDenseFeatures，收到 {type(features).__name__}")
    target = Path(out_dir)
    manifest_path = target / AUDIO_MANIFEST_NAME
    npz_path = target / AUDIO_NPZ_NAME
    if not force and is_audio_bundle_complete(target, version=features.version):
        return {"npz": npz_path, "manifest": manifest_path, "skipped": True}  # type: ignore[dict-item]
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AudioDenseError(f"无法创建输出目录（{target}）：{exc}") from exc
    arrays = {
        "frame_features": np.asarray(features.frame_features, dtype=np.float16),
        "frame_times": np.asarray(features.frame_times, dtype=np.float32),
        "bin_features": np.asarray(features.bin_features, dtype=np.float32),
        "bin_std": np.asarray(features.bin_std, dtype=np.float32),
        "bin_times": np.asarray(features.bin_times, dtype=np.float32),
        "bin_mask": np.asarray(features.bin_mask, dtype=bool),
        "duration_s": np.float64(features.duration_s),
    }

    def _write_npz(handle: Any) -> None:
        np.savez(handle, **arrays)

    _atomic_write_bytes(npz_path, _write_npz)
    manifest_text = json.dumps(
        _manifest_payload(features), ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    _atomic_write_bytes(
        manifest_path, lambda handle: handle.write(manifest_text.encode("utf-8"))
    )
    logger.info(
        "音频特征已保存: %s（%d 帧 / %d bin, %.1fs 音频）",
        target,
        int(features.frame_features.shape[0]),
        int(features.bin_features.shape[0]),
        features.duration_s,
    )
    return {"npz": npz_path, "manifest": manifest_path}


def load_audio_bundle(out_dir: str | Path) -> AudioDenseFeatures | None:
    """读取 audio.npz + manifest；缺失返回 None，损坏抛 AudioDenseError。"""

    target = Path(out_dir)
    npz_path = target / AUDIO_NPZ_NAME
    manifest_path = target / AUDIO_MANIFEST_NAME
    if not target.is_dir() or not npz_path.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AudioDenseError(f"音频 manifest 无法读取（{manifest_path}）：{exc}") from exc
    if not isinstance(manifest, dict):
        raise AudioDenseError("音频 manifest 顶层必须是对象")
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
    except (OSError, ValueError, KeyError) as exc:
        raise AudioDenseError(f"音频 npz 读取失败（{npz_path}）：{exc}") from exc
    required = {
        "frame_features",
        "frame_times",
        "bin_features",
        "bin_std",
        "bin_times",
        "bin_mask",
        "duration_s",
    }
    missing = required - set(arrays)
    if missing:
        raise AudioDenseError(f"音频 npz 缺少数组: {sorted(missing)}")
    try:
        return AudioDenseFeatures(
            frame_features=arrays["frame_features"],
            frame_times=arrays["frame_times"],
            bin_features=arrays["bin_features"],
            bin_std=arrays["bin_std"],
            bin_times=arrays["bin_times"],
            bin_mask=arrays["bin_mask"],
            duration_s=float(np.asarray(arrays["duration_s"]).item()),
            version=str(manifest.get("version") or ""),
            encoder=str(manifest.get("encoder") or ""),
            revision=None if manifest.get("revision") is None else str(manifest["revision"]),
            sample_rate=int(manifest.get("sample_rate", 0) or 0),
            context_s=float(manifest.get("context_s", 0.0) or 0.0),
            hop_s=float(manifest.get("hop_s", 0.0) or 0.0),
            layer=None if manifest.get("layer") is None else int(manifest["layer"]),
            pool=str(manifest.get("pool") or ""),
            bin_seconds=float(manifest.get("bin_seconds", 0.0) or 0.0),
            normalize=str(manifest.get("normalize") or ""),
            frame_rate=float(manifest.get("frame_rate", 0.0) or 0.0),
            dim=int(manifest.get("dim", 0) or 0),
            decode_s=float(manifest.get("decode_s", 0.0)),
            encode_s=float(manifest.get("encode_s", 0.0)),
            model_load_s=float(manifest.get("model_load_s", 0.0)),
            wall_s=float(manifest.get("wall_s", 0.0)),
            decoded_duration_s=float(manifest.get("decoded_duration_s", 0.0)),
            created_at=str(manifest.get("created_at", "")),
            work_id=(
                None if manifest.get("work_id") is None else int(manifest["work_id"])
            ),
        )
    except (AudioDenseError, KeyError, TypeError, ValueError) as exc:
        raise AudioDenseError(f"音频 manifest 字段非法（{manifest_path}）：{exc}") from exc


def is_audio_bundle_complete(out_dir: str | Path, *, version: str | None = None) -> bool:
    """断点复用判定：manifest+npz 存在、n_frames/n_bins 一致、版本匹配。"""

    target = Path(out_dir)
    npz_path = target / AUDIO_NPZ_NAME
    manifest_path = target / AUDIO_MANIFEST_NAME
    if not npz_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if version is not None and str(manifest.get("version")) != str(version):
            return False
        with np.load(npz_path, allow_pickle=False) as data:
            n_frames = int(np.asarray(data["frame_features"]).shape[0])
            n_bins = int(np.asarray(data["bin_features"]).shape[0])
        return bool(
            n_frames == int(manifest.get("n_frames", -1))
            and n_bins == int(manifest.get("n_bins", -1))
            and n_frames > 0
            and n_bins > 0
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
