"""v0.4.3 密集视觉语义编码：冻结视觉编码器 + 帧级向量 + 窗口池化。

背景：v0.4 的 PE-AV 是 8s / 16 帧（约 2fps）稀疏语义，更高频的只有 96x54 灰度
事件统计，缺少逐帧视觉语义。本模块先搭基础设施：用单条 ffmpeg 命令按目标帧率
流式解码整部视频，交给冻结的视觉编码器（CLIP / DINOv2 等）得到帧级语义向量，
再按 8s 窗口（重叠时长加权）池化为窗口向量并保存。

设计要点：

- 解码不整片进内存：ffmpeg rawvideo 经 pipe 流式产出 (B,H,W,3) uint8 批次；
- 编码器通过 VisualEncoder Protocol 注入，默认实现 HFImageEncoder 懒加载
  transformers 冻结模型（CLIP 用 get_image_features，其他用 AutoModel CLS），
  默认测试用 fake encoder，不加载 torch / 下载模型；
- 向量逐行 L2 归一化后保存为 fp16 npz + JSON manifest（原子写）；
- 池化按帧与窗口的重叠时长加权平均，再做 L2 归一化；可选 pool_windows_std
  返回窗口内逐维加权标准差；
- 结果只读源视频、只写用户指定目录，不写数据库。

边界：模块导入时不 import torch/transformers；真实模型加载与推理由 pilot CLI
在具备 GPU / 权重缓存的环境执行。
"""

from __future__ import annotations

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
from .util import now_utc_iso

logger = logging.getLogger(__name__)

DEFAULT_FPS = 8.0
"""默认密集视觉采样帧率。"""

DEFAULT_SIZE = 224
"""默认编码器输入边长（正方形，ffmpeg 直接缩放到该尺寸）。"""

DEFAULT_BATCH_SIZE = 16
"""默认每批送入编码器的帧数。"""

DENSE_VISUAL_NPZ_NAME = "dense_visual.npz"
"""帧级向量 npz 文件名。"""

DENSE_VISUAL_MANIFEST_NAME = "dense_visual_manifest.json"
"""元数据 manifest 文件名。"""

_EPS = 1e-12
_STDERR_TAIL_CHARS = 1000
_SUPPORTED_DEVICES = ("auto", "cpu", "cuda")
_SUPPORTED_DTYPES = ("auto", "float32", "bfloat16")


class DenseVisualError(NebulaError):
    """密集视觉编解码、池化或存取失败。"""


# ──────────────────────────────────────────────────────────────────────────────
# 参数校验小工具
# ──────────────────────────────────────────────────────────────────────────────


def _require_positive_number(value: Any, *, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DenseVisualError(f"{what} 必须是数字，收到 {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise DenseVisualError(f"{what} 必须大于 0，收到 {value!r}")
    return number


def _require_positive_int(value: Any, *, what: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DenseVisualError(f"{what} 必须是整数，收到 {value!r}")
    if value < minimum:
        raise DenseVisualError(f"{what} 必须 >= {minimum}，收到 {value!r}")
    return int(value)


def _stderr_tail(stderr: bytes | str | None, limit: int = _STDERR_TAIL_CHARS) -> str:
    if not stderr:
        return ""
    text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else str(stderr)
    text = text.strip()
    if len(text) > limit:
        return "..." + text[-limit:]
    return text


def _resolve_ffprobe(ffmpeg: str) -> str:
    """尽量找到与 ffmpeg 同目录的 ffprobe，否则退回 PATH。"""

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


def _which(name: str) -> str | None:
    import shutil

    return shutil.which(name)


def _probe_duration_s(source: Path, *, ffmpeg: str, timeout_s: float) -> float:
    """用 ffprobe 读取容器时长（秒）；失败抛 DenseVisualError。"""

    ffprobe = _resolve_ffprobe(ffmpeg)
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(source),
    ]
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
        raise DenseVisualError(f"找不到 ffprobe 可执行文件（{ffprobe}）：请检查 PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise DenseVisualError(f"读取视频时长超时（>{timeout_s:g}s）：{source}") from exc
    except OSError as exc:
        raise DenseVisualError(f"无法启动 ffprobe（{ffprobe}）：{exc}") from exc
    if proc.returncode != 0:
        tail = _stderr_tail(proc.stderr)
        detail = f"；stderr 摘要: {tail}" if tail else ""
        raise DenseVisualError(f"ffprobe 无法读取视频时长（{source}）{detail}")
    text = (proc.stdout or "").strip()
    try:
        duration_s = float(text)
    except ValueError as exc:
        raise DenseVisualError(f"ffprobe 时长输出无法解析: {text!r}（{source}）") from exc
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise DenseVisualError(f"视频时长非法: {duration_s!r}（{source}）")
    return duration_s


def _read_exact(stream: Any, size: int) -> bytes | None:
    """从管道精确读 size 字节；干净 EOF 返回 None，读到一半 EOF 抛错。"""

    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        data = stream.read(remaining)
        if not data:
            break
        chunks.append(data)
        remaining -= len(data)
    if remaining == 0:
        return b"".join(chunks)
    if not chunks:
        return None
    raise DenseVisualError("ffmpeg 原始帧输出被截断（帧数据不完整）")


# ──────────────────────────────────────────────────────────────────────────────
# 解码：单条 ffmpeg 流式产出帧批次
# ──────────────────────────────────────────────────────────────────────────────


def iter_frame_batches(
    source: str | Path,
    *,
    fps: float = DEFAULT_FPS,
    size: int = DEFAULT_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    ffmpeg: str = "ffmpeg",
    timeout_s: float = 600.0,
) -> Iterator[np.ndarray]:
    """按固定帧率流式解码视频，逐批产出 (B,H,W,3) uint8 RGB 帧。

    只发一条 ffmpeg 命令：
    -vf fps={fps},scale={size}:{size},format=rgb24 -f rawvideo -pix_fmt rgb24 pipe:1。
    不会把整部视频读进内存；每批最多 batch_size 帧，最后一维为 RGB 三通道。

    校验：ffprobe 时长可读、ffmpeg 退出码为 0、帧数据完整（非截断）、实际帧数与
    duration*fps 在容差内（防止管道被截断/时基异常）、至少解码出一帧。

    异常
    ----
    DenseVisualError:
        文件不存在、参数非法、ffprobe/ffmpeg 失败、输出截断或帧数明显不符时抛出。
    """

    src = Path(source)
    if not src.is_file():
        raise DenseVisualError(f"视频文件不存在: {src}")
    fps_value = _require_positive_number(fps, what="fps")
    size_value = _require_positive_int(size, what="size", minimum=2)
    batch_value = _require_positive_int(batch_size, what="batch_size")
    timeout = _require_positive_number(timeout_s, what="timeout_s")
    duration_s = _probe_duration_s(src, ffmpeg=ffmpeg, timeout_s=timeout)

    command = [
        str(ffmpeg),
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(src),
        "-an",
        "-sn",
        "-vf",
        f"fps={fps_value:g},scale={size_value}:{size_value},format=rgb24",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise DenseVisualError(f"找不到 ffmpeg 可执行文件（{ffmpeg}）：请检查 PATH") from exc
    except OSError as exc:
        raise DenseVisualError(f"无法启动 ffmpeg 解码（{ffmpeg}）：{exc}") from exc

    frame_bytes = size_value * size_value * 3
    expected_frames = max(1, int(round(duration_s * fps_value)))
    tolerance = max(2, int(math.ceil(fps_value * 0.5)))
    emitted = 0
    buffer: list[np.ndarray] = []
    stderr = b""
    returncode = -1
    deadline = time.monotonic() + timeout
    try:
        assert process.stdout is not None and process.stderr is not None
        while True:
            if time.monotonic() > deadline:
                process.kill()
                raise DenseVisualError(f"视频解码超时（>{timeout:g}s）：{src}")
            raw = _read_exact(process.stdout, frame_bytes)
            if raw is None:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                size_value, size_value, 3
            ).copy()
            buffer.append(frame)
            if len(buffer) >= batch_value:
                batch = np.stack(buffer).astype(np.uint8, copy=False)
                buffer.clear()
                emitted += int(batch.shape[0])
                yield batch
        if buffer:
            batch = np.stack(buffer).astype(np.uint8, copy=False)
            buffer.clear()
            emitted += int(batch.shape[0])
            yield batch
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
        raise DenseVisualError(f"ffmpeg 解码失败（退出码 {returncode}）：{src}{detail}")
    if emitted == 0:
        raise DenseVisualError(f"没有解码出任何帧: {src}")
    if abs(emitted - expected_frames) > tolerance:
        raise DenseVisualError(
            f"解码帧数与时长不匹配: {emitted} 帧，预期约 {expected_frames} 帧"
            f"（duration={duration_s:.3f}s, fps={fps_value:g}）: {src}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 编码器协议与 transformers 冻结视觉编码器
# ──────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class VisualEncoder(Protocol):
    """帧级视觉编码器统一接口（测试可注入 fake 实现）。"""

    version: str
    dim: int

    def encode(self, frames: np.ndarray) -> np.ndarray:
        """把 (B,H,W,3) uint8 帧编码成 (B,D) float32 逐行归一化向量。"""
        ...

    def close(self) -> None:
        """释放模型 / 显存；允许重复调用。"""
        ...


def _resolve_hub_cache_dir(cache_dir: str | Path | None) -> Path | None:
    """把配置里的模型缓存根目录解析成 huggingface_hub 缓存目录。

    与 nebula.embed 的兼容逻辑一致：config 的 .cache/hf 是 HF_HOME 布局，
    真正的 hub 缓存位于 <cache_dir>/hub；from_pretrained(cache_dir=...) 需要
    直接指向含 models--* 的目录。
    """

    if cache_dir is None:
        return None
    root = Path(cache_dir)
    hub = root / "hub"
    if hub.is_dir():
        return hub
    return root


class HFImageEncoder:
    """transformers 冻结图像编码器（懒加载，CLIP 或通用 AutoModel）。

    - CLIP 家族（model_type == "clip"）：用 CLIPModel.get_image_features 的
      投影后图像特征（兼容旧版直接返回 tensor / 新版返回 model output）；
    - 其他模型（如 DINOv2）：AutoModel 前向取 last_hidden_state 的 CLS（第 0 行），
      没有 last_hidden_state 时退回 pooler_output。

    CPU 用 float32；CUDA 在 dtype=auto/bfloat16 时用 bfloat16 autocast（权重保持
    float32），输出统一转 CPU float32 并逐行 L2 归一化。模型加载耗时记录在
    model_load_s；close 释放模型与显存。
    """

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        device: str = "auto",
        dtype: str = "auto",
        local_files_only: bool = False,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise DenseVisualError(f"model_id 不能为空，收到 {model_id!r}")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise DenseVisualError(f"revision 必须是字符串或 None，收到 {revision!r}")
        if device not in _SUPPORTED_DEVICES:
            raise DenseVisualError(f"device 必须是 {_SUPPORTED_DEVICES}，收到 {device!r}")
        if dtype not in _SUPPORTED_DTYPES:
            raise DenseVisualError(f"dtype 必须是 {_SUPPORTED_DTYPES}，收到 {dtype!r}")
        self.model_id = model_id.strip()
        self.revision = revision.strip() if isinstance(revision, str) else None
        self.cache_dir = _resolve_hub_cache_dir(cache_dir)
        self.device = device
        self.dtype = dtype
        self.local_files_only = bool(local_files_only)
        self.version = (
            self.model_id if self.revision is None else f"{self.model_id}@{self.revision}"
        )
        self.dim = 0
        self.model_load_s = 0.0
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._resolved_device: str = "cpu"
        self._use_autocast = False
        self._is_clip = False
        self._loaded = False

    # ---------------------------------------------------------------- 懒加载
    def _load_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"local_files_only": self.local_files_only}
        if self.revision is not None:
            kwargs["revision"] = self.revision
        if self.cache_dir is not None:
            kwargs["cache_dir"] = str(self.cache_dir)
        return kwargs

    def _select_device(self, torch: Any) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise DenseVisualError("device=cuda 但当前环境没有可用的 CUDA 设备")
        return self.device

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        started = time.perf_counter()
        try:
            import torch
            from transformers import AutoConfig, AutoImageProcessor, AutoModel
        except ImportError as exc:  # pragma: no cover - 依赖缺失环境才会触发
            raise DenseVisualError(
                f"加载视觉编码器需要 torch/transformers：{exc}"
            ) from exc
        kwargs = self._load_kwargs()
        try:
            config = AutoConfig.from_pretrained(self.model_id, **kwargs)
            is_clip = str(getattr(config, "model_type", "")) == "clip"
            if is_clip:
                from transformers import CLIPModel

                model = CLIPModel.from_pretrained(self.model_id, **kwargs)
            else:
                model = AutoModel.from_pretrained(self.model_id, **kwargs)
            processor = AutoImageProcessor.from_pretrained(self.model_id, **kwargs)
        except Exception as exc:  # noqa: BLE001 - 统一转项目异常
            raise DenseVisualError(
                f"加载视觉编码器失败（{self.model_id}）：{exc}"
            ) from exc

        resolved_device = self._select_device(torch)
        use_autocast = False
        try:
            model.eval()
            if resolved_device == "cpu":
                model = model.float()
            elif self.dtype in ("auto", "bfloat16"):
                use_autocast = True
            model = model.to(resolved_device)
        except (RuntimeError, ValueError) as exc:
            raise DenseVisualError(
                f"视觉编码器无法移动到设备 {resolved_device}：{exc}"
            ) from exc

        self._model = model
        self._processor = processor
        self._torch = torch
        self._resolved_device = resolved_device
        self._use_autocast = use_autocast
        self._is_clip = is_clip
        self.dim = self._infer_dim(config)
        self._loaded = True
        self.model_load_s = time.perf_counter() - started
        logger.info(
            "视觉编码器加载完成：%s（device=%s dtype=%s is_clip=%s dim=%s 耗时 %.1fs）",
            self.version,
            resolved_device,
            "bfloat16(autocast)" if use_autocast else "float32",
            self._is_clip,
            self.dim,
            self.model_load_s,
        )

    @staticmethod
    def _infer_dim(config: Any) -> int:
        """从 config 猜输出维度（CLIP 用 projection_dim，DINOv2 用 hidden_size）。"""

        candidates = [
            getattr(config, "projection_dim", None),
            getattr(config, "hidden_size", None),
            getattr(getattr(config, "vision_config", None), "hidden_size", None),
        ]
        for value in candidates:
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if number > 0:
                return number
        return 0

    # ---------------------------------------------------------------- 前向
    def _forward_features(self, pixel_values: Any) -> Any:
        assert self._model is not None
        torch = self._torch
        if self._is_clip:
            outputs = self._model.get_image_features(pixel_values=pixel_values)
            if torch.is_tensor(outputs):
                return outputs
            for attribute in ("pooler_output", "image_embeds"):
                value = getattr(outputs, attribute, None)
                if value is not None:
                    return value
            raise DenseVisualError("CLIP 输出中没有图像特征（pooler_output/image_embeds）")
        outputs = self._model(pixel_values=pixel_values, return_dict=True)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is not None:
            return hidden[:, 0]
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is not None:
            return pooled
        raise DenseVisualError("AutoModel 输出中没有 last_hidden_state / pooler_output")

    def encode(self, frames: np.ndarray) -> np.ndarray:
        """把 (B,H,W,3) uint8 RGB 帧编码成 (B,D) float32 逐行归一化向量。"""

        array = np.asarray(frames)
        if array.ndim != 4 or array.shape[3] != 3:
            raise DenseVisualError(
                f"frames 必须是 (B,H,W,3) uint8 数组，收到 {array.shape}"
            )
        if array.dtype != np.uint8:
            raise DenseVisualError(f"frames 必须是 uint8，收到 {array.dtype}")
        if array.shape[0] < 1:
            raise DenseVisualError("frames 批次为空")
        self._ensure_loaded()
        assert self._torch is not None and self._processor is not None
        torch = self._torch
        try:
            images = [array[index] for index in range(int(array.shape[0]))]
            inputs = self._processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self._resolved_device)
        except DenseVisualError:
            raise
        except Exception as exc:  # noqa: BLE001 - 预处理异常统一包装
            raise DenseVisualError(f"视觉编码预处理失败：{exc}") from exc

        cuda = self._resolved_device == "cuda"
        if cuda:
            torch.cuda.synchronize()
        try:
            with torch.inference_mode():
                if self._use_autocast:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        features = self._forward_features(pixel_values)
                else:
                    features = self._forward_features(pixel_values)
        except DenseVisualError:
            raise
        except Exception as exc:  # noqa: BLE001 - 前向异常统一包装
            raise DenseVisualError(f"视觉编码前向失败：{exc}") from exc
        if cuda:
            torch.cuda.synchronize()

        matrix = features.detach().to(device="cpu", dtype=torch.float32).numpy()
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.ndim != 2 or matrix.shape[0] != array.shape[0]:
            raise DenseVisualError(f"编码器输出形状异常: {matrix.shape}")
        if self.dim > 0 and int(matrix.shape[1]) != self.dim:
            raise DenseVisualError(
                f"编码器输出维度 {matrix.shape[1]} 与 dim={self.dim} 不一致"
            )
        self.dim = int(matrix.shape[1])
        return _normalize_rows(matrix, what="视觉编码器输出")

    def close(self) -> None:
        """释放模型与显存；允许重复调用。"""

        was_loaded = self._loaded
        model = self._model
        self._model = None
        self._processor = None
        self._torch = None
        self._loaded = False
        self.model_load_s = 0.0
        if not was_loaded:
            return
        del model
        try:
            import torch
        except ImportError:  # pragma: no cover - 加载过模型时必然可用
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (RuntimeError, AssertionError):  # pragma: no cover - 仅 CUDA 环境
            logger.debug("清理视觉编码器显存缓存失败", exc_info=True)


# ──────────────────────────────────────────────────────────────────────────────
# 向量工具 / 特征数据结构
# ──────────────────────────────────────────────────────────────────────────────


def _normalize_rows(matrix: Any, *, what: str) -> np.ndarray:
    """把二维矩阵逐行 L2 归一化，返回 float32（不修改输入数组）。"""

    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2:
        raise DenseVisualError(f"{what}必须是二维矩阵，收到 {array.shape}")
    if array.shape[0] == 0:
        raise DenseVisualError(f"{what}为空（0 行）")
    if not np.all(np.isfinite(array)):
        raise DenseVisualError(f"{what}含 NaN/Inf")
    norms = np.linalg.norm(array, axis=1)
    bad = (~np.isfinite(norms)) | (norms <= _EPS)
    if np.any(bad):
        row = int(np.flatnonzero(bad)[0])
        raise DenseVisualError(f"{what}第 {row} 行范数为 0 或非有限")
    return np.ascontiguousarray(array / norms[:, None], dtype=np.float32)


def compress_version(encoder_version: str, fps: float, size: int) -> str:
    """compose DenseVisualFeatures.version：编码器版本 + fps + 输入尺寸。"""

    if not isinstance(encoder_version, str) or not encoder_version.strip():
        raise DenseVisualError(f"encoder_version 不能为空，收到 {encoder_version!r}")
    fps_value = _require_positive_number(fps, what="fps")
    size_value = _require_positive_int(size, what="size", minimum=2)
    return f"{encoder_version.strip()}__fps{fps_value:g}_s{size_value}"


def _as_feature_matrix(
    value: Any, *, expected_rows: int, what: str
) -> np.ndarray:
    """校验编码器输出为 (expected_rows, dim) 的有限 float32 矩阵。"""

    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2:
        raise DenseVisualError(f"{what}必须是二维矩阵，收到 {matrix.shape}")
    if matrix.shape[0] != expected_rows:
        raise DenseVisualError(
            f"{what}行数异常: {matrix.shape[0]}（期望 {expected_rows}）"
        )
    if matrix.shape[1] < 1:
        raise DenseVisualError(f"{what}维度为 0")
    if not np.all(np.isfinite(matrix)):
        raise DenseVisualError(f"{what}含 NaN/Inf")
    return matrix


@dataclass(frozen=True)
class DenseVisualFeatures:
    """一部作品的密集视觉语义序列。

    per_frame 为 (N,D) float32 逐行 L2 归一化帧向量；下标的帧时间区间为
    [i/fps, (i+1)/fps)。version 由编码器版本 + fps + 输入尺寸组成，用于区分
    不同编码方案的序列，禁止混用。
    """

    per_frame: np.ndarray
    fps: float
    duration_s: float
    n_frames: int
    dim: int
    version: str
    decode_s: float
    encode_s: float
    model_load_s: float
    encoder: str = ""
    revision: str | None = None
    size: int = 0
    created_at: str = ""

    def __post_init__(self) -> None:
        per_frame = np.asarray(self.per_frame)
        if per_frame.ndim != 2:
            raise DenseVisualError(f"per_frame 必须是二维矩阵，收到 {per_frame.shape}")
        if int(self.n_frames) != int(per_frame.shape[0]):
            raise DenseVisualError(
                f"n_frames={self.n_frames} 与 per_frame 行数 {per_frame.shape[0]} 不一致"
            )
        if int(self.dim) != int(per_frame.shape[1]):
            raise DenseVisualError(
                f"dim={self.dim} 与 per_frame 列数 {per_frame.shape[1]} 不一致"
            )
        if self.n_frames < 1 or self.dim < 1:
            raise DenseVisualError("密集视觉序列不能为空")
        if not np.all(np.isfinite(per_frame)):
            raise DenseVisualError("per_frame 含 NaN/Inf")
        fps_value = _require_positive_number(self.fps, what="fps")
        _require_positive_number(self.duration_s, what="duration_s")
        for name, value in (
            ("decode_s", self.decode_s),
            ("encode_s", self.encode_s),
            ("model_load_s", self.model_load_s),
        ):
            if float(value) < 0.0 or not math.isfinite(float(value)):
                raise DenseVisualError(f"{name} 必须是非负有限数，收到 {value!r}")
        if not isinstance(self.version, str) or not self.version.strip():
            raise DenseVisualError("DenseVisualFeatures.version 不能为空")
        if self.size and (self.size < 2 or self.size % 1 != 0):
            raise DenseVisualError(f"size 非法: {self.size!r}")
        object.__setattr__(self, "fps", fps_value)
        object.__setattr__(self, "per_frame", per_frame.astype(np.float32, copy=False))


# ──────────────────────────────────────────────────────────────────────────────
# 全流程：解码 + 编码
# ──────────────────────────────────────────────────────────────────────────────


def extract_dense_features(
    source: str | Path,
    encoder: VisualEncoder,
    *,
    fps: float = DEFAULT_FPS,
    size: int = DEFAULT_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    ffmpeg: str = "ffmpeg",
    timeout_s: float = 600.0,
) -> DenseVisualFeatures:
    """整部视频 -> 帧级视觉语义向量序列（流式解码 + 冻结编码器）。

    输出 per_frame 逐行 L2 归一化；decode_s 为累计解码（含管道读取）耗时，
    encode_s 为累计编码器前向耗时，model_load_s 取自编码器懒加载耗时。
    """

    src = Path(source)
    if not src.is_file():
        raise DenseVisualError(f"视频文件不存在: {src}")
    fps_value = _require_positive_number(fps, what="fps")
    size_value = _require_positive_int(size, what="size", minimum=2)
    batch_value = _require_positive_int(batch_size, what="batch_size")
    timeout = _require_positive_number(timeout_s, what="timeout_s")
    if not callable(getattr(encoder, "encode", None)):
        raise DenseVisualError("encoder 缺少可调用的 encode(frames)")
    if not isinstance(getattr(encoder, "version", None), str):
        raise DenseVisualError("encoder 缺少字符串 version 属性")

    duration_s = _probe_duration_s(src, ffmpeg=ffmpeg, timeout_s=timeout)
    batches = iter_frame_batches(
        src,
        fps=fps_value,
        size=size_value,
        batch_size=batch_value,
        ffmpeg=ffmpeg,
        timeout_s=timeout,
    )
    vectors: list[np.ndarray] = []
    n_frames = 0
    decode_s = 0.0
    encode_s = 0.0
    try:
        while True:
            started = time.perf_counter()
            try:
                batch = next(batches)
            except StopIteration:
                decode_s += time.perf_counter() - started
                break
            decode_s += time.perf_counter() - started
            started = time.perf_counter()
            try:
                encoded = encoder.encode(batch)
            except DenseVisualError:
                raise
            except Exception as exc:  # noqa: BLE001 - 统一包装第三方异常
                raise DenseVisualError(
                    f"视觉编码失败（{type(encoder).__name__}）：{exc}"
                ) from exc
            encode_s += time.perf_counter() - started
            matrix = _as_feature_matrix(
                encoded, expected_rows=int(batch.shape[0]), what="视觉编码器输出"
            )
            declared_dim = int(getattr(encoder, "dim", 0) or 0)
            if declared_dim > 0 and int(matrix.shape[1]) != declared_dim:
                raise DenseVisualError(
                    f"编码器输出维度 {matrix.shape[1]} 与 encoder.dim={declared_dim} 不一致"
                )
            vectors.append(matrix)
            n_frames += int(matrix.shape[0])
    finally:
        close = getattr(batches, "close", None)
        if callable(close):
            close()
    if not vectors or n_frames < 1:
        raise DenseVisualError(f"没有可用的帧级向量: {src}")

    per_frame = _normalize_rows(np.concatenate(vectors, axis=0), what="帧级向量")
    dim = int(per_frame.shape[1])
    feature_version = compress_version(str(encoder.version), fps_value, size_value)
    return DenseVisualFeatures(
        per_frame=per_frame,
        fps=fps_value,
        duration_s=duration_s,
        n_frames=n_frames,
        dim=dim,
        version=feature_version,
        decode_s=decode_s,
        encode_s=encode_s,
        model_load_s=float(getattr(encoder, "model_load_s", 0.0) or 0.0),
        encoder=str(encoder.version),
        revision=getattr(encoder, "revision", None),
        size=size_value,
        created_at=now_utc_iso(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 窗口池化
# ──────────────────────────────────────────────────────────────────────────────


def _window_overlaps(
    features: DenseVisualFeatures, windows: Sequence[tuple[float, float]]
) -> tuple[np.ndarray, list[tuple[int, int, np.ndarray, float]]]:
    """把窗口换算成帧区间与逐帧权重（重叠时长）。"""

    if not isinstance(features, DenseVisualFeatures):
        raise DenseVisualError("pool_windows 需要 DenseVisualFeatures")
    if not windows:
        raise DenseVisualError("windows 不能为空")
    frame = np.asarray(features.per_frame, dtype=np.float64)
    n_frames = int(frame.shape[0])
    fps_value = float(features.fps)
    frame_starts = np.arange(n_frames, dtype=np.float64) / fps_value
    frame_ends = (np.arange(n_frames, dtype=np.float64) + 1.0) / fps_value
    duration_s = float(features.duration_s)
    prepared: list[tuple[int, int, np.ndarray, float]] = []
    for index, item in enumerate(windows):
        try:
            start, end = item
        except (TypeError, ValueError) as exc:
            raise DenseVisualError(f"窗口 {index} 必须是 (start, end) 二元组") from exc
        if isinstance(start, bool) or isinstance(end, bool):
            raise DenseVisualError(f"窗口 {index} 的 start/end 必须是数字")
        start_value = float(start)
        end_value = float(end)
        if not math.isfinite(start_value) or not math.isfinite(end_value):
            raise DenseVisualError(f"窗口 {index} 的 start/end 必须有限")
        if start_value < 0 or end_value <= start_value:
            raise DenseVisualError(
                f"窗口 {index} 非法: start={start_value!r} end={end_value!r}"
            )
        if end_value > duration_s + 1.0 / fps_value + 1e-6:
            raise DenseVisualError(
                f"窗口 {index} 超出视频时长: end={end_value:.3f}s duration={duration_s:.3f}s"
            )
        first = max(0, int(math.floor(start_value * fps_value)))
        last = min(n_frames, int(math.ceil(end_value * fps_value)))
        if last <= first:
            raise DenseVisualError(
                f"窗口 {index} [{start_value:.3f}, {end_value:.3f}] 没有覆盖任何帧"
            )
        overlaps = np.clip(
            np.minimum(frame_ends[first:last], end_value)
            - np.maximum(frame_starts[first:last], start_value),
            0.0,
            None,
        )
        total = float(overlaps.sum())
        if total <= _EPS:
            raise DenseVisualError(
                f"窗口 {index} [{start_value:.3f}, {end_value:.3f}] 没有有效帧重叠"
            )
        prepared.append((first, last, overlaps, total))
    return frame, prepared


def pool_windows(
    features: DenseVisualFeatures, windows: Sequence[tuple[float, float]]
) -> np.ndarray:
    """按窗口与帧的重叠时长加权平均帧向量，再逐行 L2 归一化。

    返回 (T,D) float32；T = len(windows)，窗口顺序与输入一致。
    窗口覆盖不到任何帧、越界或参数非法时抛 DenseVisualError。
    """

    frame, prepared = _window_overlaps(features, windows)
    pooled = np.zeros((len(prepared), frame.shape[1]), dtype=np.float64)
    for row, (first, last, overlaps, total) in enumerate(prepared):
        pooled[row] = (overlaps[:, None] * frame[first:last]).sum(axis=0) / total
    return _normalize_rows(pooled, what="窗口池化向量")


def pool_windows_std(
    features: DenseVisualFeatures, windows: Sequence[tuple[float, float]]
) -> np.ndarray:
    """窗口内逐维加权标准差（总体标准差），返回 (T,D) float32。

    可作为窗口内的变化量特征；与 pool_windows 使用同一套重叠时长权重。
    """

    frame, prepared = _window_overlaps(features, windows)
    result = np.zeros((len(prepared), frame.shape[1]), dtype=np.float32)
    for row, (first, last, overlaps, total) in enumerate(prepared):
        slice_frame = frame[first:last]
        mean = (overlaps[:, None] * slice_frame).sum(axis=0) / total
        variance = (
            overlaps[:, None] * (slice_frame - mean) ** 2
        ).sum(axis=0) / total
        result[row] = np.sqrt(np.clip(variance, 0.0, None)).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise DenseVisualError("窗口标准差含 NaN/Inf")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 保存 / 读取（fp16 npz + JSON manifest，原子写）
# ──────────────────────────────────────────────────────────────────────────────


def _validate_features_for_save(features: DenseVisualFeatures) -> DenseVisualFeatures:
    if not isinstance(features, DenseVisualFeatures):
        raise DenseVisualError(
            f"需要 DenseVisualFeatures，收到 {type(features).__name__}"
        )
    return features


def _atomic_write_bytes(path: Path, writer: Any) -> None:
    """用同目录临时文件写内容再 os.replace（writer 接收文件对象）。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
        os.replace(tmp_path, target)
    except OSError as exc:
        raise DenseVisualError(f"写入文件失败（{target}）：{exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


def _manifest_payload(features: DenseVisualFeatures) -> dict[str, Any]:
    created_at = features.created_at or now_utc_iso()
    return {
        "version": features.version,
        "encoder": features.encoder,
        "revision": features.revision,
        "fps": float(features.fps),
        "size": int(features.size),
        "duration_s": float(features.duration_s),
        "n_frames": int(features.n_frames),
        "dim": int(features.dim),
        "created_at": created_at,
        "dtype": "float16",
        "decode_s": float(features.decode_s),
        "encode_s": float(features.encode_s),
        "model_load_s": float(features.model_load_s),
        "files": {
            "per_frame": DENSE_VISUAL_NPZ_NAME,
            "manifest": DENSE_VISUAL_MANIFEST_NAME,
        },
    }


def save_features(
    features: DenseVisualFeatures, out_dir: str | Path
) -> dict[str, Path]:
    """把密集视觉序列原子写入 out_dir：dense_visual.npz + dense_visual_manifest.json。

    per_frame 以 float16 保存；返回 {"npz": Path, "manifest": Path}。
    """

    data = _validate_features_for_save(features)
    target = Path(out_dir)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DenseVisualError(f"无法创建输出目录 {target}：{exc}") from exc

    arrays = {
        "per_frame": np.asarray(data.per_frame, dtype=np.float16),
        "fps": np.float64(data.fps),
        "duration_s": np.float64(data.duration_s),
        "n_frames": np.int64(data.n_frames),
        "dim": np.int64(data.dim),
        "version": np.asarray(data.version),
    }
    npz_path = target / DENSE_VISUAL_NPZ_NAME
    manifest_path = target / DENSE_VISUAL_MANIFEST_NAME

    def _write_npz(handle: Any) -> None:
        np.savez_compressed(handle, **arrays)

    _atomic_write_bytes(npz_path, _write_npz)
    manifest_text = json.dumps(
        _manifest_payload(data), ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    _atomic_write_bytes(
        manifest_path, lambda handle: handle.write(manifest_text.encode("utf-8"))
    )
    logger.info(
        "密集视觉序列已保存: %s（%d 帧，dim=%d，版本 %s）",
        target,
        data.n_frames,
        data.dim,
        data.version,
    )
    return {"npz": npz_path, "manifest": manifest_path}


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DenseVisualError(f"密集视觉 manifest 无法读取（{path}）：{exc}") from exc
    if not isinstance(payload, dict):
        raise DenseVisualError(f"密集视觉 manifest 顶层必须是对象（{path}）")
    return payload


def save_dense_visual(
    features: DenseVisualFeatures, out_dir: str | Path
) -> dict[str, Path]:
    """save_features 的语义化别名（save/load npz 契约）。"""

    return save_features(features, out_dir)


def load_dense_visual(out_dir: str | Path) -> DenseVisualFeatures | None:
    """load_features 的语义化别名（save/load npz 契约）。"""

    return load_features(out_dir)


def load_features(out_dir: str | Path) -> DenseVisualFeatures | None:
    """读取 dense_visual.npz + manifest。

    两个文件任一缺失返回 None；结构损坏 / 版本或尺寸不一致抛 DenseVisualError。
    per_frame 以 float32 返回（写入时为 float16）。
    """

    target = Path(out_dir)
    if not target.is_dir():
        return None
    npz_path = target / DENSE_VISUAL_NPZ_NAME
    manifest_path = target / DENSE_VISUAL_MANIFEST_NAME
    if not npz_path.is_file() or not manifest_path.is_file():
        return None
    manifest = _load_manifest(manifest_path)
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            per_frame = np.asarray(data["per_frame"], dtype=np.float32)
            fps = float(np.asarray(data["fps"]).item())
            duration_s = float(np.asarray(data["duration_s"]).item())
            n_frames = int(np.asarray(data["n_frames"]).item())
            dim = int(np.asarray(data["dim"]).item())
            version = str(np.asarray(data["version"]).item())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DenseVisualError(f"密集视觉 npz 读取失败（{npz_path}）：{exc}") from exc

    if n_frames != int(per_frame.shape[0]) or dim != int(per_frame.shape[1]):
        raise DenseVisualError(
            f"npz 内部不一致: n_frames={n_frames} dim={dim} 实际 {per_frame.shape}"
        )
    if not (math.isfinite(fps) and fps > 0 and math.isfinite(duration_s) and duration_s > 0):
        raise DenseVisualError(f"npz 中的 fps/duration 非法: fps={fps} duration={duration_s}")
    if str(manifest.get("version")) != version:
        raise DenseVisualError(
            f"manifest 与 npz 版本不一致: {manifest.get('version')!r} != {version!r}"
        )
    if int(manifest.get("n_frames", -1)) != n_frames:
        raise DenseVisualError("manifest 与 npz 的 n_frames 不一致")
    if int(manifest.get("dim", -1)) != dim:
        raise DenseVisualError("manifest 与 npz 的 dim 不一致")
    try:
        fps_manifest = float(manifest["fps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DenseVisualError(f"manifest fps 非法: {exc}") from exc
    if abs(fps_manifest - fps) > 1e-9:
        raise DenseVisualError("manifest 与 npz 的 fps 不一致")

    try:
        return DenseVisualFeatures(
            per_frame=per_frame,
            fps=fps,
            duration_s=duration_s,
            n_frames=n_frames,
            dim=dim,
            version=version,
            decode_s=float(manifest.get("decode_s", 0.0)),
            encode_s=float(manifest.get("encode_s", 0.0)),
            model_load_s=float(manifest.get("model_load_s", 0.0)),
            encoder=str(manifest.get("encoder", "")),
            revision=(
                str(manifest["revision"])
                if manifest.get("revision") is not None
                else None
            ),
            size=int(manifest.get("size", 0)),
            created_at=str(manifest.get("created_at", "")),
        )
    except (DenseVisualError, KeyError, TypeError, ValueError) as exc:
        raise DenseVisualError(f"密集视觉 manifest 字段非法（{manifest_path}）：{exc}") from exc
