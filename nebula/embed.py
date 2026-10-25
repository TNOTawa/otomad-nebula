"""PE-AV 音画联合向量提取、片段聚合与向量文件读写。

设计要点：

- 模块导入时不加载 torch / transformers；PEAVEmbedder 只有在第一次调用
  encode_pairs 时才懒加载 processor 与模型，因此默认 pytest 与全链路测试
  不会真的加载权重、也不会访问网络。
- HashTestEmbedder 只依赖 hashlib + numpy，从（视频, 音频）文件内容确定性
  地生成向量，供离线测试与演示使用，不依赖 torch。
- 所有输出向量都是 float32、CPU、逐行 L2 归一化；作品级向量按
  “逐行归一化 -> 求平均 -> 归一化”聚合，与方案文档一致。

数值精度说明：向量数值会受运行设备与 dtype 影响。本机 CUDA 上默认使用
bfloat16 autocast（见 config.yaml 的 model.device/dtype=auto），与 float32
相比同一片段的联合向量 cosine 约 0.996 量级、逐元素最大差约 1e-2。因此
跨硬件 / 跨 dtype 的向量对比应视为趋势而非 bit 级一致；同一配置（同一
机器、同一 dtype）下结果是确定的：同一输入重复编码的逐元素最大绝对差为 0。

transformers 5.20.2 workaround（必须保留；升级 transformers 后需重新验证）：

    该版本的 PeAudioVideoConfig.from_pretrained 在嵌套 TimmWrapperConfig 的
    from_dict 路径上没有把 config.json 里的 num_classes(1024) 转成 num_labels，
    视觉编码器 head/proj 会被初始化成 2 类，与检查点权重形状不匹配。
    这里显式按 num_classes 重建 num_labels / id2label / label2id，再把修好的
    config 传给 PeAudioVideoModel.from_pretrained。
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from .config import ModelConfig
from .errors import EmbeddingError

logger = logging.getLogger(__name__)

HASH_TEST_VERSION = "test-hash-v1"

# 与 config.yaml 的 media.audio_sample_rate 保持一致；PE-AV 特征提取器要求 48k
_AUDIO_SAMPLE_RATE = 48000

_ZERO_NORM_EPS = 1e-12
_FILE_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class EmbeddingBatch:
    """一批片段的向量。

    joint / audio / video 的形状都是 (n, dim)，dtype 为 float32，逐行 L2 归一化；
    模型不提供单模态向量时对应字段为 None。
    """

    joint: np.ndarray
    audio: np.ndarray | None = None
    video: np.ndarray | None = None


@runtime_checkable
class Embedder(Protocol):
    """嵌入器统一接口；流水线只依赖本协议，便于注入测试用实现。"""

    version: str
    is_test_only: bool

    def encode_pairs(self, pairs: Sequence[tuple[Path, Path]]) -> EmbeddingBatch:
        """编码 (视频文件, 音频文件) 片段对，返回逐行归一化的向量批次。"""
        ...

    def close(self) -> None:
        """释放模型 / 显存等资源；允许重复调用。"""
        ...


def _normalize_rows(matrix: np.ndarray, *, what: str) -> np.ndarray:
    """把二维矩阵逐行做 L2 归一化，返回 float32（不修改输入数组）。"""

    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2:
        raise EmbeddingError(f"{what}必须是二维矩阵 (n, dim)，收到形状 {array.shape}")
    if array.shape[0] == 0:
        raise EmbeddingError(f"{what}为空（0 行），无法归一化")
    if not np.all(np.isfinite(array)):
        raise EmbeddingError(f"{what}包含 NaN/Inf，无法归一化")
    norms = np.linalg.norm(array, axis=1)
    bad = (~np.isfinite(norms)) | (norms <= _ZERO_NORM_EPS)
    if np.any(bad):
        row = int(np.flatnonzero(bad)[0])
        raise EmbeddingError(f"{what}第 {row} 行的范数为 0 或非有限，无法归一化")
    result = array / norms[:, None]
    return np.ascontiguousarray(result, dtype=np.float32)


def _resolve_hub_cache_dir(cache_dir: Path) -> Path:
    """把配置中的模型缓存根目录解析成 huggingface_hub 的缓存目录。

    config.yaml 的 .cache/hf 采用 HF_HOME 布局（缓存实际位于 .cache/hf/hub），
    而 from_pretrained(cache_dir=...) 需要直接指向包含 models--* 的目录，
    所以存在 <cache_dir>/hub 时使用它，否则原样返回。
    """

    root = Path(cache_dir)
    hub = root / "hub"
    if hub.is_dir():
        return hub
    return root


def _patch_vision_num_labels(config: Any) -> None:
    """修正 PE-AV 嵌套视觉配置的 num_labels（transformers 5.20.2 workaround）。"""

    try:
        vision = config.audio_video_config.video_config.vision_config
    except AttributeError:
        logger.warning(
            "PE-AV 配置缺少 audio_video_config.video_config.vision_config，跳过 num_labels 修正"
        )
        return
    num_classes = getattr(vision, "num_classes", None)
    current = getattr(vision, "num_labels", 0)
    try:
        current_int = int(current)
    except (TypeError, ValueError):
        current_int = 0
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0:
        return
    if num_classes == current_int:
        return
    vision.num_labels = int(num_classes)
    vision.id2label = {i: f"LABEL_{i}" for i in range(num_classes)}
    vision.label2id = {f"LABEL_{i}": i for i in range(num_classes)}
    logger.info("PE-AV workaround: 视觉头 num_labels %s -> %s", current_int, num_classes)


def _file_digest(path: Path) -> bytes:
    """流式计算文件内容的 BLAKE2b 摘要（用于确定性测试向量）。"""

    if not path.is_file():
        raise EmbeddingError(f"文件不存在或不是普通文件: {path}")
    digest = hashlib.blake2b(digest_size=32)
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(_FILE_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EmbeddingError(f"读取文件失败（{path}）：{exc}") from exc
    return digest.digest()


def _hash_vector(domain: bytes, digests: tuple[bytes, ...], dim: int) -> np.ndarray:
    """由域前缀 + 文件内容摘要确定性地生成单位向量。"""

    seed = hashlib.sha256(domain)
    for digest in digests:
        seed.update(digest)
    seed_bytes = seed.digest()
    raw = bytearray()
    counter = 0
    while len(raw) < dim * 4:
        raw.extend(hashlib.sha256(seed_bytes + counter.to_bytes(4, "big")).digest())
        counter += 1
    integers = np.frombuffer(bytes(raw[: dim * 4]), dtype=">u4").astype(np.float64)
    values = (integers / float(2**31) - 1.0).astype(np.float32)
    row = _normalize_rows(values[None, :], what="哈希测试向量")
    return row[0]


class PEAVEmbedder:
    """PE-AV 多模态嵌入器：懒加载模型，按配置的 batch_size 分批编码片段对。"""

    is_test_only = False

    def __init__(self, model: ModelConfig) -> None:
        if not isinstance(model, ModelConfig):
            raise EmbeddingError(f"PEAVEmbedder 需要 ModelConfig，收到: {type(model).__name__}")
        self._config = model
        self.version = model.version
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._device = "cpu"
        self._use_autocast = False
        self._loaded = False
        self._batch_size = max(1, int(model.batch_size))

    @staticmethod
    def _resolve_device(device: str, torch: Any) -> str:
        """按配置选择运行设备；auto 优先 CUDA。"""

        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise EmbeddingError(
                "config.yaml 配置了 model.device=cuda，但当前环境没有可用的 CUDA 设备"
            )
        if device not in ("cpu", "cuda"):
            raise EmbeddingError(
                f"model.device 配置非法: {device!r}（只支持 auto/cpu/cuda）"
            )
        return device

    def _ensure_loaded(self) -> None:
        """懒加载 processor 与模型（只执行一次）。"""

        if self._loaded:
            return
        try:
            import torch
            from transformers import PeAudioVideoConfig, PeAudioVideoModel, PeAudioVideoProcessor
        except ImportError as exc:
            raise EmbeddingError(
                f"PE-AV 依赖不可用（需要 torch / transformers / timm）：{exc}"
            ) from exc

        cfg = self._config
        cache_dir = _resolve_hub_cache_dir(cfg.cache_dir)
        load_kwargs: dict[str, Any] = {
            "cache_dir": str(cache_dir),
            "local_files_only": bool(cfg.local_files_only),
        }
        if cfg.revision:
            load_kwargs["revision"] = str(cfg.revision)
        logger.info(
            "加载 PE-AV 模型 %s（cache_dir=%s, local_files_only=%s）",
            cfg.checkpoint,
            cache_dir,
            cfg.local_files_only,
        )
        try:
            hf_config = PeAudioVideoConfig.from_pretrained(cfg.checkpoint, **load_kwargs)
            _patch_vision_num_labels(hf_config)
            processor = PeAudioVideoProcessor.from_pretrained(cfg.checkpoint, **load_kwargs)
            model = PeAudioVideoModel.from_pretrained(cfg.checkpoint, config=hf_config, **load_kwargs)
        except Exception as exc:
            raise EmbeddingError(f"加载 PE-AV 模型失败（{cfg.checkpoint}）：{exc}") from exc

        device = self._resolve_device(cfg.device, torch)
        use_autocast = False
        try:
            model.eval()
            if device == "cpu":
                model = model.float()
            elif cfg.dtype in ("auto", "bfloat16"):
                use_autocast = True
            model = model.to(device)
        except (RuntimeError, ValueError) as exc:
            raise EmbeddingError(f"PE-AV 模型无法移动到设备 {device}: {exc}") from exc

        self._processor = processor
        self._model = model
        self._torch = torch
        self._device = device
        self._use_autocast = use_autocast
        self._loaded = True
        logger.info(
            "PE-AV 模型加载完成：device=%s dtype=%s",
            device,
            "bfloat16(autocast)" if use_autocast else "float32",
        )

    def _extract_embeds(
        self, outputs: Any, name: str, *, expected_rows: int
    ) -> np.ndarray | None:
        """从模型输出里取指定字段，转 float32 CPU 并逐行归一化。"""

        tensor = getattr(outputs, name, None)
        if tensor is None:
            return None
        torch = self._torch
        array = tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2:
            raise EmbeddingError(f"PE-AV 输出 {name} 形状异常: {array.shape}（期望 (n, dim)）")
        if array.shape[0] != expected_rows:
            raise EmbeddingError(
                f"PE-AV 输出 {name} 行数异常: {array.shape[0]}（期望 {expected_rows}）"
            )
        return _normalize_rows(array, what=f"PE-AV 输出 {name} ")

    def _encode_chunk(
        self, chunk: Sequence[tuple[Path, Path]]
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """编码一个 batch 的片段对，返回（联合, 音频, 视频）归一化向量。"""

        torch = self._torch
        try:
            inputs = self._processor(
                videos=[str(video) for video, _ in chunk],
                audio=[str(audio) for _, audio in chunk],
                sampling_rate=_AUDIO_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
        except Exception as exc:
            raise EmbeddingError(f"PE-AV 预处理失败（{len(chunk)} 个片段）：{exc}") from exc
        model_inputs = {
            key: value.to(self._device) if torch.is_tensor(value) else value
            for key, value in dict(inputs).items()
        }
        try:
            with torch.inference_mode():
                if self._use_autocast:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = self._model(**model_inputs)
                else:
                    outputs = self._model(**model_inputs)
        except Exception as exc:
            raise EmbeddingError(f"PE-AV 推理失败（{len(chunk)} 个片段）：{exc}") from exc

        joint = self._extract_embeds(outputs, "audio_video_embeds", expected_rows=len(chunk))
        if joint is None:
            raise EmbeddingError("PE-AV 模型没有返回 audio_video_embeds（联合向量）")
        audio = self._extract_embeds(outputs, "audio_embeds", expected_rows=len(chunk))
        video = self._extract_embeds(outputs, "video_embeds", expected_rows=len(chunk))
        return joint, audio, video

    def encode_pairs(self, pairs: Sequence[tuple[Path, Path]]) -> EmbeddingBatch:
        """分批编码 (视频文件, 音频文件) 片段对。

        参数
        ----
        pairs:
            片段对序列；每个元素是（视频路径, 音频路径）。

        返回
        ----
        EmbeddingBatch:
            joint 为 (n, dim) float32 归一化向量；模型提供单模态向量时
            audio / video 也为 (n, dim)，否则为 None。

        异常
        ----
        EmbeddingError:
            片段为空、文件不存在、模型加载或推理失败、输出形状异常时抛出。
        """

        if not pairs:
            raise EmbeddingError("encode_pairs 收到空列表，没有可编码的片段")
        normalized_pairs: list[tuple[Path, Path]] = []
        for item in pairs:
            try:
                video, audio = item
            except (TypeError, ValueError) as exc:
                raise EmbeddingError(
                    f"片段对格式非法（需要 (视频, 音频) 二元组）：{item!r}"
                ) from exc
            video_path = Path(video)
            audio_path = Path(audio)
            if not video_path.is_file():
                raise EmbeddingError(f"片段视频文件不存在: {video_path}")
            if not audio_path.is_file():
                raise EmbeddingError(f"片段音频文件不存在: {audio_path}")
            normalized_pairs.append((video_path, audio_path))

        self._ensure_loaded()

        joint_parts: list[np.ndarray] = []
        audio_parts: list[np.ndarray] = []
        video_parts: list[np.ndarray] = []
        for offset in range(0, len(normalized_pairs), self._batch_size):
            chunk = normalized_pairs[offset : offset + self._batch_size]
            joint, audio, video = self._encode_chunk(chunk)
            joint_parts.append(joint)
            if audio is not None:
                audio_parts.append(audio)
            if video is not None:
                video_parts.append(video)

        joint_all = np.concatenate(joint_parts, axis=0).astype(np.float32, copy=False)
        audio_all = (
            np.concatenate(audio_parts, axis=0).astype(np.float32, copy=False)
            if audio_parts
            else None
        )
        video_all = (
            np.concatenate(video_parts, axis=0).astype(np.float32, copy=False)
            if video_parts
            else None
        )
        for name, array in (("audio_embeds", audio_all), ("video_embeds", video_all)):
            if array is not None and array.shape[0] != joint_all.shape[0]:
                raise EmbeddingError(
                    f"PE-AV 单模态向量 {name} 数量与联合向量不一致: "
                    f"{array.shape[0]} != {joint_all.shape[0]}"
                )
        return EmbeddingBatch(joint=joint_all, audio=audio_all, video=video_all)

    def close(self) -> None:
        """释放模型与显存；允许重复调用。"""

        was_loaded = self._loaded
        model = self._model
        self._processor = None
        self._model = None
        self._torch = None
        self._loaded = False
        if not was_loaded:
            return
        del model
        try:
            import torch
        except ImportError:  # pragma: no cover - 已加载过模型时必然可用
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (RuntimeError, AssertionError):  # pragma: no cover - 仅在 CUDA 环境
            logger.debug("清理 CUDA 显存缓存失败", exc_info=True)


class HashTestEmbedder:
    """仅测试 / 演示用的确定性嵌入器：不依赖 torch、不访问网络。

    视频与音频文件的内容各算一个内容摘要，再用带域前缀的 SHA-256 派生
    float32 向量并做 L2 归一化。相同文件内容永远得到相同向量；联合向量
    同时依赖视频与音频摘要，因此可以模拟“音画联合”的行为，用于离线
    全链路测试与消融实验的接口验证。
    """

    version = HASH_TEST_VERSION
    is_test_only = True

    def __init__(self, dim: int = 32) -> None:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
            raise EmbeddingError(f"HashTestEmbedder 维度必须为正整数，收到: {dim!r}")
        self.dim = dim

    def encode_pairs(self, pairs: Sequence[tuple[Path, Path]]) -> EmbeddingBatch:
        """从文件内容确定性地生成 joint / audio / video 三组向量。"""

        if not pairs:
            raise EmbeddingError("encode_pairs 收到空列表，没有可编码的片段")
        video_hashes: dict[Path, bytes] = {}
        audio_hashes: dict[Path, bytes] = {}
        joint_rows: list[np.ndarray] = []
        audio_rows: list[np.ndarray] = []
        video_rows: list[np.ndarray] = []
        for item in pairs:
            try:
                video, audio = item
            except (TypeError, ValueError) as exc:
                raise EmbeddingError(
                    f"片段对格式非法（需要 (视频, 音频) 二元组）：{item!r}"
                ) from exc
            video_path = Path(video)
            audio_path = Path(audio)
            if video_path not in video_hashes:
                video_hashes[video_path] = _file_digest(video_path)
            if audio_path not in audio_hashes:
                audio_hashes[audio_path] = _file_digest(audio_path)
            video_digest = video_hashes[video_path]
            audio_digest = audio_hashes[audio_path]
            joint_rows.append(_hash_vector(b"joint", (video_digest, audio_digest), self.dim))
            audio_rows.append(_hash_vector(b"audio", (audio_digest,), self.dim))
            video_rows.append(_hash_vector(b"video", (video_digest,), self.dim))
        return EmbeddingBatch(
            joint=np.stack(joint_rows).astype(np.float32, copy=False),
            audio=np.stack(audio_rows).astype(np.float32, copy=False),
            video=np.stack(video_rows).astype(np.float32, copy=False),
        )

    def close(self) -> None:
        """无资源需要释放（保持接口一致）。"""


def aggregate_work_vector(clip_vectors: np.ndarray) -> np.ndarray:
    """把片段向量聚合为作品向量：逐行归一化 -> 求平均 -> 归一化。

    参数
    ----
    clip_vectors:
        (n, dim) 片段向量矩阵；已归一化时本操作幂等。

    返回
    ----
    np.ndarray:
        (dim,) float32 单位向量。

    异常
    ----
    EmbeddingError:
        输入不是二维非空矩阵、含 NaN/Inf、任意行范数为 0，或平均后范数为 0 时抛出。
    """

    try:
        array = np.asarray(clip_vectors, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise EmbeddingError(f"片段向量无法转换为 float32 矩阵：{exc}") from exc
    if array.ndim != 2:
        raise EmbeddingError(f"片段向量必须是二维矩阵 (n, dim)，收到形状 {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise EmbeddingError(f"片段向量矩阵为空: {array.shape}")
    normalized = _normalize_rows(array, what="片段向量")
    mean = normalized.mean(axis=0, dtype=np.float32)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(norm) or norm <= _ZERO_NORM_EPS:
        raise EmbeddingError("片段向量均值范数为 0，无法生成作品向量")
    return np.ascontiguousarray(mean / norm, dtype=np.float32)


def save_vector(path: Path, array: np.ndarray) -> None:
    """原子写入 .npy 向量文件（先写同目录临时文件，再 os.replace）。

    这样即使进程中途被杀，也不会留下半个向量文件被后续流程误读。
    """

    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise EmbeddingError(f"无法创建向量输出目录 {target.parent}：{exc}") from exc
    payload = np.asarray(array)
    tmp_path: Path | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp.npy", dir=str(target.parent)
        )
        tmp_path = Path(tmp_name)
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, payload, allow_pickle=False)
        os.replace(tmp_path, target)
    except (OSError, ValueError) as exc:
        raise EmbeddingError(f"写入向量文件失败（{target}）：{exc}") from exc
    finally:
        if tmp_path is not None:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:  # pragma: no cover - 清理失败不改变主结果
                logger.warning("无法删除向量临时文件: %s", tmp_path)


def load_vector(path: Path) -> np.ndarray:
    """读取 .npy 向量文件；文件缺失或内容损坏时抛 EmbeddingError。"""

    source = Path(path)
    if not source.is_file():
        raise EmbeddingError(f"向量文件不存在: {source}")
    try:
        return np.load(source, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise EmbeddingError(f"读取向量文件失败（{source}）：{exc}") from exc
