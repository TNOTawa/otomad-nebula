"""v0.4.3 正式密集视觉批处理流水线（canonical / 下载一次 / 状态 / 断点续跑 / 清理）。

职责：把 canonical 且 completed 的作品批量跑通冻结视觉编码器的密集视觉语义通道：

- 主任务：每个配置编码器（CLIP / DINOv2 等）× visual_dense.fps（默认 8fps）；
- 子集任务：动态选取 visual_dense.subset_count 部作品，额外跑 subset_fps（默认 30fps）；
- 每部作品只下载一次：本地作品用原文件；Bilibili 先复用 temp/work_XXXXXX/download
  下可 probe 的源文件，否则用 download.download_bilibili 下载一次，供该作品所有任务复用；
- 断点续跑：visual_sequences 中该 (work, version) 已 completed 且文件存在则跳过；
  否则先记 in_progress（中断可见），成功后记 completed（帧/窗口路径、数量、维度、耗时），
  失败记 failed(error) 并继续其它任务；
- 清理时机：该作品所有需要的视觉任务都 completed 后才删除它的 temp 目录；失败保留。
- 版本命名：使用 nebula.dense_visual.compress_version 规则（若 t1 新版本支持 preprocess
  参数则自动带上；旧版自动兼容）。
- 30fps 子集：select_thirty_fps_works 基于全片序列相邻 clip_joint 的 1-cos 动态 +
  事件窗口变化，按时长桶（短/中/长）确定性选样。

与 t1 dense_visual 的接口兼容策略：优先使用增强后的
extract_dense_features(..., preprocess=...) / save_bundle(...)；若当前工作区还是
pilot 版（无 preprocess 参数、无 save_bundle），自动回退到旧签名并自行写
windows.npy / window_std.npy / window_times.npy，便于并行开发期间先跑通测试。
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import dense_visual
from .config import Config
from .db import Catalog
from .dense_visual import DenseVisualFeatures
from .errors import NebulaError
from .media import MediaError, probe_media
from .sampling import plan_full_coverage

LOGGER = logging.getLogger(__name__)

WINDOWS_NAME = "windows.npy"
WINDOW_STD_NAME = "window_std.npy"
WINDOW_TIMES_NAME = "window_times.npy"
FRAMES_NAME = "frames.npy"
DEFAULT_DENSE_NPZ_NAME = getattr(dense_visual, "DENSE_VISUAL_NPZ_NAME", "dense_visual.npz")
DEFAULT_DENSE_MANIFEST_NAME = getattr(
    dense_visual, "DENSE_VISUAL_MANIFEST_NAME", "dense_visual_manifest.json"
)
_VIDEO_SUFFIXES = frozenset({".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".m4v"})
_INCOMPLETE_SUFFIXES = frozenset({".part", ".ytdl", ".temp", ".tmp", ".download"})
_ERROR_LIMIT = 2000
_SEMANTIC_WEIGHT = 0.5
_EVENT_WEIGHT = 0.5

EncoderFactory = Callable[[str, str], Any]
Extractor = Callable[..., DenseVisualFeatures]
Downloader = Callable[..., Path]
ProgressCallback = Callable[[str], None]


class VisualPipelineError(NebulaError):
    """密集视觉批处理流水线的可预期错误（配置、版本、窗口等）。"""


@dataclass(frozen=True)
class VisualVersionSpec:
    """一个视觉序列版本（编码器 + 帧率 + 尺寸 + 预处理）。"""

    version: str
    encoder_version: str
    model: str
    revision: str
    fps: float
    size: int
    preprocess: str
    subset: bool

    @property
    def label(self) -> str:
        return "30fps 子集" if self.subset else "主帧率"


@dataclass(frozen=True)
class VisualTask:
    """一部作品需要跑的一个视觉序列版本。"""

    work_id: int
    spec: VisualVersionSpec

    @property
    def version(self) -> str:
        return self.spec.version

    @property
    def fps(self) -> float:
        return self.spec.fps


@dataclass(frozen=True)
class VisualRunSummary:
    """一次批处理的统计结果。"""

    selected_works: int
    subset_works: int
    tasks_total: int
    tasks_completed: int
    tasks_skipped: int
    tasks_failed: int
    downloaded_works: int
    cleaned_temp_dirs: int
    versions: tuple[str, ...]
    elapsed_s: float
    dry_run: bool = False

    def summary_text(self) -> str:
        mode = "dry-run（未写 DB/文件）" if self.dry_run else "已写入"
        lines = [
            f"密集视觉批处理完成（{mode}）：作品 {self.selected_works} 部"
            f"（30fps 子集 {self.subset_works} 部），任务 {self.tasks_total} 个",
            f"  完成 {self.tasks_completed}，跳过（已存在）{self.tasks_skipped}，"
            f"失败 {self.tasks_failed}",
            f"  下载作品 {self.downloaded_works} 部，清理 temp 目录 {self.cleaned_temp_dirs} 个，"
            f"耗时 {self.elapsed_s:.2f}s",
            f"  版本: {', '.join(self.versions) if self.versions else '（无）'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------- 版本与作品选择


def compose_visual_version(
    encoder_version: str, fps: float, size: int, preprocess: str
) -> str:
    """按 dense_visual.compress_version 规则生成视觉版本号，并文件系统安全化。

    compress_version 保留原始 model id/revision（逻辑身份）；视觉数据目录与 DB
    version 使用 visual_version_slug 的 slug 形式，避免 model id 中的 "/" 造成
    多层目录或 Windows 长路径问题。model_id/revision 原值记录在 visual_versions 表，
    所以 slug 化不丢失身份信息。
    """

    try:
        raw = str(
            dense_visual.compress_version(encoder_version, fps, size, preprocess)  # type: ignore[call-arg]
        )
    except TypeError:
        # t1 增强前的 pilot 版 compress_version 只有三个参数
        raw = str(dense_visual.compress_version(encoder_version, fps, size))
    slug_fn = getattr(dense_visual, "visual_version_slug", None)
    return str(slug_fn(raw)) if callable(slug_fn) else raw


def encoder_version_for(model: str, revision: str) -> str:
    return f"{model}@{revision}"


def visual_version_specs(config: Config, *, include_subset: bool) -> list[VisualVersionSpec]:
    """构造当前配置需要的视觉版本列表（主版本 + 可选子集版本）。"""

    dense = config.visual_dense
    specs: list[VisualVersionSpec] = []
    for encoder in dense.encoders:
        encoder_version = encoder_version_for(encoder.model, encoder.revision)
        specs.append(
            VisualVersionSpec(
                version=compose_visual_version(
                    encoder_version, float(dense.fps), int(dense.size), dense.preprocess
                ),
                encoder_version=encoder_version,
                model=encoder.model,
                revision=encoder.revision,
                fps=float(dense.fps),
                size=int(dense.size),
                preprocess=str(dense.preprocess),
                subset=False,
            )
        )
        if include_subset and int(dense.subset_count) > 0:
            specs.append(
                VisualVersionSpec(
                    version=compose_visual_version(
                        encoder_version, float(dense.subset_fps), int(dense.size), dense.preprocess
                    ),
                    encoder_version=encoder_version,
                    model=encoder.model,
                    revision=encoder.revision,
                    fps=float(dense.subset_fps),
                    size=int(dense.size),
                    preprocess=str(dense.preprocess),
                    subset=True,
                )
            )
    return specs


def select_canonical_completed_works(catalog: Catalog) -> list[dict[str, Any]]:
    """返回 canonical（排除 work_links duplicate）且 status=completed 的作品。"""

    duplicates = catalog.get_duplicate_work_ids()
    works: list[dict[str, Any]] = []
    for work in catalog.list_works():
        work_id = int(work["id"])
        if work_id in duplicates:
            continue
        if str(work.get("status")) != "completed":
            continue
        if str(work.get("source")) not in ("local", "bilibili"):
            continue
        works.append(work)
    return works


def _minmax_normalize(values: Sequence[float | None]) -> list[float]:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not valid:
        return [0.0 for _ in values]
    low = min(valid)
    high = max(valid)
    span = high - low
    if span <= 1e-12:
        return [0.0 for _ in values]
    normalized = []
    for value in values:
        if value is None or not math.isfinite(float(value)):
            normalized.append(0.0)
        else:
            normalized.append((float(value) - low) / span)
    return normalized


def _semantic_dynamics(config: Config, version: str, work_id: int) -> float | None:
    """相邻 clip_joint 的 1-cos 均值（全片序列动态分数）。"""

    path = config.paths.embeddings / version / f"{work_id:06d}" / "clip_joint.npy"
    if not path.is_file():
        return None
    try:
        matrix = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    except (OSError, ValueError) as exc:
        LOGGER.warning("读取 clip_joint 失败（%s）: %s", path, exc)
        return None
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return 0.0
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-9)
    cosine = np.sum(normalized[:-1] * normalized[1:], axis=1)
    return float(max(0.0, 1.0 - float(np.mean(cosine))))


def _event_dynamics(config: Config, version: str, work_id: int) -> float | None:
    """事件窗口特征相邻变化的平均 L2（按列标准化后，避免量纲差异）。"""

    path = config.paths.embeddings / version / f"{work_id:06d}" / "event_features.npy"
    if not path.is_file():
        return None
    try:
        matrix = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    except (OSError, ValueError) as exc:
        LOGGER.warning("读取 event_features 失败（%s）: %s", path, exc)
        return None
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        return 0.0
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    standardized = (matrix - mean) / np.maximum(std, 1e-9)
    difference = np.diff(standardized, axis=0)
    return float(np.mean(np.linalg.norm(difference, axis=1)))


def _duration_buckets(durations: Sequence[float]) -> list[str]:
    """把候选时长分成短/中/长三个桶（三分位，确定性）。"""

    ordered = sorted(float(value) for value in durations)
    if len(ordered) < 3:
        return ["short" for _ in durations]
    low = ordered[len(ordered) // 3]
    high = ordered[(2 * len(ordered)) // 3]
    labels: list[str] = []
    for value in durations:
        if value <= low:
            labels.append("short")
        elif value <= high:
            labels.append("medium")
        else:
            labels.append("long")
    return labels


def select_thirty_fps_works(catalog: Catalog, config: Config, count: int) -> list[int]:
    """确定性选择 30fps 对照作品（动态分数 + 短/中/长时长桶覆盖）。

    动态分数 = 0.5 * minmax(相邻 clip_joint 1-cos 均值)
             + 0.5 * minmax(事件窗口特征相邻变化的平均 L2)；
    每部作品先归入短/中/长时长桶，按 count 平均分配名额，桶内按分数降序取，
    名额不足时从剩余作品按分数补齐。返回 work_id 升序列表，保证可复现。
    """

    total = int(count)
    if total <= 0:
        return []
    works = select_canonical_completed_works(catalog)
    full_version = config.full_sequence_version
    scored: list[dict[str, Any]] = []
    for work in works:
        duration = work.get("duration_s")
        if duration is None:
            continue
        duration_value = float(duration)
        if not math.isfinite(duration_value) or duration_value <= 0:
            continue
        work_id = int(work["id"])
        scored.append(
            {
                "work_id": work_id,
                "duration": duration_value,
                "semantic": _semantic_dynamics(config, full_version, work_id),
                "event": _event_dynamics(config, full_version, work_id),
            }
        )
    if not scored:
        return []
    semantic_scores = _minmax_normalize([row["semantic"] for row in scored])
    event_scores = _minmax_normalize([row["event"] for row in scored])
    for row, semantic, event in zip(scored, semantic_scores, event_scores, strict=True):
        row["score"] = _SEMANTIC_WEIGHT * semantic + _EVENT_WEIGHT * event

    buckets = _duration_buckets([row["duration"] for row in scored])
    grouped: dict[str, list[dict[str, Any]]] = {"short": [], "medium": [], "long": []}
    for row, bucket in zip(scored, buckets, strict=True):
        grouped[bucket].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: (-row["score"], row["work_id"]))

    base = total // 3
    remainder = total % 3
    quotas = {"short": base, "medium": base, "long": base}
    for index, name in enumerate(("short", "medium", "long")):
        if index < remainder:
            quotas[name] += 1
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    for name in ("short", "medium", "long"):
        candidate_rows = grouped[name][: quotas[name]]
        for row in candidate_rows:
            selected.append(row)
            selected_ids.add(int(row["work_id"]))
    if len(selected) < total:
        remaining = sorted(
            (row for row in scored if int(row["work_id"]) not in selected_ids),
            key=lambda row: (-row["score"], row["work_id"]),
        )
        selected.extend(remaining[: total - len(selected)])
    return sorted(int(row["work_id"]) for row in selected)


def plan_visual_tasks(
    config: Config,
    works: Sequence[Mapping[str, Any]],
    subset_work_ids: Sequence[int],
    *,
    include_subset: bool,
) -> tuple[list[VisualVersionSpec], dict[int, list[VisualTask]]]:
    """为选中的作品生成每个需要执行的任务。"""

    subset_set = set(int(work_id) for work_id in subset_work_ids) if include_subset else set()
    specs = visual_version_specs(config, include_subset=bool(subset_set))
    main_specs = [spec for spec in specs if not spec.subset]
    subset_specs = [spec for spec in specs if spec.subset]
    tasks_by_work: dict[int, list[VisualTask]] = {}
    for work in works:
        work_id = int(work["id"])
        tasks = [VisualTask(work_id=work_id, spec=spec) for spec in main_specs]
        if work_id in subset_set:
            tasks.extend(VisualTask(work_id=work_id, spec=spec) for spec in subset_specs)
        tasks_by_work[work_id] = tasks
    return specs, tasks_by_work


# ---------------------------------------------------------------- dense_visual 兼容适配


def _extract_dense_features(
    source: Path,
    encoder: Any,
    *,
    fps: float,
    size: int,
    batch_size: int,
    ffmpeg: str,
    preprocess: str,
    timeout_s: float,
) -> DenseVisualFeatures:
    """调用 t1 的 extract_dense_features；兼容尚未加入 preprocess 参数的版本。"""

    try:
        return dense_visual.extract_dense_features(
            source,
            encoder,
            fps=fps,
            size=size,
            batch_size=batch_size,
            ffmpeg=ffmpeg,
            preprocess=preprocess,
            timeout_s=timeout_s,
        )
    except TypeError as exc:
        if "preprocess" not in str(exc):
            raise
        LOGGER.warning("当前 dense_visual.extract_dense_features 尚不支持 preprocess，使用兼容调用")
        return dense_visual.extract_dense_features(
            source,
            encoder,
            fps=fps,
            size=size,
            batch_size=batch_size,
            ffmpeg=ffmpeg,
            timeout_s=timeout_s,
        )


def _atomic_save_npy(path: Path, array: np.ndarray) -> Path:
    """原子保存 npy（同目录临时文件 + os.replace）。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "wb") as file_handle:
            np.save(file_handle, np.asarray(array), allow_pickle=False)
        os.replace(tmp_path, target)
    finally:
        tmp_path.unlink(missing_ok=True)
    return target


def _coerce_saved_paths(
    target: Path, result: Any
) -> dict[str, Path]:
    """把 save_bundle/save_features 的返回值整理成 {frames,windows,window_std,window_times}。"""

    paths: dict[str, Path] = {}
    if isinstance(result, Mapping):
        aliases = {
            "frames": ("frames", "npz", "per_frame", "dense_visual"),
            "windows": ("windows", "window_features", "pooled_windows"),
            "window_std": ("window_std", "std", "windows_std"),
            "window_times": ("window_times", "times", "window_bounds"),
        }
        for key, names in aliases.items():
            for name in names:
                value = result.get(name)
                if value:
                    candidate = Path(str(value))
                    paths[key] = candidate if candidate.is_absolute() else target / candidate
                    break
    if "frames" not in paths:
        for name in (DEFAULT_DENSE_NPZ_NAME, FRAMES_NAME):
            candidate = target / name
            if candidate.is_file():
                paths["frames"] = candidate
                break
    if "windows" not in paths:
        candidate = target / WINDOWS_NAME
        if candidate.is_file():
            paths["windows"] = candidate
    if "window_std" not in paths:
        candidate = target / WINDOW_STD_NAME
        if candidate.is_file():
            paths["window_std"] = candidate
    if "window_times" not in paths:
        candidate = target / WINDOW_TIMES_NAME
        if candidate.is_file():
            paths["window_times"] = candidate
    return paths


def _save_dense_bundle(
    out_dir: Path,
    features: DenseVisualFeatures,
    *,
    pooled: np.ndarray,
    window_std: np.ndarray,
    window_times: Sequence[tuple[float, float]],
) -> dict[str, Path]:
    """优先调用 t1 的 save_bundle；旧版 dense_visual 则写文件后补窗口 npy。

    t1 增强版 save_bundle 会把窗口向量 / 标准差 / 窗口时间（作为 (start,end) 序列）
    一起原子写入 bundle（时间写 manifest）；pilot 旧版 save_features 只写帧级 npz，
    这里补写 windows.npy / window_std.npy / window_times.npy 以保持 DB 路径完整。
    """

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    save_bundle = getattr(dense_visual, "save_bundle", None)
    used_save_bundle = callable(save_bundle)
    result: Any = None
    if used_save_bundle:
        result = save_bundle(
            target,
            features,
            windows=np.asarray(pooled, dtype=np.float32),
            window_std=np.asarray(window_std, dtype=np.float32),
            window_times=[(float(start), float(end)) for start, end in window_times],
        )
    else:
        save_features = getattr(dense_visual, "save_features", None)
        if not callable(save_features):
            raise VisualPipelineError("dense_visual 缺少 save_bundle/save_features，无法保存视觉序列")
        result = save_features(features, target)

    paths = _coerce_saved_paths(target, result)
    if "frames" not in paths:
        raise VisualPipelineError(f"视觉序列保存后找不到帧级文件: {target}")
    if "windows" not in paths:
        paths["windows"] = _atomic_save_npy(
            target / WINDOWS_NAME, np.asarray(pooled, dtype=np.float32)
        )
    if "window_std" not in paths:
        paths["window_std"] = _atomic_save_npy(
            target / WINDOW_STD_NAME, np.asarray(window_std, dtype=np.float32)
        )
    if "window_times" not in paths and not used_save_bundle:
        # 增强版 save_bundle 已把窗口时间写入 manifest，无需额外 npy
        paths["window_times"] = _atomic_save_npy(
            target / WINDOW_TIMES_NAME, np.asarray(window_times, dtype=np.float32)
        )
    return paths


# ---------------------------------------------------------------- 源文件（下载一次）


def _default_encoder_factory(config: Config) -> EncoderFactory:
    dense = config.visual_dense

    def factory(model: str, revision: str) -> Any:
        return dense_visual.HFImageEncoder(
            model,
            revision=revision,
            cache_dir=dense.cache_dir,
            device=dense.device,
            dtype=dense.dtype,
        )

    return factory


def _default_downloader(bvid: str, *, part: int, out_dir: Path, download: Any) -> Path:
    from .download import download_bilibili

    return download_bilibili(bvid, part=part, out_dir=out_dir, download=download)


def _find_reusable_source(
    download_dir: Path, bvid: str, part: int, config: Config
) -> Path | None:
    """在 temp 下载目录里找可 probe 的既有源文件（避免重复下载）。"""

    if not download_dir.is_dir():
        return None
    candidates = sorted(
        path
        for path in download_dir.glob(f"{bvid}_p{part}.*")
        if path.is_file()
        and path.suffix.lower() in _VIDEO_SUFFIXES
        and path.suffix.lower() not in _INCOMPLETE_SUFFIXES
    )
    for candidate in candidates:
        try:
            info = probe_media(
                candidate,
                ffprobe=config.media.ffprobe,
                timeout_s=config.media.probe_timeout_seconds,
            )
        except MediaError as exc:
            LOGGER.warning("temp 源文件无法 probe，忽略: %s: %s", candidate, exc)
            continue
        if info.has_video and info.duration_s > 0:
            LOGGER.info("复用 temp 中的源文件: %s", candidate)
            return candidate
    return None


def _resolve_source(
    work: Mapping[str, Any],
    config: Config,
    downloader: Downloader,
) -> tuple[Path, Path | None, bool]:
    """返回 (源文件路径, 作品 temp 根目录或 None, 本次是否真的下载了)。"""

    work_id = int(work["id"])
    source = str(work.get("source"))
    if source == "local":
        rel_path = work.get("local_path")
        if not rel_path:
            raise VisualPipelineError(f"本地作品缺少 local_path: work_id={work_id}")
        path = config.local_work_path(str(rel_path))
        if not path.is_file():
            raise VisualPipelineError(f"本地视频不存在: {path}")
        return path, None, False
    if source == "bilibili":
        bvid = str(work.get("bvid") or "").strip()
        part = int(work.get("part") or 1)
        if not bvid:
            raise VisualPipelineError(f"Bilibili 作品缺少 bvid: work_id={work_id}")
        temp_root = config.paths.temp / f"work_{work_id:06d}"
        download_dir = temp_root / "download"
        reusable = _find_reusable_source(download_dir, bvid, part, config)
        if reusable is not None:
            return reusable, temp_root, False
        LOGGER.info("下载视觉源文件: work_id=%s bvid=%s part=%s", work_id, bvid, part)
        path = Path(
            downloader(bvid, part=part, out_dir=download_dir, download=config.download)
        )
        if not path.is_file():
            raise VisualPipelineError(f"下载后找不到视频文件: {path}")
        return path, temp_root, True
    raise VisualPipelineError(f"未知作品来源: {source!r}（work_id={work_id}）")


def _sequence_files_exist(config: Config, sequence: Mapping[str, Any]) -> bool:
    for key in ("frames_path", "windows_path"):
        relative = sequence.get(key)
        if not relative:
            return False
        path = config.data_path(str(relative))
        if not path.is_file() or path.stat().st_size <= 0:
            return False
    return True


def _error_text(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text[:_ERROR_LIMIT]


# ---------------------------------------------------------------- 主流程


def run_visual_pipeline(
    catalog: Catalog,
    config: Config,
    *,
    limit: int | None = None,
    work_ids: Sequence[int] | None = None,
    include_subset: bool = True,
    dry_run: bool = False,
    encoder_factory: EncoderFactory | None = None,
    extractor: Extractor | None = None,
    downloader: Downloader | None = None,
    progress: ProgressCallback | None = None,
) -> VisualRunSummary:
    """批量生成密集视觉语义序列。

    参数
    ----
    catalog, config:
        编目与配置。
    limit:
        最多处理多少部作品（按 work_id 升序）。
    work_ids:
        只处理指定作品（仍要求 canonical + completed）。
    include_subset:
        是否为 30fps 子集作品追加 subset_fps 任务。
    dry_run:
        只计算作品/任务与版本，不写数据库、不建文件、不下载。
    encoder_factory / extractor / downloader:
        可注入的编码器工厂、提取函数与下载函数（测试用 fake 实现，默认走真实
        dense_visual.HFImageEncoder / extract_dense_features / download_bilibili）。
    progress:
        可选进度回调，接收一行中文状态文本。
    """

    started = time.perf_counter()
    dense = config.visual_dense
    subset_ids = (
        select_thirty_fps_works(catalog, config, int(dense.subset_count))
        if include_subset and int(dense.subset_count) > 0
        else []
    )
    subset_set = set(int(work_id) for work_id in subset_ids)

    works = select_canonical_completed_works(catalog)
    if work_ids:
        allowed = {int(value) for value in work_ids}
        works = [work for work in works if int(work["id"]) in allowed]
    works.sort(key=lambda work: int(work["id"]))
    if limit is not None:
        if int(limit) < 0:
            raise VisualPipelineError(f"limit 不能为负: {limit}")
        works = works[: int(limit)]

    specs, tasks_by_work = plan_visual_tasks(
        config, works, subset_ids, include_subset=include_subset
    )
    tasks_total = sum(len(tasks) for tasks in tasks_by_work.values())
    if dry_run:
        return VisualRunSummary(
            selected_works=len(works),
            subset_works=len(subset_set & {int(work["id"]) for work in works}),
            tasks_total=tasks_total,
            tasks_completed=0,
            tasks_skipped=0,
            tasks_failed=0,
            downloaded_works=0,
            cleaned_temp_dirs=0,
            versions=tuple(spec.version for spec in specs),
            elapsed_s=time.perf_counter() - started,
            dry_run=True,
        )
    if not works:
        return VisualRunSummary(
            selected_works=0,
            subset_works=0,
            tasks_total=0,
            tasks_completed=0,
            tasks_skipped=0,
            tasks_failed=0,
            downloaded_works=0,
            cleaned_temp_dirs=0,
            versions=(),
            elapsed_s=time.perf_counter() - started,
        )

    factory = encoder_factory or _default_encoder_factory(config)
    active_extractor: Extractor = extractor or _extract_dense_features
    active_downloader: Downloader = downloader or _default_downloader
    encoders: dict[tuple[str, str], Any] = {}

    def get_encoder(model: str, revision: str) -> Any:
        key = (model, revision)
        encoder = encoders.get(key)
        if encoder is None:
            encoder = factory(model, revision)
            encoders[key] = encoder
        return encoder

    # 仅登记实际会用到的版本（主版本 + 实际有子集作品时的子集版本）
    used_versions: dict[str, VisualVersionSpec] = {}
    for tasks in tasks_by_work.values():
        for task in tasks:
            used_versions.setdefault(task.version, task.spec)
    for version, spec in sorted(used_versions.items()):
        encoder = get_encoder(spec.model, spec.revision)
        encoder_dim = int(getattr(encoder, "dim", 0) or 0)
        catalog.register_visual_version(
            version,
            model_id=spec.model,
            revision=spec.revision,
            preprocessing=spec.preprocess,
            fps=spec.fps,
            size=spec.size,
            dim=encoder_dim if encoder_dim > 0 else None,
            params={
                "encoder_version": spec.encoder_version,
                "encoder_dim": encoder_dim,
                "preprocess": spec.preprocess,
                "batch_size": int(dense.batch_size),
                "subset": bool(spec.subset),
                "source": "nebula.visual_pipeline",
            },
        )

    completed = 0
    skipped = 0
    failed = 0
    downloaded_works = 0
    cleaned_dirs = 0
    timeout_s = max(300.0, float(config.media.extract_timeout_seconds))

    try:
        for work_index, work in enumerate(works, start=1):
            work_id = int(work["id"])
            tasks = tasks_by_work.get(work_id, [])
            source_path: Path | None = None
            temp_root: Path | None = None
            source_error: BaseException | None = None
            work_downloaded = False
            if progress:
                progress(
                    f"[{work_index}/{len(works)}] work_id={work_id} 任务 {len(tasks)} 个"
                )
            for task in tasks:
                existing = catalog.get_visual_sequence(work_id, version=task.version)
                if (
                    existing is not None
                    and str(existing.get("status")) == "completed"
                    and _sequence_files_exist(config, existing)
                ):
                    skipped += 1
                    continue
                if source_error is not None:
                    catalog.upsert_visual_sequence(
                        work_id,
                        version=task.version,
                        status="failed",
                        error=_error_text(source_error),
                    )
                    failed += 1
                    continue
                if source_path is None:
                    try:
                        source_path, temp_root, work_downloaded = _resolve_source(
                            work, config, active_downloader
                        )
                    except (NebulaError, OSError) as exc:
                        source_error = exc
                        LOGGER.error(
                            "work_id=%s 获取视觉源文件失败: %s", work_id, _error_text(exc)
                        )
                        catalog.upsert_visual_sequence(
                            work_id,
                            version=task.version,
                            status="failed",
                            error=_error_text(exc),
                        )
                        failed += 1
                        continue
                catalog.upsert_visual_sequence(
                    work_id, version=task.version, status="in_progress"
                )
                task_started = time.perf_counter()
                try:
                    encoder = get_encoder(task.spec.model, task.spec.revision)
                    features = active_extractor(
                        source_path,
                        encoder,
                        fps=task.fps,
                        size=task.spec.size,
                        batch_size=int(dense.batch_size),
                        ffmpeg=config.media.ffmpeg,
                        preprocess=task.spec.preprocess,
                        timeout_s=timeout_s,
                    )
                    # bundle manifest 的 version 与 DB/目录版本保持一致（slug 形式）；
                    # 原始 encoder 身份保存在 visual_versions.model_id/revision 与 manifest.encoder。
                    features = replace(features, version=task.version)
                    duration_s = _choose_duration(work, features)
                    windows = _work_windows(config, duration_s)
                    pooled = dense_visual.pool_windows(features, windows)
                    window_std = dense_visual.pool_windows_std(features, windows)
                    out_dir = config.paths.visual_dense / task.version / f"{work_id:06d}"
                    saved = _save_dense_bundle(
                        out_dir,
                        features,
                        pooled=pooled,
                        window_std=window_std,
                        window_times=windows,
                    )
                    total_s = time.perf_counter() - task_started
                    catalog.upsert_visual_sequence(
                        work_id,
                        version=task.version,
                        status="completed",
                        frames_path=config.data_relpath(saved["frames"]),
                        windows_path=config.data_relpath(saved["windows"]),
                        n_frames=int(features.n_frames),
                        n_windows=int(np.asarray(pooled).shape[0]),
                        dim=int(features.dim),
                        decode_s=float(features.decode_s),
                        encode_s=float(features.encode_s),
                        model_load_s=float(getattr(encoder, "model_load_s", 0.0) or 0.0),
                        total_s=total_s,
                    )
                    completed += 1
                    LOGGER.info(
                        "work_id=%s 视觉序列完成: %s（%d 帧 / %d 窗口, %.2fs）",
                        work_id,
                        task.version,
                        int(features.n_frames),
                        int(np.asarray(pooled).shape[0]),
                        total_s,
                    )
                    if progress:
                        progress(
                            f"  work_id={work_id} {task.version}: "
                            f"{features.n_frames} 帧 / {np.asarray(pooled).shape[0]} 窗口"
                        )
                except Exception as exc:  # noqa: BLE001 - 单任务失败不阻塞其它任务
                    failed += 1
                    LOGGER.exception(
                        "work_id=%s 视觉序列失败: %s", work_id, task.version
                    )
                    try:
                        catalog.upsert_visual_sequence(
                            work_id,
                            version=task.version,
                            status="failed",
                            error=_error_text(exc),
                            total_s=time.perf_counter() - task_started,
                        )
                    except NebulaError:  # pragma: no cover - DB 异常只记录
                        LOGGER.exception("记录视觉序列失败状态时出错: work_id=%s", work_id)
            if work_downloaded:
                downloaded_works += 1
            if temp_root is not None and tasks:
                all_done = True
                for task in tasks:
                    row = catalog.get_visual_sequence(work_id, version=task.version)
                    if (
                        row is None
                        or str(row.get("status")) != "completed"
                        or not _sequence_files_exist(config, row)
                    ):
                        all_done = False
                        break
                if all_done and temp_root.is_dir():
                    shutil.rmtree(temp_root, ignore_errors=True)
                    cleaned_dirs += 1
                    LOGGER.info("work_id=%s 全部视觉任务完成，清理 temp: %s", work_id, temp_root)
    finally:
        for encoder in encoders.values():
            close = getattr(encoder, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - 关闭失败不影响主结果
                    LOGGER.warning("关闭视觉编码器失败", exc_info=True)

    return VisualRunSummary(
        selected_works=len(works),
        subset_works=len(subset_set & {int(work["id"]) for work in works}),
        tasks_total=tasks_total,
        tasks_completed=completed,
        tasks_skipped=skipped,
        tasks_failed=failed,
        downloaded_works=downloaded_works,
        cleaned_temp_dirs=cleaned_dirs,
        versions=tuple(sorted(used_versions)),
        elapsed_s=time.perf_counter() - started,
    )


def _choose_duration(work: Mapping[str, Any], features: DenseVisualFeatures) -> float:
    """窗口时长优先用 DB 中 PE-AV 使用的 probe 时长，避免与片段序列错位。"""

    db_duration = work.get("duration_s")
    features_duration = float(features.duration_s)
    if db_duration is None:
        return features_duration
    db_value = float(db_duration)
    tolerance = max(1.0, 1.0 / max(float(features.fps), 1e-9))
    if abs(db_value - features_duration) <= tolerance:
        return db_value
    LOGGER.warning(
        "作品时长不一致（DB=%.3fs, 提取=%.3fs），窗口按提取时长生成",
        db_value,
        features_duration,
    )
    return features_duration


def _work_windows(config: Config, duration_s: float) -> list[tuple[float, float]]:
    plans = plan_full_coverage(
        float(duration_s),
        clip_seconds=float(config.media.clip_seconds),
        overlap_seconds=float(config.sampling.overlap_seconds),
        min_duration_seconds=float(config.media.min_duration_seconds),
    )
    return [(float(plan.start_s), float(plan.end_s)) for plan in plans]


# ---------------------------------------------------------------- 状态 / 清理


def visual_status(catalog: Catalog, config: Config) -> dict[str, Any]:
    """汇总 visual_versions 与 visual_sequences 状态计数与存储占用。"""

    expected = len(select_canonical_completed_works(catalog))
    versions: list[dict[str, Any]] = []
    total_bytes = 0
    for record in catalog.list_visual_versions():
        version = str(record["version"])
        counts = catalog.visual_status_summary(version)
        sequences = catalog.list_visual_sequences(version=version)
        storage_bytes = 0
        missing_files = 0
        for sequence in sequences:
            for key in ("frames_path", "windows_path"):
                relative = sequence.get(key)
                if not relative:
                    continue
                path = config.data_path(str(relative))
                if path.is_file():
                    storage_bytes += path.stat().st_size
                elif str(sequence.get("status")) == "completed":
                    missing_files += 1
        total_bytes += storage_bytes
        versions.append(
            {
                "version": version,
                "model_id": record.get("model_id"),
                "revision": record.get("revision"),
                "preprocessing": record.get("preprocessing"),
                "fps": float(record.get("fps") or 0.0),
                "size": int(record.get("size") or 0),
                "dim": record.get("dim"),
                "status_counts": counts,
                "n_registered": len(sequences),
                "expected_works": expected,
                "missing_files": missing_files,
                "storage_bytes": storage_bytes,
            }
        )
    return {
        "versions": versions,
        "expected_works": expected,
        "total_storage_bytes": total_bytes,
    }


def _directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def clean_completed_temp_dirs(catalog: Catalog, config: Config) -> dict[str, Any]:
    """清理"该作品所有需要的视觉任务都已完成"的 temp/work_XXXXXX 目录。

    需要的任务按当前配置与 30fps 子集选择重新计算：主版本全部 + 子集作品再加
    subset 版本；只有对应 visual_sequences 全部 completed 且文件存在时才删除。
    失败/未完成的作品保留 temp 供续跑。
    """

    dense = config.visual_dense
    subset_ids = (
        set(select_thirty_fps_works(catalog, config, int(dense.subset_count)))
        if int(dense.subset_count) > 0
        else set()
    )
    specs = visual_version_specs(config, include_subset=bool(subset_ids))
    main_versions = [spec.version for spec in specs if not spec.subset]
    subset_versions = [spec.version for spec in specs if spec.subset]
    duplicates = catalog.get_duplicate_work_ids()
    removed = 0
    kept = 0
    skipped = 0
    freed_bytes = 0
    for path in sorted(config.paths.temp.glob("work_*")):
        if not path.is_dir():
            skipped += 1
            continue
        try:
            work_id = int(path.name.split("_", 1)[1])
        except (IndexError, ValueError):
            skipped += 1
            continue
        work = catalog.get_work(work_id)
        if (
            work is None
            or work_id in duplicates
            or str(work.get("status")) != "completed"
        ):
            skipped += 1
            continue
        needed = list(main_versions)
        if work_id in subset_ids:
            needed.extend(subset_versions)
        rows = [catalog.get_visual_sequence(work_id, version=version) for version in needed]
        if any(row is None for row in rows):
            kept += 1
            continue
        if not all(
            str(row.get("status")) == "completed" and _sequence_files_exist(config, row)
            for row in rows
        ):
            kept += 1
            continue
        size = _directory_size(path)
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
        freed_bytes += size
        LOGGER.info("清理已完成作品的 temp 目录: %s（%.2f MB）", path, size / 1024 / 1024)
    return {
        "removed_dirs": removed,
        "kept_dirs": kept,
        "skipped_dirs": skipped,
        "freed_bytes": freed_bytes,
    }
