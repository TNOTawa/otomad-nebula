"""v0.4.3.1 密集视觉批处理流水线：帧编码层与窗口计划层分离。

三层模型（v0.4.3.1）：

- 帧编码层：冻结视觉编码器按固定 fps 逐帧编码，产物
  data/visual_dense/<frame_version>/<work_id>/（dense_visual.npz + manifest），
  由 DB visual_versions / visual_sequences 登记；同一帧版本只编码一次。
- 窗口计划层：把帧级向量按 window plan（窗口时长 + overlap + 权重方案）池化成
  窗口级向量，产物
  data/visual_windows/<window_plan_version>/<frame_version>/<work_id>/
  （windows.npy / window_std.npy / window_weights.npy / manifest.json），
  由 DB visual_window_plans / visual_window_sequences 登记。改变窗口计划只重建
  窗口层，绝不重新编码帧。
- 表征层：由窗口层聚合（另见 tools/build_dense_visual_representation.py）。

窗口计划版本号规则（唯一来源，与 config.visual_window_plan_version 一致）：
    slugify("win{clip:g}s_ov{overlap:g}__{weight_scheme}")
默认 win8s_ov0__coverage-v1；weight_scheme 允许 coverage-v1（覆盖计数积分，
sum=1）与 equal（1/T，复现旧等权行为）。

Bilibili 源视频使用持久缓存：nebula.download.resolve_bilibili_source 先查
data/sources/<bvid>/<bvid>_p<part>.* 可 probe 文件，再下载到暂存后原子移动进缓存；
缓存不随 temp 清理删除，每部作品最多下载一次。
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import dense_visual, sampling
from .config import Config
from .db import Catalog
from .dense_visual import DenseVisualFeatures
from .errors import NebulaError
from .sampling import plan_full_coverage
from .util import now_utc_iso, slugify

LOGGER = logging.getLogger(__name__)

WINDOWS_NAME = "windows.npy"
WINDOW_STD_NAME = "window_std.npy"
WINDOW_WEIGHTS_NAME = "window_weights.npy"
WINDOW_MANIFEST_NAME = "manifest.json"
WINDOW_TIMES_NAME = "window_times.npy"
FRAMES_NAME = "frames.npy"
DEFAULT_DENSE_NPZ_NAME = getattr(dense_visual, "DENSE_VISUAL_NPZ_NAME", "dense_visual.npz")
DEFAULT_DENSE_MANIFEST_NAME = getattr(
    dense_visual, "DENSE_VISUAL_MANIFEST_NAME", "manifest.json"
)
WEIGHT_SCHEMES = ("coverage-v1", "equal")
DEFAULT_WEIGHT_SCHEME = "coverage-v1"
_ERROR_LIMIT = 2000
_SEMANTIC_WEIGHT = 0.5
_EVENT_WEIGHT = 0.5
_FPS_IN_VERSION_RE = re.compile(r"__fps(?P<fps>[0-9]+(?:\.[0-9]+)?)")

EncoderFactory = Callable[[str, str], Any]
Extractor = Callable[..., DenseVisualFeatures]
Downloader = Callable[..., Path]
ProgressCallback = Callable[[str], None]


class VisualPipelineError(NebulaError):
    """密集视觉流水线的可预期错误（配置、版本、窗口、帧文件等）。"""


# ---------------------------------------------------------------- 数据结构


@dataclass(frozen=True)
class VisualVersionSpec:
    """帧编码版本（编码器 + fps + 尺寸 + 预处理）。"""

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
    """一部作品需要跑的一个帧编码版本。"""

    work_id: int
    spec: VisualVersionSpec

    @property
    def version(self) -> str:
        return self.spec.version

    @property
    def fps(self) -> float:
        return self.spec.fps


@dataclass(frozen=True)
class WindowPlan:
    """窗口计划：窗口时长 + 重叠 + 权重方案。"""

    version: str
    clip_seconds: float
    overlap_seconds: float
    weight_scheme: str

    @property
    def label(self) -> str:
        return (
            f"{self.clip_seconds:g}s/ov{self.overlap_seconds:g}"
            f"/{self.weight_scheme}"
        )


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
    window_plan_version: str = ""
    windows_total: int = 0
    windows_completed: int = 0
    windows_skipped: int = 0
    windows_failed: int = 0

    def summary_text(self) -> str:
        mode = "dry-run（未写 DB/文件）" if self.dry_run else "已写入"
        lines = [
            f"密集视觉批处理完成（{mode}）：作品 {self.selected_works} 部"
            f"（30fps 子集 {self.subset_works} 部），帧任务 {self.tasks_total} 个",
            f"  帧层：完成 {self.tasks_completed}，跳过（已存在）{self.tasks_skipped}，"
            f"失败 {self.tasks_failed}",
            f"  窗口层（{self.window_plan_version or '（无）'}）："
            f"完成 {self.windows_completed}，跳过 {self.windows_skipped}，"
            f"失败 {self.windows_failed}（对应 {self.windows_total} 个帧任务）",
            f"  下载作品 {self.downloaded_works} 部，清理 temp 目录 {self.cleaned_temp_dirs} 个，"
            f"耗时 {self.elapsed_s:.2f}s",
            f"  帧版本: {', '.join(self.versions) if self.versions else '（无）'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------- 版本与窗口计划


def compose_visual_version(
    encoder_version: str, fps: float, size: int, preprocess: str
) -> str:
    """按 dense_visual.compress_version 规则生成帧编码版本号（兼容 t1 新旧签名）。"""

    try:
        raw = str(
            dense_visual.compress_version(encoder_version, fps, size, preprocess)  # type: ignore[call-arg]
        )
    except TypeError:
        raw = str(dense_visual.compress_version(encoder_version, fps, size))
    # DB/data 目录契约使用文件系统安全的 slug；model_id/revision 原值保存在
    # visual_versions 行与 manifest 的 encoder/revision 字段中。
    return str(dense_visual.visual_version_slug(raw))


def encoder_version_for(model: str, revision: str) -> str:
    return f"{model}@{revision}"


def window_plan_version_for(
    clip_seconds: float, overlap_seconds: float, weight_scheme: str
) -> str:
    """窗口计划版本号唯一公式（与 config.visual_window_plan_version 一致）。"""

    scheme = str(weight_scheme or "").strip()
    if scheme not in WEIGHT_SCHEMES:
        raise VisualPipelineError(
            f"weight_scheme 必须是 {WEIGHT_SCHEMES}，收到 {weight_scheme!r}"
        )
    clip = float(clip_seconds)
    overlap = float(overlap_seconds)
    if not math.isfinite(clip) or clip <= 0:
        raise VisualPipelineError(f"窗口时长非法: {clip_seconds!r}")
    if not math.isfinite(overlap) or overlap < 0 or overlap >= clip:
        raise VisualPipelineError(
            f"窗口重叠非法: overlap={overlap_seconds!r} clip={clip_seconds!r}"
        )
    return slugify(f"win{clip:g}s_ov{overlap:g}__{scheme}")


def resolve_window_plan(
    config: Config,
    *,
    window_seconds: float | None = None,
    overlap_seconds: float | None = None,
    weight_scheme: str | None = None,
) -> WindowPlan:
    """解析窗口计划；无覆盖参数时使用 config 默认值（config 属性是唯一来源）。"""

    if (
        window_seconds is None
        and overlap_seconds is None
        and weight_scheme is None
    ):
        return WindowPlan(
            version=str(config.visual_window_plan_version),
            clip_seconds=float(config.media.clip_seconds),
            overlap_seconds=float(config.sampling.overlap_seconds),
            weight_scheme=str(getattr(config.visual_dense, "weight_scheme", DEFAULT_WEIGHT_SCHEME)),
        )
    clip = float(
        config.media.clip_seconds if window_seconds is None else window_seconds
    )
    overlap = float(
        config.sampling.overlap_seconds if overlap_seconds is None else overlap_seconds
    )
    scheme = str(
        getattr(config.visual_dense, "weight_scheme", DEFAULT_WEIGHT_SCHEME)
        if weight_scheme is None
        else weight_scheme
    )
    return WindowPlan(
        version=window_plan_version_for(clip, overlap, scheme),
        clip_seconds=clip,
        overlap_seconds=overlap,
        weight_scheme=scheme,
    )


def visual_version_specs(config: Config, *, include_subset: bool) -> list[VisualVersionSpec]:
    """构造当前配置需要的帧编码版本列表（主版本 + 可选子集版本）。"""

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
                        encoder_version,
                        float(dense.subset_fps),
                        int(dense.size),
                        dense.preprocess,
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


# ---------------------------------------------------------------- 作品与 30fps 子集


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
    valid = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
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
    """事件窗口特征相邻变化的平均 L2（按列标准化后）。"""

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
        for row in grouped[name][: quotas[name]]:
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
    """为选中的作品生成每个需要执行的帧编码任务。"""

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


# ---------------------------------------------------------------- 帧编码（dense_visual 适配）


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
        LOGGER.warning("dense_visual.extract_dense_features 尚不支持 preprocess，使用兼容调用")
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


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file_handle:
            json.dump(dict(payload), file_handle, ensure_ascii=False, indent=2, sort_keys=True)
            file_handle.write("\n")
        os.replace(tmp_path, target)
    finally:
        tmp_path.unlink(missing_ok=True)
    return target


def _coerce_saved_paths(target: Path, result: Any) -> dict[str, Path]:
    """把 save_bundle/save_features 的返回值整理成 frames 路径。"""

    paths: dict[str, Path] = {}
    if isinstance(result, Mapping):
        for name in ("frames", "npz", "per_frame", "dense_visual"):
            value = result.get(name)
            if value:
                candidate = Path(str(value))
                paths["frames"] = candidate if candidate.is_absolute() else target / candidate
                break
    if "frames" not in paths:
        for name in (DEFAULT_DENSE_NPZ_NAME, FRAMES_NAME):
            candidate = target / name
            if candidate.is_file():
                paths["frames"] = candidate
                break
    return paths


def _save_frame_bundle(out_dir: Path, features: DenseVisualFeatures) -> dict[str, Path]:
    """只保存帧级向量（帧编码层）；窗口层由 save_visual_window_bundle 另存。"""

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    save_features = getattr(dense_visual, "save_features", None)
    if callable(save_features):
        result = save_features(features, target)
    else:
        save_bundle = getattr(dense_visual, "save_bundle", None)
        if not callable(save_bundle):
            raise VisualPipelineError("dense_visual 缺少 save_features/save_bundle")
        result = save_bundle(target, features)
    paths = _coerce_saved_paths(target, result)
    if "frames" not in paths:
        raise VisualPipelineError(f"帧编码保存后找不到帧级文件: {target}")
    return paths


# ---------------------------------------------------------------- 窗口层


def window_plan_root(config: Config) -> Path:
    return Path(
        getattr(config.paths, "visual_windows", config.data_path("visual_windows"))
    )


def visual_windows_dir(
    config: Config, window_plan_version: str, frame_version: str, work_id: int
) -> Path:
    return (
        window_plan_root(config)
        / str(window_plan_version)
        / str(frame_version)
        / f"{int(work_id):06d}"
    )


def _window_files_exist(config: Config, sequence: Mapping[str, Any]) -> bool:
    for key in ("windows_path", "window_std_path", "window_weights_path"):
        relative = sequence.get(key)
        if not relative:
            return False
        path = config.data_path(str(relative))
        if not path.is_file() or path.stat().st_size <= 0:
            return False
    return True


def _sequence_files_exist(config: Config, sequence: Mapping[str, Any]) -> bool:
    """帧层记录：只要求 frames_path 存在。"""

    relative = sequence.get("frames_path")
    if not relative:
        return False
    path = config.data_path(str(relative))
    return path.is_file() and path.stat().st_size > 0


def _coverage_weights_compat(
    windows: Sequence[tuple[float, float]], duration_s: float
) -> tuple[float, ...]:
    """调用 t1 的 public coverage_weights；未就绪时回退到 sampling 内部实现。"""

    public = getattr(sampling, "coverage_weights", None)
    if callable(public):
        values = public(windows, duration_s)
    else:
        private = getattr(sampling, "_coverage_weights", None)
        if not callable(private):
            raise VisualPipelineError(
                "sampling.coverage_weights 未就绪，无法计算窗口覆盖权重"
            )
        LOGGER.debug("sampling.coverage_weights 尚未公开，回退到 _coverage_weights")
        values = private(windows, duration_s)
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(windows),):
        raise VisualPipelineError(
            f"窗口权重数量异常: {array.shape} vs {len(windows)} 个窗口"
        )
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise VisualPipelineError("窗口权重含非法值（NaN/Inf/负数）")
    total = float(array.sum())
    if total <= 0:
        raise VisualPipelineError("窗口权重之和必须为正")
    return tuple(float(value / total) for value in array)


def window_weights_for(
    windows: Sequence[tuple[float, float]],
    duration_s: float,
    weight_scheme: str,
) -> np.ndarray:
    """按权重方案生成窗口权重（float32，和归一为 1）。"""

    if not windows:
        raise VisualPipelineError("窗口列表不能为空")
    if weight_scheme == "coverage-v1":
        weights = _coverage_weights_compat(windows, duration_s)
    elif weight_scheme == "equal":
        weights = tuple(1.0 / len(windows) for _ in windows)
    else:
        raise VisualPipelineError(f"未知 weight_scheme: {weight_scheme!r}")
    array = np.asarray(weights, dtype=np.float32)
    array = array / np.float32(array.sum())
    return np.ascontiguousarray(array, dtype=np.float32)


def plan_windows(
    config: Config, plan: WindowPlan, duration_s: float
) -> list[tuple[float, float]]:
    """按窗口计划生成与 PE-AV 全片序列对齐的窗口 (start, end) 列表。"""

    plans = plan_full_coverage(
        float(duration_s),
        clip_seconds=float(plan.clip_seconds),
        overlap_seconds=float(plan.overlap_seconds),
        min_duration_seconds=float(config.media.min_duration_seconds),
    )
    return [(float(item.start_s), float(item.end_s)) for item in plans]


def save_visual_window_bundle(
    out_dir: Path,
    *,
    pooled: np.ndarray,
    window_std: np.ndarray,
    window_weights: np.ndarray,
    window_times: Sequence[tuple[float, float]],
    frame_version: str,
    plan: WindowPlan,
    features: DenseVisualFeatures,
    duration_s: float,
) -> dict[str, Path]:
    """原子保存窗口层 bundle（manifest 最后写，作为完成标记）。"""

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    windows_array = np.ascontiguousarray(np.asarray(pooled, dtype=np.float32))
    std_array = np.ascontiguousarray(np.asarray(window_std, dtype=np.float32))
    weights_array = np.ascontiguousarray(np.asarray(window_weights, dtype=np.float32))
    n_windows = int(windows_array.shape[0])
    if std_array.shape != windows_array.shape:
        raise VisualPipelineError("window_std 与 windows 形状不一致")
    if weights_array.shape != (n_windows,):
        raise VisualPipelineError("window_weights 与窗口数不一致")
    if len(window_times) != n_windows:
        raise VisualPipelineError("window_times 与窗口数不一致")
    if not np.all(np.isfinite(windows_array)) or not np.all(np.isfinite(std_array)):
        raise VisualPipelineError("窗口向量/标准差含 NaN/Inf")
    if not np.all(np.isfinite(weights_array)) or np.any(weights_array < 0):
        raise VisualPipelineError("窗口权重含非法值")
    total_weight = float(weights_array.sum())
    if total_weight <= 0:
        raise VisualPipelineError("窗口权重之和必须为正")

    windows_path = _atomic_save_npy(target / WINDOWS_NAME, windows_array)
    std_path = _atomic_save_npy(target / WINDOW_STD_NAME, std_array)
    weights_path = _atomic_save_npy(target / WINDOW_WEIGHTS_NAME, weights_array)
    manifest_payload = {
        "window_plan_version": plan.version,
        "frame_version": str(frame_version),
        "clip_seconds": float(plan.clip_seconds),
        "overlap_seconds": float(plan.overlap_seconds),
        "weight_scheme": str(plan.weight_scheme),
        "duration_s": float(duration_s),
        "fps": float(features.fps),
        "n_frames": int(features.n_frames),
        "n_windows": n_windows,
        "dim": int(features.dim),
        "encoder": str(features.encoder),
        "revision": features.revision,
        "size": int(features.size),
        "preprocess": str(features.preprocess),
        "weight_sum": total_weight,
        "window_times": [
            [float(start), float(end)] for start, end in window_times
        ],
        "files": {
            "windows": WINDOWS_NAME,
            "window_std": WINDOW_STD_NAME,
            "window_weights": WINDOW_WEIGHTS_NAME,
        },
        "created_at": now_utc_iso(),
    }
    manifest_path = _atomic_write_json(target / WINDOW_MANIFEST_NAME, manifest_payload)
    return {
        "windows": windows_path,
        "window_std": std_path,
        "window_weights": weights_path,
        "manifest": manifest_path,
    }


def load_frame_features(
    config: Config, frame_version: str, work_id: int
) -> DenseVisualFeatures | None:
    """从已保存的帧编码 bundle 读取帧级特征（不下载、不解码、不跑模型）。

    v0.4.3 旧 bundle 的 npz/manifest version 表示差异由 dense_visual.load_bundle
    统一兼容（slug 等价），这里不再保留第二套读取逻辑。
    """

    out_dir = config.paths.visual_dense / str(frame_version) / f"{int(work_id):06d}"
    return dense_visual.load_features(out_dir)


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


def _register_window_plan(catalog: Catalog, plan: WindowPlan) -> None:
    catalog.register_visual_window_plan(
        plan.version,
        clip_seconds=plan.clip_seconds,
        overlap_seconds=plan.overlap_seconds,
        weight_scheme=plan.weight_scheme,
        params={
            "layout": "visual_windows/<window_plan_version>/<frame_version>/<work_id>",
            "source": "nebula.visual_pipeline",
            "weight_scheme": plan.weight_scheme,
        },
    )


def build_windows_for_work(
    catalog: Catalog,
    config: Config,
    work: Mapping[str, Any],
    frame_version: str,
    *,
    plan: WindowPlan | None = None,
    features: DenseVisualFeatures | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """为一部作品的一个帧版本重建窗口层并登记 DB。

    返回 {"status": "created"|"skipped", "n_windows", "total_s", "path"}；
    失败时先写 visual_window_sequences.failed，再抛出原异常。
    """

    resolved_plan = plan or resolve_window_plan(config)
    work_id = int(work["id"])
    _register_window_plan(catalog, resolved_plan)
    existing = catalog.get_visual_window_sequence(
        work_id,
        frame_version=str(frame_version),
        window_plan_version=resolved_plan.version,
    )
    if (
        existing is not None
        and str(existing.get("status")) == "completed"
        and _window_files_exist(config, existing)
        and not force
    ):
        return {
            "status": "skipped",
            "n_windows": int(existing.get("n_windows") or 0),
            "total_s": 0.0,
            "path": visual_windows_dir(
                config, resolved_plan.version, str(frame_version), work_id
            ),
        }

    started = time.perf_counter()
    catalog.upsert_visual_window_sequence(
        work_id,
        frame_version=str(frame_version),
        window_plan_version=resolved_plan.version,
        status="in_progress",
    )
    try:
        active_features = features
        if active_features is None:
            active_features = load_frame_features(config, str(frame_version), work_id)
        if active_features is None:
            raise VisualPipelineError(
                f"帧级特征不存在: frame_version={frame_version}, work_id={work_id}"
            )
        duration_s = _choose_duration(work, active_features)
        windows = plan_windows(config, resolved_plan, duration_s)
        pooled = dense_visual.pool_windows(active_features, windows)
        window_std = dense_visual.pool_windows_std(active_features, windows)
        weights = window_weights_for(windows, duration_s, resolved_plan.weight_scheme)
        out_dir = visual_windows_dir(
            config, resolved_plan.version, str(frame_version), work_id
        )
        saved = save_visual_window_bundle(
            out_dir,
            pooled=pooled,
            window_std=window_std,
            window_weights=weights,
            window_times=windows,
            frame_version=str(frame_version),
            plan=resolved_plan,
            features=active_features,
            duration_s=duration_s,
        )
        total_s = time.perf_counter() - started
        catalog.upsert_visual_window_sequence(
            work_id,
            frame_version=str(frame_version),
            window_plan_version=resolved_plan.version,
            status="completed",
            windows_path=config.data_relpath(saved["windows"]),
            window_std_path=config.data_relpath(saved["window_std"]),
            window_weights_path=config.data_relpath(saved["window_weights"]),
            n_windows=int(np.asarray(pooled).shape[0]),
            dim=int(active_features.dim),
            total_s=total_s,
        )
        return {
            "status": "created",
            "n_windows": int(np.asarray(pooled).shape[0]),
            "total_s": total_s,
            "path": out_dir,
        }
    except Exception as exc:
        with_error = _error_text(exc)
        try:
            catalog.upsert_visual_window_sequence(
                work_id,
                frame_version=str(frame_version),
                window_plan_version=resolved_plan.version,
                status="failed",
                error=with_error,
                total_s=time.perf_counter() - started,
            )
        except NebulaError:  # pragma: no cover - DB 异常时只记录
            LOGGER.exception("记录窗口层失败状态时出错: work_id=%s", work_id)
        raise


# ---------------------------------------------------------------- 帧版本发现（重建工具用）


def _frame_version_fps(
    catalog: Catalog, frame_version: str, version_row: Mapping[str, Any] | None = None
) -> float | None:
    row = version_row
    if row is None:
        row = catalog.get_visual_version(frame_version)
    if row is not None:
        try:
            return float(row.get("fps"))
        except (TypeError, ValueError):
            return None
    match = _FPS_IN_VERSION_RE.search(str(frame_version))
    if match:
        return float(match.group("fps"))
    return None


def discover_frame_versions(config: Config) -> list[str]:
    """扫描 data/visual_dense 下含至少一个帧 bundle 的版本目录。"""

    root = config.paths.visual_dense
    if not root.is_dir():
        return []
    versions: list[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        has_bundle = any(
            (work_dir / DEFAULT_DENSE_NPZ_NAME).is_file()
            for work_dir in child.iterdir()
            if work_dir.is_dir()
        )
        if has_bundle:
            versions.append(child.name)
    return versions


def resolve_frame_versions(
    catalog: Catalog,
    config: Config,
    *,
    frame_versions: Sequence[str] | None = None,
    all_versions: bool = False,
    fps: float | None = None,
) -> list[str]:
    """确定要重建窗口层的帧编码版本列表。

    - 显式 frame_versions：原样返回（不要求已注册，工具会报告缺失）；
    - all_versions=True：扫描 data/visual_dense 目录（含未登记版本）；
    - 默认：DB 中已登记的帧编码版本。
    """

    if frame_versions:
        versions = [str(value) for value in frame_versions]
    elif all_versions:
        versions = discover_frame_versions(config)
    else:
        versions = [str(row["version"]) for row in catalog.list_visual_versions()]
    versions = sorted(dict.fromkeys(versions))
    if fps is not None:
        wanted = float(fps)
        rows = {str(row["version"]): row for row in catalog.list_visual_versions()}
        filtered = []
        for version in versions:
            actual = _frame_version_fps(catalog, version, rows.get(version))
            if actual is not None and abs(actual - wanted) <= 1e-6:
                filtered.append(version)
        versions = filtered
    return versions


def build_visual_windows(
    catalog: Catalog,
    config: Config,
    *,
    window_seconds: float | None = None,
    overlap_seconds: float | None = None,
    weight_scheme: str | None = None,
    frame_versions: Sequence[str] | None = None,
    all_versions: bool = False,
    fps: float | None = None,
    limit: int | None = None,
    force: bool = False,
    dry_run: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """从已保存的帧级特征批量重建窗口层（不下载、不解码、不跑模型）。"""

    started = time.perf_counter()
    plan = resolve_window_plan(
        config,
        window_seconds=window_seconds,
        overlap_seconds=overlap_seconds,
        weight_scheme=weight_scheme,
    )
    works = select_canonical_completed_works(catalog)
    works.sort(key=lambda work: int(work["id"]))
    if limit is not None:
        if int(limit) < 0:
            raise VisualPipelineError(f"limit 不能为负: {limit}")
        works = works[: int(limit)]
    versions = resolve_frame_versions(
        catalog, config, frame_versions=frame_versions, all_versions=all_versions, fps=fps
    )
    known_versions = {str(row["version"]) for row in catalog.list_visual_versions()}
    if not dry_run:
        _register_window_plan(catalog, plan)
    created = 0
    skipped = 0
    missing_frames = 0
    failed: list[dict[str, Any]] = []
    would_process = 0
    total_windows = 0
    processed_pairs = 0
    total_pairs = len(versions) * len(works)

    for version_index, frame_version in enumerate(versions, start=1):
        if not dry_run and frame_version not in known_versions:
            missing_frames += len(works)
            failed.append(
                {
                    "frame_version": frame_version,
                    "reason": "帧编码版本未在 visual_versions 登记（无法建立窗口外键）",
                }
            )
            continue
        for work in works:
            work_id = int(work["id"])
            processed_pairs += 1
            frame_dir = (
                config.paths.visual_dense / frame_version / f"{work_id:06d}"
            )
            bundle = frame_dir / DEFAULT_DENSE_NPZ_NAME
            if not bundle.is_file():
                missing_frames += 1
                continue
            existing = catalog.get_visual_window_sequence(
                work_id,
                frame_version=frame_version,
                window_plan_version=plan.version,
            )
            if (
                existing is not None
                and str(existing.get("status")) == "completed"
                and _window_files_exist(config, existing)
                and not force
            ):
                skipped += 1
                continue
            if dry_run:
                would_process += 1
                continue
            if progress:
                progress(
                    f"[{version_index}/{len(versions)}] frame_version={frame_version} "
                    f"work_id={work_id} -> {plan.version}"
                )
            try:
                result = build_windows_for_work(
                    catalog, config, work, frame_version, plan=plan, force=force
                )
            except Exception as exc:  # noqa: BLE001 - 单作品失败不阻塞批处理
                failed.append(
                    {
                        "frame_version": frame_version,
                        "work_id": work_id,
                        "error": _error_text(exc),
                    }
                )
                LOGGER.error(
                    "窗口层重建失败: frame_version=%s work_id=%s: %s",
                    frame_version,
                    work_id,
                    _error_text(exc),
                )
                continue
            if result["status"] == "skipped":
                skipped += 1
            else:
                created += 1
                total_windows += int(result.get("n_windows") or 0)

    return {
        "window_plan_version": plan.version,
        "plan": plan,
        "frame_versions": versions,
        "works": len(works),
        "pairs": processed_pairs,
        "total_pairs": total_pairs,
        "created": created,
        "skipped": skipped,
        "missing_frames": missing_frames,
        "failed": failed,
        "would_process": would_process,
        "total_windows": total_windows,
        "dry_run": bool(dry_run),
        "elapsed_s": time.perf_counter() - started,
    }


# ---------------------------------------------------------------- Bilibili 源缓存 / 本地源


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


def _existing_cached_sources(config: Config, bvid: str, part: int) -> set[Path]:
    cache_dir = Path(getattr(config.paths, "sources", config.data_path("sources"))) / bvid
    if not cache_dir.is_dir():
        return set()
    return {path.resolve() for path in cache_dir.glob(f"{bvid}_p{int(part)}.*")}


def _resolve_source(
    work: Mapping[str, Any],
    config: Config,
    downloader: Downloader,
) -> tuple[Path, Path | None, bool]:
    """返回 (源文件路径, 作品 temp 根目录或 None, 本次是否真的下载了)。

    本地作品用原文件；Bilibili 走 download.resolve_bilibili_source 持久缓存
    （data/sources），缓存命中不下载，缺失才下载到 temp 暂存后原子移动。
    """

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
        before = _existing_cached_sources(config, bvid, part)
        from .download import resolve_bilibili_source

        path = Path(
            resolve_bilibili_source(
                config,
                None,
                bvid,
                part,
                temp_root,
                config.download,
                timeout_s=float(config.media.probe_timeout_seconds),
                downloader=downloader,
            )
        )
        downloaded = path.resolve() not in before
        LOGGER.info(
            "work_id=%s 视觉源文件: %s（%s）",
            work_id,
            path,
            "本次下载" if downloaded else "缓存命中",
        )
        return path, temp_root, downloaded
    raise VisualPipelineError(f"未知作品来源: {source!r}（work_id={work_id}）")


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
    """批量生成帧编码层与窗口计划层。

    帧任务判定：visual_sequences 对应帧版本 completed 且 frames_path 存在 -> 跳过编码；
    窗口判定：visual_window_sequences 对应 (work, frame_version, window_plan_version)
    completed 且 windows/std/weights 文件存在 -> 跳过；仅窗口缺失时用已保存帧重建，
    不重新编码。Bilibili 源视频来自 data/sources 持久缓存。
    """

    started = time.perf_counter()
    dense = config.visual_dense
    plan = resolve_window_plan(config)
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
            window_plan_version=plan.version,
            windows_total=tasks_total,
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
            window_plan_version=plan.version,
            windows_total=0,
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
    _register_window_plan(catalog, plan)

    completed = 0
    skipped = 0
    failed = 0
    downloaded_works = 0
    cleaned_dirs = 0
    windows_completed = 0
    windows_skipped = 0
    windows_failed = 0
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
                    f"[{work_index}/{len(works)}] work_id={work_id} "
                    f"帧任务 {len(tasks)} 个，窗口计划 {plan.version}"
                )
            for task in tasks:
                existing = catalog.get_visual_sequence(work_id, version=task.version)
                frame_done = (
                    existing is not None
                    and str(existing.get("status")) == "completed"
                    and _sequence_files_exist(config, existing)
                )
                features: DenseVisualFeatures | None = None
                if frame_done:
                    skipped += 1
                else:
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
                                "work_id=%s 获取视觉源文件失败: %s",
                                work_id,
                                _error_text(exc),
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
                        features = replace(features, version=task.version)
                        out_dir = (
                            config.paths.visual_dense / task.version / f"{work_id:06d}"
                        )
                        saved = _save_frame_bundle(out_dir, features)
                        total_s = time.perf_counter() - task_started
                        catalog.upsert_visual_sequence(
                            work_id,
                            version=task.version,
                            status="completed",
                            frames_path=config.data_relpath(saved["frames"]),
                            n_frames=int(features.n_frames),
                            dim=int(features.dim),
                            decode_s=float(features.decode_s),
                            encode_s=float(features.encode_s),
                            model_load_s=float(
                                getattr(encoder, "model_load_s", 0.0) or 0.0
                            ),
                            total_s=total_s,
                        )
                        completed += 1
                        LOGGER.info(
                            "work_id=%s 帧编码完成: %s（%d 帧, %.2fs）",
                            work_id,
                            task.version,
                            int(features.n_frames),
                            total_s,
                        )
                        if progress:
                            progress(
                                f"  work_id={work_id} {task.version}: "
                                f"{features.n_frames} 帧"
                            )
                    except Exception as exc:  # noqa: BLE001 - 单任务失败不阻塞
                        failed += 1
                        LOGGER.exception(
                            "work_id=%s 帧编码失败: %s", work_id, task.version
                        )
                        try:
                            catalog.upsert_visual_sequence(
                                work_id,
                                version=task.version,
                                status="failed",
                                error=_error_text(exc),
                                total_s=time.perf_counter() - task_started,
                            )
                        except NebulaError:  # pragma: no cover
                            LOGGER.exception(
                                "记录帧编码失败状态时出错: work_id=%s", work_id
                            )
                        continue
                try:
                    result = build_windows_for_work(
                        catalog,
                        config,
                        work,
                        task.version,
                        plan=plan,
                        features=features,
                    )
                except Exception as exc:  # noqa: BLE001 - 窗口失败不阻塞其它任务
                    windows_failed += 1
                    LOGGER.exception(
                        "work_id=%s 窗口层失败: %s", work_id, task.version
                    )
                    if progress:
                        progress(f"  work_id={work_id} 窗口层失败: {_error_text(exc)}")
                    continue
                if result["status"] == "skipped":
                    windows_skipped += 1
                else:
                    windows_completed += 1
            if work_downloaded:
                downloaded_works += 1
            if temp_root is not None and tasks:
                all_frames_done = True
                for task in tasks:
                    row = catalog.get_visual_sequence(work_id, version=task.version)
                    if (
                        row is None
                        or str(row.get("status")) != "completed"
                        or not _sequence_files_exist(config, row)
                    ):
                        all_frames_done = False
                        break
                if all_frames_done and temp_root.is_dir():
                    shutil.rmtree(temp_root, ignore_errors=True)
                    cleaned_dirs += 1
                    LOGGER.info(
                        "work_id=%s 全部帧任务完成，清理 temp: %s", work_id, temp_root
                    )
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
        window_plan_version=plan.version,
        windows_total=tasks_total,
        windows_completed=windows_completed,
        windows_skipped=windows_skipped,
        windows_failed=windows_failed,
    )


# ---------------------------------------------------------------- 状态 / 清理


def visual_status(catalog: Catalog, config: Config) -> dict[str, Any]:
    """汇总帧编码层与窗口计划层的状态计数与存储占用。"""

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
            relative = sequence.get("frames_path")
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

    window_plans: list[dict[str, Any]] = []
    window_total_bytes = 0
    for record in catalog.list_visual_window_plans():
        plan_version = str(record["version"])
        counts = catalog.visual_window_status_summary(plan_version)
        sequences = catalog.list_visual_window_sequences(
            window_plan_version=plan_version
        )
        storage_bytes = 0
        missing_files = 0
        for sequence in sequences:
            for key in ("windows_path", "window_std_path", "window_weights_path"):
                relative = sequence.get(key)
                if not relative:
                    continue
                path = config.data_path(str(relative))
                if path.is_file():
                    storage_bytes += path.stat().st_size
                elif str(sequence.get("status")) == "completed":
                    missing_files += 1
        window_total_bytes += storage_bytes
        window_plans.append(
            {
                "version": plan_version,
                "clip_seconds": float(record.get("clip_seconds") or 0.0),
                "overlap_seconds": float(record.get("overlap_seconds") or 0.0),
                "weight_scheme": record.get("weight_scheme"),
                "status_counts": counts,
                "n_registered": len(sequences),
                "expected_works": expected,
                "missing_files": missing_files,
                "storage_bytes": storage_bytes,
            }
        )
    return {
        "versions": versions,
        "window_plans": window_plans,
        "expected_works": expected,
        "total_storage_bytes": total_bytes,
        "window_total_storage_bytes": window_total_bytes,
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
    """清理帧编码任务全部完成的 temp/work_XXXXXX 目录。

    Bilibili 源视频已改为 data/sources 持久缓存，temp 只放下载暂存与切片中间物；
    只有对应帧版本全部 completed 且 frames 文件存在时才删除，失败/未完成保留。
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
        rows = [
            catalog.get_visual_sequence(work_id, version=version) for version in needed
        ]
        if any(row is None for row in rows):
            kept += 1
            continue
        if not all(
            str(row.get("status")) == "completed"
            and _sequence_files_exist(config, row)
            for row in rows
        ):
            kept += 1
            continue
        size = _directory_size(path)
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
        freed_bytes += size
        LOGGER.info(
            "清理已完成作品的 temp 目录: %s（%.2f MB）", path, size / 1024 / 1024
        )
    return {
        "removed_dirs": removed,
        "kept_dirs": kept,
        "skipped_dirs": skipped,
        "freed_bytes": freed_bytes,
    }
