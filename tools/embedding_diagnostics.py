r"""v0.4.3 低维投影诊断：PCA / UMAP 邻域保持率与距离相关。

对已登记的作品级表征（默认 full_mean_v1、temporal_stats_v1）做低维投影诊断，
回答“当前 2D/3D 星云结构杂乱主要来自哪里”：

1. 高维空间：执行 L2 归一化后的余弦距离矩阵；
2. 低维投影：
   - PCA 2D / 3D（sklearn PCA，svd_solver="full"，random_state=42）；
   - UMAP 2D / 3D × n_neighbors ∈ {15, 30, 50}（metric=cosine、min_dist=0.1、
     random_state=42、n_jobs=1；小样本自动把 n_neighbors 限制到 n-1）；
3. 指标（每个低维配置相对高维）：
   - Top-k 邻域保持率：每部作品高维 Top-k 与低维 Top-k 交集 / 有效 k 的均值；
   - Top-1 保持率；
   - 全对距离相关：高维余弦距离 vs 低维欧氏距离的 Pearson / Spearman；
4. 一致性：不同 UMAP n_neighbors 之间的 Top-k 保持率；2D vs 3D 的 Top-k 保持率；
5. 输出 logs/dim_diagnostics.json + CSV，可选为每个投影写 Plotly HTML；
6. CLI 支持多个 --representation 与 --k/--umap-neighbors/--components/--out/--csv/
   --html-dir。

本工具只读取已登记向量（nebula.representations.load_representation_matrix，
canonical only），不写数据库、不加载 PE-AV、不做模型编码、不访问网络。
工具本身不修改 data/；输出默认写入配置的 logs 目录。真实 UMAP 首次调用需要
numba 编译（本机约 20-30s），测试通过注入轻量 UMAP 替身覆盖 schema/确定性。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nebula.config import Config, load_config  # noqa: E402
from nebula.db import Catalog  # noqa: E402
from nebula.errors import NebulaError  # noqa: E402
from nebula.highdim import (  # noqa: E402
    cosine_distance_matrix,
    normalize_vectors,
    topk_distance_neighbors,
)
from nebula.log import configure_logging  # noqa: E402
from nebula.representations import (  # noqa: E402
    load_representation_matrix,
    representation_label,
    resolve_representation,
)
from nebula.util import now_utc_iso, slugify  # noqa: E402

TOOL_NAME = "embedding_diagnostics"
TOOL_VERSION = "v0.4.3"

#: 默认诊断的表征（v0.4 三个空间中除 legacy 外的两个主空间）
DEFAULT_REPRESENTATIONS: tuple[str, ...] = ("full_mean_v1", "temporal_stats_v1")
DEFAULT_K = 10
DEFAULT_UMAP_NEIGHBORS: tuple[int, ...] = (15, 30, 50)
DEFAULT_COMPONENTS: tuple[int, ...] = (2, 3)
SUPPORTED_COMPONENTS: tuple[int, ...] = (2, 3)
DEFAULT_KIND = "joint"
VECTOR_KINDS: tuple[str, ...] = ("joint", "audio", "video")

PCA_RANDOM_STATE = 42
UMAP_METRIC = "cosine"
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42
UMAP_N_JOBS = 1
UMAP_TRANSFORM_SEED = 42

_EPS = 1e-12


# --------------------------------------------------------------------------- 校验
def _positive_int(value: Any, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NebulaError(f"{name} 必须是整数，收到 {value!r}")
    if value < minimum:
        raise NebulaError(f"{name} 必须 >= {minimum}，收到 {value}")
    return value


def _validated_k(k: int) -> int:
    return _positive_int(k, name="k", minimum=1)


def _validated_components(components: Sequence[int]) -> tuple[int, ...]:
    if not components:
        raise NebulaError("components 不能为空（支持 2 和/或 3）")
    result: list[int] = []
    for value in components:
        if isinstance(value, bool) or not isinstance(value, int):
            raise NebulaError(f"components 必须为整数，收到 {value!r}")
        if value not in SUPPORTED_COMPONENTS:
            raise NebulaError(f"components 只支持 {SUPPORTED_COMPONENTS}，收到 {value}")
        if value in result:
            raise NebulaError(f"components 重复: {value}")
        result.append(value)
    return tuple(result)


def _validated_neighbors(neighbors: Sequence[int]) -> tuple[int, ...]:
    if not neighbors:
        raise NebulaError("umap_neighbors 不能为空")
    result: list[int] = []
    for value in neighbors:
        checked = _positive_int(value, name="umap_neighbors", minimum=2)
        if checked in result:
            raise NebulaError(f"umap_neighbors 重复: {checked}")
        result.append(checked)
    return tuple(result)


def _validated_kind(kind: str) -> str:
    if kind not in VECTOR_KINDS:
        raise NebulaError(f"未知向量种类: {kind}（支持 {VECTOR_KINDS}）")
    return kind


def _validated_names(names: Sequence[str]) -> tuple[str, ...]:
    if not names:
        raise NebulaError("至少需要一个 --representation")
    result: list[str] = []
    for name in names:
        text = str(name or "").strip()
        if not text:
            raise NebulaError("--representation 不能为空")
        if text in result:
            raise NebulaError(f"--representation 重复: {text}")
        result.append(text)
    return tuple(result)


# --------------------------------------------------------------------------- 数据空间
@dataclass(frozen=True)
class SpaceMatrix:
    """一个表征的 canonical 作品矩阵（行 = work_id 升序，列 = 高维特征）。"""

    name: str
    label: str
    sequence_version: str
    representation_id: int
    work_ids: tuple[int, ...]
    matrix: np.ndarray  # (n, dim) float64，逐行 L2 归一化

    @property
    def n_works(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])

    @classmethod
    def from_matrix(
        cls,
        matrix: Any,
        *,
        name: str = "synthetic",
        label: str | None = None,
        sequence_version: str = "",
        representation_id: int = 0,
        work_ids: Sequence[int] | None = None,
    ) -> "SpaceMatrix":
        normalized = normalize_vectors(matrix, where=f"{name} 向量矩阵")
        count = int(normalized.shape[0])
        ids = (
            tuple(int(value) for value in work_ids)
            if work_ids is not None
            else tuple(range(count))
        )
        if len(ids) != count:
            raise NebulaError(f"{name} 的 work_ids 数量 {len(ids)} 与矩阵行数 {count} 不一致")
        if len(set(ids)) != len(ids):
            raise NebulaError(f"{name} 的 work_ids 有重复")
        return cls(
            name=str(name),
            label=str(label or name),
            sequence_version=str(sequence_version),
            representation_id=int(representation_id),
            work_ids=ids,
            matrix=normalized,
        )


def load_space(
    catalog: Catalog,
    config: Config,
    name: str,
    *,
    sequence_version: str | None = None,
    kind: str = DEFAULT_KIND,
) -> SpaceMatrix:
    """解析表征并加载其 canonical 作品矩阵（按 work_id 排序）。"""

    kind = _validated_kind(kind)
    rep = resolve_representation(
        catalog, config, name=name, sequence_version=sequence_version
    )
    work_ids, matrix = load_representation_matrix(catalog, config, rep, kind=kind)
    normalized = normalize_vectors(
        matrix, where=f"{rep['name']}@{rep['sequence_version']} 向量矩阵"
    )
    return SpaceMatrix(
        name=str(rep["name"]),
        label=representation_label(str(rep["name"])),
        sequence_version=str(rep["sequence_version"]),
        representation_id=int(rep["id"]),
        work_ids=tuple(int(work_id) for work_id in work_ids),
        matrix=normalized,
    )


# --------------------------------------------------------------------------- 距离与邻域
# 共享实现见 nebula.highdim；本工具旧内部名保留为别名，避免测试/外部脚本失效。
topk_neighbors = topk_distance_neighbors


def euclidean_distance_matrix(points: Any) -> np.ndarray:
    """低维坐标的欧氏距离矩阵（对称、对角线为 0）。"""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1:
        raise NebulaError(f"低维坐标形状非法: {values.shape}")
    if not np.all(np.isfinite(values)):
        raise NebulaError("低维坐标包含 NaN/Inf")
    squared_norms = np.sum(values * values, axis=1)
    squared = squared_norms[:, None] + squared_norms[None, :] - 2.0 * (values @ values.T)
    distance = np.sqrt(np.maximum(squared, 0.0))
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    return distance


def _pearson(first: Any, second: Any) -> float:
    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
    if a.size != b.size:
        raise NebulaError(f"Pearson 输入长度不一致: {a.size} != {b.size}")
    if a.size < 2:
        return 0.0
    a = a - float(a.mean())
    b = b - float(b.mean())
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= _EPS:
        return 0.0
    return float(np.dot(a, b) / denominator)


def _average_ranks(values: Any) -> np.ndarray:
    """平均秩（处理并列），用于 Spearman 相关。"""

    array = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(array, kind="stable")
    sorted_values = array[order]
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start
        while end + 1 < array.size and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2.0 + 1.0
        start = end + 1
    return ranks


def _spearman(first: Any, second: Any) -> float:
    return _pearson(_average_ranks(first), _average_ranks(second))


def pair_distance_correlation(high_distance: Any, low_distance: Any) -> dict[str, Any]:
    """全对（i<j）高维距离 vs 低维距离的 Pearson / Spearman。"""

    high = np.asarray(high_distance, dtype=np.float64)
    low = np.asarray(low_distance, dtype=np.float64)
    if high.shape != low.shape or high.ndim != 2 or high.shape[0] != high.shape[1]:
        raise NebulaError(
            f"距离矩阵形状不一致: {high.shape} != {low.shape}（需要同形方阵）"
        )
    count = int(high.shape[0])
    if count < 2:
        return {"pearson": 0.0, "spearman": 0.0, "n_pairs": 0}
    rows, columns = np.triu_indices(count, k=1)
    high_values = high[rows, columns]
    low_values = low[rows, columns]
    return {
        "pearson": round(_pearson(high_values, low_values), 6),
        "spearman": round(_spearman(high_values, low_values), 6),
        "n_pairs": int(high_values.size),
    }


def neighborhood_metrics(
    reference_topk: Any, candidate_topk: Any
) -> dict[str, Any]:
    """参考 Top-k 与候选 Top-k 的邻域保持率（每作品交集比例均值）。"""

    reference = np.asarray(reference_topk, dtype=np.int64)
    candidate = np.asarray(candidate_topk, dtype=np.int64)
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise NebulaError(
            f"Top-k 矩阵形状不一致: {reference.shape} != {candidate.shape}"
        )
    count, effective = int(reference.shape[0]), int(reference.shape[1])
    if count < 1 or effective < 1:
        return {
            "topk_preservation": 0.0,
            "top1_preservation": 0.0,
            "mean_overlap": 0.0,
            "stable_ratio": 0.0,
            "effective_k": effective,
            "n_works": count,
        }
    overlaps = np.zeros(count, dtype=np.int64)
    for row in range(count):
        overlaps[row] = len(
            set(int(value) for value in reference[row])
            & set(int(value) for value in candidate[row])
        )
    top1 = float(np.mean(reference[:, 0] == candidate[:, 0]))
    return {
        "topk_preservation": round(float(overlaps.mean()) / effective, 6),
        "top1_preservation": round(top1, 6),
        "mean_overlap": round(float(overlaps.mean()), 4),
        "stable_ratio": round(float(np.mean(overlaps == effective)), 6),
        "effective_k": effective,
        "n_works": count,
    }


# --------------------------------------------------------------------------- 投影
def pca_projection(
    matrix: Any,
    n_components: int,
    *,
    random_state: int = PCA_RANDOM_STATE,
) -> np.ndarray:
    """PCA 投影（完整 SVD，确定性）；输入会先做 L2 归一化。"""

    components = _positive_int(n_components, name="n_components", minimum=1)
    normalized = normalize_vectors(matrix, where="PCA 输入")
    if components > min(normalized.shape):
        raise NebulaError(
            f"n_components={components} 超过 min(n_samples, dim)={min(normalized.shape)}"
        )
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:  # pragma: no cover - 项目依赖已包含 scikit-learn
        raise NebulaError(f"缺少 scikit-learn 依赖，无法做 PCA：{exc}") from exc
    try:
        model = PCA(
            n_components=components,
            svd_solver="full",
            random_state=int(random_state),
        )
        coordinates = model.fit_transform(normalized)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise NebulaError(f"PCA 投影失败: {exc}") from exc
    return np.asarray(coordinates, dtype=np.float64)


def umap_projection(
    matrix: Any,
    n_components: int,
    n_neighbors: int,
    *,
    metric: str = UMAP_METRIC,
    min_dist: float = UMAP_MIN_DIST,
    random_state: int = UMAP_RANDOM_STATE,
    n_jobs: int = UMAP_N_JOBS,
) -> tuple[np.ndarray, int]:
    """UMAP 投影（固定种子、单线程）；返回 (坐标, 实际使用的 n_neighbors)。

    小样本时 n_neighbors 自动限制为 n_samples - 1（UMAP 要求小于样本数）。
    """

    components = _positive_int(n_components, name="n_components", minimum=1)
    requested_neighbors = _positive_int(n_neighbors, name="n_neighbors", minimum=2)
    normalized = normalize_vectors(matrix, where="UMAP 输入")
    count = int(normalized.shape[0])
    if components >= count:
        raise NebulaError(
            f"UMAP 要求 n_components < n_samples，收到 {components} >= {count}"
        )
    effective_neighbors = min(requested_neighbors, count - 1)
    if effective_neighbors < 2:
        raise NebulaError(
            f"UMAP 需要至少 3 部作品，收到 {count}（n_neighbors 无法 >= 2）"
        )
    try:
        import umap
    except ImportError as exc:
        raise NebulaError(f"缺少 umap-learn 依赖，无法做 UMAP：{exc}") from exc
    try:
        reducer = umap.UMAP(
            n_components=components,
            n_neighbors=effective_neighbors,
            min_dist=float(min_dist),
            metric=str(metric),
            random_state=int(random_state),
            n_jobs=int(n_jobs),
            transform_seed=UMAP_TRANSFORM_SEED,
        )
        coordinates = reducer.fit_transform(normalized)
    except Exception as exc:  # noqa: BLE001 - 第三方库异常统一转 NebulaError
        raise NebulaError(f"UMAP 投影失败: {exc}") from exc
    array = np.asarray(coordinates, dtype=np.float64)
    if array.shape != (count, components):
        raise NebulaError(
            f"UMAP 输出形状异常: {array.shape} != ({count}, {components})"
        )
    if not np.all(np.isfinite(array)):
        raise NebulaError("UMAP 输出包含 NaN/Inf")
    return array, effective_neighbors


# --------------------------------------------------------------------------- 单表征诊断
@dataclass(frozen=True)
class ProjectionResult:
    """一个低维配置的投影坐标、低维距离、Top-k 与指标。"""

    method: str
    label: str
    n_components: int
    n_neighbors: int | None
    effective_n_neighbors: int | None
    coordinates: np.ndarray
    distances: np.ndarray
    topk: np.ndarray
    metrics: dict[str, Any]


@dataclass(frozen=True)
class RepresentationDiagnostics:
    """一个表征的完整诊断结果（报告字典 + 可复用坐标）。"""

    space: SpaceMatrix
    high_distance: np.ndarray
    high_topk: np.ndarray
    projections: tuple[ProjectionResult, ...]
    umap_neighbor_consistency: tuple[dict[str, Any], ...]
    dimension_consistency: tuple[dict[str, Any], ...]
    report: dict[str, Any]


def _high_dim_summary(distance: np.ndarray) -> dict[str, Any]:
    count = int(distance.shape[0])
    if count < 2:
        return {
            "metric": "cosine_distance",
            "n_pairs": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    rows, columns = np.triu_indices(count, k=1)
    values = distance[rows, columns]
    return {
        "metric": "cosine_distance",
        "n_pairs": int(values.size),
        "mean": round(float(values.mean()), 6),
        "median": round(float(np.median(values)), 6),
        "min": round(float(values.min()), 6),
        "max": round(float(values.max()), 6),
    }


def _projection_report(projection: ProjectionResult) -> dict[str, Any]:
    metrics = projection.metrics
    return {
        "method": projection.method,
        "label": projection.label,
        "n_components": int(projection.n_components),
        "n_neighbors": projection.n_neighbors,
        "effective_n_neighbors": projection.effective_n_neighbors,
        "effective_k": int(metrics["effective_k"]),
        "topk_preservation": float(metrics["topk_preservation"]),
        "top1_preservation": float(metrics["top1_preservation"]),
        "mean_overlap": float(metrics["mean_overlap"]),
        "stable_ratio": float(metrics["stable_ratio"]),
        "distance_pearson": float(metrics["pearson"]),
        "distance_spearman": float(metrics["spearman"]),
        "n_pairs": int(metrics["n_pairs"]),
    }


def analyze_space(
    space: SpaceMatrix,
    *,
    k: int = DEFAULT_K,
    umap_neighbors: Sequence[int] = DEFAULT_UMAP_NEIGHBORS,
    components: Sequence[int] = DEFAULT_COMPONENTS,
    random_state: int = PCA_RANDOM_STATE,
) -> RepresentationDiagnostics:
    """对一个表征计算高维距离、PCA/UMAP 投影、保持率与一致性指标。"""

    k_value = _validated_k(k)
    component_list = _validated_components(components)
    neighbor_list = _validated_neighbors(umap_neighbors)
    count = space.n_works
    min_required = max(component_list) + 1
    if count < min_required:
        raise NebulaError(
            f"{space.name} 只有 {count} 部作品，至少需要 {min_required} 部"
            f"（components={list(component_list)}）"
        )

    high_distance = cosine_distance_matrix(space.matrix)
    high_topk = topk_neighbors(high_distance, k_value)

    projections: list[ProjectionResult] = []
    by_key: dict[tuple[str, int | None, int], ProjectionResult] = {}

    for component_count in component_list:
        coordinates = pca_projection(
            space.matrix, component_count, random_state=random_state
        )
        distances = euclidean_distance_matrix(coordinates)
        topk = topk_neighbors(distances, k_value)
        metrics = {
            **neighborhood_metrics(high_topk, topk),
            **pair_distance_correlation(high_distance, distances),
        }
        projection = ProjectionResult(
            method="pca",
            label=f"pca_{component_count}d",
            n_components=component_count,
            n_neighbors=None,
            effective_n_neighbors=None,
            coordinates=coordinates,
            distances=distances,
            topk=topk,
            metrics=metrics,
        )
        projections.append(projection)
        by_key[("pca", None, component_count)] = projection

    for neighbor_count in neighbor_list:
        for component_count in component_list:
            coordinates, effective_neighbors = umap_projection(
                space.matrix,
                component_count,
                neighbor_count,
                random_state=random_state,
            )
            distances = euclidean_distance_matrix(coordinates)
            topk = topk_neighbors(distances, k_value)
            metrics = {
                **neighborhood_metrics(high_topk, topk),
                **pair_distance_correlation(high_distance, distances),
            }
            projection = ProjectionResult(
                method="umap",
                label=f"umap_n{neighbor_count}_{component_count}d",
                n_components=component_count,
                n_neighbors=neighbor_count,
                effective_n_neighbors=effective_neighbors,
                coordinates=coordinates,
                distances=distances,
                topk=topk,
                metrics=metrics,
            )
            projections.append(projection)
            by_key[("umap", neighbor_count, component_count)] = projection

    umap_neighbor_consistency: list[dict[str, Any]] = []
    for first_index in range(len(neighbor_list)):
        for second_index in range(first_index + 1, len(neighbor_list)):
            first_neighbors = neighbor_list[first_index]
            second_neighbors = neighbor_list[second_index]
            for component_count in component_list:
                first = by_key[("umap", first_neighbors, component_count)]
                second = by_key[("umap", second_neighbors, component_count)]
                metrics = neighborhood_metrics(first.topk, second.topk)
                umap_neighbor_consistency.append(
                    {
                        "components": int(component_count),
                        "a": int(first_neighbors),
                        "b": int(second_neighbors),
                        "topk_preservation": float(metrics["topk_preservation"]),
                        "top1_preservation": float(metrics["top1_preservation"]),
                        "mean_overlap": float(metrics["mean_overlap"]),
                        "effective_k": int(metrics["effective_k"]),
                    }
                )

    dimension_consistency: list[dict[str, Any]] = []
    if 2 in component_list and 3 in component_list:
        pca_two = by_key[("pca", None, 2)]
        pca_three = by_key[("pca", None, 3)]
        metrics = neighborhood_metrics(pca_two.topk, pca_three.topk)
        dimension_consistency.append(
            {
                "method": "pca",
                "n_neighbors": None,
                "components_a": 2,
                "components_b": 3,
                "topk_preservation": float(metrics["topk_preservation"]),
                "top1_preservation": float(metrics["top1_preservation"]),
                "mean_overlap": float(metrics["mean_overlap"]),
                "effective_k": int(metrics["effective_k"]),
            }
        )
        for neighbor_count in neighbor_list:
            two_dim = by_key[("umap", neighbor_count, 2)]
            three_dim = by_key[("umap", neighbor_count, 3)]
            metrics = neighborhood_metrics(two_dim.topk, three_dim.topk)
            dimension_consistency.append(
                {
                    "method": "umap",
                    "n_neighbors": int(neighbor_count),
                    "components_a": 2,
                    "components_b": 3,
                    "topk_preservation": float(metrics["topk_preservation"]),
                    "top1_preservation": float(metrics["top1_preservation"]),
                    "mean_overlap": float(metrics["mean_overlap"]),
                    "effective_k": int(metrics["effective_k"]),
                }
            )

    report = {
        "name": space.name,
        "label": space.label,
        "sequence_version": space.sequence_version,
        "representation_id": space.representation_id,
        "n_works": count,
        "dim": space.dim,
        "high_dim": _high_dim_summary(high_distance),
        "projections": [_projection_report(projection) for projection in projections],
        "umap_neighbor_consistency": umap_neighbor_consistency,
        "dimension_consistency": dimension_consistency,
    }
    return RepresentationDiagnostics(
        space=space,
        high_distance=high_distance,
        high_topk=high_topk,
        projections=tuple(projections),
        umap_neighbor_consistency=tuple(umap_neighbor_consistency),
        dimension_consistency=tuple(dimension_consistency),
        report=report,
    )


# --------------------------------------------------------------------------- 多表征
@dataclass(frozen=True)
class DiagnosticsResult:
    """一次诊断的完整结果（报告 + 每个表征的坐标/指标）。"""

    report: dict[str, Any]
    analyses: tuple[RepresentationDiagnostics, ...]


def run_diagnostics(
    catalog: Catalog,
    config: Config,
    *,
    representations: Sequence[str] | None = None,
    sequence_version: str | None = None,
    kind: str = DEFAULT_KIND,
    k: int = DEFAULT_K,
    umap_neighbors: Sequence[int] = DEFAULT_UMAP_NEIGHBORS,
    components: Sequence[int] = DEFAULT_COMPONENTS,
    random_state: int = PCA_RANDOM_STATE,
) -> DiagnosticsResult:
    """读取多个表征并做低维投影诊断；不修改数据库。"""

    names = _validated_names(
        DEFAULT_REPRESENTATIONS if representations is None else representations
    )
    kind_value = _validated_kind(kind)
    k_value = _validated_k(k)
    neighbor_list = _validated_neighbors(umap_neighbors)
    component_list = _validated_components(components)
    analyses: list[RepresentationDiagnostics] = []
    for name in names:
        space = load_space(
            catalog, config, name, sequence_version=sequence_version, kind=kind_value
        )
        analyses.append(
            analyze_space(
                space,
                k=k_value,
                umap_neighbors=neighbor_list,
                components=component_list,
                random_state=random_state,
            )
        )
    report = {
        "tool": TOOL_NAME,
        "version": TOOL_VERSION,
        "generated_at": now_utc_iso(),
        "params": {
            "kind": kind_value,
            "k": int(k_value),
            "umap_neighbors": [int(value) for value in neighbor_list],
            "components": [int(value) for value in component_list],
            "representations": list(names),
            "pca_random_state": int(random_state),
            "umap": {
                "metric": UMAP_METRIC,
                "min_dist": UMAP_MIN_DIST,
                "random_state": UMAP_RANDOM_STATE,
                "n_jobs": UMAP_N_JOBS,
            },
        },
        "representations": [analysis.report for analysis in analyses],
    }
    return DiagnosticsResult(report=report, analyses=tuple(analyses))


# --------------------------------------------------------------------------- 输出
CSV_COLUMNS: tuple[str, ...] = (
    "representation",
    "sequence_version",
    "representation_id",
    "n_works",
    "dim",
    "projection",
    "method",
    "components",
    "n_neighbors",
    "effective_n_neighbors",
    "effective_k",
    "topk_preservation",
    "top1_preservation",
    "mean_overlap",
    "stable_ratio",
    "distance_pearson",
    "distance_spearman",
    "n_pairs",
)


def _report_of(report_or_result: DiagnosticsResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(report_or_result, DiagnosticsResult):
        return report_or_result.report
    if isinstance(report_or_result, Mapping):
        return dict(report_or_result)
    raise NebulaError(f"无法识别诊断结果类型: {type(report_or_result).__name__}")


def _atomic_text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def write_reports(
    report_or_result: DiagnosticsResult | Mapping[str, Any],
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> tuple[Path, Path]:
    """原子写 JSON + CSV（一个低维配置一行）；返回 (json_path, csv_path)。"""

    report = _report_of(report_or_result)
    json_target = Path(json_path)
    csv_target = Path(csv_path)
    _atomic_text_write(
        json_target,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
    )

    csv_target.parent.mkdir(parents=True, exist_ok=True)
    temp_csv = csv_target.with_name(csv_target.name + ".tmp")
    with temp_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for representation in report.get("representations", []):
            for projection in representation.get("projections", []):
                writer.writerow(
                    {
                        "representation": representation.get("name"),
                        "sequence_version": representation.get("sequence_version"),
                        "representation_id": representation.get("representation_id"),
                        "n_works": representation.get("n_works"),
                        "dim": representation.get("dim"),
                        "projection": projection.get("label"),
                        "method": projection.get("method"),
                        "components": projection.get("n_components"),
                        "n_neighbors": projection.get("n_neighbors"),
                        "effective_n_neighbors": projection.get(
                            "effective_n_neighbors"
                        ),
                        "effective_k": projection.get("effective_k"),
                        "topk_preservation": projection.get("topk_preservation"),
                        "top1_preservation": projection.get("top1_preservation"),
                        "mean_overlap": projection.get("mean_overlap"),
                        "stable_ratio": projection.get("stable_ratio"),
                        "distance_pearson": projection.get("distance_pearson"),
                        "distance_spearman": projection.get("distance_spearman"),
                        "n_pairs": projection.get("n_pairs"),
                    }
                )
    os.replace(temp_csv, csv_target)
    return json_target, csv_target


def _work_hover_text(
    work_ids: Sequence[int], works: Mapping[int, Mapping[str, Any]] | None
) -> list[str]:
    labels: list[str] = []
    for work_id in work_ids:
        meta = (works or {}).get(int(work_id), {})
        title = meta.get("title")
        if title:
            labels.append(f"work_id={int(work_id)} | {title}")
        else:
            labels.append(f"work_id={int(work_id)}")
    return labels


def write_plotly_html(
    result: DiagnosticsResult,
    html_dir: str | Path,
    *,
    works: Mapping[int, Mapping[str, Any]] | None = None,
) -> list[Path]:
    """为每个表征的每个低维配置写一个 Plotly HTML（可选输出）。"""

    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - 项目依赖已包含 plotly
        raise NebulaError(f"缺少 plotly 依赖，无法写 HTML：{exc}") from exc

    target_dir = Path(html_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for analysis in result.analyses:
        hover_text = _work_hover_text(analysis.space.work_ids, works)
        for projection in analysis.projections:
            coordinates = projection.coordinates
            if projection.n_components == 2:
                trace = go.Scatter(
                    x=coordinates[:, 0],
                    y=coordinates[:, 1],
                    mode="markers",
                    text=hover_text,
                    hoverinfo="text",
                    marker={"size": 6, "opacity": 0.85},
                )
            else:
                trace = go.Scatter3d(
                    x=coordinates[:, 0],
                    y=coordinates[:, 1],
                    z=coordinates[:, 2],
                    mode="markers",
                    text=hover_text,
                    hoverinfo="text",
                    marker={"size": 4, "opacity": 0.85},
                )
            figure = go.Figure(data=[trace])
            figure.update_layout(
                title=(
                    f"{analysis.space.label} · {projection.label} · "
                    f"Top{projection.metrics['effective_k']} 保持率="
                    f"{projection.metrics['topk_preservation']:.3f}"
                ),
                margin={"l": 40, "r": 20, "t": 50, "b": 40},
            )
            filename = f"{slugify(analysis.space.name)}__{projection.label}.html"
            path = target_dir / filename
            temp = path.with_name(path.name + ".tmp")
            try:
                figure.write_html(
                    str(temp),
                    include_plotlyjs="cdn",
                    auto_open=False,
                    full_html=True,
                )
            except (OSError, ValueError) as exc:
                raise NebulaError(f"写 Plotly HTML 失败（{path}）：{exc}") from exc
            os.replace(temp, path)
            paths.append(path)
    return paths


def _print_summary(
    result: DiagnosticsResult,
    *,
    json_path: Path,
    csv_path: Path,
    html_paths: Sequence[Path] = (),
) -> None:
    report = result.report
    params = report["params"]
    print("低维投影诊断（v0.4.3：PCA / UMAP 邻域保持率与距离相关）")
    print(
        f"  k={params['k']}  components={params['components']}  "
        f"umap_neighbors={params['umap_neighbors']}  kind={params['kind']}"
    )
    for analysis in result.analyses:
        space = analysis.space
        print(
            f"\n  {space.label} [{space.name}@{space.sequence_version}]"
            f"  n_works={space.n_works}  dim={space.dim}"
        )
        print(
            "    projection          topk_pres  top1_pres  "
            "pearson  spearman  n_pairs"
        )
        for projection in analysis.projections:
            metrics = projection.metrics
            print(
                f"    {projection.label:<20}"
                f"{metrics['topk_preservation']:>9.4f}  "
                f"{metrics['top1_preservation']:>9.4f}  "
                f"{metrics['pearson']:>8.4f}  "
                f"{metrics['spearman']:>8.4f}  "
                f"{metrics['n_pairs']:>7d}"
            )
        if analysis.umap_neighbor_consistency:
            pairs = ", ".join(
                f"{item['a']}↔{item['b']}/{item['components']}D="
                f"{item['topk_preservation']:.3f}"
                for item in analysis.umap_neighbor_consistency
            )
            print(f"    UMAP 邻居一致性: {pairs}")
        if analysis.dimension_consistency:
            pairs = ", ".join(
                f"{item['method']}"
                f"{'' if item['n_neighbors'] is None else '_n' + str(item['n_neighbors'])}"
                f" 2D↔3D={item['topk_preservation']:.3f}"
                for item in analysis.dimension_consistency
            )
            print(f"    2D vs 3D 保持率: {pairs}")
    print(f"\n  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    if html_paths:
        print(f"  HTML: {len(html_paths)} 个文件 -> {html_paths[0].parent}")


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/embedding_diagnostics.py",
        description=(
            "v0.4.3 低维投影诊断：PCA/UMAP 2D/3D 的 Top-k 邻域保持率、"
            "Top1 保持率与高低维距离相关。只读已登记表征，不写数据库。"
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="配置文件路径（默认取 NEBULA_CONFIG 环境变量，否则当前目录 config.yaml）",
    )
    parser.add_argument(
        "--representation",
        nargs="+",
        action="extend",
        default=None,
        help=(
            "要诊断的表征名（可多值/可重复；默认 "
            + ",".join(DEFAULT_REPRESENTATIONS)
            + "）"
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Top-k 邻域（默认 {DEFAULT_K}）",
    )
    parser.add_argument(
        "--umap-neighbors",
        nargs="+",
        type=int,
        default=None,
        help=f"UMAP n_neighbors 列表（默认 {list(DEFAULT_UMAP_NEIGHBORS)}）",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        type=int,
        default=None,
        help=f"低维分量列表（默认 {list(DEFAULT_COMPONENTS)}；只支持 2/3）",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="JSON 输出路径（默认 <logs>/dim_diagnostics.json）",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="CSV 输出路径（默认与 JSON 同目录同名 .csv）",
    )
    parser.add_argument(
        "--html-dir",
        default=None,
        help="可选：为每个低维配置写 Plotly HTML 的目录（默认不写）",
    )
    return parser


def _resolve_config_path(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    env_value = os.environ.get("NEBULA_CONFIG")
    if env_value:
        return Path(env_value).expanduser()
    return Path.cwd() / "config.yaml"


def _collect_work_meta(
    catalog: Catalog, result: DiagnosticsResult
) -> dict[int, dict[str, Any]]:
    work_ids: set[int] = set()
    for analysis in result.analyses:
        work_ids.update(int(work_id) for work_id in analysis.space.work_ids)
    meta: dict[int, dict[str, Any]] = {}
    for work_id in sorted(work_ids):
        work = catalog.get_work(work_id) or {}
        meta[work_id] = {"work_id": work_id, "title": work.get("title")}
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(_resolve_config_path(args.config))
    except NebulaError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2
    configure_logging(config.logging.level)

    catalog = Catalog(config.paths.db_path)
    try:
        catalog.initialize()
        result = run_diagnostics(
            catalog,
            config,
            representations=args.representation,
            k=args.k,
            umap_neighbors=(
                args.umap_neighbors
                if args.umap_neighbors is not None
                else DEFAULT_UMAP_NEIGHBORS
            ),
            components=(
                args.components if args.components is not None else DEFAULT_COMPONENTS
            ),
        )
        work_meta = _collect_work_meta(catalog, result)
    except NebulaError as exc:
        print(f"低维投影诊断失败: {exc}", file=sys.stderr)
        return 1
    finally:
        catalog.close()

    json_path = (
        Path(args.out).expanduser()
        if args.out
        else config.paths.logs / "dim_diagnostics.json"
    )
    csv_path = (
        Path(args.csv).expanduser() if args.csv else json_path.with_suffix(".csv")
    )
    try:
        write_reports(result, json_path=json_path, csv_path=csv_path)
        html_paths = (
            write_plotly_html(result, Path(args.html_dir), works=work_meta)
            if args.html_dir
            else []
        )
    except NebulaError as exc:
        print(f"写入诊断报告失败: {exc}", file=sys.stderr)
        return 1

    _print_summary(result, json_path=json_path, csv_path=csv_path, html_paths=html_paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
