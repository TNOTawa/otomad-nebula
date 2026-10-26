r"""v0.4.4 跨模型近邻共识图与轻量融合评估。

回答的问题：多个模型（PE-AV 全片表征 + CLIP/DINOv2 密集视觉表征）共同认可的
相似作品，是否比单一模型的近邻更有解释力？本工具只读取已登记的作品级表征，
不训练模型、不加载 PE-AV、不写数据库、不访问网络。

核心定义（每个空间独立计算）：

- 空间：一个已登记表征的 canonical 作品级向量矩阵（load_representation_matrix，
  自动排除 work_links duplicate，按 work_id 排序）；
- 高维余弦 Top-k 有向邻接：每部作品取余弦相似度最高的 k 个邻居（默认 k=5）；
- mutual 对：a 在 b 的 Top-k 且 b 在 a 的 Top-k（无序作品对）；
- 共识边：在 >= min_spaces 个空间中互为 Top-k 的无序对（默认 min_spaces=2）；
  在全部空间 mutual 的边称全共识边；
- 共识强度：mutual 空间数 / 空间数（0..1）。

输出：

- logs/neighbor_consensus.json：meta/representations/params/spaces/mutual_edges/consensus/
  nodes/edges/consensus_edges/fusion；
- logs/neighbor_consensus.csv：边表（work_a/work_b、共识数/强度、每空间 rank/cos/mutual）；
- logs/neighbor_consensus_nodes.csv：节点表（每空间 mutual 度数、共识度数等）。

轻量融合基线（只在内存中评估，不写 DB、不注册正式表征）：

1. rank-average：每空间 Top-k 排名（缺失记 k+1）取平均，按平均排名取 Top-k；
   与每个单空间 Top-k 以及共识边分别计算 Jaccard；
2. concat 向量融合：每空间向量 L2 归一化后按块权重（等权或 1/sqrt(dim)）拼接，
   再整体 L2 归一化，计算融合空间 Top-k 与单空间/共识的 Jaccard、平均近邻余弦。

CLI 示例（项目根目录）：

    python tools/build_consensus_graph.py
    python tools/build_consensus_graph.py --representations full_mean_v1 dense_visual_clip_v1 dense_visual_dino_v1 --k 5
    python tools/build_consensus_graph.py --min-spaces 3 --fusion concat
    python tools/build_consensus_graph.py --out logs/neighbor_consensus.json --csv logs/neighbor_consensus.csv

默认表征为 full_mean_v1（PE-AV）与 dense_visual_clip_v2 / dense_visual_dino_v2
（v0.4.3.1 重建后的密集视觉表征）；在 v2 尚未重建时可用 --representations
显式传现有 v1 名称做开发验证。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from nebula.log import configure_logging  # noqa: E402
from nebula.representations import (  # noqa: E402
    load_representation_matrix,
    representation_label,
    resolve_representation,
)
from nebula.util import now_utc_iso  # noqa: E402

TOOL_NAME = "neighbor_consensus"
TOOL_VERSION = "v0.4.4"
NEWLINE = chr(10)

#: 默认参与共识图的空间：PE-AV Full Mean + v0.4.3.1 重建后的两种密集视觉表征
DEFAULT_REPRESENTATIONS: tuple[str, ...] = (
    "full_mean_v1",
    "dense_visual_clip_v2",
    "dense_visual_dino_v2",
)
DEFAULT_K = 5
DEFAULT_MIN_SPACES = 2
FUSION_MODES: tuple[str, ...] = ("none", "rank", "concat", "both")
DEFAULT_FUSION = "both"
DEFAULT_KIND = "joint"
VECTOR_KINDS: tuple[str, ...] = ("joint", "audio", "video")

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


def _validated_min_spaces(min_spaces: int, *, n_spaces: int) -> int:
    value = _positive_int(min_spaces, name="min_spaces", minimum=2)
    if value > n_spaces:
        raise NebulaError(
            f"min_spaces={value} 大于空间数 {n_spaces}；无法形成共识边"
        )
    return value


def _validated_fusion(fusion: str) -> str:
    text = str(fusion or "").strip()
    if text not in FUSION_MODES:
        raise NebulaError(f"fusion 必须是 {FUSION_MODES} 之一，收到 {fusion!r}")
    return text


def _validated_kind(kind: str) -> str:
    if kind not in VECTOR_KINDS:
        raise NebulaError(f"未知向量种类: {kind}（支持 {VECTOR_KINDS}）")
    return kind


def _validated_names(names: Sequence[str]) -> tuple[str, ...]:
    if not names:
        raise NebulaError("至少需要一个 --representations")
    result: list[str] = []
    for name in names:
        text = str(name or "").strip()
        if not text:
            raise NebulaError("--representations 不能为空")
        if text in result:
            raise NebulaError(f"--representations 重复: {text}")
        result.append(text)
    return tuple(result)


# --------------------------------------------------------------------------- 数据空间
def normalize_vectors(matrix: Any, *, where: str = "向量矩阵") -> np.ndarray:
    """校验二维矩阵并逐行 L2 归一化（float64）。"""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise NebulaError(f"{where} 形状非法: {values.shape}")
    if not np.all(np.isfinite(values)):
        raise NebulaError(f"{where} 包含 NaN/Inf")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    zero_rows = np.where(norms <= _EPS)[0]
    if zero_rows.size:
        raise NebulaError(f"{where} 有 {zero_rows.size} 个零向量")
    return values / norms


@dataclass(frozen=True)
class SpaceData:
    """一个表征空间的作品矩阵（行 = work_id 升序，列 = 高维特征，逐行 L2）。"""

    name: str
    label: str
    sequence_version: str
    representation_id: int
    source_kind: str
    work_ids: tuple[int, ...]
    matrix: np.ndarray
    n_works_total: int

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
        source_kind: str = "pe_av",
        work_ids: Sequence[int] | None = None,
        n_works_total: int | None = None,
    ) -> "SpaceData":
        normalized = normalize_vectors(matrix, where=f"{name} 向量矩阵")
        count = int(normalized.shape[0])
        ids = (
            tuple(int(value) for value in work_ids)
            if work_ids is not None
            else tuple(range(1, count + 1))
        )
        if len(ids) != count:
            raise NebulaError(f"{name} 的 work_ids 数量 {len(ids)} 与矩阵行数 {count} 不一致")
        if len(set(ids)) != len(ids):
            raise NebulaError(f"{name} 的 work_ids 有重复")
        total = count if n_works_total is None else int(n_works_total)
        if total < count:
            raise NebulaError(f"{name} 的 n_works_total={total} 小于行数 {count}")
        return cls(
            name=str(name),
            label=str(label or name),
            sequence_version=str(sequence_version),
            representation_id=int(representation_id),
            source_kind=str(source_kind),
            work_ids=ids,
            matrix=normalized,
            n_works_total=total,
        )


def load_space(
    catalog: Catalog,
    config: Config,
    name: str,
    *,
    kind: str = DEFAULT_KIND,
) -> SpaceData:
    """解析表征并加载其 canonical 作品级矩阵。"""

    kind_value = _validated_kind(kind)
    rep = resolve_representation(catalog, config, name=name)
    if str(rep.get("status")) != "ready":
        raise NebulaError(
            f"表征 {rep['name']}@{rep['sequence_version']} 状态为 {rep.get('status')}，"
            "需要 ready 才能构建共识图"
        )
    work_ids, matrix = load_representation_matrix(catalog, config, rep, kind=kind_value)
    normalized = normalize_vectors(
        matrix, where=f"{rep['name']}@{rep['sequence_version']} 向量矩阵"
    )
    return SpaceData(
        name=str(rep["name"]),
        label=representation_label(str(rep["name"])),
        sequence_version=str(rep["sequence_version"]),
        representation_id=int(rep["id"]),
        source_kind=str(rep.get("source_kind", "pe_av")),
        work_ids=tuple(int(work_id) for work_id in work_ids),
        matrix=normalized,
        n_works_total=len(work_ids),
    )


def align_spaces(spaces: Sequence[SpaceData]) -> tuple[SpaceData, ...]:
    """取所有空间的共同 canonical 作品（按 work_id 排序）并对齐矩阵。"""

    if len(spaces) < 2:
        raise NebulaError("至少需要 2 个空间才能构建共识图")
    names = [space.name for space in spaces]
    if len(set(names)) != len(names):
        raise NebulaError(f"空间名重复: {names}")
    common = set(spaces[0].work_ids)
    for space in spaces[1:]:
        common &= set(space.work_ids)
    common_ids = tuple(sorted(common))
    if len(common_ids) < 2:
        raise NebulaError(
            f"所有空间的共同 canonical 作品不足 2 部（{len(common_ids)} 部）"
        )
    aligned: list[SpaceData] = []
    for space in spaces:
        index = {int(work_id): position for position, work_id in enumerate(space.work_ids)}
        rows = np.asarray([index[work_id] for work_id in common_ids], dtype=np.int64)
        aligned.append(
            SpaceData(
                name=space.name,
                label=space.label,
                sequence_version=space.sequence_version,
                representation_id=space.representation_id,
                source_kind=space.source_kind,
                work_ids=common_ids,
                matrix=np.ascontiguousarray(space.matrix[rows]),
                n_works_total=space.n_works_total,
            )
        )
    return tuple(aligned)


# --------------------------------------------------------------------------- 邻接
def cosine_similarity_matrix(matrix: Any) -> np.ndarray:
    """逐行归一化后的余弦相似度矩阵（对称、对角线 1）。"""

    normalized = normalize_vectors(matrix, where="余弦相似度输入")
    similarity = normalized @ normalized.T
    np.clip(similarity, -1.0, 1.0, out=similarity)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def topk_neighbors(similarity: Any, k: int) -> np.ndarray:
    """每行 Top-k 邻居索引（不含自身；相似度降序，并列按索引稳定排序）。"""

    values = np.asarray(similarity, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise NebulaError(f"相似度矩阵必须是方阵，收到 {values.shape}")
    if not np.all(np.isfinite(values)):
        raise NebulaError("相似度矩阵包含 NaN/Inf")
    k_value = _validated_k(k)
    count = int(values.shape[0])
    effective = min(k_value, max(count - 1, 0))
    if effective == 0:
        return np.zeros((count, 0), dtype=np.int64)
    work = values.copy()
    np.fill_diagonal(work, -np.inf)
    order = np.argsort(-work, axis=1, kind="stable")
    return order[:, :effective].astype(np.int64)


@dataclass(frozen=True)
class SpaceTopology:
    """一个空间的 Top-k 有向边、mutual 对与统计量。"""

    space: SpaceData
    similarity: np.ndarray
    neighbors: np.ndarray
    work_sets: tuple[frozenset[int], ...]
    directed: frozenset[tuple[int, int]]
    mutual: frozenset[tuple[int, int]]
    rank_by_source: tuple[dict[int, int], ...]
    mean_topk_cosine: float

    @property
    def n_works(self) -> int:
        return self.space.n_works

    @property
    def effective_k(self) -> int:
        return int(self.neighbors.shape[1])


def build_topology(space: SpaceData, k: int) -> SpaceTopology:
    """计算一个空间的 Top-k 有向邻接与 mutual 对。"""

    similarity = cosine_similarity_matrix(space.matrix)
    neighbors = topk_neighbors(similarity, k)
    ids = space.work_ids
    count = len(ids)
    work_sets: list[frozenset[int]] = []
    for row in range(count):
        work_sets.append(frozenset(ids[int(index)] for index in neighbors[row]))
    directed: set[tuple[int, int]] = set()
    rank_by_source: list[dict[int, int]] = []
    cosine_values: list[float] = []
    for row in range(count):
        ranks: dict[int, int] = {}
        row_values: list[float] = []
        for position, index in enumerate(neighbors[row]):
            target = ids[int(index)]
            ranks[target] = position + 1
            directed.add((ids[row], target))
            row_values.append(float(similarity[row, int(index)]))
        rank_by_source.append(ranks)
        if row_values:
            cosine_values.append(float(np.mean(row_values)))
    mutual: set[tuple[int, int]] = set()
    for row in range(count):
        for index in neighbors[row]:
            target_row = int(index)
            if ids[row] in work_sets[target_row]:
                first, second = ids[row], ids[target_row]
                if first > second:
                    first, second = second, first
                mutual.add((first, second))
    mean_topk_cosine = float(np.mean(cosine_values)) if cosine_values else 0.0
    return SpaceTopology(
        space=space,
        similarity=similarity,
        neighbors=neighbors,
        work_sets=tuple(work_sets),
        directed=frozenset(directed),
        mutual=frozenset(mutual),
        rank_by_source=tuple(rank_by_source),
        mean_topk_cosine=mean_topk_cosine,
    )


def _pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _jaccard(first: set[Any] | frozenset[Any], second: set[Any] | frozenset[Any]) -> float:
    union = first | second
    if not union:
        return 1.0
    return len(first & second) / len(union)


# --------------------------------------------------------------------------- 共识图
def build_consensus(
    spaces: Sequence[SpaceData],
    *,
    k: int = DEFAULT_K,
    min_spaces: int = DEFAULT_MIN_SPACES,
    fusion: str = DEFAULT_FUSION,
    kind: str = DEFAULT_KIND,
    works_meta: Mapping[int, Mapping[str, Any]] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """构建跨空间近邻共识图与轻量融合评估（纯函数，确定性输出）。"""

    k_value = _validated_k(k)
    fusion_value = _validated_fusion(fusion)
    kind_value = _validated_kind(kind)
    aligned = align_spaces(spaces)
    n_spaces = len(aligned)
    min_spaces_value = _validated_min_spaces(min_spaces, n_spaces=n_spaces)
    common_ids = aligned[0].work_ids
    count = len(common_ids)
    effective_k = min(k_value, max(count - 1, 0))
    index_by_id = {work_id: position for position, work_id in enumerate(common_ids)}

    topologies: list[SpaceTopology] = [
        build_topology(space, k_value) for space in aligned
    ]

    # 逐空间统计
    space_stats: list[dict[str, Any]] = []
    for topology in topologies:
        space = topology.space
        mutual_edges = len(topology.mutual)
        directed_edges = len(topology.directed)
        space_stats.append(
            {
                "name": space.name,
                "label": space.label,
                "sequence_version": space.sequence_version,
                "representation_id": space.representation_id,
                "source_kind": space.source_kind,
                "n_works_total": int(space.n_works_total),
                "n_works_common": int(space.n_works),
                "dim": int(space.dim),
                "directed_edges": int(directed_edges),
                "mutual_edges": int(mutual_edges),
                "mutual_ratio": round(
                    (2.0 * mutual_edges / directed_edges) if directed_edges else 0.0, 6
                ),
                "mean_topk_cosine": round(float(topology.mean_topk_cosine), 6),
            }
        )

    # 跨空间保留率（有向边与 mutual 对）
    cross_space: list[dict[str, Any]] = []
    for first_index in range(n_spaces):
        for second_index in range(first_index + 1, n_spaces):
            first = topologies[first_index]
            second = topologies[second_index]
            directed_intersection = len(first.directed & second.directed)
            directed_union = len(first.directed | second.directed)
            mutual_intersection = len(first.mutual & second.mutual)
            mutual_union = len(first.mutual | second.mutual)
            cross_space.append(
                {
                    "a": first.space.name,
                    "b": second.space.name,
                    "directed_intersection": int(directed_intersection),
                    "directed_union": int(directed_union),
                    "directed_jaccard": round(
                        directed_intersection / directed_union if directed_union else 0.0, 6
                    ),
                    "directed_retention_a_to_b": round(
                        directed_intersection / len(first.directed)
                        if first.directed
                        else 0.0,
                        6,
                    ),
                    "directed_retention_b_to_a": round(
                        directed_intersection / len(second.directed)
                        if second.directed
                        else 0.0,
                        6,
                    ),
                    "mutual_intersection": int(mutual_intersection),
                    "mutual_union": int(mutual_union),
                    "mutual_jaccard": round(
                        mutual_intersection / mutual_union if mutual_union else 0.0, 6
                    ),
                    "mutual_retention_a_to_b": round(
                        mutual_intersection / len(first.mutual) if first.mutual else 0.0,
                        6,
                    ),
                    "mutual_retention_b_to_a": round(
                        mutual_intersection / len(second.mutual) if second.mutual else 0.0,
                        6,
                    ),
                }
            )

    # 每个空间对每部作品的排名表：work_id -> {neighbor: rank}
    rank_maps: list[dict[int, dict[int, int]]] = []
    for topology in topologies:
        mapping: dict[int, dict[int, int]] = {}
        for position, work_id in enumerate(common_ids):
            mapping[int(work_id)] = dict(topology.rank_by_source[position])
        rank_maps.append(mapping)

    # 所有空间 Top-k 有向边的无序并集
    union_pairs: set[tuple[int, int]] = set()
    for topology in topologies:
        for work_a, work_b in topology.directed:
            union_pairs.add(_pair(int(work_a), int(work_b)))

    mutual_sets = [set(topology.mutual) for topology in topologies]
    edges: list[dict[str, Any]] = []
    for work_a, work_b in sorted(union_pairs):
        mutual_count = sum(1 for mutual in mutual_sets if (work_a, work_b) in mutual)
        mutual_space_names = [
            topology.space.name
            for topology, mutual in zip(topologies, mutual_sets, strict=True)
            if (work_a, work_b) in mutual
        ]
        per_space: dict[str, Any] = {}
        for space_index, topology in enumerate(topologies):
            space = topology.space
            position_a = index_by_id[work_a]
            position_b = index_by_id[work_b]
            per_space[space.name] = {
                "rank_a": rank_maps[space_index][work_a].get(work_b),
                "rank_b": rank_maps[space_index][work_b].get(work_a),
                "cos": round(
                    float(topology.similarity[position_a, position_b]), 6
                ),
                "mutual": bool((work_a, work_b) in mutual_sets[space_index]),
            }
        edges.append(
            {
                "work_id_a": int(work_a),
                "work_id_b": int(work_b),
                "consensus_count": int(mutual_count),
                "consensus_strength": round(mutual_count / n_spaces, 6),
                "is_consensus": bool(mutual_count >= min_spaces_value),
                "is_full_consensus": bool(mutual_count == n_spaces),
                "mutual_spaces": mutual_space_names,
                "per_space": per_space,
            }
        )

    edges.sort(key=lambda item: (-item["consensus_count"], item["work_id_a"], item["work_id_b"]))
    consensus_edges = [edge for edge in edges if edge["is_consensus"]]
    full_consensus_edges = [edge for edge in edges if edge["is_full_consensus"]]
    histogram: dict[str, int] = {str(value): 0 for value in range(n_spaces + 1)}
    for edge in edges:
        histogram[str(edge["consensus_count"])] += 1

    # 节点统计
    per_space_mutual_degree: dict[str, dict[int, int]] = {}
    for topology in topologies:
        counter: dict[int, int] = {int(work_id): 0 for work_id in common_ids}
        for work_a, work_b in topology.mutual:
            counter[int(work_a)] += 1
            counter[int(work_b)] += 1
        per_space_mutual_degree[topology.space.name] = counter
    consensus_degree: dict[int, int] = {int(work_id): 0 for work_id in common_ids}
    full_consensus_degree: dict[int, int] = {int(work_id): 0 for work_id in common_ids}
    consensus_strength_sum: dict[int, float] = {int(work_id): 0.0 for work_id in common_ids}
    for edge in consensus_edges:
        work_a, work_b = edge["work_id_a"], edge["work_id_b"]
        consensus_degree[work_a] += 1
        consensus_degree[work_b] += 1
        consensus_strength_sum[work_a] += edge["consensus_strength"]
        consensus_strength_sum[work_b] += edge["consensus_strength"]
        if edge["is_full_consensus"]:
            full_consensus_degree[work_a] += 1
            full_consensus_degree[work_b] += 1
    consensus_pairs = {(_pair(edge["work_id_a"], edge["work_id_b"])) for edge in consensus_edges}
    consensus_works = sorted(
        {work_id for edge in consensus_edges for work_id in (edge["work_id_a"], edge["work_id_b"])}
    )

    meta = works_meta or {}
    nodes: list[dict[str, Any]] = []
    for work_id in common_ids:
        work_meta = meta.get(int(work_id), {})
        nodes.append(
            {
                "work_id": int(work_id),
                "title": work_meta.get("title"),
                "source": work_meta.get("source"),
                "mutual_degree": {
                    space.name: int(per_space_mutual_degree[space.name][int(work_id)])
                    for space in aligned
                },
                "consensus_degree": int(consensus_degree[int(work_id)]),
                "full_consensus_degree": int(full_consensus_degree[int(work_id)]),
                "consensus_strength_sum": round(
                    float(consensus_strength_sum[int(work_id)]), 6
                ),
            }
        )

    coverage = len(consensus_works)
    mean_strength = (
        float(np.mean([edge["consensus_strength"] for edge in consensus_edges]))
        if consensus_edges
        else 0.0
    )
    mean_degree = (
        float(np.mean([node["consensus_degree"] for node in nodes])) if nodes else 0.0
    )
    max_degree = max((node["consensus_degree"] for node in nodes), default=0)
    strength_values = [float(edge["consensus_strength"]) for edge in consensus_edges]
    strength_mean = float(np.mean(strength_values)) if strength_values else 0.0
    strength_std = float(np.std(strength_values)) if strength_values else 0.0
    strength_min = float(np.min(strength_values)) if strength_values else 0.0
    strength_max = float(np.max(strength_values)) if strength_values else 0.0
    full_consensus_works = sorted(
        {
            work_id
            for edge in full_consensus_edges
            for work_id in (edge["work_id_a"], edge["work_id_b"])
        }
    )
    full_coverage = len(full_consensus_works)

    representation_names = [space.name for space in aligned]
    graph: dict[str, Any] = {
        "n_spaces": int(n_spaces),
        "n_works_common": int(count),
        "k": int(k_value),
        "effective_k": int(effective_k),
        "min_spaces": int(min_spaces_value),
        "n_edges": int(len(edges)),
        "n_consensus_edges": int(len(consensus_edges)),
        "n_full_consensus_edges": int(len(full_consensus_edges)),
        "consensus_histogram": histogram,
        "consensus_coverage_works": int(coverage),
        "consensus_coverage_ratio": round(coverage / count if count else 0.0, 6),
        "full_consensus_coverage_works": int(full_coverage),
        "full_consensus_coverage_ratio": round(
            full_coverage / count if count else 0.0, 6
        ),
        "mean_consensus_strength": round(mean_strength, 6),
        "consensus_strength_stats": {
            "mean": round(strength_mean, 6),
            "std": round(strength_std, 6),
            "min": round(strength_min, 6),
            "max": round(strength_max, 6),
        },
        "mean_consensus_degree": round(mean_degree, 6),
        "max_consensus_degree": int(max_degree),
        "cross_space": cross_space,
    }
    report: dict[str, Any] = {
        "tool": TOOL_NAME,
        "version": TOOL_VERSION,
        "generated_at": generated_at,
        "meta": {
            "tool": TOOL_NAME,
            "version": TOOL_VERSION,
            "kind": kind_value,
            "spaces": representation_names,
            "generated_at": generated_at,
        },
        "representations": representation_names,
        "params": {
            "k": int(k_value),
            "min_spaces": int(min_spaces_value),
            "fusion": fusion_value,
            "kind": kind_value,
            "representations": representation_names,
        },
        "spaces": space_stats,
        "mutual_edges": {
            str(stat["name"]): int(stat["mutual_edges"]) for stat in space_stats
        },
        "consensus": graph,
        "nodes": nodes,
        "edges": edges,
        "consensus_edges": consensus_edges,
        "fusion": None,
    }

    if fusion_value in ("rank", "both"):
        report["fusion"] = {"rank_average": None, "concat": None}
        report["fusion"]["rank_average"] = rank_average_fusion(
            topologies, common_ids, k=k_value, consensus_pairs=consensus_pairs
        )
    if fusion_value in ("concat", "both"):
        if report["fusion"] is None:
            report["fusion"] = {"rank_average": None, "concat": None}
        concat: dict[str, Any] = {}
        for scheme in ("equal", "inv_sqrt_dim"):
            concat[scheme] = concat_fusion(
                aligned,
                topologies,
                common_ids,
                k=k_value,
                consensus_pairs=consensus_pairs,
                weight_scheme=scheme,
            )
        report["fusion"]["concat"] = concat
    return report


# --------------------------------------------------------------------------- 融合基线
def _fused_vs_spaces(
    fused_orders: Sequence[Sequence[int]],
    topologies: Sequence[SpaceTopology],
    common_ids: Sequence[int],
) -> dict[str, Any]:
    per_space: dict[str, Any] = {}
    all_jaccards: list[float] = []
    for topology in topologies:
        jaccards: list[float] = []
        overlaps: list[int] = []
        same_top1 = 0
        for position, _work_id in enumerate(common_ids):
            space_set = {int(value) for value in topology.work_sets[position]}
            fused_set = set(fused_orders[position])
            intersection = len(space_set & fused_set)
            jaccards.append(_jaccard(space_set, fused_set))
            overlaps.append(intersection)
            space_top1 = (
                int(common_ids[int(topology.neighbors[position, 0])])
                if topology.effective_k
                else None
            )
            fused_top1 = (
                int(fused_orders[position][0]) if fused_orders[position] else None
            )
            if space_top1 is not None and space_top1 == fused_top1:
                same_top1 += 1
        total = len(common_ids)
        per_space[topology.space.name] = {
            "mean_jaccard": round(float(np.mean(jaccards)), 6) if jaccards else 0.0,
            "mean_overlap": round(float(np.mean(overlaps)), 4) if overlaps else 0.0,
            "same_top1_ratio": round(same_top1 / total, 6) if total else 0.0,
        }
        all_jaccards.extend(jaccards)
    return {
        "per_space": per_space,
        "mean_jaccard_vs_spaces": round(float(np.mean(all_jaccards)), 6)
        if all_jaccards
        else 0.0,
    }


def _fused_vs_consensus(
    fused_sets: Sequence[set[int]],
    common_ids: Sequence[int],
    consensus_pairs: set[tuple[int, int]],
) -> dict[str, Any]:
    consensus_neighbors: dict[int, set[int]] = {int(work_id): set() for work_id in common_ids}
    for work_a, work_b in consensus_pairs:
        consensus_neighbors[int(work_a)].add(int(work_b))
        consensus_neighbors[int(work_b)].add(int(work_a))
    jaccards: list[float] = []
    evaluated = 0
    for position, work_id in enumerate(common_ids):
        reference = consensus_neighbors[int(work_id)]
        if not reference:
            continue
        fused = fused_sets[position]
        union = len(reference | fused)
        jaccards.append(len(reference & fused) / union if union else 1.0)
        evaluated += 1
    fused_pairs: set[tuple[int, int]] = set()
    for position, work_id in enumerate(common_ids):
        for target in fused_sets[position]:
            fused_pairs.add(_pair(int(work_id), int(target)))
    edge_union = fused_pairs | consensus_pairs
    edge_jaccard = (
        len(fused_pairs & consensus_pairs) / len(edge_union) if edge_union else 0.0
    )
    return {
        "n_consensus_pairs": int(len(consensus_pairs)),
        "n_works_with_consensus": int(evaluated),
        "n_works_evaluated": int(evaluated),
        "jaccard_mean_works_with_consensus": round(
            float(np.mean(jaccards)), 6
        )
        if jaccards
        else 0.0,
        "edge_jaccard": round(float(edge_jaccard), 6),
    }


def rank_average_fusion(
    topologies: Sequence[SpaceTopology],
    common_ids: Sequence[int],
    *,
    k: int,
    consensus_pairs: set[tuple[int, int]],
) -> dict[str, Any]:
    """rank-average 融合：每空间 Top-k 排名（缺失记 k+1）平均后取 Top-k。"""

    k_value = _validated_k(k)
    count = len(common_ids)
    effective = min(k_value, max(count - 1, 0))
    rank_maps: list[dict[int, dict[int, int]]] = []
    for topology in topologies:
        mapping: dict[int, dict[int, int]] = {}
        for position, work_id in enumerate(common_ids):
            mapping[int(work_id)] = dict(topology.rank_by_source[position])
        rank_maps.append(mapping)
    n_spaces = len(topologies)
    fused_orders: list[tuple[int, ...]] = []
    mean_scores: list[float] = []
    for work_id in common_ids:
        totals: dict[int, int] = {}
        appearances: dict[int, int] = {}
        for mapping in rank_maps:
            for target, rank in mapping[int(work_id)].items():
                totals[int(target)] = totals.get(int(target), 0) + int(rank)
                appearances[int(target)] = appearances.get(int(target), 0) + 1
        candidates: dict[int, float] = {
            target: (total + (k_value + 1) * (n_spaces - appearances[target])) / n_spaces
            for target, total in totals.items()
        }
        ordered = sorted(candidates.items(), key=lambda item: (item[1], item[0]))
        chosen = tuple(target for target, _score in ordered[:effective])
        fused_orders.append(chosen)
        if chosen:
            mean_scores.append(
                float(np.mean([candidates[target] for target in chosen]))
            )
    fused_sets = [set(order) for order in fused_orders]
    metrics = {
        "k": int(k_value),
        "effective_k": int(effective),
        "n_works": int(count),
        "n_spaces": int(n_spaces),
        "mean_fused_rank_score": round(float(np.mean(mean_scores)), 6)
        if mean_scores
        else 0.0,
        **_fused_vs_spaces(fused_orders, topologies, common_ids),
        **_fused_vs_consensus(fused_sets, common_ids, consensus_pairs),
    }
    return metrics


def concat_fusion(
    spaces: Sequence[SpaceData],
    topologies: Sequence[SpaceTopology],
    common_ids: Sequence[int],
    *,
    k: int,
    consensus_pairs: set[tuple[int, int]],
    weight_scheme: str,
) -> dict[str, Any]:
    """concat 向量融合：每空间 L2 后按块权重拼接，再整体 L2，评估 Top-k 邻域。"""

    if weight_scheme not in ("equal", "inv_sqrt_dim"):
        raise NebulaError(f"未知 concat 权重方案: {weight_scheme}")
    k_value = _validated_k(k)
    count = len(common_ids)
    effective = min(k_value, max(count - 1, 0))
    blocks: list[np.ndarray] = []
    block_dims: list[int] = []
    for space in spaces:
        block = normalize_vectors(space.matrix, where=f"{space.name} 融合输入")
        dim = int(block.shape[1])
        weight = 1.0 if weight_scheme == "equal" else 1.0 / math.sqrt(float(dim))
        blocks.append(block * weight)
        block_dims.append(dim)
    fused_matrix = normalize_vectors(np.concatenate(blocks, axis=1), where="concat 融合向量")
    similarity = cosine_similarity_matrix(fused_matrix)
    neighbors = topk_neighbors(similarity, k_value)
    fused_orders: list[tuple[int, ...]] = []
    neighbor_cosines: list[float] = []
    for row in range(count):
        order = tuple(int(common_ids[int(index)]) for index in neighbors[row])
        fused_orders.append(order)
        if neighbors.shape[1]:
            neighbor_cosines.append(
                float(np.mean([similarity[row, int(index)] for index in neighbors[row]]))
            )
    fused_sets = [set(order) for order in fused_orders]
    metrics = {
        "k": int(k_value),
        "effective_k": int(effective),
        "n_works": int(count),
        "weight_scheme": weight_scheme,
        "vector_dim": int(fused_matrix.shape[1]),
        "block_dims": [int(dim) for dim in block_dims],
        "mean_fused_neighbor_cosine": round(
            float(np.mean(neighbor_cosines)), 6
        )
        if neighbor_cosines
        else 0.0,
        **_fused_vs_spaces(fused_orders, topologies, common_ids),
        **_fused_vs_consensus(fused_sets, common_ids, consensus_pairs),
    }
    return metrics


# --------------------------------------------------------------------------- 输出
def _node_csv_path(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.stem + "_nodes.csv")


def _atomic_text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def _space_columns(report: Mapping[str, Any]) -> list[str]:
    return [str(space["name"]) for space in report.get("spaces", [])]


def write_reports(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> tuple[Path, Path, Path]:
    """原子写 JSON（nodes+edges+fusion）、边 CSV 与节点 CSV。"""

    json_target = Path(json_path)
    edge_target = Path(csv_path)
    node_target = _node_csv_path(edge_target)
    _atomic_text_write(
        json_target,
        json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=False) + NEWLINE,
    )
    space_names = _space_columns(report)

    edge_columns: list[str] = [
        "work_id_a",
        "work_id_b",
        "consensus_count",
        "consensus_strength",
        "is_consensus",
        "is_full_consensus",
        "mutual_spaces",
    ]
    for name in space_names:
        edge_columns.extend(
            [f"{name}_rank_a", f"{name}_rank_b", f"{name}_cos", f"{name}_mutual"]
        )
    edge_target.parent.mkdir(parents=True, exist_ok=True)
    edge_temp = edge_target.with_name(edge_target.name + ".tmp")
    with edge_temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=edge_columns)
        writer.writeheader()
        for edge in report.get("edges", []):
            row: dict[str, Any] = {
                "work_id_a": edge.get("work_id_a"),
                "work_id_b": edge.get("work_id_b"),
                "consensus_count": edge.get("consensus_count"),
                "consensus_strength": edge.get("consensus_strength"),
                "is_consensus": edge.get("is_consensus"),
                "is_full_consensus": edge.get("is_full_consensus"),
                "mutual_spaces": " ".join(edge.get("mutual_spaces", [])),
            }
            for name in space_names:
                space_values = edge.get("per_space", {}).get(name, {})
                row[f"{name}_rank_a"] = space_values.get("rank_a", "")
                row[f"{name}_rank_b"] = space_values.get("rank_b", "")
                row[f"{name}_cos"] = space_values.get("cos", "")
                row[f"{name}_mutual"] = space_values.get("mutual", "")
            writer.writerow(row)
    os.replace(edge_temp, edge_target)

    node_columns: list[str] = ["work_id", "title"]
    for name in space_names:
        node_columns.append(f"{name}_mutual_degree")
    node_columns.extend(
        ["consensus_degree", "full_consensus_degree", "consensus_strength_sum"]
    )
    node_target.parent.mkdir(parents=True, exist_ok=True)
    node_temp = node_target.with_name(node_target.name + ".tmp")
    with node_temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=node_columns)
        writer.writeheader()
        for node in report.get("nodes", []):
            row = {"work_id": node.get("work_id"), "title": node.get("title")}
            mutual_degree = node.get("mutual_degree", {})
            for name in space_names:
                row[f"{name}_mutual_degree"] = mutual_degree.get(name, "")
            row["consensus_degree"] = node.get("consensus_degree")
            row["full_consensus_degree"] = node.get("full_consensus_degree")
            row["consensus_strength_sum"] = node.get("consensus_strength_sum")
            writer.writerow(row)
    os.replace(node_temp, node_target)
    return json_target, edge_target, node_target


def _print_summary(
    report: Mapping[str, Any],
    *,
    json_path: Path,
    csv_path: Path,
    node_csv_path: Path,
) -> None:
    params = report["params"]
    graph = report["consensus"]
    print("跨模型近邻共识图（v0.4.4）")
    print(
        "  空间: " + ", ".join(params["representations"])
        + f"；共同作品 {graph['n_works_common']}；k={graph['k']}"
        + f"（effective {graph['effective_k']}）；min_spaces={graph['min_spaces']}"
    )
    for space in report["spaces"]:
        print(
            f"  {space['name']}: n={space['n_works_common']}"
            f"（总 {space['n_works_total']}）dim={space['dim']}"
            f" directed={space['directed_edges']} mutual={space['mutual_edges']}"
            f" mutual_ratio={space['mutual_ratio']:.4f}"
            f" mean_topk_cos={space['mean_topk_cosine']:.4f}"
        )
    for pair in graph["cross_space"]:
        print(
            f"  跨空间 {pair['a']} × {pair['b']}: "
            f"directed_jaccard={pair['directed_jaccard']:.4f}"
            f" retention_a_to_b={pair['directed_retention_a_to_b']:.4f}"
            f" mutual_jaccard={pair['mutual_jaccard']:.4f}"
        )
    print(
        f"  共识边: {graph['n_consensus_edges']}（全共识 {graph['n_full_consensus_edges']}）"
        f"；覆盖作品 {graph['consensus_coverage_works']}/{graph['n_works_common']}"
        f"（{graph['consensus_coverage_ratio']:.2%}）"
        f"；平均强度 {graph['mean_consensus_strength']:.4f}"
        f"；平均共识度 {graph['mean_consensus_degree']:.4f}"
    )
    fusion = report.get("fusion")
    if fusion:
        rank = fusion.get("rank_average")
        if rank:
            print(
                "  fusion rank-average: "
                f"Jaccard(vs spaces)={rank['mean_jaccard_vs_spaces']:.4f}"
                f" Jaccard(vs consensus)={rank['jaccard_mean_works_with_consensus']:.4f}"
                f" edge_jaccard={rank['edge_jaccard']:.4f}"
            )
        concat = fusion.get("concat") or {}
        for scheme in ("equal", "inv_sqrt_dim"):
            values = concat.get(scheme)
            if not values:
                continue
            print(
                f"  fusion concat({scheme}): dim={values['vector_dim']}"
                f" Jaccard(vs spaces)={values['mean_jaccard_vs_spaces']:.4f}"
                f" Jaccard(vs consensus)={values['jaccard_mean_works_with_consensus']:.4f}"
                f" mean_neighbor_cos={values['mean_fused_neighbor_cosine']:.4f}"
            )
    print(f"  JSON: {json_path}")
    print(f"  CSV(edges):  {csv_path}")
    print(f"  CSV(nodes):  {node_csv_path}")


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/build_consensus_graph.py",
        description=(
            "v0.4.4 跨模型近邻共识图：每空间高维余弦 Top-k 有向邻接与 mutual 对、"
            "跨空间保留率、共识边与轻量融合基线（不写 DB）。"
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="配置文件路径（默认取 NEBULA_CONFIG 环境变量，否则当前目录 config.yaml）",
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        action="extend",
        default=None,
        help=(
            "参与共识图的空间（可多值/可重复；默认 "
            + ", ".join(DEFAULT_REPRESENTATIONS)
            + "）"
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"每空间高维 Top-k（默认 {DEFAULT_K}）",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="JSON 输出路径（默认 <logs>/neighbor_consensus.json）",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="边 CSV 输出路径（默认与 JSON 同目录同名 .csv；节点 CSV 自动加 _nodes 后缀）",
    )
    parser.add_argument(
        "--min-spaces",
        type=int,
        default=DEFAULT_MIN_SPACES,
        help=f"共识边所需的最少 mutual 空间数（默认 {DEFAULT_MIN_SPACES}）",
    )
    parser.add_argument(
        "--fusion",
        choices=FUSION_MODES,
        default=DEFAULT_FUSION,
        help=f"轻量融合基线（默认 {DEFAULT_FUSION}：rank-average + concat 两种权重）",
    )
    return parser


def _resolve_config_path(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    env_value = os.environ.get("NEBULA_CONFIG")
    if env_value:
        return Path(env_value).expanduser()
    return Path.cwd() / "config.yaml"


def run_consensus(
    catalog: Catalog,
    config: Config,
    *,
    representations: Sequence[str] | None = None,
    k: int = DEFAULT_K,
    min_spaces: int = DEFAULT_MIN_SPACES,
    fusion: str = DEFAULT_FUSION,
    kind: str = DEFAULT_KIND,
) -> dict[str, Any]:
    """从已登记表征构建共识图报告（只读 DB）。"""

    names = _validated_names(
        DEFAULT_REPRESENTATIONS if representations is None else representations
    )
    spaces = [load_space(catalog, config, name, kind=kind) for name in names]
    aligned = align_spaces(spaces)
    work_ids = aligned[0].work_ids
    works_meta: dict[int, dict[str, Any]] = {}
    for work_id in work_ids:
        work = catalog.get_work(int(work_id)) or {}
        works_meta[int(work_id)] = {
            "work_id": int(work_id),
            "title": work.get("title"),
            "source": work.get("source"),
        }
    return build_consensus(
        spaces,
        k=k,
        min_spaces=min_spaces,
        fusion=fusion,
        kind=kind,
        works_meta=works_meta,
        generated_at=now_utc_iso(),
    )


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
        report = run_consensus(
            catalog,
            config,
            representations=args.representations,
            k=args.k,
            min_spaces=args.min_spaces,
            fusion=args.fusion,
        )
    except NebulaError as exc:
        print(f"共识图构建失败: {exc}", file=sys.stderr)
        return 1
    finally:
        catalog.close()

    json_path = (
        Path(args.out).expanduser()
        if args.out
        else config.paths.logs / "neighbor_consensus.json"
    )
    csv_path = (
        Path(args.csv).expanduser() if args.csv else json_path.with_suffix(".csv")
    )
    try:
        json_target, edge_target, node_target = write_reports(
            report, json_path=json_path, csv_path=csv_path
        )
    except NebulaError as exc:
        print(f"写入共识图报告失败: {exc}", file=sys.stderr)
        return 1

    _print_summary(
        report, json_path=json_target, csv_path=edge_target, node_csv_path=node_target
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
