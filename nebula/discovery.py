r"""v0.4.5 高维意外关系发现（观察性证据，不做分类/主模型）。

按用户规格实现：

- 空间：full_mean_v1 的 joint/audio/video、temporal_stats_v1 joint、
  dense_visual_clip_v2 joint、dense_visual_dino_v2 joint；canonical 共同作品按 work_id 对齐；
- 排名百分位：每空间对每行按余弦降序排名（不含自身，第 1 名 r=1，并列按 work_id 升序），
  p = 1 - (r - 1) / (N - 2)（N >= 3；第 1 名 p=1，第 N-1 名 p=0）；
  无向对 bar_p = (p(i,j) + p(j,i)) / 2；跨模型比较只用 p，原始 cos 仅展示；
- 模式 A：full_mean joint 的 Top-k 有向边取无序并集为候选对，
  U = bar_p_joint - (bar_p_audio + bar_p_video) / 2，按 U 降序；
- 模式 B：temporal joint Top-k 候选对，U = bar_p_temporal - bar_p_full_mean，按 U 降序；
- 模式 C：PE-AV / CLIP / DINOv2 三空间 mutual Top-k 对比：
  common = mutual 空间数 >= 2（common_level 标注 all3 / peav+clip / peav+dino / visual_only），
  peav_only / clip_only / dino_only = 恰好 1 个空间 mutual，
  visual_vs_peav = CLIP 与 DINOv2 都 mutual 但 PE-AV 不 mutual；
  分数全部用 p 差/组合，禁止原始余弦直接相减。

输出 logs/unexpected_relations.json + 单文件长表 CSV（各模式/类别 top_n 行）。

本模块只读已登记表征（nebula.representations.load_representation_matrix，canonical only）
与作品标题（仅展示），不写数据库、不建地图、不加载模型、不访问网络。
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import Config
from .db import Catalog
from .errors import NebulaError
from .highdim import (
    cosine_similarity_matrix,
    normalize_vectors,
    topk_similarity_neighbors,
)
from .representations import (
    load_representation_matrix,
    resolve_representation,
)
from .util import now_utc_iso

TOOL_NAME = "unexpected_relations"
DISCOVERY_VERSION = "v0.4.5"

DEFAULT_TOP_N = 50
DEFAULT_K = 10
MIN_COMMON_WORKS = 3
MODE_CHOICES: tuple[str, ...] = ("all", "a", "b", "c")

PEAV_NAME = "full_mean_v1"
TEMPORAL_NAME = "temporal_stats_v1"
CLIP_NAME = "dense_visual_clip_v2"
DINO_NAME = "dense_visual_dino_v2"

SPACE_KEYS: tuple[str, ...] = (
    "full_mean_joint",
    "full_mean_audio",
    "full_mean_video",
    "temporal_joint",
    "clip_joint",
    "dino_joint",
)
SPACE_LABELS: dict[str, str] = {
    "full_mean_joint": "PE-AV Full Mean (joint)",
    "full_mean_audio": "PE-AV Full Mean (audio)",
    "full_mean_video": "PE-AV Full Mean (video)",
    "temporal_joint": "Temporal Stats (joint)",
    "clip_joint": "Dense Visual·CLIP v2 (joint)",
    "dino_joint": "Dense Visual·DINOv2 v2 (joint)",
}

#: 需要加载的 (space_key, representation_name, kind)
SPACE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("full_mean_joint", PEAV_NAME, "joint"),
    ("full_mean_audio", PEAV_NAME, "audio"),
    ("full_mean_video", PEAV_NAME, "video"),
    ("temporal_joint", TEMPORAL_NAME, "joint"),
    ("clip_joint", CLIP_NAME, "joint"),
    ("dino_joint", DINO_NAME, "joint"),
)

CSV_SPACE_GROUPS: tuple[str, ...] = (
    "joint",
    "audio",
    "video",
    "temporal",
    "full_mean",
    "peav",
    "clip",
    "dino",
)
CSV_BASE_COLUMNS: tuple[str, ...] = (
    "mode",
    "category",
    "rank",
    "work_id_a",
    "work_id_b",
    "title_a",
    "title_b",
    "score",
    "mutual_count",
    "mutual_spaces",
)
CSV_SPACE_COLUMNS: tuple[str, ...] = tuple(
    f"{group}_{field}"
    for group in CSV_SPACE_GROUPS
    for field in ("rank_a", "rank_b", "p_a", "p_b", "bar_p", "cos")
)
CSV_COLUMNS: tuple[str, ...] = CSV_BASE_COLUMNS + CSV_SPACE_COLUMNS


class DiscoveryError(NebulaError):
    """意外关系发现错误（共同作品不足、零向量、空间缺失等）。"""


# --------------------------------------------------------------------------- 数据空间
@dataclass(frozen=True)
class SpaceVectors:
    """一个空间的作品级向量（行按 work_id 升序，逐行 L2 归一化）。"""

    key: str
    name: str
    label: str
    kind: str
    ids: tuple[int, ...]
    matrix: np.ndarray
    representation_id: int = 0
    sequence_version: str = ""
    source_kind: str = "pe_av"

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
        key: str,
        name: str,
        label: str | None = None,
        kind: str = "joint",
        ids: Sequence[int] | None = None,
        representation_id: int = 0,
        sequence_version: str = "",
        source_kind: str = "pe_av",
    ) -> "SpaceVectors":
        """构造并按 work_id 升序对齐行（行顺序不影响结果）。"""

        normalized = normalize_vectors(matrix, where=f"{name} 向量矩阵")
        count = int(normalized.shape[0])
        raw_ids = (
            tuple(int(value) for value in ids)
            if ids is not None
            else tuple(range(1, count + 1))
        )
        if len(raw_ids) != count:
            raise DiscoveryError(
                f"{name} 的 work_ids 数量 {len(raw_ids)} 与矩阵行数 {count} 不一致"
            )
        if len(set(raw_ids)) != len(raw_ids):
            raise DiscoveryError(f"{name} 的 work_ids 有重复")
        order = sorted(range(count), key=lambda index: raw_ids[index])
        sorted_ids = tuple(raw_ids[index] for index in order)
        sorted_matrix = np.ascontiguousarray(normalized[order])
        return cls(
            key=str(key),
            name=str(name),
            label=str(label or SPACE_LABELS.get(key, name)),
            kind=str(kind),
            ids=sorted_ids,
            matrix=sorted_matrix,
            representation_id=int(representation_id),
            sequence_version=str(sequence_version),
            source_kind=str(source_kind),
        )


def _subset(space: SpaceVectors, common_ids: Sequence[int]) -> SpaceVectors:
    index = {work_id: position for position, work_id in enumerate(space.ids)}
    rows = np.asarray([index[work_id] for work_id in common_ids], dtype=np.int64)
    return SpaceVectors(
        key=space.key,
        name=space.name,
        label=space.label,
        kind=space.kind,
        ids=tuple(int(work_id) for work_id in common_ids),
        matrix=np.ascontiguousarray(space.matrix[rows]),
        representation_id=space.representation_id,
        sequence_version=space.sequence_version,
        source_kind=space.source_kind,
    )


def align_spaces(
    spaces: Mapping[str, SpaceVectors | None],
    required: Sequence[str],
) -> tuple[dict[str, SpaceVectors], tuple[int, ...], str | None]:
    """取 required 空间的共同 canonical 作品并对齐；不可用时返回跳过原因。"""

    missing = [key for key in required if spaces.get(key) is None]
    if missing:
        return {}, tuple(), "缺少空间: " + ", ".join(missing)
    common = set(spaces[required[0]].ids)  # type: ignore[union-attr]
    for key in required[1:]:
        common &= set(spaces[key].ids)  # type: ignore[union-attr]
    common_ids = tuple(sorted(common))
    if len(common_ids) < MIN_COMMON_WORKS:
        return {}, common_ids, (
            f"共同作品不足 {MIN_COMMON_WORKS} 部（{len(common_ids)} 部）"
        )
    aligned = {key: _subset(spaces[key], common_ids) for key in required}  # type: ignore[arg-type]
    return aligned, common_ids, None


# --------------------------------------------------------------------------- 排名 / 百分位
def neighbor_rank_table(
    similarity: Any,
    *,
    ids: Sequence[int] | None = None,
) -> np.ndarray:
    """每行邻居排名表 ranks[i, j]（1-start，不含自身；并列按 id 升序）。"""

    values = np.asarray(similarity, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise DiscoveryError(f"相似度矩阵必须是方阵，收到 {values.shape}")
    if not np.all(np.isfinite(values)):
        raise DiscoveryError("相似度矩阵包含 NaN/Inf")
    count = int(values.shape[0])
    if ids is not None:
        id_list = [int(value) for value in ids]
        if len(id_list) != count or sorted(id_list) != id_list:
            raise DiscoveryError("ids 必须与矩阵行数一致且按 work_id 升序")
    work = values.copy()
    np.fill_diagonal(work, -np.inf)
    order = np.argsort(-work, axis=1, kind="stable")
    ranks = np.zeros((count, count), dtype=np.int64)
    positions = np.arange(1, count, dtype=np.int64)
    for row in range(count):
        ranks[row, order[row, : count - 1]] = positions
    return ranks


def percentile_table(ranks: Any) -> np.ndarray:
    """r -> p = 1 - (r-1)/(N-2)，对角为 0；要求 N >= 3。"""

    values = np.asarray(ranks, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise DiscoveryError(f"排名矩阵必须是方阵，收到 {values.shape}")
    count = int(values.shape[0])
    if count < MIN_COMMON_WORKS:
        raise DiscoveryError(
            f"排名百分位要求共同作品 N >= {MIN_COMMON_WORKS}，收到 {count}"
        )
    result = np.zeros((count, count), dtype=np.float64)
    mask = ~np.eye(count, dtype=bool)
    result[mask] = 1.0 - (values[mask] - 1.0) / float(count - 2)
    return result


def bar_percentile(p: np.ndarray, first: int, second: int) -> float:
    """无向对的平均百分位。"""

    return float((p[first, second] + p[second, first]) / 2.0)


def _pair_metrics(
    similarity: np.ndarray,
    ranks: np.ndarray,
    percentiles: np.ndarray,
    first: int,
    second: int,
) -> dict[str, Any]:
    return {
        "rank_a": int(ranks[first, second]),
        "rank_b": int(ranks[second, first]),
        "p_a": round(float(percentiles[first, second]), 6),
        "p_b": round(float(percentiles[second, first]), 6),
        "bar_p": round(bar_percentile(percentiles, first, second), 6),
        "cos": round(float(similarity[first, second]), 6),
    }


def _space_tables(space: SpaceVectors) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    similarity = cosine_similarity_matrix(
        space.matrix, where=f"{space.key} 余弦相似度"
    )
    ranks = neighbor_rank_table(similarity, ids=space.ids)
    percentiles = percentile_table(ranks)
    return similarity, ranks, percentiles


def _directed_pairs(
    similarity: np.ndarray, ids: Sequence[int], k: int
) -> set[tuple[int, int]]:
    neighbors = topk_similarity_neighbors(similarity, k)
    pairs: set[tuple[int, int]] = set()
    for row, row_neighbors in enumerate(neighbors):
        for column in row_neighbors:
            first, second = int(ids[row]), int(ids[int(column)])
            if first == second:
                continue
            pairs.add((min(first, second), max(first, second)))
    return pairs


def _mutual_pairs(ranks: np.ndarray, ids: Sequence[int], k: int) -> set[tuple[int, int]]:
    effective = min(int(k), max(len(ids) - 1, 1))
    pairs: set[tuple[int, int]] = set()
    for first in range(len(ids)):
        for second in range(first + 1, len(ids)):
            if ranks[first, second] <= effective and ranks[second, first] <= effective:
                pairs.add((int(ids[first]), int(ids[second])))
    return pairs


# --------------------------------------------------------------------------- 模式
def _base_row(
    work_a: int,
    work_b: int,
    titles: Mapping[int, str | None] | None,
) -> dict[str, Any]:
    title_map = titles or {}
    return {
        "work_id_a": int(work_a),
        "work_id_b": int(work_b),
        "title_a": title_map.get(int(work_a)),
        "title_b": title_map.get(int(work_b)),
    }


def _sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["score"]),
            int(row["work_id_a"]),
            int(row["work_id_b"]),
        ),
    )


def _mode_a(
    spaces: Mapping[str, SpaceVectors | None],
    *,
    k: int,
    top_n: int,
    titles: Mapping[int, str | None] | None,
) -> dict[str, Any]:
    required = ["full_mean_joint", "full_mean_audio", "full_mean_video"]
    aligned, common_ids, reason = align_spaces(spaces, required)
    if reason:
        return {"skipped": True, "reason": reason, "top": []}
    tables = {key: _space_tables(space) for key, space in aligned.items()}
    index = {work_id: position for position, work_id in enumerate(common_ids)}
    candidates = _directed_pairs(tables["full_mean_joint"][0], common_ids, k)
    rows: list[dict[str, Any]] = []
    for work_a, work_b in sorted(candidates):
        first, second = index[work_a], index[work_b]
        joint_metrics = _pair_metrics(*tables["full_mean_joint"], first, second)
        audio_metrics = _pair_metrics(*tables["full_mean_audio"], first, second)
        video_metrics = _pair_metrics(*tables["full_mean_video"], first, second)
        score = joint_metrics["bar_p"] - (
            audio_metrics["bar_p"] + video_metrics["bar_p"]
        ) / 2.0
        rows.append(
            {
                **_base_row(work_a, work_b, titles),
                "score": round(float(score), 6),
                "joint": joint_metrics,
                "audio": audio_metrics,
                "video": video_metrics,
                "joint_candidate_a_to_b": bool(joint_metrics["rank_a"] <= k),
                "joint_candidate_b_to_a": bool(joint_metrics["rank_b"] <= k),
            }
        )
    ordered = _sorted_rows(rows)
    return {
        "skipped": False,
        "n_works": len(common_ids),
        "n_candidates": len(rows),
        "top_n": int(top_n),
        "k": int(k),
        "top": ordered[:top_n],
    }


def _mode_b(
    spaces: Mapping[str, SpaceVectors | None],
    *,
    k: int,
    top_n: int,
    titles: Mapping[int, str | None] | None,
) -> dict[str, Any]:
    required = ["temporal_joint", "full_mean_joint"]
    aligned, common_ids, reason = align_spaces(spaces, required)
    if reason:
        return {"skipped": True, "reason": reason, "top": []}
    temporal = aligned["temporal_joint"]
    full_mean = aligned["full_mean_joint"]
    temporal_tables = _space_tables(temporal)
    full_tables = _space_tables(full_mean)
    index = {work_id: position for position, work_id in enumerate(common_ids)}
    candidates = _directed_pairs(temporal_tables[0], common_ids, k)
    rows: list[dict[str, Any]] = []
    for work_a, work_b in sorted(candidates):
        first, second = index[work_a], index[work_b]
        temporal_metrics = _pair_metrics(*temporal_tables, first, second)
        full_metrics = _pair_metrics(*full_tables, first, second)
        score = temporal_metrics["bar_p"] - full_metrics["bar_p"]
        rows.append(
            {
                **_base_row(work_a, work_b, titles),
                "score": round(float(score), 6),
                "temporal": temporal_metrics,
                "full_mean": full_metrics,
                "temporal_candidate_a_to_b": bool(temporal_metrics["rank_a"] <= k),
                "temporal_candidate_b_to_a": bool(temporal_metrics["rank_b"] <= k),
            }
        )
    ordered = _sorted_rows(rows)
    return {
        "skipped": False,
        "n_works": len(common_ids),
        "n_candidates": len(rows),
        "top_n": int(top_n),
        "k": int(k),
        "top": ordered[:top_n],
    }


def _common_level(flags: Mapping[str, bool]) -> str | None:
    """多模型共同认可只统计包含 PE-AV 的组合（Lead 确认口径）。

    CLIP 与 DINOv2 都 mutual 但 PE-AV 不 mutual 不属于 common，单独归入
    visual_vs_peav。
    """

    if not flags.get("peav"):
        return None
    mutual_count = sum(1 for value in flags.values() if value)
    if mutual_count >= 3:
        return "all3"
    if flags.get("clip"):
        return "peav_clip"
    if flags.get("dino"):
        return "peav_dino"
    return None


def _mode_c(
    spaces: Mapping[str, SpaceVectors | None],
    *,
    k: int,
    top_n: int,
    titles: Mapping[int, str | None] | None,
) -> dict[str, Any]:
    required = ["full_mean_joint", "clip_joint", "dino_joint"]
    aligned, common_ids, reason = align_spaces(spaces, required)
    if reason:
        return {"skipped": True, "reason": reason, "top": []}
    space_names = {"peav": "full_mean_joint", "clip": "clip_joint", "dino": "dino_joint"}
    tables = {label: _space_tables(aligned[key]) for label, key in space_names.items()}
    index = {work_id: position for position, work_id in enumerate(common_ids)}
    mutual_sets = {
        label: _mutual_pairs(tables[label][1], common_ids, k)
        for label in ("peav", "clip", "dino")
    }
    candidates: set[tuple[int, int]] = set()
    for pairs in mutual_sets.values():
        candidates |= pairs

    rows: list[dict[str, Any]] = []
    for work_a, work_b in sorted(candidates):
        first, second = index[work_a], index[work_b]
        flags = {
            label: (work_a, work_b) in mutual_sets[label]
            for label in ("peav", "clip", "dino")
        }
        mutual_count = int(sum(1 for value in flags.values() if value))
        metrics = {
            label: {
                **_pair_metrics(*tables[label], first, second),
                "mutual": bool(flags[label]),
            }
            for label in ("peav", "clip", "dino")
        }
        rows.append(
            {
                **_base_row(work_a, work_b, titles),
                "mutual_count": mutual_count,
                "mutual_spaces": [
                    label for label in ("peav", "clip", "dino") if flags[label]
                ],
                "common_level": _common_level(flags),
                "peav": metrics["peav"],
                "clip": metrics["clip"],
                "dino": metrics["dino"],
            }
        )

    common_rows: list[dict[str, Any]] = []
    peav_only: list[dict[str, Any]] = []
    clip_only: list[dict[str, Any]] = []
    dino_only: list[dict[str, Any]] = []
    visual_vs_peav: list[dict[str, Any]] = []
    for row in rows:
        flags = {
            "peav": bool(row["peav"]["mutual"]),
            "clip": bool(row["clip"]["mutual"]),
            "dino": bool(row["dino"]["mutual"]),
        }
        mutual_count = int(row["mutual_count"])
        bar = {
            label: float(row[label]["bar_p"])
            for label in ("peav", "clip", "dino")
        }
        if mutual_count >= 2 and flags["peav"]:
            mutual_labels = [label for label in ("peav", "clip", "dino") if flags[label]]
            score = float(np.mean([bar[label] for label in mutual_labels]))
            common_rows.append({**row, "score": round(score, 6)})
        if mutual_count == 1:
            only = next(label for label in ("peav", "clip", "dino") if flags[label])
            others = [label for label in ("peav", "clip", "dino") if label != only]
            score = bar[only] - float(np.mean([bar[label] for label in others]))
            target = {
                "peav": peav_only,
                "clip": clip_only,
                "dino": dino_only,
            }[only]
            target.append({**row, "score": round(score, 6)})
        if flags["clip"] and flags["dino"] and not flags["peav"]:
            score = (bar["clip"] + bar["dino"]) / 2.0 - bar["peav"]
            visual_vs_peav.append({**row, "score": round(score, 6)})

    common_ordered = _sorted_rows(common_rows)
    visual_ordered = _sorted_rows(visual_vs_peav)
    peav_ordered = _sorted_rows(peav_only)
    clip_ordered = _sorted_rows(clip_only)
    dino_ordered = _sorted_rows(dino_only)
    levels = {"all3": 0, "peav_clip": 0, "peav_dino": 0}
    for row in common_ordered:
        level = row["common_level"]
        if level in levels:
            levels[level] += 1
    return {
        "skipped": False,
        "n_works": len(common_ids),
        "n_candidate_pairs": len(rows),
        "top_n": int(top_n),
        "k": int(k),
        "common_summary": {
            "n_common_ge2": len(common_rows),
            "n_common_all3": levels["all3"],
            "levels": levels,
        },
        "common": common_ordered[:top_n],
        "visual_vs_peav": visual_ordered[:top_n],
        "peav_only": peav_ordered[:top_n],
        "clip_only": clip_ordered[:top_n],
        "dino_only": dino_ordered[:top_n],
        "counts": {
            "common_ge2": len(common_rows),
            "visual_vs_peav": len(visual_vs_peav),
            "peav_only": len(peav_only),
            "clip_only": len(clip_only),
            "dino_only": len(dino_only),
        },
    }


# --------------------------------------------------------------------------- 主函数
def _normalize_mode(mode: str) -> str:
    text = str(mode or "all").strip().lower()
    aliases = {"mode_a": "a", "mode_b": "b", "mode_c": "c"}
    text = aliases.get(text, text)
    if text not in MODE_CHOICES:
        raise DiscoveryError(f"mode 必须是 {MODE_CHOICES} 之一，收到 {mode!r}")
    return text


def _requested_modes(mode: str) -> tuple[str, ...]:
    normalized = _normalize_mode(mode)
    return ("a", "b", "c") if normalized == "all" else (normalized,)


def discover_from_spaces(
    spaces: Mapping[str, SpaceVectors | None],
    *,
    top_n: int = DEFAULT_TOP_N,
    k: int = DEFAULT_K,
    mode: str = "all",
    titles: Mapping[int, str | None] | None = None,
    generated_at: str | None = None,
    skipped: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """纯函数：给定已对齐的 SpaceVectors 计算三种模式（确定性输出）。"""

    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise DiscoveryError(f"top_n 必须是 >= 1 的整数，收到 {top_n!r}")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise DiscoveryError(f"k 必须是 >= 1 的整数，收到 {k!r}")
    requested = _requested_modes(mode)
    loaded = [key for key in SPACE_KEYS if spaces.get(key) is not None]
    if not loaded:
        raise DiscoveryError("没有可用的表征空间（所有空间加载失败）")
    mode_results: dict[str, dict[str, Any]] = {}
    if "a" in requested:
        mode_results["a"] = _mode_a(spaces, k=k, top_n=top_n, titles=titles)
    else:
        mode_results["a"] = None  # type: ignore[assignment]
    if "b" in requested:
        mode_results["b"] = _mode_b(spaces, k=k, top_n=top_n, titles=titles)
    else:
        mode_results["b"] = None  # type: ignore[assignment]
    if "c" in requested:
        mode_results["c"] = _mode_c(spaces, k=k, top_n=top_n, titles=titles)
    else:
        mode_results["c"] = None  # type: ignore[assignment]
    active = [name for name in ("a", "b", "c") if mode_results[name] is not None]
    runnable = [
        name
        for name in active
        if not bool(mode_results[name].get("skipped"))  # type: ignore[union-attr]
    ]
    if not runnable:
        reasons = {
            name: mode_results[name].get("reason")  # type: ignore[union-attr]
            for name in active
        }
        raise DiscoveryError(f"所有请求的模式都不可用: {reasons}")

    report = {
        "meta": {
            "tool": TOOL_NAME,
            "version": DISCOVERY_VERSION,
            "generated_at": generated_at or now_utc_iso(),
            "params": {
                "top_n": int(top_n),
                "k": int(k),
                "mode": _normalize_mode(mode),
            },
            "definitions": {
                "rank_percentile": "p = 1 - (r - 1) / (N - 2); r 为不含自身的余弦近邻排名（第 1=1），并列按 work_id 升序",
                "bar_percentile": "无向对 bar_p = (p(i,j) + p(j,i)) / 2",
                "cross_model": "跨模型比较只用 p；cos 仅作展示",
                "mode_a": "joint Top-k 有向边无序并集候选；U = bar_p_joint - (bar_p_audio + bar_p_video) / 2",
                "mode_b": "temporal Top-k 候选；U = bar_p_temporal - bar_p_full_mean",
                "mode_c": "PE-AV/CLIP/DINO mutual Top-k 对比；common 仅含包含 PE-AV 的 >=2 空间 mutual；visual_vs_peav = CLIP+DINO mutual 但 PE-AV 不 mutual；分数全部用 p 差/组合",
            },
            "spaces_loaded": list(loaded),
            "skipped": [dict(item) for item in skipped],
        },
        "spaces": [
            {
                "key": space.key,
                "name": space.name,
                "label": space.label,
                "kind": space.kind,
                "representation_id": int(space.representation_id),
                "sequence_version": space.sequence_version,
                "source_kind": space.source_kind,
                "n_works": int(space.n_works),
                "dim": int(space.dim),
            }
            for space in (spaces[key] for key in loaded)
        ],
        "mode_a": mode_results["a"],
        "mode_b": mode_results["b"],
        "mode_c": mode_results["c"],
    }
    return report


def discover(
    catalog: Catalog,
    config: Config,
    *,
    top_n: int = DEFAULT_TOP_N,
    k: int = DEFAULT_K,
    mode: str = "all",
) -> dict[str, Any]:
    """只读已登记表征，构建 v0.4.5 意外关系报告（不写 DB/地图/模型）。"""

    _requested_modes(mode)
    spaces: dict[str, SpaceVectors | None] = {key: None for key in SPACE_KEYS}
    skipped: list[dict[str, Any]] = []
    rep_cache: dict[str, dict[str, Any]] = {}
    for space_key, rep_name, kind in SPACE_SPECS:
        try:
            rep = rep_cache.get(rep_name)
            if rep is None:
                rep = resolve_representation(catalog, config, name=rep_name)
                rep_cache[rep_name] = rep
            if str(rep.get("status")) != "ready":
                raise DiscoveryError(
                    f"表征 {rep_name} 状态为 {rep.get('status')}，需要 ready"
                )
            ids, matrix = load_representation_matrix(
                catalog, config, rep, kind=kind
            )
            spaces[space_key] = SpaceVectors.from_matrix(
                matrix,
                key=space_key,
                name=str(rep["name"]),
                label=SPACE_LABELS.get(space_key, str(rep["name"])),
                kind=kind,
                ids=ids,
                representation_id=int(rep["id"]),
                sequence_version=str(rep["sequence_version"]),
                source_kind=str(rep.get("source_kind", "pe_av")),
            )
        except NebulaError as exc:
            spaces[space_key] = None
            skipped.append(
                {
                    "space": space_key,
                    "representation": rep_name,
                    "kind": kind,
                    "reason": str(exc),
                }
            )
    titles: dict[int, str | None] = {}
    work_ids: set[int] = set()
    for space in spaces.values():
        if space is not None:
            work_ids.update(int(work_id) for work_id in space.ids)
    for work_id in sorted(work_ids):
        work = catalog.get_work(work_id) or {}
        titles[work_id] = (
            work.get("title")
            or work.get("local_path")
            or work.get("work_key")
        )
    return discover_from_spaces(
        spaces,
        top_n=top_n,
        k=k,
        mode=mode,
        titles=titles,
        generated_at=now_utc_iso(),
        skipped=skipped,
    )


# --------------------------------------------------------------------------- 输出
def _atomic_text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def _csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add_row(
        mode: str,
        category: str,
        rank: int,
        item: Mapping[str, Any],
        groups: Sequence[str],
    ) -> None:
        row: dict[str, Any] = {
            "mode": mode,
            "category": category,
            "rank": rank,
            "work_id_a": item.get("work_id_a"),
            "work_id_b": item.get("work_id_b"),
            "title_a": item.get("title_a"),
            "title_b": item.get("title_b"),
            "score": item.get("score"),
            "mutual_count": item.get("mutual_count"),
            "mutual_spaces": " ".join(item.get("mutual_spaces", [])),
        }
        for group in groups:
            values = item.get(group) or {}
            row[f"{group}_rank_a"] = values.get("rank_a", "")
            row[f"{group}_rank_b"] = values.get("rank_b", "")
            row[f"{group}_p_a"] = values.get("p_a", "")
            row[f"{group}_p_b"] = values.get("p_b", "")
            row[f"{group}_bar_p"] = values.get("bar_p", "")
            row[f"{group}_cos"] = values.get("cos", "")
        rows.append(row)

    mode_a = report.get("mode_a")
    if isinstance(mode_a, Mapping) and not mode_a.get("skipped"):
        for rank, item in enumerate(mode_a.get("top", ()), start=1):
            add_row("mode_a", "top", rank, item, ("joint", "audio", "video"))
    mode_b = report.get("mode_b")
    if isinstance(mode_b, Mapping) and not mode_b.get("skipped"):
        for rank, item in enumerate(mode_b.get("top", ()), start=1):
            add_row("mode_b", "top", rank, item, ("temporal", "full_mean"))
    mode_c = report.get("mode_c")
    if isinstance(mode_c, Mapping) and not mode_c.get("skipped"):
        categories = (
            ("common", "common"),
            ("visual_vs_peav", "visual_vs_peav"),
            ("peav_only", "peav_only"),
            ("clip_only", "clip_only"),
            ("dino_only", "dino_only"),
        )
        for key, label in categories:
            for rank, item in enumerate(mode_c.get(key, ()), start=1):
                add_row(f"mode_c/{label}", label, rank, item, ("peav", "clip", "dino"))
    return rows


def write_reports(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> tuple[Path, Path]:
    """原子写 JSON 与长表 CSV；返回 (json_path, csv_path)。"""

    json_target = Path(json_path)
    csv_target = Path(csv_path)
    _atomic_text_write(
        json_target,
        json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=False)
        + chr(10),
    )
    rows = _csv_rows(report)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    temp = csv_target.with_name(csv_target.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
    os.replace(temp, csv_target)
    return json_target, csv_target


def _format_pair(item: Mapping[str, Any], groups: Sequence[str]) -> str:
    pieces = [
        f"{item.get('work_id_a')}「{item.get('title_a') or '-'}」×"
        f"{item.get('work_id_b')}「{item.get('title_b') or '-'}」",
        f"score={float(item.get('score', 0.0)):.4f}",
    ]
    for group in groups:
        values = item.get(group) or {}
        pieces.append(
            f"{group}: rank={values.get('rank_a')}/{values.get('rank_b')}"
            f" p={values.get('bar_p')} cos={values.get('cos')}"
        )
    return " | ".join(pieces)


def print_summary(
    report: Mapping[str, Any],
    *,
    json_path: Path,
    csv_path: Path,
    max_rows: int = 5,
) -> None:
    meta = report["meta"]
    params = meta["params"]
    print("高维意外关系发现（v0.4.5）")
    print(f"  参数: top_n={params['top_n']} k={params['k']} mode={params['mode']}")
    print(
        "  已加载空间: "
        + ", ".join(str(space["key"]) for space in report.get("spaces", []))
    )
    blocked = meta.get("skipped") or []
    if blocked:
        print(f"  跳过空间（{len(blocked)}）:")
        for item in blocked[:5]:
            print(f"    {item.get('space')}: {item.get('reason')}")
    mode_a = report.get("mode_a")
    if isinstance(mode_a, Mapping):
        if mode_a.get("skipped"):
            print(f"  Mode A 跳过: {mode_a.get('reason')}")
        else:
            print(
                f"  Mode A（joint 候选 {mode_a['n_candidates']} 对，"
                f"U=bar_p_joint-(bar_p_audio+bar_p_video)/2）:"
            )
            for rank, item in enumerate(mode_a.get("top", [])[:max_rows], start=1):
                print(f"    {rank}. {_format_pair(item, ('joint', 'audio', 'video'))}")
    mode_b = report.get("mode_b")
    if isinstance(mode_b, Mapping):
        if mode_b.get("skipped"):
            print(f"  Mode B 跳过: {mode_b.get('reason')}")
        else:
            print(
                f"  Mode B（temporal 候选 {mode_b['n_candidates']} 对，"
                "U=bar_p_temporal-bar_p_full_mean）:"
            )
            for rank, item in enumerate(mode_b.get("top", [])[:max_rows], start=1):
                print(f"    {rank}. {_format_pair(item, ('temporal', 'full_mean'))}")
    mode_c = report.get("mode_c")
    if isinstance(mode_c, Mapping):
        if mode_c.get("skipped"):
            print(f"  Mode C 跳过: {mode_c.get('reason')}")
        else:
            counts = mode_c.get("counts", {})
            summary = mode_c.get("common_summary", {})
            print(
                "  Mode C（三空间 mutual 对比）: "
                f"common>=2={counts.get('common_ge2')}"
                f"（all3={summary.get('n_common_all3')}，"
                f"levels={summary.get('levels')}），"
                f"visual_vs_peav={counts.get('visual_vs_peav')}，"
                f"peav_only={counts.get('peav_only')}，"
                f"clip_only={counts.get('clip_only')}，"
                f"dino_only={counts.get('dino_only')}"
            )
            for category in ("common", "visual_vs_peav", "peav_only"):
                rows = mode_c.get(category) or []
                if not rows:
                    continue
                print(f"    [{category}]")
                for rank, item in enumerate(rows[:max_rows], start=1):
                    print(
                        f"      {rank}. {_format_pair(item, ('peav', 'clip', 'dino'))}"
                    )
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
