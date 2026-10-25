"""v0.4 邻域迁移对比 CLI：比较两个表征空间的 Top-k 邻域结构。

用法（项目根目录）::

    .\\.venv\\Scripts\\python.exe tools/compare_spaces.py

    # 显式指定表征名/序列版本/表征 id
    .\\.venv\\Scripts\\python.exe tools/compare_spaces.py --a legacy_5clip_mean --b full_mean_v1
    .\\.venv\\Scripts\\python.exe tools/compare_spaces.py --a full_mean_v1 --b temporal_stats_v1
    .\\.venv\\Scripts\\python.exe tools/compare_spaces.py --a legacy_5clip_mean --a-seq <sequence_version> --b-id 7

输出（默认写到配置的 logs 目录）::

    logs/space_compare_<a>_<b>.json   汇总 + 逐作品明细
    logs/space_compare_<a>_<b>.csv    逐作品一行（Jaccard/新旧 Top1/丢失/新增邻居等）

指标：

1. 每作品 Top-k 邻居重合：Jaccard(|交集| / |并集|)、交集数量；
   stable_works = Top-k 集合完全相同的作品数；
2. 整体稳定对比例：把所有 canonical 作品的无序对视为“是否在任一方 Top-k 中相邻”，
   两个空间判定一致的 pair 占比；同时给出有向 Top-k 边的保留率/交并比与互为邻居对的稳定性；
3. 被替换案例：Jaccard 最低的作品，含标题/来源与新旧邻居（lost/gained）；
4. 两空间作品对余弦矩阵相关：共同作品上、上三角作品对余弦的 Pearson/Spearman 相关、
   平均绝对差与 Top1 一致率（注意：不同表征维度可以不同，余弦本身可比）；
5. hubness 前后变化：每部作品被其他作品列入 Top-k 的次数（在各自空间的全部 canonical
   作品上统计），给出 Top hub、变化最大的作品与共同作品上的秩相关。

本工具只读已登记向量（legacy 走 embeddings 表，派生表征走 representation_vectors），
不写数据库、不加载模型、不做 UMAP；不同表征/不同序列的矩阵不会混在一起。
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
from nebula.log import configure_logging  # noqa: E402
from nebula.representations import (  # noqa: E402
    load_representation_matrix,
    representation_label,
    resolve_representation,
)
from nebula.util import now_utc_iso, slugify  # noqa: E402

DEFAULT_K = 5
DEFAULT_TOP_N = 10
DEFAULT_KIND = "joint"


@dataclass(frozen=True)
class Space:
    """一个空间（某个表征在某序列版本下的 canonical 作品矩阵）。"""

    label: str
    name: str
    sequence_version: str
    representation_id: int
    works: tuple[int, ...]
    matrix: np.ndarray

    @property
    def n_works(self) -> int:
        return len(self.works)

    def index(self) -> dict[int, int]:
        return {work_id: position for position, work_id in enumerate(self.works)}


def _normalize_matrix(matrix: np.ndarray, *, where: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise NebulaError(f"{where} 向量矩阵形状非法: {values.shape}")
    if not np.all(np.isfinite(values)):
        raise NebulaError(f"{where} 向量矩阵包含 NaN/Inf")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    zero_rows = np.where(norms <= 0)[0]
    if zero_rows.size:
        raise NebulaError(f"{where} 有 {zero_rows.size} 个零向量，无法计算余弦相似度")
    return (values / norms).astype(np.float32)


def load_space(
    catalog: Catalog,
    config: Config,
    *,
    name: str | None,
    sequence_version: str | None = None,
    representation_id: int | None = None,
    kind: str = DEFAULT_KIND,
) -> Space:
    """解析表征并加载其 canonical 作品矩阵（按 work_id 排序）。"""

    if representation_id is not None and name is not None:
        raise NebulaError("--a/--b 与 --a-id/--b-id 不能同时指定")
    rep = resolve_representation(
        catalog,
        config,
        name=name,
        sequence_version=sequence_version,
        representation_id=representation_id,
    )
    work_ids, matrix = load_representation_matrix(catalog, config, rep, kind=kind)
    if len(work_ids) < 2:
        raise NebulaError(
            f"表征 {rep['name']}@{rep['sequence_version']} 只有 {len(work_ids)} 部作品，"
            "至少需要 2 部才能比较邻域"
        )
    normalized = _normalize_matrix(matrix, where=f"{rep['name']}@{rep['sequence_version']}")
    return Space(
        label=str(rep["name"]),
        name=str(rep["name"]),
        sequence_version=str(rep["sequence_version"]),
        representation_id=int(rep["id"]),
        works=tuple(int(work_id) for work_id in work_ids),
        matrix=normalized,
    )


def _similarity_matrix(space: Space) -> np.ndarray:
    values = space.matrix.astype(np.float64) @ space.matrix.astype(np.float64).T
    return np.clip(values, -1.0, 1.0)


def compute_topk(similarities: np.ndarray, works: Sequence[int], k: int) -> dict[int, tuple[int, ...]]:
    """每部作品的 Top-k 邻居（不含自身；相似度降序，并列按 work_id 升序）。"""

    count = len(works)
    take = min(int(k), count - 1)
    if take < 1:
        return {int(work_id): () for work_id in works}
    topk: dict[int, tuple[int, ...]] = {}
    for position, work_id in enumerate(works):
        row = similarities[position].copy()
        row[position] = -np.inf
        order = np.argsort(-row, kind="stable")[:take]
        topk[int(work_id)] = tuple(int(works[int(index)]) for index in order)
    return topk


def compute_hubness(topk: Mapping[int, Sequence[int]]) -> dict[int, int]:
    """每部作品被其他作品列入 Top-k 的次数（hubness 入度）。"""

    counts: dict[int, int] = {int(work_id): 0 for work_id in topk}
    for targets in topk.values():
        for target in targets:
            counts[int(target)] = counts.get(int(target), 0) + 1
    return counts


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2:
        return 0.0
    a = first.astype(np.float64)
    b = second.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def _ranks(values: np.ndarray) -> np.ndarray:
    """平均秩（处理并列），用于 Spearman 相关。"""

    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        rank = (start + end) / 2.0 + 1.0
        ranks[order[start : end + 1]] = rank
        start = end + 1
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    return _pearson(
        _ranks(np.asarray(first, dtype=np.float64)),
        _ranks(np.asarray(second, dtype=np.float64)),
    )


def _work_meta(catalog: Catalog, work_id: int) -> dict[str, Any]:
    work = catalog.get_work(int(work_id)) or {}
    return {
        "work_id": int(work_id),
        "title": work.get("title"),
        "source": work.get("source"),
        "bvid": work.get("bvid"),
        "local_path": work.get("local_path"),
    }


def _neighbor_rows(catalog: Catalog, ids: Sequence[int]) -> list[dict[str, Any]]:
    return [_work_meta(catalog, work_id) for work_id in ids]


def _undirected_pairs(
    works: Sequence[int],
    topk: Mapping[int, Sequence[int]],
    allowed: set[int],
) -> set[tuple[int, int]]:
    """把有向 Top-k 边折叠成无序对（只保留两端都在 allowed 内的边）。"""

    pairs: set[tuple[int, int]] = set()
    for work_id in works:
        for target in topk.get(int(work_id), ()):
            if int(target) in allowed:
                pairs.add((min(int(work_id), int(target)), max(int(work_id), int(target))))
    return pairs


def _mutual_pairs(
    works: Sequence[int],
    topk: Mapping[int, Sequence[int]],
    allowed: set[int],
) -> set[tuple[int, int]]:
    mutual: set[tuple[int, int]] = set()
    for work_id in works:
        for target in topk.get(int(work_id), ()):
            if int(target) in allowed and int(work_id) in topk.get(int(target), ()):
                mutual.add((min(int(work_id), int(target)), max(int(work_id), int(target))))
    return mutual


def _cosine_comparison(
    space_a: Space,
    space_b: Space,
    common: Sequence[int],
) -> dict[str, Any]:
    positions_a = space_a.index()
    positions_b = space_b.index()
    rows_a = np.asarray([positions_a[work_id] for work_id in common], dtype=np.int64)
    rows_b = np.asarray([positions_b[work_id] for work_id in common], dtype=np.int64)
    matrix_a = _similarity_matrix(space_a)[np.ix_(rows_a, rows_a)]
    matrix_b = _similarity_matrix(space_b)[np.ix_(rows_b, rows_b)]
    upper = np.triu_indices(len(common), k=1)
    values_a = matrix_a[upper]
    values_b = matrix_b[upper]
    if values_a.size == 0:
        return {
            "n_pairs": 0,
            "pearson": 0.0,
            "spearman": 0.0,
            "mean_abs_diff": 0.0,
            "max_abs_diff": 0.0,
            "same_top1_ratio": 0.0,
        }
    same_top1 = 0
    for position, _work_id in enumerate(common):
        row_a = matrix_a[position].copy()
        row_a[position] = -np.inf
        row_b = matrix_b[position].copy()
        row_b[position] = -np.inf
        if int(np.argmax(row_a)) == int(np.argmax(row_b)):
            same_top1 += 1
    return {
        "n_pairs": int(values_a.size),
        "pearson": round(_pearson(values_a, values_b), 6),
        "spearman": round(_spearman(values_a, values_b), 6),
        "mean_abs_diff": round(float(np.mean(np.abs(values_a - values_b))), 6),
        "max_abs_diff": round(float(np.max(np.abs(values_a - values_b))), 6),
        "same_top1_ratio": round(same_top1 / len(common), 6),
    }


def compare_spaces(
    catalog: Catalog,
    space_a: Space,
    space_b: Space,
    *,
    k: int = DEFAULT_K,
    top_n: int = DEFAULT_TOP_N,
    kind: str = DEFAULT_KIND,
) -> dict[str, Any]:
    """比较两个空间的邻域迁移；返回可直接写 JSON 的字典。"""

    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise NebulaError(f"k 必须是 >= 1 的整数，收到 {k!r}")
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise NebulaError(f"top-n 必须是 >= 1 的整数，收到 {top_n!r}")

    works_a = set(space_a.works)
    works_b = set(space_b.works)
    common = sorted(works_a & works_b)
    if len(common) < 2:
        raise NebulaError(
            f"两个空间的共同 canonical 作品不足 2 部（{len(common)} 部），无法比较邻域"
        )
    only_a = sorted(works_a - works_b)
    only_b = sorted(works_b - works_a)
    allowed = set(common)
    effective_k = min(int(k), len(common) - 1)

    similarities_a = _similarity_matrix(space_a)
    similarities_b = _similarity_matrix(space_b)
    topk_a = compute_topk(similarities_a, space_a.works, k)
    topk_b = compute_topk(similarities_b, space_b.works, k)

    jaccards: list[float] = []
    overlaps: list[int] = []
    replaced: list[dict[str, Any]] = []
    stable_works = 0
    same_top1 = 0
    for work_id in common:
        set_a = set(topk_a[work_id])
        set_b = set(topk_b[work_id])
        intersection = set_a & set_b
        union = set_a | set_b
        jaccard = len(intersection) / len(union) if union else 1.0
        jaccards.append(jaccard)
        overlaps.append(len(intersection))
        if jaccard == 1.0:
            stable_works += 1
        if topk_a[work_id][:1] == topk_b[work_id][:1]:
            same_top1 += 1
        if jaccard < 1.0:
            replaced.append(
                {
                    "work_id": int(work_id),
                    "jaccard": round(jaccard, 6),
                    "overlap": len(intersection),
                    "effective_k": effective_k,
                    "old_top1": topk_a[work_id][0] if topk_a[work_id] else None,
                    "new_top1": topk_b[work_id][0] if topk_b[work_id] else None,
                    "lost_neighbors": _neighbor_rows(
                        catalog, sorted(set_a - set_b, key=int)
                    ),
                    "gained_neighbors": _neighbor_rows(
                        catalog, sorted(set_b - set_a, key=int)
                    ),
                    "old_topk": _neighbor_rows(catalog, topk_a[work_id]),
                    "new_topk": _neighbor_rows(catalog, topk_b[work_id]),
                }
            )

    replaced.sort(key=lambda item: (item["jaccard"], item["work_id"]))
    work_meta = {work_id: _work_meta(catalog, work_id) for work_id in common}

    edges_a = {
        (int(work_id), int(target))
        for work_id in common
        for target in topk_a[work_id]
        if int(target) in allowed
    }
    edges_b = {
        (int(work_id), int(target))
        for work_id in common
        for target in topk_b[work_id]
        if int(target) in allowed
    }
    pairs_a = _undirected_pairs(common, topk_a, allowed)
    pairs_b = _undirected_pairs(common, topk_b, allowed)
    mutual_a = _mutual_pairs(common, topk_a, allowed)
    mutual_b = _mutual_pairs(common, topk_b, allowed)
    all_pairs = len(common) * (len(common) - 1) // 2
    pair_agree = 0
    for position, first in enumerate(common):
        for second in common[position + 1 :]:
            if ((first, second) in pairs_a) == ((first, second) in pairs_b):
                pair_agree += 1

    def _edge_metrics(
        first: set[tuple[int, int]], second: set[tuple[int, int]]
    ) -> dict[str, Any]:
        intersection = len(first & second)
        union = len(first | second)
        return {
            "edges_a": len(first),
            "edges_b": len(second),
            "intersection": intersection,
            "union": union,
            "jaccard": round(intersection / union, 6) if union else 0.0,
            "retained_ratio": round(intersection / len(first), 6) if first else 0.0,
        }

    cosine = _cosine_comparison(space_a, space_b, common)

    hub_a = compute_hubness(topk_a)
    hub_b = compute_hubness(topk_b)
    deltas = [
        {
            "work_id": int(work_id),
            "title": work_meta[work_id]["title"],
            "source": work_meta[work_id]["source"],
            "count_a": int(hub_a.get(work_id, 0)),
            "count_b": int(hub_b.get(work_id, 0)),
            "delta": int(hub_b.get(work_id, 0)) - int(hub_a.get(work_id, 0)),
        }
        for work_id in common
    ]
    increased = sorted(
        (item for item in deltas if item["delta"] > 0),
        key=lambda item: (-item["delta"], item["work_id"]),
    )[:top_n]
    decreased = sorted(
        (item for item in deltas if item["delta"] < 0),
        key=lambda item: (item["delta"], item["work_id"]),
    )[:top_n]
    hub_top_a = sorted(
        ((work_id, count) for work_id, count in hub_a.items()),
        key=lambda item: (-item[1], item[0]),
    )[:top_n]
    hub_top_b = sorted(
        ((work_id, count) for work_id, count in hub_b.items()),
        key=lambda item: (-item[1], item[0]),
    )[:top_n]
    hub_rank_correlation = (
        _spearman(
            np.asarray([hub_a.get(work_id, 0) for work_id in common], dtype=np.float64),
            np.asarray([hub_b.get(work_id, 0) for work_id in common], dtype=np.float64),
        )
        if len(common) >= 2
        else 0.0
    )

    return {
        "generated_at": now_utc_iso(),
        "kind": kind,
        "k": int(k),
        "effective_k": int(effective_k),
        "top_n": int(top_n),
        "a": {
            "name": space_a.name,
            "label": representation_label(space_a.name),
            "sequence_version": space_a.sequence_version,
            "representation_id": space_a.representation_id,
            "n_works": space_a.n_works,
        },
        "b": {
            "name": space_b.name,
            "label": representation_label(space_b.name),
            "sequence_version": space_b.sequence_version,
            "representation_id": space_b.representation_id,
            "n_works": space_b.n_works,
        },
        "scope": {
            "n_common": len(common),
            "n_only_in_a": len(only_a),
            "n_only_in_b": len(only_b),
            "only_in_a": _neighbor_rows(catalog, only_a[:top_n]),
            "only_in_b": _neighbor_rows(catalog, only_b[:top_n]),
            "note": "Jaccard/余弦相关只统计共同 canonical 作品；hubness 在各自空间全部 canonical 作品上统计。",
        },
        "topk": {
            "mean_jaccard": round(float(np.mean(jaccards)), 6),
            "median_jaccard": round(float(np.median(jaccards)), 6),
            "mean_overlap": round(float(np.mean(overlaps)), 4),
            "stable_works": int(stable_works),
            "stable_ratio": round(stable_works / len(common), 6),
            "same_top1_ratio": round(same_top1 / len(common), 6),
            "n_works": len(common),
        },
        "pairs": {
            "directed_edges": _edge_metrics(edges_a, edges_b),
            "undirected_edges": _edge_metrics(pairs_a, pairs_b),
            "mutual_edges": _edge_metrics(mutual_a, mutual_b),
            "stable_pair_ratio": round(pair_agree / all_pairs, 6) if all_pairs else 0.0,
            "n_pairs": all_pairs,
        },
        "cosine": cosine,
        "hubness": {
            "rank_correlation": round(hub_rank_correlation, 6),
            "top_hubs_a": [
                {"work_id": int(work_id), "count": int(count), **_work_meta(catalog, work_id)}
                for work_id, count in hub_top_a
            ],
            "top_hubs_b": [
                {"work_id": int(work_id), "count": int(count), **_work_meta(catalog, work_id)}
                for work_id, count in hub_top_b
            ],
            "top_increased": increased,
            "top_decreased": decreased,
        },
        "replaced_works": replaced[:top_n],
        "works": [
            {
                **work_meta[work_id],
                "jaccard": round(jaccards[position], 6),
                "overlap": overlaps[position],
                "old_topk": list(topk_a[work_id]),
                "new_topk": list(topk_b[work_id]),
                "hub_count_a": int(hub_a.get(work_id, 0)),
                "hub_count_b": int(hub_b.get(work_id, 0)),
                "hub_delta": int(hub_b.get(work_id, 0)) - int(hub_a.get(work_id, 0)),
            }
            for position, work_id in enumerate(common)
        ],
    }


def _report_basename(space_a: Space, space_b: Space) -> str:
    name = f"{slugify(space_a.name)}_vs_{slugify(space_b.name)}"
    sequences = slugify(f"{space_a.sequence_version}_{space_b.sequence_version}")
    return f"space_compare_{name}__{sequences}"


def write_reports(
    report: Mapping[str, Any],
    *,
    json_path: Path,
    csv_path: Path,
) -> None:
    """原子写入 JSON 与 CSV 报告。"""

    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=False)
    temp_json = json_path.with_name(json_path.name + ".tmp")
    temp_json.write_text(payload + "\n", encoding="utf-8")
    os.replace(temp_json, json_path)

    columns = (
        "work_id",
        "title",
        "source",
        "bvid",
        "jaccard",
        "overlap",
        "old_top1",
        "new_top1",
        "old_topk",
        "new_topk",
        "hub_count_a",
        "hub_count_b",
        "hub_delta",
    )
    temp_csv = csv_path.with_name(csv_path.name + ".tmp")
    with temp_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in report.get("works", []):
            values = dict(row)
            values["old_topk"] = " ".join(str(item) for item in values.get("old_topk", []))
            values["new_topk"] = " ".join(str(item) for item in values.get("new_topk", []))
            writer.writerow({column: values.get(column, "") for column in columns})
    os.replace(temp_csv, csv_path)


def _print_summary(report: Mapping[str, Any], *, json_path: Path, csv_path: Path) -> None:
    print("空间邻域迁移对比（v0.4）")
    print(f"  A: {report['a']['name']}@{report['a']['sequence_version']}（{report['a']['n_works']} 部）")
    print(f"  B: {report['b']['name']}@{report['b']['sequence_version']}（{report['b']['n_works']} 部）")
    scope = report["scope"]
    print(
        f"  共同作品: {scope['n_common']} 部（仅 A: {scope['n_only_in_a']}，仅 B: {scope['n_only_in_b']}）"
    )
    topk = report["topk"]
    print(
        f"  Top-{report['k']}: 平均 Jaccard={topk['mean_jaccard']:.4f}，"
        f"中位={topk['median_jaccard']:.4f}，完全稳定 {topk['stable_works']}/{topk['n_works']}"
        f"（{topk['stable_ratio']:.2%}），Top1 一致率={topk['same_top1_ratio']:.2%}"
    )
    pairs = report["pairs"]
    print(
        f"  稳定对比例: {pairs['stable_pair_ratio']:.2%}；有向边保留率="
        f"{pairs['directed_edges']['retained_ratio']:.2%}"
        f"（交并比 {pairs['directed_edges']['jaccard']:.4f}）；"
        f"互为邻居对交并比={pairs['mutual_edges']['jaccard']:.4f}"
    )
    cosine = report["cosine"]
    print(
        f"  作品对余弦矩阵: Pearson={cosine['pearson']:.4f}，Spearman={cosine['spearman']:.4f}，"
        f"平均绝对差={cosine['mean_abs_diff']:.4f}"
    )
    hub = report["hubness"]
    top_a = hub["top_hubs_a"][0] if hub["top_hubs_a"] else None
    top_b = hub["top_hubs_b"][0] if hub["top_hubs_b"] else None
    print(
        "  hubness 秩相关: "
        f"{hub['rank_correlation']:.4f}；Top hub A={top_a['work_id'] if top_a else '-'}"
        f"({top_a['count'] if top_a else 0} 次) / B={top_b['work_id'] if top_b else '-'}"
        f"({top_b['count'] if top_b else 0} 次)"
    )
    replaced = report.get("replaced_works", [])
    if replaced:
        first = replaced[0]
        print(
            f"  替换最大案例: work {first['work_id']}（Jaccard={first['jaccard']:.3f}，"
            f"{first.get('title')}）"
        )
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/compare_spaces.py",
        description="比较两个表征空间的邻域迁移（Top-k 重合/稳定对/余弦相关/hubness）。",
    )
    parser.add_argument("--config", default=None, help="配置文件路径（默认 NEBULA_CONFIG 或 config.yaml）")
    parser.add_argument("--a", default=None, help="A 表征名（legacy_5clip_mean/full_mean_v1/temporal_stats_v1/别名）")
    parser.add_argument("--b", default=None, help="B 表征名")
    parser.add_argument("--a-seq", default=None, help="A 的序列版本（默认按表征/当前模式解析）")
    parser.add_argument("--b-seq", default=None, help="B 的序列版本")
    parser.add_argument("--a-id", type=int, default=None, help="A 的表征 id（与 --a 互斥）")
    parser.add_argument("--b-id", type=int, default=None, help="B 的表征 id（与 --b 互斥）")
    parser.add_argument(
        "--kind",
        default=DEFAULT_KIND,
        choices=("joint", "audio", "video"),
        help="比较的模态（默认 joint）",
    )
    parser.add_argument("-k", "--k", type=int, default=DEFAULT_K, help=f"Top-k（默认 {DEFAULT_K}）")
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"替换案例/Top hub 数量（默认 {DEFAULT_TOP_N}）",
    )
    parser.add_argument("--out-dir", default=None, help="输出目录（默认配置里的 logs）")
    parser.add_argument(
        "--log-level", default=None, choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    return parser


def _resolve_config_path(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    env_value = os.environ.get("NEBULA_CONFIG")
    if env_value:
        return Path(env_value).expanduser()
    return Path.cwd() / "config.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.a is not None and args.a_id is not None:
        print("--a 与 --a-id 不能同时指定", file=sys.stderr)
        return 2
    if args.b is not None and args.b_id is not None:
        print("--b 与 --b-id 不能同时指定", file=sys.stderr)
        return 2
    # 默认比较 legacy × full_mean（各自取当前/最新 ready 表征）
    if args.a is None and args.a_id is None:
        args.a = "legacy"
    if args.b is None and args.b_id is None:
        args.b = "full_mean"
    try:
        config = load_config(_resolve_config_path(args.config))
    except NebulaError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2
    configure_logging(args.log_level or config.logging.level)

    catalog = Catalog(config.paths.db_path)
    try:
        catalog.initialize()
        space_a = load_space(
            catalog,
            config,
            name=args.a,
            sequence_version=args.a_seq,
            representation_id=args.a_id,
            kind=args.kind,
        )
        space_b = load_space(
            catalog,
            config,
            name=args.b,
            sequence_version=args.b_seq,
            representation_id=args.b_id,
            kind=args.kind,
        )
        report = compare_spaces(
            catalog, space_a, space_b, k=args.k, top_n=args.top_n, kind=args.kind
        )
    except NebulaError as exc:
        print(f"空间对比失败: {exc}", file=sys.stderr)
        return 1
    finally:
        catalog.close()

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else config.paths.logs
    basename = _report_basename(space_a, space_b)
    json_path = out_dir / f"{basename}.json"
    csv_path = out_dir / f"{basename}.csv"
    try:
        write_reports(report, json_path=json_path, csv_path=csv_path)
    except NebulaError as exc:
        print(f"写入报告失败: {exc}", file=sys.stderr)
        return 1

    _print_summary(report, json_path=json_path, csv_path=csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
