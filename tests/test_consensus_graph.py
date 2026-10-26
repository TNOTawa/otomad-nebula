"""tools/build_consensus_graph.py 的测试。

覆盖：
- 3 个合成空间的已知线型邻域结构：Top-k 有向邻接、mutual 对、跨空间保留率、
  共识边（>=2 / 全共识）、覆盖作品比例、共识强度与节点度数；
- rank-average 融合与 concat 向量融合（等权/1/sqrt(dim)）的独立手算指标；
- 确定性、JSON/edges CSV/nodes CSV schema；
- DB 加载（canonical/状态校验）、CLI --help、多表征端到端与默认表征缺失错误；
- 全部测试只用内存合成矩阵或临时 SQLite，不加载 PE-AV、不联网、不写 data/。
"""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from nebula import representations as R
from nebula.errors import NebulaError
from tools import build_consensus_graph as bcg

ORDERS: tuple[tuple[int, ...], ...] = (
    (1, 2, 3, 4, 5, 6),
    (2, 3, 4, 5, 6, 1),
    (3, 4, 5, 6, 1, 2),
)
DIMS: tuple[int, ...] = (2, 3, 4)


# --------------------------------------------------------------------------- 合成线型空间
#: 相邻位置的角间隔（互不相等且总角 <= pi；任意多段和都大于单个间隔，且所有距离互异）
GAPS: tuple[float, ...] = (0.50, 0.54, 0.58, 0.62, 0.66)
POSITION_ANGLES: tuple[float, ...] = tuple(
    float(np.cumsum((0.0, *GAPS))[index]) for index in range(len(GAPS) + 1)
)


def _line_matrix(order: tuple[int, ...], dim: int) -> np.ndarray:
    """把 order 中的作品按位置放到单位圆上（角度间隔非均匀，Top-2 邻居可手算）。"""

    count = len(order)
    unit = np.stack(
        [np.cos(np.asarray(POSITION_ANGLES)), np.sin(np.asarray(POSITION_ANGLES))],
        axis=1,
    )
    matrix = np.zeros((count, dim), dtype=np.float64)
    for position, work_id in enumerate(order):
        matrix[work_id - 1, :2] = unit[position]
    return matrix


def _line_space(index: int) -> bcg.SpaceData:
    order = ORDERS[index]
    return bcg.SpaceData.from_matrix(
        _line_matrix(order, DIMS[index]),
        name=f"space_{chr(ord('a') + index)}",
        label=f"Space {index}",
        sequence_version=f"synthetic-v{index}",
        representation_id=index + 1,
        work_ids=tuple(range(1, len(order) + 1)),
    )


def _line_spaces() -> tuple[bcg.SpaceData, ...]:
    return tuple(_line_space(index) for index in range(len(ORDERS)))


def _line_cosine(order: tuple[int, ...], first: int, second: int) -> float:
    position = {work_id: index for index, work_id in enumerate(order)}
    delta = abs(POSITION_ANGLES[position[first]] - POSITION_ANGLES[position[second]])
    return float(np.cos(delta))


def _expected_topk(order: tuple[int, ...], k: int) -> dict[int, tuple[int, ...]]:
    """独立按余弦降序 + work_id 升序 tie-break 计算 Top-k 邻居。"""

    work_ids = tuple(range(1, len(order) + 1))
    result: dict[int, tuple[int, ...]] = {}
    for work_id in work_ids:
        ranked = sorted(
            (
                (-_line_cosine(order, work_id, other), other)
                for other in work_ids
                if other != work_id
            ),
            key=lambda item: (item[0], item[1]),
        )
        result[work_id] = tuple(other for _negative_cos, other in ranked[:k])
    return result


def _expected_directed(topk: dict[int, tuple[int, ...]]) -> set[tuple[int, int]]:
    return {(work_id, other) for work_id, others in topk.items() for other in others}


def _expected_mutual(topk: dict[int, tuple[int, ...]]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for work_id, others in topk.items():
        for other in others:
            if work_id in topk[other]:
                pairs.add((min(work_id, other), max(work_id, other)))
    return pairs


def _jaccard(first: set[int], second: set[int]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def _topk_indices_from_similarity(similarity: np.ndarray, k: int) -> list[tuple[int, ...]]:
    work = np.asarray(similarity, dtype=np.float64).copy()
    np.fill_diagonal(work, -np.inf)
    order = np.argsort(-work, axis=1, kind="stable")
    return [tuple(int(value) for value in row[:k]) for row in order]


def _topk_from_similarity(similarity: np.ndarray, k: int) -> list[tuple[int, ...]]:
    return _topk_indices_from_similarity(similarity, k)


def _fused_metrics_expected(
    fused_orders: list[tuple[int, ...]],
    topks: list[dict[int, tuple[int, ...]]],
    consensus_pairs: set[tuple[int, int]],
) -> dict[str, float]:
    work_ids = sorted(topks[0])
    jaccards_all: list[float] = []
    consensus_jaccards: list[float] = []
    for topk in topks:
        for position, work_id in enumerate(work_ids):
            jaccards_all.append(
                _jaccard(set(fused_orders[position]), set(topk[work_id]))
            )
    consensus_neighbors: dict[int, set[int]] = {work_id: set() for work_id in work_ids}
    for first, second in consensus_pairs:
        consensus_neighbors[first].add(second)
        consensus_neighbors[second].add(first)
    for position, work_id in enumerate(work_ids):
        reference = consensus_neighbors[work_id]
        if reference:
            consensus_jaccards.append(_jaccard(set(fused_orders[position]), reference))
    fused_pairs: set[tuple[int, int]] = set()
    for position, work_id in enumerate(work_ids):
        for target in fused_orders[position]:
            fused_pairs.add((min(work_id, target), max(work_id, target)))
    edge_union = fused_pairs | consensus_pairs
    return {
        "mean_jaccard_vs_spaces": float(np.mean(jaccards_all)),
        "jaccard_mean_works_with_consensus": float(np.mean(consensus_jaccards)),
        "edge_jaccard": (
            len(fused_pairs & consensus_pairs) / len(edge_union) if edge_union else 0.0
        ),
    }


# --------------------------------------------------------------------------- 邻接与共识
def test_known_line_topology():
    space = _line_space(0)
    topology = bcg.build_topology(space, 2)
    expected = _expected_topk(ORDERS[0], 2)
    for position, work_id in enumerate(space.work_ids):
        observed = tuple(
            space.work_ids[int(index)] for index in topology.neighbors[position]
        )
        assert observed == expected[work_id]
    assert len(topology.directed) == 12
    assert topology.directed == frozenset(_expected_directed(expected))
    assert topology.mutual == frozenset(_expected_mutual(expected))
    assert len(topology.mutual) == 5
    assert 0.0 < topology.mean_topk_cosine < 1.0
    repeated = bcg.build_topology(space, 2)
    assert repeated.neighbors.tolist() == topology.neighbors.tolist()
    assert repeated.mutual == topology.mutual


def test_known_cross_space_retention_and_consensus_graph():
    spaces = _line_spaces()
    report = bcg.build_consensus(spaces, k=2, min_spaces=2, fusion="both")
    topks = [_expected_topk(order, 2) for order in ORDERS]
    directed = [_expected_directed(topk) for topk in topks]
    mutual = [_expected_mutual(topk) for topk in topks]

    for space_report, mutual_set in zip(report["spaces"], mutual, strict=True):
        assert space_report["directed_edges"] == 12
        assert space_report["mutual_edges"] == len(mutual_set) == 5
        assert space_report["dim"] in DIMS
        assert space_report["mutual_ratio"] == pytest.approx(2 * 5 / 12, abs=1e-6)

    # 跨空间保留率
    for first in range(3):
        for second in range(first + 1, 3):
            entry = next(
                item
                for item in report["consensus"]["cross_space"]
                if item["a"] == spaces[first].name and item["b"] == spaces[second].name
            )
            directed_intersection = len(directed[first] & directed[second])
            directed_union = len(directed[first] | directed[second])
            mutual_intersection = len(mutual[first] & mutual[second])
            mutual_union = len(mutual[first] | mutual[second])
            assert entry["directed_intersection"] == directed_intersection
            assert entry["directed_union"] == directed_union
            assert entry["directed_jaccard"] == pytest.approx(
                directed_intersection / directed_union, abs=1e-6
            )
            assert entry["directed_retention_a_to_b"] == pytest.approx(
                directed_intersection / len(directed[first]), abs=1e-6
            )
            assert entry["mutual_intersection"] == mutual_intersection
            assert entry["mutual_union"] == mutual_union
            assert entry["mutual_jaccard"] == pytest.approx(
                mutual_intersection / mutual_union, abs=1e-6
            )

    # 共识边计数
    expected_counts: dict[tuple[int, int], int] = {}
    for mutual_set in mutual:
        for pair in mutual_set:
            expected_counts[pair] = expected_counts.get(pair, 0) + 1
    expected_consensus = {
        pair for pair, count in expected_counts.items() if count >= 2
    }
    expected_full = {pair for pair, count in expected_counts.items() if count == 3}
    observed_consensus = {
        (edge["work_id_a"], edge["work_id_b"]) for edge in report["consensus_edges"]
    }
    observed_full = {
        (edge["work_id_a"], edge["work_id_b"]) for edge in report["consensus_edges"]
        if edge["is_full_consensus"]
    }
    assert observed_consensus == expected_consensus
    assert observed_full == expected_full
    assert len(expected_consensus) == 6
    assert len(expected_full) == 3
    observed_counts = {
        (edge["work_id_a"], edge["work_id_b"]): edge["consensus_count"]
        for edge in report["edges"]
        if edge["consensus_count"] > 0
    }
    assert observed_counts == {
        pair: count for pair, count in expected_counts.items() if count > 0
    }
    for edge in report["edges"]:
        pair = (edge["work_id_a"], edge["work_id_b"])
        assert edge["consensus_count"] == expected_counts.get(pair, 0)
        assert edge["consensus_strength"] == pytest.approx(
            expected_counts.get(pair, 0) / 3, abs=1e-6
        )
        assert edge["is_consensus"] == (pair in expected_consensus)
        assert edge["is_full_consensus"] == (pair in expected_full)
        assert edge["mutual_spaces"] == [
            space.name for space, mutual_set in zip(spaces, mutual, strict=True)
            if pair in mutual_set
        ]
        for space_index, space in enumerate(spaces):
            values = edge["per_space"][space.name]
            assert values["mutual"] == (pair in mutual[space_index])

    graph = report["consensus"]
    assert graph["n_works_common"] == 6
    undirected_union = {
        (min(work_a, work_b), max(work_a, work_b))
        for work_a, work_b in set().union(*directed)
    }
    assert graph["n_edges"] == len(undirected_union)
    assert graph["n_consensus_edges"] == 6
    assert graph["n_full_consensus_edges"] == 3
    assert graph["consensus_histogram"] == {"0": 6, "1": 0, "2": 3, "3": 3}
    assert graph["consensus_coverage_works"] == 6
    assert graph["consensus_coverage_ratio"] == 1.0
    assert graph["mean_consensus_strength"] == pytest.approx(
        5.0 / 6.0, abs=1e-6
    )
    assert graph["mean_consensus_degree"] == 2.0
    assert graph["max_consensus_degree"] == 2
    assert graph["consensus_strength_stats"] == {
        "mean": pytest.approx(5 / 6, abs=1e-6),
        "std": pytest.approx(
            float(
                np.std(
                    [2 / 3, 2 / 3, 2 / 3, 1.0, 1.0, 1.0]
                )
            ),
            abs=1e-6,
        ),
        "min": pytest.approx(2 / 3, abs=1e-6),
        "max": pytest.approx(1.0, abs=1e-6),
    }
    assert graph["full_consensus_coverage_works"] == 4
    assert graph["full_consensus_coverage_ratio"] == pytest.approx(4 / 6)

    # 节点度数
    nodes = {node["work_id"]: node for node in report["nodes"]}
    assert set(nodes) == {1, 2, 3, 4, 5, 6}
    for work_id, node in nodes.items():
        assert node["consensus_degree"] == 2
        expected_full_degree = sum(
            1 for pair in expected_full if work_id in pair
        )
        assert node["full_consensus_degree"] == expected_full_degree
        expected_strength = sum(
            count / 3
            for pair, count in expected_counts.items()
            if count >= 2 and work_id in pair
        )
        assert node["consensus_strength_sum"] == pytest.approx(
            expected_strength, abs=1e-6
        )
    assert nodes[3]["full_consensus_degree"] == 1
    assert nodes[4]["full_consensus_degree"] == 2


def test_min_spaces_and_fusion_modes():
    spaces = _line_spaces()
    report = bcg.build_consensus(spaces, k=2, min_spaces=3, fusion="none")
    assert report["consensus"]["n_consensus_edges"] == 3
    assert report["consensus"]["n_full_consensus_edges"] == 3
    assert report["consensus"]["consensus_coverage_works"] == 4
    assert report["consensus"]["consensus_coverage_ratio"] == pytest.approx(4 / 6)
    assert report["fusion"] is None

    rank_only = bcg.build_consensus(spaces, k=2, fusion="rank")
    assert rank_only["fusion"]["rank_average"] is not None
    assert rank_only["fusion"]["concat"] is None

    concat_only = bcg.build_consensus(spaces, k=2, fusion="concat")
    assert concat_only["fusion"]["rank_average"] is None
    assert set(concat_only["fusion"]["concat"]) == {"equal", "inv_sqrt_dim"}

    with pytest.raises(NebulaError):
        bcg.build_consensus(spaces, k=2, min_spaces=1)
    with pytest.raises(NebulaError):
        bcg.build_consensus(spaces, k=2, min_spaces=4)
    with pytest.raises(NebulaError):
        bcg.build_consensus(spaces, k=0)
    with pytest.raises(NebulaError):
        bcg.build_consensus(spaces, fusion="bad")
    with pytest.raises(NebulaError):
        bcg.build_consensus(spaces[:1])
    duplicate = (_line_space(0), _line_space(0))
    with pytest.raises(NebulaError):
        bcg.build_consensus(duplicate)

    disjoint = (
        bcg.SpaceData.from_matrix(np.eye(2), name="a", work_ids=[1, 2]),
        bcg.SpaceData.from_matrix(np.eye(2), name="b", work_ids=[3, 4]),
    )
    with pytest.raises(NebulaError):
        bcg.build_consensus(disjoint)


def test_consensus_works_meta_and_generated_at():
    spaces = _line_spaces()
    works_meta = {
        work_id: {"title": f"作品{work_id}", "source": "local"}
        for work_id in range(1, 7)
    }
    report = bcg.build_consensus(
        spaces,
        k=2,
        works_meta=works_meta,
        generated_at="2026-10-26T00:00:00Z",
    )
    assert report["generated_at"] == "2026-10-26T00:00:00Z"
    assert [node["title"] for node in report["nodes"]] == [
        f"作品{work_id}" for work_id in range(1, 7)
    ]
    assert all(node["source"] == "local" for node in report["nodes"])


# --------------------------------------------------------------------------- 融合基线
def _rank_fusion_expected(k: int = 2):
    topks = [_expected_topk(order, k) for order in ORDERS]
    work_ids = sorted(topks[0])
    fused_orders: list[tuple[int, ...]] = []
    mean_scores: list[float] = []
    for work_id in work_ids:
        totals: dict[int, int] = {}
        appearances: dict[int, int] = {}
        for topk in topks:
            for rank, other in enumerate(topk[work_id], start=1):
                totals[other] = totals.get(other, 0) + rank
                appearances[other] = appearances.get(other, 0) + 1
        scores = {
            other: (totals[other] + (k + 1) * (3 - appearances[other])) / 3
            for other in totals
        }
        ordered = sorted(scores.items(), key=lambda item: (item[1], item[0]))
        chosen = tuple(other for other, _score in ordered[:k])
        fused_orders.append(chosen)
        mean_scores.append(float(np.mean([scores[other] for other in chosen])))
    consensus_counts: dict[tuple[int, int], int] = {}
    for topk in topks:
        for pair in _expected_mutual(topk):
            consensus_counts[pair] = consensus_counts.get(pair, 0) + 1
    consensus_pairs = {pair for pair, count in consensus_counts.items() if count >= 2}
    metrics = _fused_metrics_expected(fused_orders, topks, consensus_pairs)
    metrics["mean_fused_rank_score"] = float(np.mean(mean_scores))
    return metrics, fused_orders, consensus_pairs


def test_rank_average_fusion_known_values():
    spaces = _line_spaces()
    report = bcg.build_consensus(spaces, k=2, min_spaces=2, fusion="rank")
    expected, _fused_orders, _consensus = _rank_fusion_expected(2)
    values = report["fusion"]["rank_average"]
    assert values["k"] == 2
    assert values["n_works"] == 6
    assert values["n_spaces"] == 3
    assert values["mean_fused_rank_score"] == pytest.approx(
        expected["mean_fused_rank_score"], abs=1e-6
    )
    assert values["mean_jaccard_vs_spaces"] == pytest.approx(
        expected["mean_jaccard_vs_spaces"], abs=1e-6
    )
    assert values["jaccard_mean_works_with_consensus"] == pytest.approx(
        expected["jaccard_mean_works_with_consensus"], abs=1e-6
    )
    assert values["edge_jaccard"] == pytest.approx(
        expected["edge_jaccard"], abs=1e-6
    )
    assert 0.0 <= values["jaccard_mean_works_with_consensus"] <= 1.0
    assert 0.0 <= values["edge_jaccard"] <= 1.0
    assert set(values["per_space"]) == {space.name for space in spaces}


def test_concat_fusion_known_values():
    spaces = _line_spaces()
    report = bcg.build_consensus(spaces, k=2, min_spaces=2, fusion="concat")
    topks = [_expected_topk(order, 2) for order in ORDERS]
    consensus_counts: dict[tuple[int, int], int] = {}
    for topk in topks:
        for pair in _expected_mutual(topk):
            consensus_counts[pair] = consensus_counts.get(pair, 0) + 1
    consensus_pairs = {pair for pair, count in consensus_counts.items() if count >= 2}
    work_ids = sorted(topks[0])
    for scheme, weight in (("equal", None), ("inv_sqrt_dim", True)):
        blocks = []
        for space in spaces:
            block = np.asarray(space.matrix, dtype=np.float64)
            block_weight = (
                1.0
                if weight is None
                else 1.0 / math.sqrt(float(block.shape[1]))
            )
            blocks.append(block * block_weight)
        fused = np.concatenate(blocks, axis=1)
        fused = fused / np.linalg.norm(fused, axis=1, keepdims=True)
        similarity = np.clip(fused @ fused.T, -1.0, 1.0)
        fused_indices = _topk_indices_from_similarity(similarity, 2)
        fused_orders = [
            tuple(work_ids[int(index)] for index in row) for row in fused_indices
        ]
        expected = _fused_metrics_expected(fused_orders, topks, consensus_pairs)
        expected_neighbor_cosine = float(
            np.mean(
                [
                    np.mean(
                        [similarity[row, int(index)] for index in fused_indices[row]]
                    )
                    for row in range(len(work_ids))
                ]
            )
        )
        values = report["fusion"]["concat"][scheme]
        assert values["weight_scheme"] == scheme
        assert values["vector_dim"] == sum(space.dim for space in spaces)
        assert values["block_dims"] == [space.dim for space in spaces]
        assert values["mean_jaccard_vs_spaces"] == pytest.approx(
            expected["mean_jaccard_vs_spaces"], abs=1e-6
        )
        assert values["jaccard_mean_works_with_consensus"] == pytest.approx(
            expected["jaccard_mean_works_with_consensus"], abs=1e-6
        )
        assert values["edge_jaccard"] == pytest.approx(
            expected["edge_jaccard"], abs=1e-6
        )
        assert values["mean_fused_neighbor_cosine"] == pytest.approx(
            expected_neighbor_cosine, abs=1e-6
        )
        assert 0.0 <= values["mean_fused_neighbor_cosine"] <= 1.0


# --------------------------------------------------------------------------- schema / 输出
def test_report_schema_and_determinism():
    spaces = _line_spaces()
    report = bcg.build_consensus(spaces, k=2, min_spaces=2, fusion="both")
    repeated = bcg.build_consensus(spaces, k=2, min_spaces=2, fusion="both")
    assert report == repeated
    assert report["generated_at"] is None
    assert set(report) == {
        "tool",
        "version",
        "generated_at",
        "meta",
        "representations",
        "params",
        "spaces",
        "mutual_edges",
        "consensus",
        "nodes",
        "edges",
        "consensus_edges",
        "fusion",
    }
    assert report["meta"]["tool"] == bcg.TOOL_NAME
    assert report["meta"]["spaces"] == [space.name for space in spaces]
    assert report["representations"] == [space.name for space in spaces]
    assert report["mutual_edges"] == {space.name: 5 for space in spaces}
    assert report["params"] == {
        "k": 2,
        "min_spaces": 2,
        "fusion": "both",
        "kind": "joint",
        "representations": [space.name for space in spaces],
    }
    assert set(report["consensus"]) == {
        "n_spaces",
        "n_works_common",
        "k",
        "effective_k",
        "min_spaces",
        "n_edges",
        "n_consensus_edges",
        "n_full_consensus_edges",
        "consensus_histogram",
        "consensus_coverage_works",
        "consensus_coverage_ratio",
        "full_consensus_coverage_works",
        "full_consensus_coverage_ratio",
        "mean_consensus_strength",
        "consensus_strength_stats",
        "mean_consensus_degree",
        "max_consensus_degree",
        "cross_space",
    }
    edge_keys = {
        "work_id_a",
        "work_id_b",
        "consensus_count",
        "consensus_strength",
        "is_consensus",
        "is_full_consensus",
        "mutual_spaces",
        "per_space",
    }
    for edge in report["edges"]:
        assert set(edge) == edge_keys
        assert set(edge["per_space"]) == {space.name for space in spaces}
        for values in edge["per_space"].values():
            assert set(values) == {"rank_a", "rank_b", "cos", "mutual"}
            assert -1.0 <= values["cos"] <= 1.0
    node_keys = {
        "work_id",
        "title",
        "source",
        "mutual_degree",
        "consensus_degree",
        "full_consensus_degree",
        "consensus_strength_sum",
    }
    for node in report["nodes"]:
        assert set(node) == node_keys
        assert set(node["mutual_degree"]) == {space.name for space in spaces}
    # edges 排序：共识数降序，其次 work_a/work_b
    ordering = [
        (-edge["consensus_count"], edge["work_id_a"], edge["work_id_b"])
        for edge in report["edges"]
    ]
    assert ordering == sorted(ordering)
    # 报告可直接 JSON 序列化
    json.dumps(report, ensure_ascii=False)


def test_write_reports_json_and_csv(tmp_path):
    spaces = _line_spaces()
    report = bcg.build_consensus(spaces, k=2, min_spaces=2, fusion="both")
    json_path = tmp_path / "out" / "neighbor_consensus.json"
    csv_path = tmp_path / "out" / "neighbor_consensus.csv"
    written_json, edge_csv, node_csv = bcg.write_reports(
        report, json_path=json_path, csv_path=csv_path
    )
    assert written_json == json_path
    assert edge_csv == csv_path
    assert node_csv == tmp_path / "out" / "neighbor_consensus_nodes.csv"
    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    assert not list((tmp_path / "out").glob("*.tmp"))

    with edge_csv.open("r", encoding="utf-8", newline="") as handle:
        edge_rows = list(csv.DictReader(handle))
    expected_columns = [
        "work_id_a",
        "work_id_b",
        "consensus_count",
        "consensus_strength",
        "is_consensus",
        "is_full_consensus",
        "mutual_spaces",
    ]
    for space in spaces:
        expected_columns.extend(
            [
                f"{space.name}_rank_a",
                f"{space.name}_rank_b",
                f"{space.name}_cos",
                f"{space.name}_mutual",
            ]
        )
    assert edge_rows and tuple(edge_rows[0]) == tuple(expected_columns)
    assert len(edge_rows) == len(report["edges"])
    assert edge_rows[0]["work_id_a"] == str(report["edges"][0]["work_id_a"])
    assert edge_rows[0]["mutual_spaces"] == " ".join(
        report["edges"][0]["mutual_spaces"]
    )

    with node_csv.open("r", encoding="utf-8", newline="") as handle:
        node_rows = list(csv.DictReader(handle))
    assert len(node_rows) == len(report["nodes"])
    assert "consensus_degree" in node_rows[0]
    assert f"{spaces[0].name}_mutual_degree" in node_rows[0]
    assert [int(row["work_id"]) for row in node_rows] == [
        node["work_id"] for node in report["nodes"]
    ]


# --------------------------------------------------------------------------- DB / CLI
def _register_representation(
    catalog,
    config,
    *,
    name: str,
    version: str,
    matrix: np.ndarray,
    work_ids: list[int],
    status: str = "ready",
    kind: str = "full_mean",
) -> int:
    catalog.register_embedding_version(
        version,
        model_checkpoint="test-only:hash",
        model_revision=None,
        frames=16,
        clip_seconds=8.0,
        clips_per_work=0,
        audio_sample_rate=48000,
        params={},
    )
    rep_id = catalog.register_representation(
        name,
        sequence_version=version,
        kind=kind,
        params={"test": "consensus-graph", "name": name},
    )
    rep = catalog.get_representation(rep_id)
    assert rep is not None
    for work_id, vector in zip(work_ids, matrix, strict=True):
        R.save_representation_vectors(
            catalog,
            config,
            rep,
            work_id,
            {"joint": vector.astype(np.float32)},
        )
    catalog.set_representation_status(rep_id, status, dim=int(matrix.shape[1]))
    return rep_id


def _prepare_works(catalog, count: int = 6) -> list[int]:
    work_ids: list[int] = []
    for index in range(count):
        work_id, _ = catalog.upsert_local_work(
            f"consensus/w{index}.mp4", title=f"作品{index}"
        )
        work_ids.append(int(work_id))
    return work_ids


def test_load_space_and_run_consensus(config, catalog):
    work_ids = _prepare_works(catalog)
    for index, space in enumerate(_line_spaces()):
        _register_representation(
            catalog,
            config,
            name=space.name,
            version="consensus-test-v1",
            matrix=_line_matrix(ORDERS[index], DIMS[index]),
            work_ids=work_ids,
        )

    loaded = bcg.load_space(catalog, config, "space_a")
    assert loaded.name == "space_a"
    assert loaded.source_kind == "pe_av"
    assert loaded.work_ids == tuple(sorted(work_ids))
    assert loaded.dim == 2
    np.testing.assert_allclose(np.linalg.norm(loaded.matrix, axis=1), np.ones(6))

    report = bcg.run_consensus(
        catalog,
        config,
        representations=["space_a", "space_b", "space_c"],
        k=2,
        min_spaces=2,
        fusion="both",
    )
    assert report["generated_at"]
    assert report["params"]["representations"] == ["space_a", "space_b", "space_c"]
    assert report["consensus"]["n_works_common"] == 6
    assert report["consensus"]["n_consensus_edges"] == 6
    assert all(node["title"].startswith("作品") for node in report["nodes"])

    with pytest.raises(NebulaError):
        bcg.load_space(catalog, config, "missing_representation")

    # 未 ready 的表征拒绝加载
    not_ready = _register_representation(
        catalog,
        config,
        name="space_not_ready",
        version="consensus-test-v1",
        matrix=_line_matrix(ORDERS[0], 2),
        work_ids=work_ids,
        status="building",
    )
    assert not_ready > 0
    with pytest.raises(NebulaError, match="ready"):
        bcg.load_space(catalog, config, "space_not_ready")


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as excinfo:
        bcg.main(["--help"])
    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    for flag in (
        "--config",
        "--representations",
        "--k",
        "--out",
        "--csv",
        "--min-spaces",
        "--fusion",
    ):
        assert flag in output
    parser = bcg.build_parser()
    defaults = parser.parse_args([])
    assert defaults.representations is None
    assert defaults.k == bcg.DEFAULT_K == 5
    assert defaults.min_spaces == bcg.DEFAULT_MIN_SPACES == 2
    assert defaults.fusion == bcg.DEFAULT_FUSION == "both"
    assert bcg.DEFAULT_REPRESENTATIONS == (
        "full_mean_v1",
        "dense_visual_clip_v2",
        "dense_visual_dino_v2",
    )


def test_cli_end_to_end(config, catalog, tmp_path, capsys):
    work_ids = _prepare_works(catalog)
    versions = ["consensus-cli-v1", "consensus-cli-v2", "consensus-cli-v3"]
    for index, _space in enumerate(_line_spaces()):
        _register_representation(
            catalog,
            config,
            name=f"consensus_cli_{index}",
            version=versions[index],
            matrix=_line_matrix(ORDERS[index], DIMS[index]),
            work_ids=work_ids,
        )
    json_path = tmp_path / "cli" / "neighbor_consensus.json"
    csv_path = tmp_path / "cli" / "neighbor_consensus.csv"
    code = bcg.main(
        [
            "--config",
            str(config.path),
            "--representations",
            "consensus_cli_0",
            "consensus_cli_1",
            "consensus_cli_2",
            "--k",
            "2",
            "--min-spaces",
            "2",
            "--fusion",
            "both",
            "--out",
            str(json_path),
            "--csv",
            str(csv_path),
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "共识边" in output
    assert "consensus_cli_0" in output
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["consensus"]["n_consensus_edges"] == 6
    assert payload["consensus"]["n_full_consensus_edges"] == 3
    assert payload["params"]["representations"] == [
        "consensus_cli_0",
        "consensus_cli_1",
        "consensus_cli_2",
    ]
    assert "fusion" in payload and payload["fusion"]["rank_average"] is not None
    assert csv_path.is_file()
    assert (tmp_path / "cli" / "neighbor_consensus_nodes.csv").is_file()


def test_cli_missing_default_representations_returns_error(config, catalog, tmp_path, capsys):
    code = bcg.main(
        ["--config", str(config.path), "--out", str(tmp_path / "missing.json")]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert "共识图构建失败" in captured.err
    assert "full_mean_v1" in captured.err

def test_report_is_consumed_by_appdata(config, tmp_path):
    """与 t2 界面/近邻层约定：logs/neighbor_consensus.json 可被 appdata 解析。"""

    from nebula import appdata

    report = bcg.build_consensus(_line_spaces(), k=2, min_spaces=2, fusion="none")
    report["generated_at"] = "2026-10-26T00:00:00Z"
    log_path = config.paths.logs / "neighbor_consensus.json"
    bcg.write_reports(
        report,
        json_path=log_path,
        csv_path=tmp_path / "neighbor_consensus.csv",
    )
    consensus = appdata.load_neighbor_consensus(config)
    assert consensus is not None
    assert consensus["n_edges"] == len(report["edges"])
    assert consensus["meta"]["tool"] == bcg.TOOL_NAME
    filtered = appdata.consensus_edges_for_visible(
        consensus, [1, 2, 3, 4, 5, 6], min_count=2
    )
    assert len(filtered["edges"]) == report["consensus"]["n_consensus_edges"]
    rows = appdata.consensus_neighbors(consensus, 3, min_count=2)
    assert rows
    assert all(row["consensus_count"] >= 2 for row in rows)

def test_cli_default_output_paths(config, catalog, capsys):
    work_ids = _prepare_works(catalog)
    names = ["default_path_a", "default_path_b", "default_path_c"]
    for index, _space in enumerate(_line_spaces()):
        _register_representation(
            catalog,
            config,
            name=names[index],
            version=f"consensus-default-path-{index}",
            matrix=_line_matrix(ORDERS[index], DIMS[index]),
            work_ids=work_ids,
        )
    code = bcg.main(
        [
            "--config",
            str(config.path),
            "--representations",
            *names,
            "--k",
            "2",
            "--fusion",
            "none",
        ]
    )
    assert code == 0
    assert (config.paths.logs / "neighbor_consensus.json").is_file()
    assert (config.paths.logs / "neighbor_consensus.csv").is_file()
    assert (config.paths.logs / "neighbor_consensus_nodes.csv").is_file()

