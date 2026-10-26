"""nebula.discovery / tools.explore_relations 的测试（v0.4.5 意外关系发现）。

覆盖：
- 排名百分位 p = 1-(r-1)/(N-2)、无向 bar_p 的手算值、并列 work_id 稳定排序；
- 模式 A/B 的候选并集、U 排序、每空间 rank/p/cos 字段；
- 模式 C 的 common（仅含 PE-AV 的 >=2 mutual）/visual_vs_peav/单模型类别与 p 差分数；
- 行顺序打乱不变、标题不进入分数、缺失空间 skipped、N<3/零向量报错；
- JSON/CSV schema、DB 只读（不写地图/表征）、CLI --help 与临时库端到端；
- 全部合成数据，不加载模型/不联网；真实 110 部运行由 CLI 手动执行并另行报告。
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from nebula import discovery as disc
from nebula import representations as R
from nebula.errors import NebulaError

#: 非均匀角度间隔：保证线型空间里所有距离互异（Top-2 邻居可手算）
GAPS: tuple[float, ...] = (0.50, 0.54, 0.58, 0.62, 0.66)
ANGLES: tuple[float, ...] = tuple(
    float(np.cumsum((0.0, *GAPS))[index]) for index in range(len(GAPS) + 1)
)
ORDERS: dict[str, tuple[int, ...]] = {
    "full_mean_joint": (1, 2, 3, 4, 5, 6),
    "full_mean_audio": (2, 3, 4, 5, 6, 1),
    "full_mean_video": (3, 4, 5, 6, 1, 2),
    "temporal_joint": (4, 5, 6, 1, 2, 3),
    "clip_joint": (5, 6, 1, 2, 3, 4),
    "dino_joint": (6, 1, 2, 3, 4, 5),
}
DIMS: dict[str, int] = {key: 2 + index for index, key in enumerate(ORDERS)}


# --------------------------------------------------------------------------- 合成线型空间
def _line_matrix(order: tuple[int, ...], dim: int) -> np.ndarray:
    count = len(order)
    unit = np.stack([np.cos(np.asarray(ANGLES)), np.sin(np.asarray(ANGLES))], axis=1)
    matrix = np.zeros((count, dim), dtype=np.float64)
    for position, work_id in enumerate(order):
        matrix[work_id - 1, :2] = unit[position]
    return matrix


def _line_space(key: str, order: tuple[int, ...] | None = None, dim: int | None = None):
    actual_order = ORDERS[key] if order is None else order
    actual_dim = DIMS[key] if dim is None else dim
    return disc.SpaceVectors.from_matrix(
        _line_matrix(actual_order, actual_dim),
        key=key,
        name=key,
        label=key,
        kind="joint",
        ids=tuple(range(1, len(actual_order) + 1)),
    )


def _line_spaces() -> dict[str, disc.SpaceVectors]:
    return {key: _line_space(key) for key in ORDERS}


def _position(order: tuple[int, ...], work_id: int) -> int:
    return order.index(work_id)


def _line_cos(order: tuple[int, ...], first: int, second: int) -> float:
    delta = abs(ANGLES[_position(order, first)] - ANGLES[_position(order, second)])
    return float(np.cos(delta))


def _line_ranks(order: tuple[int, ...], work_id: int) -> dict[int, int]:
    others = [other for other in range(1, len(order) + 1) if other != work_id]
    others.sort(key=lambda other: (-_line_cos(order, work_id, other), other))
    return {other: rank for rank, other in enumerate(others, start=1)}


def _line_percentile(order: tuple[int, ...], work_id: int, other: int) -> float:
    count = len(order)
    rank = _line_ranks(order, work_id)[other]
    return 1.0 - (rank - 1) / float(count - 2)


def _line_bar(order: tuple[int, ...], first: int, second: int) -> float:
    return (
        _line_percentile(order, first, second)
        + _line_percentile(order, second, first)
    ) / 2.0


def _expected_candidates(order: tuple[int, ...], k: int = 2) -> set[tuple[int, int]]:
    """线型空间 Top-k 有向边的无序并集（k=2 时按角度距离手算）。"""

    count = len(order)
    pairs: set[tuple[int, int]] = set()
    for work_id in order:
        position = _position(order, work_id)
        if position == 0:
            neighbors = [order[1], order[2]]
        elif position == count - 1:
            neighbors = [order[count - 2], order[count - 3]]
        else:
            neighbors = [order[position - 1], order[position + 1]]
        for neighbor in neighbors[:k]:
            pairs.add((min(work_id, neighbor), max(work_id, neighbor)))
    return pairs


def _expected_mutual_top2(order: tuple[int, ...]) -> set[tuple[int, int]]:
    directed = set()
    count = len(order)
    for work_id in order:
        position = _position(order, work_id)
        if position == 0:
            neighbors = [order[1], order[2]]
        elif position == count - 1:
            neighbors = [order[count - 2], order[count - 3]]
        else:
            neighbors = [order[position - 1], order[position + 1]]
        directed.update((work_id, neighbor) for neighbor in neighbors)
    mutual: set[tuple[int, int]] = set()
    for first, second in directed:
        if (second, first) in directed:
            mutual.add((min(first, second), max(first, second)))
    return mutual


# --------------------------------------------------------------------------- 手算百分位
def test_rank_percentile_hand_computed():
    similarity = np.asarray(
        [
            [1.00, 0.90, 0.80, 0.70, 0.60, 0.50],
            [0.90, 1.00, 0.85, 0.75, 0.65, 0.55],
            [0.80, 0.85, 1.00, 0.95, 0.70, 0.60],
            [0.70, 0.75, 0.95, 1.00, 0.80, 0.65],
            [0.60, 0.65, 0.70, 0.80, 1.00, 0.90],
            [0.50, 0.55, 0.60, 0.65, 0.90, 1.00],
        ]
    )
    ids = tuple(range(1, 7))
    ranks = disc.neighbor_rank_table(similarity, ids=ids)
    assert ranks[0, 1] == 1 and ranks[0, 2] == 2 and ranks[0, 5] == 5
    assert ranks[2, 0] == 3  # row2: 3(.95) > 1(.85) > 0(.8)
    assert ranks[0, 0] == 0

    p = disc.percentile_table(ranks)
    assert p[0, 1] == pytest.approx(1.0)
    assert p[0, 2] == pytest.approx(0.75)
    assert p[0, 3] == pytest.approx(0.5)
    assert p[0, 4] == pytest.approx(0.25)
    assert p[0, 5] == pytest.approx(0.0)
    assert p[0, 0] == 0.0
    # bar_p(0,2) = (p(0→2) + p(2→0)) / 2 = (0.75 + 0.5) / 2 = 0.625
    assert disc.bar_percentile(p, 0, 2) == pytest.approx(0.625)
    assert disc.bar_percentile(p, 0, 1) == pytest.approx(1.0)

    with pytest.raises(NebulaError):
        disc.neighbor_rank_table(np.ones((2, 3)), ids=ids)
    with pytest.raises(NebulaError):
        disc.percentile_table(np.zeros((2, 2)))


def test_ties_are_stable_by_work_id():
    similarity = np.asarray(
        [
            [1.0, 0.5, 0.5, 0.5],
            [0.5, 1.0, 0.9, 0.8],
            [0.5, 0.9, 1.0, 0.7],
            [0.5, 0.8, 0.7, 1.0],
        ]
    )
    ranks = disc.neighbor_rank_table(similarity, ids=(1, 2, 3, 4))
    # row0 三个并列 0.5，按 work_id 升序为 1,2,3
    assert [ranks[0, index] for index in (1, 2, 3)] == [1, 2, 3]
    with pytest.raises(NebulaError):
        disc.neighbor_rank_table(similarity, ids=(2, 1, 3, 4))


def test_space_vectors_alignment_and_shuffle_invariance():
    spaces = _line_spaces()
    matrix = spaces["full_mean_joint"].matrix
    ids = list(spaces["full_mean_joint"].ids)
    shuffled_ids = ids[::-1]
    shuffled_matrix = matrix[::-1]
    rebuilt = disc.SpaceVectors.from_matrix(
        shuffled_matrix,
        key="full_mean_joint",
        name="full_mean_joint",
        ids=shuffled_ids,
    )
    np.testing.assert_allclose(rebuilt.matrix, matrix)
    assert rebuilt.ids == tuple(ids)

    first = disc.discover_from_spaces(
        spaces, top_n=10, k=2, generated_at="2026-10-26T00:00:00Z"
    )
    reordered = {
        key: disc.SpaceVectors.from_matrix(
            spaces[key].matrix[::-1],
            key=key,
            name=key,
            label=key,
            ids=list(spaces[key].ids)[::-1],
        )
        for key in spaces
    }
    second = disc.discover_from_spaces(
        reordered, top_n=10, k=2, generated_at="2026-10-26T00:00:00Z"
    )
    assert first == second


# --------------------------------------------------------------------------- 模式 A/B
def _expected_mode_a_rows() -> list[tuple[float, int, int]]:
    joint = ORDERS["full_mean_joint"]
    audio = ORDERS["full_mean_audio"]
    video = ORDERS["full_mean_video"]
    rows: list[tuple[float, int, int]] = []
    for work_a, work_b in _expected_candidates(joint, k=2):
        score = _line_bar(joint, work_a, work_b) - (
            _line_bar(audio, work_a, work_b) + _line_bar(video, work_a, work_b)
        ) / 2.0
        rows.append((score, work_a, work_b))
    rows.sort(key=lambda item: (-item[0], item[1], item[2]))
    return rows


def test_mode_a_candidates_ordering_and_fields():
    spaces = _line_spaces()
    report = disc.discover_from_spaces(
        spaces, top_n=50, k=2, generated_at="2026-10-26T00:00:00Z"
    )
    mode_a = report["mode_a"]
    assert mode_a["skipped"] is False
    observed = {
        (item["work_id_a"], item["work_id_b"]) for item in mode_a["top"]
    }
    assert observed == _expected_candidates(ORDERS["full_mean_joint"], k=2)

    expected = _expected_mode_a_rows()
    top = mode_a["top"]
    assert len(top) == len(expected)
    for rank, (expected_score, first, second) in enumerate(expected[:10]):
        row = top[rank]
        assert (row["work_id_a"], row["work_id_b"]) == (first, second)
        assert row["score"] == pytest.approx(expected_score, abs=1e-6)
        assert row["work_id_a"] != row["work_id_b"]
        for group in ("joint", "audio", "video"):
            metric = row[group]
            assert 1 <= metric["rank_a"] <= 5
            assert 1 <= metric["rank_b"] <= 5
            assert 0.0 <= metric["p_a"] <= 1.0
            assert 0.0 <= metric["bar_p"] <= 1.0
            assert -1.0 <= metric["cos"] <= 1.0
        assert isinstance(row["joint_candidate_a_to_b"], bool)
        assert isinstance(row["joint_candidate_b_to_a"], bool)
    scores = [item["score"] for item in top]
    assert scores == sorted(scores, reverse=True)


def test_mode_b_candidates_ordering_and_fields():
    spaces = _line_spaces()
    report = disc.discover_from_spaces(
        spaces, top_n=50, k=2, generated_at="2026-10-26T00:00:00Z"
    )
    mode_b = report["mode_b"]
    assert mode_b["skipped"] is False
    observed = {
        (item["work_id_a"], item["work_id_b"]) for item in mode_b["top"]
    }
    assert observed == _expected_candidates(ORDERS["temporal_joint"], k=2)

    temporal = ORDERS["temporal_joint"]
    full_mean = ORDERS["full_mean_joint"]
    expected: list[tuple[float, int, int]] = []
    for work_a, work_b in _expected_candidates(temporal, k=2):
        score = _line_bar(temporal, work_a, work_b) - _line_bar(
            full_mean, work_a, work_b
        )
        expected.append((score, work_a, work_b))
    expected.sort(key=lambda item: (-item[0], item[1], item[2]))
    for rank, (expected_score, first, second) in enumerate(expected[:10]):
        row = mode_b["top"][rank]
        assert (row["work_id_a"], row["work_id_b"]) == (first, second)
        assert row["score"] == pytest.approx(expected_score, abs=1e-6)
        assert row["temporal"]["bar_p"] - row["full_mean"]["bar_p"] == pytest.approx(
            expected_score, abs=1e-6
        )
    assert all(
        item["work_id_a"] != item["work_id_b"] for item in mode_b["top"]
    )


# --------------------------------------------------------------------------- 模式 C
def test_mode_c_categories_and_p_scores():
    spaces = _line_spaces()
    report = disc.discover_from_spaces(
        spaces, top_n=50, k=2, generated_at="2026-10-26T00:00:00Z"
    )
    mode_c = report["mode_c"]
    assert mode_c["skipped"] is False

    peav_mutual = _expected_mutual_top2(ORDERS["full_mean_joint"])
    clip_mutual = _expected_mutual_top2(ORDERS["clip_joint"])
    dino_mutual = _expected_mutual_top2(ORDERS["dino_joint"])
    all_pairs = peav_mutual | clip_mutual | dino_mutual

    expected_counts: dict[tuple[int, int], int] = {}
    for pair in all_pairs:
        count = sum(
            1
            for mutual in (peav_mutual, clip_mutual, dino_mutual)
            if pair in mutual
        )
        expected_counts[pair] = count
    common_pairs = {
        pair
        for pair, count in expected_counts.items()
        if count >= 2 and pair in peav_mutual
    }
    visual_pairs = {
        pair
        for pair in clip_mutual & dino_mutual
        if pair not in peav_mutual
    }
    peav_only = {
        pair for pair, count in expected_counts.items() if count == 1 and pair in peav_mutual
    }
    clip_only = {
        pair for pair, count in expected_counts.items() if count == 1 and pair in clip_mutual
    }
    dino_only = {
        pair for pair, count in expected_counts.items() if count == 1 and pair in dino_mutual
    }

    observed_common = {
        (row["work_id_a"], row["work_id_b"]) for row in mode_c["common"]
    }
    assert observed_common == common_pairs
    assert {
        (row["work_id_a"], row["work_id_b"]) for row in mode_c["peav_only"]
    } == peav_only
    assert {
        (row["work_id_a"], row["work_id_b"]) for row in mode_c["clip_only"]
    } == clip_only
    assert {
        (row["work_id_a"], row["work_id_b"]) for row in mode_c["dino_only"]
    } == dino_only
    assert {
        (row["work_id_a"], row["work_id_b"]) for row in mode_c["visual_vs_peav"]
    } == visual_pairs
    assert len(common_pairs) == 5
    assert len(visual_pairs) == 1
    assert not (common_pairs & visual_pairs)
    counts = mode_c["counts"]
    assert counts["common_ge2"] + counts["visual_vs_peav"] + counts["peav_only"] + counts["clip_only"] + counts["dino_only"] == mode_c["n_candidate_pairs"]
    assert mode_c["common_summary"]["n_common_ge2"] == len(common_pairs)
    assert counts["visual_vs_peav"] == len(visual_pairs)

    levels = {"all3": 0, "peav_clip": 0, "peav_dino": 0}
    for pair, count in expected_counts.items():
        if pair not in common_pairs:
            continue
        if count >= 3:
            levels["all3"] += 1
        elif pair in clip_mutual:
            levels["peav_clip"] += 1
        else:
            levels["peav_dino"] += 1
    assert mode_c["common_summary"]["levels"] == levels

    for row in mode_c["common"]:
        mutual_labels = list(row["mutual_spaces"])
        expected_score = float(
            np.mean(
                [
                    row[label]["bar_p"]
                    for label in mutual_labels
                ]
            )
        )
        assert row["score"] == pytest.approx(expected_score, abs=1e-6)
        assert row["mutual_count"] == len(mutual_labels)
        if row["mutual_count"] >= 3:
            assert row["common_level"] == "all3"
        elif "clip" in mutual_labels:
            assert row["common_level"] == "peav_clip"
        else:
            assert row["common_level"] == "peav_dino"
    for row in mode_c["visual_vs_peav"]:
        expected_score = (
            row["clip"]["bar_p"] + row["dino"]["bar_p"]
        ) / 2.0 - row["peav"]["bar_p"]
        assert row["score"] == pytest.approx(expected_score, abs=1e-6)
        assert row["common_level"] is None
        assert row["peav"]["mutual"] is False
        assert row["clip"]["mutual"] is True
        assert row["dino"]["mutual"] is True


# --------------------------------------------------------------------------- 缺失 / 边界
def test_missing_space_skipped_and_mode_filter():
    spaces = _line_spaces()
    spaces["clip_joint"] = None
    report = disc.discover_from_spaces(
        spaces, top_n=10, k=2, generated_at="2026-10-26T00:00:00Z"
    )
    assert report["mode_c"]["skipped"] is True
    assert "clip_joint" in report["mode_c"]["reason"]
    assert report["mode_a"]["skipped"] is False
    assert report["mode_b"]["skipped"] is False
    assert "clip_joint" not in report["meta"]["spaces_loaded"]

    with pytest.raises(NebulaError):
        disc.discover_from_spaces(spaces, top_n=10, k=2, mode="c")

    only_a = disc.discover_from_spaces(
        _line_spaces(), top_n=5, k=2, mode="a", generated_at="t"
    )
    assert only_a["mode_b"] is None and only_a["mode_c"] is None
    assert only_a["mode_a"]["skipped"] is False


def test_errors_for_min_works_zero_vectors_and_bad_params():
    with pytest.raises(NebulaError):
        disc.SpaceVectors.from_matrix(
            np.zeros((3, 4)), key="bad", name="bad", ids=[1, 2, 3]
        )
    with pytest.raises(NebulaError):
        disc.SpaceVectors.from_matrix(
            np.ones((3, 4)), key="bad", name="bad", ids=[1, 2]
        )
    spaces = _line_spaces()
    small = {
        key: disc.SpaceVectors.from_matrix(
            space.matrix[:2], key=key, name=key, ids=space.ids[:2]
        )
        for key, space in spaces.items()
    }
    with pytest.raises(NebulaError):
        disc.discover_from_spaces(small, top_n=5, k=2)
    single = {
        key: disc.SpaceVectors.from_matrix(
            space.matrix[:1], key=key, name=key, ids=space.ids[:1]
        )
        for key, space in spaces.items()
    }
    with pytest.raises(NebulaError):
        disc.discover_from_spaces(single, top_n=5, k=2)
    with pytest.raises(NebulaError):
        disc.discover_from_spaces(spaces, top_n=0, k=2)
    with pytest.raises(NebulaError):
        disc.discover_from_spaces(spaces, top_n=5, k=0)
    with pytest.raises(NebulaError):
        disc.neighbor_rank_table(np.ones((3, 4)), ids=(1, 2, 3))


# --------------------------------------------------------------------------- 元数据 / 输出
def _strip_titles(report: dict) -> dict:
    cloned = json.loads(json.dumps(report, ensure_ascii=False))
    for mode_key in ("mode_a", "mode_b"):
        mode = cloned.get(mode_key) or {}
        for row in mode.get("top", ()):
            row.pop("title_a", None)
            row.pop("title_b", None)
    mode_c = cloned.get("mode_c") or {}
    for category in (
        "common",
        "visual_vs_peav",
        "peav_only",
        "clip_only",
        "dino_only",
    ):
        for row in mode_c.get(category, ()):
            row.pop("title_a", None)
            row.pop("title_b", None)
    return cloned


def test_titles_do_not_change_scores():
    spaces = _line_spaces()
    first = disc.discover_from_spaces(
        spaces,
        top_n=10,
        k=2,
        titles={work_id: "标题A" for work_id in range(1, 7)},
        generated_at="t",
    )
    second = disc.discover_from_spaces(
        spaces,
        top_n=10,
        k=2,
        titles={work_id: "完全不同" for work_id in range(1, 7)},
        generated_at="t",
    )
    assert _strip_titles(first) == _strip_titles(second)
    assert first["mode_a"]["top"][0]["title_a"] == "标题A"
    assert second["mode_a"]["top"][0]["title_a"] == "完全不同"


def test_write_reports_json_and_csv(tmp_path):
    report = disc.discover_from_spaces(
        _line_spaces(), top_n=5, k=2, generated_at="2026-10-26T00:00:00Z"
    )
    json_path = tmp_path / "out" / "unexpected_relations.json"
    csv_path = tmp_path / "out" / "unexpected_relations.csv"
    written_json, written_csv = disc.write_reports(
        report, json_path=json_path, csv_path=csv_path
    )
    assert written_json == json_path and written_csv == csv_path
    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    assert not list((tmp_path / "out").glob("*.tmp"))

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == disc.CSV_COLUMNS
    assert rows
    modes = {row["mode"] for row in rows}
    assert "mode_a" in modes and "mode_b" in modes
    assert any(mode.startswith("mode_c/") for mode in modes)
    for row in rows:
        assert int(row["work_id_a"]) != int(row["work_id_b"])
        if row["mode"] == "mode_a":
            assert float(row["joint_bar_p"]) != ""
            assert float(row["joint_cos"]) != ""
        if row["mode"].startswith("mode_c/"):
            assert row["mutual_count"] != ""
            assert row["mutual_spaces"] != ""


# --------------------------------------------------------------------------- DB / CLI
def _register_sequence(catalog) -> None:
    catalog.register_embedding_version(
        "discovery-test-seq",
        model_checkpoint="test-only:hash",
        model_revision=None,
        frames=16,
        clip_seconds=8.0,
        clips_per_work=0,
        audio_sample_rate=48000,
        params={"test": "discovery"},
    )
    catalog.register_visual_version(
        "discovery-visual-test-v1",
        model_id="test/clip",
        revision="test-revision",
        preprocessing="letterbox",
        fps=8.0,
        size=224,
        params={"test": "discovery"},
    )


def _install_representation(
    catalog,
    config,
    *,
    name: str,
    matrices: dict[str, np.ndarray],
    ids: list[int],
    kind: str,
    source_kind: str = "pe_av",
    sequence_version: str = "discovery-test-seq",
) -> int:
    rep_id = catalog.register_representation(
        name,
        sequence_version=sequence_version,
        kind=kind,
        source_kind=source_kind,
        params={"test": "discovery", "name": name},
    )
    rep = catalog.get_representation(rep_id)
    assert rep is not None
    for matrix_kind, matrix in matrices.items():
        for row, work_id in enumerate(ids):
            R.save_representation_vectors(
                catalog,
                config,
                rep,
                int(work_id),
                {matrix_kind: np.asarray(matrix[row], dtype=np.float32)},
            )
    first = next(iter(matrices.values()))
    catalog.set_representation_status(rep_id, "ready", dim=int(first.shape[1]))
    return rep_id


def _setup_db(catalog, config) -> dict:
    _register_sequence(catalog)
    spaces = _line_spaces()
    ids: list[int] = []
    for index in range(1, 7):
        work_id, _ = catalog.upsert_local_work(
            f"discovery/w{index}.mp4", title=f"作品{index}"
        )
        catalog.begin_attempt(work_id, "encoding")
        catalog.complete(work_id)
        ids.append(int(work_id))
    _install_representation(
        catalog,
        config,
        name=disc.PEAV_NAME,
        matrices={
            "joint": spaces["full_mean_joint"].matrix,
            "audio": spaces["full_mean_audio"].matrix,
            "video": spaces["full_mean_video"].matrix,
        },
        ids=ids,
        kind="full_mean",
    )
    _install_representation(
        catalog,
        config,
        name=disc.TEMPORAL_NAME,
        matrices={"joint": spaces["temporal_joint"].matrix},
        ids=ids,
        kind="temporal_stats",
    )
    _install_representation(
        catalog,
        config,
        name=disc.CLIP_NAME,
        matrices={"joint": spaces["clip_joint"].matrix},
        ids=ids,
        kind="visual",
        source_kind="visual",
        sequence_version="discovery-visual-test-v1",
    )
    _install_representation(
        catalog,
        config,
        name=disc.DINO_NAME,
        matrices={"joint": spaces["dino_joint"].matrix},
        ids=ids,
        kind="visual",
        source_kind="visual",
        sequence_version="discovery-visual-test-v1",
    )
    return {"ids": ids, "spaces": spaces}


def test_discover_reads_db_without_writing(config, catalog):
    setup = _setup_db(catalog, config)
    maps_before = catalog.list_maps()
    representations_before = catalog.list_representations()
    works_before = catalog.list_works()

    report = disc.discover(catalog, config, top_n=10, k=2, mode="all")
    assert report["meta"]["tool"] == disc.TOOL_NAME
    assert len(report["spaces"]) == 6
    loaded_keys = {space["key"] for space in report["spaces"]}
    assert loaded_keys == set(disc.SPACE_KEYS)
    assert report["mode_a"]["skipped"] is False
    assert report["mode_b"]["skipped"] is False
    assert report["mode_c"]["skipped"] is False
    first = report["mode_a"]["top"][0]
    assert first["title_a"] == f"作品{first['work_id_a'] - setup['ids'][0] + 1}"
    assert report["meta"]["generated_at"]

    assert catalog.list_maps() == maps_before
    assert catalog.list_representations() == representations_before
    assert catalog.list_works() == works_before


def test_cli_help(capsys):
    from tools import explore_relations

    with pytest.raises(SystemExit) as excinfo:
        explore_relations.main(["--help"])
    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    for flag in ("--config", "--top", "--k", "--out", "--csv", "--mode"):
        assert flag in output


def test_cli_end_to_end(config, catalog, tmp_path, capsys):
    from tools import explore_relations

    _setup_db(catalog, config)
    json_path = tmp_path / "cli" / "unexpected_relations.json"
    csv_path = tmp_path / "cli" / "unexpected_relations.csv"
    code = explore_relations.main(
        [
            "--config",
            str(config.path),
            "--top",
            "5",
            "--k",
            "2",
            "--mode",
            "all",
            "--out",
            str(json_path),
            "--csv",
            str(csv_path),
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "高维意外关系发现" in output
    assert "Mode A" in output and "Mode C" in output
    assert json_path.is_file() and csv_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["meta"]["params"]["top_n"] == 5
    assert payload["meta"]["params"]["k"] == 2
    assert payload["mode_c"]["counts"]["common_ge2"] == 5
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
