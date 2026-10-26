"""tools/embedding_diagnostics.py 的测试。

覆盖：
- 高维余弦距离 / 低维欧氏距离 / Top-k 邻域 / 邻域保持率 / Pearson+Spearman 的
  已知数值与边界（并列、有效 k、非法输入）；
- PCA 2D/3D 形状与确定性；
- UMAP 2D/3D × n_neighbors=15/30/50 的 schema、参数接线与小样本钳制
  （通过注入轻量 UMAP 替身，避免 numba 首次编译 ~20-30s；真实 UMAP 冒烟测试由
  NEBULA_REAL_UMAP_TESTS=1 环境变量显式开启）；
- 合成簇结构矩阵上的指标范围、报告 schema、确定性；
- JSON/CSV 写入 schema、Plotly HTML（可选输出）；
- CLI --help 与临时库端到端（--representation/--k/--umap-neighbors/--components/
  --out/--csv/--html-dir）。
"""

from __future__ import annotations

import csv
import json
import os
import sys
import types

import numpy as np
import pytest

from nebula import representations as R
from nebula.errors import NebulaError
from tools import embedding_diagnostics as diag


# --------------------------------------------------------------------------- 测试工具
def _synthetic_matrix(
    *,
    n_per_cluster: int = 15,
    n_clusters: int = 4,
    dim: int = 8,
    noise: float = 0.3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """已知簇结构 + 高斯噪声的合成矩阵（行已 L2 归一化，标签为簇号）。"""

    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for cluster in range(n_clusters):
        for _ in range(n_per_cluster):
            rows.append(centers[cluster] + noise * rng.normal(size=dim))
            labels.append(cluster)
    matrix = np.asarray(rows, dtype=np.float64)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix, np.asarray(labels)


def _space(
    matrix: np.ndarray,
    *,
    name: str = "synthetic",
    labels: np.ndarray | None = None,
) -> diag.SpaceMatrix:
    work_ids = None
    if labels is not None:
        work_ids = [1000 + int(index) for index in range(len(labels))]
    return diag.SpaceMatrix.from_matrix(
        matrix, name=name, label=f"{name}-label", work_ids=work_ids
    )


class _SvdUmap:
    """轻量确定性 UMAP 替身：用 SVD 前缀坐标代替投影，并记录构造参数。"""

    calls: list[dict] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = dict(kwargs)
        type(self).calls.append(self.kwargs)

    def fit_transform(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        centered = values - values.mean(axis=0, keepdims=True)
        left, singular, _right = np.linalg.svd(centered, full_matrices=False)
        components = int(self.kwargs["n_components"])
        coordinates = left[:, :components] * singular[:components]
        if coordinates.shape[1] < components:
            coordinates = np.pad(
                coordinates, ((0, 0), (0, components - coordinates.shape[1]))
            )
        return np.ascontiguousarray(coordinates)


@pytest.fixture
def fake_umap(monkeypatch):
    calls: list[dict] = []

    class FakeUmap(_SvdUmap):
        pass

    FakeUmap.calls = calls
    monkeypatch.setitem(sys.modules, "umap", types.SimpleNamespace(UMAP=FakeUmap))
    return calls


def _make_analysis(
    *,
    matrix: np.ndarray,
    k: int = 3,
    neighbors: tuple[int, ...] = (3, 5),
    components: tuple[int, ...] = (2,),
) -> diag.RepresentationDiagnostics:
    space = _space(matrix)
    return diag.analyze_space(
        space, k=k, umap_neighbors=neighbors, components=components
    )


def _diagnostics_result(
    analysis: diag.RepresentationDiagnostics, *, k: int = 3
) -> diag.DiagnosticsResult:
    report = {
        "tool": diag.TOOL_NAME,
        "version": diag.TOOL_VERSION,
        "generated_at": "2026-10-26T00:00:00Z",
        "params": {
            "kind": "joint",
            "k": k,
            "umap_neighbors": [3, 5],
            "components": [2],
            "representations": [analysis.space.name],
            "pca_random_state": diag.PCA_RANDOM_STATE,
            "umap": {},
        },
        "representations": [analysis.report],
    }
    return diag.DiagnosticsResult(report=report, analyses=(analysis,))


def _install_representation(
    catalog,
    config,
    *,
    name: str,
    version: str,
    work_count: int = 12,
    dim: int = 4,
    seed: int = 9,
) -> int:
    """在临时 DB 中登记一个 ready 表征（只用于测试，不触碰真实 data/）。"""

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
    representation_id = catalog.register_representation(
        name,
        sequence_version=version,
        kind="full_mean",
        params={"test": "embedding-diagnostics", "name": name},
    )
    rep = catalog.get_representation(representation_id)
    assert rep is not None
    rng = np.random.default_rng(seed)
    for index in range(work_count):
        work_id, _ = catalog.upsert_local_work(
            f"diag/{name}/{index}.mp4", title=f"作品-{name}-{index}"
        )
        R.save_representation_vectors(
            catalog,
            config,
            rep,
            work_id,
            {"joint": rng.normal(size=dim).astype(np.float32)},
        )
    catalog.set_representation_status(representation_id, "ready", dim=dim)
    return representation_id


# --------------------------------------------------------------------------- 距离与邻域
def test_normalize_vectors_and_cosine_distance_matrix():
    raw = np.array([[3.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    normalized = diag.normalize_vectors(raw)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), np.ones(3))
    with pytest.raises(NebulaError):
        diag.normalize_vectors(np.zeros((2, 3)))
    with pytest.raises(NebulaError):
        diag.normalize_vectors(np.ones(3))
    with pytest.raises(NebulaError):
        diag.normalize_vectors(np.array([[1.0, np.nan]]))

    distance = diag.cosine_distance_matrix(raw)
    assert distance.shape == (3, 3)
    np.testing.assert_allclose(np.diag(distance), np.zeros(3))
    np.testing.assert_allclose(distance, distance.T)
    assert np.all(distance >= 0.0) and np.all(distance <= 2.0 + 1e-9)
    np.testing.assert_allclose(distance[0, 0], 0.0)
    np.testing.assert_allclose(distance[0, 1], 1.0)  # 正交
    np.testing.assert_allclose(distance[0, 2], 1.0 - 3.0 / np.sqrt(18.0))


def test_euclidean_distance_matrix_known_values():
    points = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
    distance = diag.euclidean_distance_matrix(points)
    assert distance.shape == (3, 3)
    np.testing.assert_allclose(np.diag(distance), np.zeros(3))
    np.testing.assert_allclose(distance, distance.T)
    np.testing.assert_allclose(distance[0, 1], 5.0)
    np.testing.assert_allclose(distance[0, 2], 0.0)
    np.testing.assert_allclose(distance[1, 2], 5.0)
    with pytest.raises(NebulaError):
        diag.euclidean_distance_matrix(np.array([[1.0, np.inf]]))


def test_topk_neighbors_known_and_effective_k():
    distance = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    np.testing.assert_array_equal(
        diag.topk_neighbors(distance, 1), np.array([[1], [0], [0]])
    )
    np.testing.assert_array_equal(
        diag.topk_neighbors(distance, 10),
        np.array([[1, 2], [0, 2], [0, 1]]),
    )
    # 并列时按原索引稳定排序（不含自身）
    tie = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 2.0], [1.0, 2.0, 0.0]])
    np.testing.assert_array_equal(
        diag.topk_neighbors(tie, 1), np.array([[1], [0], [0]])
    )
    with pytest.raises(NebulaError):
        diag.topk_neighbors(np.ones((2, 3)), 1)
    with pytest.raises(NebulaError):
        diag.topk_neighbors(distance, 0)


def test_neighborhood_metrics_known_values():
    reference = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    candidate = np.array([[1, 2, 4], [0, 2, 4], [0, 1, 4], [0, 1, 2]])
    metrics = diag.neighborhood_metrics(reference, candidate)
    assert metrics["effective_k"] == 3
    assert metrics["n_works"] == 4
    assert metrics["topk_preservation"] == pytest.approx(0.75, abs=1e-6)
    assert metrics["top1_preservation"] == pytest.approx(1.0)
    assert metrics["mean_overlap"] == pytest.approx(2.25)
    assert metrics["stable_ratio"] == pytest.approx(0.25)
    assert 0.0 <= metrics["topk_preservation"] <= 1.0
    assert 0.0 <= metrics["top1_preservation"] <= 1.0
    with pytest.raises(NebulaError):
        diag.neighborhood_metrics(reference, candidate[:2])


def test_pair_distance_correlation_known_values():
    high = np.array(
        [[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 4.0, 5.0], [2.0, 4.0, 0.0, 6.0], [3.0, 5.0, 6.0, 0.0]]
    )
    low = 2.0 * high
    metrics = diag.pair_distance_correlation(high, low)
    assert metrics["n_pairs"] == 6
    assert metrics["pearson"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["spearman"] == pytest.approx(1.0, abs=1e-6)
    # 常数低维距离 -> 方差为 0 -> 相关定义为 0（而不是 NaN）
    constant = np.zeros_like(high)
    zero = diag.pair_distance_correlation(high, constant)
    assert zero["pearson"] == 0.0 and zero["spearman"] == 0.0
    with pytest.raises(NebulaError):
        diag.pair_distance_correlation(high, low[:2])


# --------------------------------------------------------------------------- 投影
def test_pca_projection_shape_and_determinism():
    matrix, _labels = _synthetic_matrix(n_per_cluster=6, n_clusters=3, dim=8, seed=1)
    two = diag.pca_projection(matrix, 2)
    three = diag.pca_projection(matrix, 3)
    assert two.shape == (18, 2) and three.shape == (18, 3)
    assert np.all(np.isfinite(two))
    np.testing.assert_array_equal(two, diag.pca_projection(matrix, 2))
    np.testing.assert_array_equal(three, diag.pca_projection(matrix, 3))
    with pytest.raises(NebulaError):
        diag.pca_projection(matrix, 99)


def test_umap_projection_wiring_and_clamping(fake_umap):
    matrix, _labels = _synthetic_matrix(n_per_cluster=4, n_clusters=2, dim=6, seed=2)
    coordinates, effective = diag.umap_projection(matrix, 2, 50)
    assert coordinates.shape == (8, 2)
    assert np.all(np.isfinite(coordinates))
    assert effective == 7
    kwargs = fake_umap[-1]
    assert kwargs["n_components"] == 2
    assert kwargs["n_neighbors"] == 7
    assert kwargs["metric"] == "cosine"
    assert kwargs["min_dist"] == pytest.approx(0.1)
    assert kwargs["random_state"] == 42
    assert kwargs["n_jobs"] == 1
    assert kwargs["transform_seed"] == 42
    repeat, repeat_effective = diag.umap_projection(matrix, 2, 50)
    assert repeat_effective == effective
    np.testing.assert_array_equal(coordinates, repeat)
    with pytest.raises(NebulaError):
        diag.umap_projection(matrix, 8, 5)  # n_components >= n_samples


def test_umap_projection_missing_dependency(monkeypatch):
    matrix, _labels = _synthetic_matrix(n_per_cluster=3, n_clusters=2, dim=4, seed=3)
    monkeypatch.setitem(sys.modules, "umap", None)
    with pytest.raises(NebulaError, match="umap"):
        diag.umap_projection(matrix, 2, 3)


# --------------------------------------------------------------------------- 合成簇诊断
def test_synthetic_cluster_high_dim_structure():
    matrix, labels = _synthetic_matrix(n_per_cluster=15, n_clusters=4, dim=8, seed=0)
    distance = diag.cosine_distance_matrix(matrix)
    within: list[float] = []
    between: list[float] = []
    for first in range(len(labels)):
        for second in range(first + 1, len(labels)):
            value = distance[first, second]
            if labels[first] == labels[second]:
                within.append(value)
            else:
                between.append(value)
    assert float(np.mean(within)) < float(np.mean(between))
    summary = diag._high_dim_summary(distance)
    assert summary["metric"] == "cosine_distance"
    assert summary["n_pairs"] == 60 * 59 // 2
    assert 0.0 < summary["mean"] < 2.0
    assert summary["min"] <= summary["median"] <= summary["max"]


def test_analyze_space_schema_ranges_and_determinism(fake_umap):
    matrix, _labels = _synthetic_matrix(n_per_cluster=15, n_clusters=4, dim=8, seed=0)
    space = _space(matrix, name="cluster-space")
    analysis = diag.analyze_space(
        space, k=10, umap_neighbors=(15, 30, 50), components=(2, 3)
    )
    report = analysis.report

    expected_labels = [
        "pca_2d",
        "pca_3d",
        "umap_n15_2d",
        "umap_n15_3d",
        "umap_n30_2d",
        "umap_n30_3d",
        "umap_n50_2d",
        "umap_n50_3d",
    ]
    assert [item["label"] for item in report["projections"]] == expected_labels
    assert len(analysis.projections) == 8
    assert report["n_works"] == 60 and report["dim"] == 8
    assert report["high_dim"]["n_pairs"] == 60 * 59 // 2
    assert report["high_dim"]["metric"] == "cosine_distance"

    required = {
        "method",
        "label",
        "n_components",
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
    }
    for projection in report["projections"]:
        assert required <= set(projection)
        assert 0.0 <= projection["topk_preservation"] <= 1.0
        assert 0.0 <= projection["top1_preservation"] <= 1.0
        assert 0.0 <= projection["stable_ratio"] <= 1.0
        assert -1.0 <= projection["distance_pearson"] <= 1.0
        assert -1.0 <= projection["distance_spearman"] <= 1.0
        assert projection["effective_k"] == 10
        assert projection["n_pairs"] == 60 * 59 // 2
        assert projection["method"] in ("pca", "umap")
        if projection["method"] == "pca":
            assert projection["n_neighbors"] is None
        else:
            assert projection["n_neighbors"] in (15, 30, 50)
            assert projection["effective_n_neighbors"] == projection["n_neighbors"]

    # UMAP 15/30/50 的两两 Top-10 保持率：3 对 × 2 个维度
    consistency = report["umap_neighbor_consistency"]
    assert len(consistency) == 6
    assert {(item["a"], item["b"]) for item in consistency} == {(15, 30), (15, 50), (30, 50)}
    for item in consistency:
        assert 0.0 <= item["topk_preservation"] <= 1.0
        assert 0.0 <= item["top1_preservation"] <= 1.0
    # 2D vs 3D：PCA 1 条 + UMAP 每个 neighbors 1 条
    dimension = report["dimension_consistency"]
    assert len(dimension) == 4
    assert [item["method"] for item in dimension] == ["pca", "umap", "umap", "umap"]
    for item in dimension:
        assert item["components_a"] == 2 and item["components_b"] == 3
        assert 0.0 <= item["topk_preservation"] <= 1.0

    # 报告可直接 JSON 序列化；重复运行逐元素确定
    json.dumps(report, ensure_ascii=False)
    repeat = diag.analyze_space(
        space, k=10, umap_neighbors=(15, 30, 50), components=(2, 3)
    )
    assert repeat.report == report
    for first, second in zip(analysis.projections, repeat.projections, strict=True):
        np.testing.assert_array_equal(first.coordinates, second.coordinates)
        np.testing.assert_allclose(first.distances, second.distances)
    assert fake_umap  # 确实走到了 UMAP 接线


def test_analyze_space_effective_k_with_small_sample(fake_umap):
    matrix, _labels = _synthetic_matrix(n_per_cluster=4, n_clusters=2, dim=5, seed=4)
    analysis = diag.analyze_space(
        _space(matrix), k=10, umap_neighbors=(15,), components=(2,)
    )
    report = analysis.report
    assert report["high_dim"]["n_pairs"] == 8 * 7 // 2
    pca = report["projections"][0]
    umap = report["projections"][1]
    assert pca["effective_k"] == 7
    assert umap["effective_k"] == 7
    assert umap["n_neighbors"] == 15
    assert umap["effective_n_neighbors"] == 7
    for item in report["projections"]:
        assert 0.0 <= item["topk_preservation"] <= 1.0
        assert -1.0 <= item["distance_pearson"] <= 1.0


def test_analyze_space_validation_errors():
    matrix, _labels = _synthetic_matrix(n_per_cluster=1, n_clusters=3, dim=4, seed=5)
    with pytest.raises(NebulaError):
        diag.analyze_space(_space(matrix), k=0)
    with pytest.raises(NebulaError):
        diag.analyze_space(_space(matrix), components=(1,))
    with pytest.raises(NebulaError):
        diag.analyze_space(_space(matrix), umap_neighbors=(1,))
    with pytest.raises(NebulaError):
        diag.analyze_space(_space(matrix), components=(2, 3))  # 3 部 < 4
    with pytest.raises(NebulaError):
        diag.SpaceMatrix.from_matrix(np.zeros((3, 4)))
    with pytest.raises(NebulaError):
        diag.SpaceMatrix.from_matrix(np.array([[1.0, np.nan]]))
    with pytest.raises(NebulaError):
        diag.SpaceMatrix.from_matrix(np.ones((3, 4)), work_ids=[1, 2])


# --------------------------------------------------------------------------- 报告输出
def test_write_reports_json_and_csv(tmp_path, fake_umap):
    matrix, _labels = _synthetic_matrix(n_per_cluster=3, n_clusters=4, dim=6, seed=6)
    analysis = _make_analysis(matrix=matrix, k=3, neighbors=(3, 5), components=(2,))
    result = _diagnostics_result(analysis)
    json_path = tmp_path / "reports" / "dim_diagnostics.json"
    csv_path = tmp_path / "reports" / "dim_diagnostics.csv"
    written_json, written_csv = diag.write_reports(
        result, json_path=json_path, csv_path=csv_path
    )
    assert written_json == json_path and written_csv == csv_path
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["tool"] == diag.TOOL_NAME
    assert payload["representations"][0]["n_works"] == 12
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0].keys()) == diag.CSV_COLUMNS
    assert len(rows) == len(analysis.projections)  # 1 PCA + 2 UMAP
    assert rows[0]["representation"] == analysis.space.name
    assert rows[0]["projection"] == "pca_2d"
    assert rows[0]["n_neighbors"] == ""
    for row in rows:
        assert float(row["topk_preservation"]) >= 0.0
        assert abs(float(row["distance_pearson"])) <= 1.0


def test_write_plotly_html(tmp_path, fake_umap):
    matrix, _labels = _synthetic_matrix(n_per_cluster=3, n_clusters=3, dim=5, seed=7)
    analysis = _make_analysis(matrix=matrix, k=3, neighbors=(3,), components=(2, 3))
    result = _diagnostics_result(analysis)
    works = {int(work_id): {"title": f"作品{work_id}"} for work_id in analysis.space.work_ids}
    paths = diag.write_plotly_html(result, tmp_path / "html", works=works)
    assert len(paths) == len(analysis.projections)  # pca_2d/pca_3d/umap_n3_2d/umap_n3_3d
    for path in paths:
        assert path.is_file()
        text = path.read_text(encoding="utf-8")
        assert "<html" in text.lower()
        assert "plotly" in text.lower()
    names = {path.name for path in paths}
    assert f"{diag.slugify(analysis.space.name)}__pca_2d.html" in names


# --------------------------------------------------------------------------- DB 加载
def test_load_space_canonical_only(config, catalog):
    version = "diag-seq-v1"
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
    representation_id = catalog.register_representation(
        "diag_space_v1",
        sequence_version=version,
        kind="full_mean",
        params={"test": "diagnostics"},
    )
    rep = catalog.get_representation(representation_id)
    assert rep is not None
    rng = np.random.default_rng(8)
    work_ids: list[int] = []
    for index in range(3):
        work_id, _ = catalog.upsert_local_work(f"diag/w{index}.mp4", title=f"作品{index}")
        work_ids.append(work_id)
        saved = R.save_representation_vectors(
            catalog, config, rep, work_id, {"joint": rng.normal(size=6).astype(np.float32)}
        )
        assert "joint" in saved
    catalog.link_works(work_ids[0], work_ids[2], method="test", confidence=1.0)
    catalog.set_representation_status(representation_id, "ready", dim=6)

    space = diag.load_space(catalog, config, "diag_space_v1", sequence_version=version)
    assert space.name == "diag_space_v1"
    assert space.sequence_version == version
    assert space.representation_id == representation_id
    assert space.n_works == 2  # duplicate 被排除
    assert space.work_ids == (work_ids[0], work_ids[1])
    assert space.dim == 6
    np.testing.assert_allclose(np.linalg.norm(space.matrix, axis=1), np.ones(2))

    with pytest.raises(NebulaError):
        diag.load_space(catalog, config, "missing_representation")


# --------------------------------------------------------------------------- CLI
def test_cli_help(capsys):
    with pytest.raises(SystemExit) as excinfo:
        diag.main(["--help"])
    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    for flag in (
        "--representation",
        "--k",
        "--umap-neighbors",
        "--components",
        "--out",
        "--csv",
        "--html-dir",
    ):
        assert flag in output
    parser = diag.build_parser()
    assert parser.parse_args([]).representation is None
    assert parser.parse_args(["--representation", "full_mean_v1", "temporal_stats_v1"]).representation == [
        "full_mean_v1",
        "temporal_stats_v1",
    ]
    repeated = parser.parse_args(
        ["--representation", "a", "--representation", "b"]
    )
    assert repeated.representation == ["a", "b"]


def test_cli_config_error(tmp_path):
    assert diag.main(["--config", str(tmp_path / "missing.yaml")]) == 2


def test_cli_end_to_end(config, catalog, tmp_path, fake_umap, capsys):
    version = "diag-cli-seq-v1"
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
    representation_id = catalog.register_representation(
        "diag_cli_space_v1",
        sequence_version=version,
        kind="full_mean",
        params={"test": "diagnostics-cli"},
    )
    rep = catalog.get_representation(representation_id)
    assert rep is not None
    rng = np.random.default_rng(9)
    for index in range(12):
        work_id, _ = catalog.upsert_local_work(f"diag/cli{index}.mp4", title=f"作品{index}")
        R.save_representation_vectors(
            catalog,
            config,
            rep,
            work_id,
            {"joint": rng.normal(size=4).astype(np.float32)},
        )
    catalog.set_representation_status(representation_id, "ready", dim=4)

    json_path = tmp_path / "cli" / "dim_diagnostics.json"
    csv_path = tmp_path / "cli" / "dim_diagnostics.csv"
    html_dir = tmp_path / "cli" / "html"
    code = diag.main(
        [
            "--config",
            str(config.path),
            "--representation",
            "diag_cli_space_v1",
            "--k",
            "3",
            "--umap-neighbors",
            "3",
            "5",
            "--components",
            "2",
            "--out",
            str(json_path),
            "--csv",
            str(csv_path),
            "--html-dir",
            str(html_dir),
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "diag_cli_space_v1" in output
    assert "pca_2d" in output
    assert "umap_n3_2d" in output

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["params"]["k"] == 3
    assert payload["params"]["umap_neighbors"] == [3, 5]
    assert payload["params"]["components"] == [2]
    representation = payload["representations"][0]
    assert representation["name"] == "diag_cli_space_v1"
    assert representation["n_works"] == 12
    labels = [item["label"] for item in representation["projections"]]
    assert labels == ["pca_2d", "umap_n3_2d", "umap_n5_2d"]
    assert all(0.0 <= item["topk_preservation"] <= 1.0 for item in representation["projections"])

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    html_files = sorted(html_dir.glob("*.html"))
    assert len(html_files) == 3


@pytest.mark.skipif(
    os.environ.get("NEBULA_REAL_UMAP_TESTS") != "1",
    reason="真实 UMAP 首次 numba 编译约 20-30s；设置 NEBULA_REAL_UMAP_TESTS=1 手动验证",
)
def test_real_umap_projection_smoke():
    matrix, _labels = _synthetic_matrix(n_per_cluster=5, n_clusters=3, dim=6, seed=10)
    coordinates, effective = diag.umap_projection(matrix, 2, 5)
    assert coordinates.shape == (15, 2)
    assert effective == 5
    assert np.all(np.isfinite(coordinates))
    analysis = diag.analyze_space(
        _space(matrix), k=3, umap_neighbors=(5,), components=(2,)
    )
    assert len(analysis.projections) == 2

def test_cli_default_representations(config, catalog, tmp_path, fake_umap):
    _install_representation(
        catalog, config, name=R.FULL_MEAN_V1, version="diag-default-seq-v1", seed=21
    )
    _install_representation(
        catalog,
        config,
        name=R.TEMPORAL_STATS_V1,
        version="diag-default-seq-v1",
        seed=22,
    )
    json_path = tmp_path / "default" / "dim_diagnostics.json"
    code = diag.main(
        [
            "--config",
            str(config.path),
            "--k",
            "2",
            "--umap-neighbors",
            "3",
            "--components",
            "2",
            "--out",
            str(json_path),
        ]
    )
    assert code == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["params"]["representations"] == list(diag.DEFAULT_REPRESENTATIONS)
    names = [item["name"] for item in payload["representations"]]
    assert names == list(diag.DEFAULT_REPRESENTATIONS)


def test_cli_default_output_paths(config, catalog, fake_umap):
    _install_representation(
        catalog, config, name="diag_default_path_v1", version="diag-default-path-v1", seed=23
    )
    code = diag.main(
        [
            "--config",
            str(config.path),
            "--representation",
            "diag_default_path_v1",
            "--k",
            "2",
            "--umap-neighbors",
            "3",
            "--components",
            "2",
        ]
    )
    assert code == 0
    assert (config.paths.logs / "dim_diagnostics.json").is_file()
    assert (config.paths.logs / "dim_diagnostics.csv").is_file()
