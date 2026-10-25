"""A3 片段稳定性（nebula.stability + tools/clip_stability.py）的离线测试。

用临时 DB + 合成 npy 验证：

- leave_one_out_vectors 的数学性质（去掉第 i 个片段后的归一化平均、置换一致、
  输入自动归一化、退化输入报中文 NebulaError）；
- stability_report 的 Top-k 重叠 / 名次变化 / 余弦漂移计算（对手工可推导的小例子断言）；
- duplicate 排除、work_ids 过滤、缺失片段向量/片段数不足/维度不一致的跳过语义；
- write_report 的 JSON/CSV 内容与原子落盘；
- tools/clip_stability.py CLI 的 --help 与离线端到端运行。

不加载模型、不访问网络、不写仓库数据目录（全部走临时目录）。
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nebula.config import Config
from nebula.db import Catalog
from nebula.embed import save_vector
from nebula.errors import NebulaError
from nebula.stability import (
    DEFAULT_K,
    leave_one_out_vectors,
    stability_report,
    write_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "tools" / "clip_stability.py"
VERSION = "v-test"


# ──────────────────────────────────────────────────────────────────────────────
# 合成数据
# ──────────────────────────────────────────────────────────────────────────────


def _unit(vector) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    assert norm > 0
    return array / norm


def _basis(dim: int, index: int) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    vector[index] = 1.0
    return vector


def _seed_catalog(config: Config, works: list[dict]) -> Catalog:
    """创建临时 Catalog 并登记作品/向量/片段；返回打开着的 catalog。"""

    catalog = Catalog(config.paths.db_path)
    catalog.initialize()
    catalog.register_embedding_version(
        VERSION,
        model_checkpoint="synthetic",
        model_revision=None,
        frames=1,
        clip_seconds=8.0,
        clips_per_work=5,
        audio_sample_rate=48000,
        params={"test": "clip-stability"},
    )
    for index, spec in enumerate(works, start=1):
        bvid = f"BV1xx411c7m{index}"
        work_id, _created = catalog.upsert_bilibili_work(
            bvid, title=spec.get("title", bvid), duration_s=spec.get("duration_s", 40.0)
        )
        catalog.begin_attempt(work_id, "encoding")
        work_dir = config.paths.embeddings / VERSION / f"{work_id:06d}"
        work_vector = _unit(spec["work"])
        save_vector(work_dir / "work_joint.npy", work_vector)
        catalog.register_embedding(
            work_id,
            version=VERSION,
            kind="work_joint",
            path=config.data_relpath(work_dir / "work_joint.npy"),
            dim=int(work_vector.shape[0]),
        )
        clips = spec.get("clips")
        if clips is not None:
            matrix = np.stack([_unit(clip) for clip in clips]).astype(np.float32)
            save_vector(work_dir / "clip_joint.npy", matrix)
            catalog.register_embedding(
                work_id,
                version=VERSION,
                kind="clip_joint",
                path=config.data_relpath(work_dir / "clip_joint.npy"),
                dim=int(matrix.shape[1]),
                n_clips=int(matrix.shape[0]),
            )
            catalog.replace_clips(
                work_id,
                version=VERSION,
                clips=[(i, i * 8.0, 8.0) for i in range(matrix.shape[0])],
            )
        catalog.complete(work_id)
    return catalog


@pytest.fixture
def make_catalog(config: Config):
    """返回一个工厂：seed(works) -> Catalog；测试结束时统一关闭。"""

    opened: list[Catalog] = []

    def factory(works: list[dict]) -> Catalog:
        catalog = _seed_catalog(config, works)
        opened.append(catalog)
        return catalog

    yield factory
    for catalog in opened:
        catalog.close()


def _four_work_case() -> list[dict]:
    """手工可推导的 4 部作品（dim=3，k=1 时 w2 的 Top-1 会因留一而变化）。"""

    diag = _unit(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    return [
        {"title": "稳定-1", "work": _basis(3, 0), "clips": [_basis(3, 0), _basis(3, 0)]},
        {
            "title": "不稳定-2",
            "work": _basis(3, 1),
            "clips": [_basis(3, 1), _basis(3, 0)],
        },
        {"title": "稳定-3", "work": _basis(3, 2), "clips": [_basis(3, 2), _basis(3, 2)]},
        {"title": "稳定-4", "work": diag, "clips": [diag, diag]},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# leave_one_out_vectors 数学性质
# ──────────────────────────────────────────────────────────────────────────────


def test_loo_equal_clips_returns_same_direction() -> None:
    base = _unit(np.array([0.3, -0.4, 0.5, 0.6]))
    matrix = np.stack([base, base, base])
    result = leave_one_out_vectors(matrix)
    assert result.shape == (3, 4)
    assert result.dtype == np.float32
    for row in result:
        assert np.allclose(row, base, atol=1e-6)
        assert float(np.linalg.norm(row)) == pytest.approx(1.0, abs=1e-6)


def test_loo_removes_the_expected_clip() -> None:
    matrix = np.stack([_basis(3, 0), _basis(3, 1), _basis(3, 2)])
    result = leave_one_out_vectors(matrix)
    assert np.allclose(result[0], _unit(_basis(3, 1) + _basis(3, 2)), atol=1e-6)
    assert np.allclose(result[1], _unit(_basis(3, 0) + _basis(3, 2)), atol=1e-6)
    assert np.allclose(result[2], _unit(_basis(3, 0) + _basis(3, 1)), atol=1e-6)


def test_loo_result_is_permutation_equivalent() -> None:
    raw = np.array(
        [
            [0.2, 0.4, -0.3, 0.1],
            [-0.5, 0.2, 0.7, 0.0],
            [0.1, -0.2, 0.3, 0.9],
            [0.4, 0.4, 0.4, 0.4],
        ],
        dtype=np.float32,
    )
    permutation = [2, 0, 3, 1]
    direct = leave_one_out_vectors(raw)
    permuted = leave_one_out_vectors(raw[permutation])
    # 在置换后的矩阵里，第 j 行对应原矩阵的第 permutation[j] 行被去掉
    assert np.allclose(permuted, direct[permutation], atol=1e-6)


def test_loo_normalizes_input_rows_by_default() -> None:
    matrix = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 0.1],
        ],
        dtype=np.float32,
    )
    recorded = leave_one_out_vectors(matrix)
    normalized = np.stack([_unit(row) for row in matrix])
    assert np.allclose(recorded, leave_one_out_vectors(normalized), atol=1e-6)


@pytest.mark.parametrize(
    "matrix",
    [
        np.zeros((1, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((2, 0), dtype=np.float32),
        np.zeros((2, 2, 2), dtype=np.float32),
        np.array([[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    ],
)
def test_loo_invalid_input_raises_chinese_error(matrix: np.ndarray) -> None:
    with pytest.raises(NebulaError):
        leave_one_out_vectors(matrix)


def test_loo_zero_cancellation_raises() -> None:
    matrix = np.stack([_basis(3, 0), _basis(3, 1), -_basis(3, 1)])
    with pytest.raises(NebulaError, match="抵消"):
        leave_one_out_vectors(matrix)


# ──────────────────────────────────────────────────────────────────────────────
# stability_report：手工可推导的小例子
# ──────────────────────────────────────────────────────────────────────────────


def test_stability_report_hand_computed_case(config: Config, make_catalog) -> None:
    catalog = make_catalog(_four_work_case())
    works = catalog.list_works()
    ids = {work["title"]: int(work["id"]) for work in works}
    report = stability_report(catalog, config, version=VERSION, k=1)
    default_report = stability_report(catalog, config, version=VERSION)
    assert default_report["k"] == DEFAULT_K
    assert default_report["effective_k"] == 3  # 4 部作品最多只有 3 个邻居

    assert report["embedding_version"] == VERSION
    assert report["k"] == 1
    assert report["effective_k"] == 1
    assert report["n_canonical_works"] == 4
    assert report["n_evaluated"] == 4
    assert report["n_skipped"] == 0
    assert report["n_excluded_duplicates"] == 0
    assert report["mean_overlap"] == pytest.approx(0.875)
    assert report["unstable_count"] == 1
    assert report["mean_cosine_drift"] == pytest.approx(0.0)

    by_id = {item["work_id"]: item for item in report["works"]}
    assert list(by_id) == sorted(by_id)

    unstable = by_id[ids["不稳定-2"]]
    assert unstable["n_clips"] == 2
    assert unstable["mean_overlap"] == pytest.approx(0.5)
    assert unstable["min_overlap"] == pytest.approx(0.0)
    assert unstable["max_rank_shift"] == 1
    assert unstable["unstable"] is True
    # 正式 Top-1 是 w4（相似度 0.707）；去掉片段 0 后 Top-1 变成 w1（相似度≈1）
    assert unstable["official_neighbors"][0]["work_id"] == ids["稳定-4"]
    assert unstable["leave_one_out"][0]["neighbors"][0]["work_id"] == ids["稳定-1"]
    assert unstable["leave_one_out"][0]["overlap"] == 0
    assert unstable["leave_one_out"][1]["overlap"] == 1
    assert unstable["leave_one_out"][0]["mean_cosine_drift"] == pytest.approx(0.0)

    for title in ("稳定-1", "稳定-3", "稳定-4"):
        entry = by_id[ids[title]]
        assert entry["mean_overlap"] == pytest.approx(1.0)
        assert entry["unstable"] is False


def test_stability_report_excludes_linked_duplicates(config: Config, make_catalog) -> None:
    catalog = make_catalog(_four_work_case())
    works = catalog.list_works()
    ids = {work["title"]: int(work["id"]) for work in works}
    catalog.link_works(
        ids["稳定-1"], ids["稳定-3"], method="title+duration", confidence=0.99, note="测试"
    )
    report = stability_report(catalog, config, version=VERSION, k=1)

    assert report["n_canonical_works"] == 3
    assert report["n_excluded_duplicates"] == 1
    assert report["n_evaluated"] == 3
    evaluated_ids = {item["work_id"] for item in report["works"]}
    assert ids["稳定-3"] not in evaluated_ids
    for item in report["works"]:
        neighbor_ids = {neighbor["work_id"] for neighbor in item["official_neighbors"]}
        for entry in item["leave_one_out"]:
            neighbor_ids |= {neighbor["work_id"] for neighbor in entry["neighbors"]}
        assert ids["稳定-3"] not in neighbor_ids


def test_stability_report_work_ids_filter(config: Config, make_catalog) -> None:
    catalog = make_catalog(_four_work_case())
    works = catalog.list_works()
    ids = {work["title"]: int(work["id"]) for work in works}
    report = stability_report(
        catalog,
        config,
        version=VERSION,
        k=1,
        work_ids=[ids["不稳定-2"], 99999, ids["不稳定-2"]],
    )
    assert report["n_evaluated"] == 1
    assert [item["work_id"] for item in report["works"]] == [ids["不稳定-2"]]
    assert report["n_skipped"] == 2
    reasons = "；".join(item["reason"] for item in report["skipped"])
    assert "canonical" in reasons
    assert "重复指定" in reasons


def test_stability_report_skips_missing_clip_vectors(config: Config, make_catalog) -> None:
    works = [
        {"title": "有片段", "work": _basis(3, 0), "clips": [_basis(3, 0), _basis(3, 0)]},
        {"title": "无片段", "work": _basis(3, 1), "clips": None},
        {"title": "单片段", "work": _basis(3, 2), "clips": [_basis(3, 2)]},
    ]
    catalog = make_catalog(works)
    report = stability_report(catalog, config, version=VERSION, k=1)
    assert report["n_canonical_works"] == 3
    assert report["n_evaluated"] == 1
    assert report["n_skipped"] == 2
    reasons = "；".join(item["reason"] for item in report["skipped"])
    assert "clip_joint" in reasons
    assert "片段数不足" in reasons
    assert report["mean_overlap"] == pytest.approx(1.0)


def test_stability_report_skips_clip_dimension_mismatch(config: Config, make_catalog) -> None:
    works = [
        {"title": "正常", "work": _basis(3, 0), "clips": [_basis(3, 0), _basis(3, 0)]},
        {"title": "维度不符", "work": _basis(3, 1), "clips": [_basis(4, 0), _basis(4, 1)]},
    ]
    catalog = make_catalog(works)
    report = stability_report(catalog, config, version=VERSION, k=1)
    assert report["n_evaluated"] == 1
    assert report["n_skipped"] == 1
    assert "维度" in report["skipped"][0]["reason"]


def test_stability_report_work_dimension_mismatch_raises(config: Config, make_catalog) -> None:
    works = [
        {"title": "dim3", "work": _basis(3, 0), "clips": [_basis(3, 0), _basis(3, 0)]},
        {"title": "dim4", "work": _basis(4, 0), "clips": [_basis(4, 0), _basis(4, 0)]},
    ]
    catalog = make_catalog(works)
    with pytest.raises(NebulaError, match="维度不一致"):
        stability_report(catalog, config, version=VERSION, k=1)


def test_stability_report_requires_registered_vectors(config: Config) -> None:
    catalog = Catalog(config.paths.db_path)
    catalog.initialize()
    try:
        with pytest.raises(NebulaError, match="没有已登记"):
            stability_report(catalog, config)
    finally:
        catalog.close()


@pytest.mark.parametrize("bad_k", [0, -1, True, 2.5])
def test_stability_report_rejects_bad_k(config: Config, make_catalog, bad_k) -> None:
    catalog = make_catalog(_four_work_case())
    with pytest.raises(NebulaError, match="k 必须"):
        stability_report(catalog, config, version=VERSION, k=bad_k)


def test_stability_report_rejects_bad_threshold(config: Config, make_catalog) -> None:
    catalog = make_catalog(_four_work_case())
    with pytest.raises(NebulaError, match="阈值"):
        stability_report(catalog, config, version=VERSION, k=1, unstable_threshold=0.0)


def test_stability_report_is_deterministic(config: Config, make_catalog) -> None:
    catalog = make_catalog(_four_work_case())
    first = stability_report(catalog, config, version=VERSION, k=1)
    second = stability_report(catalog, config, version=VERSION, k=1)
    first.pop("generated_at")
    second.pop("generated_at")
    assert first == second
    assert first["mean_overlap"] == pytest.approx(0.875)


# ──────────────────────────────────────────────────────────────────────────────
# write_report 落盘
# ──────────────────────────────────────────────────────────────────────────────


def test_write_report_json_and_csv(tmp_path: Path, config: Config, make_catalog) -> None:
    catalog = make_catalog(_four_work_case())
    report = stability_report(catalog, config, version=VERSION, k=1)
    out_dir = tmp_path / "nested" / "reports"
    json_path = out_dir / "clip_stability.json"
    csv_path = out_dir / "clip_stability.csv"
    write_report(report, json_path=json_path, csv_path=csv_path)

    assert json_path.is_file() and csv_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["embedding_version"] == VERSION
    assert len(payload["works"]) == 4
    assert payload["caveats"]

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[0]["work_id"]
    assert rows[0]["unstable"] in {"true", "false"}
    assert {row["k"] for row in rows} == {"1"}

    # 原子写：目录里不应残留临时文件
    assert not list(out_dir.glob("*.tmp"))
    assert not list(out_dir.glob(".*.tmp"))
    # 重复写入幂等（内容一致）
    before = json_path.read_text(encoding="utf-8")
    write_report(report, json_path=json_path, csv_path=csv_path)
    assert json_path.read_text(encoding="utf-8") == before


def test_write_report_requires_dict(tmp_path: Path) -> None:
    with pytest.raises(NebulaError, match="dict"):
        write_report(["not-a-report"], json_path=tmp_path / "a.json", csv_path=tmp_path / "a.csv")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def _run_cli(args: list[str], *, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return subprocess.run(
        [sys.executable, str(CLI_PATH), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        env=env,
        check=False,
    )


def test_cli_help_runs() -> None:
    proc = _run_cli(["--help"])
    assert proc.returncode == 0, proc.stderr
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "usage" in combined.lower()
    for option in ("--config", "--version", "--k", "--out-dir"):
        assert option in combined


def test_cli_end_to_end_offline(tmp_path: Path, config: Config, make_catalog) -> None:
    catalog = make_catalog(_four_work_case())
    catalog.close()
    out_dir = tmp_path / "cli-reports"
    proc = _run_cli(
        [
            "--config",
            str(config.path),
            "--version",
            VERSION,
            "--k",
            "1",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    assert "片段稳定性报告" in proc.stdout
    assert "评估作品: 4 部" in proc.stdout
    assert "平均 Top-k 重叠率" in proc.stdout
    json_path = out_dir / "clip_stability.json"
    csv_path = out_dir / "clip_stability.csv"
    assert json_path.is_file() and csv_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["embedding_version"] == VERSION
    assert payload["n_evaluated"] == 4
    assert payload["unstable_count"] == 1


def test_cli_work_ids_filter(tmp_path: Path, config: Config, make_catalog) -> None:
    """CLI --work-ids 只评估指定作品（近邻空间仍用全部 canonical）。"""

    catalog = make_catalog(_four_work_case())
    ids = {work["title"]: int(work["id"]) for work in catalog.list_works()}
    catalog.close()
    out_dir = tmp_path / "cli-work-ids"
    target = ids["不稳定-2"]
    proc = _run_cli(
        [
            "--config",
            str(config.path),
            "--version",
            VERSION,
            "--k",
            "1",
            "--out-dir",
            str(out_dir),
            "--work-ids",
            str(target),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads((out_dir / "clip_stability.json").read_text(encoding="utf-8"))
    assert payload["n_evaluated"] == 1
    assert payload["works"][0]["work_id"] == target
    assert payload["n_canonical_works"] == 4  # 近邻空间不受 work-ids 限制


def test_cli_missing_vectors_returns_error_code(tmp_path: Path, config: Config) -> None:
    # 空 DB：CLI 应给出中文错误并返回非 0，而不是抛 traceback
    out_dir = tmp_path / "empty-reports"
    proc = _run_cli(["--config", str(config.path), "--out-dir", str(out_dir)])
    assert proc.returncode == 1
    assert "稳定性报告失败" in proc.stderr
    assert "Traceback" not in proc.stderr
