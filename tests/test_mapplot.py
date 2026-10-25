"""nebula.mapplot 单元测试：UMAP 构建、版本登记、增量更新与 CLI。

测试使用 numpy 合成向量 + 临时 SQLite，不依赖真实模型/权重/网络。
合成数据只取 8 部作品（n_neighbors 自动降为 7），保证 UMAP 快速且确定性可验证。
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from nebula import mapplot
from nebula.config import Config
from nebula.db import Catalog
from nebula.errors import MapError
from nebula.mapplot import (
    COORD_COLUMNS,
    build_map,
    load_embedding_matrix,
    main,
    transform_new,
)

VECTOR_DIM = 8
WORK_COUNT = 8


# --------------------------------------------------------------------------- 合成数据
def _ensure_embedding_version(
    catalog: Catalog, config: Config, version: str | None = None
) -> str:
    target = version or config.embedding_version
    catalog.register_embedding_version(
        target,
        model_checkpoint=config.model.checkpoint,
        model_revision=config.model.revision,
        frames=config.model.frames,
        clip_seconds=config.media.clip_seconds,
        clips_per_work=config.media.clips_per_work,
        audio_sample_rate=config.media.audio_sample_rate,
        params={},
    )
    return target


def _add_works(
    catalog: Catalog,
    config: Config,
    count: int,
    *,
    dim: int = VECTOR_DIM,
    seed: int = 0,
    start: int = 0,
    version: str | None = None,
    normalize: bool = False,
) -> list[int]:
    """登记 count 部合成作品及其 work_joint 向量文件，返回 work_id 列表。"""

    embedding_version = _ensure_embedding_version(catalog, config, version)
    rng = np.random.default_rng(seed)
    work_ids: list[int] = []
    for index in range(count):
        number = start + index
        work_id, _ = catalog.upsert_bilibili_work(
            f"BV1xx411c7{number:02d}",
            title=f"合成作品 {number}",
            author_uid=f"uid{number % 3}",
            duration_s=30.0 + number,
            publish_date="2025-01-02",
        )
        vector = rng.normal(size=dim).astype(np.float32)
        if normalize:
            vector = vector / np.linalg.norm(vector)
        path = config.paths.embeddings / embedding_version / str(work_id) / "work_joint.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vector)
        catalog.register_embedding(
            work_id,
            version=embedding_version,
            kind="work_joint",
            path=config.data_relpath(path),
            dim=dim,
            n_clips=2,
        )
        work_ids.append(work_id)
    return work_ids


def _vector_path(config: Config, version: str, work_id: int) -> Path:
    return config.paths.embeddings / version / str(work_id) / "work_joint.npy"


def _read_coords(result) -> pd.DataFrame:
    return pd.read_parquet(result.coords_path)


# --------------------------------------------------------------------------- 向量读取
def test_load_embedding_matrix_normalizes(config: Config, catalog: Catalog):
    ids = _add_works(catalog, config, 4, normalize=False)
    work_ids, matrix = load_embedding_matrix(catalog, config)

    assert work_ids == ids
    assert matrix.shape == (4, VECTOR_DIM)
    assert matrix.dtype == np.float32
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    np.testing.assert_allclose(norms, np.ones(4), atol=1e-6)


def test_load_embedding_matrix_too_few_works(config: Config, catalog: Catalog):
    _add_works(catalog, config, mapplot.MIN_WORKS - 1)
    with pytest.raises(MapError, match="作品数量不足"):
        load_embedding_matrix(catalog, config)


def test_load_embedding_matrix_missing_version(config: Config, catalog: Catalog):
    _add_works(catalog, config, 3)
    with pytest.raises(MapError, match="作品数量不足"):
        load_embedding_matrix(catalog, config, version="不存在的向量版本")


def test_load_embedding_matrix_dim_mismatch(config: Config, catalog: Catalog):
    _add_works(catalog, config, 2, dim=8)
    _add_works(catalog, config, 1, dim=4, start=10)
    with pytest.raises(MapError, match="维度不一致"):
        load_embedding_matrix(catalog, config)


def test_load_embedding_matrix_missing_vector_file(config: Config, catalog: Catalog):
    version = _ensure_embedding_version(catalog, config)
    missing_id, _ = catalog.upsert_bilibili_work("BV1xx411c7ob", title="缺向量")
    catalog.register_embedding(
        missing_id,
        version=version,
        kind="work_joint",
        path=config.data_relpath(config.paths.embeddings / "不存在" / "work_joint.npy"),
        dim=VECTOR_DIM,
    )
    _add_works(catalog, config, 2, start=20)
    with pytest.raises(MapError, match="向量文件不存在"):
        load_embedding_matrix(catalog, config)


def test_load_embedding_matrix_zero_vector(config: Config, catalog: Catalog):
    ids = _add_works(catalog, config, 3)
    np.save(_vector_path(config, config.embedding_version, ids[0]), np.zeros(VECTOR_DIM, dtype=np.float32))
    with pytest.raises(MapError, match="零向量"):
        load_embedding_matrix(catalog, config)


def test_load_embedding_matrix_registered_dim_mismatch(config: Config, catalog: Catalog):
    ids = _add_works(catalog, config, 3)
    np.save(_vector_path(config, config.embedding_version, ids[1]), np.ones(5, dtype=np.float32))
    with pytest.raises(MapError, match="维度与登记不一致"):
        load_embedding_matrix(catalog, config)


# --------------------------------------------------------------------------- 构建星云
def test_build_map_outputs_and_registration(config: Config, catalog: Catalog):
    ids = _add_works(catalog, config, WORK_COUNT)
    result = build_map(catalog, config, note="单元测试")

    assert result.version == "map_v001"
    assert result.embedding_version == config.embedding_version
    assert result.n_works == WORK_COUNT
    assert result.coords_path == config.paths.maps / "map_v001" / "coords.parquet"
    for path in (result.coords_path, result.model_2d_path, result.model_3d_path):
        assert path.is_file()
    assert (config.paths.maps / "map_v001" / "meta.json").is_file()

    frame = _read_coords(result)
    assert list(frame.columns) == list(COORD_COLUMNS)
    assert frame["work_id"].tolist() == ids
    assert len(frame) == WORK_COUNT
    coordinate_values = frame[["x2", "y2", "x3", "y3", "z3"]].to_numpy(dtype=float)
    assert np.all(np.isfinite(coordinate_values))
    # 2D 与 3D 是两个独立模型：形状不同，不应是简单复制
    assert not np.allclose(frame["x2"].to_numpy(), frame["x3"].to_numpy())

    model_2d = joblib.load(result.model_2d_path)
    model_3d = joblib.load(result.model_3d_path)
    assert callable(model_2d.transform)
    assert callable(model_3d.transform)
    assert model_2d.n_components == 2
    assert model_3d.n_components == 3
    assert model_2d.get_params()["n_neighbors"] == min(config.umap.n_neighbors, WORK_COUNT - 1)

    meta = json.loads((config.paths.maps / "map_v001" / "meta.json").read_text(encoding="utf-8"))
    assert meta["version"] == "map_v001"
    assert meta["embedding_version"] == config.embedding_version
    assert meta["n_works"] == WORK_COUNT
    assert meta["params"]["metric"] == "cosine"
    assert meta["params"]["random_state"] == config.umap.random_state
    assert meta["params"]["n_neighbors"] == min(config.umap.n_neighbors, WORK_COUNT - 1)
    assert meta["dim"] == VECTOR_DIM

    map_row = catalog.get_map("map_v001")
    assert map_row is not None
    assert map_row["status"] == "ready"
    assert map_row["n_works"] == WORK_COUNT
    assert map_row["note"] == "单元测试"
    assert map_row["coords_path"] == "maps/map_v001/coords.parquet"
    assert map_row["model_2d_path"] == "maps/map_v001/map_2d.joblib"
    assert map_row["model_3d_path"] == "maps/map_v001/map_3d.joblib"
    assert len(catalog.load_map_rows("map_v001")) == WORK_COUNT


def test_build_map_versions_increment_and_determinism(config: Config, catalog: Catalog):
    _add_works(catalog, config, WORK_COUNT)
    first = build_map(catalog, config)
    second = build_map(catalog, config)

    assert first.version == "map_v001"
    assert second.version == "map_v002"
    assert catalog.next_map_version() == "map_v003"

    first_frame = _read_coords(first).set_index("work_id")
    second_frame = _read_coords(second).set_index("work_id")
    assert first_frame.index.tolist() == second_frame.index.tolist()
    # 固定 random_state 与 n_jobs=1 时，同一输入应得到完全一致的坐标
    for column in ("x2", "y2", "x3", "y3", "z3"):
        assert np.array_equal(
            first_frame[column].to_numpy(), second_frame[column].to_numpy()
        ), f"{column} 在相同随机种子下不一致"


def test_build_map_with_minimum_works(config: Config, catalog: Catalog):
    """达到 MIN_WORKS 下限（3 部）也应能生成 2D/3D 星云（自动改用 random 初始化）。"""

    _add_works(catalog, config, mapplot.MIN_WORKS)
    result = build_map(catalog, config)
    assert result.n_works == mapplot.MIN_WORKS
    frame = _read_coords(result)
    assert len(frame) == mapplot.MIN_WORKS
    assert np.all(np.isfinite(frame[["x2", "y2", "x3", "y3", "z3"]].to_numpy(dtype=float)))


def test_build_map_cleans_up_on_failure(config: Config, catalog: Catalog, monkeypatch):
    _add_works(catalog, config, WORK_COUNT)

    def boom(model, path):
        raise OSError("模拟磁盘写入失败")

    monkeypatch.setattr(mapplot, "_dump_model", boom)
    with pytest.raises(MapError, match="生成星云失败"):
        build_map(catalog, config)

    assert not (config.paths.maps / "map_v001").exists()
    assert catalog.list_maps() == []


# --------------------------------------------------------------------------- 增量更新
def test_transform_new_appends_and_keeps_old_coords(config: Config, catalog: Catalog):
    old_ids = _add_works(catalog, config, WORK_COUNT)
    result = build_map(catalog, config)
    old_frame = _read_coords(result).set_index("work_id")

    new_ids = _add_works(catalog, config, 2, seed=99, start=100)
    added = transform_new(catalog, config, map_version=result.version)
    assert added == 2

    new_frame = _read_coords(result).set_index("work_id")
    assert len(new_frame) == WORK_COUNT + 2
    assert set(new_ids).issubset(set(new_frame.index))
    for work_id in old_ids:
        for column in ("x2", "y2", "x3", "y3", "z3"):
            assert new_frame.loc[work_id, column] == old_frame.loc[work_id, column]
    new_values = new_frame.loc[new_ids, ["x2", "y2", "x3", "y3", "z3"]].to_numpy(dtype=float)
    assert np.all(np.isfinite(new_values))

    db_rows = catalog.load_map_rows(result.version)
    assert len(db_rows) == WORK_COUNT + 2
    assert catalog.get_map(result.version)["n_works"] == WORK_COUNT + 2

    # 没有新作品时应是无操作
    assert transform_new(catalog, config, map_version=result.version) == 0
    assert len(_read_coords(result)) == WORK_COUNT + 2


def test_transform_new_uses_latest_map(config: Config, catalog: Catalog):
    _add_works(catalog, config, 5)
    build_map(catalog, config)
    _add_works(catalog, config, 1, start=50)
    assert transform_new(catalog, config) == 1
    assert catalog.latest_map()["n_works"] == 6


def test_transform_new_errors(config: Config, catalog: Catalog):
    with pytest.raises(MapError, match="还没有可用的星云版本"):
        transform_new(catalog, config)

    _add_works(catalog, config, 4)
    result = build_map(catalog, config)
    with pytest.raises(MapError, match="星云版本不存在"):
        transform_new(catalog, config, map_version="map_v999")

    _add_works(catalog, config, 1, start=60)
    result.model_3d_path.unlink()
    with pytest.raises(MapError, match="模型文件缺失"):
        transform_new(catalog, config, map_version=result.version)


# --------------------------------------------------------------------------- CLI
def test_cli_build_list_transform(config: Config, catalog: Catalog, capsys):
    _add_works(catalog, config, WORK_COUNT)

    assert main(["build", "--config", str(config.path), "--note", "CLI 测试"]) == 0
    out = capsys.readouterr().out
    assert "map_v001" in out
    assert f"{WORK_COUNT} 部作品" in out

    assert main(["list", "--config", str(config.path)]) == 0
    out = capsys.readouterr().out
    assert "map_v001" in out
    assert "CLI 测试" in out

    _add_works(catalog, config, 1, start=70)
    assert main(["transform", "--config", str(config.path), "--map-version", "map_v001"]) == 0
    out = capsys.readouterr().out
    assert "新增 1 部作品" in out


def test_cli_build_uses_embedding_version_option(config: Config, catalog: Catalog, capsys):
    custom_version = "custom-vector-v1"
    _add_works(catalog, config, 6, version=custom_version)

    assert main(["build", "--config", str(config.path), "--embedding-version", custom_version]) == 0
    capsys.readouterr()
    map_row = catalog.latest_map()
    assert map_row is not None
    assert map_row["embedding_version"] == custom_version


def test_cli_build_too_few_works_returns_error(config: Config, capsys):
    assert main(["build", "--config", str(config.path)]) == 1
    captured = capsys.readouterr()
    assert "错误" in captured.err
    assert "作品数量不足" in captured.err


def test_cli_missing_config_returns_error(tmp_path: Path, capsys):
    missing = tmp_path / "no-such-config.yaml"
    assert main(["list", "--config", str(missing)]) == 1
    captured = capsys.readouterr()
    assert "错误" in captured.err
