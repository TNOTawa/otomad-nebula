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
    build_map_variants,
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


# --------------------------------------------------------------------------- 筛选条件
def _add_work(
    catalog: Catalog,
    config: Config,
    number: int,
    *,
    source: str = "bilibili",
    dim: int = VECTOR_DIM,
    seed: int | None = None,
    version: str | None = None,
) -> int:
    """登记一部指定来源的合成作品，返回 work_id。"""

    embedding_version = _ensure_embedding_version(catalog, config, version)
    if source == "local":
        work_id, _ = catalog.upsert_local_work(
            f"clips/作品 {number}.mp4", title=f"作品 {number}"
        )
    else:
        work_id, _ = catalog.upsert_bilibili_work(
            f"BV1xx411c7{number:02d}", title=f"作品 {number}", duration_s=30.0 + number
        )
    rng = np.random.default_rng(number if seed is None else seed)
    vector = rng.normal(size=dim).astype(np.float32)
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
    return work_id


def _seed_mixed_works(
    catalog: Catalog,
    config: Config,
    *,
    local_numbers: tuple[int, ...] = (0, 2, 4, 6),
    bilibili_numbers: tuple[int, ...] = (1, 3, 5, 7),
) -> dict[str, list[int]]:
    """本地/Bilibili 混合作品 + 2025/2024 两个榜单。"""

    local_ids = [_add_work(catalog, config, number, source="local") for number in local_numbers]
    bilibili_ids = [_add_work(catalog, config, number) for number in bilibili_numbers]
    all_ids = local_ids + bilibili_ids
    votes_2025 = catalog.ensure_list(
        "votes_2025", name="2025年音MAD选票", kind="votes", vote_year=2025
    )
    for index, work_id in enumerate(all_ids):
        catalog.add_appearance(work_id, votes_2025, rank=index + 1, votes=100 - index)
    votes_2024 = catalog.ensure_list(
        "votes_2024", name="2024年音MAD选票", kind="votes", vote_year=2024
    )
    for work_id in (local_ids[0], local_ids[1], bilibili_ids[0]):
        catalog.add_appearance(work_id, votes_2024, rank=1)
    return {"local": local_ids, "bilibili": bilibili_ids, "all": all_ids}


def test_load_embedding_matrix_filter_spec(config: Config, catalog: Catalog):
    groups = _seed_mixed_works(catalog, config)

    ids, matrix = load_embedding_matrix(catalog, config, filter_spec={"dataset": "local"})
    assert ids == groups["local"]
    assert matrix.shape == (len(groups["local"]), VECTOR_DIM)

    ids, _ = load_embedding_matrix(catalog, config, filter_spec={"dataset": "bilibili"})
    assert ids == groups["bilibili"]

    expected_2024 = sorted(
        [groups["local"][0], groups["local"][1], groups["bilibili"][0]]
    )
    ids, _ = load_embedding_matrix(catalog, config, filter_spec={"vote_year": 2024})
    assert ids == expected_2024

    ids, _ = load_embedding_matrix(catalog, config, filter_spec={"list_key": "votes_2024"})
    assert ids == expected_2024

    ids, _ = load_embedding_matrix(
        catalog, config, filter_spec={"dataset": "local", "vote_year": 2025}
    )
    assert ids == groups["local"]

    # 默认不过滤 = 全部 canonical 作品
    ids, _ = load_embedding_matrix(catalog, config)
    assert ids == sorted(groups["all"])


def test_load_embedding_matrix_invalid_filter_spec(config: Config, catalog: Catalog):
    _seed_mixed_works(catalog, config)
    with pytest.raises(MapError, match="未知的筛选条件"):
        load_embedding_matrix(catalog, config, filter_spec={"source": "local"})
    with pytest.raises(MapError, match="dataset 只支持"):
        load_embedding_matrix(catalog, config, filter_spec={"dataset": "youtube"})
    with pytest.raises(MapError, match="vote_year 必须是整数"):
        load_embedding_matrix(catalog, config, filter_spec={"vote_year": "2025年"})
    with pytest.raises(MapError, match="include_duplicates 必须是布尔值"):
        load_embedding_matrix(catalog, config, filter_spec={"include_duplicates": "yes"})


def test_filter_too_few_works_raises(config: Config, catalog: Catalog):
    _seed_mixed_works(catalog, config)
    # 不存在的年份 -> 0 部作品，筛选后不足 MIN_WORKS
    with pytest.raises(MapError, match="筛选后作品数量不足"):
        load_embedding_matrix(catalog, config, filter_spec={"vote_year": 2023})


def test_duplicates_excluded_by_default(config: Config, catalog: Catalog):
    groups = _seed_mixed_works(catalog, config)
    canonical = groups["bilibili"][0]
    duplicate = groups["local"][0]
    catalog.link_works(canonical, duplicate, method="title+duration", confidence=0.99)

    ids, _ = load_embedding_matrix(catalog, config)
    assert duplicate not in ids
    assert canonical in ids
    assert len(ids) == len(groups["all"]) - 1

    ids_included, _ = load_embedding_matrix(
        catalog, config, filter_spec={"include_duplicates": True}
    )
    assert duplicate in ids_included
    assert len(ids_included) == len(groups["all"])


def test_build_map_persists_filter_spec(config: Config, catalog: Catalog):
    groups = _seed_mixed_works(catalog, config)
    result = build_map(
        catalog,
        config,
        filter_spec={"dataset": "bilibili", "vote_year": 2025},
        note="仅 B 站 2025",
    )

    assert result.n_works == len(groups["bilibili"])
    frame = _read_coords(result)
    assert set(frame["work_id"].tolist()) == set(groups["bilibili"])

    map_row = catalog.get_map(result.version)
    stored_filter = json.loads(map_row["filter_json"])
    assert stored_filter == {"dataset": "bilibili", "vote_year": 2025}
    params = json.loads(map_row["params_json"])
    assert params["filter_spec"] == {"dataset": "bilibili", "vote_year": 2025}

    meta = json.loads((result.coords_path.parent / "meta.json").read_text(encoding="utf-8"))
    assert meta["filter_spec"] == {"dataset": "bilibili", "vote_year": 2025}
    assert meta["params"]["filter_spec"] == {"dataset": "bilibili", "vote_year": 2025}

    # 无筛选的星云 filter_json 为 NULL
    plain = build_map(catalog, config)
    assert catalog.get_map(plain.version)["filter_json"] is None


def test_build_map_excludes_duplicates_by_default(config: Config, catalog: Catalog):
    groups = _seed_mixed_works(catalog, config)
    catalog.link_works(
        groups["bilibili"][0], groups["local"][0], method="title+duration", confidence=0.99
    )
    result = build_map(catalog, config)
    assert result.n_works == len(groups["all"]) - 1
    assert groups["local"][0] not in _read_coords(result)["work_id"].tolist()

    included = build_map(catalog, config, filter_spec={"include_duplicates": True})
    assert included.n_works == len(groups["all"])


def test_build_map_n_neighbors_override(config: Config, catalog: Catalog):
    _seed_mixed_works(catalog, config)
    result = build_map(catalog, config, n_neighbors=4)
    params = json.loads(catalog.get_map(result.version)["params_json"])
    assert params["n_neighbors"] == 4
    model_2d = joblib.load(result.model_2d_path)
    assert model_2d.get_params()["n_neighbors"] == 4

    # 覆盖值超过样本数时自动降为 n-1
    capped = build_map(catalog, config, n_neighbors=100)
    params = json.loads(catalog.get_map(capped.version)["params_json"])
    assert params["n_neighbors"] == capped.n_works - 1

    with pytest.raises(MapError, match="n_neighbors 必须 >= 2"):
        build_map(catalog, config, n_neighbors=1)


def test_build_map_variants(config: Config, catalog: Catalog):
    _seed_mixed_works(catalog, config)
    results = build_map_variants(catalog, config, n_neighbors_list=(3, 5, 3))
    assert [result.version for result in results] == ["map_v001", "map_v002"]
    neighbors = [
        json.loads(catalog.get_map(result.version)["params_json"])["n_neighbors"]
        for result in results
    ]
    assert neighbors == [3, 5]
    notes = [catalog.get_map(result.version)["note"] for result in results]
    assert all("n_neighbors=" in note for note in notes)

    with pytest.raises(MapError, match="不能为空"):
        build_map_variants(catalog, config, n_neighbors_list=())
    with pytest.raises(MapError, match="n_neighbors 必须 >= 2"):
        build_map_variants(catalog, config, n_neighbors_list=(1,))


def test_transform_new_respects_map_filter(config: Config, catalog: Catalog):
    groups = _seed_mixed_works(catalog, config)
    result = build_map(catalog, config, filter_spec={"dataset": "bilibili"})
    assert result.n_works == len(groups["bilibili"])

    # 本地新作品不满足筛选条件，不应被 transform 进来
    local_new = _add_work(catalog, config, 40, source="local")
    assert transform_new(catalog, config, map_version=result.version) == 0
    assert local_new not in _read_coords(result)["work_id"].tolist()

    # Bilibili 新作品满足筛选条件
    bilibili_new = _add_work(catalog, config, 41)
    assert transform_new(catalog, config, map_version=result.version) == 1
    assert bilibili_new in _read_coords(result)["work_id"].tolist()


def test_cli_build_with_filter_and_neighbors(config: Config, catalog: Catalog, capsys):
    groups = _seed_mixed_works(catalog, config)
    code = main(
        [
            "build",
            "--config",
            str(config.path),
            "--dataset",
            "local",
            "--vote-year",
            "2025",
            "--n-neighbors",
            "5",
            "--note",
            "本地筛选",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "筛选条件" in out
    assert "来源=本地" in out
    assert "不可直接比较" in out

    map_row = catalog.latest_map()
    assert json.loads(map_row["filter_json"]) == {"dataset": "local", "vote_year": 2025}
    # 4 部作品时 n_neighbors 会被截断为 min(5, n-1)=3，并如实记录
    assert json.loads(map_row["params_json"])["n_neighbors"] == 3
    coords = pd.read_parquet(config.paths.maps / map_row["version"] / "coords.parquet")
    assert set(coords["work_id"].tolist()) == set(groups["local"])


def test_cli_variants(config: Config, catalog: Catalog, capsys):
    _seed_mixed_works(catalog, config)
    code = main(
        [
            "variants",
            "--config",
            str(config.path),
            "--n-neighbors",
            "3",
            "5",
            "--note-prefix",
            "对比",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "2 个星云变体" in out
    maps = catalog.list_maps()
    assert len(maps) == 2
    assert [json.loads(map_row["params_json"])["n_neighbors"] for map_row in maps] == [3, 5]

    listed = main(["list", "--config", str(config.path)])
    assert listed == 0
    out = capsys.readouterr().out
    assert "n_neighbors=3" in out
    assert "筛选=" in out


def test_cli_filter_error_returns_nonzero(config: Config, catalog: Catalog, capsys):
    _seed_mixed_works(catalog, config)
    code = main(["build", "--config", str(config.path), "--vote-year", "2023"])
    assert code == 1
    captured = capsys.readouterr()
    assert "错误" in captured.err
    assert "筛选后作品数量不足" in captured.err


# --------------------------------------------------------------------------- v0.4 表征空间
def _register_derived_representation(
    catalog: Catalog,
    config: Config,
    *,
    name: str = "full_mean_v1",
    sequence_version: str = "full-seq-v1",
    kind: str = "full_mean",
    work_ids: list[int],
    seed: int = 11,
    dim: int = 8,
) -> int:
    """登记一个 synthetic 派生表征（joint 向量），返回 representation_id。"""

    from nebula import representations

    _ensure_embedding_version(catalog, config, sequence_version)
    rep_id = catalog.register_representation(
        name,
        sequence_version=sequence_version,
        kind=kind,
        params={"synthetic": True, "dim": dim},
    )
    rep = catalog.get_representation(rep_id)
    assert rep is not None
    rng = np.random.default_rng(seed)
    for work_id in work_ids:
        vector = rng.normal(size=dim).astype(np.float32)
        vector = (vector / np.linalg.norm(vector)).astype(np.float32)
        representations.save_representation_vectors(
            catalog, config, rep, work_id, {"joint": vector}
        )
    catalog.set_representation_status(rep_id, "ready", dim=dim)
    return rep_id


def test_build_map_representation_and_isolation(config: Config, catalog: Catalog):
    from nebula import representations

    groups = _seed_mixed_works(catalog, config)
    derived_ids = groups["all"][:5]
    rep_id = _register_derived_representation(
        catalog, config, work_ids=derived_ids, seed=3
    )

    derived = build_map(
        catalog,
        config,
        representation=representations.FULL_MEAN_V1,
        sequence_version="full-seq-v1",
        note="full mean 地图",
    )
    assert derived.representation_id == rep_id
    assert derived.representation_name == representations.FULL_MEAN_V1
    assert derived.embedding_version == "full-seq-v1"
    assert derived.n_works == len(derived_ids)
    derived_frame = _read_coords(derived)
    assert set(derived_frame["work_id"].tolist()) == set(derived_ids)

    map_row = catalog.get_map(derived.version)
    assert map_row["representation_id"] == rep_id
    assert map_row["embedding_version"] == "full-seq-v1"
    params = json.loads(map_row["params_json"])
    assert params["representation"]["name"] == representations.FULL_MEAN_V1
    assert params["representation"]["id"] == rep_id
    meta = json.loads((derived.coords_path.parent / "meta.json").read_text(encoding="utf-8"))
    assert meta["representation"]["name"] == representations.FULL_MEAN_V1
    assert meta["representation"]["sequence_version"] == "full-seq-v1"

    # 旧行为：不指定表征时按当前 sampling.mode（默认 legacy）建图，且使用旧序列版本
    legacy = build_map(catalog, config, note="legacy 地图")
    assert legacy.embedding_version == config.legacy_embedding_version
    assert legacy.representation_id is not None
    assert legacy.representation_id != rep_id
    legacy_frame = _read_coords(legacy)
    assert set(legacy_frame["work_id"].tolist()) == set(groups["all"])
    assert legacy_frame["work_id"].tolist() != derived_frame["work_id"].tolist()

    # 数据库层表征过滤：不同表征的地图互不混淆
    assert [row["version"] for row in catalog.list_maps(representation_id=rep_id)] == [
        derived.version
    ]
    latest = catalog.latest_map(representation_id=rep_id)
    assert latest is not None and latest["version"] == derived.version


def test_build_map_representation_filter_spec(config: Config, catalog: Catalog):
    from nebula import representations

    groups = _seed_mixed_works(catalog, config)
    rep_id = _register_derived_representation(
        catalog, config, work_ids=groups["all"], seed=9
    )
    result = build_map(
        catalog,
        config,
        representation=representations.FULL_MEAN_V1,
        sequence_version="full-seq-v1",
        filter_spec={"dataset": "bilibili"},
    )
    frame = _read_coords(result)
    assert set(frame["work_id"].tolist()) == set(groups["bilibili"])
    assert result.n_works == len(groups["bilibili"])
    map_row = catalog.get_map(result.version)
    assert json.loads(map_row["filter_json"]) == {"dataset": "bilibili"}
    assert map_row["representation_id"] == rep_id

    # 派生表征默认排除 duplicate，include_duplicates 明确不支持
    with pytest.raises(MapError, match="include_duplicates"):
        build_map(
            catalog,
            config,
            representation=representations.FULL_MEAN_V1,
            sequence_version="full-seq-v1",
            filter_spec={"include_duplicates": True},
        )


def test_build_map_unknown_representation_raises(config: Config, catalog: Catalog):
    _seed_mixed_works(catalog, config)
    from nebula import representations

    with pytest.raises(MapError, match="无法解析表征"):
        build_map(
            catalog,
            config,
            representation=representations.TEMPORAL_STATS_V1,
            sequence_version="full-seq-v1",
        )
    with pytest.raises(MapError, match="表征不存在"):
        build_map(catalog, config, representation_id=99999)


def test_build_map_variants_representation(config: Config, catalog: Catalog):
    from nebula import representations

    groups = _seed_mixed_works(catalog, config)
    rep_id = _register_derived_representation(
        catalog, config, work_ids=groups["all"], seed=5
    )
    results = build_map_variants(
        catalog,
        config,
        n_neighbors_list=(3, 5),
        representation=representations.FULL_MEAN_V1,
        sequence_version="full-seq-v1",
    )
    assert len(results) == 2
    for result in results:
        assert result.representation_id == rep_id
        map_row = catalog.get_map(result.version)
        assert map_row["representation_id"] == rep_id
        params = json.loads(map_row["params_json"])
        assert params["representation"]["name"] == representations.FULL_MEAN_V1
    assert [
        json.loads(catalog.get_map(result.version)["params_json"])["n_neighbors"]
        for result in results
    ] == [3, 5]


def test_transform_new_representation(config: Config, catalog: Catalog):
    from nebula import representations

    groups = _seed_mixed_works(catalog, config)
    rep_id = _register_derived_representation(
        catalog, config, work_ids=groups["all"][:5], seed=7
    )
    result = build_map(
        catalog,
        config,
        representation=representations.FULL_MEAN_V1,
        sequence_version="full-seq-v1",
    )
    old_frame = _read_coords(result).set_index("work_id")

    new_work_id = _add_work(catalog, config, 50, source="local")
    rep = catalog.get_representation(rep_id)
    assert rep is not None
    representations.save_representation_vectors(
        catalog,
        config,
        rep,
        new_work_id,
        {"joint": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)},
    )
    added = transform_new(catalog, config, map_version=result.version)
    assert added == 1
    new_frame = _read_coords(result).set_index("work_id")
    assert new_work_id in new_frame.index
    for work_id in old_frame.index:
        for column in ("x2", "y2", "x3", "y3", "z3"):
            assert new_frame.loc[work_id, column] == old_frame.loc[work_id, column]
    map_row = catalog.get_map(result.version)
    assert map_row["representation_id"] == rep_id
    assert catalog.get_representation(rep_id)["id"] == rep_id


def test_cli_representation_build_and_list(config: Config, catalog: Catalog, capsys):
    from nebula import representations

    groups = _seed_mixed_works(catalog, config)
    rep_id = _register_derived_representation(
        catalog, config, work_ids=groups["all"], seed=13
    )
    code = main(
        [
            "build",
            "--config",
            str(config.path),
            "--representation",
            "full_mean_v1",
            "--sequence-version",
            "full-seq-v1",
            "--note",
            "表征 CLI",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert representations.REPRESENTATION_LABELS[representations.FULL_MEAN_V1] in out
    map_row = catalog.latest_map(representation_id=rep_id)
    assert map_row is not None
    assert map_row["embedding_version"] == "full-seq-v1"

    assert main(["list", "--config", str(config.path)]) == 0
    out = capsys.readouterr().out
    assert representations.REPRESENTATION_LABELS[representations.FULL_MEAN_V1] in out
    assert "全片平均" in out

    assert main(["variants", "--config", str(config.path), "--representation", "full_mean_v1",
                 "--sequence-version", "full-seq-v1", "--n-neighbors", "3"]) == 0
    out = capsys.readouterr().out
    assert "星云变体" in out

    assert main(["build", "--config", str(config.path), "--representation", "not_a_rep"]) == 1
    captured = capsys.readouterr()
    assert "错误" in captured.err
    assert "无法解析表征" in captured.err


def test_cli_legacy_embedding_version_alias(config: Config, catalog: Catalog, capsys):
    groups = _seed_mixed_works(catalog, config)
    code = main(
        [
            "build",
            "--config",
            str(config.path),
            "--embedding-version",
            config.legacy_embedding_version,
        ]
    )
    assert code == 0
    capsys.readouterr()
    map_row = catalog.latest_map()
    assert map_row is not None
    assert map_row["embedding_version"] == config.legacy_embedding_version
    rep = catalog.get_representation(int(map_row["representation_id"]))
    assert rep is not None
    assert rep["name"] == "legacy_5clip_mean"
    assert len(groups["all"]) == 8
