"""nebula.appdata 纯数据层单元测试。

覆盖：读取星云 DataFrame、榜单汇总、染色字段/密度/连续/离散色、
文件搜索、Bilibili 链接与 hover 文本。全部使用临时 SQLite 合成数据。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nebula.appdata import (
    COLOR_FIELDS,
    ColorSpec,
    apply_color,
    available_color_fields,
    external_url,
    filter_frame,
    hover_text,
    load_nebula_frame,
    neighbor_rows,
    search_frame,
)
from nebula.config import Config
from nebula.db import Catalog
from nebula.errors import MapError, NebulaError
from nebula.similarity import load_similarity_index


# --------------------------------------------------------------------------- 合成数据
def _ensure_embedding_version(catalog: Catalog, config: Config) -> str:
    catalog.register_embedding_version(
        config.embedding_version,
        model_checkpoint=config.model.checkpoint,
        model_revision=config.model.revision,
        frames=config.model.frames,
        clip_seconds=config.media.clip_seconds,
        clips_per_work=config.media.clips_per_work,
        audio_sample_rate=config.media.audio_sample_rate,
        params={},
    )
    return config.embedding_version


def _seed_works(catalog: Catalog, config: Config, count: int = 6) -> list[int]:
    """登记混合来源的合成作品（约 1/3 本地作品，其余 Bilibili）。"""

    _ensure_embedding_version(catalog, config)
    work_ids: list[int] = []
    for index in range(count):
        if index % 3 == 0:
            work_id, _ = catalog.upsert_local_work(
                f"clips/本地作品 {index}.mp4", title=f"本地作品 {index}"
            )
        else:
            kwargs: dict = {
                "title": f"合成作品 {index}",
                "author_uid": f"100{index}",
                "author_name": f"作者{index}",
                "duration_s": 60.0 + index * 10,
                "publish_date": f"2025-0{1 + index % 9}-0{1 + index % 9}",
                "part": 2 if index == 5 else 1,
            }
            if index == 2:  # 覆盖缺失值分支
                kwargs["duration_s"] = None
                kwargs["publish_date"] = None
                kwargs["author_uid"] = None
                kwargs["author_name"] = None
            work_id, _ = catalog.upsert_bilibili_work(f"BV1xx411c7{index:02d}", **kwargs)
        work_ids.append(work_id)
    return work_ids


def _add_embeddings(
    catalog: Catalog, config: Config, work_ids: list[int], *, dim: int = 8, seed: int = 0
) -> str:
    """为作品登记三种作品级向量（合成 npy），供近邻相关测试使用。"""

    version = _ensure_embedding_version(catalog, config)
    rng = np.random.default_rng(seed)
    for work_id in work_ids:
        for kind in ("work_joint", "work_audio", "work_video"):
            vector = rng.normal(size=dim).astype(np.float32)
            path = config.paths.embeddings / version / str(work_id) / f"{kind}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vector)
            catalog.register_embedding(
                work_id,
                version=version,
                kind=kind,
                path=config.data_relpath(path),
                dim=dim,
            )
    return version


def _seed_appearances(catalog: Catalog, work_ids: list[int]) -> None:
    votes_2025 = catalog.ensure_list(
        "votes_2025", name="2025年音MAD选票", kind="votes", vote_year=2025
    )
    weekly = catalog.ensure_list("weekly_001", name="第1期周报", kind="weekly")
    for index, work_id in enumerate(work_ids):
        catalog.add_appearance(work_id, votes_2025, rank=index + 1, votes=100 - index * 10)
        if index % 2 == 0:
            catalog.add_appearance(work_id, weekly)
    votes_2024 = catalog.ensure_list(
        "votes_2024", name="2024年音MAD选票", kind="votes", vote_year=2024
    )
    catalog.add_appearance(work_ids[3], votes_2024, rank=2, votes=200)


def _seed_map(
    catalog: Catalog,
    config: Config,
    *,
    work_ids: list[int],
    version: str = "map_v001",
    note: str = "测试星云",
) -> None:
    coords = [
        (
            work_id,
            float(index),
            float(index * 2),
            float(-index),
            float(index * 3),
            float(-index * 2),
        )
        for index, work_id in enumerate(work_ids)
    ]
    catalog.create_map_run(
        version,
        embedding_version=config.embedding_version,
        params={"metric": "cosine", "random_state": 42},
        coords=coords,
        coords_path=f"maps/{version}/coords.parquet",
        model_2d_path=f"maps/{version}/map_2d.joblib",
        model_3d_path=f"maps/{version}/map_3d.joblib",
        note=note,
    )


@pytest.fixture
def seeded(config: Config, catalog: Catalog) -> tuple[Config, Catalog, list[int]]:
    work_ids = _seed_works(catalog, config)
    _seed_appearances(catalog, work_ids)
    _seed_map(catalog, config, work_ids=work_ids)
    return config, catalog, work_ids


@pytest.fixture
def seeded_with_vectors(seeded):
    """在 seeded 基础上补充三种模态向量，并构造近邻索引。"""

    config, catalog, work_ids = seeded
    _add_embeddings(catalog, config, work_ids)
    index = load_similarity_index(catalog, config)
    try:
        yield config, catalog, work_ids, index
    finally:
        index.close()


# --------------------------------------------------------------------------- 读取星云
def test_load_nebula_frame_basic(seeded):
    config, catalog, work_ids = seeded
    map_info, frame = load_nebula_frame(catalog, config)

    assert map_info["version"] == "map_v001"
    assert map_info["embedding_version"] == config.embedding_version
    assert map_info["params"]["metric"] == "cosine"
    assert len(frame) == len(work_ids)

    required = {
        "work_id",
        "title",
        "work_key",
        "source",
        "bvid",
        "url",
        "author_uid",
        "author_name",
        "duration_s",
        "publish_date",
        "local_path",
        "file_name",
        "status",
        "x",
        "y",
        "z",
        "x2",
        "y2",
        "x3",
        "y3",
        "z3",
        "rank",
        "votes",
        "vote_year",
        "vote_years",
        "list_keys",
        "list_names",
        "local_file",
        "is_canonical",
        "search_text",
        "hover_text",
    }
    assert required <= set(frame.columns)
    assert frame["work_id"].tolist() == work_ids

    local_row = frame.iloc[0]
    assert local_row["source"] == "local"
    assert local_row["file_name"] == "本地作品 0.mp4"
    assert local_row["local_file"] == "本地作品 0.mp4"
    assert local_row["local_path"].startswith("clips/")
    assert local_row["is_canonical"] is True or bool(local_row["is_canonical"]) is True
    assert external_url(local_row) is None

    bilibili_row = frame.iloc[1]
    assert bilibili_row["source"] == "bilibili"
    assert bilibili_row["file_name"] == "BV1xx411c701.mp4"
    assert bilibili_row["local_file"] == ""
    assert external_url(bilibili_row) == "https://www.bilibili.com/video/BV1xx411c701"

    # x/y 是 2D 默认列，z 取 3D 的 z3
    assert frame["x"].equals(frame["x2"])
    assert frame["y"].equals(frame["y2"])
    assert frame["z"].equals(frame["z3"])
    assert frame["x2"].notna().all()
    assert frame["z3"].notna().all()

    assert "合成作品 1" in frame.iloc[1]["hover_text"]


def test_load_nebula_frame_defaults_to_latest_map(seeded):
    config, catalog, work_ids = seeded
    _seed_map(catalog, config, work_ids=work_ids[:3], version="map_v002", note="第二版")

    map_info, frame = load_nebula_frame(catalog, config)
    assert map_info["version"] == "map_v002"
    assert len(frame) == 3


def test_load_nebula_frame_errors(config: Config, catalog: Catalog):
    with pytest.raises(MapError, match="还没有可用的星云版本"):
        load_nebula_frame(catalog, config)

    _add_works = _seed_works(catalog, config, 3)
    _seed_map(catalog, config, work_ids=_add_works)
    with pytest.raises(MapError, match="星云版本不存在"):
        load_nebula_frame(catalog, config, map_version="map_v999")


def test_load_nebula_frame_without_appearances(config: Config, catalog: Catalog):
    _ensure_embedding_version(catalog, config)
    work_id, _ = catalog.upsert_local_work("clips/无榜单.mp4", title="无榜单作品")
    catalog.update_work_metadata(work_id, duration_s=12.5, publish_date="2025-01-01")
    _seed_map(catalog, config, work_ids=[work_id])

    _, frame = load_nebula_frame(catalog, config)
    row = frame.iloc[0]
    assert pd.isna(row["rank"])
    assert pd.isna(row["votes"])
    assert row["vote_year"] is None
    assert row["vote_years"] == []
    assert row["list_names"] == ""


# --------------------------------------------------------------------------- 榜单汇总
def test_appearance_summary(seeded):
    config, catalog, work_ids = seeded
    _, frame = load_nebula_frame(catalog, config)
    by_id = frame.set_index("work_id")

    first = by_id.loc[work_ids[0]]
    assert first["rank"] == 1.0
    assert first["votes"] == 100.0
    assert first["vote_year"] == 2025
    assert first["vote_years"] == [2025]
    assert "2025年音MAD选票" in first["list_names"]
    assert "第1期周报" in first["list_names"]
    assert sorted(first["list_keys"]) == ["votes_2025", "weekly_001"]

    third = by_id.loc[work_ids[3]]
    # 2024 年榜单中排名更好（第 2 名），应选它作为展示记录
    assert third["rank"] == 2.0
    assert third["votes"] == 200.0
    assert third["vote_year"] == 2024
    assert third["vote_years"] == [2024, 2025]
    assert "2024年音MAD选票" in third["list_names"]


# --------------------------------------------------------------------------- 染色字段
def test_available_color_fields(seeded):
    config, catalog, work_ids = seeded
    _, frame = load_nebula_frame(catalog, config)
    fields = available_color_fields(frame)
    field_ids = [field_id for field_id, _ in fields]
    assert field_ids == [
        "none",
        "density_knn",
        "duration_s",
        "publish_date",
        "rank",
        "votes",
        "author_uid",
        "vote_year",
        "source",
    ]
    assert dict(fields)["none"] == "不染色/密度"
    assert dict(fields)["density_knn"] == "高维近邻密度"
    # 没有近邻索引时不提供高维密度
    without_similarity = available_color_fields(frame, has_similarity=False)
    assert "density_knn" not in [field_id for field_id, _ in without_similarity]


def test_available_color_fields_without_optional_data(config: Config, catalog: Catalog):
    _ensure_embedding_version(catalog, config)
    work_id, _ = catalog.upsert_local_work("clips/最简.mp4")
    catalog.update_work_metadata(work_id, duration_s=12.5, publish_date="2025-01-01")
    _seed_map(catalog, config, work_ids=[work_id])
    _, frame = load_nebula_frame(catalog, config)

    field_ids = [field_id for field_id, _ in available_color_fields(frame)]
    assert field_ids == ["none", "density_knn", "duration_s", "publish_date", "source"]
    fallback_ids = [
        field_id for field_id, _ in available_color_fields(frame, has_similarity=False)
    ]
    assert fallback_ids == ["none", "duration_s", "publish_date", "source"]


def test_apply_color_density_2d(seeded):
    config, catalog, _ = seeded
    _, frame = load_nebula_frame(catalog, config)
    spec = apply_color(frame, "none", projection="2d", density_grid=4)

    assert isinstance(spec, ColorSpec)
    assert spec.categorical is False
    assert spec.colorscale == "Viridis"
    assert "密度" in spec.title
    assert len(spec.values) == len(frame)
    assert all(value is not None for value in spec.values)
    assert all(float(value) >= 1.0 for value in spec.values)
    assert sum(float(value) for value in spec.values) >= len(frame)


def test_apply_color_density_3d(seeded):
    config, catalog, _ = seeded
    _, frame = load_nebula_frame(catalog, config)
    spec = apply_color(frame, "none", projection="3d", density_grid=4)
    assert "3" in spec.title or "×" in spec.title
    assert spec.colorscale == "Viridis"
    assert len(spec.values) == len(frame)
    assert all(value is not None for value in spec.values)


def test_apply_color_continuous(seeded):
    config, catalog, _ = seeded
    _, frame = load_nebula_frame(catalog, config)

    duration_spec = apply_color(frame, "duration_s")
    assert "时长" in duration_spec.title
    assert duration_spec.values[0] is None  # 本地作品没有时长
    assert duration_spec.values[1] == pytest.approx(70.0)
    assert duration_spec.values[2] is None  # 缺时长的作品

    publish_spec = apply_color(frame, "publish_date")
    assert "发布" in publish_spec.title
    assert publish_spec.values[0] is None
    expected_days = (pd.Timestamp("2025-02-02") - pd.Timestamp("1970-01-01")).days
    assert publish_spec.values[1] == pytest.approx(float(expected_days))
    assert publish_spec.tickvals is not None
    assert "2025" in (publish_spec.ticktext or [])

    rank_spec = apply_color(frame, "rank")
    assert rank_spec.values[0] == pytest.approx(1.0)
    assert isinstance(rank_spec.colorscale, list)
    assert len(rank_spec.colorscale) >= 2
    # 排名数值越小越好：色带被反转，低值应使用原来色带的高端色
    assert rank_spec.colorscale[0] == "#fde725"

    votes_spec = apply_color(frame, "votes")
    assert votes_spec.values[0] == pytest.approx(100.0)
    assert votes_spec.colorscale == "Plasma"


def test_apply_color_categorical(seeded):
    config, catalog, _ = seeded
    _, frame = load_nebula_frame(catalog, config)

    author_spec = apply_color(frame, "author_uid")
    assert author_spec.categorical is True
    assert isinstance(author_spec.colorscale, list)
    assert author_spec.tickvals == list(range(len(author_spec.colorscale)))
    assert author_spec.values[0] is None  # 本地作品没有作者 UID
    assert author_spec.values[1] is not None
    assert author_spec.values[2] is None

    year_spec = apply_color(frame, "vote_year")
    assert year_spec.categorical is True
    assert "2025年" in (year_spec.ticktext or [])
    assert year_spec.values[0] is not None

    source_spec = apply_color(frame, "source")
    assert source_spec.categorical is True
    assert set(source_spec.ticktext or []) == {"Bilibili", "本地"}
    assert len(source_spec.values) == len(frame)
    assert all(value is not None for value in source_spec.values)


def test_apply_color_invalid_arguments(seeded):
    config, catalog, _ = seeded
    _, frame = load_nebula_frame(catalog, config)
    with pytest.raises(MapError, match="未知染色维度"):
        apply_color(frame, "no_such_field")
    with pytest.raises(MapError, match="未知投影方式"):
        apply_color(frame, "duration_s", projection="4d")


# --------------------------------------------------------------------------- 搜索
def test_search_frame(seeded):
    config, catalog, _ = seeded
    _, frame = load_nebula_frame(catalog, config)

    assert len(search_frame(frame, "")) == len(frame)
    assert len(search_frame(frame, "   ")) == len(frame)

    title_result = search_frame(frame, "合成作品 2")
    assert len(title_result) == 1
    assert title_result.iloc[0]["work_id"] == frame.iloc[2]["work_id"]

    assert len(search_frame(frame, "bv1xx411C7")) == 4  # 大小写不敏感
    assert len(search_frame(frame, "本地作品 3.mp4")) == 1  # 按文件名
    assert len(search_frame(frame, "作者4")) == 1  # 按作者
    assert len(search_frame(frame, "不存在的作品xyz")) == 0
    # 过滤结果是副本，不改变原数据
    assert len(frame) == 6


# --------------------------------------------------------------------------- 链接与 hover
def test_external_url_variants():
    assert external_url({"source": "local", "local_path": "a.mp4"}) is None
    assert (
        external_url({"source": "bilibili", "bvid": "BV1xx411c7mD"})
        == "https://www.bilibili.com/video/BV1xx411c7mD"
    )
    assert (
        external_url({"source": "bilibili", "bvid": "BV1xx411c7mD", "part": 2})
        == "https://www.bilibili.com/video/BV1xx411c7mD?p=2"
    )
    assert (
        external_url({"source": "bilibili", "url": "https://www.bilibili.com/video/BV1xx411c7mD"})
        == "https://www.bilibili.com/video/BV1xx411c7mD"
    )
    assert external_url({"source": "bilibili"}) is None


def test_hover_text_escapes_and_handles_missing():
    frame = pd.DataFrame(
        [
            {
                "title": "<b>危险标题</b>",
                "author_name": "作者A",
                "duration_s": 65.0,
                "source": "bilibili",
                "bvid": "BV1xx411c7mD",
                "file_name": "a.mp4",
                "rank": 3.0,
                "votes": 50.0,
                "vote_year": 2025.0,
                "list_names": "2025年音MAD选票",
            },
            {"title": None, "file_name": "只文件名.mp4", "source": "local"},
        ]
    )
    texts = hover_text(frame)
    assert len(texts) == 2
    assert texts[0].startswith("<b>")
    assert "&lt;b&gt;危险标题&lt;/b&gt;" in texts[0]
    assert "作者A" in texts[0]
    assert "1:05" in texts[0]
    assert "Bilibili" in texts[0]
    assert "只文件名.mp4" in texts[1]
    assert "本地" in texts[1]


def test_hover_text_empty_frame():
    assert hover_text(pd.DataFrame()) == []


def test_color_fields_contract_order():
    field_ids = [field_id for field_id, _ in COLOR_FIELDS]
    assert field_ids == [
        "none",
        "density_knn",
        "duration_s",
        "publish_date",
        "rank",
        "votes",
        "author_uid",
        "vote_year",
        "source",
    ]


# --------------------------------------------------------------------------- 身份关联补充列
def test_local_file_from_linked_duplicate(seeded):
    """canonical 的 Bilibili 作品可从关联的本地 duplicate 补充 local_file。"""

    config, catalog, work_ids = seeded
    canonical = work_ids[1]
    duplicate, _ = catalog.upsert_local_work("clips/关联本地.mp4", title="关联本地")
    catalog.link_works(canonical, duplicate, method="title+duration", confidence=0.99)
    _seed_map(catalog, config, work_ids=[canonical, duplicate], version="map_v002")

    _, frame = load_nebula_frame(catalog, config)
    by_id = frame.set_index("work_id")
    assert by_id.loc[canonical, "local_file"] == "关联本地.mp4"
    assert bool(by_id.loc[canonical, "is_canonical"]) is True
    assert by_id.loc[duplicate, "local_file"] == "关联本地.mp4"
    assert bool(by_id.loc[duplicate, "is_canonical"]) is False
    # 搜索结果也能通过关联的本地文件名命中（local_file 参与 search_text）
    assert len(search_frame(frame, "关联本地.mp4")) == 2


def test_filter_frame(seeded):
    config, catalog, work_ids = seeded
    _, frame = load_nebula_frame(catalog, config)

    assert len(filter_frame(frame)) == len(frame)
    assert len(filter_frame(frame, sources=[], list_keys=[], vote_years=[])) == len(frame)

    local = filter_frame(frame, sources=["local"])
    assert len(local) == 2
    assert set(local["source"]) == {"local"}

    weekly = filter_frame(frame, list_keys=["weekly_001"])
    assert len(weekly) == 3
    assert all("weekly_001" in keys for keys in weekly["list_keys"])

    year_2024 = filter_frame(frame, vote_years=[2024])
    assert len(year_2024) == 1
    assert int(year_2024.iloc[0]["work_id"]) == work_ids[3]

    combined = filter_frame(frame, sources=["bilibili"], vote_years=[2025])
    assert len(combined) == 4
    assert set(combined["source"]) == {"bilibili"}

    empty = filter_frame(frame, sources=["不存在的来源"])
    assert len(empty) == 0
    # 原数据不被修改
    assert len(frame) == 6


def test_filter_frame_missing_columns():
    frame = pd.DataFrame([{"work_id": 1, "source": "local"}])
    with pytest.raises(MapError, match="缺少 list_keys"):
        filter_frame(frame, list_keys=["votes_2025"])
    with pytest.raises(MapError, match="缺少 vote_years"):
        filter_frame(frame, vote_years=[2025])


# --------------------------------------------------------------------------- 高维近邻密度
def test_density_knn_color(seeded_with_vectors):
    config, catalog, work_ids, index = seeded_with_vectors
    _, frame = load_nebula_frame(catalog, config)

    spec = apply_color(frame, "density_knn", similarity_index=index)
    assert isinstance(spec, ColorSpec)
    assert spec.categorical is False
    assert "近邻" in spec.title
    assert "稀疏" in spec.title
    assert len(spec.values) == len(frame)
    expected = index.kth_neighbor_distance(k=5, kind="work_joint")
    for work_id, value in zip(frame["work_id"].tolist(), spec.values, strict=True):
        assert value == pytest.approx(expected[int(work_id)], abs=1e-6)
    # 6 部作品都不足 5 个邻居，取最后一个可用邻居；距离均在 [0, 2]
    assert all(0.0 <= float(value) <= 2.0 for value in spec.values)

    with pytest.raises(MapError, match="需要 SimilarityIndex"):
        apply_color(frame, "density_knn")


def test_density_knn_missing_work_is_none(seeded_with_vectors):
    config, catalog, work_ids, index = seeded_with_vectors
    no_vector, _ = catalog.upsert_local_work("clips/没有向量.mp4", title="没有向量")
    _seed_map(catalog, config, work_ids=[*work_ids, no_vector], version="map_v002")
    _, frame = load_nebula_frame(catalog, config)

    spec = apply_color(frame, "density_knn", similarity_index=index)
    position = frame.index[frame["work_id"] == no_vector][0]
    assert spec.values[position] is None


def test_density_knn_invalid_index(seeded):
    config, catalog, _ = seeded
    _, frame = load_nebula_frame(catalog, config)
    with pytest.raises(MapError):
        apply_color(frame, "density_knn")


# --------------------------------------------------------------------------- 近邻展示行
def test_neighbor_rows(seeded_with_vectors):
    config, catalog, work_ids, index = seeded_with_vectors
    _, frame = load_nebula_frame(catalog, config)

    rows = neighbor_rows(frame, index, work_ids[1], "work_joint", 3)
    assert len(rows) == 3
    assert [row["rank"] for row in rows] == [1, 2, 3]
    assert all(row["work_id"] != work_ids[1] for row in rows)
    similarities = [row["similarity"] for row in rows]
    assert similarities == sorted(similarities, reverse=True)
    by_id = frame.set_index("work_id")
    for row in rows:
        assert row["title"] == by_id.loc[row["work_id"], "title"]
        assert row["in_map"] is True
        assert row["similarity"] == pytest.approx(
            index.similarity(work_ids[1], row["work_id"]), abs=1e-6
        )

    # 近邻不在当前 DataFrame 时仍返回占位标题
    partial = frame.iloc[[0]]
    partial_rows = neighbor_rows(partial, index, work_ids[1], "work_joint", 2)
    assert len(partial_rows) == 2
    assert all(row["in_map"] is False for row in partial_rows)
    assert all(row["title"].startswith("作品 ") for row in partial_rows)


def test_neighbor_rows_missing_modality(seeded_with_vectors):
    config, catalog, work_ids, index = seeded_with_vectors
    _, frame = load_nebula_frame(catalog, config)

    with pytest.raises(NebulaError, match="没有 work_audio 向量"):
        neighbor_rows(frame, index, 99999, "work_audio", 5)
    assert len(neighbor_rows(frame, index, work_ids[0], "work_audio", 5)) == 5
