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
    clip_heatmap_data,
    clip_pair_rows,
    external_url,
    filter_frame,
    hover_text,
    hub_rows,
    load_event_timeline,
    load_nebula_frame,
    load_representation_options,
    map_matches_representation,
    neighbor_rows,
    search_frame,
)
from nebula.config import Config
from nebula.db import Catalog
from nebula.errors import MapError, NebulaError
from nebula.similarity import load_similarity_index


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


def _add_clip_embeddings(
    catalog: Catalog,
    config: Config,
    work_ids: list[int],
    *,
    seed: int = 100,
    n_clips: int = 5,
    dim: int = 8,
) -> None:
    """为作品登记三种片段级向量（堆叠 npy）与 clips 表起始时间。"""

    version = config.embedding_version
    rng = np.random.default_rng(seed)
    for work_id in work_ids:
        for kind in ("clip_joint", "clip_audio", "clip_video"):
            matrix = rng.normal(size=(n_clips, dim)).astype(np.float32)
            path = config.paths.embeddings / version / str(work_id) / f"{kind}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, matrix)
            catalog.register_embedding(
                work_id,
                version=version,
                kind=kind,
                path=config.data_relpath(path),
                dim=dim,
                n_clips=n_clips,
            )
        catalog.replace_clips(
            work_id,
            version=version,
            clips=[(idx, float(idx) * 10.0, 8.0) for idx in range(n_clips)],
        )


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
    representation_id: int | None = None,
    embedding_version: str | None = None,
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
        embedding_version=embedding_version or config.embedding_version,
        params={"metric": "cosine", "random_state": 42},
        coords=coords,
        coords_path=f"maps/{version}/coords.parquet",
        model_2d_path=f"maps/{version}/map_2d.joblib",
        model_3d_path=f"maps/{version}/map_3d.joblib",
        note=note,
        representation_id=representation_id,
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
        # 迭代 P1：解释字段
        assert 0.0 <= row["pair_percentile"] <= 100.0
        assert isinstance(row["is_mutual"], bool)
        assert row["pair_percentile"] == pytest.approx(
            index.pair_percentile(work_ids[1], row["work_id"]), abs=1e-6
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


# --------------------------------------------------------------------------- P1 枢纽作品
def test_hub_rows(seeded_with_vectors):
    config, catalog, work_ids, index = seeded_with_vectors
    _, frame = load_nebula_frame(catalog, config)

    rows = hub_rows(frame, index, kind="work_joint", k=5, n=3)
    assert len(rows) == 3
    assert [row["rank"] for row in rows] == [1, 2, 3]
    counts = [row["count"] for row in rows]
    assert counts == sorted(counts, reverse=True)
    by_id = frame.set_index("work_id")
    for row in rows:
        assert row["title"] == by_id.loc[row["work_id"], "title"]
        assert row["in_map"] is True
    # hubness 每部作品 k=5 时总票数 = 作品数 × 5（作品数 > k）
    assert sum(index.hubness(k=5).values()) == len(work_ids) * 5

    # n 大于作品数时不会报错
    wide = hub_rows(frame, index, kind="work_joint", k=2, n=99)
    assert len(wide) == len(work_ids)


# --------------------------------------------------------------------------- P2 片段级展示数据
@pytest.fixture
def seeded_with_clips(seeded):
    """作品级 + 片段级向量都注册好之后再构造索引。"""

    config, catalog, work_ids = seeded
    _add_embeddings(catalog, config, work_ids)
    _add_clip_embeddings(catalog, config, work_ids)
    index = load_similarity_index(catalog, config)
    try:
        yield config, catalog, work_ids, index
    finally:
        index.close()


def test_clip_heatmap_data(seeded_with_clips):
    config, catalog, work_ids, index = seeded_with_clips
    _, frame = load_nebula_frame(catalog, config)

    data = clip_heatmap_data(index, frame, work_ids[1], work_ids[2], "clip_joint")
    assert data["a_work_id"] == work_ids[1]
    assert data["b_work_id"] == work_ids[2]
    assert data["kind"] == "clip_joint"
    assert data["n_a"] == 5 and data["n_b"] == 5
    assert len(data["matrix"]) == 5
    assert all(len(row) == 5 for row in data["matrix"])
    assert all(-1.0 <= float(value) <= 1.0 for row in data["matrix"] for value in row)
    assert len(data["row_labels"]) == 5 and len(data["col_labels"]) == 5
    assert all(label.startswith("片段 ") for label in data["row_labels"])
    assert "0.0s" in data["row_labels"][0]
    by_id = frame.set_index("work_id")
    assert data["a_title"] == by_id.loc[work_ids[1], "title"]
    assert data["b_title"] == by_id.loc[work_ids[2], "title"]

    with pytest.raises(NebulaError):
        clip_heatmap_data(index, frame, 99999, work_ids[2], "clip_joint")


def test_clip_pair_rows(seeded_with_clips):
    config, catalog, work_ids, index = seeded_with_clips
    _, frame = load_nebula_frame(catalog, config)

    rows = clip_pair_rows(index, frame, work_ids[1], work_ids[2], "clip_joint", 5)
    assert len(rows) == 5
    assert [row["rank"] for row in rows] == [1, 2, 3, 4, 5]
    similarities = [row["similarity"] for row in rows]
    assert similarities == sorted(similarities, reverse=True)
    by_id = frame.set_index("work_id")
    for row in rows:
        assert set(row) == {
            "rank",
            "a_title",
            "a_idx",
            "a_start_s",
            "b_title",
            "b_idx",
            "b_start_s",
            "similarity",
        }
        assert row["a_title"] == by_id.loc[work_ids[1], "title"]
        assert row["b_title"] == by_id.loc[work_ids[2], "title"]
        assert row["a_start_s"] == pytest.approx(row["a_idx"] * 10.0)
        assert row["b_start_s"] == pytest.approx(row["b_idx"] * 10.0)
        assert row["similarity"] == pytest.approx(
            index.clip_similarity(work_ids[1], work_ids[2])[row["a_idx"], row["b_idx"]],
            abs=1e-6,
        )


# --------------------------------------------------------------------------- v0.4 表征信息与时间轴
def _create_representation(
    catalog: Catalog,
    config: Config,
    work_ids: list[int],
    *,
    name: str = "full_mean_v1",
    sequence_version: str = "full-seq-v1",
    kind: str = "full_mean",
    seed: int = 21,
    dim: int = 8,
) -> int:
    from nebula import representations

    _ensure_embedding_version(catalog, config, sequence_version)
    rep_id = catalog.register_representation(
        name,
        sequence_version=sequence_version,
        kind=kind,
        params={"synthetic": True},
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


def test_load_nebula_frame_representation_info(seeded):
    config, catalog, work_ids = seeded
    rep_id = _create_representation(catalog, config, work_ids[:3])
    _seed_map(
        catalog,
        config,
        work_ids=work_ids[:3],
        version="map_v002",
        representation_id=rep_id,
        embedding_version="full-seq-v1",
        note="full mean 地图",
    )

    map_info, frame = load_nebula_frame(catalog, config)
    assert map_info["version"] == "map_v002"
    assert map_info["representation_id"] == rep_id
    assert map_info["representation_name"] == "full_mean_v1"
    assert map_info["representation_label"] == "Full Mean（全片平均）"
    assert map_info["representation_sequence_version"] == "full-seq-v1"
    assert map_info["representation_kind"] == "full_mean"
    assert len(frame) == 3

    # 旧地图（representation_id=NULL）按 legacy 展示
    legacy_info, legacy_frame = load_nebula_frame(
        catalog, config, map_version="map_v001"
    )
    assert legacy_info["representation_id"] is None
    assert legacy_info["representation_name"] == "legacy_5clip_mean"
    assert legacy_info["representation_label"] == "Legacy（旧 5 片段平均）"
    assert legacy_info["representation_sequence_version"] == config.embedding_version
    assert len(legacy_frame) == len(work_ids)


def test_load_representation_options(seeded):
    from nebula import representations

    config, catalog, work_ids = seeded
    rep_id = _create_representation(catalog, config, work_ids[:3])
    _seed_map(
        catalog,
        config,
        work_ids=work_ids[:3],
        version="map_v002",
        representation_id=rep_id,
        embedding_version="full-seq-v1",
    )
    # 再登记一个没有任何地图的表征
    _ensure_embedding_version(catalog, config, "temporal-seq-v1")
    catalog.register_representation(
        "temporal_stats_v1",
        sequence_version="temporal-seq-v1",
        kind="temporal_stats",
        params={"synthetic": True},
    )
    temporal_row = catalog.get_representation(
        name="temporal_stats_v1", sequence_version="temporal-seq-v1"
    )
    assert temporal_row is not None
    catalog.set_representation_status(int(temporal_row["id"]), "ready", dim=8)

    options = load_representation_options(catalog, config)
    names = [str(option["representation_name"]) for option in options]
    assert names[:2] == ["legacy_5clip_mean", "full_mean_v1"]
    assert "temporal_stats_v1" in names
    by_name = {str(option["representation_name"]): option for option in options}
    legacy_option = by_name["legacy_5clip_mean"]
    assert legacy_option["map_version"] == "map_v001"
    assert legacy_option["n_maps"] == 1
    assert legacy_option["label"] == "Legacy（旧 5 片段平均）"
    full_option = by_name["full_mean_v1"]
    assert full_option["representation_id"] == rep_id
    assert full_option["map_version"] == "map_v002"
    assert full_option["sequence_version"] == "full-seq-v1"
    assert by_name["temporal_stats_v1"]["map_version"] is None
    assert by_name["temporal_stats_v1"]["n_maps"] == 0

    maps = {str(row["version"]): row for row in catalog.list_maps()}
    assert map_matches_representation(maps["map_v001"], legacy_option) is True
    assert map_matches_representation(maps["map_v001"], full_option) is False
    assert map_matches_representation(maps["map_v002"], full_option) is True
    assert map_matches_representation(maps["map_v002"], legacy_option) is False
    assert (
        representations.representation_label("full_mean_v1") == "Full Mean（全片平均）"
    )


def test_load_event_timeline(seeded, monkeypatch):
    from types import SimpleNamespace

    config, catalog, work_ids = seeded
    work_id = work_ids[1]  # bilibili 作品，带 duration
    sequence_version = config.embedding_version
    catalog.update_work_metadata(work_id, duration_s=40.0)
    catalog.replace_clips(
        work_id,
        version=sequence_version,
        clips=[(idx, float(idx) * 8.0, 8.0, 8.0, 0.2) for idx in range(5)],
    )
    rng = np.random.default_rng(3)
    stub = SimpleNamespace(
        audio=rng.normal(size=(100, 6)).astype(np.float32),
        video=rng.normal(size=(30, 6)).astype(np.float32),
        av=rng.normal(size=(100, 6)).astype(np.float32),
        window_features=rng.normal(size=(5, 32)).astype(np.float32),
        audio_rate_hz=100.0,
        video_fps=30,
        video_fps_effective=30.0,
        duration_s=40.0,
        version="events_v1",
    )
    monkeypatch.setattr("nebula.appdata._load_events_bundle", lambda work_dir: stub)

    timeline = load_event_timeline(catalog, config, sequence_version, work_id)
    assert timeline is not None
    assert timeline["work_id"] == work_id
    assert timeline["sequence_version"] == sequence_version
    assert timeline["event_version"] == "events_v1"
    assert timeline["duration_s"] == pytest.approx(40.0)
    assert timeline["n_clips"] == 5
    assert timeline["coverage_ratio"] == pytest.approx(1.0)
    assert timeline["coverage_s"] == pytest.approx(40.0)
    boundaries = timeline["window_boundaries"]
    assert len(boundaries) == 5
    assert boundaries[0]["start_s"] == pytest.approx(0.0)
    assert boundaries[-1]["end_s"] == pytest.approx(40.0)
    for boundary in boundaries:
        assert boundary["duration_s"] == pytest.approx(8.0)
        assert boundary["valid_duration_s"] == pytest.approx(8.0)
        assert boundary["weight"] == pytest.approx(0.2)
    series = timeline["series"]
    assert set(series) == {"audio", "video", "av", "window"}
    assert len(series["audio"]["time_s"]) == 100
    assert len(series["video"]["values"]) == 30
    assert len(series["av"]["values"]) == 100
    assert len(series["window"]["time_s"]) == 5
    for key in ("audio", "video", "av", "window"):
        values = series[key]["values"]
        assert all(0.0 <= float(value) <= 1.0 for value in values)
    assert timeline["feature_names"]["audio"]
    assert timeline["feature_names"]["video"]
    assert timeline["audio_rate_hz"] == pytest.approx(100.0)
    assert timeline["video_fps_effective"] == pytest.approx(30.0)

    # 事件文件缺失 -> None
    monkeypatch.setattr("nebula.appdata._load_events_bundle", lambda work_dir: None)
    assert load_event_timeline(catalog, config, sequence_version, work_id) is None
    with pytest.raises(MapError, match="作品不存在"):
        load_event_timeline(catalog, config, sequence_version, 99999)
