"""nebula.appdata 纯数据层单元测试。

覆盖：读取星云 DataFrame、榜单汇总、染色字段/密度/连续/离散色、
文件搜索、Bilibili 链接与 hover 文本。全部使用临时 SQLite 合成数据。
"""

from __future__ import annotations

import pandas as pd
import pytest

from nebula.appdata import (
    COLOR_FIELDS,
    ColorSpec,
    apply_color,
    available_color_fields,
    external_url,
    hover_text,
    load_nebula_frame,
    search_frame,
)
from nebula.config import Config
from nebula.db import Catalog
from nebula.errors import MapError


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
        "list_names",
        "search_text",
        "hover_text",
    }
    assert required <= set(frame.columns)
    assert frame["work_id"].tolist() == work_ids

    local_row = frame.iloc[0]
    assert local_row["source"] == "local"
    assert local_row["file_name"] == "本地作品 0.mp4"
    assert local_row["local_path"].startswith("clips/")
    assert external_url(local_row) is None

    bilibili_row = frame.iloc[1]
    assert bilibili_row["source"] == "bilibili"
    assert bilibili_row["file_name"] == "BV1xx411c701.mp4"
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
        "duration_s",
        "publish_date",
        "rank",
        "votes",
        "author_uid",
        "vote_year",
        "source",
    ]
    assert dict(fields)["none"] == "不染色/密度"


def test_available_color_fields_without_optional_data(config: Config, catalog: Catalog):
    _ensure_embedding_version(catalog, config)
    work_id, _ = catalog.upsert_local_work("clips/最简.mp4")
    catalog.update_work_metadata(work_id, duration_s=12.5, publish_date="2025-01-01")
    _seed_map(catalog, config, work_ids=[work_id])
    _, frame = load_nebula_frame(catalog, config)

    field_ids = [field_id for field_id, _ in available_color_fields(frame)]
    assert field_ids == ["none", "duration_s", "publish_date", "source"]


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
        "duration_s",
        "publish_date",
        "rank",
        "votes",
        "author_uid",
        "vote_year",
        "source",
    ]
