"""app.py Streamlit 页面测试（streamlit.testing.v1.AppTest）。

用临时配置 + 临时 SQLite 验证：无数据提示、有数据时正常渲染、
2D/3D 切换、染色切换、搜索过滤、手动选择作品详情。
不需要真实向量/模型权重/网络。
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from nebula.config import Config
from nebula.db import Catalog


# --------------------------------------------------------------------------- 合成数据
def _ensure_embedding_version(catalog: Catalog, config: Config) -> None:
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


def _seed_app_data(catalog: Catalog, config: Config, count: int = 6) -> list[int]:
    """登记若干作品、一个选票榜单与一个就绪星云版本。"""

    _ensure_embedding_version(catalog, config)
    work_ids: list[int] = []
    for index in range(count):
        work_id, _ = catalog.upsert_bilibili_work(
            f"BV1xx411c7{index:02d}",
            title=f"合成作品 {index}",
            author_uid=f"100{index}",
            author_name=f"作者{index}",
            duration_s=60.0 + index * 10,
            publish_date="2025-01-02",
        )
        work_ids.append(work_id)
    list_id = catalog.ensure_list(
        "votes_2025", name="2025年音MAD选票", kind="votes", vote_year=2025
    )
    for index, work_id in enumerate(work_ids):
        catalog.add_appearance(work_id, list_id, rank=index + 1, votes=100 - index)
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
        "map_v001",
        embedding_version=config.embedding_version,
        params={"metric": "cosine", "random_state": 42},
        coords=coords,
        coords_path="maps/map_v001/coords.parquet",
        model_2d_path="maps/map_v001/map_2d.joblib",
        model_3d_path="maps/map_v001/map_3d.joblib",
        note="AppTest 测试",
    )
    return work_ids


# --------------------------------------------------------------------------- 工具
def _run_app(monkeypatch: pytest.MonkeyPatch, repo_root: Path, config: Config) -> AppTest:
    monkeypatch.setenv("NEBULA_CONFIG", str(config.path))
    app_test = AppTest.from_file(str(repo_root / "app.py"), default_timeout=60)
    app_test.run()
    return app_test


# plotly 7 会把 numpy 数组编码成 {dtype, bdata, shape} 的二进制数组，测试里需要还原
_PLOTLY_DTYPES = {
    "i1": np.int8,
    "u1": np.uint8,
    "i2": np.int16,
    "u2": np.uint16,
    "i4": np.int32,
    "u4": np.uint32,
    "f4": np.float32,
    "f8": np.float64,
}


def _decode_array(value):
    if isinstance(value, dict) and "bdata" in value and "dtype" in value:
        dtype = np.dtype(_PLOTLY_DTYPES[str(value["dtype"])])
        array = np.frombuffer(base64.b64decode(str(value["bdata"])), dtype=dtype)
        shape = value.get("shape")
        if shape:
            dims = tuple(int(part) for part in str(shape).split(","))
            array = array.reshape(dims)
        return array
    return value


def _chart_spec(app_test: AppTest) -> dict:
    charts = app_test.get("plotly_chart")
    assert len(charts) == 1, "页面上应只有一张星云图"
    spec = charts[0].proto.spec
    return json.loads(spec)


# --------------------------------------------------------------------------- 测试
def test_app_without_data_shows_hint(
    config: Config, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    app_test = _run_app(monkeypatch, repo_root, config)

    assert not app_test.exception
    warnings = [str(element.value) for element in app_test.warning]
    assert any("还没有可用的星云" in message for message in warnings)
    assert len(app_test.get("plotly_chart")) == 0


def test_app_with_data_renders_controls_and_chart(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    work_ids = _seed_app_data(catalog, config)
    app_test = _run_app(monkeypatch, repo_root, config)

    assert not app_test.exception
    spec = _chart_spec(app_test)
    assert len(spec["data"]) >= 1
    first_trace = spec["data"][0]
    assert first_trace["type"] == "scatter"
    assert first_trace["mode"] == "markers"
    assert len(_decode_array(first_trace["x"])) == len(work_ids)
    assert "音MAD" in spec["layout"]["title"]["text"]

    assert app_test.selectbox(key="map_version").value == "map_v001"
    assert app_test.selectbox(key="color_field").value == "none"
    assert app_test.radio(key="projection").value == "2d"
    assert app_test.text_input(key="search").value == ""
    assert app_test.selectbox(key="manual_work") is not None
    assert len(app_test.slider) == 2


def test_app_switch_to_3d(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _seed_app_data(catalog, config)
    app_test = _run_app(monkeypatch, repo_root, config)

    app_test.radio(key="projection").set_value("3d").run()
    assert not app_test.exception
    spec = _chart_spec(app_test)
    assert spec["data"][0]["type"] == "scatter3d"
    assert len(_decode_array(spec["data"][0]["z"])) == 6


def test_app_switch_color_field(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _seed_app_data(catalog, config)
    app_test = _run_app(monkeypatch, repo_root, config)

    app_test.selectbox(key="color_field").set_value("duration_s").run()
    assert not app_test.exception
    spec = _chart_spec(app_test)
    marker = spec["data"][0]["marker"]
    assert marker["colorbar"]["title"]["text"] == "视频时长（秒）"
    assert len(_decode_array(marker["color"])) == 6

    app_test.selectbox(key="color_field").set_value("source").run()
    assert not app_test.exception
    spec = _chart_spec(app_test)
    marker = spec["data"][0]["marker"]
    assert marker["colorbar"]["title"]["text"] == "来源"


def test_app_search_filters_points(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _seed_app_data(catalog, config)
    app_test = _run_app(monkeypatch, repo_root, config)

    app_test.text_input(key="search").input("合成作品 2").run()
    assert not app_test.exception
    spec = _chart_spec(app_test)
    assert len(_decode_array(spec["data"][0]["x"])) == 1
    captions = [str(element.value) for element in app_test.caption]
    assert any("当前显示 1 / 6" in caption for caption in captions)


def test_app_search_no_match_shows_hint(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _seed_app_data(catalog, config)
    app_test = _run_app(monkeypatch, repo_root, config)

    app_test.text_input(key="search").input("绝对不存在的作品xyz").run()
    assert not app_test.exception
    assert len(app_test.get("plotly_chart")) == 0
    infos = [str(element.value) for element in app_test.info]
    assert any("没有匹配" in message for message in infos)


def test_app_manual_selection_shows_detail(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    work_ids = _seed_app_data(catalog, config)
    app_test = _run_app(monkeypatch, repo_root, config)

    target = work_ids[3]
    app_test.selectbox(key="manual_work").set_value(target).run()
    assert not app_test.exception
    headers = [str(element.value) for element in app_test.subheader]
    assert any("合成作品 3" in header for header in headers)
    markdowns = [str(element.value) for element in app_test.markdown]
    assert any("https://www.bilibili.com/video/BV1xx411c703" in text for text in markdowns)
