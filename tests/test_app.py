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

from nebula.appdata import mutual_neighbor_edges
from nebula.config import Config
from nebula.db import Catalog
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


def _add_map_version(
    catalog: Catalog,
    config: Config,
    work_ids: list[int],
    version: str,
    *,
    filter_spec: dict | None = None,
    note: str = "AppTest 测试",
    representation_id: int | None = None,
    embedding_version: str | None = None,
) -> None:
    """登记一个指定版本/筛选条件/表征的星云（坐标仅用于 AppTest）。"""

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
        filter_spec=filter_spec,
        representation_id=representation_id,
    )


def _create_map(catalog: Catalog, config: Config, work_ids: list[int]) -> None:
    _add_map_version(catalog, config, work_ids, "map_v001")


def _seed_app_data(
    catalog: Catalog, config: Config, count: int = 6, *, create_map: bool = True
) -> list[int]:
    """登记若干 Bilibili 作品、一个选票榜单与一个就绪星云版本。"""

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
    if create_map:
        _create_map(catalog, config, work_ids)
    return work_ids


def _seed_mixed_data(catalog: Catalog, config: Config, count: int = 6) -> list[int]:
    """本地/Bilibili 混合作品 + 2025/2024 榜单 + 周报，用于筛选测试。"""

    _ensure_embedding_version(catalog, config)
    work_ids: list[int] = []
    for index in range(count):
        if index % 3 == 0:
            work_id, _ = catalog.upsert_local_work(
                f"clips/本地作品 {index}.mp4", title=f"本地作品 {index}"
            )
        else:
            work_id, _ = catalog.upsert_bilibili_work(
                f"BV1xx411c7{index:02d}", title=f"合成作品 {index}"
            )
        work_ids.append(work_id)
    votes_2025 = catalog.ensure_list(
        "votes_2025", name="2025年音MAD选票", kind="votes", vote_year=2025
    )
    votes_2024 = catalog.ensure_list(
        "votes_2024", name="2024年音MAD选票", kind="votes", vote_year=2024
    )
    weekly = catalog.ensure_list("weekly_001", name="第1期周报", kind="weekly")
    for index, work_id in enumerate(work_ids):
        catalog.add_appearance(work_id, votes_2025, rank=index + 1, votes=100 - index)
        if index % 2 == 0:
            catalog.add_appearance(work_id, weekly)
    for work_id in (work_ids[0], work_ids[1]):
        catalog.add_appearance(work_id, votes_2024, rank=1)
    _create_map(catalog, config, work_ids)
    return work_ids


def _add_vectors(
    catalog: Catalog,
    config: Config,
    work_ids: list[int],
    *,
    seed: int = 0,
    kinds_by_work: dict[int, tuple[str, ...]] | None = None,
    with_clips: bool = False,
    n_clips: int = 5,
) -> None:
    """为作品写三种模态的合成 npy，供高维近邻功能使用。

    with_clips=True 时同时写片段级堆叠向量（5×8）与 clips 起始时间，
    供片段级对应面板测试使用。
    """

    version = config.embedding_version
    rng = np.random.default_rng(seed)
    for work_id in work_ids:
        if kinds_by_work is None:
            kinds = ("work_joint", "work_audio", "work_video")
        else:
            kinds = kinds_by_work.get(work_id, ("work_joint", "work_audio", "work_video"))
        for kind in kinds:
            vector = rng.normal(size=8).astype(np.float32)
            path = config.paths.embeddings / version / str(work_id) / f"{kind}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vector)
            catalog.register_embedding(
                work_id,
                version=version,
                kind=kind,
                path=config.data_relpath(path),
                dim=8,
            )
        if with_clips:
            for clip_kind in ("clip_joint", "clip_audio", "clip_video"):
                matrix = rng.normal(size=(n_clips, 8)).astype(np.float32)
                path = config.paths.embeddings / version / str(work_id) / f"{clip_kind}.npy"
                np.save(path, matrix)
                catalog.register_embedding(
                    work_id,
                    version=version,
                    kind=clip_kind,
                    path=config.data_relpath(path),
                    dim=8,
                    n_clips=n_clips,
                )
            catalog.replace_clips(
                work_id,
                version=version,
                clips=[(idx, float(idx) * 10.0, 8.0) for idx in range(n_clips)],
            )


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


# --------------------------------------------------------------------------- 迭代 A2：近邻探索
def test_app_neighbor_panel_three_modalities(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    work_ids = _seed_app_data(catalog, config)
    _add_vectors(catalog, config, work_ids)
    app_test = _run_app(monkeypatch, repo_root, config)

    assert not app_test.exception
    tables = [element.value for element in app_test.dataframe]
    assert len(tables) == 3  # 三个模态各一张 Top5 表
    for table in tables:
        assert list(table.columns) == ["排名", "作品", "余弦相似度", "全库分位", "双向"]
        assert len(table) == 5
        assert table["排名"].tolist() == [1, 2, 3, 4, 5]
        values = [float(value) for value in table["余弦相似度"]]
        assert values == sorted(values, reverse=True)
        assert all(-1.0 <= value <= 1.0 for value in values)
    captions = [str(element.value) for element in app_test.caption]
    assert any("高维近邻已就绪" in caption for caption in captions)


def test_app_highlight_and_lines_2d(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    work_ids = _seed_app_data(catalog, config)
    _add_vectors(catalog, config, work_ids)
    app_test = _run_app(monkeypatch, repo_root, config)

    assert not app_test.exception
    spec = _chart_spec(app_test)
    traces = {trace.get("name"): trace for trace in spec["data"]}
    assert "选中作品" in traces
    assert "Top5 近邻" in traces
    assert "近邻连线" in traces
    assert len(_decode_array(traces["Top5 近邻"]["x"])) == 5
    line_trace = traces["近邻连线"]
    assert line_trace["mode"] == "lines"
    assert len(line_trace["x"]) == 15  # 5 条线段 × (起点, 终点, None)


def test_app_3d_lines_toggle(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    work_ids = _seed_app_data(catalog, config)
    _add_vectors(catalog, config, work_ids)
    app_test = _run_app(monkeypatch, repo_root, config)

    app_test.radio(key="projection").set_value("3d").run()
    assert not app_test.exception
    spec = _chart_spec(app_test)
    names = [trace.get("name") for trace in spec["data"]]
    assert "选中作品" in names
    assert "Top5 近邻" in names
    assert "近邻连线" not in names  # 3D 默认关闭连线
    assert app_test.checkbox(key="show_lines_3d").value is False

    app_test.checkbox(key="show_lines_3d").set_value(True).run()
    assert not app_test.exception
    spec = _chart_spec(app_test)
    line_traces = [trace for trace in spec["data"] if trace.get("name") == "近邻连线"]
    assert len(line_traces) == 1
    assert line_traces[0]["type"] == "scatter3d"
    assert len(line_traces[0]["z"]) == 15


def test_app_density_knn_color(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    work_ids = _seed_app_data(catalog, config)
    _add_vectors(catalog, config, work_ids)
    app_test = _run_app(monkeypatch, repo_root, config)

    app_test.selectbox(key="color_field").set_value("density_knn").run()
    assert not app_test.exception
    spec = _chart_spec(app_test)
    marker = spec["data"][0]["marker"]
    assert "近邻" in marker["colorbar"]["title"]["text"]
    assert len(_decode_array(marker["color"])) == len(work_ids)


def test_app_filters(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    work_ids = _seed_mixed_data(catalog, config)
    _add_vectors(catalog, config, work_ids)
    app_test = _run_app(monkeypatch, repo_root, config)

    spec = _chart_spec(app_test)
    assert len(_decode_array(spec["data"][0]["x"])) == 6

    app_test.selectbox(key="filter_source").set_value("local").run()
    assert not app_test.exception
    spec = _chart_spec(app_test)
    assert len(_decode_array(spec["data"][0]["x"])) == 2

    app_test.multiselect(key="filter_years").set_value([2024]).run()
    assert not app_test.exception
    spec = _chart_spec(app_test)
    assert len(_decode_array(spec["data"][0]["x"])) == 1

    # 恢复来源筛选后按周报榜单过滤
    app_test.selectbox(key="filter_source").set_value("all").run()
    app_test.multiselect(key="filter_years").set_value([]).run()
    app_test.multiselect(key="filter_lists").set_value(["weekly_001"]).run()
    assert not app_test.exception
    spec = _chart_spec(app_test)
    assert len(_decode_array(spec["data"][0]["x"])) == 3

    # 组合筛选不匹配时给出中文提示（Bilibili 作品都不在 2024 年周报里）
    app_test.selectbox(key="filter_source").set_value("bilibili").run()
    app_test.multiselect(key="filter_lists").set_value(["votes_2024"]).run()
    app_test.multiselect(key="filter_years").set_value([2024]).run()
    app_test.multiselect(key="filter_lists").set_value(["weekly_001"]).run()
    assert not app_test.exception
    infos = [str(element.value) for element in app_test.info]
    assert any("没有作品" in message for message in infos)


def test_app_neighbor_jump(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    from nebula.similarity import load_similarity_index

    work_ids = _seed_app_data(catalog, config)
    _add_vectors(catalog, config, work_ids)
    index = load_similarity_index(catalog, config)
    try:
        target = index.neighbors(work_ids[0], kind="work_joint", k=1)[0].work_id
    finally:
        index.close()

    app_test = _run_app(monkeypatch, repo_root, config)
    app_test.selectbox(key="neighbor_jump").set_value(target).run()
    assert not app_test.exception
    headers = [str(element.value) for element in app_test.subheader]
    assert any(str(target) in header or "作品" in header for header in headers)
    target_work = catalog.get_work(target)
    assert any(str(target_work["title"]) in header for header in headers)
    assert app_test.selectbox(key="manual_work").value == target


def test_app_missing_modality_caption(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    work_ids = _seed_app_data(catalog, config)
    _add_vectors(
        catalog,
        config,
        work_ids,
        kinds_by_work={work_ids[5]: ("work_joint", "work_video")},
    )
    app_test = _run_app(monkeypatch, repo_root, config)

    app_test.selectbox(key="manual_work").set_value(work_ids[5]).run()
    assert not app_test.exception
    captions = [str(element.value) for element in app_test.caption]
    assert any("无音频模态向量" in caption for caption in captions)
    tables = [element.value for element in app_test.dataframe]
    assert len(tables) == 2  # 只有联合与画面两个模态


# --------------------------------------------------------------------------- 默认星云版本选择（UX）
def _seed_version_choice_env(
    catalog: Catalog, config: Config, *, with_total: bool = True
) -> list[int]:
    """构造多个星云版本：总星云 + 2025 筛选星云 + n_neighbors 变体。"""

    work_ids = _seed_app_data(catalog, config, count=6, create_map=with_total)
    if not with_total:
        _add_map_version(
            catalog,
            config,
            work_ids[:4],
            "map_v001",
            filter_spec={"list_key": "votes_2025"},
        )
        _add_map_version(
            catalog,
            config,
            work_ids[:2],
            "map_v002",
            filter_spec={"vote_year": 2025},
        )
        return work_ids
    # 总星云 map_v001 之后又生成了 2025 子集与 n_neighbors 变体
    _add_map_version(
        catalog,
        config,
        work_ids[:4],
        "map_v002",
        filter_spec={"list_key": "votes_2025"},
        note="2025 独立星云",
    )
    _add_map_version(
        catalog,
        config,
        work_ids[:2],
        "map_v003",
        filter_spec={"vote_year": 2025},
        note="2025 变体 n_neighbors=30",
    )
    return work_ids


def test_app_default_prefers_latest_unfiltered_map(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    """默认打开“最新且 filter_json 为空的总星云”，而不是最新版本。"""

    _seed_version_choice_env(catalog, config, with_total=True)
    app_test = _run_app(monkeypatch, repo_root, config)

    assert not app_test.exception
    # 最新版本是 map_v003（筛选变体），但默认应选总星云 map_v001
    assert app_test.selectbox(key="map_version").value == "map_v001"
    spec = _chart_spec(app_test)
    assert len(_decode_array(spec["data"][0]["x"])) == 6


def test_app_manual_map_switch_keeps_user_choice(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    """用户手动切到筛选星云后，后续交互不应被默认逻辑改回总星云。"""

    _seed_version_choice_env(catalog, config, with_total=True)
    app_test = _run_app(monkeypatch, repo_root, config)
    assert app_test.selectbox(key="map_version").value == "map_v001"

    app_test.selectbox(key="map_version").set_value("map_v003").run()
    assert not app_test.exception
    assert app_test.selectbox(key="map_version").value == "map_v003"
    spec = _chart_spec(app_test)
    assert len(_decode_array(spec["data"][0]["x"])) == 2

    # 其它控件交互后仍保持用户选择的版本
    app_test.selectbox(key="filter_source").set_value("bilibili").run()
    assert not app_test.exception
    assert app_test.selectbox(key="map_version").value == "map_v003"

    # 再切回总星云也正常
    app_test.selectbox(key="map_version").set_value("map_v001").run()
    assert not app_test.exception
    assert app_test.selectbox(key="map_version").value == "map_v001"
    spec = _chart_spec(app_test)
    assert len(_decode_array(spec["data"][0]["x"])) == 6


def test_app_default_falls_back_to_latest_when_all_maps_filtered(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    """没有总星云（全部带筛选）时，回退到最新 ready 版本。"""

    _seed_version_choice_env(catalog, config, with_total=False)
    app_test = _run_app(monkeypatch, repo_root, config)

    assert not app_test.exception
    assert app_test.selectbox(key="map_version").value == "map_v002"
    spec = _chart_spec(app_test)
    assert len(_decode_array(spec["data"][0]["x"])) == 2


# --------------------------------------------------------------------------- P1 近邻解释与枢纽
def test_app_neighbor_table_has_explanation_columns(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    work_ids = _seed_app_data(catalog, config)
    _add_vectors(catalog, config, work_ids)
    app_test = _run_app(monkeypatch, repo_root, config)

    assert not app_test.exception
    tables = [element.value for element in app_test.dataframe]
    assert len(tables) == 3
    for table in tables:
        assert "全库分位" in table.columns
        assert "双向" in table.columns
        assert all(str(value).endswith("%") for value in table["全库分位"])
        assert set(table["双向"]) <= {"", "是"}
    captions = [str(element.value) for element in app_test.caption]
    assert any("全库分位" in caption for caption in captions)


def test_app_hubness_table(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    work_ids = _seed_app_data(catalog, config)
    _add_vectors(catalog, config, work_ids)
    app_test = _run_app(monkeypatch, repo_root, config)

    assert not app_test.exception
    tables = at_tables = app_test.get("table")
    assert len(tables) == 1, "应有一个枢纽作品静态表（st.table，不计入 dataframe）"
    hub_table = tables[0].value
    assert list(hub_table.columns) == ["排名", "作品", "被列次数"]
    assert 1 <= len(hub_table) <= 8
    counts = [int(value) for value in hub_table["被列次数"]]
    assert counts == sorted(counts, reverse=True)
    assert [int(value) for value in hub_table["排名"]] == list(range(1, len(hub_table) + 1))
    captions = [str(element.value) for element in app_test.caption]
    assert any("hubness" in caption for caption in captions)
    assert len(at_tables) == 1


# --------------------------------------------------------------------------- P2 片段级对应面板
def test_app_clip_panel(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    work_ids = _seed_app_data(catalog, config)
    _add_vectors(catalog, config, work_ids, with_clips=True)
    app_test = _run_app(monkeypatch, repo_root, config)

    assert not app_test.exception
    # 默认不加载片段面板：只有主图与三张近邻表
    assert len(app_test.get("plotly_chart")) == 1
    assert len(app_test.dataframe) == 3
    assert app_test.checkbox(key="show_clip_panel").value is False

    app_test.checkbox(key="show_clip_panel").set_value(True).run()
    assert not app_test.exception
    charts = app_test.get("plotly_chart")
    assert len(charts) == 2
    heatmap_spec = json.loads(charts[1].proto.spec)
    heatmap_trace = heatmap_spec["data"][0]
    assert heatmap_trace["type"] == "heatmap"
    matrix = np.asarray(_decode_array(heatmap_trace["z"]), dtype=float)
    assert matrix.shape == (5, 5)
    assert np.all(matrix >= -1.0) and np.all(matrix <= 1.0)

    tables = [element.value for element in app_test.dataframe]
    assert len(tables) == 4
    pair_table = tables[3]
    assert list(pair_table.columns) == [
        "排名",
        "A 片段",
        "A 起始(s)",
        "B 片段",
        "B 起始(s)",
        "余弦",
    ]
    assert len(pair_table) == 5
    similarities = [float(value) for value in pair_table["余弦"]]
    assert similarities == sorted(similarities, reverse=True)
    captions = [str(element.value) for element in app_test.caption]
    assert any("片段级统计" in caption for caption in captions)

    # 切换片段模态与对比作品都不应报错
    assert app_test.selectbox(key="clip_kind").value == "clip_joint"
    app_test.selectbox(key="clip_kind").set_value("clip_audio").run()
    assert not app_test.exception
    assert app_test.selectbox(key="clip_kind").value == "clip_audio"
    compare_options = app_test.selectbox(key="clip_compare_work").options
    assert len(compare_options) >= 1
    new_compare = app_test.selectbox(key="clip_compare_work").value
    other_ids = [
        work_id
        for work_id in work_ids
        if work_id != int(app_test.selectbox(key="manual_work").value)
    ]
    app_test.selectbox(key="clip_compare_work").set_value(other_ids[0]).run()
    assert not app_test.exception
    assert app_test.selectbox(key="clip_compare_work").value == other_ids[0]
    assert new_compare is not None
    assert len(app_test.get("plotly_chart")) == 2


def test_app_clip_panel_absent_without_clip_vectors(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    work_ids = _seed_app_data(catalog, config)
    _add_vectors(catalog, config, work_ids)  # 只有作品级向量
    app_test = _run_app(monkeypatch, repo_root, config)

    assert not app_test.exception
    checkbox_keys = [element.key for element in app_test.checkbox]
    assert "show_clip_panel" not in checkbox_keys
    assert len(app_test.get("plotly_chart")) == 1


# --------------------------------------------------------------------------- v0.4 表征空间切换与时间轴
def _add_derived_representation(
    catalog: Catalog,
    config: Config,
    work_ids: list[int],
    *,
    name: str = "full_mean_v1",
    sequence_version: str = "full-seq-v1",
    kind: str = "full_mean",
    seed: int = 31,
    dim: int = 8,
    source_kind: str = "pe_av",
    register_embedding: bool = True,
) -> int:
    """登记一个只含 joint 向量的 synthetic 派生表征，返回 id。"""

    from nebula import representations

    if register_embedding:
        _ensure_embedding_version(catalog, config, sequence_version)
    rep_id = catalog.register_representation(
        name,
        sequence_version=sequence_version,
        kind=kind,
        params={"synthetic": True},
        source_kind=source_kind,
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


def test_app_representation_space_switch(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    work_ids = _seed_app_data(catalog, config)  # map_v001：legacy / 6 部
    derived_ids = work_ids[:4]
    rep_id = _add_derived_representation(catalog, config, derived_ids)
    _add_map_version(
        catalog,
        config,
        derived_ids,
        "map_v002",
        representation_id=rep_id,
        embedding_version="full-seq-v1",
        note="Full Mean 地图",
    )

    app_test = _run_app(monkeypatch, repo_root, config)
    assert not app_test.exception

    space_select = app_test.selectbox(key="representation_space")
    labels = [str(label) for label in space_select.options]
    full_index = next(index for index, label in enumerate(labels) if "Full Mean" in label)
    legacy_index = next(index for index, label in enumerate(labels) if "Legacy" in label)
    # 默认打开最新无筛选地图 map_v002，自动定位到 Full Mean 空间
    assert space_select.value == full_index
    assert app_test.selectbox(key="map_version").value == "map_v002"
    spec = _chart_spec(app_test)
    assert len(_decode_array(spec["data"][0]["x"])) == len(derived_ids)
    # 派生表征只有 joint 向量 -> 只有一张近邻表；无片段向量 -> 片段面板禁用
    assert len(app_test.dataframe) == 1
    assert "show_clip_panel" not in [element.key for element in app_test.checkbox]
    captions = [str(element.value) for element in app_test.caption]
    assert any("Full Mean（全片平均）" in caption for caption in captions)

    # 切回 Legacy 空间 -> 旧地图 map_v001（6 部）
    app_test.selectbox(key="representation_space").set_value(legacy_index).run()
    assert not app_test.exception
    assert app_test.selectbox(key="map_version").value == "map_v001"
    spec = _chart_spec(app_test)
    assert len(_decode_array(spec["data"][0]["x"])) == len(work_ids)
    captions = [str(element.value) for element in app_test.caption]
    assert any("Legacy（旧 5 片段平均）" in caption for caption in captions)

    # 再切回 Full Mean 空间
    app_test.selectbox(key="representation_space").set_value(full_index).run()
    assert not app_test.exception
    assert app_test.selectbox(key="map_version").value == "map_v002"
    assert len(_decode_array(_chart_spec(app_test)["data"][0]["x"])) == len(derived_ids)


def test_app_timeline_panel(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    from types import SimpleNamespace

    from nebula import appdata as appdata_module

    work_ids = _seed_app_data(catalog, config)
    sequence_version = config.embedding_version
    catalog.update_work_metadata(work_ids[0], duration_s=24.0)
    catalog.replace_clips(
        work_ids[0],
        version=sequence_version,
        clips=[(idx, float(idx) * 8.0, 8.0, 8.0, 1.0 / 3.0) for idx in range(3)],
    )
    rng = np.random.default_rng(5)
    stub = SimpleNamespace(
        audio=rng.normal(size=(100, 6)).astype(np.float32),
        video=rng.normal(size=(30, 6)).astype(np.float32),
        av=rng.normal(size=(100, 6)).astype(np.float32),
        window_features=rng.normal(size=(3, 32)).astype(np.float32),
        audio_rate_hz=100.0,
        video_fps=30,
        video_fps_effective=30.0,
        duration_s=24.0,
        version="events_v1",
    )
    monkeypatch.setattr(appdata_module, "_load_events_bundle", lambda work_dir: stub)

    app_test = _run_app(monkeypatch, repo_root, config)
    assert not app_test.exception
    charts = app_test.get("plotly_chart")
    assert len(charts) == 2  # 星云主图 + 完整时间轴
    timeline_spec = json.loads(charts[1].proto.spec)
    names = [str(trace.get("name") or "") for trace in timeline_spec["data"]]
    assert any("音频" in name for name in names)
    assert any("画面" in name for name in names)
    assert len(timeline_spec["layout"]["shapes"]) == 3  # 3 个片段窗口竖线
    captions = [str(element.value) for element in app_test.caption]
    assert any("覆盖 100.0%" in caption for caption in captions)
    assert any("事件版本 events_v1" in caption for caption in captions)


def test_app_timeline_without_events_shows_hint(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _seed_app_data(catalog, config)
    app_test = _run_app(monkeypatch, repo_root, config)

    assert not app_test.exception
    assert len(app_test.get("plotly_chart")) == 1
    captions = [str(element.value) for element in app_test.caption]
    assert any("暂无事件扫描数据" in caption for caption in captions)


def test_app_timeline_chart_count_guard(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    """保护：有事件文件时 2 张图（星云+时间轴），没有事件文件时只有 1 张图。

    同一个 AppTest 实例先验证“无事件 -> 1 图”，再补事件数据并清缓存复跑验证“有事件 -> 2 图”。
    """

    from types import SimpleNamespace

    from streamlit import cache_data as streamlit_cache_data

    from nebula import appdata as appdata_module

    work_ids = _seed_app_data(catalog, config)
    sequence_version = config.embedding_version

    app_test = _run_app(monkeypatch, repo_root, config)
    assert not app_test.exception
    assert len(app_test.get("plotly_chart")) == 1
    captions = [str(element.value) for element in app_test.caption]
    assert any("暂无事件扫描数据" in caption for caption in captions)

    # 补 clips 表 + 事件包络 stub，清 cache_data 后用同一实例复跑
    catalog.update_work_metadata(work_ids[0], duration_s=24.0)
    catalog.replace_clips(
        work_ids[0],
        version=sequence_version,
        clips=[(idx, float(idx) * 8.0, 8.0, 8.0, 1.0 / 3.0) for idx in range(3)],
    )
    rng = np.random.default_rng(11)
    stub = SimpleNamespace(
        audio=rng.normal(size=(100, 6)).astype(np.float32),
        video=rng.normal(size=(30, 6)).astype(np.float32),
        av=rng.normal(size=(100, 6)).astype(np.float32),
        window_features=rng.normal(size=(3, 32)).astype(np.float32),
        audio_rate_hz=100.0,
        video_fps=30,
        video_fps_effective=30.0,
        duration_s=24.0,
        version="events_v1",
    )
    monkeypatch.setattr(appdata_module, "_load_events_bundle", lambda work_dir: stub)
    streamlit_cache_data.clear()
    app_test.run()
    assert not app_test.exception
    assert len(app_test.get("plotly_chart")) == 2
    timeline_spec = json.loads(app_test.get("plotly_chart")[1].proto.spec)
    names = [str(trace.get("name") or "") for trace in timeline_spec["data"]]
    assert any("音频" in name for name in names)
    assert any("画面" in name for name in names)
    captions = [str(element.value) for element in app_test.caption]
    assert any("覆盖 100.0%" in caption for caption in captions)



# --------------------------------------------------------------------------- v0.4.3 视觉表征空间
def _add_visual_representation(
    catalog: Catalog,
    config: Config,
    work_ids: list[int],
    *,
    name: str,
    sequence_version: str,
    seed: int = 91,
    dim: int = 8,
) -> int:
    """登记 visual_versions + source_kind=visual 的 joint 表征并写向量。"""

    is_dino = "dino" in name
    catalog.register_visual_version(
        sequence_version,
        model_id=(
            "facebook/dinov2-small" if is_dino else "openai/clip-vit-base-patch16"
        ),
        revision=(
            "ed25f3a31f01632728cabb09d1542f84ab7b0056"
            if is_dino
            else "57c216476eefef5ab752ec549e440a49ae4ae5f3"
        ),
        preprocessing="letterbox",
        fps=8.0,
        size=224,
        dim=dim,
    )
    return _add_derived_representation(
        catalog,
        config,
        work_ids,
        name=name,
        sequence_version=sequence_version,
        kind="visual",
        seed=seed,
        dim=dim,
        source_kind="visual",
        register_embedding=False,
    )


def test_app_all_spaces_including_visual(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    work_ids = _seed_app_data(catalog, config)  # map_v001：legacy / 6 部
    _add_vectors(catalog, config, work_ids)  # legacy 空间有 3 类作品向量
    full_rep = _add_derived_representation(
        catalog, config, work_ids, sequence_version="full-seq-v1", seed=92
    )
    _add_map_version(
        catalog,
        config,
        work_ids,
        "map_v002",
        representation_id=full_rep,
        embedding_version="full-seq-v1",
        note="Full Mean 地图",
    )
    temporal_rep = _add_derived_representation(
        catalog,
        config,
        work_ids,
        name="temporal_stats_v1",
        sequence_version="temporal-seq-v1",
        kind="temporal_stats",
        seed=93,
    )
    _add_map_version(
        catalog,
        config,
        work_ids,
        "map_v003",
        representation_id=temporal_rep,
        embedding_version="temporal-seq-v1",
        note="Temporal 地图",
    )
    clip_rep = _add_visual_representation(
        catalog,
        config,
        work_ids,
        name="dense_visual_clip_v1",
        sequence_version="visual-clip-v1",
        seed=94,
    )
    _add_map_version(
        catalog,
        config,
        work_ids,
        "map_v004",
        representation_id=clip_rep,
        embedding_version="visual-clip-v1",
        note="视觉 CLIP 地图",
    )
    dino_rep = _add_visual_representation(
        catalog,
        config,
        work_ids,
        name="dense_visual_dino_v1",
        sequence_version="visual-dino-v1",
        seed=95,
    )
    _add_map_version(
        catalog,
        config,
        work_ids,
        "map_v005",
        representation_id=dino_rep,
        embedding_version="visual-dino-v1",
        note="视觉 DINOv2 地图",
    )

    app_test = _run_app(monkeypatch, repo_root, config)
    assert not app_test.exception
    assert not app_test.error
    labels = [str(label) for label in app_test.selectbox(key="representation_space").options]
    assert labels == [
        "Legacy（旧 5 片段平均）",
        "Full Mean（全片平均）",
        "Temporal Stats（时间结构）",
        "Dense Visual·CLIP（密集视觉语义）",
        "Dense Visual·DINOv2（密集视觉结构）",
    ]
    # 默认优先最新无筛选的 PE-AV 空间（Temporal Stats -> map_v003），视觉空间为对照
    assert app_test.selectbox(key="representation_space").value == 2
    assert app_test.selectbox(key="map_version").value == "map_v003"
    assert len(_decode_array(_chart_spec(app_test)["data"][0]["x"])) == len(work_ids)
    assert len(app_test.dataframe) == 1
    assert len(app_test.get("plotly_chart")) >= 1
    captions = [str(element.value) for element in app_test.caption]
    assert any("Temporal Stats" in caption for caption in captions)

    expected = [
        (
            0,
            "map_v001",
            "Legacy（旧 5 片段平均）",
            3,
            f"序列版本 {config.embedding_version}",
        ),
        (1, "map_v002", "Full Mean（全片平均）", 1, "序列版本 full-seq-v1"),
        (2, "map_v003", "Temporal Stats（时间结构）", 1, "序列版本 temporal-seq-v1"),
        (
            3,
            "map_v004",
            "Dense Visual·CLIP（密集视觉语义）",
            1,
            "视觉版本 visual-clip-v1",
        ),
    ]
    for space_index, map_version, label, n_tables, version_text in expected:
        app_test.selectbox(key="representation_space").set_value(space_index).run()
        assert not app_test.exception
        assert not app_test.error
        assert app_test.selectbox(key="map_version").value == map_version
        assert len(_decode_array(_chart_spec(app_test)["data"][0]["x"])) == len(work_ids)
        assert len(app_test.dataframe) == n_tables
        captions = [str(element.value) for element in app_test.caption]
        assert any(label in caption for caption in captions), (label, captions[-3:])
        assert any(
            version_text in caption for caption in captions
        ), (space_index, version_text, captions[-3:])
        if space_index == 0:
            assert any("序列 peav" in caption for caption in captions)

    # 手动切到 DINOv2 视觉空间：版本措辞/时间轴提示/近邻表降级均正常
    app_test.selectbox(key="representation_space").set_value(4).run()
    assert not app_test.exception
    assert not app_test.error
    assert app_test.selectbox(key="map_version").value == "map_v005"
    assert len(app_test.dataframe) == 1
    assert len(app_test.get("plotly_chart")) == 1
    assert "show_clip_panel" not in [element.key for element in app_test.checkbox]
    captions = [str(element.value) for element in app_test.caption]
    assert any("Dense Visual·DINOv2" in caption for caption in captions)
    assert any("视觉版本 visual-dino-v1" in caption for caption in captions)
    assert any("视觉表征空间不含事件时间轴" in caption for caption in captions)



# --------------------------------------------------------------------------- v0.4.4 邻域连线
def _seed_three_spaces(
    catalog: Catalog, config: Config, *, count: int = 6
) -> tuple[list[int], int]:
    """Legacy / Full Mean / Dense Visual 三空间 + 各自地图（默认落 Full Mean）。"""

    work_ids = _seed_app_data(catalog, config, count=count)  # map_v001 legacy
    _add_vectors(catalog, config, work_ids)
    full_rep = _add_derived_representation(
        catalog, config, work_ids, sequence_version="full-seq-v1", seed=201
    )
    _add_map_version(
        catalog,
        config,
        work_ids,
        "map_v002",
        representation_id=full_rep,
        embedding_version="full-seq-v1",
        note="Full Mean 地图",
    )
    visual_rep = _add_visual_representation(
        catalog,
        config,
        work_ids,
        name="dense_visual_clip_v1",
        sequence_version="visual-clip-v1",
        seed=202,
    )
    _add_map_version(
        catalog,
        config,
        work_ids,
        "map_v003",
        representation_id=visual_rep,
        embedding_version="visual-clip-v1",
        note="视觉 CLIP 地图",
    )
    return work_ids, full_rep


def _expected_mutual_edges(
    catalog: Catalog, config: Config, rep_id: int, work_ids: list[int], k: int = 5
) -> list[dict]:
    index = load_similarity_index(catalog, config, representation_id=rep_id)
    try:
        return mutual_neighbor_edges(index, work_ids, k=k)["edges"]
    finally:
        index.close()


def _chart_trace(spec: dict, name: str) -> dict | None:
    for trace in spec.get("data", []):
        if trace.get("name") == name:
            return trace
    return None


def test_app_mutual_edges_default_off_and_overlay(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    work_ids, full_rep = _seed_three_spaces(catalog, config)
    expected = _expected_mutual_edges(catalog, config, full_rep, work_ids)
    assert expected, "合成数据应有互为近邻边"

    app_test = _run_app(monkeypatch, repo_root, config)
    assert not app_test.exception
    assert app_test.checkbox(key="show_mutual_edges").value is False
    names = [trace.get("name") for trace in _chart_spec(app_test)["data"]]
    assert "高维互为近邻" not in names
    assert len(app_test.get("plotly_chart")) == 1

    app_test.checkbox(key="show_mutual_edges").set_value(True).run()
    assert not app_test.exception
    assert app_test.checkbox(key="show_mutual_edges").value is True
    assert app_test.selectbox(key="mutual_edges_k").value == 5
    spec = _chart_spec(app_test)
    trace = _chart_trace(spec, "高维互为近邻")
    assert trace is not None, [entry.get("name") for entry in spec["data"]]
    assert trace["mode"] == "lines"
    assert len(_decode_array(trace["x"])) == 3 * len(expected)
    assert len(_decode_array(trace["y"])) == 3 * len(expected)
    captions = [str(element.value) for element in app_test.caption]
    assert any("高维互为近邻边" in caption for caption in captions)
    assert any("k=5" in caption for caption in captions)
    assert len(app_test.get("plotly_chart")) == 1


def test_app_mutual_edges_truncation_caption(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    work_ids, full_rep = _seed_three_spaces(catalog, config)
    expected = _expected_mutual_edges(catalog, config, full_rep, work_ids)
    assert len(expected) > 2
    monkeypatch.setattr("nebula.appdata.DEFAULT_MUTUAL_EDGE_LIMIT", 2)

    app_test = _run_app(monkeypatch, repo_root, config)
    app_test.checkbox(key="show_mutual_edges").set_value(True).run()
    assert not app_test.exception
    spec = _chart_spec(app_test)
    trace = _chart_trace(spec, "高维互为近邻")
    assert trace is not None
    assert len(_decode_array(trace["x"])) == 2 * 3  # 只画前 2 条边
    captions = [str(element.value) for element in app_test.caption]
    assert any("边数过多" in caption for caption in captions)
    assert any("仅显示前 2 条" in caption for caption in captions)
    assert any("已截断" in caption for caption in captions)


def test_app_mutual_edges_3d(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    work_ids, full_rep = _seed_three_spaces(catalog, config)
    expected = _expected_mutual_edges(catalog, config, full_rep, work_ids)

    app_test = _run_app(monkeypatch, repo_root, config)
    app_test.radio(key="projection").set_value("3d").run()
    assert not app_test.exception
    app_test.checkbox(key="show_mutual_edges").set_value(True).run()
    assert not app_test.exception
    spec = _chart_spec(app_test)
    trace = _chart_trace(spec, "高维互为近邻")
    assert trace is not None
    assert trace["type"] == "scatter3d"
    assert trace["mode"] == "lines"
    assert len(_decode_array(trace["z"])) == 3 * len(expected)
    assert len(app_test.get("plotly_chart")) == 1
    # 3D 原有的选中作品连线开关仍默认关闭且不影响
    assert app_test.checkbox(key="show_lines_3d").value is False


def test_app_mutual_edges_missing_consensus_fallback(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _seed_three_spaces(catalog, config)
    app_test = _run_app(monkeypatch, repo_root, config)
    app_test.checkbox(key="show_mutual_edges").set_value(True).run()
    assert not app_test.exception
    captions = [str(element.value) for element in app_test.caption]
    assert any("未找到跨模型共识边文件" in caption for caption in captions)
    # 文件缺失时没有“仅显示共识边”开关，普通互邻仍绘制
    assert "show_consensus_only" not in [element.key for element in app_test.checkbox]
    assert _chart_trace(_chart_spec(app_test), "高维互为近邻") is not None


def test_app_consensus_edges_overlay(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    work_ids, full_rep = _seed_three_spaces(catalog, config)
    config.paths.logs.mkdir(parents=True, exist_ok=True)
    consensus_payload = {
        "edges": [
            {
                "work_pair": [work_ids[0], work_ids[1]],
                "consensus_count": 3,
                "spaces": ["full_mean_v1", "dense_visual_clip_v1", "temporal_stats_v1"],
            },
            {
                "work_pair": [work_ids[0], work_ids[2]],
                "consensus_count": 2,
                "spaces": ["full_mean_v1", "dense_visual_clip_v1"],
            },
            {
                "work_pair": [work_ids[1], work_ids[2]],
                "consensus_count": 1,
                "spaces": ["full_mean_v1"],
            },
        ]
    }
    (config.paths.logs / "neighbor_consensus.json").write_text(
        json.dumps(consensus_payload, ensure_ascii=False), encoding="utf-8"
    )

    app_test = _run_app(monkeypatch, repo_root, config)
    app_test.checkbox(key="show_mutual_edges").set_value(True).run()
    assert not app_test.exception
    assert app_test.checkbox(key="show_consensus_only").value is False
    app_test.checkbox(key="show_consensus_only").set_value(True).run()
    assert not app_test.exception
    assert app_test.selectbox(key="consensus_min_count").value == 2
    spec = _chart_spec(app_test)
    assert "高维互为近邻" not in [trace.get("name") for trace in spec["data"]]
    consensus_trace = _chart_trace(spec, "跨模型共识边")
    assert consensus_trace is not None
    assert consensus_trace["mode"] == "lines"
    assert len(_decode_array(consensus_trace["x"])) == 2 * 3  # 阈值 >=2 共 2 条
    captions = [str(element.value) for element in app_test.caption]
    assert any("跨模型共识边 2 条" in caption for caption in captions)

    # 提到 >=3 后只剩 1 条
    app_test.selectbox(key="consensus_min_count").set_value(3).run()
    assert not app_test.exception
    consensus_trace = _chart_trace(_chart_spec(app_test), "跨模型共识边")
    assert consensus_trace is not None
    assert len(_decode_array(consensus_trace["x"])) == 1 * 3
    captions = [str(element.value) for element in app_test.caption]
    assert any("跨模型共识边 1 条" in caption for caption in captions)
    # 详情面板的共识摘要（markdown 行）
    markdowns = [str(element.value) for element in app_test.markdown]
    assert any("共识 3 个空间" in markdown for markdown in markdowns), markdowns[-3:]
    assert any("work_pair" not in markdown for markdown in markdowns)
    assert len(app_test.get("plotly_chart")) == 1


def test_app_diagnostics_hint(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _seed_app_data(catalog, config)
    app_test = _run_app(monkeypatch, repo_root, config)
    captions = [str(element.value) for element in app_test.caption]
    assert any("PCA" in caption and "UMAP" in caption for caption in captions)
    assert any("未找到诊断产物" in caption for caption in captions)

    # 有诊断产物时列出可用路径，且不改变默认 UMAP/图表数量
    config.paths.logs.mkdir(parents=True, exist_ok=True)
    html_dir = config.paths.logs / "dim_diagnostics_v2_html"
    html_dir.mkdir(parents=True, exist_ok=True)
    (config.paths.logs / "dim_diagnostics_v2.json").write_text("{}", encoding="utf-8")
    fresh = _run_app(monkeypatch, repo_root, config)
    captions = [str(element.value) for element in fresh.caption]
    assert any("PCA" in caption and "UMAP" in caption for caption in captions)
    assert any("诊断 HTML 目录" in caption for caption in captions)
    markdowns = [str(element.value) for element in fresh.markdown]
    assert any(
        "dim_diagnostics_v2_html" in markdown and "file://" in markdown
        for markdown in markdowns
    )
    assert len(fresh.get("plotly_chart")) == 1



# --------------------------------------------------------------------------- v0.4.5 意外关系探索
def _space_metrics(rank_a: int, rank_b: int, bar_p: float, cos: float) -> dict:
    return {
        "rank_a": rank_a,
        "rank_b": rank_b,
        "p_a": bar_p,
        "p_b": bar_p,
        "bar_p": bar_p,
        "cos": cos,
    }


def _unexpected_payload(
    work_ids: list[int],
    mutual_pair: tuple[int, int],
    non_mutual_pair: tuple[int, int],
    third_pair: tuple[int, int],
) -> dict:
    a, b = mutual_pair
    c, d = non_mutual_pair
    e, f = third_pair

    def mode_a_row(first: int, second: int, score: float) -> dict:
        return {
            "work_id_a": first,
            "work_id_b": second,
            "score": score,
            "joint": _space_metrics(1, 2, 0.9, 0.88),
            "audio": _space_metrics(3, 4, 0.7, 0.6),
            "video": _space_metrics(5, 6, 0.6, 0.5),
        }

    def mode_c_row(first: int, second: int, score: float, mutual: bool) -> dict:
        return {
            "work_id_a": first,
            "work_id_b": second,
            "score": score,
            "mutual_count": 3 if mutual else 2,
            "mutual_spaces": ["peav", "clip", "dino"] if mutual else ["clip", "dino"],
            "peav": {**_space_metrics(1, 1, 0.95, 0.9), "mutual": mutual},
            "clip": {**_space_metrics(2, 3, 0.85, 0.75), "mutual": True},
            "dino": {**_space_metrics(4, 2, 0.8, 0.7), "mutual": True},
        }

    return {
        "meta": {
            "tool": "unexpected_relations",
            "version": "v0.4.5",
            "definitions": {
                "mode_a": "joint Top-k 有向边无序并集候选；U = bar_p_joint - ...",
                "mode_b": "temporal Top-k 候选；U = ...",
                "mode_c": "PE-AV/CLIP/DINO mutual Top-k 对比；分数全部用 p 差/组合",
            },
        },
        "spaces": [
            {"key": "full_mean_joint"},
            {"key": "full_mean_audio"},
            {"key": "full_mean_video"},
            {"key": "temporal_joint"},
            {"key": "clip_joint"},
            {"key": "dino_joint"},
        ],
        "mode_a": {
            "skipped": False,
            "top": [
                mode_a_row(a, b, 0.9),
                mode_a_row(c, d, 0.8),
                mode_a_row(e, f, 0.7),
            ],
        },
        "mode_b": {
            "skipped": False,
            "top": [
                {
                    "work_id_a": a,
                    "work_id_b": b,
                    "score": 0.6,
                    "temporal": _space_metrics(2, 1, 0.8, 0.7),
                    "full_mean": _space_metrics(8, 9, 0.5, 0.4),
                }
            ],
        },
        "mode_c": {
            "skipped": False,
            "common": [mode_c_row(a, b, 0.95, True)],
            "visual_vs_peav": [mode_c_row(c, d, 0.3, False)],
            "counts": {"common_ge2": 1, "visual_vs_peav": 1},
        },
    }


def _unexpected_pairs_for_app(
    catalog: Catalog, config: Config, full_rep: int, work_ids: list[int]
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """用默认空间（Full Mean）的 SimilarityIndex 选一对互邻、两对非互邻。"""

    index = load_similarity_index(catalog, config, representation_id=full_rep)
    try:
        neighbor_sets = {
            work_id: {
                item.work_id
                for item in index.neighbors(work_id, kind="work_joint", k=5)
            }
            for work_id in work_ids
        }
    finally:
        index.close()

    def is_mutual(first: int, second: int) -> bool:
        return second in neighbor_sets[first] and first in neighbor_sets[second]

    pairs = [
        (first, second)
        for position, first in enumerate(work_ids)
        for second in work_ids[position + 1 :]
    ]
    mutual = [pair for pair in pairs if is_mutual(*pair)]
    non_mutual = [pair for pair in pairs if not is_mutual(*pair)]
    assert mutual, "合成数据应有互为近邻对"
    assert len(non_mutual) >= 2, "合成数据应有两对非互邻"
    return mutual[0], non_mutual[0], non_mutual[1]


def _write_unexpected(config: Config, payload: dict) -> None:
    config.paths.logs.mkdir(parents=True, exist_ok=True)
    (config.paths.logs / "unexpected_relations.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def _unexpected_table(app_test) -> object:
    tables = [element.value for element in app_test.dataframe]
    matches = [table for table in tables if "分数" in table.columns]
    assert len(matches) == 1, [list(table.columns) for table in tables]
    return matches[0]


def test_app_unexpected_relations_panel_and_modes(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    work_ids, full_rep = _seed_three_spaces(catalog, config, count=8)
    mutual, non_mutual_a, non_mutual_b = _unexpected_pairs_for_app(
        catalog, config, full_rep, work_ids
    )
    _write_unexpected(
        config, _unexpected_payload(work_ids, mutual, non_mutual_a, non_mutual_b)
    )

    app_test = _run_app(monkeypatch, repo_root, config)
    assert not app_test.exception
    # 默认地图/空间不变（Full Mean map_v002）
    assert app_test.selectbox(key="map_version").value == "map_v002"
    mode_select = app_test.selectbox(key="unexpected_mode")
    assert mode_select.value == "a"
    assert [str(option) for option in mode_select.options] == ["模式 A", "模式 B", "模式 C"]
    pair_select = app_test.selectbox(key="unexpected_pair")
    assert pair_select.value == f"{mutual[0]}:{mutual[1]}"
    assert app_test.checkbox(key="unexpected_only_mutual").value is False

    # 星图高亮这一对作品
    spec = _chart_spec(app_test)
    pair_trace = _chart_trace(spec, "意外关系配对")
    assert pair_trace is not None
    assert pair_trace["mode"] == "markers"
    assert len(_decode_array(pair_trace["x"])) == 2
    pair_line = _chart_trace(spec, "意外关系连线")
    assert pair_line is not None and pair_line["mode"] == "lines"
    assert len(_decode_array(pair_line["x"])) == 4  # 两段 (起点,None) 连线

    # 排序表：模式 A 共 3 行，按分数降序，含各空间统计
    table = _unexpected_table(app_test)
    assert len(table) == 3
    assert set(table.columns) >= {
        "分数",
        "A 作品",
        "A 链接",
        "B 作品",
        "B 链接",
        "类别",
        "互为近邻",
        "空间统计",
    }
    scores = [float(value) for value in table["分数"]]
    assert scores == sorted(scores, reverse=True)
    assert "rank=" in str(table.loc[0, "空间统计"]) or "rA=" in str(table.loc[0, "空间统计"])
    assert "pct=" in str(table.loc[0, "空间统计"])
    assert "cos=" in str(table.loc[0, "空间统计"])

    # 两者 Top-k 邻域（当前 Full Mean 空间只有 joint 模态）
    markdowns = [str(element.value) for element in app_test.markdown]
    assert any("A 作品 · Top5 邻域" in markdown for markdown in markdowns)
    assert any("B 作品 · Top5 邻域" in markdown for markdown in markdowns)
    neighbor_tables = [
        element.value
        for element in app_test.dataframe
        if "余弦相似度" in element.value.columns
    ]
    assert len(neighbor_tables) >= 3  # A + B + 既有选中作品面板

    # CSV 下载按钮存在
    assert len(app_test.get("download_button")) == 1

    # 模式 B：仅 1 行，高亮切到 B 模式的对
    app_test.selectbox(key="unexpected_mode").set_value("b").run()
    assert not app_test.exception
    assert app_test.selectbox(key="unexpected_pair").value == f"{mutual[0]}:{mutual[1]}"
    assert len(_unexpected_table(app_test)) == 1
    assert _chart_trace(_chart_spec(app_test), "意外关系配对") is not None

    # 模式 C：2 行，类别列区分 common/visual_vs_peav
    app_test.selectbox(key="unexpected_mode").set_value("c").run()
    assert not app_test.exception
    table_c = _unexpected_table(app_test)
    assert len(table_c) == 2
    categories = {str(value) for value in table_c["类别"]}
    assert categories == {"common", "visual_vs_peav"}
    assert [float(value) for value in table_c["分数"]] == sorted(
        [float(value) for value in table_c["分数"]], reverse=True
    )

    # 只看互为近邻：模式 A 只剩 mutual 一行，高亮只保留该对
    app_test.selectbox(key="unexpected_mode").set_value("a").run()
    app_test.checkbox(key="unexpected_only_mutual").set_value(True).run()
    assert not app_test.exception
    table_only = _unexpected_table(app_test)
    assert len(table_only) == 1
    assert app_test.selectbox(key="unexpected_pair").value == f"{mutual[0]}:{mutual[1]}"
    pair_trace = _chart_trace(_chart_spec(app_test), "意外关系配对")
    assert pair_trace is not None
    assert len(_decode_array(pair_trace["x"])) == 2

    # 染色变化不影响意外关系分数/表
    before = _unexpected_table(app_test).copy()
    app_test.selectbox(key="color_field").set_value("duration_s").run()
    assert not app_test.exception
    after = _unexpected_table(app_test)
    assert [float(value) for value in before["分数"]] == [
        float(value) for value in after["分数"]
    ]
    assert before["A 作品"].tolist() == after["A 作品"].tolist()

    # 3D 下配对高亮为 Scatter3d
    app_test.radio(key="projection").set_value("3d").run()
    assert not app_test.exception
    pair_trace = _chart_trace(_chart_spec(app_test), "意外关系配对")
    assert pair_trace is not None and pair_trace["type"] == "scatter3d"


def test_app_unexpected_relations_jump_buttons(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    work_ids, full_rep = _seed_three_spaces(catalog, config, count=8)
    mutual, non_mutual_a, non_mutual_b = _unexpected_pairs_for_app(
        catalog, config, full_rep, work_ids
    )
    _write_unexpected(
        config, _unexpected_payload(work_ids, mutual, non_mutual_a, non_mutual_b)
    )
    app_test = _run_app(monkeypatch, repo_root, config)
    app_test.button(key="unexpected_jump_a").click().run()
    assert not app_test.exception
    assert int(app_test.selectbox(key="manual_work").value) == mutual[0]
    app_test.button(key="unexpected_jump_b").click().run()
    assert not app_test.exception
    assert int(app_test.selectbox(key="manual_work").value) == mutual[1]


def test_app_unexpected_relations_missing_file(
    config: Config, catalog: Catalog, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    work_ids = _seed_app_data(catalog, config)
    _add_vectors(catalog, config, work_ids)
    app_test = _run_app(monkeypatch, repo_root, config)
    assert not app_test.exception
    captions = [str(element.value) for element in app_test.caption]
    assert any("未找到意外关系文件" in caption for caption in captions)
    names = [trace.get("name") for trace in _chart_spec(app_test)["data"]]
    assert "意外关系配对" not in names
    # 默认行为不变：旧地图、3 张近邻表、1 张图
    assert app_test.selectbox(key="map_version").value == "map_v001"
    assert len(app_test.get("plotly_chart")) == 1
    assert len(app_test.dataframe) == 3
