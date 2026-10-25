"""音MAD 星云 Streamlit 可视化界面。

启动：在项目根目录运行 streamlit run app.py（使用 .venv 中的 streamlit 可执行文件）。

环境变量 NEBULA_CONFIG 可指定配置文件（默认项目根目录 config.yaml），
tests/test_app.py 通过该变量注入临时数据库。

交互说明：

- 侧栏可切换星云版本、2D/3D 投影、染色维度，并可搜索作品、调整点大小/透明度；
- 2D 视图支持 Plotly 原生点选，点击后主图下方显示作品详情；
- 3D 视图的点击选择受 Plotly/Streamlit 选择机制限制，可能不会触发，
  因此界面始终提供“搜索 + 下拉选择作品”的等价入口；
- 没有数据或没有星云时显示中文提示，不抛 traceback。
"""

from __future__ import annotations

import html
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nebula.appdata import (
    ColorSpec,
    apply_color,
    available_color_fields,
    external_url,
    load_nebula_frame,
    search_frame,
)
from nebula.config import Config, load_config
from nebula.db import Catalog
from nebula.errors import NebulaError

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
_MISSING_COLOR = "#b0b0b0"
_HEIGHT_2D = 680
_HEIGHT_3D = 720

_PROJECTION_LABELS = {"2d": "2D（平面）", "3d": "3D（三维）"}


def _config_path() -> Path:
    env_value = os.environ.get("NEBULA_CONFIG")
    return Path(env_value) if env_value else _DEFAULT_CONFIG


# --------------------------------------------------------------------------- 缓存入口
# 所有数据库读取都用 st.cache_data 包装，避免每次交互都重新查询 SQLite。
@st.cache_resource(show_spinner=False)
def _load_config_cached(config_path: str) -> Config:
    return load_config(config_path)


@st.cache_data(show_spinner=False)
def _load_map_list(config_path: str) -> list[dict[str, Any]]:
    cfg = load_config(config_path)
    cat = Catalog(cfg.paths.db_path)
    try:
        cat.initialize()
        return [dict(row) for row in cat.list_maps()]
    finally:
        cat.close()


@st.cache_data(show_spinner="正在读取星云数据…")
def _load_frame(config_path: str, map_version: str) -> tuple[dict[str, Any], pd.DataFrame]:
    cfg = load_config(config_path)
    cat = Catalog(cfg.paths.db_path)
    try:
        cat.initialize()
        return load_nebula_frame(cat, cfg, map_version=map_version)
    finally:
        cat.close()


# --------------------------------------------------------------------------- 小工具
def _safe(value: Any) -> str:
    """转义用户数据，避免破坏 hover / markdown 结构。"""

    return html.escape(str(value), quote=True).replace("[", "&#91;").replace("]", "&#93;")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # pragma: no cover - 防御性分支
        return False


def _format_duration(seconds: Any) -> str:
    total = int(round(float(seconds)))
    minutes, second = divmod(max(total, 0), 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{hours}:{minutes:02d}:{second:02d}"
    return f"{minutes}:{second:02d}"


def _work_label(row: pd.Series) -> str:
    title = row.get("title") or row.get("file_name") or row.get("work_key") or "未知作品"
    text = str(title)
    if len(text) > 60:
        text = text[:57] + "…"
    file_name = row.get("file_name")
    if file_name and str(file_name) not in text:
        return f"{text} ｜ {file_name}"
    return text


# --------------------------------------------------------------------------- 图表
def _build_figure(
    view: pd.DataFrame,
    values: list,
    spec: ColorSpec,
    *,
    projection: str,
    point_size: int,
    opacity: float,
    title: str,
) -> go.Figure:
    """按投影方式构造 Plotly 散点图；缺失染色值的点用灰色单独绘制。"""

    is_3d = projection == "3d"
    x_column, y_column = ("x3", "y3") if is_3d else ("x2", "y2")
    xs = view[x_column].to_numpy(dtype=float)
    ys = view[y_column].to_numpy(dtype=float)
    zs = view["z3"].to_numpy(dtype=float) if is_3d else None
    labels = view["hover_text"].tolist()
    missing = np.array([value is None for value in values], dtype=bool)
    present = ~missing
    colorbar: dict[str, Any] = {"title": spec.title}
    if spec.tickvals is not None and spec.ticktext is not None:
        colorbar["tickvals"] = list(spec.tickvals)
        colorbar["ticktext"] = list(spec.ticktext)

    traces: list[Any] = []
    if present.any():
        present_labels = [labels[index] for index in np.where(present)[0]]
        marker: dict[str, Any] = {"size": int(point_size), "opacity": float(opacity)}
        if spec.categorical:
            marker.update(
                color=[int(values[index]) for index in np.where(present)[0]],
                colorscale=spec.colorscale,
                cmin=0.0,
                cmax=float(max(1, len(spec.colorscale) - 1))
                if isinstance(spec.colorscale, list)
                else 1.0,
                colorbar=colorbar,
            )
        else:
            marker.update(
                color=[float(values[index]) for index in np.where(present)[0]],
                colorscale=spec.colorscale,
                colorbar=colorbar,
            )
        common = {
            "mode": "markers",
            "marker": marker,
            "hovertext": present_labels,
            "hovertemplate": "%{hovertext}<extra></extra>",
            "name": spec.title,
        }
        if is_3d:
            traces.append(
                go.Scatter3d(x=xs[present], y=ys[present], z=zs[present], **common)  # type: ignore[index]
            )
        else:
            traces.append(go.Scatter(x=xs[present], y=ys[present], **common))

    if missing.any():
        missing_labels = [labels[index] for index in np.where(missing)[0]]
        common = {
            "mode": "markers",
            "marker": {"size": int(point_size), "opacity": float(opacity), "color": _MISSING_COLOR},
            "hovertext": missing_labels,
            "hovertemplate": "%{hovertext}<extra></extra>",
            "name": "缺失值/未分类",
        }
        if is_3d:
            traces.append(
                go.Scatter3d(x=xs[missing], y=ys[missing], z=zs[missing], **common)  # type: ignore[index]
            )
        else:
            traces.append(go.Scatter(x=xs[missing], y=ys[missing], **common))

    figure = go.Figure(data=traces)
    figure.update_layout(
        title=title,
        template="plotly_dark",
        height=_HEIGHT_3D if is_3d else _HEIGHT_2D,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        legend={"itemsizing": "constant"},
        hoverlabel={"font_size": 12},
    )
    if is_3d:
        figure.update_layout(
            scene={
                "xaxis": {"title": "UMAP-1", "showgrid": False, "zeroline": False},
                "yaxis": {"title": "UMAP-2", "showgrid": False, "zeroline": False},
                "zaxis": {"title": "UMAP-3", "showgrid": False, "zeroline": False},
                "aspectmode": "cube",
            }
        )
    else:
        figure.update_layout(
            xaxis={"title": "UMAP-1", "showgrid": False, "zeroline": False},
            yaxis={"title": "UMAP-2", "showgrid": False, "zeroline": False},
        )
    return figure


def _event_points(event: Any) -> list[Mapping[str, Any]]:
    """从 st.plotly_chart 的 on_select 返回中取出被选中的点。"""

    if not isinstance(event, Mapping):
        return []
    selection = event.get("selection")
    if not isinstance(selection, Mapping):
        return []
    points = selection.get("points") or []
    return [point for point in points if isinstance(point, Mapping)]


def _clicked_index(event: Any, view: pd.DataFrame, projection: str) -> int | None:
    """把 Plotly 返回的点击坐标映射回 DataFrame 行索引。"""

    points = _event_points(event)
    if not points or len(view) == 0:
        return None
    if projection == "3d":
        x_column, y_column, z_column = "x3", "y3", "z3"
    else:
        x_column, y_column, z_column = "x2", "y2", "z3"
    xs = view[x_column].to_numpy(dtype=float)
    ys = view[y_column].to_numpy(dtype=float)
    zs = view[z_column].to_numpy(dtype=float)
    for point in points:
        px, py = point.get("x"), point.get("y")
        if px is None or py is None:
            continue
        try:
            distance = (xs - float(px)) ** 2 + (ys - float(py)) ** 2
            if projection == "3d":
                pz = point.get("z")
                if pz is None:
                    continue
                distance = distance + (zs - float(pz)) ** 2
        except (TypeError, ValueError):
            continue
        return int(view.index[int(np.argmin(distance))])
    return None


# --------------------------------------------------------------------------- 详情
def _render_detail(row: pd.Series, cfg: Config) -> None:
    """展示一部作品的详细信息（标题、链接、时长、作者、排名、本地文件）。"""

    title = row.get("title") or row.get("file_name") or row.get("work_key") or "未知作品"
    st.subheader(str(title))
    lines: list[str] = []
    url = external_url(row)
    if url:
        bvid = row.get("bvid")
        label = f"Bilibili 视频（{_safe(bvid)}）" if bvid else "在 Bilibili 打开"
        lines.append(f"- 链接：[{label}]({url})")
    elif row.get("url"):
        lines.append(f"- 原始链接：{_safe(row.get('url'))}")
    duration = row.get("duration_s")
    if not _is_missing(duration):
        lines.append(f"- 时长：{_format_duration(duration)}")
    publish_date = row.get("publish_date")
    if publish_date:
        lines.append(f"- 发布日期：{_safe(publish_date)}")
    author_name = row.get("author_name")
    author_uid = row.get("author_uid")
    if author_name or author_uid:
        author = _safe(author_name) if author_name else ""
        if author_uid:
            author = f"{author}（UID {_safe(author_uid)}）" if author else f"UID {_safe(author_uid)}"
        lines.append(f"- 作者：{author}")
    rank = row.get("rank")
    votes = row.get("votes")
    vote_year = row.get("vote_year")
    if not _is_missing(rank) or not _is_missing(votes):
        pieces = []
        if vote_year is not None and not _is_missing(vote_year):
            pieces.append(f"{int(float(vote_year))}年")
        if not _is_missing(rank):
            pieces.append(f"第 {int(float(rank))} 名")
        if not _is_missing(votes):
            pieces.append(f"{int(float(votes))} 票")
        lines.append(f"- 选票成绩：{' / '.join(pieces)}")
    list_names = row.get("list_names")
    if list_names:
        lines.append(f"- 所属榜单：{_safe(list_names)}")
    source = row.get("source")
    if source:
        lines.append(f"- 来源：{_safe(source)}")
    local_path = row.get("local_path")
    if local_path:
        lines.append(f"- 本地文件：{_safe(local_path)}")
    work_key = row.get("work_key")
    if work_key:
        lines.append(f"- 内部标识：{_safe(work_key)}")
    if lines:
        st.markdown(chr(10).join(lines))
    else:
        st.caption("该作品没有更多元数据。")


# --------------------------------------------------------------------------- 侧栏
def _render_sidebar_map_settings(
    versions: list[str],
    latest: str,
    cfg: Config,
) -> tuple[str, str]:
    with st.sidebar:
        st.header("显示设置")
        if st.button("刷新数据", key="refresh_data", help="重新读取数据库与星云版本"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        if st.session_state.get("map_version") not in versions:
            st.session_state["map_version"] = latest
        version = st.selectbox(
            "星云版本",
            options=versions,
            index=versions.index(st.session_state["map_version"]),
            key="map_version",
        )
        projection = st.radio(
            "投影方式",
            options=["2d", "3d"],
            index=0,
            format_func=lambda value: _PROJECTION_LABELS.get(value, value),
            key="projection",
        )
    return str(version), str(projection)


def _render_sidebar_display_settings(
    frame: pd.DataFrame,
    cfg: Config,
) -> tuple[str, str, int, float]:
    fields = available_color_fields(frame)
    labels = dict(fields)
    field_ids = [field_id for field_id, _ in fields]
    with st.sidebar:
        if st.session_state.get("color_field") not in field_ids:
            st.session_state["color_field"] = field_ids[0]
        color_field = st.selectbox(
            "染色维度",
            options=field_ids,
            format_func=lambda field_id: labels.get(field_id, field_id),
            key="color_field",
        )
        query = st.text_input(
            "搜索（文件名 / 标题 / BV 号）",
            value="",
            key="search",
            placeholder="例如：BV1xx411c7mD 或作品标题",
        )
        point_size = st.slider(
            "点大小",
            min_value=2,
            max_value=20,
            value=int(cfg.app.point_size),
            step=1,
            key="point_size",
        )
        opacity = st.slider(
            "透明度",
            min_value=0.1,
            max_value=1.0,
            value=float(cfg.app.opacity),
            step=0.05,
            key="opacity",
        )
    return str(color_field), str(query), int(point_size), float(opacity)


# --------------------------------------------------------------------------- 主界面
def main() -> None:
    """Streamlit 页面入口。"""

    config_path = str(_config_path())
    try:
        cfg = _load_config_cached(config_path)
    except NebulaError as exc:
        st.set_page_config(page_title="音MAD 星云", layout="wide")
        st.error(f"无法加载配置：{exc}")
        st.caption(f"配置文件：{config_path}（可用环境变量 NEBULA_CONFIG 指定）")
        return

    st.set_page_config(page_title=cfg.app.title, page_icon="🌌", layout="wide")
    st.title(f"🌌 {cfg.app.title}")
    st.caption(
        "音MAD 作品的多模态向量经 UMAP 投影得到的星云；"
        "年份、作者、排名、时长等外部数据只用于染色，不参与降维。"
    )

    try:
        maps = _load_map_list(config_path)
    except NebulaError as exc:
        st.error(f"读取星云版本列表失败：{exc}")
        return
    versions = [str(row["version"]) for row in maps if str(row.get("status")) == "ready"]
    if not versions:
        st.warning(
            "还没有可用的星云数据。请先完成作品向量提取，再运行："
        )
        st.code("map.py build", language="text")
        st.info(f"数据库：{cfg.paths.db_path}")
        return

    latest = versions[-1]
    version, projection = _render_sidebar_map_settings(versions, latest, cfg)

    try:
        map_info, frame = _load_frame(config_path, version)
    except NebulaError as exc:
        st.error(f"读取星云数据失败：{exc}")
        return
    if frame.empty:
        st.warning(f"星云版本 {version} 没有作品坐标，请重新运行 map.py build。")
        return

    color_field, query, point_size, opacity = _render_sidebar_display_settings(frame, cfg)

    try:
        spec = apply_color(
            frame,
            color_field,
            projection=projection,
            density_grid=int(cfg.app.density_grid),
        )
    except NebulaError as exc:
        st.error(f"计算染色数据失败：{exc}")
        return

    view = search_frame(frame, query)
    if view.empty:
        st.info(f"没有匹配「{query}」的作品，请调整搜索关键词。")
        return

    positions = view.index.to_numpy()
    view_values = [spec.values[int(position)] for position in positions]
    chart_title = f"{cfg.app.title} · {version} · {projection.upper()} · {spec.title}"
    figure = _build_figure(
        view,
        view_values,
        spec,
        projection=projection,
        point_size=point_size,
        opacity=opacity,
        title=chart_title,
    )
    event = st.plotly_chart(
        figure,
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key="nebula_chart",
    )

    clicked = _clicked_index(event, view, projection)
    st.caption(
        f"当前显示 {len(view)} / {len(frame)} 部作品；"
        "2D 视图点击数据点可查看详情，3D 视图或图表选择不可用时可用下方下拉框。"
    )
    if clicked is not None:
        st.caption("已通过图表点击选中作品：")
        _render_detail(frame.loc[clicked], cfg)

    st.divider()
    options = [int(work_id) for work_id in view["work_id"].tolist()]
    labels = {int(row["work_id"]): _work_label(row) for _, row in view.iterrows()}
    manual_id = st.selectbox(
        "或手动选择作品查看详情（3D 点击的等价入口）",
        options=options,
        format_func=lambda work_id: labels.get(int(work_id), str(work_id)),
        key="manual_work",
    )
    if clicked is None and manual_id is not None:
        selected = view.loc[view["work_id"] == int(manual_id)]
        if not selected.empty:
            _render_detail(selected.iloc[0], cfg)

    st.caption(
        f"星云版本 {version} · 向量版本 {map_info.get('embedding_version')} · "
        f"作品数 {map_info.get('n_works')} · 创建于 {map_info.get('created_at')}"
    )


if __name__ == "__main__":
    main()
