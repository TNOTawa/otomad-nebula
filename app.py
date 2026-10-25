"""音MAD 星云 Streamlit 可视化界面（迭代 A2：近邻探索与观察升级）。

启动：在项目根目录运行 streamlit run app.py（使用 .venv 中的 streamlit 可执行文件）。

环境变量 NEBULA_CONFIG 可指定配置文件（默认项目根目录 config.yaml），
tests/test_app.py 通过该变量注入临时数据库。

交互说明：

- 侧栏：星云版本、2D/3D 投影、染色维度（含高维近邻密度）、搜索、点大小/透明度、
  数据集/榜单/年份筛选、作品选择；
- 选中作品（2D 点击或侧栏下拉）后显示作品详情和三个模态的真实高维 Top5 近邻；
- 星图高亮选中作品与 Top5 近邻，2D 默认画出连线，3D 提供开关（默认关）；
- 近邻下拉可直接跳到近邻作品继续探索；
- 3D 点击选择受 Plotly/Streamlit 限制，侧栏“选择作品/跳到近邻作品”是等价入口；
- 没有数据或没有星云时显示中文提示，不抛 traceback。
"""

from __future__ import annotations

import html
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    filter_frame,
    load_nebula_frame,
    neighbor_rows,
    search_frame,
)
from nebula.config import Config, load_config
from nebula.db import Catalog
from nebula.errors import NebulaError
from nebula.similarity import KIND_LABELS, KINDS, SimilarityIndex, load_similarity_index

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
_MISSING_COLOR = "#b0b0b0"
_SELECTED_COLOR = "#ffd166"
_NEIGHBOR_COLOR = "#ef476f"
_LINE_COLOR = "rgba(255, 209, 102, 0.55)"
_HEIGHT_2D = 680
_HEIGHT_3D = 720
_NEIGHBOR_K = 5

_PROJECTION_LABELS = {"2d": "2D（平面）", "3d": "3D（三维）"}
_SOURCE_LABELS = {"all": "全部", "local": "本地", "bilibili": "Bilibili"}


@dataclass(frozen=True)
class _SimilarityState:
    """近邻索引加载结果：失败时保留中文原因，界面降级但不崩溃。"""

    index: SimilarityIndex | None
    error: str | None


def _config_path() -> Path:
    env_value = os.environ.get("NEBULA_CONFIG")
    return Path(env_value) if env_value else _DEFAULT_CONFIG


def _db_mtime_ns(db_path: Path) -> int:
    try:
        return db_path.stat().st_mtime_ns
    except OSError:
        return 0


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


@st.cache_resource(show_spinner="正在加载高维近邻索引…")
def _load_similarity_index(
    config_path: str,
    version: str,
    db_mtime_ns: int,
) -> _SimilarityState:
    """加载高维近邻索引；db 文件 mtime 参与缓存 key，新增向量后自动失效。"""

    del db_mtime_ns  # 仅用于缓存 key
    cfg = load_config(config_path)
    cat = Catalog(cfg.paths.db_path)
    try:
        cat.initialize()
        index = load_similarity_index(cat, cfg, version=version)
        return _SimilarityState(index=index, error=None)
    except NebulaError as exc:
        return _SimilarityState(index=None, error=str(exc))
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


def _work_label(row: pd.Series | Mapping[str, Any]) -> str:
    title = row.get("title") or row.get("file_name") or row.get("work_key") or "未知作品"
    text = str(title)
    if len(text) > 60:
        text = text[:57] + "…"
    file_name = row.get("file_name")
    if file_name and str(file_name) not in text:
        return f"{text} ｜ {file_name}"
    return text


def _describe_filter_spec(filter_spec: Mapping[str, Any] | None) -> str:
    if not filter_spec:
        return "全量"
    parts: list[str] = []
    dataset = filter_spec.get("dataset")
    if dataset:
        labels = {"local": "本地", "bilibili": "Bilibili"}
        parts.append(f"来源={labels.get(str(dataset), dataset)}")
    if filter_spec.get("vote_year") is not None:
        parts.append(f"选票年份={int(filter_spec['vote_year'])}")
    if filter_spec.get("list_key"):
        parts.append(f"榜单={filter_spec['list_key']}")
    if filter_spec.get("include_duplicates"):
        parts.append("包含重复实体")
    return "，".join(parts) if parts else "全量"


def _lookup_row(frame: pd.DataFrame, work_id: int) -> pd.Series | None:
    matches = frame.loc[frame["work_id"] == int(work_id)]
    if matches.empty:
        return None
    return matches.iloc[0]


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
    selected_work_id: int | None = None,
    neighbor_ids: Sequence[int] = (),
    show_lines: bool = False,
) -> go.Figure:
    """按投影方式构造 Plotly 散点图。

    除染色主 trace 外，选中作品与 Top5 近邻会用单独 trace 高亮；
    2D/3D 是否画连线由 show_lines 控制（3D 默认关）。
    """

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
        present_positions = np.where(present)[0]
        present_labels = [labels[index] for index in present_positions]
        marker: dict[str, Any] = {"size": int(point_size), "opacity": float(opacity)}
        if spec.categorical:
            marker.update(
                color=[int(values[index]) for index in present_positions],
                colorscale=spec.colorscale,
                cmin=0.0,
                cmax=float(max(1, len(spec.colorscale) - 1))
                if isinstance(spec.colorscale, list)
                else 1.0,
                colorbar=colorbar,
            )
        else:
            marker.update(
                color=[float(values[index]) for index in present_positions],
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
                go.Scatter3d(  # type: ignore[arg-type]
                    x=xs[present], y=ys[present], z=zs[present], **common  # type: ignore[index]
                )
            )
        else:
            traces.append(go.Scatter(x=xs[present], y=ys[present], **common))

    if missing.any():
        missing_positions = np.where(missing)[0]
        missing_labels = [labels[index] for index in missing_positions]
        common = {
            "mode": "markers",
            "marker": {"size": int(point_size), "opacity": float(opacity), "color": _MISSING_COLOR},
            "hovertext": missing_labels,
            "hovertemplate": "%{hovertext}<extra></extra>",
            "name": "缺失值/未分类",
        }
        if is_3d:
            traces.append(
                go.Scatter3d(  # type: ignore[arg-type]
                    x=xs[missing], y=ys[missing], z=zs[missing], **common  # type: ignore[index]
                )
            )
        else:
            traces.append(go.Scatter(x=xs[missing], y=ys[missing], **common))

    # ---------------------------------------------------------------- 近邻高亮
    positions = {int(work_id): index for index, work_id in enumerate(view["work_id"].tolist())}
    selected_position = None
    if selected_work_id is not None:
        selected_position = positions.get(int(selected_work_id))
    if selected_position is not None:
        selected_hover = labels[selected_position]
        if is_3d:
            traces.append(
                go.Scatter3d(  # type: ignore[arg-type]
                    x=[xs[selected_position]],
                    y=[ys[selected_position]],
                    z=[zs[selected_position]],  # type: ignore[index]
                    mode="markers",
                    marker={"size": point_size * 2.4, "color": _SELECTED_COLOR, "opacity": 1.0},
                    hovertext=[selected_hover],
                    hovertemplate="%{hovertext}<extra></extra>",
                    name="选中作品",
                )
            )
        else:
            traces.append(
                go.Scatter(
                    x=[xs[selected_position]],
                    y=[ys[selected_position]],
                    mode="markers",
                    marker={
                        "size": point_size * 2.6,
                        "color": _SELECTED_COLOR,
                        "symbol": "star",
                        "line": {"width": 1, "color": "#222222"},
                    },
                    hovertext=[selected_hover],
                    hovertemplate="%{hovertext}<extra></extra>",
                    name="选中作品",
                )
            )

    highlight_positions = [
        positions[int(work_id)] for work_id in neighbor_ids if int(work_id) in positions
    ]
    if highlight_positions:
        highlight_hover = [labels[index] for index in highlight_positions]
        if is_3d:
            traces.append(
                go.Scatter3d(  # type: ignore[arg-type]
                    x=xs[highlight_positions],
                    y=ys[highlight_positions],
                    z=zs[highlight_positions],  # type: ignore[index]
                    mode="markers",
                    marker={"size": point_size * 1.6, "color": _NEIGHBOR_COLOR, "opacity": 0.95},
                    hovertext=highlight_hover,
                    hovertemplate="%{hovertext}<extra></extra>",
                    name="Top5 近邻",
                )
            )
        else:
            traces.append(
                go.Scatter(
                    x=xs[highlight_positions],
                    y=ys[highlight_positions],
                    mode="markers",
                    marker={"size": point_size * 1.7, "color": _NEIGHBOR_COLOR, "opacity": 0.95},
                    hovertext=highlight_hover,
                    hovertemplate="%{hovertext}<extra></extra>",
                    name="Top5 近邻",
                )
            )
        if show_lines and selected_position is not None:
            line_x: list[float | None] = []
            line_y: list[float | None] = []
            line_z: list[float | None] = []
            for position in highlight_positions:
                line_x.extend([xs[selected_position], xs[position], None])
                line_y.extend([ys[selected_position], ys[position], None])
                if is_3d:
                    line_z.extend(
                        [zs[selected_position], zs[position], None]  # type: ignore[index]
                    )
            line_common: dict[str, Any] = {
                "mode": "lines",
                "line": {"color": _LINE_COLOR, "width": 1.5},
                "hoverinfo": "skip",
                "name": "近邻连线",
            }
            if is_3d:
                traces.append(go.Scatter3d(x=line_x, y=line_y, z=line_z, **line_common))
            else:
                traces.append(go.Scatter(x=line_x, y=line_y, **line_common))

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
    """把 Plotly 返回的点击坐标映射回 view 的行位置。"""

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
        return int(np.argmin(distance))
    return None


def _selected_highlight_ids(
    index: SimilarityIndex | None,
    selected_work_id: int | None,
) -> list[int]:
    """选中作品的联合模态 Top5 近邻 id（用于星图高亮/连线）。"""

    if index is None or selected_work_id is None:
        return []
    try:
        return [
            neighbor.work_id
            for neighbor in index.neighbors(selected_work_id, kind="work_joint", k=_NEIGHBOR_K)
        ]
    except NebulaError:
        return []


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
    local_file = row.get("local_file")
    if local_file and not local_path:
        lines.append(f"- 关联本地文件：{_safe(local_file)}")
    work_key = row.get("work_key")
    if work_key:
        lines.append(f"- 内部标识：{_safe(work_key)}")
    if lines:
        st.markdown(chr(10).join(lines))
    else:
        st.caption("该作品没有更多元数据。")


def _render_neighbor_panel(
    frame: pd.DataFrame,
    view: pd.DataFrame,
    index: SimilarityIndex | None,
    selected_work_id: int,
    error: str | None,
) -> None:
    """显示选中作品三个模态的 Top5 近邻，并提供跳到近邻作品的下拉入口。"""

    st.markdown("#### 高维近邻（真实余弦相似度）")
    if index is None:
        st.caption(f"高维近邻功能不可用：{error or '当前向量版本没有可读向量'}")
        return

    by_id = frame.set_index("work_id")
    jump_ids: list[int] = []
    for kind in KINDS:
        st.markdown(f"**{KIND_LABELS[kind]} Top{_NEIGHBOR_K}**")
        if not index.has_vector(selected_work_id, kind):
            st.caption(f"该作品无{KIND_LABELS[kind]}模态向量")
            continue
        try:
            rows = neighbor_rows(frame, index, selected_work_id, kind, _NEIGHBOR_K)
        except NebulaError as exc:
            st.caption(f"无该模态向量：{exc}")
            continue
        if not rows:
            st.caption("可比较作品不足，暂无近邻")
            continue
        table = pd.DataFrame(
            [
                {
                    "排名": row["rank"],
                    "作品": row["title"],
                    "余弦相似度": f"{float(row['similarity']):.4f}",
                }
                for row in rows
            ]
        )
        st.dataframe(table, width="stretch", hide_index=True)
        jump_ids.extend(int(row["work_id"]) for row in rows)

    if not jump_ids:
        return
    unique_ids = list(dict.fromkeys(jump_ids))
    jump_labels: dict[int, str] = {}
    for work_id in unique_ids:
        row = by_id.loc[work_id] if work_id in by_id.index else None
        jump_labels[work_id] = _work_label(row) if row is not None else f"作品 {work_id}"
    if st.session_state.get("neighbor_jump") not in unique_ids:
        st.session_state.pop("neighbor_jump", None)
    st.selectbox(
        "跳到近邻作品继续探索",
        options=unique_ids,
        index=None,
        placeholder="选择一部近邻作品…",
        format_func=lambda work_id: jump_labels.get(int(work_id), str(work_id)),
        key="neighbor_jump",
        on_change=_on_neighbor_jump,
        args=("neighbor_jump",),
    )


def _on_neighbor_jump(selectbox_key: str) -> None:
    """近邻下拉变化时记录待跳转作品，下一次 rerun 生效。"""

    value = st.session_state.get(selectbox_key)
    if value is not None:
        st.session_state["pending_work_id"] = int(value)


# --------------------------------------------------------------------------- 侧栏
def _render_sidebar_map_settings(
    versions: list[str],
    latest: str,
    cfg: Config,
) -> tuple[str, str]:
    with st.sidebar:
        st.header("显示设置")
        if st.button("刷新数据", key="refresh_data", help="重新读取数据库、星云版本与高维向量索引"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        # 默认打开最新的总星云；用户手动切换后由 selectbox 的 key 记住选择。
        # 不在这里写 session_state（否则 Streamlit 会警告“default + Session State 同时设置”）。
        current_version = st.session_state.get("map_version")
        if current_version is not None and current_version not in versions:
            st.session_state.pop("map_version", None)
            current_version = None
        default_index = (
            versions.index(current_version)
            if current_version in versions
            else versions.index(latest)
        )
        version = st.selectbox(
            "星云版本",
            options=versions,
            index=default_index,
            key="map_version",
            help="默认打开最新的总星云；可切换到 2025 独立星云或 n_neighbors 变体（坐标不可直接比较）",
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
    *,
    has_similarity: bool,
) -> tuple[str, str, int, float]:
    fields = available_color_fields(frame, has_similarity=has_similarity)
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


def _render_sidebar_filters(
    frame: pd.DataFrame,
    map_info: Mapping[str, Any],
) -> tuple[str, list[str], list[int]]:
    """数据集 / 榜单 / 年份筛选；只影响界面显示，不改变已生成的星云坐标。"""

    list_options = sorted(
        {str(key) for keys in frame["list_keys"].tolist() for key in (keys or [])}
    )
    year_options = sorted(
        {int(year) for years in frame["vote_years"].tolist() for year in (years or [])}
    )
    with st.sidebar:
        st.subheader("筛选")
        source = st.selectbox(
            "数据集",
            options=["all", "local", "bilibili"],
            index=0,
            format_func=lambda value: _SOURCE_LABELS.get(value, value),
            key="filter_source",
        )
        selected_lists = st.multiselect(
            "榜单（留空 = 全部）",
            options=list_options,
            default=[],
            key="filter_lists",
        )
        selected_years = st.multiselect(
            "选票年份（留空 = 全部）",
            options=year_options,
            default=[],
            format_func=lambda year: f"{year}年",
            key="filter_years",
        )
        filter_spec = map_info.get("filter_spec")
        if filter_spec:
            st.caption(f"该星云构建时筛选：{_describe_filter_spec(filter_spec)}")
    return (
        str(source),
        [str(value) for value in selected_lists],
        [int(value) for value in selected_years],
    )


def _render_sidebar_selection(
    view: pd.DataFrame,
    projection: str,
) -> tuple[int | None, bool]:
    """作品选择与近邻连线开关；图表点击/近邻跳转都会同步到这个下拉框。"""

    options = [int(work_id) for work_id in view["work_id"].tolist()]
    if not options:
        return None, False
    labels = {int(row["work_id"]): _work_label(row) for _, row in view.iterrows()}

    pending = st.session_state.pop("pending_work_id", None)
    if pending is not None:
        st.session_state.pop("neighbor_jump", None)
        if int(pending) in options:
            st.session_state["manual_work"] = int(pending)
    if st.session_state.get("manual_work") not in options:
        st.session_state["manual_work"] = options[0]

    with st.sidebar:
        st.subheader("作品选择")
        manual_id = st.selectbox(
            "选择作品（点击图表或用下拉切换）",
            options=options,
            format_func=lambda work_id: labels.get(int(work_id), str(work_id)),
            key="manual_work",
        )
        lines_label = (
            "显示近邻连线（3D，默认关）" if projection == "3d" else "显示近邻连线（2D）"
        )
        show_lines = st.checkbox(
            lines_label,
            value=(projection != "3d"),
            key=f"show_lines_{projection}",
        )
    return (int(manual_id) if manual_id is not None else None), bool(show_lines)


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
        st.warning("还没有可用的星云数据。请先完成作品向量提取，再运行：")
        st.code("map.py build", language="text")
        st.info(f"数据库：{cfg.paths.db_path}")
        return

    # 默认打开“最新的总星云”（没有筛选条件的版本）；筛选地图与变体坐标不可直接比较，
    # 让用户先看到全量空间，再主动切到 2025 独立星云/n_neighbors 变体。
    total_versions = [
        str(row["version"])
        for row in maps
        if str(row.get("status")) == "ready" and not row.get("filter_json")
    ]
    latest = total_versions[-1] if total_versions else versions[-1]
    version, projection = _render_sidebar_map_settings(versions, latest, cfg)

    try:
        map_info, frame = _load_frame(config_path, version)
    except NebulaError as exc:
        st.error(f"读取星云数据失败：{exc}")
        return
    if frame.empty:
        st.warning(f"星云版本 {version} 没有作品坐标，请重新运行 map.py build。")
        return

    embedding_version = str(map_info.get("embedding_version") or cfg.embedding_version)
    similarity_state = _load_similarity_index(
        config_path, embedding_version, _db_mtime_ns(cfg.paths.db_path)
    )
    index = similarity_state.index
    if index is not None:
        st.sidebar.caption(
            f"高维近邻已就绪：{len(index.work_ids)} 部作品，"
            f"模态 {', '.join(KIND_LABELS[kind] for kind in index.available_kinds)}"
        )

    color_field, query, point_size, opacity = _render_sidebar_display_settings(
        frame, cfg, has_similarity=index is not None
    )
    if similarity_state.error:
        st.sidebar.caption(f"高维近邻不可用：{similarity_state.error}")
    source, selected_lists, selected_years = _render_sidebar_filters(frame, map_info)

    try:
        spec = apply_color(
            frame,
            color_field,
            projection=projection,
            density_grid=int(cfg.app.density_grid),
            similarity_index=index,
        )
    except NebulaError as exc:
        st.error(f"计算染色数据失败：{exc}")
        return

    view = search_frame(frame, query)
    view = filter_frame(
        view,
        sources=[source] if source != "all" else None,
        list_keys=selected_lists,
        vote_years=selected_years,
    )
    if view.empty:
        if query:
            st.info(f"没有匹配「{query}」的作品，请调整搜索关键词。")
        else:
            st.info("当前筛选条件下没有作品，请调整筛选条件。")
        return

    selected_id, show_lines = _render_sidebar_selection(view, projection)
    neighbor_ids = _selected_highlight_ids(index, selected_id)
    positions = view.index.to_numpy()
    view_values = [spec.values[int(position)] for position in positions]
    chart_title = (
        f"{cfg.app.title} · {version} · {projection.upper()} · {spec.title}"
    )
    figure = _build_figure(
        view,
        view_values,
        spec,
        projection=projection,
        point_size=point_size,
        opacity=opacity,
        title=chart_title,
        selected_work_id=selected_id,
        neighbor_ids=neighbor_ids,
        show_lines=show_lines,
    )
    event = st.plotly_chart(
        figure,
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key="nebula_chart",
    )

    clicked = _clicked_index(event, view, projection)
    if clicked is not None:
        clicked_work_id = int(view.iloc[clicked]["work_id"])
        if clicked_work_id != st.session_state.get("_last_chart_click"):
            st.session_state["_last_chart_click"] = clicked_work_id
            st.session_state["pending_work_id"] = clicked_work_id
            st.rerun()

    st.caption(
        f"当前显示 {len(view)} / {len(frame)} 部作品；"
        "2D 点击数据点或使用侧栏下拉选择作品，选中后显示三个模态的高维近邻。"
    )

    selected_row = _lookup_row(frame, selected_id) if selected_id is not None else None
    if selected_row is not None and selected_id is not None:
        st.divider()
        _render_detail(selected_row, cfg)
        _render_neighbor_panel(frame, view, index, selected_id, similarity_state.error)

    st.caption(
        f"星云版本 {version} · 构建筛选 {_describe_filter_spec(map_info.get('filter_spec'))} · "
        f"向量版本 {embedding_version} · 作品数 {map_info.get('n_works')} · "
        f"创建于 {map_info.get('created_at')}"
    )


if __name__ == "__main__":
    main()
