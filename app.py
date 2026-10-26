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

import contextlib
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

from nebula import representations
from nebula.appdata import (
    ColorSpec,
    apply_color,
    available_color_fields,
    clip_heatmap_data,
    clip_pair_rows,
    consensus_edges_for_visible,
    consensus_neighbors,
    diagnostics_artifacts,
    external_url,
    filter_frame,
    hub_rows,
    load_event_timeline,
    load_nebula_frame,
    load_neighbor_consensus,
    load_representation_options,
    load_unexpected_relations,
    map_matches_representation,
    mutual_neighbor_edges,
    neighbor_rows,
    pick_mutual_edge_kind,
    search_frame,
    unexpected_modes,
    unexpected_pair_rows,
    unexpected_rows_csv,
)
from nebula.config import Config, load_config
from nebula.db import Catalog
from nebula.errors import NebulaError
from nebula.similarity import (
    CLIP_KIND_LABELS,
    KIND_LABELS,
    KINDS,
    SimilarityIndex,
    load_similarity_index,
)

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
_MISSING_COLOR = "#b0b0b0"
_SELECTED_COLOR = "#ffd166"
_NEIGHBOR_COLOR = "#ef476f"
_LINE_COLOR = "rgba(255, 209, 102, 0.55)"
_HEIGHT_2D = 680
_HEIGHT_3D = 720
_HEIGHT_CLIP_HEATMAP = 520
_NEIGHBOR_K = 5
_MUTUAL_EDGE_COLOR = "rgba(126, 200, 227, 0.50)"
_CONSENSUS_EDGE_COLOR = "rgba(144, 190, 109, 0.60)"
_MUTUAL_TRACE_NAME = "高维互为近邻"
_CONSENSUS_TRACE_NAME = "跨模型共识边"

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


@st.cache_data(show_spinner=False)
def _load_representation_options_cached(config_path: str) -> list[dict[str, Any]]:
    cfg = load_config(config_path)
    cat = Catalog(cfg.paths.db_path)
    try:
        cat.initialize()
        return load_representation_options(cat, cfg)
    finally:
        cat.close()


@st.cache_data(show_spinner=False)
def _load_timeline_cached(
    config_path: str,
    sequence_version: str,
    work_id: int,
) -> dict[str, Any] | None:
    cfg = load_config(config_path)
    cat = Catalog(cfg.paths.db_path)
    try:
        cat.initialize()
        return load_event_timeline(cat, cfg, sequence_version, work_id)
    finally:
        cat.close()


@st.cache_data(show_spinner=False)
def _load_consensus_cached(config_path: str) -> dict[str, Any] | None:
    cfg = load_config(config_path)
    return load_neighbor_consensus(cfg)


@st.cache_data(show_spinner=False)
def _load_unexpected_relations_cached(config_path: str) -> dict[str, Any] | None:
    cfg = load_config(config_path)
    return load_unexpected_relations(cfg)


@st.cache_resource(show_spinner="正在加载高维近邻索引…")
def _load_similarity_index(
    config_path: str,
    version: str,
    representation_id: int | None,
    db_mtime_ns: int,
) -> _SimilarityState:
    """加载高维近邻索引；db 文件 mtime 与表征 id 参与缓存 key。"""

    del db_mtime_ns  # 仅用于缓存 key
    cfg = load_config(config_path)
    cat = Catalog(cfg.paths.db_path)
    try:
        cat.initialize()
        index = load_similarity_index(
            cat, cfg, version=version, representation_id=representation_id
        )
        return _SimilarityState(index=index, error=None)
    except NebulaError as exc:
        return _SimilarityState(index=None, error=str(exc))
    finally:
        cat.close()


def _build_representation_spaces(
    maps: list[dict[str, Any]],
    rep_options: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """把表征选项与 ready 地图合并成侧栏可切换的“表征空间”列表。"""

    ready_rows = [row for row in maps if str(row.get("status")) == "ready"]
    spaces: list[dict[str, Any]] = []
    for option in rep_options:
        rows = [row for row in ready_rows if map_matches_representation(row, option)]
        if not rows:
            continue
        versions = [str(row["version"]) for row in rows]
        default_version = option.get("map_version")
        if default_version not in versions:
            default_version = versions[-1]
        spaces.append({**option, "versions": versions, "default_version": default_version})
    if spaces:
        return spaces
    # 数据库里还没有表征登记（旧库/测试环境）：所有 ready 地图归入一个 legacy 空间
    versions = [str(row["version"]) for row in ready_rows]
    unfiltered = [row for row in ready_rows if not row.get("filter_json")]
    default_version = (
        str(unfiltered[-1]["version"]) if unfiltered else versions[-1]
    )
    return [
        {
            "representation_id": None,
            "representation_name": representations.LEGACY_NAME,
            "label": representations.representation_label(representations.LEGACY_NAME),
            "kind": "legacy",
            "sequence_version": None,
            "map_version": default_version,
            "n_maps": len(ready_rows),
            "registered": False,
            "versions": versions,
            "default_version": default_version,
        }
    ]


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


def _edge_work_id(edge: Mapping[str, Any], key: str) -> int:
    return int(edge[key])


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
    mutual_edges: Sequence[Mapping[str, Any]] = (),
    mutual_trace_name: str = _MUTUAL_TRACE_NAME,
    mutual_line_color: str = _MUTUAL_EDGE_COLOR,
    pair_work_ids: Sequence[int] = (),
) -> go.Figure:
    """按投影方式构造 Plotly 散点图。

    除染色主 trace 外，选中作品与 Top5 近邻会用单独 trace 高亮；
    2D/3D 是否画连线由 show_lines 控制（3D 默认关）；
    mutual_edges 为可见作品间的高维互邻/共识边，叠加为单条 line trace。
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

    positions = {int(work_id): index for index, work_id in enumerate(view["work_id"].tolist())}

    # ---------------------------------------------------------------- 互邻/共识边
    if mutual_edges:
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        edge_z: list[float | None] = []
        for edge in mutual_edges:
            first = positions.get(_edge_work_id(edge, "a_work_id"))
            second = positions.get(_edge_work_id(edge, "b_work_id"))
            if first is None or second is None:
                continue
            edge_x.extend([xs[first], xs[second], None])
            edge_y.extend([ys[first], ys[second], None])
            if is_3d:
                edge_z.extend([zs[first], zs[second], None])  # type: ignore[index]
        if edge_x:
            line_common: dict[str, Any] = {
                "mode": "lines",
                "line": {"color": mutual_line_color, "width": 1.0},
                "hoverinfo": "skip",
                "name": mutual_trace_name,
            }
            if is_3d:
                traces.append(
                    go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, **line_common)
                )
            else:
                traces.append(go.Scatter(x=edge_x, y=edge_y, **line_common))

    # ---------------------------------------------------------------- 近邻高亮
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

    # ---------------------------------------------------------------- 意外关系配对高亮
    if pair_work_ids:
        pair_positions = [
            positions[int(work_id)]
            for work_id in pair_work_ids
            if int(work_id) in positions
        ]
        if pair_positions:
            pair_labels = [labels[index] for index in pair_positions]
            pair_marker: dict[str, Any] = {
                "size": point_size * 2.2,
                "color": "#00d1b2",
                "symbol": "diamond",
                "line": {"width": 1, "color": "#083344"},
            }
            if is_3d:
                traces.append(
                    go.Scatter3d(  # type: ignore[arg-type]
                        x=xs[pair_positions],
                        y=ys[pair_positions],
                        z=zs[pair_positions],  # type: ignore[index]
                        mode="markers",
                        marker=pair_marker,
                        hovertext=pair_labels,
                        hovertemplate="%{hovertext}<extra></extra>",
                        name="意外关系配对",
                    )
                )
            else:
                traces.append(
                    go.Scatter(
                        x=xs[pair_positions],
                        y=ys[pair_positions],
                        mode="markers",
                        marker=pair_marker,
                        hovertext=pair_labels,
                        hovertemplate="%{hovertext}<extra></extra>",
                        name="意外关系配对",
                    )
                )
            if len(pair_positions) >= 2:
                pair_line_x: list[float | None] = []
                pair_line_y: list[float | None] = []
                pair_line_z: list[float | None] = []
                for position in pair_positions[:2]:
                    pair_line_x.extend([xs[position], None])
                    pair_line_y.extend([ys[position], None])
                    if is_3d:
                        pair_line_z.extend([zs[position], None])  # type: ignore[index]
                pair_line_common: dict[str, Any] = {
                    "mode": "lines",
                    "line": {"color": "rgba(0, 209, 178, 0.75)", "width": 1.6, "dash": "dash"},
                    "hoverinfo": "skip",
                    "name": "意外关系连线",
                }
                if is_3d:
                    traces.append(
                        go.Scatter3d(
                            x=pair_line_x, y=pair_line_y, z=pair_line_z, **pair_line_common
                        )
                    )
                else:
                    traces.append(
                        go.Scatter(x=pair_line_x, y=pair_line_y, **pair_line_common)
                    )

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



# --------------------------------------------------------------------------- 共识摘要
def _render_consensus_summary(
    consensus: Mapping[str, Any] | None,
    frame: pd.DataFrame,
    work_id: int,
) -> None:
    """详情面板展示当前作品的跨空间/跨模型共识近邻摘要（有 consensus JSON 时）。"""

    if consensus is None:
        return
    rows = consensus_neighbors(consensus, work_id, min_count=2, limit=5)
    with st.expander("跨空间 / 跨模型共识近邻", expanded=False):
        if not rows:
            st.caption(
                "当前作品在共识图（logs/neighbor_consensus.json）中"
                "没有 >=2 个空间的近邻边。"
            )
            return
        by_id = frame.set_index("work_id")
        lines: list[str] = []
        for rank, row in enumerate(rows, start=1):
            neighbor_id = int(row["work_id"])
            if neighbor_id in by_id.index:
                title = _work_label(by_id.loc[neighbor_id])
            else:
                title = f"作品 {neighbor_id}"
            spaces = "、".join(str(name) for name in row["spaces"]) or "未标注空间"
            lines.append(
                f"{rank}. {_safe(title)} · 共识 {int(row['consensus_count'])} 个空间"
                f"（{_safe(spaces)}）"
            )
        st.markdown(chr(10).join(lines))
        st.caption(
            "来自 logs/neighbor_consensus.json；仅作跨空间/跨模型一致性参考，"
            "不改变星云坐标。"
        )

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
    """显示三模态 Top5 近邻（含解释列）、枢纽作品与片段级对应探索。"""

    st.markdown("#### 高维近邻（真实余弦相似度）")
    if index is None:
        st.caption(f"高维近邻功能不可用：{error or '当前向量版本没有可读向量'}")
        return

    by_id = frame.set_index("work_id")
    jump_ids: list[int] = []
    compare_candidates: list[int] = []
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
                    "全库分位": f"{float(row['pair_percentile']):.1f}%",
                    "双向": "是" if row["is_mutual"] else "",
                }
                for row in rows
            ]
        )
        st.dataframe(table, width="stretch", hide_index=True)
        for row in rows:
            work_key = int(row["work_id"])
            jump_ids.append(work_key)
            if work_key not in compare_candidates:
                compare_candidates.append(work_key)

    st.caption(
        "全库分位：该相似度高于全库全部作品对（不含自身）的百分比；"
        "双向：目标作品也把你列入同一 Top-k。分位接近 100 表示这种相似度在库里很罕见。"
    )

    # ---------------------------------------------------------------- 枢纽作品
    with st.expander("枢纽作品（被最多作品列为 Top5）", expanded=False):
        st.caption(
            "hubness 统计每部作品被多少部其他作品列为 Top5。注意：小样本/筛选子集上该统计不稳定，"
            "只作为观察“哪些作品常被当作近邻”的线索，不代表作品质量或重要性。"
        )
        try:
            hubs = hub_rows(frame, index, kind="work_joint", k=_NEIGHBOR_K, n=8)
        except NebulaError as exc:
            st.caption(f"枢纽统计不可用：{exc}")
        else:
            if hubs:
                hub_table = pd.DataFrame(
                    [
                        {
                            "排名": row["rank"],
                            "作品": row["title"],
                            "被列次数": row["count"],
                        }
                        for row in hubs
                    ]
                )
                st.table(hub_table, hide_index=True)
            else:
                st.caption("作品数量太少，暂无枢纽统计")

    # ---------------------------------------------------------------- 片段级对应
    if jump_ids:
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
        _render_clip_panel(frame, index, selected_work_id, compare_candidates)


def _render_clip_panel(
    frame: pd.DataFrame,
    index: SimilarityIndex,
    selected_work_id: int,
    compare_candidates: list[int],
) -> None:
    """片段级对应面板：可选近邻 B，展示片段余弦热图与最高片段配对。

    默认折叠且需要勾选才渲染热图：避免每次打开页面都加载片段矩阵与额外图表。
    """

    selected_clip_kinds = [
        kind for kind in index.available_clip_kinds if index.has_clip_vector(selected_work_id, kind)
    ]
    if not selected_clip_kinds:
        return
    with st.expander("片段级对应（整体平均向量 vs 局部片段）", expanded=False):
        st.caption(
            "作品级平均向量相近，不代表局部片段也相近；热图颜色为片段对余弦，"
            "越亮表示这两个具体片段越相似。"
        )
        show_clip = st.checkbox(
            "显示片段级对应热图（需要加载片段向量）",
            value=False,
            key="show_clip_panel",
        )
        if not show_clip:
            return
        if not compare_candidates:
            st.caption("该作品暂无可比较的近邻作品")
            return
        if st.session_state.get("clip_compare_work") not in compare_candidates:
            st.session_state.pop("clip_compare_work", None)
        compare_labels: dict[int, str] = {}
        for work_id in compare_candidates:
            row = _lookup_row(frame, work_id)
            compare_labels[int(work_id)] = (
                _work_label(row) if row is not None else f"作品 {work_id}"
            )
        compare_id = st.selectbox(
            "对比的近邻作品",
            options=compare_candidates,
            index=0,
            format_func=lambda work_id: compare_labels.get(int(work_id), str(work_id)),
            key="clip_compare_work",
        )
        compare_id = int(compare_id)
        valid_kinds = [
            kind
            for kind in selected_clip_kinds
            if index.has_clip_vector(compare_id, kind)
        ]
        if not valid_kinds:
            st.caption("两部作品没有共同的片段级模态向量")
            return
        if st.session_state.get("clip_kind") not in valid_kinds:
            st.session_state.pop("clip_kind", None)
        clip_kind = st.selectbox(
            "片段模态",
            options=valid_kinds,
            format_func=lambda kind: CLIP_KIND_LABELS.get(kind, kind),
            key="clip_kind",
        )
        clip_kind = str(clip_kind)
        try:
            data = clip_heatmap_data(index, frame, selected_work_id, compare_id, clip_kind)
            pairs = clip_pair_rows(
                index, frame, selected_work_id, compare_id, clip_kind, _NEIGHBOR_K
            )
            explanation = index.explain_pair(
                selected_work_id, compare_id, kind=clip_kind
            )
        except NebulaError as exc:
            st.caption(f"片段级对应不可用：{exc}")
            return

        heatmap = go.Figure(
            go.Heatmap(
                z=data["matrix"],
                x=data["col_labels"],
                y=data["row_labels"],
                colorscale="Viridis",
                zmin=-1.0,
                zmax=1.0,
                colorbar={"title": "片段余弦"},
                hovertemplate="A %{y}<br>B %{x}<br>余弦 %{z:.4f}<extra></extra>",
            )
        )
        heatmap.update_layout(
            title=f"片段级对应：{data['a_title']} × {data['b_title']}",
            template="plotly_dark",
            height=_HEIGHT_CLIP_HEATMAP,
            margin={"l": 10, "r": 10, "t": 60, "b": 10},
            xaxis_title="B 片段",
            yaxis_title="A 片段",
        )
        st.plotly_chart(heatmap, width="stretch", key="clip_heatmap_chart")

        pair_table = pd.DataFrame(
            [
                {
                    "排名": row["rank"],
                    "A 片段": int(row["a_idx"]) + 1,
                    "A 起始(s)": (
                        "-" if row["a_start_s"] is None else f"{float(row['a_start_s']):.1f}"
                    ),
                    "B 片段": int(row["b_idx"]) + 1,
                    "B 起始(s)": (
                        "-" if row["b_start_s"] is None else f"{float(row['b_start_s']):.1f}"
                    ),
                    "余弦": f"{float(row['similarity']):.4f}",
                }
                for row in pairs
            ]
        )
        st.dataframe(pair_table, width="stretch", hide_index=True)
        st.caption(
            f"片段级统计：最高余弦 {explanation['max']:.3f}，平均 {explanation['mean']:.3f}；"
            f"阈值 {explanation['threshold']:.1f} 以上共 {explanation['count_above_threshold']} 对"
            f"（共 {explanation['n_a']}×{explanation['n_b']} 对）。"
            "整体平均向量接近时，局部片段仍可能只在对少数片段上高度相似。"
        )




# --------------------------------------------------------------------------- 时间轴
def _render_timeline_panel(
    config_path: str,
    map_info: Mapping[str, Any],
    work_id: int,
) -> None:
    """详情里的“完整时间轴”面板：事件包络 + 片段窗口边界 + 覆盖率摘要。"""

    sequence_version = str(
        map_info.get("representation_sequence_version")
        or map_info.get("embedding_version")
        or ""
    )
    with st.expander("完整时间轴（音画事件 + 片段窗口）", expanded=False):
        st.caption(
            "曲线来自高频音画事件扫描并归一到 0..1；竖线为片段窗口起点。"
            "事件通道补充节奏/运动/突变信息，不代表逐帧语义理解。"
        )
        if str(map_info.get("representation_source_kind") or "pe_av") == "visual":
            st.caption("视觉表征空间不含事件时间轴（事件层属于 PE-AV 全片序列）。")
            return
        if not sequence_version:
            st.caption("缺少序列版本信息，无法读取事件时间轴。")
            return
        try:
            timeline = _load_timeline_cached(
                config_path, sequence_version, int(work_id)
            )
        except NebulaError as exc:
            st.caption(f"读取时间轴失败：{exc}")
            return
        if timeline is None:
            st.caption(
                "该作品暂无事件扫描数据（需要 full coverage 编码并开启 events 扫描）。"
            )
            return

        series = timeline.get("series") or {}
        if not series:
            st.caption("事件文件存在，但没有可绘制的包络曲线。")
        else:
            colors = {
                "audio": "#4cc9f0",
                "video": "#f77f00",
                "window": "#ffd166",
                "av": "#90be6d",
            }
            figure = go.Figure()
            for key in ("audio", "video", "window", "av"):
                item = series.get(key)
                if not item:
                    continue
                figure.add_trace(
                    go.Scatter(
                        x=item["time_s"],
                        y=item["values"],
                        mode="lines",
                        name=str(item.get("label") or key),
                        line={"width": 1.2, "color": colors.get(key)},
                        hovertemplate=(
                            f"{item.get('label') or key}<br>t=%{{x:.1f}}s"
                            " · 强度=%{y:.3f}<extra></extra>"
                        ),
                    )
                )
            boundaries = timeline.get("window_boundaries") or []
            shapes = [
                {
                    "type": "line",
                    "x0": float(boundary["start_s"]),
                    "x1": float(boundary["start_s"]),
                    "y0": 0.0,
                    "y1": 1.0,
                    "yref": "paper",
                    "line": {"color": "rgba(210, 210, 210, 0.35)", "width": 1},
                }
                for boundary in boundaries
            ]
            figure.update_layout(
                template="plotly_dark",
                height=340,
                margin={"l": 10, "r": 10, "t": 30, "b": 10},
                shapes=shapes,
                legend={"orientation": "h", "y": 1.02},
                xaxis_title="时间（秒）",
                yaxis_title="归一化事件强度",
            )
            st.plotly_chart(figure, width="stretch", key="timeline_chart")

        duration_s = timeline.get("duration_s")
        duration_text = "-" if duration_s is None else f"{float(duration_s):.1f}s"
        coverage_ratio = float(timeline.get("coverage_ratio") or 0.0)
        coverage_s = float(timeline.get("coverage_s") or 0.0)
        st.caption(
            f"片段 {int(timeline.get('n_clips') or 0)} 个 · 时长 {duration_text} · "
            f"覆盖 {coverage_ratio * 100:.1f}%（{coverage_s:.1f}s） · "
            f"事件版本 {timeline.get('event_version') or '未知'}"
        )


def _on_neighbor_jump(selectbox_key: str) -> None:
    """近邻下拉变化时记录待跳转作品，下一次 rerun 生效。"""

    value = st.session_state.get(selectbox_key)
    if value is not None:
        st.session_state["pending_work_id"] = int(value)



# --------------------------------------------------------------------------- 邻域连线 / 诊断
def _render_sidebar_neighbor_edges(
    visible_work_ids: Sequence[int],
    index: SimilarityIndex | None,
    consensus: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """侧栏“高维邻域连线”开关：互为 Top-k 或跨模型共识边。

    返回 {edges,trace_name,line_color,truncated,n_total}；默认关闭时为空边。
    """

    overlay: dict[str, Any] = {
        "edges": [],
        "trace_name": _MUTUAL_TRACE_NAME,
        "line_color": _MUTUAL_EDGE_COLOR,
        "truncated": False,
        "n_total": 0,
    }
    with st.sidebar:
        st.subheader("高维邻域连线")
        enabled = st.checkbox(
            "显示高维互为近邻连线",
            value=False,
            key="show_mutual_edges",
            help=(
                "按当前表征空间的高维余弦计算可见作品间的互为 Top-k 近邻，"
                "只画双向边；边数上限 300，超出会提示截断。"
            ),
        )
        if not enabled:
            return overlay
        if index is None:
            st.caption("高维近邻不可用，无法计算互为近邻连线。")
            return overlay
        kind = pick_mutual_edge_kind(index)
        if kind is None:
            st.caption("当前作品没有可用的高维模态向量，无法计算连线。")
            return overlay
        k = int(
            st.selectbox(
                "互为近邻 k",
                options=(3, 5, 10),
                index=1,
                key="mutual_edges_k",
            )
        )
        only_consensus = False
        min_count = 2
        if consensus is None:
            st.caption(
                "未找到跨模型共识边文件（logs/neighbor_consensus.json）；"
                "当前仅显示普通互为近邻连线。"
            )
        else:
            only_consensus = st.checkbox(
                "仅显示跨模型共识边",
                value=False,
                key="show_consensus_only",
                help="使用 logs/neighbor_consensus.json 的 consensus_count 过滤边",
            )
            if only_consensus:
                min_count = int(
                    st.selectbox(
                        "共识强度阈值",
                        options=(2, 3),
                        index=0,
                        key="consensus_min_count",
                        format_func=lambda value: f">= {value} 个空间",
                    )
                )
        visible = [int(work_id) for work_id in visible_work_ids]
        if only_consensus and consensus is not None:
            info = consensus_edges_for_visible(
                consensus, visible, min_count=min_count
            )
            overlay = {
                "edges": info["edges"],
                "trace_name": _CONSENSUS_TRACE_NAME,
                "line_color": _CONSENSUS_EDGE_COLOR,
                "truncated": bool(info["truncated"]),
                "n_total": int(info["n_total"]),
            }
            summary = (
                f"跨模型共识边 {len(info['edges'])} 条"
                f"（阈值 >= {min_count} 个空间）"
            )
        else:
            info = mutual_neighbor_edges(index, visible, kind=kind, k=k)
            overlay = {
                "edges": info["edges"],
                "trace_name": _MUTUAL_TRACE_NAME,
                "line_color": _MUTUAL_EDGE_COLOR,
                "truncated": bool(info["truncated"]),
                "n_total": int(info["n_total"]),
            }
            summary = (
                f"高维互为近邻边 {len(info['edges'])} 条"
                f"（k={k}，模态 {KIND_LABELS.get(kind, kind)}）"
            )
        if info["truncated"]:
            st.caption(
                f"边数过多：共 {info['n_total']} 条，仅显示前 {info['max_edges']} 条"
                f"（可缩小搜索/筛选范围或调整 k）。"
            )
            summary += "（已截断）"
        st.caption(summary)
    return overlay


def _render_sidebar_diagnostics(cfg: Config) -> None:
    """侧栏低维投影诊断说明：PCA 3D 辅助观察，不新增地图、不改默认 UMAP。"""

    artifacts = diagnostics_artifacts(cfg)
    with st.sidebar, st.expander("低维投影诊断（PCA 辅助观察）", expanded=False):
        st.caption(
            "星云默认仍使用 UMAP；这里只提供 PCA/UMAP 邻域保持率诊断产物路径，"
            "用于辅助判断高维近邻是否被低维投影扭曲（不新增地图）。"
        )
        if not artifacts["available"]:
            st.caption(
                "未找到诊断产物；可运行 tools/embedding_diagnostics.py "
                "生成 logs/dim_diagnostics_v2_html。"
            )
            return
        if artifacts["html_dir"]:
            st.caption("诊断 HTML 目录（可在本地文件管理器/浏览器打开）：")
            st.code(str(artifacts["html_dir"]), language="text")
            with contextlib.suppress(ValueError):
                st.markdown(
                    f"[在本地打开诊断 HTML 目录]({Path(str(artifacts['html_dir'])).as_uri()})"
                )
        if artifacts["json"]:
            st.caption("诊断 JSON / CSV：")
            st.code(str(artifacts["json"]), language="text")
            if artifacts["csv"]:
                st.code(str(artifacts["csv"]), language="text")


# --------------------------------------------------------------------------- 意外关系探索
_UNEXPECTED_JUMP_A_KEY = "unexpected_jump_a"
_UNEXPECTED_JUMP_B_KEY = "unexpected_jump_b"


def _unexpected_pair_label(row: Mapping[str, Any]) -> str:
    a_title = str(row.get("a_title") or row.get("a_work_id"))
    b_title = str(row.get("b_title") or row.get("b_work_id"))
    if len(a_title) > 24:
        a_title = a_title[:24] + "…"
    if len(b_title) > 24:
        b_title = b_title[:24] + "…"
    return f"{a_title} ↔ {b_title}｜分数 {float(row.get('score') or 0.0):.4f}"


def _render_sidebar_unexpected_relations(
    relations: Mapping[str, Any] | None,
    frame: pd.DataFrame,
    index: SimilarityIndex | None,
) -> dict[str, Any]:
    """侧栏“意外关系探索”：模式/作品对选择 + 只看互为近邻。

    返回 {active,mode,mode_label,rows,selected,only_mutual}；
    文件缺失时显示中文提示且不改变默认星图。
    """

    state: dict[str, Any] = {
        "active": False,
        "mode": "",
        "mode_label": "",
        "rows": [],
        "selected": None,
        "only_mutual": False,
    }
    with st.sidebar, st.expander("意外关系探索（v0.4.5）", expanded=False):
        if relations is None:
            st.caption(
                "未找到意外关系文件（logs/unexpected_relations.json）；"
                "可先运行 t5 的 discovery 工具生成，默认星云不受影响。"
            )
            return state
        modes = unexpected_modes(relations)
        if not modes:
            st.caption("意外关系文件里没有可用的模式/作品对。")
            return state
        labels = {str(item["mode"]): str(item["label"]) for item in modes}
        mode_options = [str(item["mode"]) for item in modes]
        current_mode = st.session_state.get("unexpected_mode")
        if current_mode not in mode_options:
            st.session_state.pop("unexpected_mode", None)
        mode = st.selectbox(
            "模式",
            options=mode_options,
            index=0,
            format_func=lambda value: labels.get(value, value),
            key="unexpected_mode",
        )
        mode_info = next(
            (item for item in modes if str(item["mode"]) == str(mode)), modes[0]
        )
        if mode_info.get("description"):
            st.caption(str(mode_info["description"]))
        only_mutual = st.checkbox(
            "只看互为近邻",
            value=False,
            key="unexpected_only_mutual",
            help="在当前表征空间的高维 Top-k 近邻里，两两互为近邻才保留",
        )
        rows = unexpected_pair_rows(
            relations,
            str(mode),
            frame=frame,
            similarity_index=index,
            only_mutual=bool(only_mutual),
            k=_NEIGHBOR_K,
        )
        state.update(
            {
                "active": True,
                "mode": str(mode),
                "mode_label": str(mode_info["label"]),
                "rows": rows,
                "only_mutual": bool(only_mutual),
            }
        )
        if not rows:
            st.caption("当前模式/筛选下没有匹配的作品对。")
            return state
        pair_keys = [f"{row['a_work_id']}:{row['b_work_id']}" for row in rows]
        if st.session_state.get("unexpected_pair") not in pair_keys:
            st.session_state.pop("unexpected_pair", None)
        selected_key = st.selectbox(
            "选择作品对",
            options=pair_keys,
            index=0,
            format_func=lambda key: next(
                (
                    _unexpected_pair_label(row)
                    for row in rows
                    if f"{row['a_work_id']}:{row['b_work_id']}" == key
                ),
                key,
            ),
            key="unexpected_pair",
        )
        selected = next(
            (
                row
                for row in rows
                if f"{row['a_work_id']}:{row['b_work_id']}" == selected_key
            ),
            rows[0],
        )
        state["selected"] = selected
        st.caption(
            f"共 {len(rows)} 对（记录 {relations.get('n_pairs', 0)} 对）；"
            "按分数降序，染色只影响星图颜色。"
        )
    return state


def _render_unexpected_panel(
    state: Mapping[str, Any],
    frame: pd.DataFrame,
    index: SimilarityIndex | None,
) -> None:
    """主区“意外关系探索”表格 + CSV + 两者 Top-k 邻域。"""

    if not state.get("active"):
        return
    rows = list(state.get("rows") or [])
    with st.expander(
        f"意外关系探索 · {state.get('mode_label') or state.get('mode')}"
        f"（{len(rows)} 对）",
        expanded=bool(rows),
    ):
        if not rows:
            st.caption("当前模式/筛选下没有作品对。")
            return
        table = pd.DataFrame(
            [
                {
                    "分数": f"{float(row['score']):.4f}",
                    "A 作品": row["a_title"],
                    "A 链接": row["a_url"],
                    "B 作品": row["b_title"],
                    "B 链接": row["b_url"],
                    "类别": row["category"],
                    "互为近邻": "是" if row["mutual"] else "",
                    "空间统计": row["space_summary"],
                    "在图中": "是" if row["in_map"] else "否",
                }
                for row in rows
            ]
        )
        st.dataframe(
            table,
            width="stretch",
            hide_index=True,
            column_config={
                "A 链接": st.column_config.LinkColumn("A 链接"),
                "B 链接": st.column_config.LinkColumn("B 链接"),
            },
        )
        csv_text = unexpected_rows_csv(rows)
        st.download_button(
            "下载当前表 CSV",
            data=csv_text.encode("utf-8"),
            file_name=f"unexpected_relations_mode_{state.get('mode', 'a')}.csv",
            mime="text/csv",
            key="unexpected_csv",
        )
        selected = state.get("selected")
        if not isinstance(selected, Mapping):
            return
        a_work_id = int(selected["a_work_id"])
        b_work_id = int(selected["b_work_id"])
        st.markdown(
            f"**当前作品对**：{_safe(selected['a_title'])} ↔ "
            f"{_safe(selected['b_title'])}（分数 {float(selected['score']):.4f}，"
            f"类别 {_safe(selected['category'] or '—')}）"
        )
        st.caption(str(selected.get("space_summary") or ""))
        jump_a, jump_b = st.columns(2)
        with jump_a:
            if st.button(
                f"跳转到 A：{str(selected['a_title'])[:24]}",
                key=_UNEXPECTED_JUMP_A_KEY,
            ):
                st.session_state["pending_work_id"] = a_work_id
                st.rerun()
        with jump_b:
            if st.button(
                f"跳转到 B：{str(selected['b_title'])[:24]}",
                key=_UNEXPECTED_JUMP_B_KEY,
            ):
                st.session_state["pending_work_id"] = b_work_id
                st.rerun()
        if index is None:
            st.caption("高维近邻不可用，无法展示两者的 Top-k 邻域。")
            return
        for label, work_id in (("A", a_work_id), ("B", b_work_id)):
            st.markdown(f"#### {label} 作品 · Top{_NEIGHBOR_K} 邻域")
            for kind in KINDS:
                if not index.has_vector(work_id, kind):
                    st.caption(f"该作品无{KIND_LABELS[kind]}模态向量")
                    continue
                try:
                    neighbor_list = neighbor_rows(
                        frame, index, work_id, kind, _NEIGHBOR_K
                    )
                except NebulaError as exc:
                    st.caption(f"无该模态向量：{exc}")
                    continue
                if not neighbor_list:
                    st.caption("可比较作品不足，暂无近邻")
                    continue
                st.markdown(f"**{KIND_LABELS[kind]} Top{_NEIGHBOR_K}**")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "排名": row["rank"],
                                "作品": row["title"],
                                "余弦相似度": f"{float(row['similarity']):.4f}",
                                "全库分位": f"{float(row['pair_percentile']):.1f}%",
                                "双向": "是" if row["is_mutual"] else "",
                            }
                            for row in neighbor_list
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )

# --------------------------------------------------------------------------- 侧栏
def _render_sidebar_map_settings(
    spaces: list[dict[str, Any]],
    default_space_index: int,
    cfg: Config,
) -> tuple[str, str, dict[str, Any]]:
    """表征空间 + 星云版本 + 投影方式侧栏控件。"""

    with st.sidebar:
        st.header("显示设置")
        if st.button("刷新数据", key="refresh_data", help="重新读取数据库、表征与高维向量索引"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        # 默认打开最新总星云所在表征空间；用户切换后由 selectbox 的 key 记住选择。
        current_space = st.session_state.get("representation_space")
        if (
            not isinstance(current_space, int)
            or isinstance(current_space, bool)
            or not 0 <= current_space < len(spaces)
        ):
            st.session_state.pop("representation_space", None)
            current_space = None
        space_index = st.selectbox(
            "表征空间",
            options=list(range(len(spaces))),
            index=default_space_index,
            format_func=lambda index: spaces[index]["label"],
            key="representation_space",
            help=(
                "Legacy / Full Mean / Temporal Stats / Dense Visual"
                "（CLIP、DINOv2，v1/v2）等表征空间；不同空间的坐标不可直接比较"
            ),
        )
        selected_space = spaces[int(space_index)]
        versions = list(selected_space["versions"])
        current_version = st.session_state.get("map_version")
        if current_version is not None and current_version not in versions:
            st.session_state.pop("map_version", None)
            current_version = None
        fallback_version = str(selected_space["default_version"])
        if fallback_version not in versions:
            fallback_version = versions[-1]
        default_index = (
            versions.index(current_version)
            if current_version in versions
            else versions.index(fallback_version)
        )
        version = st.selectbox(
            "星云版本",
            options=versions,
            index=default_index,
            key="map_version",
            help="默认打开该表征空间最新的总星云；可切换到筛选版或 n_neighbors 变体（坐标不可直接比较）",
        )
        projection = st.radio(
            "投影方式",
            options=["2d", "3d"],
            index=0,
            format_func=lambda value: _PROJECTION_LABELS.get(value, value),
            key="projection",
        )
    return str(version), str(projection), selected_space


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

    try:
        rep_options = _load_representation_options_cached(config_path)
    except NebulaError as exc:
        st.warning(f"读取表征空间失败（降级为 legacy 空间）：{exc}")
        rep_options = []
    spaces = _build_representation_spaces(maps, rep_options)

    # 默认打开“最新 ready 且无筛选的 PE-AV 空间星云”（Dense Visual 为对照实验，
    # 不抢占默认视图），并自动定位它所属的表征空间；用户可在侧栏切换其它空间。
    ready_maps = [row for row in maps if str(row.get("status")) == "ready"]
    unfiltered = [row for row in ready_maps if not row.get("filter_json")]
    peav_unfiltered = [
        row for row in unfiltered if str(row.get("source_kind") or "pe_av") == "pe_av"
    ]
    default_pool = peav_unfiltered or unfiltered or ready_maps
    default_version = (
        str(default_pool[-1]["version"])
        if default_pool
        else versions[-1]
    )
    default_space_index = next(
        (
            index
            for index, space in enumerate(spaces)
            if default_version in space["versions"]
        ),
        0,
    )
    version, projection, selected_space = _render_sidebar_map_settings(
        spaces, default_space_index, cfg
    )

    try:
        map_info, frame = _load_frame(config_path, version)
    except NebulaError as exc:
        st.error(f"读取星云数据失败：{exc}")
        return
    if frame.empty:
        st.warning(f"星云版本 {version} 没有作品坐标，请重新运行 map.py build。")
        return

    representation_id = map_info.get("representation_id")
    representation_label = str(
        map_info.get("representation_label")
        or representations.representation_label(
            str(selected_space.get("representation_name"))
        )
    )
    embedding_version = str(
        map_info.get("representation_sequence_version")
        or map_info.get("embedding_version")
        or cfg.embedding_version
    )
    similarity_state = _load_similarity_index(
        config_path,
        embedding_version,
        representation_id,
        _db_mtime_ns(cfg.paths.db_path),
    )
    index = similarity_state.index
    version_kind_text = (
        "视觉版本"
        if str(map_info.get("representation_source_kind") or "pe_av") == "visual"
        else "序列"
    )
    st.sidebar.caption(
        f"当前表征：{representation_label} · {version_kind_text} {embedding_version}"
    )
    if index is not None:
        st.sidebar.caption(
            f"高维近邻已就绪：{len(index.work_ids)} 部作品，"
            f"模态 {', '.join(KIND_LABELS[kind] for kind in index.available_kinds)}"
        )
    consensus = _load_consensus_cached(config_path)
    unexpected_relations = _load_unexpected_relations_cached(config_path)

    color_field, query, point_size, opacity = _render_sidebar_display_settings(
        frame, cfg, has_similarity=index is not None
    )
    if similarity_state.error:
        st.sidebar.caption(f"高维近邻不可用：{similarity_state.error}")
    _render_sidebar_diagnostics(cfg)
    source, selected_lists, selected_years = _render_sidebar_filters(frame, map_info)
    unexpected_state = _render_sidebar_unexpected_relations(
        unexpected_relations, frame, index
    )

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
    edge_overlay = _render_sidebar_neighbor_edges(
        view["work_id"].tolist(), index, consensus
    )
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
        mutual_edges=edge_overlay["edges"],
        mutual_trace_name=str(edge_overlay["trace_name"]),
        mutual_line_color=str(edge_overlay["line_color"]),
        pair_work_ids=(
            [
                int(unexpected_state["selected"]["a_work_id"]),
                int(unexpected_state["selected"]["b_work_id"]),
            ]
            if unexpected_state.get("selected")
            else []
        ),
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
    _render_unexpected_panel(unexpected_state, frame, index)

    selected_row = _lookup_row(frame, selected_id) if selected_id is not None else None
    if selected_row is not None and selected_id is not None:
        st.divider()
        _render_detail(selected_row, cfg)
        _render_consensus_summary(consensus, frame, selected_id)
        _render_timeline_panel(config_path, map_info, selected_id)
        _render_neighbor_panel(frame, view, index, selected_id, similarity_state.error)

    st.caption(
        f"星云版本 {version} · 表征 {representation_label} · "
        f"构建筛选 {_describe_filter_spec(map_info.get('filter_spec'))} · "
        f"{version_kind_text}版本 {embedding_version} · 作品数 {map_info.get('n_works')} · "
        f"创建于 {map_info.get('created_at')}"
    )


if __name__ == "__main__":
    main()
