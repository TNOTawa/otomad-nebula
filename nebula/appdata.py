"""可视化用纯数据准备（不依赖 streamlit / plotly）。

职责划分：

- mapplot 只负责“模型给出坐标”，本模块只读取坐标与作品/榜单元数据，
  拼成前端可直接消费的 DataFrame；
- 染色、搜索、Bilibili 跳转链接、hover 文本全部是纯函数，便于单元测试；
- 外部元数据（年份、作者、排名、时长等）只用于展示/染色，绝不参与降维。

约定：缺失值统一为 None（ColorSpec.values）或 pandas 的 NaN/None（DataFrame）。
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .db import Catalog
from .errors import MapError

#: 有序的染色字段列表：(字段 id, 中文标签)
COLOR_FIELDS: list[tuple[str, str]] = [
    ("none", "不染色/密度"),
    ("duration_s", "视频时长"),
    ("publish_date", "发布日期"),
    ("rank", "选票排名"),
    ("votes", "得票数"),
    ("author_uid", "作者UID"),
    ("vote_year", "选票年份"),
    ("source", "来源"),
]

_CONTINUOUS_FIELDS = frozenset({"duration_s", "publish_date", "rank", "votes"})
_CATEGORICAL_FIELDS = frozenset({"author_uid", "vote_year", "source"})
_ALL_FIELD_IDS = frozenset(field_id for field_id, _ in COLOR_FIELDS)

#: 分类色板（tableau10 顺序，固定以保证可复现）
_QUALITATIVE_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

#: 来源字段的固定颜色，便于观众快速区分本地作品与 B 站作品
_SOURCE_COLORS = {"local": "#2ca02c", "bilibili": "#ff7f0e"}
_SOURCE_LABELS = {"local": "本地", "bilibili": "Bilibili"}

#: Viridis 近似色带（rank 反转后数值越小颜色越亮）
_VIRIDIS = [
    "#440154",
    "#482878",
    "#3e4989",
    "#31688e",
    "#26828e",
    "#1f9e89",
    "#35b779",
    "#6ece58",
    "#b5de2b",
    "#fde725",
]

_EPOCH = pd.Timestamp("1970-01-01")

#: 返回的 DataFrame 固定列顺序（前端与测试都依赖它）
_FRAME_COLUMNS = [
    "work_id",
    "title",
    "work_key",
    "source",
    "bvid",
    "part",
    "url",
    "author_uid",
    "author_name",
    "duration_s",
    "publish_date",
    "local_path",
    "file_name",
    "status",
    "x2",
    "y2",
    "x3",
    "y3",
    "z3",
    "x",
    "y",
    "z",
    "rank",
    "votes",
    "vote_year",
    "vote_years",
    "list_names",
    "search_text",
    "hover_text",
]

#: 搜索时参与匹配的文本列
_SEARCH_COLUMNS = (
    "file_name",
    "title",
    "work_key",
    "bvid",
    "author_name",
    "author_uid",
    "local_path",
    "list_names",
)


@dataclass(frozen=True)
class ColorSpec:
    """一个染色维度的渲染参数。

    values 与 DataFrame 行对齐；缺失为 None。categorical=True 时 values 是
    类别编码（从 0 开始的整数），colorscale 是固定调色列表；否则 colorscale
    是 plotly 内置色带名或颜色列表。
    """

    values: list
    title: str
    colorscale: str | list
    categorical: bool = False
    tickvals: list | None = None
    ticktext: list | None = None


# --------------------------------------------------------------------------- 小工具
def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # pragma: no cover - 防御性分支
        return False


def _cell(row: Mapping[str, Any] | pd.Series, key: str) -> Any:
    """同时兼容 dict 与 pandas.Series 的字段读取。"""

    if isinstance(row, Mapping):
        return row.get(key)
    return row.get(key, None)


def _label(field: str) -> str:
    for field_id, label in COLOR_FIELDS:
        if field_id == field:
            return label
    raise MapError(f"未知染色维度: {field}")


def _palette(count: int, *, base: list[str] | None = None) -> list[str]:
    """按固定色板循环生成 count 个颜色；始终至少返回两个颜色。"""

    colors = base if base is not None else _QUALITATIVE_COLORS
    if count <= 0:
        return colors[:2]
    result = [colors[index % len(colors)] for index in range(count)]
    if len(result) == 1:
        result = result * 2
    return result


def _numeric_column(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        return np.full(len(df), np.nan, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


# --------------------------------------------------------------------------- 读取星云
def _file_name(local_path: Any, bvid: Any, part: Any) -> str:
    if local_path:
        name = Path(str(local_path).replace("\\", "/")).name
        if name:
            return name
    if bvid:
        suffix = ""
        if not _is_missing(part):
            try:
                part_number = int(part)
            except (TypeError, ValueError):
                part_number = 1
            if part_number > 1:
                suffix = f"_p{part_number}"
        return f"{bvid}{suffix}.mp4"
    return ""


def _pick_primary_appearance(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """选取用于展示的“主榜单记录”：优先选票榜中排名最好的记录。"""

    vote_rows = [
        row for row in rows if row.get("list_kind") == "votes" and row.get("rank") is not None
    ]
    if vote_rows:

        def sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
            rank = int(row["rank"])
            year = int(row["vote_year"]) if row.get("vote_year") is not None else -1
            return (rank, -year, str(row.get("list_key") or ""))

        return min(vote_rows, key=sort_key)
    return rows[0] if rows else None


def _summarize_appearances(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """把一部作品的多条榜单记录汇总成展示字段。"""

    years = sorted({int(row["vote_year"]) for row in rows if row.get("vote_year") is not None})
    list_names: list[str] = []
    for row in rows:
        name = row.get("list_name") or row.get("list_key")
        if name and name not in list_names:
            list_names.append(str(name))
    primary = _pick_primary_appearance(rows)
    rank = primary.get("rank") if primary is not None else None
    votes = primary.get("votes") if primary is not None else None
    if votes is None:
        vote_values = [int(row["votes"]) for row in rows if row.get("votes") is not None]
        votes = max(vote_values) if vote_values else None
    vote_year = primary.get("vote_year") if primary is not None else None
    if vote_year is None and years:
        vote_year = years[-1]
    return {
        "rank": None if rank is None else float(rank),
        "votes": None if votes is None else float(votes),
        "vote_year": None if vote_year is None else int(vote_year),
        "vote_years": years,
        "list_names": "、".join(list_names),
    }


def _build_search_text(df: pd.DataFrame) -> pd.Series:
    pieces: list[pd.Series] = []
    for column in _SEARCH_COLUMNS:
        if column not in df.columns:
            continue
        pieces.append(df[column].fillna("").astype(str))
    if not pieces:
        return pd.Series([""] * len(df), index=df.index, dtype=object)
    text = pieces[0]
    for piece in pieces[1:]:
        text = text + " " + piece
    return text.str.strip()


def _parse_map_params(map_row: Mapping[str, Any]) -> dict[str, Any]:
    raw = map_row.get("params_json")
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def load_nebula_frame(
    cat: Catalog,
    cfg: Config,
    *,
    map_version: str | None = None,
) -> tuple[dict, pd.DataFrame]:
    """读取一个星云版本的全部可视化数据。

    返回 (map_info, frame)。frame 每行一部作品，包含作品元数据、2D/3D 坐标、
    榜单汇总、file_name、search_text、hover_text 等展示字段。
    没有可用星云时抛 MapError 并给出中文提示。
    """

    if map_version is None:
        map_row = cat.latest_map()
        if map_row is None:
            raise MapError("还没有可用的星云版本，请先运行：map.py build")
    else:
        map_row = cat.get_map(str(map_version))
        if map_row is None:
            raise MapError(f"星云版本不存在: {map_version}")
    version = str(map_row["version"])
    if str(map_row.get("status")) != "ready":
        raise MapError(f"星云版本 {version} 尚未就绪（状态 {map_row.get('status')}）")

    rows = cat.load_map_rows(version)
    if not rows:
        raise MapError(f"星云版本 {version} 没有任何作品坐标，请重新运行 map.py build")

    appearances_by_work: dict[int, list[dict[str, Any]]] = {}
    for appearance in cat.appearance_rows():
        appearances_by_work.setdefault(int(appearance["work_id"]), []).append(appearance)

    records: list[dict[str, Any]] = []
    for row in rows:
        work_id = int(row["id"])
        local_path = row.get("local_path")
        bvid = row.get("bvid")
        part = row.get("part")
        summary = _summarize_appearances(appearances_by_work.get(work_id, []))
        records.append(
            {
                "work_id": work_id,
                "title": row.get("title"),
                "work_key": row.get("work_key"),
                "source": row.get("source"),
                "bvid": bvid,
                "part": part,
                "url": row.get("url"),
                "author_uid": row.get("author_uid"),
                "author_name": row.get("author_name"),
                "duration_s": row.get("duration_s"),
                "publish_date": row.get("publish_date"),
                "local_path": local_path,
                "file_name": _file_name(local_path, bvid, part),
                "status": row.get("status"),
                "x2": row.get("x2"),
                "y2": row.get("y2"),
                "x3": row.get("x3"),
                "y3": row.get("y3"),
                "z3": row.get("z3"),
                "rank": summary["rank"],
                "votes": summary["votes"],
                "vote_year": summary["vote_year"],
                "vote_years": summary["vote_years"],
                "list_names": summary["list_names"],
            }
        )

    frame = pd.DataFrame.from_records(records)
    frame["work_id"] = frame["work_id"].astype("int64")
    frame["part"] = pd.to_numeric(frame["part"], errors="coerce").fillna(1).astype("int64")
    for column in ("duration_s", "rank", "votes", "x2", "y2", "x3", "y3", "z3"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    # 方便默认按 2D 使用；3D 视图由 app 显式选择 x3/y3/z3。
    frame["x"] = frame["x2"]
    frame["y"] = frame["y2"]
    frame["z"] = frame["z3"]
    frame["search_text"] = _build_search_text(frame)
    frame["hover_text"] = hover_text(frame)
    frame = frame[[column for column in _FRAME_COLUMNS if column in frame.columns]]

    map_info = dict(map_row)
    map_info["params"] = _parse_map_params(map_row)
    return map_info, frame


# --------------------------------------------------------------------------- 染色
def available_color_fields(df: pd.DataFrame) -> list[tuple[str, str]]:
    """返回当前数据可用的染色字段（按 COLOR_FIELDS 顺序）。

    除“不染色”外，只保留有至少一个非缺失值的列。
    """

    available: list[tuple[str, str]] = []
    for field_id, label in COLOR_FIELDS:
        if field_id == "none":
            available.append((field_id, label))
            continue
        if field_id not in df.columns:
            continue
        if bool(df[field_id].notna().any()):
            available.append((field_id, label))
    return available


def _density_spec(df: pd.DataFrame, projection: str, density_grid: int) -> ColorSpec:
    grid = max(2, int(density_grid))
    count = len(df)
    values: list[float | None] = [None] * count
    if projection == "2d":
        xs = _numeric_column(df, "x2")
        ys = _numeric_column(df, "y2")
        valid = np.isfinite(xs) & np.isfinite(ys)
        if valid.any():
            histogram, x_edges, y_edges = np.histogram2d(xs[valid], ys[valid], bins=grid)
            x_index = np.clip(np.searchsorted(x_edges, xs[valid], side="right") - 1, 0, grid - 1)
            y_index = np.clip(np.searchsorted(y_edges, ys[valid], side="right") - 1, 0, grid - 1)
            density = histogram[x_index, y_index]
            for position, value in zip(np.where(valid)[0], density, strict=True):
                values[int(position)] = float(value)
        title = f"空间密度（{grid}×{grid} 网格内作品数）"
    else:
        xs = _numeric_column(df, "x3")
        ys = _numeric_column(df, "y3")
        zs = _numeric_column(df, "z3")
        valid = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
        if valid.any():
            points = np.column_stack([xs[valid], ys[valid], zs[valid]])
            histogram, edges = np.histogramdd(points, bins=grid)
            indices = []
            for axis in range(3):
                axis_edges = np.asarray(edges[axis])
                indices.append(
                    np.clip(
                        np.searchsorted(axis_edges, points[:, axis], side="right") - 1,
                        0,
                        grid - 1,
                    )
                )
            density = histogram[indices[0], indices[1], indices[2]]
            for position, value in zip(np.where(valid)[0], density, strict=True):
                values[int(position)] = float(value)
        title = f"空间密度（{grid}×{grid}×{grid} 网格内作品数）"
    return ColorSpec(values=values, title=title, colorscale="Viridis")


def _continuous_values(df: pd.DataFrame, field: str) -> list[float | None]:
    if field == "publish_date":
        dates = pd.to_datetime(df.get("publish_date"), errors="coerce")
        values: list[float | None] = []
        for value in dates:
            if pd.isna(value):
                values.append(None)
            else:
                values.append(float((value - _EPOCH).days))
        return values
    numeric = _numeric_column(df, field)
    return [None if not np.isfinite(value) else float(value) for value in numeric]


def _publish_date_ticks(df: pd.DataFrame) -> tuple[list[int], list[str]]:
    dates = pd.to_datetime(df.get("publish_date"), errors="coerce").dropna()
    if dates.empty:
        return [], []
    min_year = int(dates.dt.year.min())
    max_year = int(dates.dt.year.max())
    tickvals: list[int] = []
    ticktext: list[str] = []
    for year in range(min_year, max_year + 1):
        tickvals.append(int((pd.Timestamp(year=year, month=1, day=1) - _EPOCH).days))
        ticktext.append(str(year))
    return tickvals, ticktext


def _continuous_spec(df: pd.DataFrame, field: str) -> ColorSpec:
    values = _continuous_values(df, field)
    tickvals: list[int] | None = None
    ticktext: list[str] | None = None
    if field == "duration_s":
        title, colorscale = "视频时长（秒）", "Viridis"
    elif field == "votes":
        title, colorscale = "得票数", "Plasma"
    elif field == "publish_date":
        title, colorscale = "发布日期", "Cividis"
        tickvals, ticktext = _publish_date_ticks(df)
    else:  # rank
        title, colorscale = "选票排名（数值越小越好）", list(reversed(_VIRIDIS))
    return ColorSpec(
        values=values,
        title=title,
        colorscale=colorscale,
        categorical=False,
        tickvals=tickvals,
        ticktext=ticktext,
    )


def _categorical_values(df: pd.DataFrame, field: str) -> list[Any]:
    if field not in df.columns:
        return [None] * len(df)
    return [None if _is_missing(value) else value for value in df[field].tolist()]


def _categorical_spec(df: pd.DataFrame, field: str) -> ColorSpec:
    raw_values = _categorical_values(df, field)
    categories = sorted(
        {value for value in raw_values if value is not None},
        key=lambda value: (str(type(value)), str(value)) if field != "vote_year" else (0, int(value)),
    )
    if field == "vote_year":
        categories = sorted({int(value) for value in raw_values if value is not None})
    codes = {category: index for index, category in enumerate(categories)}
    values: list[int | None] = [
        None if value is None else int(codes[int(value)] if field == "vote_year" else codes[value])
        for value in raw_values
    ]
    if field == "source":
        colors = [
            _SOURCE_COLORS.get(str(category), _QUALITATIVE_COLORS[index % len(_QUALITATIVE_COLORS)])
            for index, category in enumerate(categories)
        ]
        title = _label(field)
    elif field == "vote_year":
        colors = _palette(len(categories))
        title = _label(field)
    else:
        colors = _palette(len(categories))
        title = _label(field)
    if not colors:
        colors = _palette(2)
    if len(colors) == 1:
        colors = colors * 2
    ticktext = (
        [f"{int(category)}年" for category in categories]
        if field == "vote_year"
        else [_SOURCE_LABELS.get(str(category), str(category)) for category in categories]
        if field == "source"
        else [str(category) for category in categories]
    )
    return ColorSpec(
        values=values,
        title=title,
        colorscale=colors,
        categorical=True,
        tickvals=list(range(len(categories))),
        ticktext=ticktext,
    )


def apply_color(
    df: pd.DataFrame,
    field: str,
    *,
    projection: str = "2d",
    density_grid: int = 40,
) -> ColorSpec:
    """把某个染色维度换算成 Plotly 可直接使用的颜色参数。"""

    if field not in _ALL_FIELD_IDS:
        raise MapError(f"未知染色维度: {field}（可选: {', '.join(sorted(_ALL_FIELD_IDS))}）")
    if projection not in ("2d", "3d"):
        raise MapError(f"未知投影方式: {projection}（只支持 2d / 3d）")
    if field == "none":
        return _density_spec(df, projection, density_grid)
    if field in _CONTINUOUS_FIELDS:
        return _continuous_spec(df, field)
    return _categorical_spec(df, field)


# --------------------------------------------------------------------------- 搜索与跳转
def search_frame(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """按关键词过滤作品；大小写不敏感，空查询返回全部。

    关键词按整体子串匹配 search_text（其中包含文件名、标题、BV 号、
    作者、榜单名等），不区分大小写。
    """

    text = (query or "").strip()
    if not text:
        return df.copy()
    if "search_text" not in df.columns:
        raise MapError("数据缺少 search_text 列，无法搜索")
    search_text = df["search_text"].fillna("").astype(str).str.casefold()
    mask = search_text.str.contains(text.casefold(), regex=False)
    return df.loc[mask].copy()


def external_url(row: Mapping[str, Any] | pd.Series) -> str | None:
    """返回作品的 Bilibili 链接；本地作品返回 None。"""

    source = _cell(row, "source")
    if str(source) != "bilibili":
        return None
    bvid = _cell(row, "bvid")
    if not _is_missing(bvid) and str(bvid).strip():
        url = f"https://www.bilibili.com/video/{str(bvid).strip()}"
        part = _cell(row, "part")
        part_number = 1
        if not _is_missing(part):
            try:
                part_number = int(part)
            except (TypeError, ValueError):
                part_number = 1
        if part_number > 1:
            url += f"?p={part_number}"
        return url
    url = _cell(row, "url")
    if url and str(url).startswith(("http://", "https://")):
        return str(url)
    return None


def _format_duration(seconds: Any) -> str:
    total = int(round(float(seconds)))
    minutes, second = divmod(max(total, 0), 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{hours}:{minutes:02d}:{second:02d}"
    return f"{minutes}:{second:02d}"


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _hover_one(record: Mapping[str, Any]) -> str:
    title = record.get("title") or record.get("file_name") or record.get("work_key") or "未知作品"
    parts = [f"<b>{_escape(title)}</b>"]
    author = record.get("author_name") or record.get("author_uid")
    if author:
        parts.append(f"作者：{_escape(author)}")
    duration = record.get("duration_s")
    if not _is_missing(duration):
        parts.append(f"时长：{_escape(_format_duration(duration))}")
    publish_date = record.get("publish_date")
    if publish_date:
        parts.append(f"发布日期：{_escape(publish_date)}")
    rank = record.get("rank")
    if not _is_missing(rank):
        parts.append(f"选票排名：第 {int(float(rank))} 名")
    votes = record.get("votes")
    if not _is_missing(votes):
        parts.append(f"得票数：{int(float(votes))}")
    vote_year = record.get("vote_year")
    if not _is_missing(vote_year):
        parts.append(f"选票年份：{int(float(vote_year))}年")
    list_names = record.get("list_names")
    if list_names:
        parts.append(f"榜单：{_escape(list_names)}")
    source = record.get("source")
    if str(source) == "bilibili":
        bvid = record.get("bvid")
        parts.append(f"来源：Bilibili{f' · {_escape(bvid)}' if bvid else ''}")
    elif source:
        parts.append(f"来源：{_escape(_SOURCE_LABELS.get(str(source), source))}")
    file_name = record.get("file_name")
    if file_name:
        parts.append(f"文件：{_escape(file_name)}")
    return "<br>".join(parts)


def hover_text(df: pd.DataFrame) -> list[str]:
    """生成与 DataFrame 行对齐的 HTML hover 文本。"""

    if len(df) == 0:
        return []
    return [_hover_one(record) for record in df.to_dict("records")]
