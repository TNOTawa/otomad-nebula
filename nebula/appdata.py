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
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from . import representations
from .config import Config
from .db import Catalog
from .errors import MapError, NebulaError

if TYPE_CHECKING:  # 仅类型提示；appdata 运行期不依赖近邻模块，保持纯数据层轻量
    from .similarity import SimilarityIndex

#: 有序的染色字段列表：(字段 id, 中文标签)
COLOR_FIELDS: list[tuple[str, str]] = [
    ("none", "不染色/密度"),
    ("density_knn", "高维近邻密度"),
    ("duration_s", "视频时长"),
    ("publish_date", "发布日期"),
    ("rank", "选票排名"),
    ("votes", "得票数"),
    ("author_uid", "作者UID"),
    ("vote_year", "选票年份"),
    ("source", "来源"),
]

_CONTINUOUS_FIELDS = frozenset({"duration_s", "publish_date", "rank", "votes"})
_NEIGHBOR_DENSITY_FIELD = "density_knn"
_NEIGHBOR_DENSITY_K = 5

#: 高维互为近邻连线默认参数（v0.4.4 界面叠加，不改变地图坐标语义）
DEFAULT_MUTUAL_EDGE_LIMIT = 300
DEFAULT_MUTUAL_K = 5
MUTUAL_EDGE_DEFAULT_KIND = "work_joint"
_MUTUAL_EDGE_KIND_ORDER: tuple[str, ...] = ("work_joint", "work_audio", "work_video")

#: 跨模型共识边产物（t5 build_consensus_graph）与低维诊断产物文件名
CONSENSUS_FILE_NAME = "neighbor_consensus.json"
DIAGNOSTICS_HTML_DIR_NAME = "dim_diagnostics_v2_html"
DIAGNOSTICS_JSON_NAME = "dim_diagnostics_v2.json"
DIAGNOSTICS_CSV_NAME = "dim_diagnostics_v2.csv"
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
    "list_keys",
    "list_names",
    "local_file",
    "is_canonical",
    "search_text",
    "hover_text",
]

logger = logging.getLogger(__name__)

#: 事件时间轴曲线选择偏好（按事件模块的 feature names 匹配，缺失时退回第 0 列）
_AUDIO_SERIES_PREFERENCE: tuple[str, ...] = (
    "onset_strength",
    "spectral_flux",
    "rms",
    "log_energy",
)
_VIDEO_SERIES_PREFERENCE: tuple[str, ...] = (
    "motion_energy",
    "frame_diff_mean",
    "edge_change",
    "luma_std",
)
_AV_SERIES_PREFERENCE: tuple[str, ...] = (
    "envelope_correlation",
    "max_correlation",
    "coincidence_rate",
    "corr_lag0",
)

#: 搜索时参与匹配的文本列
_SEARCH_COLUMNS = (
    "file_name",
    "title",
    "work_key",
    "bvid",
    "author_name",
    "author_uid",
    "local_path",
    "local_file",
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
    list_keys: list[str] = []
    for row in rows:
        key = row.get("list_key")
        if key and str(key) not in list_keys:
            list_keys.append(str(key))
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
        "list_keys": list_keys,
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


def _parse_filter_spec(map_row: Mapping[str, Any]) -> dict[str, Any] | None:
    """解析 map_runs.filter_json；None 表示该星云没有筛选（全量）。"""

    raw = map_row.get("filter_json")
    if not raw:
        return None
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) and parsed else None


def _basename(path: Any) -> str:
    if not path:
        return ""
    return Path(str(path).replace("\\", "/")).name


def _linked_local_file(
    work_id: int,
    *,
    row: Mapping[str, Any],
    duplicate_ids_by_canonical: Mapping[int, list[int]],
    works_by_id: Mapping[int, Mapping[str, Any]],
) -> str:
    """作品本地文件名：优先自身 local_path，其次关联 duplicate 本地作品的 local_path。"""

    own = _basename(row.get("local_path"))
    if own:
        return own
    for duplicate_id in duplicate_ids_by_canonical.get(int(work_id), []):
        duplicate = works_by_id.get(int(duplicate_id))
        if duplicate is None:
            continue
        linked = _basename(duplicate.get("local_path"))
        if linked:
            return linked
    return ""


def select_default_map(cat: Catalog) -> dict[str, Any] | None:
    """选择默认星云：优先最新 ready 且无筛选的 PE-AV 表征地图。

    v0.4.3 的 Dense Visual 空间是对照实验，不应抢占默认视图；用户仍可在侧栏
    显式切换。筛选地图不参与默认选择（与既有行为一致）。
    """

    maps = [m for m in cat.list_maps() if str(m.get("status")) == "ready"]
    if not maps:
        return cat.latest_map()

    def _source_kind(row: Mapping[str, Any]) -> str:
        rep_id = row.get("representation_id")
        if rep_id is None:
            return "pe_av"
        rep = cat.get_representation(int(rep_id))
        return str(rep.get("source_kind")) if rep else "pe_av"

    unfiltered = [m for m in maps if not m.get("filter_json")]
    preferred = [m for m in unfiltered if _source_kind(m) == "pe_av"]
    pool = preferred or unfiltered or maps
    return max(
        pool,
        key=lambda m: (str(m.get("created_at") or ""), str(m.get("version") or "")),
    )


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
        map_row = select_default_map(cat)
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

    duplicate_ids = cat.get_duplicate_work_ids()
    duplicate_ids_by_canonical: dict[int, list[int]] = {}
    for link in cat.get_work_links():
        canonical = int(link["canonical_work_id"])
        duplicate_ids_by_canonical.setdefault(canonical, []).append(int(link["duplicate_work_id"]))
    works_by_id = {int(work["id"]): work for work in cat.list_works()}

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
                "local_file": _linked_local_file(
                    work_id,
                    row=row,
                    duplicate_ids_by_canonical=duplicate_ids_by_canonical,
                    works_by_id=works_by_id,
                ),
                "status": row.get("status"),
                "is_canonical": work_id not in duplicate_ids,
                "x2": row.get("x2"),
                "y2": row.get("y2"),
                "x3": row.get("x3"),
                "y3": row.get("y3"),
                "z3": row.get("z3"),
                "rank": summary["rank"],
                "votes": summary["votes"],
                "vote_year": summary["vote_year"],
                "vote_years": summary["vote_years"],
                "list_keys": summary["list_keys"],
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
    map_info["filter_spec"] = _parse_filter_spec(map_row)
    map_info.update(_map_representation_info(cat, map_row))
    return map_info, frame


# --------------------------------------------------------------------------- 表征空间
def _map_representation_info(
    cat: Catalog,
    map_row: Mapping[str, Any],
) -> dict[str, Any]:
    """暴露地图所属表征的名称/标签/序列版本（旧地图按 legacy 处理）。"""

    rep_id = map_row.get("representation_id")
    if rep_id is not None:
        rep = cat.get_representation(int(rep_id))
        if rep is not None:
            return {
                "representation_id": int(rep["id"]),
                "representation_name": str(rep["name"]),
                "representation_label": representations.representation_label(str(rep["name"])),
                "representation_sequence_version": str(rep["sequence_version"]),
                "representation_kind": str(rep["kind"]),
                "representation_source_kind": str(rep.get("source_kind") or "pe_av"),
            }
    sequence_version = str(map_row.get("embedding_version") or "")
    return {
        "representation_id": None if rep_id is None else int(rep_id),
        "representation_name": representations.LEGACY_NAME,
        "representation_label": representations.representation_label(
            representations.LEGACY_NAME
        ),
        "representation_sequence_version": sequence_version,
        "representation_kind": "legacy",
        "representation_source_kind": "pe_av",
    }


def _latest_unfiltered_map_version(maps: Iterable[Mapping[str, Any]]) -> str | None:
    latest: Mapping[str, Any] | None = None
    for row in maps:
        if row.get("filter_json"):
            continue
        latest = row
    return None if latest is None else str(latest["version"])


def map_matches_representation(
    map_row: Mapping[str, Any],
    option: Mapping[str, Any],
) -> bool:
    """判断一个 ready 地图是否属于某个表征空间（兼容未回填的旧地图）。"""

    representation_id = map_row.get("representation_id")
    option_id = option.get("representation_id")
    if representation_id is not None:
        return option_id is not None and int(representation_id) == int(option_id)
    return (
        str(option.get("representation_name")) == representations.LEGACY_NAME
        and str(map_row.get("embedding_version"))
        == str(option.get("sequence_version"))
    )


def load_representation_options(
    catalog: Catalog,
    config: Config,
) -> list[dict[str, Any]]:
    """列出可切换的表征空间及其“最新 ready 且无筛选”的地图版本。

    返回每项：representation_id / representation_name / label / kind /
    sequence_version / map_version / n_maps。没有地图的表征也会返回，
    map_version 为 None（界面可忽略或提示尚未构建地图）。
    """

    try:
        registered = representations.list_available(catalog, config)
    except NebulaError as exc:  # pragma: no cover - 仅界面提示
        logger.warning("读取表征列表失败: %s", exc)
        registered = []
    ready_maps = [
        row for row in catalog.list_maps() if str(row.get("status")) == "ready"
    ]
    options: list[dict[str, Any]] = []

    def append_option(
        *,
        representation_id: int | None,
        name: str,
        kind: str | None,
        sequence_version: str | None,
        matches: list[Mapping[str, Any]],
        registered_flag: bool,
        source_kind: str = "pe_av",
    ) -> None:
        options.append(
            {
                "representation_id": representation_id,
                "representation_name": name,
                "label": representations.representation_label(name),
                "kind": kind,
                "source_kind": source_kind,
                "sequence_version": sequence_version,
                "map_version": _latest_unfiltered_map_version(matches),
                "n_maps": len(matches),
                "registered": registered_flag,
            }
        )

    for rep in registered:
        rep_id = int(rep["id"])
        name = str(rep["name"])
        sequence_version = str(rep["sequence_version"])
        matches = [
            row for row in ready_maps if row.get("representation_id") == rep_id
        ]
        if name == representations.LEGACY_NAME:
            # 旧地图可能仍是 representation_id=NULL，按 sequence_version 归入 legacy 空间
            matches = matches + [
                row
                for row in ready_maps
                if row.get("representation_id") is None
                and str(row.get("embedding_version")) == sequence_version
            ]
        append_option(
            representation_id=rep_id,
            name=name,
            kind=str(rep["kind"]),
            sequence_version=sequence_version,
            matches=matches,
            registered_flag=True,
            source_kind=str(rep.get("source_kind") or "pe_av"),
        )

    # 没有任何已登记 legacy 表征、但存在旧地图时，合成一个 legacy 空间
    null_maps = [row for row in ready_maps if row.get("representation_id") is None]
    covered_sequences = {
        str(option["sequence_version"])
        for option in options
        if str(option["representation_name"]) == representations.LEGACY_NAME
    }
    for sequence_version in sorted(
        {str(row.get("embedding_version")) for row in null_maps} - covered_sequences
    ):
        matches = [
            row
            for row in null_maps
            if str(row.get("embedding_version")) == sequence_version
        ]
        append_option(
            representation_id=None,
            name=representations.LEGACY_NAME,
            kind="legacy",
            sequence_version=sequence_version,
            matches=matches,
            registered_flag=False,
        )

    # 未登记/孤儿表征的地图也给出可切换空间，避免地图“消失”
    known_ids = {
        int(option["representation_id"])
        for option in options
        if option["representation_id"] is not None
    }
    orphan_maps = [
        row
        for row in ready_maps
        if row.get("representation_id") is not None
        and int(row["representation_id"]) not in known_ids
    ]
    for rep_id in sorted({int(row["representation_id"]) for row in orphan_maps}):
        matches = [
            row for row in orphan_maps if int(row["representation_id"]) == rep_id
        ]
        orphan_rep = catalog.get_representation(rep_id)
        append_option(
            representation_id=rep_id,
            name=f"representation_{rep_id}",
            kind=None if orphan_rep is None else str(orphan_rep["kind"]),
            sequence_version=str(matches[0].get("embedding_version")),
            matches=matches,
            registered_flag=False,
            source_kind=(
                "pe_av"
                if orphan_rep is None
                else str(orphan_rep.get("source_kind") or "pe_av")
            ),
        )

    priority = {
        representations.LEGACY_NAME: 0,
        representations.FULL_MEAN_V1: 1,
        representations.TEMPORAL_STATS_V1: 2,
        representations.DENSE_VISUAL_CLIP_V1: 3,
        representations.DENSE_VISUAL_DINO_V1: 4,
        representations.DENSE_VISUAL_CLIP_V2: 5,
        representations.DENSE_VISUAL_DINO_V2: 6,
        representations.TRANSFORMER_V1: 7,
    }
    options.sort(
        key=lambda option: (
            priority.get(str(option["representation_name"]), 9),
            str(option["label"]),
        )
    )
    return options




# --------------------------------------------------------------------------- 事件时间轴
def _event_module() -> Any | None:
    """延迟导入 nebula.events（t3 并行实现；模块缺失时返回 None）。"""

    try:
        from . import events
    except ImportError:
        return None
    return events


def _load_events_bundle(work_dir: Path) -> Any | None:
    """读取事件包络（便于测试 monkeypatch）；文件缺失/模块未就绪返回 None。"""

    module = _event_module()
    if module is None:
        return None
    loader = getattr(module, "load_events_bundle", None)
    if loader is None:
        return None
    try:
        return loader(work_dir)
    except NebulaError:
        return None
    except Exception as exc:  # noqa: BLE001 - 事件文件损坏只影响时间轴显示
        logger.warning("读取事件包络失败 %s: %s", work_dir, exc)
        return None


def _event_feature_names(attribute: str) -> tuple[str, ...]:
    module = _event_module()
    if module is None:
        return ()
    try:
        return tuple(str(name) for name in getattr(module, attribute, ()))
    except TypeError:  # pragma: no cover - 防御性分支
        return ()


def _normalize_display_series(values: np.ndarray) -> list[float]:
    """把任意范围的事件曲线缩放到 0..1（恒定值返回全 0）。"""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return []
    finite = np.where(np.isfinite(array), array, 0.0)
    low = float(np.min(finite))
    high = float(np.max(finite))
    if not np.isfinite(low) or not np.isfinite(high) or high - low <= 1e-12:
        return [0.0 for _ in range(finite.size)]
    return ((finite - low) / (high - low)).astype(float).tolist()


def _matrix_series(
    matrix: Any,
    *,
    names: tuple[str, ...],
    preference: tuple[str, ...],
    rate: float,
) -> dict[str, Any] | None:
    """从 (n, features) 事件矩阵中取一条包络曲线（优先按 feature 名称选列）。"""

    if matrix is None:
        return None
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        return None
    column = 0
    feature = names[0] if names else "feature_0"
    for candidate in preference:
        if candidate in names:
            column = names.index(candidate)
            feature = candidate
            break
    if column >= array.shape[1]:
        column = 0
    values = array[:, column]
    if not np.all(np.isfinite(values)):
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    safe_rate = float(rate) if float(rate) > 0 else 1.0
    return {
        "feature": feature,
        "rate_hz": safe_rate,
        "time_s": [round(index / safe_rate, 6) for index in range(values.size)],
        "values": _normalize_display_series(values),
    }


def _resample_series(values: list[float], target_length: int) -> list[float]:
    """线性插值把一条曲线重采样到目标长度（用于音画包络对齐）。"""

    if target_length <= 0 or not values:
        return []
    if len(values) == target_length:
        return list(values)
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, target_length)
    return np.interp(target, source, np.asarray(values, dtype=np.float64)).astype(float).tolist()


def _coverage_ratio(
    boundaries: list[dict[str, Any]],
    duration_s: float | None,
) -> tuple[float, float]:
    """返回 (被覆盖秒数, 覆盖率)；覆盖率以视频总时长为准。"""

    intervals = sorted(
        (float(item["start_s"]), float(item["end_s"])) for item in boundaries
    )
    merged: list[list[float]] = []
    for start, end in intervals:
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    if duration_s is not None and duration_s > 0:
        covered = sum(
            max(0.0, min(end, duration_s) - max(0.0, start))
            for start, end in merged
        )
        return float(covered), float(covered / duration_s)
    covered = sum(end - start for start, end in merged)
    return float(covered), 1.0 if covered > 0 else 0.0


def load_event_timeline(
    catalog: Catalog,
    config: Config,
    sequence_version: str,
    work_id: int,
) -> dict[str, Any] | None:
    """读取某作品的事件时间轴数据（供 app 画图）；没有事件文件返回 None。

    返回内容：片段窗口边界、被覆盖时长/覆盖率、事件包络曲线（audio/video/
    av 重合/窗口特征，均缩放至 0..1 便于叠画）、事件版本与采样率。
    """

    work = catalog.get_work(int(work_id))
    if work is None:
        raise MapError(f"作品不存在: {work_id}")
    work_dir = representations.sequence_work_dir(
        config, str(sequence_version), int(work_id)
    )
    features = _load_events_bundle(work_dir)
    if features is None:
        return None

    clip_rows = catalog.get_clips(int(work_id), version=str(sequence_version))
    boundaries: list[dict[str, Any]] = []
    for row in clip_rows:
        start = float(row["start_s"])
        duration = float(row["duration_s"])
        valid_raw = row.get("valid_duration_s")
        weight_raw = row.get("weight")
        boundaries.append(
            {
                "index": int(row["idx"]),
                "start_s": round(start, 6),
                "end_s": round(start + duration, 6),
                "duration_s": round(duration, 6),
                "valid_duration_s": (
                    round(float(valid_raw), 6) if valid_raw is not None else round(duration, 6)
                ),
                "weight": None if weight_raw is None else float(weight_raw),
            }
        )

    duration_s = work.get("duration_s")
    if duration_s is None:
        duration_s = getattr(features, "duration_s", None)
    if duration_s is None:
        state = catalog.get_sequence_state(int(work_id), version=str(sequence_version))
        if state is not None:
            duration_s = state.get("duration_s")
    duration_value = None if duration_s is None else float(duration_s)
    covered_s, coverage_ratio = _coverage_ratio(boundaries, duration_value)

    audio_matrix = getattr(features, "audio", None)
    video_matrix = getattr(features, "video", None)
    audio_rate = float(getattr(features, "audio_rate_hz", 0.0) or 0.0)
    video_rate = float(
        getattr(features, "video_fps_effective", 0.0)
        or getattr(features, "video_fps", 0.0)
        or 0.0
    )
    audio_series = _matrix_series(
        audio_matrix,
        names=_event_feature_names("AUDIO_FEATURE_NAMES"),
        preference=_AUDIO_SERIES_PREFERENCE,
        rate=audio_rate,
    )
    video_series = _matrix_series(
        video_matrix,
        names=_event_feature_names("VIDEO_FEATURE_NAMES"),
        preference=_VIDEO_SERIES_PREFERENCE,
        rate=video_rate,
    )

    series: dict[str, dict[str, Any]] = {}
    if audio_series is not None:
        audio_series["label"] = "音频事件包络"
        series["audio"] = audio_series
    if video_series is not None:
        video_series["label"] = "画面运动包络"
        series["video"] = video_series
    av_series = _matrix_series(
        getattr(features, "av", None),
        names=_event_feature_names("AV_FEATURE_NAMES"),
        preference=_AV_SERIES_PREFERENCE,
        rate=audio_rate,
    )
    if av_series is not None:
        av_series["label"] = "音画关系包络（归一化）"
        series["av"] = av_series
    elif audio_series is not None and video_series is not None:
        # 没有音画关系序列时，用音频×画面包络的乘积做替代展示
        aligned_video = _resample_series(
            video_series["values"], len(audio_series["values"])
        )
        overlap = [
            round(min(1.0, a * v), 6)
            for a, v in zip(audio_series["values"], aligned_video, strict=True)
        ]
        series["av"] = {
            "label": "音画包络重合（音频×画面，归一化）",
            "feature": "audio_x_video_normalized",
            "rate_hz": audio_series["rate_hz"],
            "time_s": list(audio_series["time_s"]),
            "values": overlap,
        }

    window_features_raw = getattr(features, "window_features", None)
    window_features: list[list[float]] | None = None
    if window_features_raw is not None:
        window_array = np.asarray(window_features_raw, dtype=np.float64)
        if window_array.ndim == 2 and window_array.shape[0] > 0:
            window_features = window_array.astype(float).tolist()
            energy = np.nanmean(window_array, axis=1)
            energy = np.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)
            if len(boundaries) == window_array.shape[0]:
                times = [
                    round((item["start_s"] + item["end_s"]) / 2.0, 6)
                    for item in boundaries
                ]
            else:
                times = [float(index) for index in range(window_array.shape[0])]
            series["window"] = {
                "label": "窗口事件特征均值（归一化）",
                "feature": "window_feature_mean",
                "rate_hz": None,
                "time_s": times,
                "values": _normalize_display_series(energy),
            }

    event_version = getattr(features, "version", None)
    return {
        "work_id": int(work_id),
        "sequence_version": str(sequence_version),
        "event_version": None if event_version is None else str(event_version),
        "duration_s": duration_value,
        "n_clips": len(boundaries),
        "window_boundaries": boundaries,
        "window_features": window_features,
        "coverage_s": round(covered_s, 6),
        "coverage_ratio": round(coverage_ratio, 6),
        "audio_rate_hz": audio_rate or None,
        "video_fps_effective": video_rate or None,
        "feature_names": {
            "audio": list(_event_feature_names("AUDIO_FEATURE_NAMES")),
            "video": list(_event_feature_names("VIDEO_FEATURE_NAMES")),
            "av": list(_event_feature_names("AV_FEATURE_NAMES")),
            "window": list(_event_feature_names("WINDOW_FEATURE_NAMES")),
        },
        "series": series,
    }


# --------------------------------------------------------------------------- 染色
def available_color_fields(
    df: pd.DataFrame,
    has_similarity: bool = True,
) -> list[tuple[str, str]]:
    """返回当前数据可用的染色字段（按 COLOR_FIELDS 顺序）。

    除“不染色”外，只保留有至少一个非缺失值的列；
    “高维近邻密度”依赖 SimilarityIndex，has_similarity=False 时不提供。
    """

    available: list[tuple[str, str]] = []
    count = len(df)
    for field_id, label in COLOR_FIELDS:
        if field_id == "none":
            available.append((field_id, label))
            continue
        if field_id == _NEIGHBOR_DENSITY_FIELD:
            if has_similarity and count > 0:
                available.append((field_id, label))
            continue
        if field_id not in df.columns:
            continue
        if bool(df[field_id].notna().any()):
            available.append((field_id, label))
    return available


def _neighbor_density_spec(
    df: pd.DataFrame,
    similarity_index: "SimilarityIndex | None",
) -> ColorSpec:
    """高维近邻密度：第 k 近邻余弦距离（越大越稀疏）。"""

    if similarity_index is None:
        raise MapError("计算高维近邻密度需要 SimilarityIndex，请先构造近邻索引")
    if "work_id" not in df.columns:
        raise MapError("数据缺少 work_id 列，无法计算高维近邻密度")
    try:
        distances = similarity_index.kth_neighbor_distance(k=_NEIGHBOR_DENSITY_K)
    except Exception as exc:  # noqa: BLE001 - 统一转成展示层可捕获的中文错误
        raise MapError(f"计算高维近邻密度失败: {exc}") from exc
    values = [distances.get(int(work_id)) for work_id in df["work_id"].tolist()]
    return ColorSpec(
        values=values,
        title=f"高维近邻密度（第{_NEIGHBOR_DENSITY_K}近邻余弦距离，越大越稀疏）",
        colorscale="Viridis",
    )


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
    similarity_index: "SimilarityIndex | None" = None,
) -> ColorSpec:
    """把某个染色维度换算成 Plotly 可直接使用的颜色参数。

    高维近邻密度（density_knn）需要传入 SimilarityIndex；其余维度不需要。
    """

    if field not in _ALL_FIELD_IDS:
        raise MapError(f"未知染色维度: {field}（可选: {', '.join(sorted(_ALL_FIELD_IDS))}）")
    if projection not in ("2d", "3d"):
        raise MapError(f"未知投影方式: {projection}（只支持 2d / 3d）")
    if field == "none":
        return _density_spec(df, projection, density_grid)
    if field == _NEIGHBOR_DENSITY_FIELD:
        return _neighbor_density_spec(df, similarity_index)
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


def _as_selection(values: Any) -> set[Any] | None:
    """把界面传来的筛选值统一成集合；None 或空集合表示不过滤。"""

    if values is None:
        return None
    if isinstance(values, (str, int)):
        return {values}
    try:
        selected = set(values)
    except TypeError as exc:
        raise MapError(f"筛选值必须是可迭代集合: {values!r}") from exc
    return selected or None


def filter_frame(
    df: pd.DataFrame,
    *,
    sources: Iterable[str] | None = None,
    list_keys: Iterable[str] | None = None,
    vote_years: Iterable[int] | None = None,
) -> pd.DataFrame:
    """按来源 / 榜单 / 选票年份过滤作品；空值表示不过滤。

    榜单筛选使用作品关联的 list_keys（可能跨多个榜单），年份使用 vote_years。
    """

    result = df.copy()
    selected_sources = _as_selection(sources)
    if selected_sources is not None:
        if "source" not in result.columns:
            raise MapError("数据缺少 source 列，无法按来源筛选")
        result = result[result["source"].isin(selected_sources)]

    selected_lists = _as_selection(list_keys)
    if selected_lists is not None:
        if "list_keys" not in result.columns:
            raise MapError("数据缺少 list_keys 列，无法按榜单筛选")
        result = result[
            result["list_keys"].map(
                lambda keys: bool(set(keys or []) & selected_lists)
            )
        ]

    selected_years = _as_selection(vote_years)
    if selected_years is not None:
        if "vote_years" not in result.columns:
            raise MapError("数据缺少 vote_years 列，无法按年份筛选")
        normalized_years = {int(year) for year in selected_years}
        result = result[
            result["vote_years"].map(
                lambda years: bool(set(int(year) for year in (years or [])) & normalized_years)
            )
        ]
    return result


def _record_lookup(df: pd.DataFrame) -> dict[int, Mapping[str, Any]]:
    """按 work_id 建立展示行索引。"""

    records: dict[int, Mapping[str, Any]] = {}
    if "work_id" in df.columns:
        for record in df.to_dict("records"):
            records[int(record["work_id"])] = record
    return records


def _record_title(record: Mapping[str, Any] | None, work_id: int) -> str:
    if record is None:
        return f"作品 {work_id}"
    title = record.get("title") or record.get("file_name") or record.get("work_key")
    return str(title) if title else f"作品 {work_id}"


def _clip_start(times: Mapping[int, Mapping[str, Any]], idx: int) -> float | None:
    item = times.get(int(idx))
    if item is None:
        return None
    start = item.get("start_s")
    return None if start is None else float(start)


def _clip_label(times: Mapping[int, Mapping[str, Any]], idx: int) -> str:
    start = _clip_start(times, idx)
    if start is None:
        return f"片段 {int(idx) + 1}"
    return f"片段 {int(idx) + 1}（{start:.1f}s）"


def neighbor_rows(
    df: pd.DataFrame,
    index: "SimilarityIndex",
    work_id: int,
    kind: str = "work_joint",
    k: int = 5,
) -> list[dict[str, Any]]:
    """把某部作品在指定模态下的 Top-k 近邻整理成展示行。

    每行包含 rank / work_id / title / file_name / similarity / pair_percentile /
    is_mutual / in_map；其中 pair_percentile 是该相似度在全库作品对中的百分位，
    is_mutual 表示目标作品是否也把当前作品列为 Top-k，in_map 表示该近邻是否
    也出现在当前 DataFrame（当前星云/筛选视图）里。
    作品没有该模态向量时由 SimilarityIndex 抛 NebulaError。
    """

    neighbors = index.neighbors(int(work_id), kind=kind, k=k)
    records = _record_lookup(df)
    rows: list[dict[str, Any]] = []
    for neighbor in neighbors:
        record = records.get(int(neighbor.work_id))
        rows.append(
            {
                "rank": neighbor.rank,
                "work_id": neighbor.work_id,
                "title": _record_title(record, neighbor.work_id),
                "file_name": str(record.get("file_name") or "") if record is not None else "",
                "similarity": float(neighbor.similarity),
                "pair_percentile": float(neighbor.pair_percentile),
                "is_mutual": bool(neighbor.is_mutual),
                "in_map": record is not None,
            }
        )
    return rows


def hub_rows(
    df: pd.DataFrame,
    index: "SimilarityIndex",
    *,
    kind: str = "work_joint",
    k: int = 5,
    n: int = 8,
) -> list[dict[str, Any]]:
    """枢纽作品展示行：被最多作品列入 Top-k 的作品。

    注意：hubness 在小样本/筛选子集上不稳定，仅作为观察线索。
    """

    records = _record_lookup(df)
    rows: list[dict[str, Any]] = []
    for rank, (work_id, count) in enumerate(index.hub_top(k=k, n=n, kind=kind), start=1):
        record = records.get(int(work_id))
        rows.append(
            {
                "rank": rank,
                "work_id": int(work_id),
                "title": _record_title(record, int(work_id)),
                "count": int(count),
                "in_map": record is not None,
            }
        )
    return rows


def clip_heatmap_data(
    index: "SimilarityIndex",
    df: pd.DataFrame,
    a_work_id: int,
    b_work_id: int,
    kind: str = "clip_joint",
) -> dict[str, Any]:
    """片段级相似热图的纯数据（不依赖 plotly）。

    返回 matrix（n_a × n_b 余弦值，list[list[float]]）、行列标签（含片段起始秒）、
    两侧作品标题与模态，供 app 直接构造 Plotly 热图。
    """

    matrix = index.clip_similarity(a_work_id, b_work_id, kind=kind)
    times_a = {int(item["idx"]): item for item in index.clip_times(a_work_id)}
    times_b = {int(item["idx"]): item for item in index.clip_times(b_work_id)}
    records = _record_lookup(df)
    return {
        "kind": kind,
        "a_work_id": int(a_work_id),
        "b_work_id": int(b_work_id),
        "a_title": _record_title(records.get(int(a_work_id)), int(a_work_id)),
        "b_title": _record_title(records.get(int(b_work_id)), int(b_work_id)),
        "matrix": matrix.astype(float).tolist(),
        "row_labels": [_clip_label(times_a, idx) for idx in range(matrix.shape[0])],
        "col_labels": [_clip_label(times_b, idx) for idx in range(matrix.shape[1])],
        "n_a": int(matrix.shape[0]),
        "n_b": int(matrix.shape[1]),
    }


def clip_pair_rows(
    index: "SimilarityIndex",
    df: pd.DataFrame,
    a_work_id: int,
    b_work_id: int,
    kind: str = "clip_joint",
    k: int = 5,
) -> list[dict[str, Any]]:
    """最高片段配对的展示行（含两侧作品标题、片段序号、起始秒与余弦）。"""

    records = _record_lookup(df)
    times_a = {int(item["idx"]): item for item in index.clip_times(a_work_id)}
    times_b = {int(item["idx"]): item for item in index.clip_times(b_work_id)}
    a_title = _record_title(records.get(int(a_work_id)), int(a_work_id))
    b_title = _record_title(records.get(int(b_work_id)), int(b_work_id))
    rows: list[dict[str, Any]] = []
    for rank, pair in enumerate(
        index.best_clip_pairs(a_work_id, b_work_id, kind=kind, k=k), start=1
    ):
        a_idx = int(pair["a_idx"])
        b_idx = int(pair["b_idx"])
        rows.append(
            {
                "rank": rank,
                "a_title": a_title,
                "a_idx": a_idx,
                "a_start_s": _clip_start(times_a, a_idx),
                "b_title": b_title,
                "b_idx": b_idx,
                "b_start_s": _clip_start(times_b, b_idx),
                "similarity": float(pair["similarity"]),
            }
        )
    return rows


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



# --------------------------------------------------------------------------- 高维邻域连线（v0.4.4）
def pick_mutual_edge_kind(index: "SimilarityIndex") -> str | None:
    """选择用于互邻连线的模态：优先 joint，其次 audio/video。"""

    for kind in _MUTUAL_EDGE_KIND_ORDER:
        if kind in index.available_kinds:
            return kind
    return None


def mutual_neighbor_edges(
    index: "SimilarityIndex",
    work_ids: Iterable[int],
    *,
    kind: str = MUTUAL_EDGE_DEFAULT_KIND,
    k: int = DEFAULT_MUTUAL_K,
    max_edges: int | None = None,
) -> dict[str, Any]:
    """对可见作品计算“互为 Top-k 高维近邻”边（只保留双向边）。

    返回 {kind,k,edges,n_total,truncated,max_edges,n_visible_works}；
    edges 每条为 {a_work_id,b_work_id,similarity}，按相似度降序、id 升序稳定排序。
    边数超过 max_edges（默认 DEFAULT_MUTUAL_EDGE_LIMIT=300）时返回截断信息，
    由界面提示“仅显示前 N 条”。
    """

    if int(k) < 1:
        raise NebulaError(f"互为近邻的 k 必须 >= 1，收到 {k}")
    visible: list[int] = []
    seen: set[int] = set()
    for raw in work_ids:
        work_id = int(raw)
        if work_id in seen:
            continue
        seen.add(work_id)
        if index.has_vector(work_id, kind):
            visible.append(work_id)
    neighbor_sets: dict[int, set[int]] = {}
    for work_id in visible:
        try:
            neighbors = index.neighbors(work_id, kind=kind, k=int(k))
        except NebulaError:
            continue
        neighbor_sets[work_id] = {int(neighbor.work_id) for neighbor in neighbors}

    edges: list[dict[str, Any]] = []
    for a_work_id in visible:
        forward = neighbor_sets.get(a_work_id, set())
        for b_work_id in forward:
            if b_work_id <= a_work_id:
                continue
            if a_work_id not in neighbor_sets.get(b_work_id, set()):
                continue
            try:
                similarity = float(index.similarity(a_work_id, b_work_id, kind=kind))
            except NebulaError:  # pragma: no cover - 防御性分支
                continue
            edges.append(
                {
                    "a_work_id": int(a_work_id),
                    "b_work_id": int(b_work_id),
                    "similarity": round(similarity, 6),
                }
            )
    edges.sort(key=lambda edge: (-edge["similarity"], edge["a_work_id"], edge["b_work_id"]))
    total = len(edges)
    limit = DEFAULT_MUTUAL_EDGE_LIMIT if max_edges is None else int(max_edges)
    if limit < 1:
        raise NebulaError(f"边数上限必须 >= 1，收到 {limit}")
    return {
        "kind": kind,
        "k": int(k),
        "edges": edges[:limit],
        "n_total": total,
        "truncated": total > limit,
        "max_edges": limit,
        "n_visible_works": len(visible),
    }


def _coerce_work_id(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _consensus_pair(edge: Mapping[str, Any]) -> tuple[int, int] | None:
    """从共识边记录里兼容解析作品对（work_pair / pair / a_work_id+b_work_id 等）。"""

    for key in ("work_pair", "pair", "work_ids", "ids"):
        value = edge.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            pair = (_coerce_work_id(value[0]), _coerce_work_id(value[1]))
            if pair[0] is not None and pair[1] is not None:
                return pair  # type: ignore[return-value]
        if isinstance(value, str):
            pieces = value.replace("-", ":").replace(",", ":").split(":")
            if len(pieces) == 2:
                pair = (_coerce_work_id(pieces[0]), _coerce_work_id(pieces[1]))
                if pair[0] is not None and pair[1] is not None:
                    return pair  # type: ignore[return-value]
    for first_key, second_key in (
        ("a_work_id", "b_work_id"),
        ("work_a", "work_b"),
        ("work_id_a", "work_id_b"),
        ("a", "b"),
    ):
        pair = (
            _coerce_work_id(edge.get(first_key)),
            _coerce_work_id(edge.get(second_key)),
        )
        if pair[0] is not None and pair[1] is not None:
            return pair  # type: ignore[return-value]
    return None


def _consensus_spaces(edge: Mapping[str, Any]) -> tuple[str, ...]:
    """共识边涉及的空间名（兼容 t5 工具 mutual_spaces 列表与 spaces 映射）。"""

    for key in (
        "mutual_spaces",
        "spaces",
        "representations",
        "space_names",
        "models",
        "model_names",
    ):
        value = edge.get(key)
        if isinstance(value, (Mapping, list, tuple)):
            names = tuple(str(name) for name in value if str(name or "").strip())
            if names:
                return names
    return ()


def _consensus_count(edge: Mapping[str, Any]) -> int:
    for key in ("consensus_count", "count", "support", "n_spaces", "n_models", "votes"):
        value = edge.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count >= 1:
            return count
    spaces = _consensus_spaces(edge)
    if spaces:
        return len(spaces)
    # 单条边至少代表 1 个空间的共识
    return 1


def load_neighbor_consensus(config: Config) -> dict[str, Any] | None:
    """读取 logs/neighbor_consensus.json（t5 跨模型共识图产物）。

    文件不存在/不可解析时返回 None（界面提示并回退普通互为近邻）。
    兼容多种字段命名：edges 列表或 pairs 映射、work_pair/a_work_id+b_work_id、
    consensus_count/count/len(spaces)。返回规范化后的
    {path,edges,pairs,n_edges,meta}；edges 按共识强度降序。
    """

    path = Path(config.paths.logs) / CONSENSUS_FILE_NAME
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("读取跨模型共识边失败 %s: %s", path, exc)
        return None

    records: list[Mapping[str, Any]] = []
    pairs_mapping: Mapping[str, Any] | None = None
    if isinstance(raw, list):
        records = [item for item in raw if isinstance(item, Mapping)]
    elif isinstance(raw, Mapping):
        edges_value = raw.get("edges")
        if isinstance(edges_value, list):
            records = [item for item in edges_value if isinstance(item, Mapping)]
        pairs_value = raw.get("pairs")
        if isinstance(pairs_value, Mapping):
            pairs_mapping = pairs_value
        elif isinstance(raw.get("edge_pairs"), list):
            records = records + [
                item for item in raw["edge_pairs"] if isinstance(item, Mapping)
            ]
    else:
        return None

    pairs: dict[tuple[int, int], dict[str, Any]] = {}
    for record in records:
        pair = _consensus_pair(record)
        if pair is None:
            continue
        a_work_id, b_work_id = sorted(pair)
        if a_work_id == b_work_id:
            continue
        spaces = _consensus_spaces(record)
        count = _consensus_count(record)
        key = (a_work_id, b_work_id)
        existing = pairs.get(key)
        if existing is None:
            pairs[key] = {
                "a_work_id": a_work_id,
                "b_work_id": b_work_id,
                "consensus_count": int(count),
                "spaces": spaces,
            }
        else:
            existing["consensus_count"] = max(int(existing["consensus_count"]), int(count))
            existing["spaces"] = tuple(dict.fromkeys(existing["spaces"] + spaces))

    if pairs_mapping is not None:
        for key_text, value in pairs_mapping.items():
            pair = _consensus_pair({"pair": key_text})
            if pair is None:
                continue
            a_work_id, b_work_id = sorted(pair)
            if a_work_id == b_work_id:
                continue
            if isinstance(value, Mapping):
                count = _consensus_count(value)
                spaces = _consensus_spaces(value)
            else:
                count = _consensus_count({"count": value})
                spaces = ()
            key = (a_work_id, b_work_id)
            if key not in pairs:
                pairs[key] = {
                    "a_work_id": a_work_id,
                    "b_work_id": b_work_id,
                    "consensus_count": int(count),
                    "spaces": spaces,
                }
            else:
                pairs[key]["consensus_count"] = max(
                    int(pairs[key]["consensus_count"]), int(count)
                )
                pairs[key]["spaces"] = tuple(
                    dict.fromkeys(pairs[key]["spaces"] + spaces)
                )

    edges = sorted(
        pairs.values(),
        key=lambda edge: (
            -int(edge["consensus_count"]),
            edge["a_work_id"],
            edge["b_work_id"],
        ),
    )
    meta = raw.get("meta") if isinstance(raw, Mapping) else None
    return {
        "path": str(path),
        "edges": edges,
        "pairs": pairs,
        "n_edges": len(edges),
        "meta": meta,
    }


def consensus_edges_for_visible(
    consensus: Mapping[str, Any],
    work_ids: Iterable[int],
    *,
    min_count: int = 2,
    max_edges: int | None = None,
) -> dict[str, Any]:
    """按可见作品与共识强度过滤共识边（>= min_count，默认 2 个空间）。"""

    if int(min_count) < 1:
        raise NebulaError(f"共识强度阈值必须 >= 1，收到 {min_count}")
    visible = {int(work_id) for work_id in work_ids}
    selected: list[dict[str, Any]] = []
    for edge in consensus.get("edges", ()):
        if int(edge["a_work_id"]) not in visible or int(edge["b_work_id"]) not in visible:
            continue
        if int(edge["consensus_count"]) < int(min_count):
            continue
        selected.append(
            {
                "a_work_id": int(edge["a_work_id"]),
                "b_work_id": int(edge["b_work_id"]),
                "consensus_count": int(edge["consensus_count"]),
                "spaces": tuple(edge.get("spaces") or ()),
            }
        )
    total = len(selected)
    limit = DEFAULT_MUTUAL_EDGE_LIMIT if max_edges is None else int(max_edges)
    if limit < 1:
        raise NebulaError(f"边数上限必须 >= 1，收到 {limit}")
    return {
        "min_count": int(min_count),
        "edges": selected[:limit],
        "n_total": total,
        "truncated": total > limit,
        "max_edges": limit,
    }


def consensus_neighbors(
    consensus: Mapping[str, Any],
    work_id: int,
    *,
    min_count: int = 2,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """某部作品在共识图里的跨空间近邻摘要行（按共识强度降序）。"""

    target = int(work_id)
    rows: list[dict[str, Any]] = []
    for edge in consensus.get("edges", ()):
        first = int(edge["a_work_id"])
        second = int(edge["b_work_id"])
        if first == target:
            other = second
        elif second == target:
            other = first
        else:
            continue
        count = int(edge["consensus_count"])
        if count < int(min_count):
            continue
        rows.append(
            {
                "work_id": other,
                "consensus_count": count,
                "spaces": tuple(edge.get("spaces") or ()),
            }
        )
    rows.sort(key=lambda row: (-row["consensus_count"], row["work_id"]))
    return rows[: max(1, int(limit))]


def diagnostics_artifacts(config: Config) -> dict[str, Any]:
    """低维投影诊断产物（PCA/UMAP 邻域保持率）路径，用于界面辅助说明。"""

    logs = Path(config.paths.logs)
    html_dir = logs / DIAGNOSTICS_HTML_DIR_NAME
    json_path = logs / DIAGNOSTICS_JSON_NAME
    csv_path = logs / DIAGNOSTICS_CSV_NAME
    return {
        "html_dir": str(html_dir) if html_dir.is_dir() else None,
        "json": str(json_path) if json_path.is_file() else None,
        "csv": str(csv_path) if csv_path.is_file() else None,
        "available": bool(
            html_dir.is_dir() or json_path.is_file() or csv_path.is_file()
        ),
    }



# --------------------------------------------------------------------------- 意外关系探索（v0.4.5）
UNEXPECTED_RELATIONS_FILE_NAME = "unexpected_relations.json"

_UNEXPECTED_MODE_ALIASES: dict[str, str] = {
    "a": "a",
    "mode_a": "a",
    "mode-a": "a",
    "modea": "a",
    "1": "a",
    "b": "b",
    "mode_b": "b",
    "mode-b": "b",
    "modeb": "b",
    "2": "b",
    "c": "c",
    "mode_c": "c",
    "mode-c": "c",
    "modec": "c",
    "3": "c",
}

_UNEXPECTED_SPACE_ALIASES: dict[str, str] = {
    "joint": "joint",
    "work_joint": "joint",
    "pe_av_joint": "joint",
    "full_mean_joint": "joint",
    "audio": "audio",
    "work_audio": "audio",
    "full_mean_audio": "audio",
    "video": "video",
    "work_video": "video",
    "full_mean_video": "video",
    "temporal": "temporal",
    "temporal_joint": "temporal",
    "temporal_stats": "temporal",
    "temporal_stats_v1": "temporal",
    "full_mean": "full_mean",
    "fullmean": "full_mean",
    "full_mean_v1": "full_mean",
    "clip": "clip",
    "visual_clip": "clip",
    "dense_visual_clip": "clip",
    "dense_visual_clip_v1": "clip",
    "dense_visual_clip_v2": "clip",
    "clip_joint": "clip",
    "peav": "peav",
    "pe_av": "peav",
    "peav_joint": "peav",
    "dino": "dino",
    "visual_dino": "dino",
    "dense_visual_dino": "dino",
    "dense_visual_dino_v1": "dino",
    "dense_visual_dino_v2": "dino",
    "dino_joint": "dino",
}

_UNEXPECTED_SPACE_ORDER: tuple[str, ...] = (
    "joint",
    "audio",
    "video",
    "temporal",
    "full_mean",
    "peav",
    "clip",
    "dino",
)

_UNEXPECTED_SPACE_LABELS: dict[str, str] = {
    "joint": "联合",
    "audio": "音频",
    "video": "画面",
    "temporal": "时间结构",
    "full_mean": "全片平均",
    "peav": "PE-AV",
    "clip": "CLIP",
    "dino": "DINOv2",
}

_UNEXPECTED_MODE_LABELS: dict[str, str] = {
    "a": "模式 A",
    "b": "模式 B",
    "c": "模式 C",
}


def _unexpected_mode(value: Any) -> str | None:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    return _UNEXPECTED_MODE_ALIASES.get(raw)


def _unexpected_space(value: Any) -> str | None:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not raw:
        return None
    return _UNEXPECTED_SPACE_ALIASES.get(raw)


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _unexpected_space_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    rank = stats.get("rank")
    if rank is None:
        rank = stats.get("rank_min", stats.get("min_rank"))
    percentile = stats.get("percentile")
    if percentile is None:
        percentile = stats.get("bar_p", stats.get("pct", stats.get("percentile_avg")))
    return {
        "rank": _int_or_none(rank),
        "rank_a": _int_or_none(stats.get("rank_a")),
        "rank_b": _int_or_none(stats.get("rank_b")),
        "percentile": _float_or_none(percentile),
        "bar_p": _float_or_none(stats.get("bar_p")),
        "p_a": _float_or_none(stats.get("p_a")),
        "p_b": _float_or_none(stats.get("p_b")),
        "cosine": _float_or_none(
            stats.get("cosine", stats.get("cos", stats.get("sim", stats.get("similarity"))))
        ),
        "mutual": (
            None if stats.get("mutual") is None else bool(stats.get("mutual"))
        ),
    }


def _unexpected_per_space(record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """兼容 discovery 真实字段（per-space dict 直接放在 pair 上）与合成 per_space 容器。"""

    raw = record.get("per_space") or record.get("space_stats")
    result: dict[str, dict[str, Any]] = {}
    candidate_sources: list[Mapping[str, Any]] = []
    if isinstance(raw, Mapping):
        candidate_sources.append(raw)
    elif isinstance(record.get("spaces"), Mapping):
        candidate_sources.append(record["spaces"])
    candidate_sources.append(record)
    for source in candidate_sources:
        for name, stats in source.items():
            space = _unexpected_space(name)
            if space is None or not isinstance(stats, Mapping):
                continue
            if space not in result:
                result[space] = _unexpected_space_stats(stats)
    return result


def _unexpected_score(record: Mapping[str, Any]) -> float:
    for key in ("score", "relation_score", "novelty", "uplift", "value"):
        number = _float_or_none(record.get(key))
        if number is not None:
            return number
    return 0.0


def _unexpected_category(record: Mapping[str, Any]) -> str:
    for key in ("category", "label", "tag", "kind", "reason", "type"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _unexpected_record_pair(
    record: Mapping[str, Any],
    default_mode: str | None,
    default_category: str = "",
) -> dict[str, Any] | None:
    pair = _consensus_pair(record)
    if pair is None:
        return None
    mode = _unexpected_mode(record.get("mode", record.get("mode_id"))) or default_mode
    a_work_id, b_work_id = sorted(pair)
    if a_work_id == b_work_id:
        return None
    title_a = record.get("title_a", record.get("a_title"))
    title_b = record.get("title_b", record.get("b_title"))
    return {
        "mode": mode or "",
        "a_work_id": a_work_id,
        "b_work_id": b_work_id,
        "score": _unexpected_score(record),
        "category": _unexpected_category(record) or default_category,
        "title_a": None if title_a is None else str(title_a),
        "title_b": None if title_b is None else str(title_b),
        "per_space": _unexpected_per_space(record),
    }


_UNEXPECTED_PAIR_CONTAINER_KEYS = (
    "pairs",
    "relations",
    "edges",
    "items",
    "data",
    "top",
)
_UNEXPECTED_MODE_C_CATEGORIES = (
    "common",
    "visual_vs_peav",
    "peav_only",
    "clip_only",
    "dino_only",
)


def _unexpected_container_records(
    raw: Any,
) -> list[tuple[Mapping[str, Any], str | None, str]]:
    """从 JSON 的多种容器布局里取出 (pair 记录, 默认模式, 默认类别) 列表。"""

    records: list[tuple[Mapping[str, Any], str | None, str]] = []

    def add_many(
        value: Any, default_mode: str | None, default_category: str = ""
    ) -> None:
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    records.append((item, default_mode, default_category))
        elif isinstance(value, Mapping):
            found = False
            for key in _UNEXPECTED_PAIR_CONTAINER_KEYS:
                child = value.get(key)
                if isinstance(child, list):
                    add_many(child, default_mode, default_category)
                    found = True
            for key in _UNEXPECTED_MODE_C_CATEGORIES:
                child = value.get(key)
                if isinstance(child, list):
                    add_many(child, default_mode, key)
                    found = True
            if not found and (
                value.get("a_work_id") is not None
                or value.get("work_id_a") is not None
            ):
                records.append((value, default_mode, default_category))

    if isinstance(raw, list):
        add_many(raw, None)
        return records
    if not isinstance(raw, Mapping):
        return records

    for key in ("pairs", "relations", "edges", "items"):
        add_many(raw.get(key), None)

    modes_value = raw.get("modes")
    if isinstance(modes_value, Mapping):
        for key, value in modes_value.items():
            default_mode = _unexpected_mode(key)
            if isinstance(value, str):
                continue
            add_many(value, default_mode)
    elif isinstance(modes_value, list):
        for entry in modes_value:
            if not isinstance(entry, Mapping):
                continue
            default_mode = _unexpected_mode(entry.get("mode", entry.get("id")))
            add_many(entry, default_mode)

    for key, value in raw.items():
        normalized = str(key).strip().lower()
        if not (normalized.startswith("mode_") or normalized in ("a", "b", "c")):
            continue
        default_mode = _unexpected_mode(normalized)
        if default_mode is None:
            continue
        add_many(value, default_mode)
    return records


def _unexpected_spaces(
    raw: Mapping[str, Any], pairs: Iterable[Mapping[str, Any]]
) -> tuple[str, ...]:
    spaces: list[str] = []
    raw_spaces = raw.get("spaces")
    candidates: list[str] = []
    if isinstance(raw_spaces, Mapping):
        candidates = [str(key) for key in raw_spaces]
    elif isinstance(raw_spaces, (list, tuple)):
        for item in raw_spaces:
            if isinstance(item, Mapping):
                key = item.get("key") or item.get("name") or item.get("label")
                if key:
                    candidates.append(str(key))
            else:
                candidates.append(str(item))
    for name in candidates:
        space = _unexpected_space(name)
        if space is not None and space not in spaces:
            spaces.append(space)
    for pair in pairs:
        for space in pair["per_space"]:
            if space not in spaces:
                spaces.append(space)
    ordered = [space for space in _UNEXPECTED_SPACE_ORDER if space in spaces]
    ordered.extend(space for space in spaces if space not in _UNEXPECTED_SPACE_ORDER)
    return tuple(ordered)


def load_unexpected_relations(config: Config) -> dict[str, Any] | None:
    """读取 logs/unexpected_relations.json（t5 discovery 产物）。

    文件不存在/不可解析时返回 None（界面提示，不影响默认星云）。
    兼容 modes/mode_a/mode_b/mode_c、pairs/relations 列表、work_pair/work_a 等
    多种字段命名；返回 {path,meta,spaces,pairs,by_mode,n_pairs}，pairs 带
    mode/a_work_id/b_work_id/score/category/per_space{rank,percentile,cosine}。
    """

    path = Path(config.paths.logs) / UNEXPECTED_RELATIONS_FILE_NAME
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("读取意外关系文件失败 %s: %s", path, exc)
        return None
    records = _unexpected_container_records(raw)
    pairs: list[dict[str, Any]] = []
    for record, default_mode, default_category in records:
        pair = _unexpected_record_pair(record, default_mode, default_category)
        if pair is not None:
            pairs.append(pair)
    counts: dict[str, int] = {}
    for pair in pairs:
        mode = str(pair["mode"])
        if mode:
            counts[mode] = counts.get(mode, 0) + 1
    meta = raw.get("meta") if isinstance(raw, Mapping) else None
    if not isinstance(meta, dict):
        meta = {}
    spaces = _unexpected_spaces(raw, pairs) if isinstance(raw, Mapping) else ()
    return {
        "path": str(path),
        "meta": meta,
        "spaces": spaces,
        "pairs": pairs,
        "by_mode": counts,
        "n_pairs": len(pairs),
    }


def unexpected_modes(relations: Mapping[str, Any]) -> list[dict[str, Any]]:
    """可用模式列表（id/label/description/count），label 优先取 meta 配置。"""

    meta = relations.get("meta") or {}
    labels_raw = meta.get("mode_labels") or meta.get("mode_label") or {}
    labels: dict[str, str] = {}
    if isinstance(labels_raw, Mapping):
        for key, value in labels_raw.items():
            mode = _unexpected_mode(key) or str(key)
            if value:
                labels[mode] = str(value)
    descriptions: dict[str, str] = {}
    definitions = meta.get("definitions")
    if isinstance(definitions, Mapping):
        for mode in ("a", "b", "c"):
            value = definitions.get(f"mode_{mode}", definitions.get(mode))
            if value:
                descriptions[mode] = str(value)
    desc_raw = meta.get("mode_descriptions") or meta.get("mode_description") or {}
    if isinstance(desc_raw, Mapping):
        for key, value in desc_raw.items():
            mode = _unexpected_mode(key) or str(key)
            if value:
                descriptions[mode] = str(value)
    meta_modes = meta.get("modes")
    if isinstance(meta_modes, Mapping):
        for key, value in meta_modes.items():
            mode = _unexpected_mode(key) or str(key)
            if isinstance(value, Mapping):
                if value.get("label"):
                    labels.setdefault(mode, str(value["label"]))
                if value.get("description"):
                    descriptions.setdefault(mode, str(value["description"]))
    elif isinstance(meta_modes, list):
        for entry in meta_modes:
            if not isinstance(entry, Mapping):
                continue
            mode = _unexpected_mode(entry.get("mode", entry.get("id")))
            if mode is None:
                continue
            if entry.get("label"):
                labels.setdefault(mode, str(entry["label"]))
            if entry.get("description"):
                descriptions.setdefault(mode, str(entry["description"]))

    by_mode = relations.get("by_mode") or {}
    ordered = [mode for mode in ("a", "b", "c") if mode in by_mode or mode in labels]
    ordered.extend(sorted(mode for mode in by_mode if mode not in ordered))
    result: list[dict[str, Any]] = []
    for mode in ordered:
        result.append(
            {
                "mode": mode,
                "label": labels.get(mode)
                or _UNEXPECTED_MODE_LABELS.get(mode, f"模式 {mode.upper()}"),
                "description": descriptions.get(mode) or "",
                "count": int(by_mode.get(mode, 0)),
            }
        )
    return result


def _unexpected_mutual(
    similarity_index: "SimilarityIndex | None",
    a_work_id: int,
    b_work_id: int,
    *,
    k: int = 5,
) -> tuple[bool, tuple[str, ...]]:
    if similarity_index is None:
        return False, ()
    kinds: list[str] = []
    for kind in similarity_index.available_kinds:
        if not (
            similarity_index.has_vector(a_work_id, kind)
            and similarity_index.has_vector(b_work_id, kind)
        ):
            continue
        forward = {
            int(item.work_id)
            for item in similarity_index.neighbors(a_work_id, kind=kind, k=int(k))
        }
        backward = {
            int(item.work_id)
            for item in similarity_index.neighbors(b_work_id, kind=kind, k=int(k))
        }
        if b_work_id in forward and a_work_id in backward:
            kinds.append(kind)
    return bool(kinds), tuple(kinds)


def unexpected_space_summary(
    per_space: Mapping[str, Mapping[str, Any]], spaces=None
) -> str:
    """把每空间 rank/percentile/cosine 压成表格/CSV 用的中文摘要。"""

    order = tuple(spaces) if spaces else _UNEXPECTED_SPACE_ORDER
    parts: list[str] = []
    for space in order:
        stats = per_space.get(space)
        if not isinstance(stats, Mapping):
            continue
        bits: list[str] = []
        rank = _int_or_none(stats.get("rank"))
        rank_a = _int_or_none(stats.get("rank_a"))
        rank_b = _int_or_none(stats.get("rank_b"))
        percentile = _float_or_none(stats.get("percentile", stats.get("bar_p")))
        cosine = _float_or_none(stats.get("cosine"))
        if rank is not None:
            bits.append(f"rank={rank}")
        elif rank_a is not None or rank_b is not None:
            bits.append(f"rA={rank_a}/rB={rank_b}")
        if percentile is not None:
            bits.append(f"pct={percentile:.1f}")
        if cosine is not None:
            bits.append(f"cos={cosine:.3f}")
        if bits:
            label = _UNEXPECTED_SPACE_LABELS.get(space, space)
            parts.append(f"{label}:" + "/".join(bits))
    return "；".join(parts)


def unexpected_pair_rows(
    relations: Mapping[str, Any],
    mode: str,
    *,
    frame: pd.DataFrame | None = None,
    similarity_index: "SimilarityIndex | None" = None,
    only_mutual: bool = False,
    k: int = 5,
) -> list[dict[str, Any]]:
    """当前模式的作品对展示行（按分数降序）；可只看互为近邻。"""

    normalized_mode = _unexpected_mode(mode) or str(mode)
    records = _record_lookup(frame) if frame is not None else {}
    rows: list[dict[str, Any]] = []
    for pair in relations.get("pairs", ()):  # type: ignore[union-attr]
        if str(pair["mode"]) != normalized_mode:
            continue
        a_work_id = int(pair["a_work_id"])
        b_work_id = int(pair["b_work_id"])
        mutual, mutual_kinds = _unexpected_mutual(
            similarity_index, a_work_id, b_work_id, k=k
        )
        if only_mutual and not mutual:
            continue
        record_a = records.get(a_work_id)
        record_b = records.get(b_work_id)
        title_a = _record_title(record_a, a_work_id)
        title_b = _record_title(record_b, b_work_id)
        if record_a is None and pair.get("title_a"):
            title_a = str(pair["title_a"])
        if record_b is None and pair.get("title_b"):
            title_b = str(pair["title_b"])
        rows.append(
            {
                "mode": normalized_mode,
                "a_work_id": a_work_id,
                "b_work_id": b_work_id,
                "a_title": title_a,
                "b_title": title_b,
                "a_url": external_url(record_a) if record_a is not None else None,
                "b_url": external_url(record_b) if record_b is not None else None,
                "score": float(pair["score"]),
                "category": str(pair["category"]),
                "per_space": pair["per_space"],
                "space_summary": unexpected_space_summary(
                    pair["per_space"], relations.get("spaces")
                ),
                "mutual": bool(mutual),
                "mutual_kinds": mutual_kinds,
                "in_map": record_a is not None and record_b is not None,
            }
        )
    rows.sort(key=lambda row: (-row["score"], row["a_work_id"], row["b_work_id"]))
    return rows


def unexpected_rows_csv(rows: Iterable[Mapping[str, Any]]) -> str:
    """当前意外关系表导出 CSV（低成本文本，不建持久库）。"""

    import csv
    import io

    buffer = io.StringIO()
    fieldnames = [
        "模式",
        "分数",
        "A work_id",
        "A 标题",
        "B work_id",
        "B 标题",
        "类别",
        "互为近邻",
        "空间统计",
        "A 链接",
        "B 链接",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "模式": row.get("mode", ""),
                "分数": row.get("score", ""),
                "A work_id": row.get("a_work_id", ""),
                "A 标题": row.get("a_title", ""),
                "B work_id": row.get("b_work_id", ""),
                "B 标题": row.get("b_title", ""),
                "类别": row.get("category", ""),
                "互为近邻": "是" if row.get("mutual") else "",
                "空间统计": row.get("space_summary", ""),
                "A 链接": row.get("a_url") or "",
                "B 链接": row.get("b_url") or "",
            }
        )
    return buffer.getvalue()
