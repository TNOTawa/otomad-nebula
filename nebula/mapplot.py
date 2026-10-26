"""UMAP 二维/三维星云构建与增量更新。

流程（对应执行方案“降维与星云”一节）：

1. 从 Catalog 读取 work_joint 作品级向量（已 L2 归一化），再次归一化后
   用余弦距离训练 UMAP；
2. 分别训练 2D 与 3D 两个独立模型；
3. 坐标写入 data/maps/<map_version>/ 并原子登记到数据库；
4. P3 增量：用保存的模型 transform() 把新作品加入已有固定空间。

星云版本号形如 map_v001，每次完整重训都生成新版本并保留旧版本，
因为 UMAP 重新训练会整体改变旧点的位置（见执行方案 P2）。

外部元数据（年份、作者、排名、时长等）绝不参与降维，本模块只关心向量本身。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import representations
from .config import Config, load_config
from .db import Catalog
from .errors import MapError, NebulaError
from .representations import LEGACY_NAME, representation_label
from .util import now_utc_iso

logger = logging.getLogger(__name__)

# 星云版本目录内的固定文件名
COORDS_FILENAME = "coords.parquet"
MODEL_2D_FILENAME = "map_2d.joblib"
MODEL_3D_FILENAME = "map_3d.joblib"
META_FILENAME = "meta.json"

#: 坐标表列顺序（与 Catalog.create_map_run 的 coords 元组一致）
COORD_COLUMNS = ("work_id", "x2", "y2", "x3", "y3", "z3")

#: 生成星云所需的最少作品数；作品太少时 UMAP 结构没有意义
MIN_WORKS = 3

#: build_map(filter_spec=...) 支持的筛选键
FILTER_KEYS: frozenset[str] = frozenset(
    {"dataset", "vote_year", "list_key", "include_duplicates"}
)

#: dataset 过滤的合法取值
DATASETS: tuple[str, ...] = ("all", "local", "bilibili")

_DEFAULT_CONFIG_NAME = "config.yaml"


@dataclass(frozen=True)
class MapBuildResult:
    """一次完整星云构建的结果。"""

    version: str
    embedding_version: str
    n_works: int
    coords_path: Path
    model_2d_path: Path
    model_3d_path: Path
    #: v0.4：地图绑定的表征（旧地图/直接调用时可能为 None 或 legacy 表征）
    representation_id: int | None = None
    representation_name: str | None = None


# --------------------------------------------------------------------------- 向量读取
def _read_work_vectors(
    cfg: Config,
    work_rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """读取多部作品的 work_joint 向量并校验，返回 (n, dim) float32 矩阵。"""

    vectors: list[np.ndarray] = []
    dims: set[int] = set()
    for row in work_rows:
        work_id = int(row["id"])
        relpath = row.get("embedding_path")
        if not relpath:
            raise MapError(f"作品 {work_id} 缺少向量路径登记，请重新生成向量")
        path = cfg.data_path(str(relpath))
        if not path.is_file():
            raise MapError(f"作品 {work_id} 的向量文件不存在: {path}")
        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise MapError(f"读取作品 {work_id} 的向量失败: {path}: {exc}") from exc
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 2 and array.shape[0] == 1:
            array = array.reshape(-1)
        if array.ndim != 1 or array.size == 0:
            raise MapError(f"作品 {work_id} 的向量形状非法: {array.shape}，需要一维向量")
        registered_dim = row.get("embedding_dim")
        if registered_dim is not None and int(registered_dim) != int(array.shape[0]):
            raise MapError(
                f"作品 {work_id} 的向量维度与登记不一致: "
                f"文件 {array.shape[0]} 维，登记 {int(registered_dim)} 维"
            )
        if not np.all(np.isfinite(array)):
            raise MapError(f"作品 {work_id} 的向量包含非有限值（NaN/Inf），请重新生成")
        dims.add(int(array.shape[0]))
        vectors.append(array)

    if not vectors:
        raise MapError("没有可用的作品向量")
    if len(dims) > 1:
        raise MapError(f"作品向量维度不一致: {sorted(dims)}；请确认使用同一个向量版本")

    matrix = np.vstack(vectors)
    norms = np.linalg.norm(matrix, axis=1)
    zero_rows = np.where(norms <= 0)[0]
    if zero_rows.size:
        zero_ids = [int(work_rows[int(i)]["id"]) for i in zero_rows]
        raise MapError(f"以下作品的向量为零向量，无法做余弦降维: {zero_ids}")
    matrix = (matrix / norms[:, None]).astype(np.float32)
    return matrix


def _normalize_filter_spec(filter_spec: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """校验并规范化筛选条件；返回 None 表示没有筛选（全量 canonical 集合）。

    默认行为是不论是否传 filter_spec 都排除 work_links 的 duplicate 作品，
    只有显式 include_duplicates=True 才包含重复实体。
    """

    if filter_spec is None:
        return None
    if not isinstance(filter_spec, Mapping):
        raise MapError(f"filter_spec 必须是映射类型，收到 {type(filter_spec).__name__}")
    unknown = set(filter_spec) - FILTER_KEYS
    if unknown:
        raise MapError(
            f"未知的筛选条件: {', '.join(sorted(map(str, unknown)))}；"
            f"支持: {', '.join(sorted(FILTER_KEYS))}"
        )
    normalized: dict[str, Any] = {}
    dataset = filter_spec.get("dataset")
    if dataset is not None:
        if str(dataset) not in DATASETS:
            raise MapError(f"dataset 只支持 {', '.join(DATASETS)}，收到 {dataset!r}")
        if str(dataset) != "all":
            normalized["dataset"] = str(dataset)
    vote_year = filter_spec.get("vote_year")
    if vote_year is not None:
        try:
            normalized["vote_year"] = int(vote_year)
        except (TypeError, ValueError) as exc:
            raise MapError(f"vote_year 必须是整数，收到 {vote_year!r}") from exc
    list_key = filter_spec.get("list_key")
    if list_key is not None and str(list_key).strip():
        normalized["list_key"] = str(list_key).strip()
    include_duplicates = filter_spec.get("include_duplicates")
    if include_duplicates is not None:
        if not isinstance(include_duplicates, bool):
            raise MapError(f"include_duplicates 必须是布尔值，收到 {include_duplicates!r}")
        if include_duplicates:
            normalized["include_duplicates"] = True
    return normalized or None


def _parse_map_filter(map_row: Mapping[str, Any]) -> dict[str, Any] | None:
    """解析 map_runs.filter_json（兼容旧库中列为 NULL / 空串的情况）。"""

    raw = map_row.get("filter_json")
    if not raw:
        return None
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError) as exc:
        raise MapError(f"星云版本 {map_row.get('version')} 的 filter_json 无法解析: {exc}") from exc
    if not isinstance(parsed, dict):
        raise MapError(f"星云版本 {map_row.get('version')} 的 filter_json 不是对象")
    return _normalize_filter_spec(parsed)


def _describe_filter_spec(filter_spec: Mapping[str, Any] | None) -> str:
    """把筛选条件转成中文描述，便于错误提示与 CLI 输出。"""

    if not filter_spec:
        return "全部作品"
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
    return "，".join(parts)


def _select_embedding_rows(
    cat: Catalog,
    version: str,
    filter_spec: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """按筛选条件挑选参与降维的作品行（work_joint 向量行）。

    默认排除 work_links 中的 duplicate 实体；include_duplicates=True 时才保留。
    """

    spec = _normalize_filter_spec(filter_spec)
    rows = cat.list_works_with_embedding(version, kind="work_joint")
    include_duplicates = bool(spec.get("include_duplicates")) if spec else False
    if not include_duplicates:
        duplicate_ids = cat.get_duplicate_work_ids()
        rows = [row for row in rows if int(row["id"]) not in duplicate_ids]
    if spec and spec.get("dataset"):
        rows = [row for row in rows if str(row.get("source")) == spec["dataset"]]
    if spec and spec.get("list_key"):
        allowed = cat.work_ids_for_list(str(spec["list_key"]), only_canonical=not include_duplicates)
        rows = [row for row in rows if int(row["id"]) in allowed]
    if spec and spec.get("vote_year") is not None:
        allowed = cat.work_ids_for_vote_year(
            int(spec["vote_year"]), only_canonical=not include_duplicates
        )
        rows = [row for row in rows if int(row["id"]) in allowed]
    return rows




# --------------------------------------------------------------------------- 表征解析与矩阵
def _representation_info(rep: Mapping[str, Any]) -> dict[str, Any]:
    """把表征行压成写入 params/meta 的简明信息。"""

    return {
        "id": int(rep["id"]),
        "name": str(rep["name"]),
        "label": representation_label(str(rep["name"])),
        "sequence_version": str(rep["sequence_version"]),
        "kind": str(rep["kind"]),
        "source_kind": str(rep.get("source_kind") or "pe_av"),
    }


def _resolve_build_representation(
    cat: Catalog,
    cfg: Config,
    *,
    representation: str | None = None,
    representation_id: int | None = None,
    sequence_version: str | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    """解析 build_map 要使用的表征。

    - representation/representation_id 显式指定（可配 sequence_version）；
    - version 是 legacy 别名（其值即旧 embedding version）；
    - 都不给时按当前 sampling.mode 选择默认表征（legacy / full_mean_v1）。
    """

    if representation_id is not None:
        row = cat.get_representation(int(representation_id))
        if row is None:
            raise MapError(f"表征不存在: {representation_id}")
        if sequence_version is not None and str(row["sequence_version"]) != str(sequence_version):
            raise MapError(
                f"表征 {row['name']} 的序列版本是 {row['sequence_version']}，"
                f"与指定的 {sequence_version} 不一致"
            )
        return row
    if representation is not None:
        if version is not None:
            raise MapError("--embedding-version 是 legacy 别名，不能与 --representation 同时使用")
        try:
            return representations.resolve_representation(
                cat, cfg, name=str(representation), sequence_version=sequence_version
            )
        except NebulaError as exc:
            raise MapError(f"无法解析表征 {representation}: {exc}") from exc
    if version is not None:
        if sequence_version is not None and str(sequence_version) != str(version):
            raise MapError(
                f"--embedding-version={version} 与 --sequence-version={sequence_version} 不一致"
            )
        try:
            return representations.resolve_representation(
                cat, cfg, name=LEGACY_NAME, sequence_version=str(version)
            )
        except NebulaError as exc:
            raise MapError(f"无法解析 legacy 向量版本 {version}: {exc}") from exc
    if sequence_version is not None:
        try:
            return representations.resolve_representation(
                cat, cfg, sequence_version=str(sequence_version)
            )
        except NebulaError as exc:
            raise MapError(f"无法解析序列版本 {sequence_version} 的默认表征: {exc}") from exc
    try:
        return representations.resolve_representation(cat, cfg)
    except NebulaError as exc:
        raise MapError(f"无法确定默认表征（请先构建表征或显式指定 --representation）: {exc}") from exc


def _normalize_work_matrix(matrix: np.ndarray, *, where: str) -> np.ndarray:
    """逐行 L2 归一化表征矩阵并校验。"""

    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2 or array.size == 0:
        raise MapError(f"{where} 向量矩阵形状非法: {array.shape}")
    if not np.all(np.isfinite(array)):
        raise MapError(f"{where} 向量包含非有限值（NaN/Inf）")
    norms = np.linalg.norm(array.astype(np.float64), axis=1)
    zero_rows = np.where(norms <= 0)[0]
    if zero_rows.size:
        raise MapError(f"{where} 有 {zero_rows.size} 个零向量，无法做余弦降维")
    return (array / norms[:, None]).astype(np.float32)


def _filter_work_ids(
    cat: Catalog,
    work_ids: Sequence[int],
    filter_spec: Mapping[str, Any] | None,
) -> list[int]:
    """按 dataset/list_key/vote_year 过滤 work_id（仅用于派生表征路径）。"""

    spec = _normalize_filter_spec(filter_spec)
    ids = [int(work_id) for work_id in work_ids]
    if not spec:
        return ids
    if spec.get("dataset"):
        works = {int(work["id"]): work for work in cat.list_works()}
        ids = [
            work_id
            for work_id in ids
            if str(works.get(work_id, {}).get("source")) == str(spec["dataset"])
        ]
    if spec.get("list_key"):
        allowed = cat.work_ids_for_list(str(spec["list_key"]), only_canonical=True)
        ids = [work_id for work_id in ids if work_id in allowed]
    if spec.get("vote_year") is not None:
        allowed = cat.work_ids_for_vote_year(
            int(spec["vote_year"]), only_canonical=True
        )
        ids = [work_id for work_id in ids if work_id in allowed]
    return ids


def _load_representation_work_matrix(
    cat: Catalog,
    cfg: Config,
    rep: Mapping[str, Any],
    filter_spec: Mapping[str, Any] | None,
) -> tuple[list[int], np.ndarray, dict[str, Any] | None]:
    """读取某表征用于 UMAP 的作品级矩阵（joint），并应用界面筛选条件。"""

    normalized = _normalize_filter_spec(filter_spec)
    rep_kind = str(rep.get("kind"))
    if rep_kind == "legacy":
        # 旧路径完全保留（legacy 向量的筛选/include_duplicates 语义不变）
        work_ids, matrix = load_embedding_matrix(
            cat, cfg, version=str(rep["sequence_version"]), filter_spec=normalized
        )
        return work_ids, matrix, normalized

    if normalized and normalized.get("include_duplicates"):
        raise MapError(
            "派生表征（非 legacy）默认排除 work_links 重复实体，"
            "不能使用 include_duplicates=True"
        )
    try:
        ids, matrix = representations.load_representation_matrix(
            cat, cfg, rep, kind="joint"
        )
    except NebulaError as exc:
        raise MapError(
            f"读取表征 {rep['name']}@{rep['sequence_version']} 的联合向量失败: {exc}"
        ) from exc
    ids_list = [int(work_id) for work_id in ids]
    if normalized:
        keep = _filter_work_ids(cat, ids_list, normalized)
        position = {work_id: index for index, work_id in enumerate(ids_list)}
        if keep:
            matrix = np.stack([matrix[position[work_id]] for work_id in keep])
        else:
            matrix = np.empty((0, matrix.shape[1]), dtype=np.float32)
        ids_list = keep
    if len(ids_list) < MIN_WORKS:
        raise MapError(
            f"筛选后作品数量不足: 表征「{representation_label(str(rep['name']))}」"
            f"在条件「{_describe_filter_spec(normalized)}」下只有 {len(ids_list)} 部作品，"
            f"至少需要 {MIN_WORKS} 部才能生成星云"
        )
    matrix = _normalize_work_matrix(
        matrix, where=f"表征 {rep['name']}@{rep['sequence_version']}"
    )
    return ids_list, matrix, normalized


def load_embedding_matrix(
    cat: Catalog,
    cfg: Config,
    *,
    version: str | None = None,
    filter_spec: Mapping[str, Any] | None = None,
) -> tuple[list[int], np.ndarray]:
    """读取某个向量版本的 work_joint 向量矩阵。

    filter_spec 支持 dataset / vote_year / list_key / include_duplicates，
    默认排除 work_links 的 duplicate 实体。返回 (work_ids, matrix)，matrix
    已再次逐行 L2 归一化；筛选后作品数少于 MIN_WORKS 时抛 MapError。
    """

    embedding_version = version or cfg.embedding_version
    normalized = _normalize_filter_spec(filter_spec)
    rows = _select_embedding_rows(cat, embedding_version, normalized)
    if len(rows) < MIN_WORKS:
        raise MapError(
            f"筛选后作品数量不足: 向量版本 {embedding_version} 在条件"
            f"「{_describe_filter_spec(normalized)}」下只有 {len(rows)} 部作品，"
            f"至少需要 {MIN_WORKS} 部才能生成星云"
        )
    logger.info(
        "读取向量: version=%s works=%d dim=%s filter=%s",
        embedding_version,
        len(rows),
        rows[0].get("embedding_dim"),
        _describe_filter_spec(normalized),
    )
    matrix = _read_work_vectors(cfg, rows)
    work_ids = [int(row["id"]) for row in rows]
    return work_ids, matrix


# --------------------------------------------------------------------------- UMAP 训练
def _require_umap() -> Any:
    """惰性导入 umap-learn。

    导入 umap 会连带加载 scipy / numba / scikit-learn，耗时可达数秒；
    为了 map.py --help 与纯数据测试保持轻量，只在真正训练/transform 时导入。
    """

    try:
        import umap
    except ImportError as exc:  # pragma: no cover - 依赖已安装，仅作防御
        raise MapError("缺少 umap-learn 依赖，无法生成星云") from exc
    return umap


def _umap_init_mode(n_samples: int, n_components: int) -> str:
    """选择 UMAP 初始化方式。

    spectral 初始化需要样本数 > n_components + 1（否则 ARPACK 无法求解）；
    作品极少时退回 random 初始化，配合固定 random_state 仍然确定性。
    """

    return "spectral" if n_samples > n_components + 1 else "random"


def _effective_neighbors(cfg: Config, n_samples: int, n_neighbors: int | None) -> int:
    """实际使用的 n_neighbors：命令行/函数参数优先，且不能超过样本数 - 1。"""

    requested = int(cfg.umap.n_neighbors if n_neighbors is None else n_neighbors)
    if requested < 2:
        raise MapError(f"n_neighbors 必须 >= 2，收到 {requested}")
    return min(requested, max(2, n_samples - 1))


def _make_umap(
    cfg: Config,
    *,
    n_samples: int,
    n_components: int,
    n_neighbors: int | None = None,
) -> Any:
    """按配置构造一个独立的 UMAP 模型（n_neighbors 可被变体覆盖）。"""

    umap_module = _require_umap()
    n_neighbors = _effective_neighbors(cfg, n_samples, n_neighbors)
    return umap_module.UMAP(
        n_components=n_components,
        metric=cfg.umap.metric,
        n_neighbors=n_neighbors,
        min_dist=float(cfg.umap.min_dist),
        random_state=int(cfg.umap.random_state),
        n_jobs=int(cfg.umap.n_jobs),
        n_epochs=cfg.umap.n_epochs,
        init=_umap_init_mode(n_samples, n_components),
        verbose=False,
    )


def _fit_projection(
    matrix: np.ndarray,
    cfg: Config,
    *,
    n_components: int,
    n_neighbors: int | None = None,
) -> tuple[Any, np.ndarray]:
    """训练单个投影模型并校验输出形状。"""

    try:
        model = _make_umap(
            cfg,
            n_samples=int(matrix.shape[0]),
            n_components=n_components,
            n_neighbors=n_neighbors,
        )
        coords = np.asarray(model.fit_transform(matrix), dtype=np.float64)
    except MapError:
        raise
    except (ValueError, TypeError, MemoryError) as exc:
        raise MapError(f"UMAP {n_components}D 训练失败: {exc}") from exc
    expected = (int(matrix.shape[0]), n_components)
    if coords.shape != expected:
        raise MapError(f"UMAP {n_components}D 输出形状异常: {coords.shape}，期望 {expected}")
    if not np.all(np.isfinite(coords)):
        raise MapError(f"UMAP {n_components}D 输出包含非有限坐标")
    return model, coords


def _fit_both_projections(
    matrix: np.ndarray,
    cfg: Config,
    *,
    n_neighbors: int | None = None,
) -> tuple[Any, np.ndarray, Any, np.ndarray]:
    """分别训练 2D 与 3D 两个独立模型（不共享模型对象）。"""

    model_2d, coords_2d = _fit_projection(
        matrix, cfg, n_components=2, n_neighbors=n_neighbors
    )
    model_3d, coords_3d = _fit_projection(
        matrix, cfg, n_components=3, n_neighbors=n_neighbors
    )
    return model_2d, coords_2d, model_3d, coords_3d


# --------------------------------------------------------------------------- 原子写文件
def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _dump_model(model: Any, path: Path) -> None:
    """原子保存 joblib 模型（先写临时文件再替换）。"""

    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        joblib.dump(model, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _coords_frame(
    work_ids: Sequence[int],
    coords_2d: np.ndarray,
    coords_3d: np.ndarray,
) -> pd.DataFrame:
    if len(work_ids) != coords_2d.shape[0] or len(work_ids) != coords_3d.shape[0]:
        raise MapError("作品数量与坐标行数不一致")
    return pd.DataFrame(
        {
            "work_id": [int(work_id) for work_id in work_ids],
            "x2": coords_2d[:, 0],
            "y2": coords_2d[:, 1],
            "x3": coords_3d[:, 0],
            "y3": coords_3d[:, 1],
            "z3": coords_3d[:, 2],
        }
    )


def _coords_tuples(frame: pd.DataFrame) -> list[tuple[int, float, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float, float]] = []
    for record in frame[list(COORD_COLUMNS)].itertuples(index=False):
        rows.append(
            (
                int(record.work_id),
                float(record.x2),
                float(record.y2),
                float(record.x3),
                float(record.y3),
                float(record.z3),
            )
        )
    return rows


def _umap_params(
    cfg: Config,
    *,
    n_works: int,
    dim: int,
    n_neighbors: int,
    filter_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "metric": str(cfg.umap.metric),
        "n_neighbors": int(n_neighbors),
        "min_dist": float(cfg.umap.min_dist),
        "random_state": int(cfg.umap.random_state),
        "n_jobs": int(cfg.umap.n_jobs),
        "n_epochs": cfg.umap.n_epochs,
        "n_components": [2, 3],
        "init": {
            "2d": _umap_init_mode(n_works, 2),
            "3d": _umap_init_mode(n_works, 3),
        },
        "filter_spec": dict(filter_spec) if filter_spec else None,
        "n_works": int(n_works),
        "dim": int(dim),
        "umap_version": getattr(_require_umap(), "__version__", None),
    }


# --------------------------------------------------------------------------- 构建星云
def _validate_n_neighbors(n_neighbors: int | None) -> int | None:
    if n_neighbors is None:
        return None
    if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int):
        raise MapError(f"n_neighbors 必须是整数，收到 {n_neighbors!r}")
    if n_neighbors < 2:
        raise MapError(f"n_neighbors 必须 >= 2，收到 {n_neighbors}")
    return int(n_neighbors)


def build_map(
    cat: Catalog,
    cfg: Config,
    *,
    version: str | None = None,
    note: str | None = None,
    filter_spec: Mapping[str, Any] | None = None,
    n_neighbors: int | None = None,
    representation: str | None = None,
    representation_id: int | None = None,
    sequence_version: str | None = None,
) -> MapBuildResult:
    """用作品向量训练 2D/3D 星云并登记为新版本。

    参数
    ----
    representation / representation_id / sequence_version:
        v0.4 表征空间定位。显式指定时通过 nebula.representations 读取作品级向量；
        不指定时按当前 sampling.mode 选择默认表征（legacy 或 full_mean_v1）。
    version:
        legacy 别名（值即旧 embedding version），仅用于兼容旧命令/测试。
    filter_spec:
        可选筛选条件（dataset / vote_year / list_key / include_duplicates）；
        默认排除 work_links 的 duplicate 实体。筛选条件写入 map_runs.filter_json
        与 meta.json，便于区分不同子集生成的星云坐标。
    n_neighbors:
        覆盖配置中的 umap.n_neighbors，并记录到 params（用于变体对比）。

    步骤：解析表征 -> 读取矩阵 -> 分别训练 2D/3D -> 写入新版本目录 ->
    数据库原子登记（记录 representation_id 与 sequence version）。
    任意一步失败都会清理本版本目录，数据库不会留下半成品版本。
    """

    normalized_filter = _normalize_filter_spec(filter_spec)
    override_neighbors = _validate_n_neighbors(n_neighbors)
    explicit_representation = (
        representation is not None
        or representation_id is not None
        or sequence_version is not None
        or version is not None
    )
    try:
        rep = _resolve_build_representation(
            cat,
            cfg,
            representation=representation,
            representation_id=representation_id,
            sequence_version=sequence_version,
            version=version,
        )
    except MapError:
        if explicit_representation:
            raise
        # 旧行为兼容：数据库尚未登记表征/序列版本时，沿用 embeddings 路径，
        # 让“作品数量不足”等旧错误信息与旧测试保持稳定。
        rep = None

    if rep is None:
        embedding_version = str(version or cfg.legacy_embedding_version)
        work_ids, matrix = load_embedding_matrix(
            cat, cfg, version=embedding_version, filter_spec=normalized_filter
        )
        rep_info: dict[str, Any] | None = None
        representation_id_value: int | None = None
    else:
        embedding_version = str(rep["sequence_version"])
        work_ids, matrix, normalized_filter = _load_representation_work_matrix(
            cat, cfg, rep, normalized_filter
        )
        rep_info = _representation_info(rep)
        representation_id_value = int(rep["id"])
    n_works = len(work_ids)
    actual_neighbors = _effective_neighbors(cfg, n_works, override_neighbors)
    logger.info(
        "开始训练星云: representation=%s@%s n_works=%d dim=%d filter=%s n_neighbors=%d",
        rep_info["name"] if rep_info else "legacy(embeddings)",
        embedding_version,
        n_works,
        matrix.shape[1],
        _describe_filter_spec(normalized_filter),
        actual_neighbors,
    )

    model_2d, coords_2d, model_3d, coords_3d = _fit_both_projections(
        matrix, cfg, n_neighbors=override_neighbors
    )

    map_version = cat.next_map_version()
    out_dir = cfg.paths.maps / map_version
    coords_path = out_dir / COORDS_FILENAME
    model_2d_path = out_dir / MODEL_2D_FILENAME
    model_3d_path = out_dir / MODEL_3D_FILENAME
    meta_path = out_dir / META_FILENAME
    frame = _coords_frame(work_ids, coords_2d, coords_3d)
    params = _umap_params(
        cfg,
        n_works=n_works,
        dim=int(matrix.shape[1]),
        n_neighbors=actual_neighbors,
        filter_spec=normalized_filter,
    )
    params["representation"] = rep_info

    try:
        out_dir.mkdir(parents=True, exist_ok=False)
        _atomic_write_parquet(frame, coords_path)
        _dump_model(model_2d, model_2d_path)
        _dump_model(model_3d, model_3d_path)
        _atomic_write_json(
            {
                "version": map_version,
                "embedding_version": embedding_version,
                "representation": rep_info,
                "n_works": n_works,
                "dim": int(matrix.shape[1]),
                "filter_spec": normalized_filter,
                "created_at": now_utc_iso(),
                "params": params,
                "files": {
                    "coords": COORDS_FILENAME,
                    "model_2d": MODEL_2D_FILENAME,
                    "model_3d": MODEL_3D_FILENAME,
                },
            },
            meta_path,
        )
        cat.create_map_run(
            map_version,
            embedding_version=embedding_version,
            params=params,
            coords=_coords_tuples(frame),
            coords_path=cfg.data_relpath(coords_path),
            model_2d_path=cfg.data_relpath(model_2d_path),
            model_3d_path=cfg.data_relpath(model_3d_path),
            note=note,
            filter_spec=normalized_filter,
            representation_id=representation_id_value,
        )
    except BaseException as exc:  # noqa: BLE001 - 清理半成品后重抛
        logger.exception("星云构建失败，清理半成品目录: %s", out_dir)
        shutil.rmtree(out_dir, ignore_errors=True)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, MapError):
            raise
        if isinstance(exc, NebulaError):
            raise MapError(f"星云登记失败（{map_version}）: {exc}") from exc
        raise MapError(f"生成星云失败（{map_version}）: {exc}") from exc

    logger.info(
        "星云构建完成: %s，表征=%s，作品数=%d",
        map_version,
        rep_info["name"] if rep_info else "legacy(embeddings)",
        n_works,
    )
    return MapBuildResult(
        version=map_version,
        embedding_version=embedding_version,
        n_works=n_works,
        coords_path=coords_path,
        model_2d_path=model_2d_path,
        model_3d_path=model_3d_path,
        representation_id=representation_id_value,
        representation_name=None if rep_info is None else rep_info["name"],
    )


def build_map_variants(
    cat: Catalog,
    cfg: Config,
    *,
    n_neighbors_list: Sequence[int] = (8, 15, 30),
    filter_spec: Mapping[str, Any] | None = None,
    note_prefix: str | None = None,
    representation: str | None = None,
    representation_id: int | None = None,
    sequence_version: str | None = None,
    version: str | None = None,
) -> list[MapBuildResult]:
    """依次用不同 n_neighbors 生成多个星云版本，便于对比不同邻域尺度下的结构。

    每个变体都会重新训练 2D/3D 模型并登记为新版本（保留旧版本，不覆盖）。
    注意：不同变体的坐标之间不可直接比较绝对位置。
    """

    candidates = list(n_neighbors_list)
    if not candidates:
        raise MapError("n_neighbors_list 不能为空")
    neighbors: list[int] = []
    for value in candidates:
        validated = _validate_n_neighbors(value)
        if validated is None:  # pragma: no cover - 循环内不会出现 None
            continue
        if validated in neighbors:
            continue
        neighbors.append(validated)
    results: list[MapBuildResult] = []
    for value in neighbors:
        prefix = f"{note_prefix} " if note_prefix else ""
        note = f"{prefix}n_neighbors={value}"
        logger.info("构建星云变体: n_neighbors=%d filter=%s", value, _describe_filter_spec(
            _normalize_filter_spec(filter_spec)
        ))
        results.append(
            build_map(
                cat,
                cfg,
                filter_spec=filter_spec,
                n_neighbors=value,
                note=note,
                representation=representation,
                representation_id=representation_id,
                sequence_version=sequence_version,
                version=version,
            )
        )
    return results


# --------------------------------------------------------------------------- 增量更新
def _load_model(path: Path, label: str) -> Any:
    if not path.is_file():
        raise MapError(f"星云模型文件缺失（{label}）: {path}")
    try:
        model = joblib.load(path)
    except (OSError, ValueError) as exc:
        raise MapError(f"读取星云模型失败（{label}）: {path}: {exc}") from exc
    if not hasattr(model, "transform"):
        raise MapError(f"星云模型对象非法（{label}）: {path} 不是可用的 UMAP 模型")
    return model


def _read_coords(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise MapError(f"星云坐标文件缺失: {path}")
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise MapError(f"读取星云坐标失败: {path}: {exc}") from exc
    missing = [column for column in COORD_COLUMNS if column not in frame.columns]
    if missing:
        raise MapError(f"星云坐标文件缺少列 {missing}: {path}")
    return frame


def transform_new(
    cat: Catalog,
    cfg: Config,
    *,
    map_version: str | None = None,
) -> int:
    """把缺坐标的新作品 transform 进已有星云，返回新增作品数。

    旧作品坐标保持完全不变；只追加新行并写回 parquet 与数据库。
    没有任何新作品时返回 0，不修改任何文件。
    """

    if map_version is None:
        map_row = cat.latest_map()
        if map_row is None:
            raise MapError("还没有可用的星云版本，请先运行：map.py build")
        version = str(map_row["version"])
    else:
        version = str(map_version)
        map_row = cat.get_map(version)
        if map_row is None:
            raise MapError(f"星云版本不存在: {version}")
    if str(map_row.get("status")) != "ready":
        raise MapError(
            f"星云版本 {version} 状态不是 ready（当前 {map_row.get('status')}），无法增量更新"
        )

    embedding_version = str(map_row["embedding_version"])
    coords_rel = map_row.get("coords_path")
    model_2d_rel = map_row.get("model_2d_path")
    model_3d_rel = map_row.get("model_3d_path")
    if not coords_rel or not model_2d_rel or not model_3d_rel:
        raise MapError(f"星云版本 {version} 缺少文件登记，无法增量更新")
    coords_path = cfg.data_path(str(coords_rel))
    model_2d_path = cfg.data_path(str(model_2d_rel))
    model_3d_path = cfg.data_path(str(model_3d_rel))

    coords_frame = _read_coords(coords_path)
    existing_ids = {int(value) for value in coords_frame["work_id"].tolist()}
    filter_spec = _parse_map_filter(map_row)

    representation_id = map_row.get("representation_id")
    rep_row = (
        None
        if representation_id is None
        else cat.get_representation(int(representation_id))
    )
    if representation_id is not None and rep_row is None:
        raise MapError(f"星云版本 {version} 关联的表征不存在: {representation_id}")

    if rep_row is not None and str(rep_row.get("kind")) != "legacy":
        # v0.4 派生表征：新作品向量来自 representation_vectors
        try:
            vectors = representations.load_representation_vectors(
                cat, cfg, rep_row, "joint"
            )
        except NebulaError as exc:
            raise MapError(
                f"读取表征 {rep_row['name']} 的向量失败，无法增量更新: {exc}"
            ) from exc
        candidate_ids = _filter_work_ids(cat, sorted(vectors), filter_spec)
        new_ids = [
            work_id for work_id in candidate_ids if int(work_id) not in existing_ids
        ]
        if not new_ids:
            logger.info("星云 %s 没有需要增量加入的新作品", version)
            return 0
        logger.info("增量更新星云 %s（表征 %s）: 新增作品 %d 部", version, rep_row["name"], len(new_ids))
        new_matrix = _normalize_work_matrix(
            np.stack([vectors[work_id] for work_id in new_ids]),
            where=f"表征 {rep_row['name']}",
        )
    else:
        # legacy 表征 / 旧地图：完全复用旧的 embeddings 路径
        rows = _select_embedding_rows(cat, embedding_version, filter_spec)
        new_rows = [row for row in rows if int(row["id"]) not in existing_ids]
        if not new_rows:
            logger.info("星云 %s 没有需要增量加入的新作品", version)
            return 0
        new_ids = [int(row["id"]) for row in new_rows]
        logger.info("增量更新星云 %s: 新增作品 %d 部", version, len(new_ids))
        new_matrix = _read_work_vectors(cfg, new_rows)
    model_2d = _load_model(model_2d_path, "2D")
    model_3d = _load_model(model_3d_path, "3D")

    try:
        coords_2d = np.asarray(model_2d.transform(new_matrix), dtype=np.float64)
        coords_3d = np.asarray(model_3d.transform(new_matrix), dtype=np.float64)
    except (ValueError, MemoryError) as exc:
        raise MapError(f"UMAP transform 失败（新作品向量与星云模型不匹配？）: {exc}") from exc
    if coords_2d.shape != (len(new_ids), 2) or coords_3d.shape != (len(new_ids), 3):
        raise MapError(
            f"UMAP transform 输出形状异常: 2D={coords_2d.shape}, 3D={coords_3d.shape}"
        )
    if not np.all(np.isfinite(coords_2d)) or not np.all(np.isfinite(coords_3d)):
        raise MapError("UMAP transform 输出包含非有限坐标")

    new_frame = _coords_frame(new_ids, coords_2d, coords_3d)
    merged = pd.concat([coords_frame, new_frame], ignore_index=True)
    merged = merged.sort_values("work_id", kind="stable").reset_index(drop=True)
    backup = coords_frame.copy()

    try:
        _atomic_write_parquet(merged, coords_path)
        cat.replace_map_coords(version, _coords_tuples(merged))
    except BaseException as exc:  # noqa: BLE001 - 回滚 parquet 后重抛
        logger.exception("增量更新失败，尝试恢复原坐标文件: %s", coords_path)
        try:
            _atomic_write_parquet(backup, coords_path)
        except OSError:
            logger.exception("恢复原星云坐标文件失败: %s", coords_path)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, MapError):
            raise
        if isinstance(exc, NebulaError):
            raise MapError(f"增量更新星云失败（{version}）: {exc}") from exc
        raise MapError(f"增量更新星云失败（{version}）: {exc}") from exc

    logger.info(
        "星云 %s 增量更新完成: 新增 %d 部，总作品数 %d", version, len(new_ids), len(merged)
    )
    return len(new_ids)


# --------------------------------------------------------------------------- CLI
def _resolve_config_path(value: str | None) -> Path:
    """解析配置文件路径：命令行优先，其次 NEBULA_CONFIG，最后项目根目录。"""

    if value:
        return Path(value)
    env_value = os.environ.get("NEBULA_CONFIG")
    if env_value:
        return Path(env_value)
    # nebula/mapplot.py -> 项目根目录；与 app.py 的默认行为保持一致
    return Path(__file__).resolve().parents[1] / _DEFAULT_CONFIG_NAME


def _open_catalog(cfg: Config) -> Catalog:
    cat = Catalog(cfg.paths.db_path)
    cat.initialize()
    return cat


def _filter_spec_from_args(args: Any) -> dict[str, Any] | None:
    """把 CLI 参数拼成 filter_spec；全部为默认值时返回 None。"""

    spec: dict[str, Any] = {}
    dataset = getattr(args, "dataset", None)
    if dataset is not None and str(dataset) != "all":
        spec["dataset"] = str(dataset)
    vote_year = getattr(args, "vote_year", None)
    if vote_year is not None:
        spec["vote_year"] = int(vote_year)
    list_key = getattr(args, "list_key", None)
    if list_key:
        spec["list_key"] = str(list_key)
    if getattr(args, "include_duplicates", False):
        spec["include_duplicates"] = True
    return spec or None


def _print_build_result(result: MapBuildResult, filter_spec: Mapping[str, Any] | None) -> None:
    rep_label = (
        representation_label(result.representation_name)
        if result.representation_name
        else "未登记"
    )
    print(
        f"已生成星云 {result.version}: {result.n_works} 部作品，"
        f"表征={rep_label}，序列版本 {result.embedding_version}"
    )
    print(f"  筛选条件: {_describe_filter_spec(filter_spec)}")
    print(f"  坐标: {result.coords_path}")
    print(f"  2D 模型: {result.model_2d_path}")
    print(f"  3D 模型: {result.model_3d_path}")
    if filter_spec:
        print("  注意：筛选后的坐标与总星云坐标不可直接比较绝对位置")


def _representation_args_from_cli(args: Any) -> dict[str, Any]:
    return {
        "representation": getattr(args, "representation", None),
        "sequence_version": getattr(args, "sequence_version", None),
        "version": getattr(args, "embedding_version", None),
    }


def _cmd_build(cfg: Config, args: Any) -> int:
    filter_spec = _filter_spec_from_args(args)
    cat = _open_catalog(cfg)
    try:
        result = build_map(
            cat,
            cfg,
            note=getattr(args, "note", None),
            filter_spec=filter_spec,
            n_neighbors=getattr(args, "n_neighbors", None),
            **_representation_args_from_cli(args),
        )
    finally:
        cat.close()
    _print_build_result(result, _normalize_filter_spec(filter_spec))
    return 0


def _cmd_variants(cfg: Config, args: Any) -> int:
    filter_spec = _filter_spec_from_args(args)
    cat = _open_catalog(cfg)
    try:
        results = build_map_variants(
            cat,
            cfg,
            n_neighbors_list=tuple(int(value) for value in args.n_neighbors),
            filter_spec=filter_spec,
            note_prefix=getattr(args, "note_prefix", None),
            **_representation_args_from_cli(args),
        )
    finally:
        cat.close()
    print(f"已生成 {len(results)} 个星云变体（n_neighbors 不同，坐标不可直接比较）:")
    for result in results:
        _print_build_result(result, _normalize_filter_spec(filter_spec))
    return 0


def _cmd_list(cfg: Config) -> int:
    cat = _open_catalog(cfg)
    try:
        maps = cat.list_maps()
        representations_by_id = {
            int(row["id"]): row for row in cat.list_representations()
        }
    finally:
        cat.close()
    if not maps:
        print("还没有生成任何星云版本。可运行：map.py build")
        return 0
    print(f"共 {len(maps)} 个星云版本（按创建时间从旧到新）:")
    for row in maps:
        note = f"  备注: {row['note']}" if row.get("note") else ""
        try:
            filter_spec = _parse_map_filter(row)
        except MapError:
            filter_spec = None
        params = {}
        try:
            params = json.loads(str(row.get("params_json") or "{}"))
        except (TypeError, ValueError):
            params = {}
        neighbors = params.get("n_neighbors") if isinstance(params, dict) else None
        rep = representations_by_id.get(int(row.get("representation_id") or 0))
        rep_label = (
            representation_label(str(rep["name"])) if rep is not None else "未登记（旧地图）"
        )
        source_kind = str(row.get("source_kind") or "pe_av")
        version_label = "视觉版本" if source_kind == "visual" else "序列版本"
        print(
            f"  {row['version']}  {int(row.get('n_works') or 0)} 部  "
            f"表征={rep_label}  {version_label}={row['embedding_version']}  状态={row['status']}  "
            f"n_neighbors={neighbors}  筛选={_describe_filter_spec(filter_spec)}  "
            f"创建于 {row['created_at']}{note}"
        )
    return 0


def _cmd_transform(cfg: Config, *, map_version: str | None) -> int:
    cat = _open_catalog(cfg)
    try:
        added = transform_new(cat, cfg, map_version=map_version)
        version = map_version or (cat.latest_map() or {}).get("version")
    finally:
        cat.close()
    if added == 0:
        print(f"星云 {version or ''} 没有需要增量加入的新作品")
    else:
        print(f"星云 {version} 增量更新完成: 新增 {added} 部作品（旧作品坐标保持不变）")
    return 0


def _add_representation_arguments(parser: Any) -> None:
    """为 build / variants 子命令添加表征空间参数（v0.4）。"""

    parser.add_argument(
        "--representation",
        default=None,
        help=(
            "表征名：legacy_5clip_mean / full_mean_v1 / temporal_stats_v1"
            "（默认按 sampling.mode 选择）"
        ),
    )
    parser.add_argument(
        "--sequence-version",
        default=None,
        help="序列版本（缺省时用表征已登记的版本）",
    )


def _add_filter_arguments(parser: Any) -> None:
    """为 build / variants 子命令添加筛选相关参数。"""

    parser.add_argument(
        "--dataset",
        choices=list(DATASETS),
        default=None,
        help="按来源筛选作品：all / local / bilibili（默认 all）",
    )
    parser.add_argument("--vote-year", type=int, default=None, help="只保留该选票年份出现的作品")
    parser.add_argument("--list-key", default=None, help="只保留该榜单出现的作品，例如 votes_2025")
    parser.add_argument(
        "--include-duplicates",
        action="store_true",
        help="包含 work_links 中登记的重复实体（默认排除）",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """map.py 命令行入口：build / variants / list / transform。"""

    import argparse

    parser = argparse.ArgumentParser(
        prog="map.py",
        description="音MAD 星云：用 UMAP 生成 2D/3D 坐标，或把新作品增量加入已有星云",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build", help="按筛选条件用作品向量重新训练 2D/3D 星云"
    )
    build_parser.add_argument(
        "--config", default=None, help="配置文件路径（默认 NEBULA_CONFIG 或 config.yaml）"
    )
    build_parser.add_argument(
        "--embedding-version",
        default=None,
        help="legacy 别名：旧 embedding version（仅用于 legacy 表征，不能与 --representation 同用）",
    )
    build_parser.add_argument("--note", default=None, help="版本备注，写入数据库")
    build_parser.add_argument(
        "--n-neighbors", type=int, default=None, help="覆盖配置中的 UMAP n_neighbors"
    )
    _add_representation_arguments(build_parser)
    _add_filter_arguments(build_parser)

    variants_parser = subparsers.add_parser(
        "variants", help="依次用不同 n_neighbors 生成多个星云版本以便对比"
    )
    variants_parser.add_argument(
        "--config", default=None, help="配置文件路径（默认 NEBULA_CONFIG 或 config.yaml）"
    )
    variants_parser.add_argument(
        "--embedding-version",
        default=None,
        help="legacy 别名：旧 embedding version（仅用于 legacy 表征）",
    )
    variants_parser.add_argument(
        "--n-neighbors",
        type=int,
        nargs="+",
        default=[8, 15, 30],
        help="要对比的 n_neighbors 列表（默认 8 15 30）",
    )
    variants_parser.add_argument("--note-prefix", default=None, help="版本备注前缀")
    _add_representation_arguments(variants_parser)
    _add_filter_arguments(variants_parser)

    list_parser = subparsers.add_parser("list", help="列出全部星云版本")
    list_parser.add_argument(
        "--config", default=None, help="配置文件路径（默认 NEBULA_CONFIG 或 config.yaml）"
    )

    transform_parser = subparsers.add_parser("transform", help="用已有模型把新作品增量加入星云")
    transform_parser.add_argument(
        "--config", default=None, help="配置文件路径（默认 NEBULA_CONFIG 或 config.yaml）"
    )
    transform_parser.add_argument(
        "--map-version", default=None, help="星云版本（默认最新 ready 版本）"
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        cfg = load_config(_resolve_config_path(args.config))
    except NebulaError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    logging.basicConfig(
        level=getattr(logging, str(cfg.logging.level).upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.command == "build":
            return _cmd_build(cfg, args)
        if args.command == "variants":
            return _cmd_variants(cfg, args)
        if args.command == "list":
            return _cmd_list(cfg)
        if args.command == "transform":
            return _cmd_transform(cfg, map_version=args.map_version)
    except NebulaError as exc:
        logger.debug("命令执行失败", exc_info=True)
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    print(f"错误: 未知子命令 {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover - 供 python -m 直接运行
    raise SystemExit(main())
