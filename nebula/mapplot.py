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

from .config import Config, load_config
from .db import Catalog
from .errors import MapError, NebulaError
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


def load_embedding_matrix(
    cat: Catalog,
    cfg: Config,
    *,
    version: str | None = None,
) -> tuple[list[int], np.ndarray]:
    """读取某个向量版本的全部 work_joint 向量。

    返回 (work_ids, matrix)，其中 matrix 已再次逐行 L2 归一化，
    保证余弦距离等价于内积距离。作品数少于 MIN_WORKS 时抛 MapError。
    """

    embedding_version = version or cfg.embedding_version
    rows = cat.list_works_with_embedding(embedding_version, kind="work_joint")
    if len(rows) < MIN_WORKS:
        raise MapError(
            f"作品数量不足: 向量版本 {embedding_version} 只有 {len(rows)} 部作品，"
            f"至少需要 {MIN_WORKS} 部才能生成星云"
        )
    logger.info(
        "读取向量: version=%s works=%d dim=%s",
        embedding_version,
        len(rows),
        rows[0].get("embedding_dim"),
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


def _make_umap(cfg: Config, *, n_samples: int, n_components: int) -> Any:
    """按配置构造一个独立的 UMAP 模型。"""

    umap_module = _require_umap()
    n_neighbors = min(int(cfg.umap.n_neighbors), max(2, n_samples - 1))
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
) -> tuple[Any, np.ndarray]:
    """训练单个投影模型并校验输出形状。"""

    try:
        model = _make_umap(cfg, n_samples=int(matrix.shape[0]), n_components=n_components)
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
) -> tuple[Any, np.ndarray, Any, np.ndarray]:
    """分别训练 2D 与 3D 两个独立模型（不共享模型对象）。"""

    model_2d, coords_2d = _fit_projection(matrix, cfg, n_components=2)
    model_3d, coords_3d = _fit_projection(matrix, cfg, n_components=3)
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


def _umap_params(cfg: Config, *, n_works: int, dim: int, n_neighbors: int) -> dict[str, Any]:
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
        "n_works": int(n_works),
        "dim": int(dim),
        "umap_version": getattr(_require_umap(), "__version__", None),
    }


# --------------------------------------------------------------------------- 构建星云
def build_map(
    cat: Catalog,
    cfg: Config,
    *,
    version: str | None = None,
    note: str | None = None,
) -> MapBuildResult:
    """用全部作品向量重新训练 2D/3D 星云并登记为新版本。

    步骤：读取矩阵 -> 分别训练 2D/3D -> 写入新版本目录 -> 数据库原子登记。
    任意一步失败都会清理本版本目录，数据库不会留下半成品版本。
    """

    embedding_version = version or cfg.embedding_version
    work_ids, matrix = load_embedding_matrix(cat, cfg, version=embedding_version)
    n_works = len(work_ids)
    logger.info(
        "开始训练星云: embedding_version=%s n_works=%d dim=%d",
        embedding_version,
        n_works,
        matrix.shape[1],
    )

    model_2d, coords_2d, model_3d, coords_3d = _fit_both_projections(matrix, cfg)

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
        n_neighbors=min(int(cfg.umap.n_neighbors), max(2, n_works - 1)),
    )

    try:
        out_dir.mkdir(parents=True, exist_ok=False)
        _atomic_write_parquet(frame, coords_path)
        _dump_model(model_2d, model_2d_path)
        _dump_model(model_3d, model_3d_path)
        _atomic_write_json(
            {
                "version": map_version,
                "embedding_version": embedding_version,
                "n_works": n_works,
                "dim": int(matrix.shape[1]),
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

    logger.info("星云构建完成: %s，作品数=%d", map_version, n_works)
    return MapBuildResult(
        version=map_version,
        embedding_version=embedding_version,
        n_works=n_works,
        coords_path=coords_path,
        model_2d_path=model_2d_path,
        model_3d_path=model_3d_path,
    )


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

    rows = cat.list_works_with_embedding(embedding_version, kind="work_joint")
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


def _cmd_build(cfg: Config, *, embedding_version: str | None, note: str | None) -> int:
    cat = _open_catalog(cfg)
    try:
        result = build_map(cat, cfg, version=embedding_version, note=note)
    finally:
        cat.close()
    print(
        f"已生成星云 {result.version}: {result.n_works} 部作品，"
        f"向量版本 {result.embedding_version}"
    )
    print(f"  坐标: {result.coords_path}")
    print(f"  2D 模型: {result.model_2d_path}")
    print(f"  3D 模型: {result.model_3d_path}")
    return 0


def _cmd_list(cfg: Config) -> int:
    cat = _open_catalog(cfg)
    try:
        maps = cat.list_maps()
    finally:
        cat.close()
    if not maps:
        print("还没有生成任何星云版本。可运行：map.py build")
        return 0
    print(f"共 {len(maps)} 个星云版本（按创建时间从旧到新）:")
    for row in maps:
        note = f"  备注: {row['note']}" if row.get("note") else ""
        print(
            f"  {row['version']}  {int(row.get('n_works') or 0)} 部  "
            f"向量版本={row['embedding_version']}  状态={row['status']}  "
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


def main(argv: Sequence[str] | None = None) -> int:
    """map.py 命令行入口：build / list / transform。"""

    import argparse

    parser = argparse.ArgumentParser(
        prog="map.py",
        description="音MAD 星云：用 UMAP 生成 2D/3D 坐标，或把新作品增量加入已有星云",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="用全部作品向量重新训练 2D/3D 星云")
    build_parser.add_argument(
        "--config", default=None, help="配置文件路径（默认 NEBULA_CONFIG 或 config.yaml）"
    )
    build_parser.add_argument(
        "--embedding-version", default=None, help="向量版本（默认取配置中的版本）"
    )
    build_parser.add_argument("--note", default=None, help="版本备注，写入数据库")

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
            return _cmd_build(cfg, embedding_version=args.embedding_version, note=args.note)
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
