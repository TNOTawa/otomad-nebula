"""从已保存的密集视觉窗口向量构建 CLIP-only / DINOv2-only 表征（不重编码）。

读取 data/visual_dense/<visual_version>/<work_id>/windows.npy（8s 窗口池化向量），
对每部作品做窗口加权平均再 L2 归一化，登记为 kind=visual 的表征
（dense_visual_clip_v1 / dense_visual_dino_v1），供 map.py 构建对照星云。

用法::

    python tools/build_dense_visual_representation.py --all-main
    python tools/build_dense_visual_representation.py --visual-version <version> --name dense_visual_clip_v1 --force
    python tools/build_dense_visual_representation.py --v2 --force
    python tools/build_dense_visual_representation.py --window-plan win8s_ov0__coverage-v1 --frame-version <frame_version>

v1 为等权/旧 manifest 权重历史行为；v2 从 v0.4.3.1 窗口计划层读取显式覆盖权重。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nebula import representations  # noqa: E402
from nebula.config import Config, load_config  # noqa: E402
from nebula.db import Catalog  # noqa: E402
from nebula.dense_visual import compress_version, visual_version_slug  # noqa: E402
from nebula.errors import NebulaError  # noqa: E402
from nebula.log import configure_logging  # noqa: E402

LOGGER = __import__("logging").getLogger("build_dense_visual_representation")

VISUAL_REP_NAMES: dict[str, str] = {
    "clip": representations.DENSE_VISUAL_CLIP_V1,
    "dinov2": representations.DENSE_VISUAL_DINO_V1,
}
VISUAL_REP_NAMES_V2: dict[str, str] = {
    "clip": representations.DENSE_VISUAL_CLIP_V2,
    "dinov2": representations.DENSE_VISUAL_DINO_V2,
}
"""编码器标记 -> 表征名（v1=等权历史；v2=窗口计划 + 覆盖权重）。"""


def aggregate_window_vectors(
    windows: np.ndarray, weights: Sequence[float] | None = None
) -> np.ndarray:
    """窗口向量 -> 作品向量：加权平均后 L2 归一化。

    参数
    ----
    windows:
        (T, D) 窗口级向量（已逐窗口 L2 归一化）。
    weights:
        可选 (T,) 窗口权重；缺省等权。权重会归一化到和为 1。
    """

    matrix = np.asarray(windows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise NebulaError(f"窗口向量形状非法: {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise NebulaError("窗口向量包含 NaN/Inf")
    if weights is None:
        weight = np.full((matrix.shape[0],), 1.0 / matrix.shape[0], dtype=np.float64)
    else:
        weight = np.asarray(weights, dtype=np.float64)
        if weight.shape != (matrix.shape[0],):
            raise NebulaError(f"窗口权重形状非法: {weight.shape} != ({matrix.shape[0]},)")
        if not np.all(np.isfinite(weight)) or float(weight.sum()) <= 0:
            raise NebulaError("窗口权重必须是有限非负且和大于 0")
        weight = weight / float(weight.sum())
    mean = matrix.T @ weight
    norm = float(np.linalg.norm(mean))
    if norm <= 1e-12:
        raise NebulaError("窗口加权平均向量范数为 0")
    return (mean / norm).astype(np.float32)


def _manifest_weights(work_dir: Path, n_windows: int) -> np.ndarray | None:
    """从 manifest 的 window_times 读取窗口权重（支持 [start,end] 或 dict）。"""

    path = work_dir / "manifest.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    explicit = payload.get("window_weights")
    if isinstance(explicit, list) and len(explicit) == n_windows:
        try:
            values = [float(item) for item in explicit]
        except (TypeError, ValueError):
            values = []
        if values and all(np.isfinite(value) and value >= 0 for value in values) and sum(values) > 0:
            return np.asarray(values, dtype=np.float64)
    times = payload.get("window_times")
    if not isinstance(times, list) or len(times) != n_windows:
        return None
    weights: list[float] = []
    for entry in times:
        value = None
        if isinstance(entry, Mapping):
            for key in ("weight", "window_weight"):
                if entry.get(key) is not None:
                    value = float(entry[key])
                    break
        elif isinstance(entry, (list, tuple)) and len(entry) >= 4:
            value = float(entry[3])
        if value is None:
            return None
        weights.append(value)
    return np.asarray(weights, dtype=np.float64)


def _encoder_tag(visual_version: str, model_id: str) -> str:
    haystack = f"{visual_version} {model_id}".lower()
    if "clip" in haystack:
        return "clip"
    if "dinov2" in haystack or "dino" in haystack:
        return "dinov2"
    raise NebulaError(f"无法从视觉版本判断编码器类型: {visual_version}")


def _resolve_main_versions(config: Config) -> list[str]:
    versions: list[str] = []
    for encoder in config.visual_dense.encoders:
        raw = compress_version(
            f"{encoder.model}@{encoder.revision}",
            float(config.visual_dense.fps),
            int(config.visual_dense.size),
            str(config.visual_dense.preprocess),
        )
        versions.append(visual_version_slug(raw))
    return versions


def _load_weight_file(path: Path, n_windows: int) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    weights = np.asarray(array, dtype=np.float64).reshape(-1)
    if weights.shape != (n_windows,):
        raise NebulaError(f"窗口权重数量 {weights.shape[0]} != 窗口数 {n_windows}: {path}")
    if not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0:
        raise NebulaError(f"窗口权重非法: {path}")
    return weights


def build_representation(
    catalog: Catalog,
    config: Config,
    *,
    visual_version: str,
    name: str | None = None,
    force: bool = False,
    limit: int | None = None,
    window_plan_version: str | None = None,
) -> dict[str, Any]:
    """构建一个视觉表征；返回统计摘要。

    window_plan_version 为空时走 v1 历史路径（visual_sequences 内置窗口 + 等权/旧 manifest
    权重）；给出时走 v0.4.3.1 的窗口计划层：从 visual_window_sequences 读取窗口与显式
    覆盖权重，默认登记 dense_visual_*_v2。
    """

    version_row = catalog.get_visual_version(visual_version)
    if version_row is None:
        raise NebulaError(
            f"视觉版本未登记: {visual_version}（先运行 tools/visual_pipeline.py run）"
        )
    encoder_tag = _encoder_tag(visual_version, str(version_row["model_id"]))
    if window_plan_version:
        plan_row = catalog.get_visual_window_plan(window_plan_version)
        if plan_row is None:
            raise NebulaError(
                f"窗口计划未登记: {window_plan_version}（先运行 tools/build_visual_windows.py）"
            )
        rep_name = name or VISUAL_REP_NAMES_V2[encoder_tag]
        if rep_name not in VISUAL_REP_NAMES_V2.values():
            raise NebulaError(
                f"窗口计划模式请使用 v2 表征名（{sorted(VISUAL_REP_NAMES_V2.values())}），收到 {rep_name}"
            )
        params = {
            "aggregation": "coverage-weighted-window-mean-l2",
            "source": "visual_window_sequences",
            "visual_version": visual_version,
            "window_plan_version": window_plan_version,
            "weight_scheme": str(plan_row["weight_scheme"]),
            "clip_seconds": float(plan_row["clip_seconds"]),
            "overlap_seconds": float(plan_row["overlap_seconds"]),
            "encoder": version_row["model_id"],
            "revision": version_row["revision"],
            "fps": float(version_row["fps"]),
            "size": int(version_row["size"]),
            "preprocess": str(version_row["preprocessing"]),
            "version": "dense_visual_v2",
        }
    else:
        rep_name = name or VISUAL_REP_NAMES[encoder_tag]
        params = {
            "aggregation": "window-weighted-mean-l2",
            "source": "visual_dense_windows",
            "visual_version": visual_version,
            "encoder": version_row["model_id"],
            "revision": version_row["revision"],
            "fps": float(version_row["fps"]),
            "size": int(version_row["size"]),
            "preprocess": str(version_row["preprocessing"]),
            "window_seconds": float(config.media.clip_seconds),
            "version": "dense_visual_v1",
        }
    existing = catalog.get_representation(name=rep_name, sequence_version=visual_version)
    if existing is not None:
        if not force:
            raise NebulaError(
                f"表征 {rep_name}@{visual_version} 已存在；重新构建请加 --force"
            )
        rep_id = int(existing["id"])
        catalog.update_representation_params(rep_id, params)
        catalog.clear_representation_vectors(rep_id)
        catalog.set_representation_status(rep_id, "building", note="dense visual 重建")
    else:
        rep_id = catalog.register_representation(
            rep_name,
            sequence_version=visual_version,
            kind="visual",
            source_kind="visual",
            params=params,
            note="密集视觉窗口向量加权平均（不重编码）",
        )
    catalog.set_representation_status(rep_id, "building")
    if window_plan_version:
        rows = [
            row
            for row in catalog.list_visual_window_sequences(
                frame_version=visual_version,
                window_plan_version=window_plan_version,
                status="completed",
            )
            if row.get("windows_path")
        ]
    else:
        rows = [
            row
            for row in catalog.list_visual_sequences(version=visual_version, status="completed")
            if row.get("windows_path")
        ]
    duplicates = catalog.get_duplicate_work_ids()
    rows = [row for row in rows if int(row["work_id"]) not in duplicates]
    rows.sort(key=lambda row: int(row["work_id"]))
    if limit is not None:
        rows = rows[: int(limit)]
    built = 0
    skipped = 0
    dims: set[int] = set()
    for row in rows:
        work_id = int(row["work_id"])
        windows_path = config.data_path(str(row["windows_path"]))
        if not windows_path.is_file():
            LOGGER.warning("work_id=%s 缺少窗口文件 %s，跳过", work_id, windows_path)
            skipped += 1
            continue
        try:
            windows = np.load(windows_path, allow_pickle=False)
            if window_plan_version:
                weights_path = row.get("window_weights_path")
                if not weights_path:
                    raise NebulaError("窗口记录缺少 window_weights_path")
                weights = _load_weight_file(
                    config.data_path(str(weights_path)), int(windows.shape[0])
                )
            else:
                weights = _manifest_weights(
                    config.paths.visual_dense / visual_version / f"{work_id:06d}",
                    int(windows.shape[0]),
                )
            vector = aggregate_window_vectors(windows, weights)
            representations.save_representation_vectors(
                catalog, config, {"id": rep_id, "name": rep_name, "sequence_version": visual_version},
                work_id, {"joint": vector},
            )
            dims.add(int(vector.size))
            built += 1
        except Exception as exc:  # noqa: BLE001 - 单部失败不中断整体
            LOGGER.error("work_id=%s 视觉表征构建失败: %s", work_id, exc)
            skipped += 1
    if built == 0:
        catalog.set_representation_status(rep_id, "failed", dim=0, note="没有可用视觉窗口")
        raise NebulaError(f"表征 {rep_name}@{visual_version} 没有构建出任何作品")
    dim = max(dims)
    catalog.set_representation_status(rep_id, "ready", dim=dim)
    summary = {
        "representation_id": rep_id,
        "name": rep_name,
        "sequence_version": visual_version,
        "n_built": built,
        "n_skipped": skipped,
        "dim": dim,
        "status": "ready",
    }
    LOGGER.info("密集视觉表征构建完成: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建 CLIP-only / DINOv2-only 密集视觉表征")
    parser.add_argument("--config", default=None, help="配置文件路径（默认 ./config.yaml）")
    parser.add_argument("--visual-version", default=None, help="视觉序列/帧编码版本；省略时按配置解析主版本")
    parser.add_argument(
        "--name",
        default=None,
        choices=sorted(set(VISUAL_REP_NAMES.values()) | set(VISUAL_REP_NAMES_V2.values())),
        help="表征名；省略时按编码器与模式自动选择（v1 或 v2）",
    )
    parser.add_argument("--all-main", action="store_true", help="为配置中的全部主帧版本各构建一个表征")
    parser.add_argument(
        "--window-plan",
        default=None,
        help="窗口计划版本（v0.4.3.1）；给出时从 visual_window_sequences 读取显式覆盖权重并登记 v2 表征",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="使用 config.visual_window_plan_version 构建覆盖权重 v2 表征（等价于 --window-plan <默认值> --all-main）",
    )
    parser.add_argument("--frame-version", default=None, help="v2 模式下的帧编码版本（默认全部主版本）")
    parser.add_argument("--force", action="store_true", help="清空并重建同版本表征")
    parser.add_argument("--limit", type=int, default=None, help="最多构建多少部（调试）")
    args = parser.parse_args(argv)
    config = load_config(args.config or Path.cwd() / "config.yaml")
    configure_logging(config.logging.level)
    catalog = Catalog(config.paths.db_path)
    catalog.initialize()
    try:
        window_plan = args.window_plan or (
            config.visual_window_plan_version if args.v2 else None
        )
        if window_plan:
            if args.frame_version:
                versions = [str(args.frame_version)]
            elif args.all_main or args.v2 or args.visual_version is None:
                versions = _resolve_main_versions(config)
            else:
                versions = [str(args.visual_version)]
            if args.name and args.name not in set(VISUAL_REP_NAMES_V2.values()):
                raise NebulaError("窗口计划模式请使用 v2 表征名（dense_visual_clip_v2 / dense_visual_dino_v2）")
        else:
            versions = (
                _resolve_main_versions(config)
                if args.all_main or args.visual_version is None
                else [str(args.visual_version)]
            )
        summaries = []
        for version in versions:
            summaries.append(
                build_representation(
                    catalog,
                    config,
                    visual_version=version,
                    name=args.name,
                    force=bool(args.force),
                    limit=args.limit,
                    window_plan_version=window_plan,
                )
            )
    finally:
        catalog.close()
    for summary in summaries:
        print(
            f"  {summary['name']}@{summary['sequence_version']}: "
            f"{summary['n_built']} 部，dim={summary['dim']}，跳过 {summary['n_skipped']}"
        )
    print(f"完成：{len(summaries)} 个密集视觉表征"
          + (f"（窗口计划 {window_plan}）" if window_plan else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
