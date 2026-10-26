"""表征层：聚合器版本、向量登记与加载。

职责边界（见 docs/plans/v0.4-full-coverage-design.md 第 7 节）：

- sequence layer（`data/embeddings/<sequence_version>/`）保存 PE-AV 片段序列，
  编码一次后长期复用；
- representation layer 保存不同聚合器生成的最终作品向量：聚合算法变化只新增
  表征，绝不重新下载/解码/编码 PE-AV；
- legacy 表征不产生新文件，作品向量来自 `embeddings` 表的 work_* 文件；
- 派生表征（full_mean / temporal_stats / transformer）的向量写入
  `data/representations/<name>/<sequence_version>/<work_id>/{joint,audio,video}.npy`。

所有读取都排除 work_links 中的 duplicate 作品；矩阵按 work_id 排序，保证同一
表征在不同进程中得到完全相同的行顺序。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import Config
from .db import LEGACY_REPRESENTATION_NAME, Catalog
from .embed import load_vector, save_vector
from .errors import NebulaError

logger = logging.getLogger(__name__)

#: 旧版“均匀抽 ≤5 段再平均”表征名（与 db.LEGACY_REPRESENTATION_NAME 相同）
LEGACY_NAME = LEGACY_REPRESENTATION_NAME
FULL_MEAN_V1 = "full_mean_v1"
TEMPORAL_STATS_V1 = "temporal_stats_v1"
TRANSFORMER_V1 = "transformer_v1"
#: 密集视觉语义表征（v0.4.3）：CLIP-only / DINOv2-only 对照空间（等权窗口平均，历史）
DENSE_VISUAL_CLIP_V1 = "dense_visual_clip_v1"
DENSE_VISUAL_DINO_V1 = "dense_visual_dino_v1"
#: v0.4.3.1 修复版：窗口覆盖权重（coverage-v1）+ 可独立演进的窗口计划
DENSE_VISUAL_CLIP_V2 = "dense_visual_clip_v2"
DENSE_VISUAL_DINO_V2 = "dense_visual_dino_v2"

#: 界面显示名
REPRESENTATION_LABELS: dict[str, str] = {
    LEGACY_NAME: "Legacy（旧 5 片段平均）",
    FULL_MEAN_V1: "Full Mean（全片平均）",
    TEMPORAL_STATS_V1: "Temporal Stats（时间结构）",
    TRANSFORMER_V1: "Temporal Transformer（v0.5，未开放）",
    DENSE_VISUAL_CLIP_V1: "Dense Visual·CLIP（密集视觉语义）",
    DENSE_VISUAL_DINO_V1: "Dense Visual·DINOv2（密集视觉结构）",
    DENSE_VISUAL_CLIP_V2: "Dense Visual·CLIP v2（覆盖权重）",
    DENSE_VISUAL_DINO_V2: "Dense Visual·DINOv2 v2（覆盖权重）",
}

#: CLI / 界面上的别名
_NAME_ALIASES: dict[str, str] = {
    "legacy": LEGACY_NAME,
    "legacy_5clip_mean": LEGACY_NAME,
    "full_mean": FULL_MEAN_V1,
    "full-mean": FULL_MEAN_V1,
    "full_mean_v1": FULL_MEAN_V1,
    "temporal": TEMPORAL_STATS_V1,
    "temporal_stats": TEMPORAL_STATS_V1,
    "temporal-stats": TEMPORAL_STATS_V1,
    "temporal_stats_v1": TEMPORAL_STATS_V1,
    "transformer": TRANSFORMER_V1,
    "transformer_v1": TRANSFORMER_V1,
    "dense_visual_clip": DENSE_VISUAL_CLIP_V1,
    "dense_visual_clip_v1": DENSE_VISUAL_CLIP_V1,
    "clip_visual": DENSE_VISUAL_CLIP_V1,
    "dense_visual_dino": DENSE_VISUAL_DINO_V1,
    "dense_visual_dino_v1": DENSE_VISUAL_DINO_V1,
    "dino_visual": DENSE_VISUAL_DINO_V1,
    "dense_visual_clip_v2": DENSE_VISUAL_CLIP_V2,
    "dense_visual_dino_v2": DENSE_VISUAL_DINO_V2,
    "clip_visual_v2": DENSE_VISUAL_CLIP_V2,
    "dino_visual_v2": DENSE_VISUAL_DINO_V2,
}

#: 表征向量支持的模态（与 DB representation_vectors.kind 一致）
VECTOR_KINDS: tuple[str, ...] = ("joint", "audio", "video")


def normalize_representation_name(name: str) -> str:
    """把用户输入的表征名/别名规范化为注册名。"""

    raw = str(name or "").strip()
    if not raw:
        raise NebulaError("表征名不能为空")
    return _NAME_ALIASES.get(raw, raw)


def representation_label(name: str) -> str:
    """返回界面显示名（未知名称原样返回）。"""

    return REPRESENTATION_LABELS.get(name, name)


@dataclass(frozen=True)
class ClipTime:
    """一个片段时间窗口（覆盖完整时间轴，含有效时长与权重）。"""

    index: int
    start_s: float
    duration_s: float
    valid_duration_s: float
    weight: float | None = None

    @property
    def end_s(self) -> float:
        return round(self.start_s + self.duration_s, 6)


@dataclass(frozen=True)
class ClipSequence:
    """一部作品的有序片段序列 + 事件特征（序列层的只读视图）。"""

    work_id: int
    version: str
    duration_s: float | None
    times: tuple[ClipTime, ...]
    joint: np.ndarray | None
    audio: np.ndarray | None
    video: np.ndarray | None
    event_features: np.ndarray | None = None
    event_version: str | None = None

    @property
    def n_clips(self) -> int:
        return len(self.times)

    def clip_array(self, kind: str) -> np.ndarray | None:
        if kind == "joint":
            return self.joint
        if kind == "audio":
            return self.audio
        if kind == "video":
            return self.video
        raise NebulaError(f"未知片段模态: {kind}")

    def weights(self) -> np.ndarray:
        """片段权重（缺失时退化为等权，保持旧序列也能被聚合）。"""

        if not self.times:
            return np.zeros((0,), dtype=np.float32)
        raw = np.asarray(
            [1.0 if t.weight is None else float(t.weight) for t in self.times],
            dtype=np.float32,
        )
        total = float(raw.sum())
        if not np.isfinite(total) or total <= 0:
            return np.full((len(self.times),), 1.0 / len(self.times), dtype=np.float32)
        return raw / total


# --------------------------------------------------------------------------- 解析
def resolve_representation(
    catalog: Catalog,
    config: Config,
    *,
    name: str | None = None,
    sequence_version: str | None = None,
    representation_id: int | None = None,
) -> dict[str, Any]:
    """解析要使用的表征行。

    - 显式 representation_id：直接取；
    - name=None：按当前 sampling.mode 选择 Full Mean 或 Legacy；
    - legacy：按 sequence_version（缺省为 config.legacy_embedding_version）自动补登；
    - 其它 name：默认取最新一条 ready 记录；未构建时给出可执行的提示。
    """

    if representation_id is not None:
        row = catalog.get_representation(int(representation_id))
        if row is None:
            raise NebulaError(f"表征不存在: {representation_id}")
        return row

    if name is None:
        if config.sampling.mode == "full_coverage":
            name = FULL_MEAN_V1
            if sequence_version is None:
                sequence_version = config.full_sequence_version
        else:
            name = LEGACY_NAME

    resolved = normalize_representation_name(str(name))
    if resolved == LEGACY_NAME:
        version = sequence_version or config.legacy_embedding_version
        rep_id = catalog.ensure_legacy_representation(version)
        row = catalog.get_representation(rep_id)
        assert row is not None
        return row

    row = catalog.get_representation(name=resolved, sequence_version=sequence_version)
    if row is None:
        available = [
            r["name"]
            for r in catalog.list_representations(status="ready")
        ]
        hint = (
            f"请先运行 tools/build_representation.py {resolved} --sequence <sequence_version> "
            "生成该表征"
        )
        raise NebulaError(
            f"表征 {resolved} 尚未构建（可用: {', '.join(sorted(set(available))) or '无'}）。{hint}"
        )
    return row


def list_available(catalog: Catalog, config: Config) -> list[dict[str, Any]]:
    """列出可用于界面切换的表征（每个名字取最新一条 ready，附 ready 地图数）。"""

    legacy_version = config.legacy_embedding_version
    if catalog.get_embedding_version(legacy_version) is not None:
        try:
            catalog.ensure_legacy_representation(legacy_version)
        except Exception:  # noqa: BLE001 - 仅用于界面提示，不阻断
            logger.debug("补登 legacy 表征失败", exc_info=True)

    rows = catalog.list_representations(status="ready")
    latest_by_name: dict[str, dict[str, Any]] = {}
    for row in reversed(rows):
        latest_by_name.setdefault(str(row["name"]), row)

    result: list[dict[str, Any]] = []
    for row in rows:
        name = str(row["name"])
        if latest_by_name.get(name) is not row:
            continue
        enriched = dict(row)
        enriched["label"] = representation_label(name)
        try:
            maps = catalog.list_maps(representation_id=int(row["id"]))
        except Exception:  # noqa: BLE001 - 地图查询失败不影响界面
            maps = []
        enriched["maps"] = sum(1 for entry in maps if entry.get("status") == "ready")
        result.append(enriched)
    return result


# --------------------------------------------------------------------------- 路径
def representation_dir(config: Config, rep: Mapping[str, Any]) -> Path:
    """派生表征向量的根目录：data/representations/<name>/<sequence_version>/。"""

    return (
        config.paths.representations
        / str(rep["name"])
        / str(rep["sequence_version"])
    )


def sequence_work_dir(config: Config, sequence_version: str, work_id: int) -> Path:
    """序列层某部作品的目录（片段 npy、事件文件、manifest 所在处）。"""

    return config.paths.embeddings / str(sequence_version) / f"{int(work_id):06d}"


def save_representation_vectors(
    catalog: Catalog,
    config: Config,
    rep: Mapping[str, Any],
    work_id: int,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """原子写入并登记一部作品在某个表征下的向量（joint/audio/video 子集）。

    返回 {kind: {"path": data 相对路径, "dim": int}}。
    """

    if not arrays:
        raise NebulaError("表征向量为空")
    work_dir = representation_dir(config, rep) / f"{int(work_id):06d}"
    work_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, dict[str, Any]] = {}
    for kind, array in arrays.items():
        if kind not in VECTOR_KINDS:
            raise NebulaError(f"未知表征向量种类: {kind}")
        vector = np.asarray(array, dtype=np.float32)
        if vector.ndim != 1 or vector.size < 1:
            raise NebulaError(f"{kind} 表征向量必须是一维非空数组，收到 {vector.shape}")
        if not np.all(np.isfinite(vector)):
            raise NebulaError(f"{kind} 表征向量包含 NaN/Inf")
        path = work_dir / f"{kind}.npy"
        save_vector(path, vector)
        relpath = config.data_relpath(path)
        catalog.register_representation_vector(
            int(rep["id"]),
            int(work_id),
            kind=kind,
            path=relpath,
            dim=int(vector.size),
        )
        saved[kind] = {"path": relpath, "dim": int(vector.size)}
    return saved


# --------------------------------------------------------------------------- 加载
def _row_work_id(row: Mapping[str, Any]) -> int:
    """兼容两种查询结果：embeddings 联合查询用 works.id，纯向量查询用 work_id。"""

    if "work_id" in row:
        return int(row["work_id"])
    return int(row["id"])


def _canonical_rows(catalog: Catalog, rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    duplicates = catalog.get_duplicate_work_ids()
    return [row for row in rows if _row_work_id(row) not in duplicates]


def load_representation_matrix(
    catalog: Catalog,
    config: Config,
    rep: Mapping[str, Any],
    *,
    kind: str = "joint",
) -> tuple[tuple[int, ...], np.ndarray]:
    """读取某表征的作品级矩阵（canonical、按 work_id 排序）。"""

    vectors = load_representation_vectors(catalog, config, rep, kind)
    if not vectors:
        raise NebulaError(
            f"表征 {rep['name']}@{rep['sequence_version']} 没有 {kind} 向量"
        )
    ids = tuple(sorted(vectors))
    matrix = np.stack([vectors[work_id] for work_id in ids]).astype(np.float32, copy=False)
    dims = matrix.shape[1]
    if any(v.size != dims for v in vectors.values()):
        raise NebulaError(f"表征 {rep['name']} 的 {kind} 向量维度不一致")
    return ids, matrix


def load_representation_vectors(
    catalog: Catalog,
    config: Config,
    rep: Mapping[str, Any],
    kind: str = "joint",
) -> dict[int, np.ndarray]:
    """读取某表征的作品级向量，返回 {work_id: 一维 float32 向量}。"""

    if kind not in VECTOR_KINDS:
        raise NebulaError(f"未知表征向量种类: {kind}")
    result: dict[int, np.ndarray] = {}
    if str(rep["kind"]) == "legacy":
        embedding_kind = f"work_{kind}"
        rows = catalog.list_works_with_embedding(
            str(rep["sequence_version"]), kind=embedding_kind
        )
        for row in _canonical_rows(catalog, rows):
            work_id = int(row["id"])
            vector = load_vector(config.data_path(str(row["embedding_path"])))
            if vector.ndim != 1:
                raise NebulaError(f"作品 {work_id} 的 {embedding_kind} 不是一维向量")
            result[work_id] = vector.astype(np.float32, copy=False)
        return result

    rep_id = int(rep["id"])
    rows = catalog.list_works_with_representation(rep_id, kind=kind)
    for row in _canonical_rows(catalog, rows):
        work_id = int(row["id"])
        vector = load_vector(config.data_path(str(row["representation_path"])))
        if vector.ndim != 1:
            raise NebulaError(f"作品 {work_id} 的 {kind} 表征向量不是一维")
        result[work_id] = vector.astype(np.float32, copy=False)
    return result


def load_clip_sequence(
    catalog: Catalog,
    config: Config,
    sequence_version: str,
    work_id: int,
) -> ClipSequence:
    """读取一部作品在某个序列版本下的有序片段向量、时间与事件特征。"""

    work = catalog.get_work(int(work_id))
    if work is None:
        raise NebulaError(f"作品不存在: {work_id}")

    def _clip(kind: str) -> np.ndarray | None:
        row = catalog.get_embedding(int(work_id), version=sequence_version, kind=kind)
        if row is None:
            return None
        array = load_vector(config.data_path(str(row["path"])))
        if array.ndim != 2:
            raise NebulaError(f"片段向量 {kind} 必须是二维矩阵，收到 {array.shape}")
        return array.astype(np.float32, copy=False)

    joint = _clip("clip_joint")
    audio = _clip("clip_audio")
    video = _clip("clip_video")
    clip_rows = catalog.get_clips(int(work_id), version=sequence_version)
    if joint is not None and len(clip_rows) != joint.shape[0]:
        raise NebulaError(
            f"片段记录数 {len(clip_rows)} 与 clip_joint 行数 {joint.shape[0]} 不一致"
        )
    times = tuple(
        ClipTime(
            index=int(row["idx"]),
            start_s=float(row["start_s"]),
            duration_s=float(row["duration_s"]),
            valid_duration_s=(
                float(row["valid_duration_s"])
                if row.get("valid_duration_s") is not None
                else float(row["duration_s"])
            ),
            weight=(
                float(row["weight"]) if row.get("weight") is not None else None
            ),
        )
        for row in clip_rows
    )

    sequence_dir = sequence_work_dir(config, sequence_version, int(work_id))
    event_features: np.ndarray | None = None
    event_version: str | None = None
    feature_path = sequence_dir / "event_features.npy"
    if feature_path.is_file() and joint is not None and feature_path.stat().st_size > 0:
        array = np.load(feature_path, allow_pickle=False)
        event_features = np.asarray(array, dtype=np.float32)
        if event_features.ndim == 2 and event_features.shape[0] != joint.shape[0]:
                logger.warning(
                    "work_id=%s 事件特征行数 %s 与片段数 %s 不一致，忽略事件特征",
                    work_id,
                    event_features.shape[0],
                    joint.shape[0],
                )
                event_features = None
    manifest_path = sequence_dir / "event_manifest.json"
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            event_version = str(payload.get("version") or "") or None
        except (OSError, json.JSONDecodeError):  # pragma: no cover - 损坏文件只影响事件
            logger.warning("读取事件 manifest 失败: %s", manifest_path, exc_info=True)

    duration_s = work.get("duration_s")
    if duration_s is None:
        state = catalog.get_sequence_state(int(work_id), version=sequence_version)
        if state is not None:
            duration_s = state.get("duration_s")
    return ClipSequence(
        work_id=int(work_id),
        version=str(sequence_version),
        duration_s=None if duration_s is None else float(duration_s),
        times=times,
        joint=joint,
        audio=audio,
        video=video,
        event_features=event_features,
        event_version=event_version,
    )


def clip_times_array(times: Sequence[ClipTime]) -> np.ndarray:
    """把片段窗口压成 clip_times.npy 的 (T,4) 矩阵。

    列顺序：start_s, duration_s, valid_duration_s, weight（weight 缺失时按等权补齐）。
    """

    if not times:
        return np.zeros((0, 4), dtype=np.float32)
    fallback = 1.0 / len(times)
    rows = [
        [
            float(t.start_s),
            float(t.duration_s),
            float(t.valid_duration_s),
            float(fallback if t.weight is None else t.weight),
        ]
        for t in times
    ]
    return np.asarray(rows, dtype=np.float32)
