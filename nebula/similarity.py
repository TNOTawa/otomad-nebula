"""高维余弦近邻索引与片段级相似探索。

职责：只读取 SQLite 中已经登记的向量文件（作品级 work_* 与片段级 clip_*），
在真实高维空间里计算余弦相似度、近邻解释信息与片段级对应，不重新编码、
不依赖 UMAP 平面坐标。

关键约束（见 docs/CONTRACTS_V2.md 第 2 节，以及本轮 P1/P2 需求）：

- 作品级矩阵在进程内缓存一次：构造时每种模态只读一遍 npy，之后重复查询不再读文件；
- 片段级矩阵按需加载并缓存（同一作品/模态只读一次）；
- 默认使用 canonical 集合（排除 work_links 中登记的 duplicate 实体）；
- 所有相似度都是高维余弦，禁止用二维/三维坐标距离代替。

解释性字段约定：

- pair_percentile：该相似度在全库所有无序作品对（不含自身）的余弦分布中
  的百分位，定义为“高于多少百分比的作品对”，即 100 * count(其他对 < 该值) / 对数；
- is_mutual：目标作品是否也把当前作品列入自己的 Top-k（同一个 k）；
- hubness：每部作品被多少部其他作品列入 Top-k。小样本/子集上该统计不稳定，
  只作为观察线索，不应当作“作品重要性”结论。

测试可通过 monkeypatch 模块级函数 _read_vector_file 注入计数 reader。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .db import Catalog
from .errors import NebulaError

logger = logging.getLogger(__name__)

#: 参与近邻计算的三种作品级模态（顺序固定，界面按此顺序展示）
KINDS: tuple[str, ...] = ("work_joint", "work_audio", "work_video")

#: 模态中文标签
KIND_LABELS: dict[str, str] = {
    "work_joint": "联合",
    "work_audio": "音频",
    "work_video": "画面",
}

#: 片段级模态（顺序固定）
CLIP_KINDS: tuple[str, ...] = ("clip_joint", "clip_audio", "clip_video")

#: 片段级模态中文标签
CLIP_KIND_LABELS: dict[str, str] = {
    "clip_joint": "联合片段",
    "clip_audio": "音频片段",
    "clip_video": "画面片段",
}

DEFAULT_KIND = "work_joint"
DEFAULT_CLIP_KIND = "clip_joint"

#: explain_pair 默认的片段相似度阈值
DEFAULT_CLIP_THRESHOLD = 0.5


@dataclass(frozen=True)
class Neighbor:
    """一部近邻作品及其解释信息。

    新增字段带默认值，保持旧调用方式（Neighbor(work_id, similarity, rank)）兼容。
    """

    work_id: int
    similarity: float
    rank: int
    #: 该相似度在全库作品对分布中的百分位（0~100，越高表示越稀少地接近）
    pair_percentile: float = 0.0
    #: 目标作品是否也把当前作品列入自己的 Top-k
    is_mutual: bool = False


def _read_vector_file(path: Path) -> np.ndarray:
    """读取一个 .npy 向量文件。

    单独封装成模块级函数，便于测试注入计数 reader（验证缓存不会重复读文件）。
    """

    return np.load(path, allow_pickle=False)


def _normalize_matrix(matrix: np.ndarray, *, where: str) -> np.ndarray:
    """逐行 L2 归一化；零向量/非有限值抛 NebulaError。"""

    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2 or array.size == 0:
        raise NebulaError(f"{where} 向量矩阵形状非法: {array.shape}")
    if not np.all(np.isfinite(array)):
        raise NebulaError(f"{where} 向量包含非有限值（NaN/Inf）")
    norms = np.linalg.norm(array.astype(np.float64), axis=1)
    zero_rows = np.where(norms <= 0)[0]
    if zero_rows.size:
        raise NebulaError(f"{where} 有 {zero_rows.size} 个零向量，无法计算余弦相似度")
    return (array / norms[:, None]).astype(np.float32)


class SimilarityIndex:
    """进程内高维余弦近邻索引 + 片段级相似探索。

    构造时从 Catalog 读取该 embedding version 的作品级向量（每种模态只读文件一次），
    默认排除 work_links 中的 duplicate 作品；片段级向量在首次使用时按需加载并缓存。
    对象不再使用时应调用 close() 释放矩阵。
    """

    def __init__(
        self,
        catalog: Catalog,
        config: Config,
        *,
        version: str | None = None,
    ) -> None:
        self._config = config
        self.version = str(version or config.embedding_version)
        self._closed = False

        duplicate_ids = catalog.get_duplicate_work_ids()
        kind_ids: dict[str, list[int]] = {kind: [] for kind in KINDS}
        kind_paths: dict[str, dict[int, str]] = {kind: {} for kind in KINDS}
        kind_dims: dict[str, dict[int, int]] = {kind: {} for kind in KINDS}
        clip_paths: dict[str, dict[int, str]] = {kind: {} for kind in CLIP_KINDS}
        clip_dims: dict[str, dict[int, int]] = {kind: {} for kind in CLIP_KINDS}
        clip_counts: dict[str, dict[int, int | None]] = {kind: {} for kind in CLIP_KINDS}
        for row in catalog.list_embeddings(self.version):
            kind = str(row.get("kind"))
            work_id = int(row["work_id"])
            if work_id in duplicate_ids:
                continue
            if kind in KINDS:
                kind_ids[kind].append(work_id)
                kind_paths[kind][work_id] = str(row["path"])
                kind_dims[kind][work_id] = int(row["dim"])
            elif kind in CLIP_KINDS:
                clip_paths[kind][work_id] = str(row["path"])
                clip_dims[kind][work_id] = int(row["dim"])
                raw_count = row.get("n_clips")
                clip_counts[kind][work_id] = None if raw_count is None else int(raw_count)

        self._kind_ids: dict[str, tuple[int, ...]] = {
            kind: tuple(sorted(work_ids)) for kind, work_ids in kind_ids.items() if work_ids
        }
        self._kind_paths = kind_paths
        self._kind_dims = kind_dims
        self._kind_pos: dict[str, dict[int, int]] = {
            kind: {work_id: index for index, work_id in enumerate(work_ids)}
            for kind, work_ids in self._kind_ids.items()
        }
        self._work_ids: tuple[int, ...] = tuple(
            sorted({work_id for work_ids in self._kind_ids.values() for work_id in work_ids})
        )
        self._pos: dict[int, int] = {
            work_id: index for index, work_id in enumerate(self._work_ids)
        }

        #: 片段级向量路径/登记维度/登记片段数（canonical 作品）
        self._clip_paths = clip_paths
        self._clip_dims = clip_dims
        self._clip_counts = clip_counts
        #: work_id -> 片段元数据（来自 clips 表：idx/start_s/duration_s）
        self._clip_times: dict[int, list[dict[str, Any]]] = {
            work_id: catalog.get_clips(work_id, version=self.version)
            for work_id in self._work_ids
        }

        self._matrices: dict[str, np.ndarray] = {}
        for kind in list(self._kind_ids):
            self._matrices[kind] = self._load_kind(kind)

        #: 派生缓存：全库相似度矩阵、成对相似度排序数组、Top-k 集合、片段矩阵
        self._full_similarity_cache: dict[str, np.ndarray] = {}
        self._pair_values_cache: dict[str, np.ndarray] = {}
        self._topk_cache: dict[tuple[str, int], dict[int, frozenset[int]]] = {}
        self._clip_matrix_cache: dict[tuple[str, int], np.ndarray] = {}

        logger.info(
            "近邻索引就绪: version=%s works=%d kinds=%s clip_kinds=%s",
            self.version,
            len(self._work_ids),
            ",".join(self.available_kinds),
            ",".join(self.available_clip_kinds),
        )

    # ------------------------------------------------------------------ 内部
    def _ensure_open(self) -> None:
        if self._closed:
            raise NebulaError("SimilarityIndex 已关闭，不能继续查询")

    def _load_kind(self, kind: str) -> np.ndarray:
        work_ids = self._kind_ids[kind]
        vectors: list[np.ndarray] = []
        dims: set[int] = set()
        for work_id in work_ids:
            relpath = self._kind_paths[kind][work_id]
            path = self._config.data_path(relpath)
            if not path.is_file():
                raise NebulaError(
                    f"作品 {work_id} 的 {kind} 向量文件不存在: {path}；"
                    "请重新运行 pipeline 生成向量"
                )
            try:
                raw = _read_vector_file(path)
            except (OSError, ValueError) as exc:
                raise NebulaError(f"读取作品 {work_id} 的 {kind} 向量失败: {path}: {exc}") from exc
            array = np.asarray(raw, dtype=np.float32)
            if array.ndim == 2 and array.shape[0] == 1:
                array = array.reshape(-1)
            if array.ndim != 1 or array.size == 0:
                raise NebulaError(f"作品 {work_id} 的 {kind} 向量形状非法: {array.shape}")
            registered_dim = int(self._kind_dims[kind][work_id])
            if registered_dim != int(array.shape[0]):
                raise NebulaError(
                    f"作品 {work_id} 的 {kind} 向量维度与登记不一致: "
                    f"文件 {array.shape[0]} 维，登记 {registered_dim} 维"
                )
            dims.add(int(array.shape[0]))
            vectors.append(array)
        if len(dims) > 1:
            raise NebulaError(f"{kind} 向量维度不一致: {sorted(dims)}，请确认使用同一个向量版本")
        return _normalize_matrix(np.vstack(vectors), where=kind)

    def _kind_data(self, kind: str) -> tuple[np.ndarray, tuple[int, ...]]:
        self._ensure_open()
        if kind not in KINDS:
            raise NebulaError(f"未知模态: {kind}（可选: {', '.join(KINDS)}）")
        if kind not in self._matrices:
            raise NebulaError(f"向量版本 {self.version} 没有 {kind} 向量数据")
        return self._matrices[kind], self._kind_ids[kind]

    def _full_similarity(self, kind: str) -> np.ndarray:
        """全库 n×n 余弦相似度矩阵（float64，缓存）。"""

        matrix, _ = self._kind_data(kind)
        if kind not in self._full_similarity_cache:
            values = matrix.astype(np.float64) @ matrix.astype(np.float64).T
            self._full_similarity_cache[kind] = np.clip(values, -1.0, 1.0)
        return self._full_similarity_cache[kind]

    def _pair_values(self, kind: str) -> np.ndarray:
        """全库无序作品对（不含自身）的相似度排序数组（缓存）。"""

        if kind not in self._pair_values_cache:
            similarities = self._full_similarity(kind)
            count = similarities.shape[0]
            if count < 2:
                self._pair_values_cache[kind] = np.empty(0, dtype=np.float64)
            else:
                upper = similarities[np.triu_indices(count, k=1)]
                self._pair_values_cache[kind] = np.sort(upper.astype(np.float64))
        return self._pair_values_cache[kind]

    def _topk_sets(self, kind: str, k: int) -> dict[int, frozenset[int]]:
        """每部作品 Top-k 近邻集合（不含自身，缓存）。"""

        cache_key = (kind, int(k))
        if cache_key not in self._topk_cache:
            similarities = self._full_similarity(kind)
            work_ids = self._kind_ids[kind]
            count = len(work_ids)
            take = min(int(k), count - 1)
            sets: dict[int, frozenset[int]] = {}
            for position, work_id in enumerate(work_ids):
                if take <= 0:
                    sets[int(work_id)] = frozenset()
                    continue
                row = similarities[position].copy()
                row[position] = -np.inf
                order = np.argsort(-row, kind="stable")[:take]
                sets[int(work_id)] = frozenset(int(work_ids[int(index)]) for index in order)
            self._topk_cache[cache_key] = sets
        return self._topk_cache[cache_key]

    def _require_k(self, k: int) -> int:
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise NebulaError(f"k 必须是 >= 1 的整数，收到 {k!r}")
        return int(k)

    # ------------------------------------------------------------------ 公开 API：作品级
    @property
    def work_ids(self) -> tuple[int, ...]:
        """canonical 作品 id（按 work_id 排序；包含至少有一种模态向量的作品）。"""

        return self._work_ids

    @property
    def available_kinds(self) -> tuple[str, ...]:
        """有向量数据的作品级模态，顺序与 KINDS 一致。"""

        return tuple(kind for kind in KINDS if kind in self._matrices)

    @property
    def available_clip_kinds(self) -> tuple[str, ...]:
        """有片段级向量数据的模态，顺序与 CLIP_KINDS 一致。"""

        return tuple(kind for kind in CLIP_KINDS if self._clip_paths.get(kind))

    def has_vector(self, work_id: int, kind: str = DEFAULT_KIND) -> bool:
        """判断某部作品是否有指定作品级模态的向量。"""

        if kind not in self._kind_pos:
            return False
        return int(work_id) in self._kind_pos[kind]

    def has_clip_vector(self, work_id: int, kind: str = DEFAULT_CLIP_KIND) -> bool:
        """判断某部作品是否有指定片段级模态的向量。"""

        return int(work_id) in self._clip_paths.get(kind, {})

    def matrix(self, kind: str = DEFAULT_KIND) -> np.ndarray:
        """返回指定模态的 (n, dim) float32 矩阵（每行 L2 归一化，缓存）。"""

        matrix, _ = self._kind_data(kind)
        return matrix

    def index_of(self, work_id: int) -> int:
        """返回 work_id 在 canonical 作品序列中的下标；不存在抛 NebulaError。"""

        self._ensure_open()
        try:
            return self._pos[int(work_id)]
        except (KeyError, ValueError) as exc:
            raise NebulaError(f"作品 {work_id} 不在近邻索引中（没有可用向量或被排除）") from exc

    def pair_percentile(
        self,
        a_work_id: int,
        b_work_id: int,
        *,
        kind: str = DEFAULT_KIND,
    ) -> float:
        """该作品对的相似度在全库作品对分布中的百分位（0~100）。

        定义为“高于多少百分比的作品对”：100 * count(其他无序对 < 本对相似度) / 总对数。
        """

        similarity = self.similarity(a_work_id, b_work_id, kind=kind)
        pairs = self._pair_values(kind)
        if pairs.size == 0:
            return 0.0
        below = int(np.searchsorted(pairs, similarity, side="left"))
        return float(100.0 * below / pairs.size)

    def neighbors(
        self,
        work_id: int,
        *,
        kind: str = DEFAULT_KIND,
        k: int = 5,
    ) -> list[Neighbor]:
        """返回指定模态下最相似的 k 部作品（不含自身，按相似度降序）。

        每条结果附带全库成对分位 pair_percentile 与互为近邻标记 is_mutual。
        作品不足 k 部时返回尽量多的条数；作品在该模态没有向量时抛 NebulaError。
        """

        take_k = self._require_k(k)
        _, work_ids = self._kind_data(kind)
        position = self._kind_pos[kind].get(int(work_id))
        if position is None:
            raise NebulaError(f"作品 {int(work_id)} 没有 {kind} 向量，无法计算近邻")
        similarities = self._full_similarity(kind)[position].copy()
        similarities[position] = -np.inf
        take = min(take_k, len(work_ids) - 1)
        if take <= 0:
            return []
        order = np.argsort(-similarities, kind="stable")[:take]
        topk_sets = self._topk_sets(kind, take_k)
        pair_values = self._pair_values(kind)
        current = int(work_id)
        neighbors: list[Neighbor] = []
        for rank, index in enumerate(order, start=1):
            target = int(work_ids[int(index)])
            similarity = float(np.clip(similarities[int(index)], -1.0, 1.0))
            if pair_values.size:
                below = int(np.searchsorted(pair_values, similarity, side="left"))
                percentile = float(100.0 * below / pair_values.size)
            else:
                percentile = 0.0
            neighbor_set = topk_sets.get(target, frozenset())
            neighbors.append(
                Neighbor(
                    work_id=target,
                    similarity=similarity,
                    rank=rank,
                    pair_percentile=percentile,
                    is_mutual=current in neighbor_set,
                )
            )
        return neighbors

    def similarity(
        self,
        a_work_id: int,
        b_work_id: int,
        *,
        kind: str = DEFAULT_KIND,
    ) -> float:
        """两部作品在指定模态下的真实高维余弦相似度。"""

        first = self._kind_pos[kind].get(int(a_work_id))
        second = self._kind_pos[kind].get(int(b_work_id))
        self._kind_data(kind)
        if first is None:
            raise NebulaError(f"作品 {a_work_id} 没有 {kind} 向量")
        if second is None:
            raise NebulaError(f"作品 {b_work_id} 没有 {kind} 向量")
        if first == second:
            return 1.0
        return float(self._full_similarity(kind)[first, second])

    def hubness(
        self,
        k: int = 5,
        kind: str = DEFAULT_KIND,
    ) -> dict[int, int]:
        """每部作品被其他作品列入 Top-k 的次数（hubness）。

        注意：小样本或筛选子集上该统计不稳定，只用于观察“哪些作品常被当作近邻”，
        不代表作品质量或重要性。
        """

        take_k = self._require_k(k)
        self._kind_data(kind)
        topk_sets = self._topk_sets(kind, take_k)
        counts: dict[int, int] = {int(work_id): 0 for work_id in self._kind_ids[kind]}
        for targets in topk_sets.values():
            for target in targets:
                counts[target] = counts.get(target, 0) + 1
        return counts

    def hub_top(
        self,
        k: int = 5,
        n: int = 5,
        kind: str = DEFAULT_KIND,
    ) -> list[tuple[int, int]]:
        """返回被最多作品列为 Top-k 的作品列表 [(work_id, 被列次数)]。

        按次数降序、work_id 升序排序；n 限制返回条数。
        """

        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise NebulaError(f"n 必须是 >= 1 的整数，收到 {n!r}")
        counts = self.hubness(k=k, kind=kind)
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [(int(work_id), int(count)) for work_id, count in ranked[: int(n)]]

    def kth_neighbor_distance(
        self,
        *,
        k: int = 5,
        kind: str = DEFAULT_KIND,
    ) -> dict[int, float]:
        """每部作品的第 k 近邻余弦距离（1 - 相似度；越大表示局部越稀疏）。

        用于 A2 的高维近邻密度染色。作品数不足 k+1 时用最后一个可用近邻；
        只有一个作品（没有近邻）时距离记为 1.0。
        """

        take_k = self._require_k(k)
        _, work_ids = self._kind_data(kind)
        count = len(work_ids)
        if count <= 1:
            return {int(work_ids[0]): 1.0} if count == 1 else {}
        take = min(take_k, count - 1)
        similarities = self._full_similarity(kind)
        distances: dict[int, float] = {}
        for index, work_id in enumerate(work_ids):
            row = similarities[index].copy()
            row[index] = -np.inf
            kth_similarity = float(np.sort(row)[::-1][take - 1])
            distances[int(work_id)] = float(1.0 - np.clip(kth_similarity, -1.0, 1.0))
        return distances

    # ------------------------------------------------------------------ 公开 API：片段级
    def _load_clip_matrix(self, work_id: int, kind: str) -> np.ndarray:
        if kind not in CLIP_KINDS:
            raise NebulaError(f"未知片段模态: {kind}（可选: {', '.join(CLIP_KINDS)}）")
        self._ensure_open()
        work_key = int(work_id)
        cache_key = (kind, work_key)
        if cache_key in self._clip_matrix_cache:
            return self._clip_matrix_cache[cache_key]
        relpath = self._clip_paths.get(kind, {}).get(work_key)
        if relpath is None:
            raise NebulaError(f"作品 {work_key} 没有 {kind} 片段向量")
        path = self._config.data_path(relpath)
        if not path.is_file():
            raise NebulaError(f"作品 {work_key} 的 {kind} 片段向量文件不存在: {path}")
        try:
            raw = _read_vector_file(path)
        except (OSError, ValueError) as exc:
            raise NebulaError(f"读取作品 {work_key} 的 {kind} 片段向量失败: {path}: {exc}") from exc
        array = np.asarray(raw, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.size == 0:
            raise NebulaError(f"作品 {work_key} 的 {kind} 片段向量形状非法: {array.shape}")
        registered_dim = int(self._clip_dims[kind][work_key])
        if registered_dim != int(array.shape[1]):
            raise NebulaError(
                f"作品 {work_key} 的 {kind} 片段维度与登记不一致: "
                f"文件 {array.shape[1]} 维，登记 {registered_dim} 维"
            )
        registered_count = self._clip_counts[kind][work_key]
        if registered_count is not None and registered_count != int(array.shape[0]):
            raise NebulaError(
                f"作品 {work_key} 的 {kind} 片段数量与登记不一致: "
                f"文件 {array.shape[0]} 段，登记 {registered_count} 段"
            )
        matrix = _normalize_matrix(array, where=f"{work_key}/{kind}")
        self._clip_matrix_cache[cache_key] = matrix
        return matrix

    def clip_matrix(self, work_id: int, kind: str = DEFAULT_CLIP_KIND) -> np.ndarray:
        """返回某部作品的片段矩阵 (n_clips, dim) float32（逐行 L2 归一化，缓存）。"""

        matrix = self._load_clip_matrix(work_id, kind)
        return matrix

    def clip_times(self, work_id: int) -> list[dict[str, Any]]:
        """返回某部作品的片段元数据（idx/start_s/duration_s，来自 clips 表）。"""

        self._ensure_open()
        work_key = int(work_id)
        if work_key not in self._pos:
            raise NebulaError(f"作品 {work_key} 不在近邻索引中（没有可用向量或被排除）")
        return [dict(item) for item in self._clip_times.get(work_key, [])]

    def clip_similarity(
        self,
        a_work_id: int,
        b_work_id: int,
        kind: str = DEFAULT_CLIP_KIND,
    ) -> np.ndarray:
        """两部作品片段之间的余弦相似度矩阵 (n_a, n_b)。"""

        first = self.clip_matrix(a_work_id, kind)
        second = self.clip_matrix(b_work_id, kind)
        values = first.astype(np.float64) @ second.astype(np.float64).T
        return np.clip(values, -1.0, 1.0)

    def best_clip_pairs(
        self,
        a_work_id: int,
        b_work_id: int,
        kind: str = DEFAULT_CLIP_KIND,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """返回两部作品最相似的 k 对片段（a_idx/b_idx/similarity，按相似度降序）。"""

        take_k = self._require_k(k)
        similarities = self.clip_similarity(a_work_id, b_work_id, kind=kind)
        flat_order = np.argsort(-similarities.reshape(-1), kind="stable")[:take_k]
        columns = similarities.shape[1]
        pairs: list[dict[str, Any]] = []
        for flat_index in flat_order:
            a_index, b_index = divmod(int(flat_index), columns)
            pairs.append(
                {
                    "a_idx": int(a_index),
                    "b_idx": int(b_index),
                    "similarity": float(similarities[a_index, b_index]),
                }
            )
        return pairs

    def explain_pair(
        self,
        a_work_id: int,
        b_work_id: int,
        kind: str = DEFAULT_CLIP_KIND,
        threshold: float = DEFAULT_CLIP_THRESHOLD,
    ) -> dict[str, Any]:
        """解释一对作品的片段级对应关系。

        返回 max（最高片段相似度）、mean（全部片段对平均相似度）、
        count_above_threshold（超过阈值的片段对数量）等。
        """

        if not 0.0 <= float(threshold) <= 1.0:
            raise NebulaError(f"threshold 必须在 0..1，收到 {threshold!r}")
        similarities = self.clip_similarity(a_work_id, b_work_id, kind=kind)
        return {
            "kind": kind,
            "a_work_id": int(a_work_id),
            "b_work_id": int(b_work_id),
            "n_a": int(similarities.shape[0]),
            "n_b": int(similarities.shape[1]),
            "max": float(np.max(similarities)),
            "mean": float(np.mean(similarities)),
            "count_above_threshold": int(
                np.count_nonzero(similarities >= float(threshold))
            ),
            "threshold": float(threshold),
        }

    def close(self) -> None:
        """释放缓存的矩阵；之后继续查询会抛 NebulaError。"""

        self._matrices.clear()
        self._full_similarity_cache.clear()
        self._pair_values_cache.clear()
        self._topk_cache.clear()
        self._clip_matrix_cache.clear()
        self._closed = True


def load_similarity_index(
    catalog: Catalog,
    config: Config,
    *,
    version: str | None = None,
) -> SimilarityIndex:
    """构造 SimilarityIndex（便捷函数，app 用 st.cache_resource 包一层）。"""

    return SimilarityIndex(catalog, config, version=version)
