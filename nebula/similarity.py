"""高维余弦近邻索引（迭代 A1）。

职责：只读取 SQLite 中已经登记的 work_joint / work_audio / work_video 向量文件，
在真实高维空间里计算余弦相似度与近邻，不重新编码、不依赖 UMAP 平面坐标。

关键约束（见 docs/CONTRACTS_V2.md 第 2 节）：

- 矩阵在进程内缓存一次：构造时每种模态只读一遍 npy，之后重复查询不再读文件；
- 默认使用 canonical 集合（排除 work_links 中登记的 duplicate 实体）；
- 所有相似度都是高维余弦，禁止用二维/三维坐标距离代替。

测试可通过 monkeypatch 模块级函数 _read_vector_file 注入计数 reader。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

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

DEFAULT_KIND = "work_joint"


@dataclass(frozen=True)
class Neighbor:
    """一部近邻作品。"""

    work_id: int
    similarity: float
    rank: int


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
    """进程内高维余弦近邻索引。

    构造时从 Catalog 读取该 embedding version 的三种作品级向量（只读文件一次），
    默认排除 work_links 中的 duplicate 作品。对象不再使用时应调用 close() 释放矩阵。
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
        for row in catalog.list_embeddings(self.version):
            kind = str(row.get("kind"))
            if kind not in KINDS:
                continue
            work_id = int(row["work_id"])
            if work_id in duplicate_ids:
                continue
            kind_ids[kind].append(work_id)
            kind_paths[kind][work_id] = str(row["path"])
            kind_dims[kind][work_id] = int(row["dim"])

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

        self._matrices: dict[str, np.ndarray] = {}
        for kind in list(self._kind_ids):
            self._matrices[kind] = self._load_kind(kind)

        logger.info(
            "近邻索引就绪: version=%s works=%d kinds=%s",
            self.version,
            len(self._work_ids),
            ",".join(self.available_kinds),
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

    # ------------------------------------------------------------------ 公开 API
    @property
    def work_ids(self) -> tuple[int, ...]:
        """canonical 作品 id（按 work_id 排序；包含至少有一种模态向量的作品）。"""

        return self._work_ids

    @property
    def available_kinds(self) -> tuple[str, ...]:
        """有向量数据的模态，顺序与 KINDS 一致。"""

        return tuple(kind for kind in KINDS if kind in self._matrices)

    def has_vector(self, work_id: int, kind: str = DEFAULT_KIND) -> bool:
        """判断某部作品是否有指定模态的向量。"""

        if kind not in self._kind_pos:
            return False
        return int(work_id) in self._kind_pos[kind]

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

    def neighbors(
        self,
        work_id: int,
        *,
        kind: str = DEFAULT_KIND,
        k: int = 5,
    ) -> list[Neighbor]:
        """返回指定模态下最相似的 k 部作品（不含自身，按相似度降序）。

        作品不足 k 部时返回尽量多的条数；作品在该模态没有向量时抛 NebulaError。
        """

        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise NebulaError(f"k 必须是 >= 1 的整数，收到 {k!r}")
        matrix, work_ids = self._kind_data(kind)
        work_key = int(work_id)
        position = self._kind_pos[kind].get(work_key)
        if position is None:
            raise NebulaError(f"作品 {work_key} 没有 {kind} 向量，无法计算近邻")
        similarities = matrix @ matrix[position]
        similarities[position] = -np.inf
        take = min(int(k), len(work_ids) - 1)
        if take <= 0:
            return []
        # 稳定排序：相似度相同时按 work_id 升序（矩阵行按 work_id 排序）
        order = np.argsort(-similarities, kind="stable")[:take]
        return [
            Neighbor(
                work_id=int(work_ids[int(index)]),
                similarity=float(np.clip(similarities[int(index)], -1.0, 1.0)),
                rank=rank,
            )
            for rank, index in enumerate(order, start=1)
        ]

    def similarity(
        self,
        a_work_id: int,
        b_work_id: int,
        *,
        kind: str = DEFAULT_KIND,
    ) -> float:
        """两部作品在指定模态下的真实高维余弦相似度。"""

        matrix, _ = self._kind_data(kind)
        first = self._kind_pos[kind].get(int(a_work_id))
        second = self._kind_pos[kind].get(int(b_work_id))
        if first is None:
            raise NebulaError(f"作品 {a_work_id} 没有 {kind} 向量")
        if second is None:
            raise NebulaError(f"作品 {b_work_id} 没有 {kind} 向量")
        if first == second:
            return 1.0
        value = float(np.dot(matrix[first], matrix[second]))
        return float(np.clip(value, -1.0, 1.0))

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

        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise NebulaError(f"k 必须是 >= 1 的整数，收到 {k!r}")
        matrix, work_ids = self._kind_data(kind)
        count = len(work_ids)
        if count <= 1:
            return {int(work_ids[0]): 1.0} if count == 1 else {}
        take = min(int(k), count - 1)
        similarities = matrix.astype(np.float64) @ matrix.astype(np.float64).T
        distances: dict[int, float] = {}
        for index, work_id in enumerate(work_ids):
            row = similarities[index].copy()
            row[index] = -np.inf
            kth_similarity = float(np.sort(row)[::-1][take - 1])
            distances[int(work_id)] = float(1.0 - np.clip(kth_similarity, -1.0, 1.0))
        return distances

    def close(self) -> None:
        """释放缓存的矩阵；之后继续查询会抛 NebulaError。"""

        self._matrices.clear()
        self._closed = True


def load_similarity_index(
    catalog: Catalog,
    config: Config,
    *,
    version: str | None = None,
) -> SimilarityIndex:
    """构造 SimilarityIndex（便捷函数，app 用 st.cache_resource 包一层）。"""

    return SimilarityIndex(catalog, config, version=version)
