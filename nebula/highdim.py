"""高维向量的共享基础运算（归一化、余弦矩阵、Top-k 邻域）。

v0.4.5 工程收敛：此前 tools/build_consensus_graph.py 与 tools/embedding_diagnostics.py
各自实现了逐行归一化、余弦矩阵和 Top-k；两处语义（相似度降序 / 距离升序）不同，
但底层数学相同。这里提供单一实现，工具只保留各自的业务排序语义。

不写数据库、不加载模型、不访问网络。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import NebulaError

EPS = 1e-9


def normalize_vectors(matrix: Any, *, where: str = "向量矩阵") -> np.ndarray:
    """校验二维矩阵并逐行 L2 归一化（float64）；零向量/NaN 显式报错。"""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise NebulaError(f"{where} 形状非法: {values.shape}")
    if not np.all(np.isfinite(values)):
        raise NebulaError(f"{where} 包含 NaN/Inf")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    zero_rows = np.where(norms <= EPS)[0]
    if zero_rows.size:
        raise NebulaError(f"{where} 有 {zero_rows.size} 个零向量")
    return values / norms


def cosine_similarity_matrix(matrix: Any, *, where: str = "余弦相似度输入") -> np.ndarray:
    """逐行归一化后的余弦相似度矩阵（对称、对角线 1，值裁剪到 [-1,1]）。"""

    normalized = normalize_vectors(matrix, where=where)
    similarity = normalized @ normalized.T
    np.clip(similarity, -1.0, 1.0, out=similarity)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def cosine_distance_matrix(matrix: Any, *, where: str = "余弦距离输入") -> np.ndarray:
    """1 - 余弦相似度（对称、对角线 0、非负）。"""

    similarity = cosine_similarity_matrix(matrix, where=where)
    distance = np.maximum(1.0 - similarity, 0.0)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    return distance


def _validate_square(values: Any, *, where: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise NebulaError(f"{where} 必须是方阵，收到 {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise NebulaError(f"{where} 包含 NaN/Inf")
    return matrix


def _validate_k(k: int) -> int:
    try:
        value = int(k)
    except (TypeError, ValueError) as exc:
        raise NebulaError(f"k 必须是正整数，收到 {k!r}") from exc
    if value < 1:
        raise NebulaError(f"k 必须 >= 1，收到 {k}")
    return value


def topk_similarity_neighbors(similarity: Any, k: int) -> np.ndarray:
    """每行相似度最高的 k 个邻居索引（不含自身；降序，并列按索引稳定）。"""

    values = _validate_square(similarity, where="相似度矩阵")
    count = int(values.shape[0])
    effective = min(_validate_k(k), max(count - 1, 0))
    if effective == 0:
        return np.zeros((count, 0), dtype=np.int64)
    work = values.copy()
    np.fill_diagonal(work, -np.inf)
    order = np.argsort(-work, axis=1, kind="stable")
    return order[:, :effective].astype(np.int64)


def topk_distance_neighbors(distance: Any, k: int) -> np.ndarray:
    """每行距离最小的 k 个邻居索引（不含自身；升序，并列按索引稳定）。"""

    values = _validate_square(distance, where="距离矩阵")
    count = int(values.shape[0])
    effective = min(_validate_k(k), max(count - 1, 0))
    if effective == 0:
        return np.zeros((count, 0), dtype=np.int64)
    work = values.copy()
    np.fill_diagonal(work, np.inf)
    order = np.argsort(work, axis=1, kind="stable")
    return order[:, :effective].astype(np.int64)
