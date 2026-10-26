"""共享高维运算（nebula.highdim）的合成数据测试。"""

from __future__ import annotations

import numpy as np
import pytest

from nebula.errors import NebulaError
from nebula.highdim import (
    cosine_distance_matrix,
    cosine_similarity_matrix,
    normalize_vectors,
    topk_distance_neighbors,
    topk_similarity_neighbors,
)


def test_normalize_vectors_and_invalid_inputs():
    matrix = np.asarray([[3.0, 4.0], [0.0, 2.0]])
    normalized = normalize_vectors(matrix)
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)
    assert np.allclose(normalized[0], [0.6, 0.8])
    with pytest.raises(NebulaError):
        normalize_vectors(np.zeros((2, 3)))
    with pytest.raises(NebulaError):
        normalize_vectors(np.asarray([[1.0, np.nan]]))
    with pytest.raises(NebulaError):
        normalize_vectors(np.ones(3))


def test_cosine_similarity_and_distance_known_values():
    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    similarity = cosine_similarity_matrix(matrix)
    assert np.allclose(np.diag(similarity), 1.0)
    assert np.allclose(similarity, similarity.T)
    assert similarity[0, 1] == pytest.approx(0.0)
    assert similarity[0, 2] == pytest.approx(1.0 / np.sqrt(2.0))
    distance = cosine_distance_matrix(matrix)
    assert np.allclose(np.diag(distance), 0.0)
    assert np.allclose(distance, distance.T)
    assert distance[0, 1] == pytest.approx(1.0)
    assert distance[0, 2] == pytest.approx(1.0 - 1.0 / np.sqrt(2.0))
    assert np.all(distance >= 0.0)


def test_topk_similarity_excludes_self_and_is_stable_for_ties():
    similarity = np.asarray(
        [
            [1.0, 0.5, 0.5, 0.1],
            [0.5, 1.0, 0.2, 0.2],
            [0.5, 0.2, 1.0, 0.9],
            [0.1, 0.2, 0.9, 1.0],
        ]
    )
    top = topk_similarity_neighbors(similarity, 2)
    assert top.shape == (4, 2)
    assert np.all(top != np.arange(4)[:, None])
    # 并列 0.5 时按索引稳定：行 0 选 1、2
    assert top[0].tolist() == [1, 2]
    assert top[2].tolist() == [3, 0]
    assert topk_similarity_neighbors(similarity, 10).shape == (4, 3)
    with pytest.raises(NebulaError):
        topk_similarity_neighbors(similarity, 0)


def test_topk_distance_ascending_and_deterministic():
    distance = np.asarray(
        [
            [0.0, 0.2, 0.2, 0.9],
            [0.2, 0.0, 0.1, 0.3],
            [0.2, 0.1, 0.0, 0.4],
            [0.9, 0.3, 0.4, 0.0],
        ]
    )
    first = topk_distance_neighbors(distance, 2)
    second = topk_distance_neighbors(distance, 2)
    assert np.array_equal(first, second)
    assert first[0].tolist() == [1, 2]
    assert first[1].tolist() == [2, 0]
    assert np.all(first != np.arange(4)[:, None])
    with pytest.raises(NebulaError):
        topk_distance_neighbors(np.ones((2, 3)), 1)
