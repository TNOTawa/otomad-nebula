"""nebula.similarity 单元测试：高维余弦近邻、缓存、缺失模态与 duplicate 排除。

全部使用临时 SQLite + 合成 npy，不加载模型、不访问网络。
"""

from __future__ import annotations

import numpy as np
import pytest

from nebula import similarity
from nebula.config import Config
from nebula.db import Catalog
from nebula.errors import NebulaError
from nebula.similarity import (
    KIND_LABELS,
    KINDS,
    Neighbor,
    load_similarity_index,
)


# --------------------------------------------------------------------------- 合成数据
def _ensure_embedding_version(catalog: Catalog, config: Config) -> str:
    catalog.register_embedding_version(
        config.embedding_version,
        model_checkpoint=config.model.checkpoint,
        model_revision=config.model.revision,
        frames=config.model.frames,
        clip_seconds=config.media.clip_seconds,
        clips_per_work=config.media.clips_per_work,
        audio_sample_rate=config.media.audio_sample_rate,
        params={},
    )
    return config.embedding_version


def _unit(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return (array / np.linalg.norm(array)).astype(np.float32)


def _add_work(
    catalog: Catalog,
    config: Config,
    number: int,
    *,
    vectors: dict[str, np.ndarray] | None = None,
    dim: int = 8,
    kinds: tuple[str, ...] = KINDS,
    version: str | None = None,
    source: str = "bilibili",
) -> int:
    """登记一部作品并写各模态向量文件。vectors 可指定具体向量，否则随机生成。"""

    embedding_version = _ensure_embedding_version(catalog, config)
    target_version = version or embedding_version
    if target_version != embedding_version:
        catalog.register_embedding_version(
            target_version,
            model_checkpoint=config.model.checkpoint,
            model_revision=config.model.revision,
            frames=config.model.frames,
            clip_seconds=config.media.clip_seconds,
            clips_per_work=config.media.clips_per_work,
            audio_sample_rate=config.media.audio_sample_rate,
            params={},
        )
    if source == "local":
        work_id, _ = catalog.upsert_local_work(f"clips/作品 {number}.mp4", title=f"作品 {number}")
    else:
        work_id, _ = catalog.upsert_bilibili_work(
            f"BV1xx411c7{number:02d}", title=f"作品 {number}", duration_s=30.0 + number
        )
    rng = np.random.default_rng(1000 + number)
    for kind in kinds:
        if vectors is not None and kind in vectors:
            vector = np.asarray(vectors[kind], dtype=np.float32)
        else:
            vector = rng.normal(size=dim).astype(np.float32)
        path = config.paths.embeddings / target_version / str(work_id) / f"{kind}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vector)
        catalog.register_embedding(
            work_id,
            version=target_version,
            kind=kind,
            path=config.data_relpath(path),
            dim=int(vector.shape[0]),
        )
    return work_id


def _add_clip_vectors(
    catalog: Catalog,
    config: Config,
    work_id: int,
    *,
    matrices: dict[str, np.ndarray],
    version: str | None = None,
    start_step: float = 10.0,
) -> None:
    """为一部作品登记片段级向量（堆叠 npy）与 clips 表起始时间。"""

    target_version = version or config.embedding_version
    max_clips = 0
    for kind, matrix in matrices.items():
        array = np.asarray(matrix, dtype=np.float32)
        path = config.paths.embeddings / target_version / str(work_id) / f"{kind}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)
        catalog.register_embedding(
            work_id,
            version=target_version,
            kind=kind,
            path=config.data_relpath(path),
            dim=int(array.shape[1]),
            n_clips=int(array.shape[0]),
        )
        max_clips = max(max_clips, int(array.shape[0]))
    catalog.replace_clips(
        work_id,
        version=target_version,
        clips=[(idx, start_step * idx, 8.0) for idx in range(max_clips)],
    )


def _seed_axes_works(catalog: Catalog, config: Config) -> dict[int, int]:
    """4 部作品，joint 向量在几个正交方向上，便于手算余弦。"""

    vectors = {
        1: {"work_joint": _unit([1.0, 0.0, 0.0, 0.0])},
        2: {"work_joint": _unit([0.9, 0.43589, 0.0, 0.0])},
        3: {"work_joint": _unit([0.0, 1.0, 0.0, 0.0])},
        4: {"work_joint": _unit([0.0, 0.0, 1.0, 0.0])},
    }
    return {
        number: _add_work(catalog, config, number, vectors=payload, kinds=("work_joint",))
        for number, payload in vectors.items()
    }


# --------------------------------------------------------------------------- 基础语义
def test_constants():
    assert KINDS == ("work_joint", "work_audio", "work_video")
    assert KIND_LABELS == {"work_joint": "联合", "work_audio": "音频", "work_video": "画面"}


def test_neighbors_sorted_and_excludes_self(config: Config, catalog: Catalog):
    ids = _seed_axes_works(catalog, config)
    index = load_similarity_index(catalog, config)
    try:
        neighbors = index.neighbors(ids[2], k=5)
        assert len(neighbors) == 3
        assert [neighbor.rank for neighbor in neighbors] == [1, 2, 3]
        assert all(isinstance(neighbor, Neighbor) for neighbor in neighbors)
        assert ids[2] not in [neighbor.work_id for neighbor in neighbors]
        assert [neighbor.work_id for neighbor in neighbors] == [ids[1], ids[3], ids[4]]
        similarities = [neighbor.similarity for neighbor in neighbors]
        assert similarities == sorted(similarities, reverse=True)
        # 与手算的高维余弦一致（不是坐标距离）
        expected = float(np.dot(_unit([0.9, 0.43589, 0.0, 0.0]), _unit([1.0, 0.0, 0.0, 0.0])))
        assert neighbors[0].similarity == pytest.approx(expected, abs=1e-5)
        assert neighbors[0].similarity == pytest.approx(
            index.similarity(ids[2], ids[1]), abs=1e-6
        )
    finally:
        index.close()


def test_k_truncation(config: Config, catalog: Catalog):
    ids = _seed_axes_works(catalog, config)
    index = load_similarity_index(catalog, config)
    try:
        assert len(index.neighbors(ids[1], k=1)) == 1
        assert len(index.neighbors(ids[1], k=2)) == 2
        assert len(index.neighbors(ids[1], k=99)) == 3
        with pytest.raises(NebulaError):
            index.neighbors(ids[1], k=0)
        with pytest.raises(NebulaError):
            index.neighbors(ids[1], k=-3)
    finally:
        index.close()


def test_similarity_symmetric_and_self(config: Config, catalog: Catalog):
    ids = _seed_axes_works(catalog, config)
    index = load_similarity_index(catalog, config)
    try:
        for first in ids.values():
            assert index.similarity(first, first) == 1.0
            for second in ids.values():
                value = index.similarity(first, second)
                assert -1.0 <= value <= 1.0
                assert value == pytest.approx(index.similarity(second, first), abs=1e-6)
        with pytest.raises(NebulaError):
            index.similarity(ids[1], 99999)
    finally:
        index.close()


def test_work_ids_and_index_of(config: Config, catalog: Catalog):
    ids = _seed_axes_works(catalog, config)
    index = load_similarity_index(catalog, config)
    try:
        assert index.work_ids == tuple(sorted(ids.values()))
        for work_id in index.work_ids:
            position = index.index_of(work_id)
            assert index.work_ids[position] == work_id
        with pytest.raises(NebulaError):
            index.index_of(99999)
    finally:
        index.close()


def test_matrix_normalized_and_cached(config: Config, catalog: Catalog):
    _seed_axes_works(catalog, config)
    index = load_similarity_index(catalog, config)
    try:
        matrix = index.matrix("work_joint")
        assert matrix.dtype == np.float32
        assert matrix.shape == (4, 4)
        norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
        np.testing.assert_allclose(norms, np.ones(4), atol=1e-6)
        assert index.matrix("work_joint") is matrix  # 返回同一缓存对象
        with pytest.raises(NebulaError):
            index.matrix("clip_joint")
    finally:
        index.close()


def test_close_releases_matrix(config: Config, catalog: Catalog):
    ids = _seed_axes_works(catalog, config)
    index = load_similarity_index(catalog, config)
    index.close()
    with pytest.raises(NebulaError):
        index.matrix("work_joint")
    with pytest.raises(NebulaError):
        index.neighbors(ids[1])


# --------------------------------------------------------------------------- 缺失模态 / 多模态
def test_missing_modality(config: Config, catalog: Catalog):
    joint = {"work_joint": _unit([1.0, 0.0, 0.0, 0.0])}
    joint_video = {**joint, "work_video": _unit([0.0, 1.0, 0.0, 0.0])}
    first = _add_work(catalog, config, 1, vectors=joint_video, kinds=("work_joint", "work_video"))
    second = _add_work(catalog, config, 2, vectors=joint_video, kinds=("work_joint", "work_video"))
    third = _add_work(catalog, config, 3, vectors=joint, kinds=("work_joint",))

    index = load_similarity_index(catalog, config)
    try:
        assert index.available_kinds == ("work_joint", "work_video")
        assert index.work_ids == tuple(sorted([first, second, third]))
        assert index.has_vector(first, "work_video") is True
        assert index.has_vector(third, "work_video") is False
        assert len(index.neighbors(first, kind="work_video", k=5)) == 1
        with pytest.raises(NebulaError, match="没有 work_video 向量"):
            index.neighbors(third, kind="work_video")
        with pytest.raises(NebulaError, match="没有 work_audio 向量数据"):
            index.neighbors(first, kind="work_audio")
    finally:
        index.close()


def test_duplicate_excluded(config: Config, catalog: Catalog):
    ids = _seed_axes_works(catalog, config)
    catalog.link_works(ids[1], ids[4], method="title+duration", confidence=0.99)
    index = load_similarity_index(catalog, config)
    try:
        assert ids[4] not in index.work_ids
        assert ids[4] not in [neighbor.work_id for neighbor in index.neighbors(ids[2], k=5)]
        assert index.has_vector(ids[4]) is False
        assert index.matrix("work_joint").shape[0] == 3
        with pytest.raises(NebulaError):
            index.similarity(ids[1], ids[4])
    finally:
        index.close()


# --------------------------------------------------------------------------- 缓存
def test_cache_reads_each_npy_only_once(config: Config, catalog: Catalog, monkeypatch):
    for number in (1, 2, 3):
        _add_work(catalog, config, number)  # 3 部作品 × 3 模态 = 9 个 npy
    reads: list[str] = []
    original = similarity._read_vector_file

    def counting_reader(path):
        reads.append(str(path))
        return original(path)

    monkeypatch.setattr(similarity, "_read_vector_file", counting_reader)
    index = load_similarity_index(catalog, config)
    assert len(reads) == 9
    try:
        for _ in range(3):
            index.matrix("work_joint")
            index.matrix("work_audio")
            index.matrix("work_video")
            for work_id in index.work_ids:
                index.neighbors(work_id, k=2)
            index.kth_neighbor_distance(k=2)
        assert len(reads) == 9  # 没有重复读 npy
    finally:
        index.close()


# --------------------------------------------------------------------------- 第 k 近邻距离
def test_kth_neighbor_distance(config: Config, catalog: Catalog):
    # w1 与 w2 向量相同 -> 最近邻距离 0；w3 与两者都正交 -> 明显更稀疏
    vectors = {
        1: {"work_joint": _unit([1.0, 0.0, 0.0, 0.0])},
        2: {"work_joint": _unit([1.0, 0.0, 0.0, 0.0])},
        3: {"work_joint": _unit([0.0, 1.0, 0.0, 0.0])},
    }
    ids = {
        number: _add_work(catalog, config, number, vectors=payload, kinds=("work_joint",))
        for number, payload in vectors.items()
    }
    index = load_similarity_index(catalog, config)
    try:
        distances = index.kth_neighbor_distance(k=1)
        assert set(distances) == set(ids.values())
        assert distances[ids[1]] == pytest.approx(0.0, abs=1e-6)
        assert distances[ids[2]] == pytest.approx(0.0, abs=1e-6)
        assert distances[ids[3]] == pytest.approx(1.0, abs=1e-6)
        # k 超过可用邻居数时退化为最远可用邻居（这里是第 2 个）
        wide = index.kth_neighbor_distance(k=5)
        assert wide[ids[1]] == pytest.approx(1.0, abs=1e-6)
        with pytest.raises(NebulaError):
            index.kth_neighbor_distance(k=0)
    finally:
        index.close()


def test_kth_neighbor_distance_marks_isolated_work_sparse(config: Config, catalog: Catalog):
    rng = np.random.default_rng(7)
    base = rng.normal(size=8)
    ids = {}
    for number in (1, 2, 3, 4):
        noise = rng.normal(scale=0.01, size=8)
        ids[number] = _add_work(
            catalog, config, number, vectors={"work_joint": (base + noise).astype(np.float32)}
        )
    isolated = _add_work(
        catalog, config, 5, vectors={"work_joint": rng.normal(size=8).astype(np.float32)}
    )
    index = load_similarity_index(catalog, config)
    try:
        distances = index.kth_neighbor_distance(k=1)
        for number in (2, 3, 4):
            assert distances[ids[number]] < distances[isolated]
    finally:
        index.close()


# --------------------------------------------------------------------------- 错误路径
def test_version_helpers(config: Config, catalog: Catalog):
    custom = "custom-vectors-v1"
    _add_work(catalog, config, 1, version=custom)
    _add_work(catalog, config, 2, version=custom)
    index = load_similarity_index(catalog, config, version=custom)
    assert index.version == custom
    assert len(index.work_ids) == 2
    index.close()

    empty = load_similarity_index(catalog, config, version="不存在")
    try:
        assert empty.available_kinds == ()
        assert empty.work_ids == ()
        with pytest.raises(NebulaError):
            empty.matrix("work_joint")
        with pytest.raises(NebulaError):
            empty.kth_neighbor_distance()
    finally:
        empty.close()


def test_missing_vector_file(config: Config, catalog: Catalog):
    work_id = _add_work(catalog, config, 1)
    version = config.embedding_version
    path = config.paths.embeddings / version / str(work_id) / "work_joint.npy"
    path.unlink()
    with pytest.raises(NebulaError, match="向量文件不存在"):
        load_similarity_index(catalog, config)


def test_registered_dim_mismatch(config: Config, catalog: Catalog):
    work_id = _add_work(catalog, config, 1)
    version = config.embedding_version
    path = config.paths.embeddings / version / str(work_id) / "work_joint.npy"
    np.save(path, np.ones(3, dtype=np.float32))
    with pytest.raises(NebulaError, match="维度与登记不一致"):
        load_similarity_index(catalog, config)


def test_zero_vector_rejected(config: Config, catalog: Catalog):
    work_id = _add_work(catalog, config, 1)
    version = config.embedding_version
    path = config.paths.embeddings / version / str(work_id) / "work_joint.npy"
    np.save(path, np.zeros(8, dtype=np.float32))
    with pytest.raises(NebulaError, match="零向量"):
        load_similarity_index(catalog, config)


# --------------------------------------------------------------------------- P1 近邻解释
def test_neighbor_dataclass_backward_compatible():
    neighbor = Neighbor(work_id=1, similarity=0.9, rank=1)
    assert neighbor.pair_percentile == 0.0
    assert neighbor.is_mutual is False


def test_neighbor_explanation_fields(config: Config, catalog: Catalog):
    ids = _seed_axes_works(catalog, config)
    index = load_similarity_index(catalog, config)
    try:
        neighbors = index.neighbors(ids[2], k=5)
        assert len(neighbors) == 3
        work_ids = index.work_ids
        matrix = index.matrix("work_joint")
        pair_similarities = [
            float(np.dot(matrix[first], matrix[second]))
            for first in range(len(work_ids))
            for second in range(first + 1, len(work_ids))
        ]
        for neighbor in neighbors:
            assert 0.0 <= neighbor.pair_percentile <= 100.0
            assert isinstance(neighbor.is_mutual, bool)
            similarity = index.similarity(ids[2], neighbor.work_id)
            below = sum(1 for value in pair_similarities if value < similarity - 1e-12)
            expected = 100.0 * below / len(pair_similarities)
            assert neighbor.pair_percentile == pytest.approx(expected, abs=1e-6)
        # 互为近邻的定义：目标作品的 Top-k 里也包含当前作品
        for neighbor in neighbors:
            target_top = index.neighbors(neighbor.work_id, k=5)
            target_ids = {item.work_id for item in target_top}
            assert neighbor.is_mutual is (ids[2] in target_ids)
    finally:
        index.close()


def test_pair_percentile_hand_calculated(config: Config, catalog: Catalog):
    # 3 部作品：A-B=0，A-C=B-C=0.7071 -> C 与 A 的分位 = 100 * 1 / 3
    vectors = {
        1: {"work_joint": _unit([1.0, 0.0, 0.0, 0.0])},
        2: {"work_joint": _unit([0.0, 1.0, 0.0, 0.0])},
        3: {"work_joint": _unit([1.0, 1.0, 0.0, 0.0])},
    }
    ids = {
        number: _add_work(catalog, config, number, vectors=payload, kinds=("work_joint",))
        for number, payload in vectors.items()
    }
    index = load_similarity_index(catalog, config)
    try:
        similarity_ab = index.similarity(ids[1], ids[2])
        similarity_ac = index.similarity(ids[1], ids[3])
        assert similarity_ab == pytest.approx(0.0, abs=1e-6)
        assert similarity_ac == pytest.approx(0.70710678, abs=1e-6)
        # A-B: 没有任何作品对低于 0
        assert index.pair_percentile(ids[1], ids[2]) == pytest.approx(0.0, abs=1e-6)
        # A-C: 只有 A-B=0 低于 0.7071 -> 1/3
        assert index.pair_percentile(ids[1], ids[3]) == pytest.approx(100.0 / 3.0, abs=1e-6)
        neighbors = {item.work_id: item for item in index.neighbors(ids[1], k=2)}
        assert neighbors[ids[3]].pair_percentile == pytest.approx(100.0 / 3.0, abs=1e-6)
    finally:
        index.close()


def test_hubness_and_hub_top_hand_calculated(config: Config, catalog: Catalog):
    # 1、2 向量相同；3/4/5 互相正交。k=1 时所有人都把 1 当作最近邻（2 也把 1 当最近）
    vectors = {
        1: {"work_joint": _unit([1.0, 0.0, 0.0, 0.0])},
        2: {"work_joint": _unit([1.0, 0.0, 0.0, 0.0])},
        3: {"work_joint": _unit([0.0, 1.0, 0.0, 0.0])},
        4: {"work_joint": _unit([0.0, 0.0, 1.0, 0.0])},
        5: {"work_joint": _unit([0.0, 0.0, 0.0, 1.0])},
    }
    ids = {
        number: _add_work(catalog, config, number, vectors=payload, kinds=("work_joint",))
        for number, payload in vectors.items()
    }
    index = load_similarity_index(catalog, config)
    try:
        counts = index.hubness(k=1)
        assert counts == {ids[1]: 4, ids[2]: 1, ids[3]: 0, ids[4]: 0, ids[5]: 0}
        assert sum(counts.values()) == len(ids)
        assert index.hub_top(k=1, n=3) == [(ids[1], 4), (ids[2], 1), (ids[3], 0)]

        # k=2 时每部作品出 2 票
        counts_k2 = index.hubness(k=2)
        assert sum(counts_k2.values()) == 2 * len(ids)
        # 互为近邻标记与 hubness 一致：1 的近邻 2 的 Top-k 包含 1
        top_pair = index.neighbors(ids[1], k=1)[0]
        assert top_pair.work_id == ids[2] and top_pair.is_mutual is True
        with pytest.raises(NebulaError):
            index.hubness(k=0)
        with pytest.raises(NebulaError):
            index.hub_top(n=0)
    finally:
        index.close()


# --------------------------------------------------------------------------- P2 片段级
def _seed_clip_pair(catalog: Catalog, config: Config):
    """两部作品，每部 3 个片段；片段向量按已知方向构造便于手算。"""

    ids = {
        number: _add_work(catalog, config, number, kinds=("work_joint",))
        for number in (1, 2)
    }
    clips_a = np.vstack(
        [
            _unit([1.0, 0.0, 0.0, 0.0]),
            _unit([0.0, 1.0, 0.0, 0.0]),
            _unit([1.0, 1.0, 0.0, 0.0]),
        ]
    ).astype(np.float32)
    clips_b = np.vstack(
        [
            _unit([1.0, 0.0, 0.0, 0.0]),
            _unit([0.0, 0.0, 1.0, 0.0]),
            _unit([1.0, 0.0, 1.0, 0.0]),
        ]
    ).astype(np.float32)
    _add_clip_vectors(catalog, config, ids[1], matrices={"clip_joint": clips_a})
    _add_clip_vectors(catalog, config, ids[2], matrices={"clip_joint": clips_b})
    return ids, clips_a, clips_b


def test_clip_matrix_and_similarity(config: Config, catalog: Catalog):
    ids, clips_a, clips_b = _seed_clip_pair(catalog, config)
    index = load_similarity_index(catalog, config)
    try:
        matrix_a = index.clip_matrix(ids[1])
        matrix_b = index.clip_matrix(ids[2], "clip_joint")
        assert matrix_a.shape == (3, 4)
        assert matrix_b.shape == (3, 4)
        np.testing.assert_allclose(np.linalg.norm(matrix_a, axis=1), np.ones(3), atol=1e-6)
        similarities = index.clip_similarity(ids[1], ids[2])
        expected = clips_a.astype(np.float64) @ clips_b.astype(np.float64).T
        assert similarities.shape == (3, 3)
        np.testing.assert_allclose(similarities, np.clip(expected, -1.0, 1.0), atol=1e-6)
        assert index.has_clip_vector(ids[1], "clip_joint") is True
        assert index.has_clip_vector(ids[1], "clip_audio") is False
        assert index.available_clip_kinds == ("clip_joint",)
        with pytest.raises(NebulaError):
            index.clip_matrix(ids[1], "clip_audio")
    finally:
        index.close()


def test_best_clip_pairs_and_explain(config: Config, catalog: Catalog):
    ids, clips_a, clips_b = _seed_clip_pair(catalog, config)
    index = load_similarity_index(catalog, config)
    try:
        pairs = index.best_clip_pairs(ids[1], ids[2], "clip_joint", 2)
        assert len(pairs) == 2
        assert set(pairs[0]) == {"a_idx", "b_idx", "similarity"}
        assert pairs[0]["a_idx"] == 0 and pairs[0]["b_idx"] == 0
        assert pairs[0]["similarity"] == pytest.approx(1.0, abs=1e-6)
        # 第二高是一对 0.707 的片段（A0-B2 与 A2-B0 并列，行优先稳定排序取前者）
        assert (pairs[1]["a_idx"], pairs[1]["b_idx"]) in {(0, 2), (2, 0)}
        assert pairs[1]["similarity"] == pytest.approx(np.sqrt(0.5), abs=1e-6)
        assert [pair["similarity"] for pair in pairs] == sorted(
            [pair["similarity"] for pair in pairs], reverse=True
        )

        explanation = index.explain_pair(ids[1], ids[2], "clip_joint")
        expected = clips_a.astype(np.float64) @ clips_b.astype(np.float64).T
        assert explanation["max"] == pytest.approx(float(np.max(expected)), abs=1e-6)
        assert explanation["mean"] == pytest.approx(float(np.mean(expected)), abs=1e-6)
        assert explanation["count_above_threshold"] == int(np.count_nonzero(expected >= 0.5))
        assert explanation["n_a"] == 3 and explanation["n_b"] == 3
        custom = index.explain_pair(ids[1], ids[2], "clip_joint", threshold=0.9)
        assert custom["count_above_threshold"] == 1
        with pytest.raises(NebulaError):
            index.explain_pair(ids[1], ids[2], "clip_joint", threshold=2.0)
    finally:
        index.close()


def test_clip_times(config: Config, catalog: Catalog):
    ids, _, _ = _seed_clip_pair(catalog, config)
    index = load_similarity_index(catalog, config)
    try:
        times = index.clip_times(ids[1])
        assert [item["idx"] for item in times] == [0, 1, 2]
        assert [item["start_s"] for item in times] == [0.0, 10.0, 20.0]
        assert all(item["duration_s"] == 8.0 for item in times)
        # 返回副本，外部修改不影响缓存
        times[0]["start_s"] = 999.0
        assert index.clip_times(ids[1])[0]["start_s"] == 0.0
        with pytest.raises(NebulaError):
            index.clip_times(99999)
    finally:
        index.close()


def test_clip_cache_reads_each_npy_once(config: Config, catalog: Catalog, monkeypatch):
    ids, _, _ = _seed_clip_pair(catalog, config)
    reads: list[str] = []
    original = similarity._read_vector_file

    def counting_reader(path):
        reads.append(str(path))
        return original(path)

    monkeypatch.setattr(similarity, "_read_vector_file", counting_reader)
    index = load_similarity_index(catalog, config)
    work_reads = len(reads)  # 2 部作品 × work_joint 1 种 = 2
    try:
        index.clip_matrix(ids[1])
        index.clip_matrix(ids[1])  # 命中缓存
        index.clip_similarity(ids[1], ids[2])
        index.best_clip_pairs(ids[1], ids[2])
        index.explain_pair(ids[1], ids[2])
        assert len(reads) == work_reads + 2  # 两部作品各读一次片段 npy
        index.clip_matrix(ids[2])
        assert len(reads) == work_reads + 2
    finally:
        index.close()


def test_close_clears_clip_cache(config: Config, catalog: Catalog):
    ids, _, _ = _seed_clip_pair(catalog, config)
    index = load_similarity_index(catalog, config)
    assert index.clip_matrix(ids[1]).shape == (3, 4)
    index.close()
    with pytest.raises(NebulaError):
        index.clip_matrix(ids[1])


def test_clip_registered_count_mismatch(config: Config, catalog: Catalog):
    ids, clips_a, _ = _seed_clip_pair(catalog, config)
    version = config.embedding_version
    path = config.paths.embeddings / version / str(ids[1]) / "clip_joint.npy"
    np.save(path, np.vstack([clips_a[0], clips_a[1]]))  # 登记 3 段，文件 2 段
    index = load_similarity_index(catalog, config)
    try:
        with pytest.raises(NebulaError, match="片段数量与登记不一致"):
            index.clip_matrix(ids[1])
    finally:
        index.close()
