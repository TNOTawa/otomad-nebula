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
