"""nebula.embed 的单元测试与可选的 PE-AV 真实模型测试。

默认 pytest 只运行不依赖模型的测试（HashTestEmbedder / 聚合 / 原子读写 /
懒加载），不会导入 torch、不会访问网络。真实模型测试标记为 model，
需要显式设置 NEBULA_MODEL_TESTS=1 才会运行：

    $env:NEBULA_MODEL_TESTS = "1"
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_embed.py -m model -q -s
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nebula.config import load_config
from nebula.embed import (
    Embedder,
    EmbeddingBatch,
    HashTestEmbedder,
    PEAVEmbedder,
    aggregate_work_vector,
    load_vector,
    save_vector,
)
from nebula.errors import EmbeddingError
from nebula.media import black_video_clip, extract_clip, mute_audio_clip, probe_media
from nebula.sampling import plan_clips

MODEL_TESTS_ENABLED = os.environ.get("NEBULA_MODEL_TESTS") == "1"
MODEL_SKIP_REASON = (
    "真实模型测试默认跳过：设置 NEBULA_MODEL_TESTS=1 后运行"
    "（需要 transformers/torch 与本地 PE-AV 权重）"
)
REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# HashTestEmbedder（离线、确定性、无需模型）
# ---------------------------------------------------------------------------


@pytest.fixture
def media_files(tmp_path: Path) -> tuple[Path, Path]:
    """两个内容固定的假文件（HashTestEmbedder 只读内容，不解析格式）。"""

    video = tmp_path / "video.bin"
    audio = tmp_path / "audio.bin"
    video.write_bytes(b"fake video content" * 128)
    audio.write_bytes(b"fake audio content" * 128)
    return video, audio


def test_hash_embedder_metadata():
    embedder = HashTestEmbedder()
    assert embedder.version == "test-hash-v1"
    assert embedder.is_test_only is True
    assert embedder.dim == 32
    assert isinstance(embedder, Embedder)


def test_hash_embedder_custom_dim(media_files: tuple[Path, Path]):
    embedder = HashTestEmbedder(dim=7)
    batch = embedder.encode_pairs([media_files])
    assert batch.joint.shape == (1, 7)
    assert batch.audio is not None and batch.audio.shape == (1, 7)
    assert batch.video is not None and batch.video.shape == (1, 7)


@pytest.mark.parametrize("dim", [0, -3, True, 2.5])
def test_hash_embedder_invalid_dim_raises(dim):
    with pytest.raises(EmbeddingError, match="维度"):
        HashTestEmbedder(dim=dim)


def test_hash_embedder_is_deterministic_across_instances(media_files: tuple[Path, Path]):
    first = HashTestEmbedder()
    second = HashTestEmbedder()
    batch_a = first.encode_pairs([media_files])
    batch_b = second.encode_pairs([media_files])
    assert np.array_equal(batch_a.joint, batch_b.joint)
    assert np.array_equal(batch_a.audio, batch_b.audio)
    assert np.array_equal(batch_a.video, batch_b.video)


def test_hash_embedder_rows_are_normalized(media_files: tuple[Path, Path]):
    batch = HashTestEmbedder().encode_pairs([media_files, media_files])
    assert batch.joint.dtype == np.float32
    assert batch.joint.shape == (2, 32)
    for name, array in (("joint", batch.joint), ("audio", batch.audio), ("video", batch.video)):
        assert array is not None, name
        norms = np.linalg.norm(array, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), (name, norms)


def test_hash_embedder_audio_change_only_changes_audio_and_joint(
    media_files: tuple[Path, Path],
):
    video, audio = media_files
    embedder = HashTestEmbedder()
    before = embedder.encode_pairs([(video, audio)])

    audio.write_bytes(b"a completely different audio payload" * 64)
    after = embedder.encode_pairs([(video, audio)])

    assert np.array_equal(before.video, after.video), "音频内容变化不应影响视频单模态向量"
    assert not np.allclose(before.audio, after.audio), "音频内容变化应改变音频向量"
    assert not np.allclose(before.joint, after.joint), "联合向量必须同时依赖音频"


def test_hash_embedder_video_change_only_changes_video_and_joint(
    media_files: tuple[Path, Path],
):
    video, audio = media_files
    embedder = HashTestEmbedder()
    before = embedder.encode_pairs([(video, audio)])

    video.write_bytes(b"a completely different video payload" * 64)
    after = embedder.encode_pairs([(video, audio)])

    assert np.array_equal(before.audio, after.audio), "视频内容变化不应影响音频单模态向量"
    assert not np.allclose(before.video, after.video), "视频内容变化应改变视频向量"
    assert not np.allclose(before.joint, after.joint), "联合向量必须同时依赖视频"


def test_hash_embedder_missing_file_raises(tmp_path: Path, media_files: tuple[Path, Path]):
    video, audio = media_files
    with pytest.raises(EmbeddingError, match="不存在"):
        HashTestEmbedder().encode_pairs([(tmp_path / "missing.mp4", audio)])


def test_hash_embedder_empty_pairs_raises():
    with pytest.raises(EmbeddingError, match="空列表"):
        HashTestEmbedder().encode_pairs([])


def test_hash_embedder_close_is_idempotent():
    embedder = HashTestEmbedder()
    embedder.close()
    embedder.close()


# ---------------------------------------------------------------------------
# aggregate_work_vector
# ---------------------------------------------------------------------------


def _unit_vector(dim: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def test_aggregate_single_vector_unchanged():
    vector = _unit_vector()
    result = aggregate_work_vector(vector[None, :])
    assert result.dtype == np.float32
    assert result.shape == vector.shape
    assert np.allclose(result, vector, atol=1e-6)
    assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-6)


def test_aggregate_equal_vectors_unchanged():
    vector = _unit_vector(seed=3)
    result = aggregate_work_vector(np.stack([vector, vector, vector]))
    assert np.allclose(result, vector, atol=1e-6)


def test_aggregate_normalizes_rows_before_averaging():
    vector = _unit_vector(seed=5)
    rows = np.stack([vector, vector * 2.0, vector * 3.0])
    result = aggregate_work_vector(rows)
    assert np.allclose(result, vector, atol=1e-6)


def test_aggregate_two_orthogonal_vectors():
    rows = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    result = aggregate_work_vector(rows)
    expected = np.array([1.0, 1.0], dtype=np.float32) / np.sqrt(2.0)
    assert np.allclose(result, expected, atol=1e-6)
    assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-6)


def test_aggregate_is_deterministic():
    rows = np.stack([_unit_vector(seed=1), _unit_vector(seed=2)])
    assert np.array_equal(aggregate_work_vector(rows), aggregate_work_vector(rows))


def test_aggregate_zero_row_raises():
    rows = np.stack([_unit_vector(), np.zeros(8, dtype=np.float32)])
    with pytest.raises(EmbeddingError, match="范数"):
        aggregate_work_vector(rows)


def test_aggregate_all_zero_raises():
    with pytest.raises(EmbeddingError, match="范数"):
        aggregate_work_vector(np.zeros((2, 8), dtype=np.float32))


def test_aggregate_nan_raises():
    rows = np.stack([_unit_vector(), np.full(8, np.nan, dtype=np.float32)])
    with pytest.raises(EmbeddingError, match="NaN/Inf"):
        aggregate_work_vector(rows)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((0, 8), dtype=np.float32),
        np.zeros((3, 0), dtype=np.float32),
        np.zeros(8, dtype=np.float32),
        np.zeros((2, 2, 2), dtype=np.float32),
    ],
)
def test_aggregate_invalid_shape_raises(bad: np.ndarray):
    with pytest.raises(EmbeddingError):
        aggregate_work_vector(bad)


# ---------------------------------------------------------------------------
# save_vector / load_vector
# ---------------------------------------------------------------------------


def test_save_load_vector_roundtrip(tmp_path: Path):
    target = tmp_path / "emb" / "work_joint.npy"
    vector = _unit_vector(dim=16)
    save_vector(target, vector)

    assert target.is_file()
    loaded = load_vector(target)
    assert np.array_equal(loaded, vector)
    assert loaded.dtype == np.float32
    leftovers = [path.name for path in target.parent.iterdir() if path != target]
    assert leftovers == [], f"原子写入不应留下临时文件: {leftovers}"


def test_save_vector_overwrites_existing(tmp_path: Path):
    target = tmp_path / "vector.npy"
    save_vector(target, np.zeros(4, dtype=np.float32))
    save_vector(target, np.ones(4, dtype=np.float32))
    assert np.array_equal(load_vector(target), np.ones(4, dtype=np.float32))


def test_load_missing_vector_raises(tmp_path: Path):
    with pytest.raises(EmbeddingError, match="不存在"):
        load_vector(tmp_path / "missing.npy")


def test_load_corrupt_vector_raises(tmp_path: Path):
    broken = tmp_path / "broken.npy"
    broken.write_bytes(b"this is not a npy file")
    with pytest.raises(EmbeddingError, match="读取向量文件失败"):
        load_vector(broken)


# ---------------------------------------------------------------------------
# 懒加载与默认不加载模型
# ---------------------------------------------------------------------------


def test_embed_module_import_does_not_load_torch_or_transformers():
    """在干净子进程里 import 本模块，确认不会顺带导入 torch / transformers。"""

    code = (
        "import sys; import nebula.embed;"
        "print(int('torch' in sys.modules), int('transformers' in sys.modules))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "0 0", f"embed 模块导入不应加载重型依赖: {proc.stdout!r}"


def test_peav_embedder_init_is_lazy(repo_root: Path):
    config = load_config(repo_root / "config.yaml")
    embedder = PEAVEmbedder(config.model)

    assert isinstance(embedder, Embedder)
    assert embedder.version == config.model.version
    assert embedder.is_test_only is False
    # 构造完成时不应加载模型权重（_loaded 仍为 False）
    assert embedder._loaded is False
    embedder.close()
    embedder.close()


def test_peav_embedder_empty_pairs_raises_without_loading(repo_root: Path):
    config = load_config(repo_root / "config.yaml")
    embedder = PEAVEmbedder(config.model)
    try:
        with pytest.raises(EmbeddingError, match="空列表"):
            embedder.encode_pairs([])
        assert embedder._loaded is False
    finally:
        embedder.close()


def test_peav_embedder_missing_files_raise_without_loading(tmp_path: Path, repo_root: Path):
    config = load_config(repo_root / "config.yaml")
    embedder = PEAVEmbedder(config.model)
    try:
        with pytest.raises(EmbeddingError, match="不存在"):
            embedder.encode_pairs([(tmp_path / "a.mp4", tmp_path / "a.wav")])
        assert embedder._loaded is False
    finally:
        embedder.close()


# ---------------------------------------------------------------------------
# PE-AV 真实模型测试（默认跳过）
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def peav_embedder(repo_root: Path):
    """加载一次真实模型供所有 model 测试复用（默认 skip，不会真的加载）。"""

    if not MODEL_TESTS_ENABLED:
        pytest.skip(MODEL_SKIP_REASON)
    config = load_config(repo_root / "config.yaml")
    embedder = PEAVEmbedder(config.model)
    try:
        yield embedder
    finally:
        embedder.close()


def _extract_two_clips(sample_video: Path, config, out_dir: Path) -> list:
    """把 12 秒测试视频按 4 秒切成 2 段，返回 ExtractedClip 列表。"""

    info = probe_media(sample_video, ffprobe=config.media.ffprobe)
    plans = plan_clips(info.duration_s, clip_seconds=4.0, clips_per_work=2)
    assert len(plans) == 2
    return [extract_clip(sample_video, plan, out_dir, media=config.media) for plan in plans]


def _pair(clips) -> list:
    return [(clip.video_path, clip.audio_path) for clip in clips]


@pytest.mark.model
@pytest.mark.skipif(not MODEL_TESTS_ENABLED, reason=MODEL_SKIP_REASON)
def test_peav_real_batch_shapes_and_norms(sample_video: Path, tmp_path: Path, repo_root: Path, peav_embedder: PEAVEmbedder):
    config = load_config(repo_root / "config.yaml")
    clips = _extract_two_clips(sample_video, config, tmp_path / "clips")

    batch = peav_embedder.encode_pairs(_pair(clips))

    assert isinstance(batch, EmbeddingBatch)
    assert batch.joint.ndim == 2
    assert batch.joint.shape[0] == 2
    assert batch.audio is not None and batch.video is not None
    assert batch.audio.shape == batch.joint.shape
    assert batch.video.shape == batch.joint.shape
    assert batch.joint.dtype == np.float32
    for name, array in (("joint", batch.joint), ("audio", batch.audio), ("video", batch.video)):
        norms = np.linalg.norm(array, axis=1)
        print(f"[peav] {name}: shape={array.shape} dtype={array.dtype} norms={norms.tolist()}")
        assert np.allclose(norms, 1.0, atol=1e-4), (name, norms)


@pytest.mark.model
@pytest.mark.skipif(not MODEL_TESTS_ENABLED, reason=MODEL_SKIP_REASON)
def test_peav_real_repeat_and_batch_deterministic(
    sample_video: Path, tmp_path: Path, repo_root: Path, peav_embedder: PEAVEmbedder, monkeypatch
):
    config = load_config(repo_root / "config.yaml")
    clips = _extract_two_clips(sample_video, config, tmp_path / "clips")
    pairs = _pair(clips)

    first = peav_embedder.encode_pairs(pairs)
    second = peav_embedder.encode_pairs(pairs)
    repeat_diff = float(np.abs(first.joint - second.joint).max())
    print(f"[peav] 重复运行 joint 最大绝对差: {repeat_diff:.3e}")
    assert repeat_diff <= 1e-5, "同一输入重复编码应逐元素一致"

    # 把 batch_size 临时改成 1，验证分批逻辑与单批结果一致
    monkeypatch.setattr(peav_embedder, "_batch_size", 1)
    chunked = peav_embedder.encode_pairs(pairs)
    chunk_diff = float(np.abs(first.joint - chunked.joint).max())
    chunk_cosine_min = min(
        float(np.dot(first.joint[row], chunked.joint[row])) for row in range(first.joint.shape[0])
    )
    print(
        f"[peav] batch_size=1 vs 默认 batch 的 joint 最大绝对差: {chunk_diff:.3e}, "
        f"最小余弦: {chunk_cosine_min:.6f}"
    )
    # bfloat16 autocast 的 GPU kernel 会带来 1e-3 量级的逐元素差，用余弦约束语义一致性
    tolerance = 5e-3 if getattr(peav_embedder, "_use_autocast", False) else 1e-5
    assert chunk_diff <= tolerance, "分批编码结果应与单批编码一致"
    assert chunk_cosine_min >= 0.9999, "分批编码结果的余弦相似度应接近 1"


@pytest.mark.model
@pytest.mark.skipif(not MODEL_TESTS_ENABLED, reason=MODEL_SKIP_REASON)
def test_peav_real_ablation_muted_and_black_change_joint(
    sample_video: Path, tmp_path: Path, repo_root: Path, peav_embedder: PEAVEmbedder
):
    """消融实验：静音 / 纯黑后联合向量必须明显变化（排查只看单一模态）。"""

    config = load_config(repo_root / "config.yaml")
    info = probe_media(sample_video, ffprobe=config.media.ffprobe)
    plans = plan_clips(info.duration_s, clip_seconds=4.0, clips_per_work=2)
    clip = extract_clip(sample_video, plans[0], tmp_path / "clips", media=config.media)

    muted_audio = mute_audio_clip(clip.duration_s, tmp_path / "muted.wav", media=config.media)
    black_video = black_video_clip(
        clip.duration_s, tmp_path / "black.mp4", media=config.media
    )

    original = peav_embedder.encode_pairs([(clip.video_path, clip.audio_path)])
    muted = peav_embedder.encode_pairs([(clip.video_path, muted_audio)])
    black = peav_embedder.encode_pairs([(black_video, clip.audio_path)])

    def cosine(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.dot(left.reshape(-1), right.reshape(-1)))

    joint_muted = cosine(original.joint[0], muted.joint[0])
    joint_black = cosine(original.joint[0], black.joint[0])
    audio_muted = cosine(original.audio[0], muted.audio[0])
    video_black = cosine(original.video[0], black.video[0])
    print(
        "[peav] 消融余弦: "
        f"joint(原,静音)={joint_muted:.4f} "
        f"joint(原,纯黑)={joint_black:.4f} "
        f"audio(原,静音)={audio_muted:.4f} "
        f"video(原,纯黑)={video_black:.4f}"
    )

    # 只换音频时：视频向量应保持不变，音频与联合向量必须变化。
    # bfloat16 autocast 下不同 batch 组成会带来 1e-4 量级的合法数值差，放宽到 5e-3。
    inert_tolerance = 5e-3 if getattr(peav_embedder, "_use_autocast", False) else 1e-5
    assert np.all(np.abs(original.video[0] - muted.video[0]) <= inert_tolerance)
    assert audio_muted < 0.99
    # 只换画面时：音频向量应保持不变，视频与联合向量必须变化
    assert np.all(np.abs(original.audio[0] - black.audio[0]) <= inert_tolerance)
    assert video_black < 0.99
    # 联合向量同时受两个模态影响：任一消融都应造成明显下降
    assert joint_muted < 0.99, f"静音后联合向量几乎不变（cos={joint_muted:.4f}）"
    assert joint_black < 0.99, f"纯黑后联合向量几乎不变（cos={joint_black:.4f}）"
