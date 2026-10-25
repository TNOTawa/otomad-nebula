from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nebula.config import load_config
from nebula.errors import ConfigError


def test_repo_config_loads(repo_root: Path):
    cfg = load_config(repo_root / "config.yaml")
    assert cfg.root == repo_root
    assert cfg.media.clip_seconds == 8.0
    assert cfg.media.clips_per_work == 5
    assert cfg.model.checkpoint == "facebook/pe-av-small-16-frame"
    assert cfg.umap.metric == "cosine"
    assert cfg.umap.random_state == 42
    assert cfg.umap.dimensions == (2, 3)
    assert cfg.paths.db_path == (repo_root / "data" / "catalog.sqlite").resolve()


def test_embedding_version_encodes_model_and_sampling(config):
    version = config.embedding_version
    assert version.startswith("peav-small-16f__f16_clip8s_x5_48k")


def test_ensure_directories(config):
    config.ensure_directories()
    for path in (
        config.paths.input_local,
        config.paths.input_lists,
        config.paths.data,
        config.paths.temp,
        config.paths.logs,
        config.paths.embeddings,
        config.paths.maps,
        config.model.cache_dir,
    ):
        assert path.is_dir()


def test_unknown_top_level_key_rejected(tmp_path: Path, repo_root: Path):
    data = yaml.safe_load((repo_root / "config.yaml").read_text(encoding="utf-8"))
    data["typo_section"] = {}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError, match="未知字段"):
        load_config(path)


def test_unknown_section_key_rejected(tmp_path: Path, repo_root: Path):
    data = yaml.safe_load((repo_root / "config.yaml").read_text(encoding="utf-8"))
    data["media"]["clip_second"] = 8.0
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError, match="未知字段"):
        load_config(path)


def test_missing_config_file(tmp_path: Path):
    with pytest.raises(ConfigError, match="找不到配置文件"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_umap_dimensions(tmp_path: Path, repo_root: Path):
    data = yaml.safe_load((repo_root / "config.yaml").read_text(encoding="utf-8"))
    data["umap"]["dimensions"] = [2, 4]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError, match="umap.dimensions"):
        load_config(path)


def test_data_relpath_roundtrip(config):
    target = config.paths.data / "embeddings" / "v1" / "000001" / "work_joint.npy"
    rel = config.data_relpath(target)
    assert rel == "embeddings/v1/000001/work_joint.npy"
    assert config.data_path(rel) == target.resolve()


def test_paths_resolve_relative_to_config(tmp_path: Path, repo_root: Path):
    data = yaml.safe_load((repo_root / "config.yaml").read_text(encoding="utf-8"))
    data["paths"]["data"] = "rel-data"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.paths.data == (tmp_path / "rel-data").resolve()
