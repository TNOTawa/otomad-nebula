from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from nebula.config import load_config
from nebula.db import Catalog

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def config(tmp_path: Path):
    data = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    data["paths"] = {
        "input_local": str(tmp_path / "input" / "local"),
        "input_lists": str(tmp_path / "input" / "lists"),
        "data": str(tmp_path / "data"),
        "temp": str(tmp_path / "temp"),
        "logs": str(tmp_path / "logs"),
    }
    data["model"]["cache_dir"] = str(tmp_path / "cache" / "hf")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    cfg = load_config(config_path)
    cfg.ensure_directories()
    return cfg


@pytest.fixture
def catalog(config):
    cat = Catalog(config.paths.db_path)
    cat.initialize()
    yield cat
    cat.close()


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """用 ffmpeg 生成一段 12 秒、含音轨的测试视频（testsrc2 + 440Hz 正弦）。"""

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("缺少 ffmpeg")
    target = tmp_path / "sample.mp4"
    import subprocess

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=10:duration=12",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=12",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(target),
        ],
        check=True,
        capture_output=True,
    )
    return target
