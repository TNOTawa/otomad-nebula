"""测试用合成视频工厂（离线、确定性、带缓存）。

本模块用本机 ffmpeg 现场合成一组规格固定的测试视频，供端到端测试与手工验收使用：

- 不同时长（4s / 12s / 40s / 17s / 6s / 0.5s），覆盖不同的采样片段数；
- 不同音高（220 / 330 / 440 / 660 / 880 Hz）与不同画面（testsrc2 / smptebars / rgbtestsrc）；
- 无音轨视频，用于验证流水线会补静音或正确记录 has_audio；
- 坏文件（随机字节、空文件），用于验证失败路径不会阻塞其他作品。

生成产物默认缓存到 tests/fixtures/out/（该目录自带 .gitignore），首次生成后复用，
避免每次跑测试都重新编码。所有生成只依赖本机 ffmpeg，不访问网络、不加载模型。

注意：本模块只负责“造数据”，不包含任何业务断言；E2E 测试应使用这里的素材，
但断言逻辑由测试自己（独立）编写。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from nebula.config import Config, load_config
from nebula.db import Catalog

# ──────────────────────────────────────────────────────────────────────────────
# 路径与常量
# ──────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
"""项目根目录（tests/fixtures/video_factory.py -> 上两级）。"""

OUT_DIR = Path(__file__).resolve().parent / "out"
"""默认缓存目录，已被 .gitignore 忽略。"""

FACTORY_VERSION = 1
"""工厂清单版本；缓存清单不匹配时会整体重建，避免旧素材误导测试。"""


@dataclass(frozen=True)
class VideoSpec:
    """一条合成视频的固定规格。"""

    name: str
    duration_s: float
    pattern: str
    freq_hz: int | None
    size: str = "160x120"
    fps: int = 10
    description: str = ""


VIDEO_SPECS: tuple[VideoSpec, ...] = (
    VideoSpec(
        name="short_4s.mp4",
        duration_s=4.0,
        pattern="testsrc2",
        freq_hz=440,
        description="4 秒短视频：不足一个 8 秒片段，整段作为唯一片段",
    ),
    VideoSpec(
        name="mid_12s.mp4",
        duration_s=12.0,
        pattern="testsrc2",
        freq_hz=880,
        description="12 秒视频：按 8 秒规则只切 1 段",
    ),
    VideoSpec(
        name="long_40s.mp4",
        duration_s=40.0,
        pattern="smptebars",
        freq_hz=220,
        description="40 秒视频：可切满 5 段，覆盖 clips_per_work 上限",
    ),
    VideoSpec(
        name="odd_17s.mp4",
        duration_s=17.0,
        pattern="rgbtestsrc",
        freq_hz=660,
        description="17 秒视频：切 2 段，边界不在整数秒",
    ),
    VideoSpec(
        name="silent_6s.mp4",
        duration_s=6.0,
        pattern="rgbtestsrc",
        freq_hz=None,
        description="6 秒无音轨视频：验证无音轨时的处理路径",
    ),
    VideoSpec(
        name="tiny_05s.mp4",
        duration_s=0.5,
        pattern="testsrc2",
        freq_hz=330,
        description="0.5 秒视频：短于 min_duration_seconds，应被记为失败",
    ),
)

GOOD_VIDEO_NAMES: tuple[str, ...] = tuple(spec.name for spec in VIDEO_SPECS)
"""全部“可正常探测”的视频名。"""

BAD_VIDEO_NAMES: tuple[str, ...] = ("corrupt.mp4", "empty.mp4")
"""故意损坏的文件名。"""

ALL_VIDEO_NAMES: tuple[str, ...] = GOOD_VIDEO_NAMES + BAD_VIDEO_NAMES


# ──────────────────────────────────────────────────────────────────────────────
# ffmpeg / ffprobe 可用性
# ──────────────────────────────────────────────────────────────────────────────


def media_tools_available(ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> bool:
    """本机是否同时具备 ffmpeg 与 ffprobe。"""

    return shutil.which(ffmpeg) is not None and shutil.which(ffprobe) is not None


def _ffmpeg_version(ffmpeg: str) -> str:
    proc = subprocess.run(
        [ffmpeg, "-version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    first_line = (proc.stdout or "").splitlines()
    return first_line[0].strip() if first_line else "unknown"


def _probe_json(path: Path, ffprobe: str = "ffprobe") -> dict[str, Any]:
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe 无法解析生成的视频 {path}: {proc.stderr.strip()[:500]}")
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:  # pragma: no cover - 正常情况下不会触发
        raise RuntimeError(f"ffprobe 输出不是合法 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"ffprobe 输出格式异常: {path}")
    return payload


def probe_duration_s(path: Path, ffprobe: str = "ffprobe") -> float:
    """读取视频容器时长（秒）；用于工厂自检与测试辅助。"""

    payload = _probe_json(path, ffprobe)
    fmt = payload.get("format")
    if isinstance(fmt, dict) and fmt.get("duration") is not None:
        return float(fmt["duration"])
    for stream in payload.get("streams", []):
        if stream.get("duration") is not None:
            return float(stream["duration"])
    raise RuntimeError(f"无法从 ffprobe 结果中读取时长: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 合成视频
# ──────────────────────────────────────────────────────────────────────────────


def _video_filter(spec: VideoSpec) -> str:
    return f"{spec.pattern}=size={spec.size}:rate={spec.fps}:duration={spec.duration_s:g}"


def _audio_filter(spec: VideoSpec) -> str:
    assert spec.freq_hz is not None
    return f"sine=frequency={spec.freq_hz}:sample_rate=48000:duration={spec.duration_s:g}"


def _build_one(spec: VideoSpec, target: Path, *, ffmpeg: str, ffprobe: str) -> Path:
    """调用 ffmpeg 生成一条视频，先写临时文件再原子替换。"""

    partial = target.with_name(f"{target.stem}.{os.getpid()}.partial{target.suffix}")
    partial.unlink(missing_ok=True)
    command: list[str] = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        _video_filter(spec),
    ]
    if spec.freq_hz is not None:
        command += ["-f", "lavfi", "-i", _audio_filter(spec)]
    command += [
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
    ]
    if spec.freq_hz is not None:
        command += ["-c:a", "aac", "-ar", "48000", "-ac", "2"]
    command += ["-movflags", "+faststart", str(partial)]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg 生成 {spec.name} 失败 (exit {proc.returncode}): "
                f"{(proc.stderr or '').strip()[:800]}"
            )
        if not partial.is_file() or partial.stat().st_size <= 0:
            raise RuntimeError(f"ffmpeg 未生成有效文件: {spec.name}")
        # 自检：时长期望误差 0.25s 以内，否则说明素材不可信
        actual = probe_duration_s(partial, ffprobe)
        tolerance = 0.25
        if abs(actual - spec.duration_s) > tolerance:
            raise RuntimeError(
                f"生成视频时长与规格不符: {spec.name} 期望 {spec.duration_s}s，实际 {actual:.3f}s"
            )
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def _build_bad_files(out_dir: Path) -> dict[str, Path]:
    """生成两个确定性的坏文件（不是视频）。"""

    corrupt = out_dir / "corrupt.mp4"
    payload = (b"THIS_IS_NOT_A_VIDEO_" * 256)[:4096]
    if not corrupt.is_file() or corrupt.stat().st_size != len(payload):
        corrupt.write_bytes(payload)
    empty = out_dir / "empty.mp4"
    if not empty.is_file():
        empty.write_bytes(b"")
    return {"corrupt.mp4": corrupt, "empty.mp4": empty}


def _manifest_payload(ffmpeg_version: str) -> dict[str, Any]:
    return {
        "factory_version": FACTORY_VERSION,
        "ffmpeg_version": ffmpeg_version,
        "specs": [
            {
                "name": spec.name,
                "duration_s": spec.duration_s,
                "pattern": spec.pattern,
                "freq_hz": spec.freq_hz,
                "size": spec.size,
                "fps": spec.fps,
            }
            for spec in VIDEO_SPECS
        ],
        "bad_files": list(BAD_VIDEO_NAMES),
    }


def _read_manifest(out_dir: Path) -> dict[str, Any] | None:
    path = out_dir / "manifest.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _cache_is_valid(out_dir: Path, expected: Mapping[str, Path]) -> bool:
    return all(path.is_file() for path in expected.values())


def build_video_library(
    out_dir: Path | None = None,
    *,
    names: Sequence[str] | None = None,
    include_bad: bool = True,
    force: bool = False,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> dict[str, Path]:
    """生成（或复用缓存）一批测试视频，返回 {名字: 绝对路径}。

    参数
    ----
    out_dir:
        缓存目录，默认 tests/fixtures/out。
    names:
        只生成指定的视频名（必须是 GOOD_VIDEO_NAMES 的子集）；省略时生成全部。
    include_bad:
        是否同时生成坏文件。
    force:
        为 True 时忽略缓存强制重建。
    ffmpeg / ffprobe:
        工具名或绝对路径。
    """

    if not media_tools_available(ffmpeg, ffprobe):
        raise RuntimeError("缺少 ffmpeg/ffprobe，无法生成测试视频；请确认它们在 PATH 中")

    target_dir = Path(out_dir) if out_dir is not None else OUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    if names is None:
        selected = list(VIDEO_SPECS)
    else:
        wanted = set(names)
        unknown = wanted - set(GOOD_VIDEO_NAMES)
        if unknown:
            raise ValueError(f"未知的测试视频名: {', '.join(sorted(unknown))}")
        selected = [spec for spec in VIDEO_SPECS if spec.name in wanted]

    ffmpeg_version = _ffmpeg_version(ffmpeg)
    manifest_path = target_dir / "manifest.json"
    manifest = _manifest_payload(ffmpeg_version)
    cached_manifest = _read_manifest(target_dir)

    result: dict[str, Path] = {spec.name: target_dir / spec.name for spec in selected}
    if include_bad:
        result.update(
            {
                "corrupt.mp4": target_dir / "corrupt.mp4",
                "empty.mp4": target_dir / "empty.mp4",
            }
        )

    cache_ok = (
        not force
        and cached_manifest == manifest
        and _cache_is_valid(target_dir, result)
    )
    if cache_ok:
        return result

    for spec in selected:
        _build_one(spec, target_dir / spec.name, ffmpeg=ffmpeg, ffprobe=ffprobe)
    if include_bad:
        _build_bad_files(target_dir)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def install_video_library(
    input_local: Path,
    *,
    layout: Mapping[str, str] | None = None,
    names: Sequence[str] | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Path]:
    """把缓存视频复制到测试用的 input/local 目录。

    参数
    ----
    input_local:
        目标目录，通常是临时配置里的 cfg.paths.input_local。
    layout:
        {视频名: 相对目标路径}，用于构造子目录场景；省略时按视频名平铺。
    names:
        只复制指定视频，省略则复制全部正常视频（不含坏文件）。
    cache_dir:
        视频缓存目录，默认 tests/fixtures/out。

    返回
    ----
    {视频名: 复制后的绝对路径}。
    """

    target_dir = Path(input_local)
    target_dir.mkdir(parents=True, exist_ok=True)
    source = build_video_library(cache_dir)
    chosen = list(names) if names is not None else list(GOOD_VIDEO_NAMES)
    unknown = set(chosen) - set(source)
    if unknown:
        raise ValueError(f"未知的测试视频名: {', '.join(sorted(unknown))}")
    installed: dict[str, Path] = {}
    for name in chosen:
        rel = layout[name] if layout is not None and name in layout else name
        destination = target_dir / Path(rel)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source[name], destination)
        installed[name] = destination
    return installed


# ──────────────────────────────────────────────────────────────────────────────
# 临时配置与数据库
# ──────────────────────────────────────────────────────────────────────────────


def _deep_merge(base: dict[str, Any], patch: Mapping[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def create_temp_config(
    tmp_path: Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    base_config: Path | None = None,
) -> Config:
    """在 tmp_path 下生成一份隔离的临时配置并加载。

    路径（input/data/temp/logs/模型缓存）全部改到 tmp_path 内，其他参数复制
    项目根目录的 config.yaml；overrides 支持按段深合并，例如::

        create_temp_config(tmp_path, overrides={"media": {"clip_seconds": 2.0}})
    """

    base = Path(base_config) if base_config is not None else REPO_ROOT / "config.yaml"
    data: dict[str, Any] = yaml.safe_load(base.read_text(encoding="utf-8"))
    if not isinstance(data, dict):  # pragma: no cover - 仓库配置损坏时才会触发
        raise RuntimeError(f"基础配置不是映射: {base}")

    root = tmp_path.resolve()
    data["paths"] = {
        "input_local": str(root / "input" / "local"),
        "input_lists": str(root / "input" / "lists"),
        "data": str(root / "data"),
        "temp": str(root / "temp"),
        "logs": str(root / "logs"),
    }
    data.setdefault("model", {})["cache_dir"] = str(root / "cache" / "hf")
    if overrides:
        _deep_merge(data, overrides)
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    cfg = load_config(config_path)
    cfg.ensure_directories()
    return cfg


def create_temp_catalog(config: Config) -> Catalog:
    """创建并初始化一份临时 SQLite 编目；调用方负责关闭。"""

    catalog = Catalog(config.paths.db_path)
    catalog.initialize()
    return catalog
