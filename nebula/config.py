"""配置加载与路径解析。

配置文件是所有运行参数的唯一来源（见项目根目录 config.yaml）。
所有相对路径都相对于配置文件所在目录解析。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import ConfigError
from .util import slugify

_TOP_LEVEL_KEYS = {
    "project",
    "paths",
    "database",
    "media",
    "sampling",
    "events",
    "temporal",
    "model",
    "umap",
    "download",
    "bilibili",
    "app",
    "logging",
}


@dataclass(frozen=True)
class PathsConfig:
    input_local: Path
    input_lists: Path
    data: Path
    temp: Path
    logs: Path
    db_path: Path
    embeddings: Path
    representations: Path
    maps: Path


@dataclass(frozen=True)
class MediaConfig:
    ffmpeg: str
    ffprobe: str
    clip_seconds: float
    clips_per_work: int
    min_duration_seconds: float
    audio_sample_rate: int
    audio_channels: int
    video_crf: int
    video_preset: str
    extract_timeout_seconds: float
    probe_timeout_seconds: float
    clip_height: int | None = None

    long_video_seconds: float = 3600.0


@dataclass(frozen=True)
class SamplingConfig:
    """全片采样模式与重叠参数。"""

    mode: str
    overlap_seconds: float


@dataclass(frozen=True)
class EventsConfig:
    """v0.4.1 高频音画事件扫描参数。"""

    enabled: bool
    audio_rate_hz: int
    video_fps: int
    version: str


@dataclass(frozen=True)
class TemporalConfig:
    """v0.4.2 固定时间结构表征参数。"""

    stages: int


@dataclass(frozen=True)
class ModelConfig:
    checkpoint: str
    revision: str | None
    version: str
    frames: int
    device: str
    dtype: str
    batch_size: int
    cache_dir: Path
    local_files_only: bool


@dataclass(frozen=True)
class UmapConfig:
    metric: str
    n_neighbors: int
    min_dist: float
    random_state: int
    n_jobs: int
    n_epochs: int | None
    dimensions: tuple[int, ...]


@dataclass(frozen=True)
class DownloadConfig:
    yt_dlp_format: str
    max_height: int
    cookies_file: Path | None
    proxy: str | None
    retries: int


@dataclass(frozen=True)
class BilibiliConfig:
    api_timeout_seconds: float
    user_agent: str


@dataclass(frozen=True)
class AppConfig:
    title: str
    point_size: int
    opacity: float
    density_grid: int


@dataclass(frozen=True)
class LoggingConfig:
    level: str


@dataclass(frozen=True)
class Config:
    """一次运行所需的全部配置。"""

    path: Path
    root: Path
    project_name: str
    paths: PathsConfig
    media: MediaConfig
    sampling: SamplingConfig
    events: EventsConfig
    temporal: TemporalConfig
    model: ModelConfig
    umap: UmapConfig
    download: DownloadConfig
    bilibili: BilibiliConfig
    app: AppConfig
    logging: LoggingConfig

    @property
    def legacy_embedding_version(self) -> str:
        """旧版“均匀抽 ≤N 段”序列版本号（v0.3 及以前的 5 片段平均）。"""

        clip = f"{self.media.clip_seconds:g}"
        version = (
            f"{slugify(self.model.version)}"
            f"__f{self.model.frames}"
            f"_clip{clip}s"
            f"_x{self.media.clips_per_work}"
            f"_{self.media.audio_sample_rate // 1000}k"
        )
        if self.media.clip_height:
            version += f"_h{int(self.media.clip_height)}"
        return version

    @property
    def full_sequence_version(self) -> str:
        """全片覆盖序列版本号（v0.4）：窗口采样参数 + 可选切片高度。"""

        clip = f"{self.media.clip_seconds:g}"
        overlap = f"{self.sampling.overlap_seconds:g}"
        version = (
            f"{slugify(self.model.version)}"
            f"__f{self.model.frames}"
            f"_full{clip}s"
            f"_ov{overlap}"
            f"_{self.media.audio_sample_rate // 1000}k"
        )
        if self.media.clip_height:
            version += f"_h{int(self.media.clip_height)}"
        return version

    @property
    def embedding_version(self) -> str:
        """当前激活的序列版本号（由 sampling.mode 决定）。

        一旦模型/采样/预处理参数变化，版本号随之变化，旧向量不会被覆盖
        （对应方案中“新的向量版本保存，而不是覆盖旧版本”的要求）。
        """

        if self.sampling.mode == "full_coverage":
            return self.full_sequence_version
        return self.legacy_embedding_version

    def ensure_directories(self) -> None:
        for directory in (
            self.paths.input_local,
            self.paths.input_lists,
            self.paths.data,
            self.paths.temp,
            self.paths.logs,
            self.paths.embeddings,
            self.paths.representations,
            self.paths.maps,
            self.model.cache_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.paths.db_path.parent.mkdir(parents=True, exist_ok=True)

    def data_path(self, relative: str | os.PathLike[str]) -> Path:
        return (self.paths.data / Path(relative)).resolve()

    def data_relpath(self, path: str | Path) -> str:
        """把 data 目录内的绝对路径转成相对路径（POSIX 风格，便于跨平台存储）。"""

        resolved = Path(path).resolve()
        try:
            return resolved.relative_to(self.paths.data).as_posix()
        except ValueError as exc:
            raise ConfigError(f"路径不在 data 目录内: {resolved}") from exc

    def local_work_path(self, rel_path: str) -> Path:
        return (self.paths.input_local / Path(rel_path)).resolve()


def _fail(message: str) -> None:
    raise ConfigError(message)


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name)
    if value is None:
        _fail(f"配置缺少 [{name}] 段")
    if not isinstance(value, Mapping):
        _fail(f"配置 [{name}] 段必须是映射")
    return value


def _check_keys(section: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(section) - allowed
    if unknown:
        _fail(f"配置 [{where}] 存在未知字段: {', '.join(sorted(map(str, unknown)))}")


def _req(section: Mapping[str, Any], key: str, where: str, kind: type | tuple[type, ...]) -> Any:
    if key not in section:
        _fail(f"配置 [{where}] 缺少字段: {key}")
    value = section[key]
    if not isinstance(value, kind):
        _fail(f"配置 [{where}.{key}] 类型错误: 需要 {kind}, 收到 {type(value).__name__}")
    return value


def _opt(section: Mapping[str, Any], key: str, where: str, kind: type | tuple[type, ...]) -> Any:
    if key not in section or section[key] is None:
        return None
    value = section[key]
    if not isinstance(value, kind):
        _fail(f"配置 [{where}.{key}] 类型错误: 需要 {kind}, 收到 {type(value).__name__}")
    return value


def _path(root: Path, value: Any, where: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        _fail(f"配置 [{where}] 必须是非空路径字符串")
    expanded = Path(os.path.expanduser(value))
    resolved = expanded if expanded.is_absolute() else root / expanded
    return resolved.resolve()


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        _fail(f"无法读取配置文件 {path}: {exc}")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        _fail(f"配置文件 YAML 解析失败: {exc}")
    if data is None:
        _fail(f"配置文件为空: {path}")
    if not isinstance(data, Mapping):
        _fail(f"配置文件顶层必须是映射: {path}")
    return data


def load_config(path: str | Path | None = None, *, root: str | Path | None = None) -> Config:
    """加载并校验配置。

    参数
    ----
    path:
        配置文件路径；省略时使用 root/config.yaml（root 省略时使用当前目录）。
    root:
        项目根目录；仅在 path 省略时使用。
    """

    if path is None:
        base = Path(root) if root is not None else Path.cwd()
        config_path = (base / "config.yaml").resolve()
    else:
        config_path = Path(path).resolve()
    project_root = config_path.parent

    if not config_path.is_file():
        _fail(f"找不到配置文件: {config_path}")

    data = _load_yaml(config_path)
    _check_keys(data, _TOP_LEVEL_KEYS, "顶层")

    project = _section(data, "project")
    _check_keys(project, {"name"}, "project")
    project_name = _req(project, "name", "project", str)

    paths_section = _section(data, "paths")
    _check_keys(
        paths_section,
        {"input_local", "input_lists", "data", "temp", "logs"},
        "paths",
    )
    input_local = _path(project_root, _req(paths_section, "input_local", "paths", str), "paths.input_local")
    input_lists = _path(project_root, _req(paths_section, "input_lists", "paths", str), "paths.input_lists")
    data_dir = _path(project_root, _req(paths_section, "data", "paths", str), "paths.data")
    temp_dir = _path(project_root, _req(paths_section, "temp", "paths", str), "paths.temp")
    logs_dir = _path(project_root, _req(paths_section, "logs", "paths", str), "paths.logs")

    database = _section(data, "database")
    _check_keys(database, {"filename"}, "database")
    db_filename = _req(database, "filename", "database", str)
    if "/" in db_filename or "\\" in db_filename:
        _fail("database.filename 不能包含目录分隔符，数据目录由 paths.data 决定")
    db_path = (data_dir / db_filename).resolve()

    media = _section(data, "media")
    _check_keys(
        media,
        {
            "ffmpeg",
            "ffprobe",
            "clip_seconds",
            "clips_per_work",
            "min_duration_seconds",
            "audio_sample_rate",
            "audio_channels",
            "video_crf",
            "video_preset",
            "extract_timeout_seconds",
            "probe_timeout_seconds",
            "clip_height",
            "long_video_seconds",
        },
        "media",
    )
    clip_seconds = float(_req(media, "clip_seconds", "media", (int, float)))
    clips_per_work = int(_req(media, "clips_per_work", "media", int))
    min_duration = float(_req(media, "min_duration_seconds", "media", (int, float)))
    sample_rate = int(_req(media, "audio_sample_rate", "media", int))
    audio_channels = int(_req(media, "audio_channels", "media", int))
    video_crf = int(_req(media, "video_crf", "media", int))
    extract_timeout = float(_req(media, "extract_timeout_seconds", "media", (int, float)))
    probe_timeout = float(_req(media, "probe_timeout_seconds", "media", (int, float)))
    if clip_seconds <= 0:
        _fail("media.clip_seconds 必须大于 0")
    if clips_per_work < 1:
        _fail("media.clips_per_work 必须 >= 1")
    if min_duration <= 0:
        _fail("media.min_duration_seconds 必须大于 0")
    if sample_rate < 8000:
        _fail("media.audio_sample_rate 过小")
    if audio_channels not in (1, 2):
        _fail("media.audio_channels 只支持 1 或 2")
    if not 0 <= video_crf <= 51:
        _fail("media.video_crf 必须在 0..51")
    if extract_timeout <= 0 or probe_timeout <= 0:
        _fail("media 超时参数必须大于 0")
    clip_height_raw = _opt(media, "clip_height", "media", int)
    if clip_height_raw is not None and clip_height_raw < 64:
        _fail("media.clip_height 必须是 >=64 的整数或 null")
    long_video_seconds = float(
        _opt(media, "long_video_seconds", "media", (int, float)) or 3600.0
    )
    if long_video_seconds <= 0:
        _fail("media.long_video_seconds 必须大于 0")

    sampling = _section(data, "sampling")
    _check_keys(sampling, {"mode", "overlap_seconds"}, "sampling")
    sampling_mode = _req(sampling, "mode", "sampling", str)
    if sampling_mode not in ("legacy", "full_coverage"):
        _fail("sampling.mode 必须是 legacy 或 full_coverage")
    overlap_seconds = float(_req(sampling, "overlap_seconds", "sampling", (int, float)))
    if overlap_seconds < 0:
        _fail("sampling.overlap_seconds 不能为负")
    if sampling_mode == "full_coverage" and overlap_seconds >= clip_seconds:
        _fail("sampling.overlap_seconds 必须小于 media.clip_seconds")

    events = _section(data, "events")
    _check_keys(events, {"enabled", "audio_rate_hz", "video_fps", "version"}, "events")
    events_enabled = bool(_req(events, "enabled", "events", bool))
    audio_rate_hz = int(_req(events, "audio_rate_hz", "events", int))
    video_fps = int(_req(events, "video_fps", "events", int))
    event_version = _req(events, "version", "events", str)
    if not 20 <= audio_rate_hz <= 1000:
        _fail("events.audio_rate_hz 必须在 20..1000")
    if not 1 <= video_fps <= 120:
        _fail("events.video_fps 必须在 1..120")
    if not event_version.strip():
        _fail("events.version 不能为空")

    temporal = _section(data, "temporal")
    _check_keys(temporal, {"stages"}, "temporal")
    stages = int(_req(temporal, "stages", "temporal", int))
    if not 2 <= stages <= 8:
        _fail("temporal.stages 必须在 2..8")

    model = _section(data, "model")
    _check_keys(
        model,
        {
            "checkpoint",
            "revision",
            "version",
            "frames",
            "device",
            "dtype",
            "batch_size",
            "cache_dir",
            "local_files_only",
        },
        "model",
    )
    device = _req(model, "device", "model", str)
    if device not in ("auto", "cpu", "cuda"):
        _fail("model.device 必须是 auto/cpu/cuda")
    dtype = _req(model, "dtype", "model", str)
    if dtype not in ("auto", "float32", "bfloat16"):
        _fail("model.dtype 必须是 auto/float32/bfloat16")
    frames = int(_req(model, "frames", "model", int))
    batch_size = int(_req(model, "batch_size", "model", int))
    if frames < 1:
        _fail("model.frames 必须 >= 1")
    if batch_size < 1:
        _fail("model.batch_size 必须 >= 1")
    revision = _opt(model, "revision", "model", str)

    umap = _section(data, "umap")
    _check_keys(
        umap,
        {"metric", "n_neighbors", "min_dist", "random_state", "n_jobs", "n_epochs", "dimensions"},
        "umap",
    )
    n_neighbors = int(_req(umap, "n_neighbors", "umap", int))
    min_dist = float(_req(umap, "min_dist", "umap", (int, float)))
    random_state = int(_req(umap, "random_state", "umap", int))
    n_jobs = int(_req(umap, "n_jobs", "umap", int))
    n_epochs = _opt(umap, "n_epochs", "umap", int)
    dimensions_raw = _req(umap, "dimensions", "umap", list)
    dimensions = tuple(sorted({int(v) for v in dimensions_raw}))
    if not dimensions or any(d not in (2, 3) for d in dimensions):
        _fail("umap.dimensions 只支持 2 和/或 3")
    if n_neighbors < 2:
        _fail("umap.n_neighbors 必须 >= 2")
    if not 0.0 <= min_dist <= 1.0:
        _fail("umap.min_dist 必须在 0..1")
    if n_jobs < 1:
        _fail("umap.n_jobs 必须 >= 1")

    download = _section(data, "download")
    _check_keys(
        download,
        {"yt_dlp_format", "max_height", "cookies_file", "proxy", "retries"},
        "download",
    )
    cookies_raw = _opt(download, "cookies_file", "download", str)
    cookies_file = (
        _path(project_root, cookies_raw, "download.cookies_file") if cookies_raw else None
    )
    retries = int(_req(download, "retries", "download", int))
    if retries < 0:
        _fail("download.retries 不能为负")

    bilibili = _section(data, "bilibili")
    _check_keys(bilibili, {"api_timeout_seconds", "user_agent"}, "bilibili")
    api_timeout = float(_req(bilibili, "api_timeout_seconds", "bilibili", (int, float)))
    if api_timeout <= 0:
        _fail("bilibili.api_timeout_seconds 必须大于 0")

    app = _section(data, "app")
    _check_keys(app, {"title", "point_size", "opacity", "density_grid"}, "app")
    point_size = int(_req(app, "point_size", "app", int))
    opacity = float(_req(app, "opacity", "app", (int, float)))
    density_grid = int(_req(app, "density_grid", "app", int))
    if point_size < 1:
        _fail("app.point_size 必须 >= 1")
    if not 0.0 < opacity <= 1.0:
        _fail("app.opacity 必须在 (0, 1]")
    if density_grid < 2:
        _fail("app.density_grid 必须 >= 2")

    logging_section = _section(data, "logging")
    _check_keys(logging_section, {"level"}, "logging")
    level = _req(logging_section, "level", "logging", str)
    if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        _fail("logging.level 非法")

    paths = PathsConfig(
        input_local=input_local,
        input_lists=input_lists,
        data=data_dir,
        temp=temp_dir,
        logs=logs_dir,
        db_path=db_path,
        embeddings=(data_dir / "embeddings").resolve(),
        representations=(data_dir / "representations").resolve(),
        maps=(data_dir / "maps").resolve(),
    )
    return Config(
        path=config_path,
        root=project_root,
        project_name=project_name,
        paths=paths,
        media=MediaConfig(
            ffmpeg=str(_req(media, "ffmpeg", "media", str)),
            ffprobe=str(_req(media, "ffprobe", "media", str)),
            clip_seconds=clip_seconds,
            clips_per_work=clips_per_work,
            min_duration_seconds=min_duration,
            audio_sample_rate=sample_rate,
            audio_channels=audio_channels,
            video_crf=video_crf,
            video_preset=str(_req(media, "video_preset", "media", str)),
            extract_timeout_seconds=extract_timeout,
            probe_timeout_seconds=probe_timeout,
            clip_height=clip_height_raw,
            long_video_seconds=long_video_seconds,
        ),
        sampling=SamplingConfig(mode=sampling_mode, overlap_seconds=overlap_seconds),
        events=EventsConfig(
            enabled=events_enabled,
            audio_rate_hz=audio_rate_hz,
            video_fps=video_fps,
            version=event_version,
        ),
        temporal=TemporalConfig(stages=stages),
        model=ModelConfig(
            checkpoint=str(_req(model, "checkpoint", "model", str)),
            revision=revision,
            version=str(_req(model, "version", "model", str)),
            frames=frames,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            cache_dir=_path(project_root, _req(model, "cache_dir", "model", str), "model.cache_dir"),
            local_files_only=bool(_req(model, "local_files_only", "model", bool)),
        ),
        umap=UmapConfig(
            metric=str(_req(umap, "metric", "umap", str)),
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            n_jobs=n_jobs,
            n_epochs=n_epochs,
            dimensions=dimensions,
        ),
        download=DownloadConfig(
            yt_dlp_format=str(_req(download, "yt_dlp_format", "download", str)),
            max_height=int(_req(download, "max_height", "download", int)),
            cookies_file=cookies_file,
            proxy=_opt(download, "proxy", "download", str),
            retries=retries,
        ),
        bilibili=BilibiliConfig(
            api_timeout_seconds=api_timeout,
            user_agent=str(_req(bilibili, "user_agent", "bilibili", str)),
        ),
        app=AppConfig(
            title=str(_req(app, "title", "app", str)),
            point_size=point_size,
            opacity=opacity,
            density_grid=density_grid,
        ),
        logging=LoggingConfig(level=level),
    )
