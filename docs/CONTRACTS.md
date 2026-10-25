# 模块契约（内部约定）

> 本文件是并列开发时的接口与边界约定。实现以本文件 + 现有代码为准；
> 发现契约不合理时先和 Lead 沟通，不要单方面改动共享文件。

## 1. 数据流

~~~text
本地视频 / 选票 CSV / 周报
        │  ingest.py（nebula.ingest）
        ▼
SQLite（works / lists / appearances / 任务状态）
        │  pipeline.py（nebula.pipeline）
        │  下载 → ffprobe → 8s 音画同步切片 → PE-AV → 保存 npy
        ▼
data/embeddings/<embedding_version>/<work_id>/work_*.npy + clip_*.npy
        │  map.py（nebula.mapplot）
        │  UMAP(cosine) 分别训练 2D / 3D
        ▼
data/maps/<map_version>/coords.parquet + map_2d.joblib + map_3d.joblib
        │  app.py（nebula.appdata + Streamlit/Plotly）
        ▼
可交互星云：2D/3D 切换、悬浮、点击详情、外部元数据染色、文件名搜索、跳转 Bilibili
~~~

## 2. 共享文件与写权限（重要）

Lead 维护、其他人不要修改：

- config.yaml、pyproject.toml、.gitignore、AGENTS.md、执行方案文档
- nebula/__init__.py、nebula/config.py、nebula/db.py、nebula/sampling.py、nebula/errors.py、nebula/util.py
- tests/conftest.py
- nebula/pipeline.py、pipeline.py（Lead 负责）
- 所有 git 操作（提交、分支、回滚）由 Lead 执行，teammate 不要运行 git 写命令

需要新的共享能力（例如 db.py 缺一个查询方法）时：先在任务板/消息里说明需要的签名与用途，由 Lead 添加；不要自己改共享文件。

## 3. 通用约定

- Python 3.10，虚拟环境：.venv（不要动全局 Python，不要重装现有依赖）。运行：
  ~~~powershell
  .\\.venv\\Scripts\\python.exe -m pytest tests -q
  .\\.venv\\Scripts\\ruff.exe check nebula tests
  ~~~
- 代码风格：ruff（E/F/W/I/B/SIM，行宽 100）。公开函数写类型注解；docstring 用中文；库代码用 logging 而不是 print；CLI 负责打印结果。
- 错误：使用 nebula.errors 中的异常类型；不要用裸 Exception。可预期错误在 CLI 顶层捕获并返回非 0，给出中文提示。
- 可复现：采样规则只用 nebula.sampling.plan_clips；不引入额外随机性；UMAP random_state 固定。
- 默认测试必须离线、快速：不下载模型、不访问网络、不需要 GPU。模型/网络测试用 pytest marker 标出。
- 不新增第三方依赖（需要的库已装好）；若确实需要，先联系 Lead。
- 文件读写函数尽量原子（先写临时文件再 os.replace），避免中断留下半个文件。

## 4. 已完成的地基（直接使用）

### 4.1 nebula/config.py

~~~python
cfg = load_config("config.yaml")   # 返回 frozen dataclass Config
cfg.paths.data / cfg.paths.embeddings / cfg.paths.maps / cfg.paths.db_path
cfg.media.ffmpeg / cfg.media.clip_seconds / cfg.media.clips_per_work ...
cfg.model.checkpoint / cfg.model.version / cfg.model.frames / cfg.model.batch_size ...
cfg.umap.metric / n_neighbors / min_dist / random_state / n_jobs / dimensions
cfg.download.yt_dlp_format / max_height / cookies_file / proxy / retries
cfg.bilibili.api_timeout_seconds / user_agent
cfg.app.title / point_size / opacity / density_grid
cfg.embedding_version   # 例如 peav-small-16f__f16_clip8s_x5_48k
cfg.data_path("embeddings/x.npy") -> Path
cfg.data_relpath(abs_path) -> "embeddings/x.npy"
cfg.ensure_directories()
~~~

### 4.2 nebula/db.py（Catalog）

~~~python
cat = Catalog(cfg.paths.db_path); cat.initialize(); cat.close()
# 作品
cat.upsert_local_work(rel_path, title=None) -> (work_id, created)
cat.upsert_bilibili_work(bvid, part=1, title=None, author_uid=None,
                         duration_s=None, publish_date=None, url=None) -> (work_id, created)
cat.get_work(work_id) / get_work_by_key / get_work_by_bvid
cat.list_works(status=None, source=None, limit=None)
cat.count_works_by_status()
cat.update_work_metadata(work_id, duration_s=..., has_audio=..., ...)  # 白名单字段
# 状态机（pending -> downloading -> encoding -> completed；任意 -> failed；failed -> 重试）
cat.begin_attempt(work_id, "downloading" | "encoding") -> old_status
cat.fail(work_id, message) / cat.complete(work_id)
cat.get_events(work_id)
# 向量版本与登记
cat.register_embedding_version(version, model_checkpoint=..., model_revision=...,
                               frames=..., clip_seconds=..., clips_per_work=...,
                               audio_sample_rate=..., params={...})
cat.register_embedding(work_id, version=..., kind="work_joint", path="embeddings/..",
                       dim=..., n_clips=None)
cat.has_embedding(work_id, version=..., kind="work_joint")
cat.get_embedding(...) / list_embeddings(version, kind=None)
cat.list_works_with_embedding(version, kind="work_joint")  # 行内带 embedding_path/dim/n_clips
cat.replace_clips(work_id, version=..., clips=[(idx, start_s, dur_s), ...])
cat.get_clips(work_id, version=...)
# 榜单
cat.ensure_list(list_key, name=..., kind="votes"|"weekly"|"manual", vote_year=None,
                url=None, source_path=None) -> list_id
cat.add_appearance(work_id, list_id, rank=None, votes=None, note=None)
cat.get_appearances(work_id) / cat.appearance_rows() / cat.list_lists()
# 星云
cat.next_map_version() -> "map_v001"
cat.create_map_run(version, embedding_version=..., params={...},
                   coords=[(work_id, x2, y2, x3, y3, z3), ...],
                   coords_path="maps/map_v001/coords.parquet",
                   model_2d_path=..., model_3d_path=..., note=None)   # 原子写入
cat.get_map(version) / cat.latest_map() / cat.list_maps()
cat.load_map_rows(version)  # 作品行 + x2,y2,x3,y3,z3
cat.replace_map_coords(version, coords)   # P3 transform 用
cat.stats()
~~~

### 4.3 nebula/sampling.py

~~~python
from nebula.sampling import plan_clips, ClipPlan
plans = plan_clips(info.duration_s, clip_seconds=cfg.media.clip_seconds,
                   clips_per_work=cfg.media.clips_per_work,
                   min_duration_seconds=cfg.media.min_duration_seconds)
# tuple[ClipPlan(index, start_s, duration_s)]
~~~

### 4.4 异常

nebula.errors: NebulaError, ConfigError, DatabaseError, MediaError, SamplingError,
EmbeddingError, DownloadError, IngestError, MapError, PipelineError。

## 5. nebula/media.py（T1 负责）

~~~python
@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration_s: float
    has_video: bool
    has_audio: bool
    width: int | None
    height: int | None
    fps: float | None
    video_codec: str | None
    audio_codec: str | None

def probe_media(path, *, ffprobe="ffprobe", timeout_s=60.0) -> MediaInfo
    # ffprobe -show_streams -show_format -of json；失败抛 MediaError；时长 <= 0 抛 MediaError

def extract_clip(source: Path, plan: ClipPlan, dst_dir: Path, *,
                 media: MediaConfig) -> ExtractedClip
    # 在 temp 目录生成 clip_{index:02d}.mp4（H.264/yuv420p）与 clip_{index:02d}.wav（48k/2ch/pcm_s16le）。
    # 音画必须来自同一时间段（-ss 同时作用于音视频），无音轨时补等长静音。
    # 失败抛 MediaError；成功后 assert 两个文件存在且非空。

@dataclass(frozen=True)
class ExtractedClip:
    index: int
    start_s: float
    duration_s: float
    video_path: Path
    audio_path: Path

def mute_audio_clip(duration_s, out_wav, *, media: MediaConfig) -> Path
    # 生成等长静音 wav（消融实验用）

def black_video_clip(duration_s, out_mp4, *, media: MediaConfig,
                     width=336, height=336, fps=25) -> Path
    # 生成等长纯黑视频（消融实验用）

def clip_duration_report(video_path, audio_path, *,
                         ffprobe="ffprobe", tolerance_s=0.15) -> dict
    # 返回 {"video_duration_s", "audio_duration_s", "delta_s", "ok"}，用于 P0 音画同步抽查
~~~

命令行实现注意：ffmpeg/ffprobe 用 cfg.media 里的路径，通过 subprocess.run 调用，捕获 stderr 写入 MediaError。

## 6. nebula/embed.py（T1 负责）

~~~python
@dataclass(frozen=True)
class EmbeddingBatch:
    joint: np.ndarray                    # (n, dim) float32，逐行 L2 归一化
    audio: np.ndarray | None             # 同上；模型不提供时为 None
    video: np.ndarray | None

class Embedder(Protocol):
    version: str
    is_test_only: bool
    def encode_pairs(self, pairs: Sequence[tuple[Path, Path]]) -> EmbeddingBatch
    def close(self) -> None

class PEAVEmbedder:
    def __init__(self, model: ModelConfig)   # 懒加载 transformers 的 PeAudioVideoModel/Processor
    version = model.version
    is_test_only = False
    # pairs 中每个元素是 (视频文件, 音频文件)。按 cfg.model.batch_size 分批：
    # inputs = processor(videos=[...], audio=[...], return_tensors="pt", padding=True)
    # outputs 里取 audio_video_embeds（joint）、audio_embeds、video_embeds
    # 全部转 float32、CPU、逐行 L2 归一化
    # device: auto -> cuda if available else cpu；CPU 用 float32，CUDA 可用 bfloat16 autocast
    # cache_dir=cfg.model.cache_dir；revision=cfg.model.revision（可为 None）；local_files_only 生效

class HashTestEmbedder:
    # 仅测试/演示用，不依赖 torch/网络，完全确定性
    version = "test-hash-v1"; is_test_only = True
    def __init__(self, dim: int = 32)

def aggregate_work_vector(clip_vectors: np.ndarray) -> np.ndarray
    # 输入 (n, dim) 已归一化片段向量 -> 逐行归一化（幂等）-> 求平均 -> 归一化；float32
    # 0 范数保护：直接抛 EmbeddingError

def save_vector(path: Path, array: np.ndarray) -> None   # 原子写 .npy
def load_vector(path: Path) -> np.ndarray
~~~

PE-AV 权重已缓存在 .cache/hf（T1 可先检查是否下载完成）。真实模型验证至少覆盖：
1 个 12 秒视频切 2 段，batch 编码，检查 joint/audio/video 形状一致、范数≈1、两次运行结果逐元素接近。

## 7. nebula/download.py（T3 负责）

~~~python
@dataclass(frozen=True)
class BilibiliMeta:
    bvid: str
    title: str
    author_uid: str | None
    author_name: str | None
    duration_s: float
    publish_date: str | None      # YYYY-MM-DD
    cover_url: str | None
    page_count: int

class BilibiliClient:
    def __init__(self, config: BilibiliConfig, *, session=None)
    def fetch_meta(self, bvid: str, *, part: int = 1) -> BilibiliMeta
        # https://api.bilibili.com/x/web-interface/view?bvid=...
        # code != 0 或网络失败抛 DownloadError；session 可注入用于测试

def http_get_text(url, *, user_agent, timeout_s, proxy=None, session=None) -> str

def download_bilibili(bvid: str, *, part: int = 1, out_dir: Path,
                      download: DownloadConfig, on_progress=None,
                      ydl_factory=None) -> Path
    # 用 yt-dlp 下载，格式 download.yt_dlp_format，限制高度 max_height。
    # 返回下载后的视频文件路径（位于 out_dir）；失败抛 DownloadError。
    # 测试通过 ydl_factory 注入假对象，默认测试不得触网。
~~~

## 8. nebula/ingest.py（T3 负责）

~~~python
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".m4v"}

@dataclass(frozen=True)
class VideoFile:
    rel_path: str        # 相对 cfg.paths.input_local 的 POSIX 路径
    abs_path: Path

def scan_local_videos(input_local: Path) -> list[VideoFile]      # 递归、按 rel_path 排序

@dataclass(frozen=True)
class LocalScanSummary:
    files: int; created: int; existing: int

def register_local_videos(cat, cfg) -> LocalScanSummary

@dataclass(frozen=True)
class VotesRow:
    bvid: str
    part: int = 1
    rank: int | None = None
    votes: int | None = None
    author_uid: str | None = None
    note: str | None = None

def parse_votes_csv(path, *, vote_year: int) -> list[VotesRow]
    # utf-8-sig CSV；必需列 bvid（兼容 BV号/BV）；可选 part/rank/votes/author_uid/note
    # 兼容中文列名：分P、排名、得票数/票数、UID/UP主UID、备注
    # bvid 校验 ^BV[0-9A-Za-z]{10}$；非法行抛 IngestError 并带行号
    # 同 (bvid, part) 重复行：后者覆盖前者

@dataclass(frozen=True)
class ImportSummary:
    list_id: int; works_created: int; works_existing: int; appearances: int; bvid_count: int

def import_votes_csv(cat, cfg, path, *, vote_year: int, list_name=None) -> ImportSummary
    # ensure_list("votes_{year}", name=f"{year}年音MAD选票", kind="votes", vote_year=year)

def extract_bvids(text: str) -> list[str]        # 去重保序

def import_weekly(cat, cfg, *, list_key: str, name: str | None = None,
                  url: str | None = None, html: str | None = None,
                  bvids: Sequence[str] | None = None,
                  fetcher=None) -> ImportSummary
    # 三种输入任选：bvids 直接给；html 给页面内容；url 用 fetcher/http_get_text 抓页面。
    # 生成 kind="weekly" 的 list，BV 全部建立 appearance。

def main(argv=None) -> int   # ingest.py 顶层 CLI 调用它；子命令 scan-local / import-votes / import-weekly
~~~

## 9. nebula/mapplot.py（T2 负责）

~~~python
@dataclass(frozen=True)
class MapBuildResult:
    version: str
    embedding_version: str
    n_works: int
    coords_path: Path
    model_2d_path: Path
    model_3d_path: Path

def load_embedding_matrix(cat, cfg, *, version: str | None = None) -> tuple[list[int], np.ndarray]
    # version 默认 cfg.embedding_version；读取 work_joint 向量并再次 L2 归一化，
    # 校验维度一致；作品数 < 3 抛 MapError

def build_map(cat, cfg, *, version: str | None = None, note: str | None = None) -> MapBuildResult
    # 1) 取矩阵；2) UMAP metric=cosine，n_neighbors=min(cfg值, n-1)，
    #    min_dist/random_state/n_jobs/n_epochs 用 cfg.umap；分别 fit 2D 与 3D（各自独立模型）
    # 3) 写 data/maps/<map_vNNN>/：coords.parquet（work_id,x2,y2,x3,y3,z3）、
    #    map_2d.joblib、map_3d.joblib、meta.json（参数、向量版本、作品数）
    # 4) cat.create_map_run(...) 原子登记；失败时清理半成品目录
    # 5) 返回 MapBuildResult

def transform_new(cat, cfg, *, map_version: str | None = None) -> int
    # P3：对已有星云版本，用保存的 UMAP 模型 transform() 把缺坐标的新作品加入固定空间，
    # 旧作品坐标不变；写回 parquet + cat.replace_map_coords；返回新增数量

def main(argv=None) -> int   # map.py CLI：build / list / transform
~~~

## 10. nebula/appdata.py 与 app.py（T2 负责）

appdata 必须是纯数据函数（不导入 streamlit/plotly），便于单元测试：

~~~python
COLOR_FIELDS: 有序列表，字段 id + 中文标签：
  none(不染色/密度), duration_s(视频时长), publish_date(发布日期),
  rank(选票排名), votes(得票数), author_uid(作者UID), vote_year(选票年份), source(来源)

@dataclass(frozen=True)
class ColorSpec:
    values: list            # 与 DataFrame 行对齐的数值；缺失为 None
    title: str
    colorscale: str | list
    categorical: bool = False
    tickvals: list | None = None
    ticktext: list | None = None

def load_nebula_frame(cat, cfg, *, map_version: str | None = None
                      ) -> tuple[dict, pandas.DataFrame]
    # 默认取 cat.latest_map()；没有星云时抛 MapError 并给中文提示。
    # DataFrame 至少包含：work_id, title, work_key, source, bvid, url, author_uid,
    # author_name, duration_s, publish_date, local_path, file_name, status,
    # x, y, z（按 2D 用 x2/y2，3D 用 x3/y3/z3 另由 app 选择）, rank, votes,
    # vote_year, vote_years, list_names, search_text, hover_text

def available_color_fields(df) -> list[tuple[str, str]]     # (id, 中文标签)
def apply_color(df, field: str, *, projection: str = "2d",
                density_grid: int = 40) -> ColorSpec
    # none -> 空间密度（2D 用 histogram2d，3D 用 histogramdd），连续色
    # duration/publish_date/rank/votes -> 连续色；缺失值 None
    # author_uid/vote_year/source -> 离散色：类别编码为整数 + 固定调色列表
    # rank 数值越小越好（app 可用 reversescale 处理，或 choose 递减色）
def search_frame(df, query: str) -> DataFrame   # 大小写不敏感；空查询返回全部
def external_url(row) -> str | None             # bilibili 链接（source=bilibili）
def hover_text(df) -> list[str]
~~~

app.py（Streamlit）至少提供：

- 侧栏：星云版本（默认最新）、2D/3D 切换、染色维度、文件名/title/BV 搜索框、点大小/透明度可选
- 主图：Plotly scatter 或 scatter3d；缩放/旋转/悬浮原生支持；点击点显示详情
- 详情：标题、BV/URL（可点击跳转 Bilibili）、时长、发布日期、作者、选票排名/得票、本地文件
- 无数据时显示清晰中文提示（不是 traceback）
- 支持 streamlit.testing.v1.AppTest 在设置 NEBULA_CONFIG 环境变量后运行
- 3D 点击选择若 plotly/streamlit 不支持，提供「搜索 + 下拉选择作品」的等价入口，并在 README 说明

## 11. 流水线（Lead 负责，接口仅参考）

pipeline.py 会读取 media/embed/ingest/download 上述接口；T1 的 media/embed 接口若需调整，
先发消息给 Lead 再改。

## 12. 验收与汇报

每个任务完成时必须给出（用 send_message 发给 lead，或作为最终回复）：

- 改了哪些文件（路径）
- 运行过的命令与结果（pytest 通过数、ruff 结果、真实模型/真实网络验证）
- 未完成项、风险、需要 Lead 决策的问题
- 禁止在未验证的情况下声称完成；不能运行的检查要明确写「未运行/原因」

完成前自检：

~~~powershell
.\\.venv\\Scripts\\python.exe -m pytest tests -q
.\\.venv\\Scripts\\ruff.exe check nebula tests
~~~
