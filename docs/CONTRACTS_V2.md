# 迭代 v0.2/v0.3 契约（内部约定）

> 在 v0.1 的 docs/CONTRACTS.md 基础上新增。共享文件仍归 Lead；本文件是新增模块的接口依据。
>
> **阅读提示**：本文件最初写于 v0.2 实现之前，正文中“将新增 / 追加 / 待实现”等未来时措辞
> 属于历史设计记录；当前真实行为以下方“实现状态”表与实际代码为准。

## 实现状态（v0.2.x）

更新时间：2026-10-25；代码基线：公开 main `b0a2dfb`（v0.2.0）。
公开 main 的 `app.py` 已实际接入 `SimilarityIndex` / `_render_neighbor_panel` /
`total_versions`，`density_knn` 由 `nebula/appdata.py` 的 `apply_color` 生效；
本文件不再是“待实现清单”。

| 项 | 状态 | 实现位置 | 测试 / QA 证据 |
| --- | --- | --- | --- |
| DB schema v2（§1） | ✅ 已实现（v0.2.0） | `nebula/db.py`：SCHEMA_VERSION=2、v1→v2 自动迁移、work_links / pipeline_runs / work_timings / map_runs.filter_json | `tests/test_db.py` 19 项（含 v2 迁移、work_links、filter_json、计时） |
| A1 高维近邻（§2） | ✅ 已实现（v0.2.0；v0.2.x 增加解释字段/片段级） | `nebula/similarity.py`；`app.py` 近邻面板/高亮/连线；`appdata.neighbor_rows`；`st.cache_resource` 缓存（key 含 db mtime + 版本）；v0.2.x：`pair_percentile / is_mutual / hubness / hub_top / clip_similarity / best_clip_pairs / explain_pair` | `tests/test_similarity.py` 26 项、`tests/test_app.py` 21 项、`tests/test_appdata.py` 28 项；独立验收手算高维余弦/分位/互为/hubness/片段配对对照 |
| A2 数据准备/界面（§3） | ✅ 已实现（v0.2.0；v0.2.x 增加解释列/枢纽/片段面板） | `nebula/appdata.py`：list_keys/vote_years/is_canonical/density_knn/filter_frame/neighbor_rows/hub_rows/clip_heatmap_data/clip_pair_rows；`app.py`：侧栏筛选/密度染色/近邻解释表/枢纽/片段面板/3D 连线开关 | `tests/test_appdata.py` 28 项、`tests/test_app.py` 21 项；真实 AppTest（map_v002/v003/v004-006）通过 |
| A2 筛选星云与变体（§4） | ✅ 已实现（v0.2.0） | `nebula/mapplot.py`：filter_spec / build_map_variants；`map.py`：--dataset/--vote-year/--list-key/--n-neighbors/variants | `tests/test_mapplot.py` 30 项；真实 map_v002(votes_2025,29)、map_v003(全量,56)、v004-006(n_neighbors 变体) filter_json/坐标核对通过 |
| B1 身份关联与去重（§5） | ✅ 已实现（v0.2.0） | `nebula/identity.py`；`ingest.py link-local`；`work_links` canonical/duplicate 单层关系 | `tests/test_identity.py` 47 项；真实数据 6 条关联、`62 completed/1 failed`、总星云 56=62−6，独立扫描无未关联高置信重复 |
| A3 片段稳定性（§6） | ✅ 已实现（v0.2.0） | `nebula/stability.py`；`tools/clip_stability.py` | `tests/test_stability.py` 30 项；三份真实报告：P0 33 部（mean_overlap=0.8978，不稳定 1/33）、v0.2 canonical 56 部（0.875，6/56）、扩量后 canonical 110 部（0.8488，21/110），文件为 `logs/clip_stability_{p0,v02,v03}.json/.csv` |
| A4 分阶段计时（§7） | ✅ 已实现（v0.2.0） | `pipeline_runs / work_timings`；`pipeline.py timings`；`logs/perf_<run_id>.csv` | `tests/test_pipeline.py` 12 项、`tests/test_db.py` 19 项、`tests/test_embed.py` 40 项；真实批次 CSV 已生成 |
| A5 低风险提速 | ✅ 已实现（v0.2.0，零漂移项） | 切片 mp4 去 AAC 音轨、`extract_clip(info=...)` 复用 ffprobe；preset/336p 对照数据见 ARCHITECTURE §6.2（未采纳） | `tests/test_media.py` 22 项、`tests/test_embed.py` 40 项；T1 基准工具 `tools/bench_ffmpeg.py` / `bench_preprocess.py` / `bench_peav.py` |
| A6 音画错位实验（v0.2.x 新增） | ✅ 已实现并完成真实 GPU 实验（2026-10-25） | `nebula/avsync.py`、`tools/desync_experiment.py`；输出 `logs/av_desync_experiment.json/.csv`；方法见 ADR-0008 | `tests/test_avsync.py` 37 项；10 部/48 片段实测：Δ_quality=0.000964、+0.5s=0.004222(4.38×)、+1.0s=0.002364(2.45×)、换音频=0.087575(90.85×)；video 分支对纯音频变体 Δ=0 |
| §8 真实 2025 数据 | ✅ 已完成（2026-10-25） | 89 部全部导入；累计 120 completed + 2 failed（8.71h 直播回放、API -404 已删除）、pending=0；身份关联 10 条；预检 `logs/preflight_89.json`（80 normal / 7 multi / 1 long / 1 unavailable）；批处理 run_id=5：54 部全部成功（730.51s，模型加载 17.05s） | 独立只读核对（120/2/10/89）+ 真实 AppTest + map_v007(87)/map_v008(110) 通过；A3 v0.3 报告已生成 |
| §9 验收与汇报 | 流程有效 | — | 本文件每项均按“证据式完成报告”验证 |

**当前门禁**（2026-10-25 v0.2.x）：仓库内 `python qa.py` = PASS 5/5（ruff 全绿；
pytest 524 收集 / 518 passed + 6 个 model/network deselected；docs/CLI/config 全过；
干净临时目录复跑同样通过）。后续迭代以本文档顶部状态表与代码为准，正文历史措辞不再表示待办。

## 0. 现状与硬约束

- P0 已完成：33 部本地视频、198 个 npy、map_v001（33 部）、v0.1 已提交并推送公开仓库。
- 本阶段目标：迭代 A（观察功能 + 计时 + 低风险提速）与迭代 B（真实 2025 选票前 30 部联网试运行）。
- 不换模型、不做人工风格标注、不重构架构、不改 5×8s 采样方案。
- 现有 33 部作品的向量必须可直接使用新的观察功能（A1/A2/A4 不得要求重新编码）。
- 默认测试离线、快速、不加载模型、不访问网络；模型/网络测试用 marker。
- 只改自己任务写权限内的文件；共享文件（config.yaml、pyproject.toml、nebula/config.py、nebula/db.py、nebula/pipeline.py、pipeline.py、tests/conftest.py、AGENTS.md、执行方案文档）归 Lead，需要新能力先发消息。
- 不要运行 git 写命令；Lead 统一提交。
- **实现状态**：本文件描述的功能已在 v0.2.0（公开 main b0a2dfb）实现，详见顶部“实现状态”表；
  阅读正文时请把未来时措辞视为历史设计记录。

## 1. SQLite schema v2（Lead 已实现，直接使用）

SCHEMA_VERSION 已升为 "2"，旧库（v1）在 Catalog.initialize() 时自动迁移（新增 map_runs.filter_json 列 + 新建三张表），无需重建数据库。

新增/变更 API：

~~~python
# 身份关联（同一作品：本地视频 vs Bilibili 条目）
cat.link_works(canonical_work_id, duplicate_work_id, method="title+duration",
               confidence=0.99, note=None) -> link_id
cat.unlink_work(duplicate_work_id, reason=None) -> canonical_id
cat.get_work_links() -> list[dict]
cat.get_work_link(work_id) -> dict | None      # canonical 或 duplicate 均可查
cat.get_canonical_work_id(work_id) -> int      # duplicate 返回 canonical，其它原样
cat.get_duplicate_work_ids() -> set[int]       # 默认星云要排除的重复实体
cat.get_linked_duplicates(canonical_work_id) -> list[dict]  # 行内带 link_method/confidence/note

# 流水线批次级指标（A4）
run_id = cat.start_pipeline_run(embedding_version=..., params={...}) -> int
cat.finish_pipeline_run(run_id, n_selected=..., n_completed=..., n_failed=...,
                        total_s=..., model_load_s=...)
cat.record_work_timing(run_id, work_id, embedding_version=...,
                       download_s=0.0, probe_s=0.0, extract_s=0.0,
                       preprocess_s=0.0, forward_s=0.0, save_s=0.0, total_s=0.0)
cat.list_work_timings(run_id=None, embedding_version=None)
cat.list_pipeline_runs(limit=20)

# 星云筛选条件落库（A2）
cat.create_map_run(..., filter_spec={"dataset": "bilibili", "vote_year": 2025,
                                     "list_key": "votes_2025", "include_duplicates": False})
# get_map()/latest_map()/list_maps() 返回行里会有 filter_json；None 表示无筛选（全量）
~~~

语义：
- canonical 是星云/展示保留的实体；duplicate 默认不进入星云，但其 npy 与作品记录保留。
- 每个 duplicate 只能属于一个 canonical；canonical 本身不能是别人的 duplicate。
- 地图/近邻索引/界面默认都用 "canonical 集合"（排除 duplicate_work_ids）。

## 2. A1 nebula/similarity.py（T2 负责，✅ 已实现 v0.2.0）

~~~python
KINDS = ("work_joint", "work_audio", "work_video")
KIND_LABELS = {"work_joint": "联合", "work_audio": "音频", "work_video": "画面"}

@dataclass(frozen=True)
class Neighbor:
    work_id: int
    similarity: float   # 真实高维余弦相似度，[-1, 1]
    rank: int           # 1-based

class SimilarityIndex:
    def __init__(self, catalog, config, *, version: str | None = None)
    version: str
    work_ids: tuple[int, ...]            # canonical 集合、按 work_id 排序
    available_kinds -> tuple[str, ...]   # 有向量的模态
    def matrix(self, kind="work_joint") -> np.ndarray   # (n, dim) float32，已 L2 归一化
    def index_of(self, work_id) -> int
    def neighbors(self, work_id, *, kind="work_joint", k=5) -> list[Neighbor]
        # 不含自身；按 similarity 降序；作品不足时返回尽量多的条数
    def similarity(self, a_work_id, b_work_id, *, kind="work_joint") -> float
    def kth_neighbor_distance(self, *, k=5, kind="work_joint") -> dict[int, float]
        # 1 - 第 k 近邻余弦相似度；不足 k 个邻居时用最后一个；用于 A2 密度染色
    def close(self) -> None

def load_similarity_index(catalog, config, *, version=None) -> SimilarityIndex
~~~

要求：
- 只读 Catalog + data/embeddings，不重新计算模型；矩阵在进程内缓存一次（重复点击不再读 npy）。
- 严格使用高维余弦，不允许用 UMAP 平面距离。
- app.py 用 st.cache_resource 包一层（key 含 db 文件 mtime + embedding version）。
- 单元测试用临时 DB + 合成 npy；覆盖：近邻排序正确、排除自身、k 截断、缺失模态、duplicate 排除、缓存不再重复读文件（可注入读取计数）。

## 3. A2 appdata.py / app.py（T2 负责，✅ 已实现 v0.2.0）

appdata 追加（保持纯数据，不 import streamlit/plotly）：
- load_nebula_frame 的 DataFrame 增加列：list_keys、list_names、vote_years、rank、votes、local_file（从关联的 duplicate 本地作品补充）、is_canonical。
- 新染色字段 density_knn：values = 1 - 第5近邻余弦相似度（越大表示局部越稀疏），需要传入 SimilarityIndex；旧 none 保留为二维网格密度。
- apply_color 增加 similarity_index 参数；available_color_fields(df, has_similarity=True) 时包含 density_knn。
- filter_frame(df, *, sources=None, list_keys=None, vote_years=None)：空值表示不过滤；用于界面筛选。
- neighbor_rows(df, index, work_id, kind, k) -> 带标题/相似度的展示行（app 用）。

app.py（✅ 已实现 v0.2.0，以下为设计记录）：
- 侧栏：星云版本、2D/3D、染色维度（含“高维近邻密度”）、搜索、点大小/透明度、数据集筛选（来源：全部/本地/Bilibili；榜单多选；年份多选）。
- 选中作品（2D 点击 / 3D 下的下拉选择）后：
  - 详情面板显示三个模态的 Top5 近邻（标题 + 余弦值），缺失模态显示“无该模态向量”；
  - 星图中高亮 Top5（不同 trace）并画出选中→近邻连线；2D 默认开，3D 提供“显示近邻连线”开关（默认关）；
  - 近邻列表可点击/下拉切换选中作品（等价入口，避免 3D 点击限制）。
- 现有 AppTest 必须继续通过；新增测试覆盖：近邻面板、三模态切换、高亮 trace 出现、筛选控件、density_knn 染色。

## 4. A2 mapplot.py / map.py 筛选与变体（T2 负责，✅ 已实现 v0.2.0）

~~~python
def load_embedding_matrix(cat, cfg, *, version=None, filter_spec=None
                          ) -> tuple[list[int], np.ndarray]
def build_map(cat, cfg, *, version=None, note=None, filter_spec=None,
              n_neighbors=None) -> MapBuildResult
def build_map_variants(cat, cfg, *, n_neighbors_list=(8, 15, 30),
                       filter_spec=None, note_prefix=None) -> list[MapBuildResult]
~~~

filter_spec 支持键（都可省略）：
~~~python
{
  "dataset": "all" | "local" | "bilibili",   # 默认 all
  "vote_year": 2025,                          # 通过 appearances/lists 过滤
  "list_key": "votes_2025",                   # 通过 appearances/lists 过滤
  "include_duplicates": False                 # 默认排除 work_links 的 duplicate
}
~~~
- build_map 把 filter_spec 写入 map_runs.filter_json（create_map_run(filter_spec=...)），并在 meta.json/params 里记录；构建前若过滤后作品数 < 3 抛中文 MapError。
- n_neighbors 覆盖 config 值并写入 params；2D/3D 仍分别训练。
- CLI（✅ 已实现 v0.2.0）：map.py build 增加 --dataset、--vote-year、--list-key、--n-neighbors；新增 map.py variants --n-neighbors 8 15 30 [--vote-year ...] [--list-key ...] 依次生成多个版本。
- 注意：筛选后的 UMAP 坐标与总星云不可直接比较；文档/输出要提示。

## 5. B1 nebula/identity.py（T3 负责，✅ 已实现 v0.2.0）

~~~python
@dataclass(frozen=True)
class IdentityCandidate:
    local_work_id: int
    bilibili_work_id: int
    local_title: str
    bilibili_title: str
    local_duration_s: float | None
    bilibili_duration_s: float | None
    method: str
    confidence: float
    accepted: bool
    reason: str

def normalize_title(value: str) -> str
    # 去换行/所有空白、统一全角半角括号引号冒号感叹号、casefold；空串抛 IngestError

def find_candidates(cat, config, *, list_key: str = "votes_2025",
                    client=None, duration_tolerance_s: float = 3.0
                    ) -> list[IdentityCandidate]
    # 1) 取该榜单的 bilibili 作品与全部 local 作品；
    # 2) 规范化标题完全相等才进入候选；
    # 3) 用 BilibiliClient 取真实标题/时长（client 可注入，测试离线）；
    # 4) |local_duration - bili_duration| <= tolerance 才 accepted=True；
    # 5) 同时把 Bilibili 元数据补全到作品记录（title/author/publish_date/duration）。

def link_candidates(cat, config, *, candidates, copy_vectors=True,
                    version: str | None = None) -> dict
    # 仅 accepted 的候选：cat.link_works(canonical=bili, duplicate=local, ...)
    # copy_vectors=True 且 local 有当前版本向量时：
    #   - 把 work_*/clip_* npy 复制到 canonical 作品的 embeddings 目录（同 embedding_version）
    #   - cat.register_embedding(...) + cat.replace_clips(...)（片段计划来自 local 的 clips 行）
    #   - canonical 状态置 completed（记录 event），从而流水线自动跳过下载/编码
    #   - 不删除本地原视频、不删除 local 的向量
    # 返回 {"linked": n, "copied": n, "skipped": [...]}

MANUAL_LINKS_FILENAME = "local_bv_links.csv"   # 放在 input/lists/，列：local_path,bvid[,part]
def load_manual_links(cat, config) -> list[tuple[int, int]]   # 本地相对路径 -> bilibili work_id
~~~

- ingest.py（✅ 已实现 v0.2.0）link-local 子命令：--list-key votes_2025、--dry-run、--no-copy、--manual；输出候选表与统计。
- 不自动合并可疑项；标题规范化后必须完全相等，且时长差 <= 3s。
- 测试：规范化边界、标题不匹配不链接、时长不符拒绝、手动覆盖、复制向量后 has_embedding=true、duplicate 排除集合正确、不删除原文件。

## 6. A3 nebula/stability.py + tools/clip_stability.py（T4 负责，✅ 已实现 v0.2.0）

~~~python
def leave_one_out_vectors(clip_matrix: np.ndarray) -> np.ndarray
    # (n_clips, dim) -> (n_clips, dim)：去掉第 i 个片段后其余片段的 归一化-平均-归一化
def stability_report(cat, config, *, version=None, k=5,
                     work_ids=None) -> dict
    # 对每部 canonical 作品：用 clip_joint 求 5 个 LOO 作品向量，
    # 在“其余作品使用正式 work_joint”的固定空间里计算 Top-k 近邻，
    # 与正式 work_joint 的 Top-k 比较：overlap、最大名次变化、平均余弦漂移；
    # 汇总 mean_overlap / unstable_count；输出 JSON + CSV。
def write_report(report: dict, *, json_path, csv_path) -> None
~~~

- 只用已保存向量，不重新提取、不重新编码、不参与坐标训练，也不作为作品质量指标。
- CLI（✅ 已实现 v0.2.0）：tools/clip_stability.py（支持 --config/--version/--k/--out-dir，另有可选 --work-ids），中文摘要。
- 测试用合成 clips/工作向量，验证 LOO 数学性质与 overlap 计算。

## 7. A4 计时（Lead + T1，✅ 已实现 v0.2.0）

- EmbeddingBatch 新增 preprocess_s / forward_s（默认 0.0，CUDA 同步计时）；PEAVEmbedder 新增 model_load_s。
- pipeline 负责：download_s / probe_s / extract_s / save_s / total_s，写 work_timings + pipeline_runs，并输出 logs/perf_<run_id>.csv。
- 每部一次的 ffprobe 结果会传给 extract_clip(info=...)，消除重复探测（T1 实现 media 侧）。
- 默认测试离线；计时字段非负；HashTestEmbedder 的计时允许为 0。

## 8. 真实 2025 数据（Lead 执行，其他人不要跑真实联网批量）

- 真实 CSV：input/lists/zuozhe_votes_2025.csv（89 行，UTF-8-BOM，列：BV号/年份/排名/票数/作品标题/来源网页；标题可能含换行）。
- 本轮只处理 **前 30 部**（按 CSV 顺序），由 Lead 导入/关联/下载/编码；用户确认后再扩大。
- 本地已有 33 部作品；其中约 10 部与榜单标题完全匹配（身份关联候选）。
- 任何真实下载/长时间 GPU 运行由 Lead 统一调度，避免重复占用 GPU。
- 状态（v0.2.x，2026-10-25）：前 30 部已完成（29 completed / 1 failed：8.7h 直播回放，已知边界），
  得到 6 条身份关联与 map_v002/map_v003；89 部选票已全部导入并完成预检（身份关联增至 10 条），
  首批 54 部批处理进行中；扩量完成后重建总星云/2025 星云。

## 9. 验收与汇报

完成后 send_message 给 lead：
- 改动文件、公开接口变化
- 运行过的测试/ruff 与真实证据（数量、关键数值）
- 未完成项、风险、需要 Lead 决策的问题
- 不要把未验证的功能写成已完成

自检：
~~~powershell
.\\.venv\\Scripts\\python.exe -m pytest tests -q
.\\.venv\\Scripts\\ruff.exe check nebula tests
~~~
