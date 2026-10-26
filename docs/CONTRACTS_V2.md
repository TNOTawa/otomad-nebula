# 迭代 v0.2–v0.4 契约（内部约定）

> 在 v0.1 的 docs/CONTRACTS.md 基础上新增。共享文件仍归 Lead；本文件是新增模块的接口依据。
>
> **阅读提示**：本文件最初写于 v0.2 实现之前，正文中“将新增 / 追加 / 待实现”等未来时措辞
> 属于历史设计记录；当前真实行为以下方“实现状态”表与实际代码为准。
> v0.4 全片覆盖与多尺度音画特征表征（config/db v3/representations/sampling/events/
> temporal/map/compare_spaces）的接口、状态与能力边界见文末 §10（准确命名不是“逐帧语义理解系统”）。

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

## 10. v0.4 全片覆盖与多尺度音画特征表征（新增，2026-10-25）

设计契约全文见 `docs/plans/v0.4-full-coverage-design.md`；本节只列接口与验收状态。
核心原则：**序列层（PE-AV 片段/事件）只编码一次；聚合器变化只新增 representation，绝不重新编码；
不同 sequence/representation 的向量禁止混入同一矩阵；旧数据/旧地图不覆盖。**

> **能力边界与命名**：v0.4 的准确名称是**「全片覆盖与多尺度音画特征表征系统」**，
> 不是「逐帧音MAD语义理解系统」。
> - PE-AV 通道每 8s 仅 16 帧（≈2fps）稀疏联合语义；高频通道是 30fps（96×54 灰度）与
>   100Hz 音频的节奏/运动/突变统计，**不是视觉语义**；
> - `temporal_stats_v1` 是**人工设计分组与权重**的固定统计聚合基线，
>   **不是训练/自监督学习模型**；它不包含可学习的音画时序关系；
> - 完整时间覆盖 ≠ 逐帧语义理解；v0.4 不声称逐帧视觉语义、可学习时序或自监督全曲表示能力。
> 详见 `ARCHITECTURE.md` §6.5 与 `docs/reports/full-coverage-analysis.md` §0。

### 10.1 状态（2026-10-25，实现与真实数据验收完成）

| 项 | 状态 | 实现位置 | 证据 |
| --- | --- | --- | --- |
| config sampling/events/temporal 段与版本属性 | ✅ 基础层已提交 `55b01f7` | `nebula/config.py`、`config.yaml` | `tests/test_core_v04.py` 10 项、`tests/test_invariants.py` 版本不变量 |
| DB schema v3（representations/representation_vectors/sequence_states、clips 5 元组、map_runs.representation_id、work_timings.events_s） | ✅ 基础层已提交 `55b01f7` | `nebula/db.py`（v2→v3 自动迁移 + legacy 表征回填） | `tests/test_core_v04.py` 迁移/登记/序列状态/地图绑定；invariants schema 不变量 |
| 表征注册/加载骨架 | ✅ 基础层已提交 `55b01f7`（t5 扩展构建 CLI） | `nebula/representations.py`：resolve/list/save/load matrix；legacy 走 embeddings，派生走 representation_vectors | `tests/test_representations.py` 18 项；invariants 隔离测试；compare_spaces 合成验收 |
| 全片采样 `plan_full_coverage` | ✅ t1 完成 | `nebula/sampling.py` | `tests/test_sampling.py` 58 项（覆盖/末尾锚定/权重/边界/确定性） |
| 全片编码流水线与片段级恢复 | ✅ t1 完成 | `nebula/pipeline.py`、`nebula/media.py`、`nebula/embed.py`、`pipeline.py` | `tests/test_pipeline.py` 26 项（4s/12s/17s/40s/180s、超长视频、批文件恢复、legacy 回归）；E2E 注入中断→只补缺失批 |
| 高频事件扫描 | ✅ t3 完成 | `nebula/events.py`、`tools/events_probe.py` | `tests/test_events.py` 20 项（合成闪帧/音脉冲/确定性/无音轨/无视频流） |
| Full Mean / Temporal Stats 聚合 | ✅ t5 完成 | `nebula/temporal.py` | `tests/test_temporal.py` 17 项（与旧聚合公式等价、顺序敏感性、维度/分组约束） |
| 构建 CLI | ✅ t5 完成 | `tools/build_representation.py` | CLI --help（qa.py 冒烟）+ 构建/增量/--force/事件变体；E2E 从已保存序列构建 full_mean |
| 地图/空间切换/时间轴/近邻集成 | ✅ t2 完成 | `map.py` `--representation`/`--sequence-version`、`nebula/mapplot.py`、`app.py`（表征空间 + 完整时间轴）、`nebula/similarity.py`、`nebula/appdata.py` | t2 测试（mapplot/app/appdata/similarity）+ E2E 地图代表征绑定/AppTest |
| 邻域迁移对比工具 | ✅ t4 完成 | `tools/compare_spaces.py` | 合成数据手算 Jaccard/hubness/Pearson + CLI；E2E 全链路 |
| 独立验收 E2E / 不变量 | ✅ t4 完成 | `tests/test_e2e.py`、`tests/test_invariants.py` | invariants 55 项（含默认 full 模式的版本断言）；E2E 27 项（v0.4 全链路 3 项） |
| 文档与报告 | ✅ t4 完成（真实数字已回填） | `ARCHITECTURE.md`、`README.md`、`AGENTS.md`、`docs/reports/full-coverage-analysis.md` | 本文件 §10 + 报告（覆盖率/耗时/存储/336p/三组邻域迁移/替换案例） |
| 真实数据（v0.4） | ✅ 全部完成（2026-10-25） | 110/110 canonical assembled；full_mean_v1(id=2,1024) / temporal_stats_v1(id=3,6356) 全量 ready；map_v011/map_v012=110；legacy=map_v008 | `logs/v04_data_analysis.json`、三组 `logs/space_compare_*.json/.csv`、336p `logs/seq_compare_*.json`；独立复核 8 PASS / 0 FAIL（覆盖率/表征/地图/三空间 compare/真实 AppTest） |

**当前门禁（v0.4，默认 sampling.mode=full_coverage）**：仓库内 `python qa.py --full` = PASS 5/5
（ruff 全绿；pytest 692 passed（model/network 已启用），174.85s；docs 10 篇 ADR 章节完整；
CLI 11 个入口 --help；config 含 v0.4 段校验，embedding_version=full 序列版本）。


### 10.2 版本命名与配置

| 概念 | 规则 |
| --- | --- |
| Sequence Version（legacy） | `Config.legacy_embedding_version`：`<slug(model.version)>__f<frames>_clip<clip_seconds>s_x<clips_per_work>_<khz>k[_h<height>]` |
| Sequence Version（full） | `Config.full_sequence_version`：`<slug>__f<frames>_full<clip_seconds>s_ov<overlap>_<khz>k[_h<height>]` |
| 当前激活 | `Config.embedding_version`：`sampling.mode` = `legacy` → legacy；`full_coverage` → full |
| Representation | `legacy_5clip_mean` / `full_mean_v1` / `temporal_stats_v1` / `transformer_v1`（v0.5 预留）；前两者是统计聚合，`temporal_stats_v1` 为固定分组/权重统计基线（非学习模型） |
| 地图 | `map_vNNN`，每次拟合新版本；`map_runs.representation_id` 绑定表征，`embedding_version` 记录其 sequence version |

`config.yaml` 新增段：

~~~yaml
sampling:
  mode: full_coverage     # v0.4 起默认全片覆盖；legacy 仅用于旧数据/旧地图
  overlap_seconds: 0.0
events:
  enabled: true
  audio_rate_hz: 100
  video_fps: 30
  version: events_v1
temporal:
  stages: 4
media:
  clip_height: null       # 336 = 336p 实验；非空时版本号追加 _h336
  long_video_seconds: 3600
~~~

### 10.3 数据布局与 DB 表（v3）

```text
data/embeddings/<sequence_version>/<work_id>/
  clip_{joint,audio,video}.npy   # (T, dim) 有序片段
  clip_times.npy                 # (T,4) start_s, duration_s, valid_duration_s, weight
  work_{joint,audio,video}.npy   # 全片均值（兼容基线）
  event_scan.npz / event_features.npy / event_manifest.json  # 事件层
  manifest.json / .partial/      # 参数清单与编码中分批状态
data/representations/<name>/<sequence_version>/<work_id>/{joint,audio,video}.npy
~~~

- `representations(id,name,sequence_version,kind,dim,path_prefix,params_json,status,note,...)`，UNIQUE(name,sequence_version)；
- `representation_vectors(representation_id,work_id,kind,path,dim)`，kind ∈ joint/audio/video；
- `sequence_states(work_id,version,duration_s,n_clips,n_clips_done,partial_relpath,status)`，status ∈ in_progress/assembled/failed；
- `clips` 新增 `valid_duration_s / weight`（旧 3 元组写入时记 NULL）；`map_runs` 新增 `representation_id`；`work_timings` 新增 `events_s`。

### 10.4 工具：tools/compare_spaces.py（T4）

~~~powershell
python tools/compare_spaces.py                              # 默认 legacy × full_mean（需已构建 full_mean_v1）
python tools/compare_spaces.py --a legacy_5clip_mean --b full_mean_v1
python tools/compare_spaces.py --a full_mean_v1 --b temporal_stats_v1 --k 5 --top-n 10
python tools/compare_spaces.py --a legacy_5clip_mean --a-seq <sequence_version> --b-id 7
~~~

- 每作品 Top-k 邻居 Jaccard/重合数、完全稳定作品比例；
- 稳定对比例（无序对有向/无序边一致率）、有向边保留率与交并比、互为邻居对交并比；
- 被替换案例（标题/来源/新旧邻居 lost/gained）；
- 共同作品上的作品对余弦矩阵 Pearson/Spearman、平均绝对差、Top1 一致率；
- hubness 前后入度、Top hub、变化最大作品、秩相关；
- 只读向量、不写 DB、不加载模型/UMAP；输出 `logs/space_compare_<a>_vs_<b>__<seqs>.json/.csv`；
- 合成数据验收：手算 Jaccard/hubness/Pearson 一致、CLI --help 可跑、默认 space 解析正确。

### 10.5 验收结果（2026-10-25）

- ✅ 全链路离线 E2E：4 部合成视频（4s/12s/17s/40s）→ HashTestEmbedder 全片编码 → 注入中断 →
  只补缺失批 → `tools/build_representation.py full_mean` →
  `map.py build --representation full_mean_v1` → `appdata.load_nebula_frame` →
  `tools/compare_spaces.py legacy×full_mean` 输出 JSON/CSV；
- ✅ 不变量：默认 full 模式/legacy 公式显式切换、full 版本格式、表征与序列隔离、地图绑定代表征且
  序列一致、schema v3 迁移幂等、旧地图坐标不被覆盖、异空间矩阵不混用；
- ✅ 真实数据：110/110 canonical 覆盖；4130 窗口（legacy 536）；权重和∈[0.99999992,1.00000003]；
  事件三件套 110/110；336p 对照（10 部：joint cos min 0.9972、video min 0.9772，total 51.08→31.85s）；
  map_v011/map_v012 均为 110 部；三组 compare（legacy×full_mean Jaccard 0.4068、Pearson 0.9569；
  full_mean×temporal Jaccard 0.2049、Pearson 0.8294；legacy×temporal Jaccard 0.1734）；
- ✅ 真实 AppTest：默认 map_v012（Temporal Stats），三空间切换 Full Mean→map_v011 /
  Legacy→map_v008 / Temporal→map_v012，时间轴事件面板与 full 序列片段面板正常；
- ✅ 报告已回填：`docs/reports/full-coverage-analysis.md`（含 replaced_works 案例与
  `logs/v04_data_analysis.json` 引用）；
- ⏳ 最终 `python qa.py --full` 在 t1/t2 默认模式测试修复后执行，结果随最终交付报告提交。
