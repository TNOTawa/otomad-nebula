# 架构设计（ARCHITECTURE）

本文描述音MAD星云计划的系统结构、数据模型、模块边界与必须保持的不变量。
v0.1 接口细节以 `docs/CONTRACTS.md` 为准，v0.2/v0.3 迭代契约见
`docs/CONTRACTS_V2.md`；用户使用说明见 `README.md`；关键取舍的理由见 `docs/adr/`。

## 1. 系统总览

~~~text
input/local/*.mp4 ──┐
votes CSV / 周报 ────┼──> ingest.py / nebula.ingest ──> SQLite catalog.sqlite
Bilibili BV 清单 ───┘                                        │
                                                             v
                              pipeline.py / nebula.pipeline（逐部作品）
   下载(Bilibili) → ffprobe 探测 → plan_clips 均匀采样 → ffmpeg 音画同步切片
        → PE-AV 推理(片段级 joint/audio/video) → 聚合为作品级向量
        → 写 data/embeddings/<embedding_version>/<work_id>/*.npy
        → 登记 SQLite（先文件后登记） → 清理 temp
                                                             │
                                                             v
                                   map.py / nebula.mapplot
                    读取 work_joint 矩阵（L2 归一化）→ UMAP(cosine)
                    2D 与 3D 分别训练（或 P3 用已保存模型 transform）
                    → data/maps/<map_version>/ + map_runs/map_coords 登记
                                                             │
                                                             v
                        app.py / nebula.appdata + Streamlit + Plotly
             2D/3D 切换、悬浮、点击详情、元数据染色、搜索、跳转 Bilibili
~~~

一次完整运行由四个顶层命令驱动：`ingest.py`（编目）、`pipeline.py`（向量化）、
`map.py`（降维）、`app.py`（可视化）；`qa.py` 是质量门禁，不参与数据流。

## 2. 分层与模块边界

| 层 | 模块 | 职责 | 允许依赖 |
| --- | --- | --- | --- |
| 基础层 | `nebula/config.py` | 配置加载、路径解析、`embedding_version` 派生 | errors、util、yaml |
| 基础层 | `nebula/db.py` | SQLite 编目：作品/榜单/向量/状态/星云版本 | errors、util |
| 基础层 | `nebula/sampling.py` | 纯函数时间轴采样（无随机、无 IO） | errors |
| 基础层 | `nebula/util.py` / `nebula/errors.py` | 路径/时间/哈希工具；统一异常类型 | 仅标准库 |
| 领域层 | `nebula/media.py` | ffprobe 探测、ffmpeg 音画同步切片、消融素材 | config、errors、util |
| 领域层 | `nebula/embed.py` | Embedder 协议、PE-AV 推理、HashTestEmbedder、聚合、npy 原子读写 | config、errors、util；**不依赖 db/pandas** |
| 编排层 | `nebula/pipeline.py` | 单部作品全流程、批处理、断点续跑、临时目录生命周期 | media、embed、ingest、download、db、config |
| 数据层 | `nebula/ingest.py` / `nebula/download.py` | 本地扫描、选票/周报解析、Bilibili 元数据与下载；v0.4.3.1 起 `resolve_bilibili_source` 优先复用 `data/sources` 持久缓存 | db、config、errors |
| 数据层 | `nebula/mapplot.py` | 向量矩阵加载、UMAP 2D/3D、版本落盘、transform 增量 | db、config、umap-learn、joblib、numpy |
| 展示层 | `nebula/appdata.py` | 纯数据准备：DataFrame、染色 `ColorSpec`（含高维近邻密度）、筛选、搜索、hover、近邻解释行、枢纽行、片段热图/配对纯数据 | pandas、numpy、db、config；**不导入 streamlit/plotly** |
| 展示层 | `app.py` | Streamlit 页面：侧栏控件、Plotly 图、详情、近邻解释表、枢纽与片段级对应面板 | streamlit、plotly、appdata、similarity（经缓存） |
| 分析层 | `nebula/similarity.py` | 高维余弦近邻索引：三模态 Top-k、全库分位/互为 Top-k、hubness、第 k 近邻距离、片段级相似（热图/配对/解释）、进程内缓存 | db、config、numpy；**不导入 umap/streamlit/plotly** |
| 数据层 | `nebula/identity.py` | 标题规范化、本地↔Bilibili 身份候选与关联（人工清单兜底）、向量复制登记 | db、config、errors；Bilibili 客户端由调用方注入 |
| 分析层 | `nebula/stability.py` | 片段留一重聚合、Top-k 稳定性报告（JSON/CSV） | config、db、embed、numpy；**不加载模型、不导入 umap/pandas** |
| 分析层 | `nebula/avsync.py` | 音画错位实验：WAV 偏移/交换、画质退化基线、Δ=1-cos 计算与 JSON/CSV 报告 | config、embed、media、sampling、numpy；不写主向量库 |
| 领域层 | `nebula/events.py` | 高频音画事件扫描：音频 100Hz/视频 30fps 特征、音画互相关、窗口压缩为 32 维 | media、config、numpy；不导入模型/UMAP/UI |
| 分析层 | `nebula/temporal.py` | 表征聚合：Full Mean（全片窗口均值）与 Temporal Stats（全局/阶段/漂移/动态/自相似/事件分组） | representations、numpy；确定性、顺序敏感性由测试约束 |
| 分析层 | `nebula/representations.py` | 表征注册/解析/向量落盘与加载；legacy 复用 embeddings，派生走 representation_vectors；矩阵按 work_id 排序、排除 duplicate；`source_kind` 区分 pe_av/visual | config、db、embed、numpy |
| 领域层 | `nebula/dense_visual.py` | v0.4.3/v0.4.3.1 冻结视觉编码器（CLIP ViT-B/16、DINOv2-small）：单次 ffmpeg 流式解码、帧级 L2 特征、窗口池化、窗口覆盖权重、bundle/manifest 原子保存 | config、errors、numpy；torch/transformers 懒加载，不导入 db/UMAP/UI |
| 编排层 | `nebula/visual_pipeline.py` | v0.4.3 密集视觉批处理：canonical+completed 作品、每部只下载一次、按编码器/版本建任务、visual_sequences 状态与断点续跑、完成后清理 temp | dense_visual、download、db、config、sampling |
| 领域层 | `nebula/v05_data.py` | v0.5-alpha token 构建：从 PE-AV 窗口/CLIP+DINOv2 帧/100Hz+30fps 事件生成 0.5s×16bin 局部 token（npz+manifest），不加载模型 | config、db、dense_visual、events、errors、numpy |
| 领域层 | `nebula/v05_model.py` | v0.5-alpha 研究模型：局部多模态 Transformer + 全曲时序 Transformer + mask/AV/order 三头（4.31M 参数） | torch、numpy |
| 领域层/编排 | `nebula/v05_train.py` | v0.5-alpha 训练/验证：确定性 90/20 split、变长 mask、三目标、指标/checkpoint/研究向量；不注册 DB/地图 | torch、numpy、config、db、errors |
| 分析层 | `nebula/discovery.py` | v0.4.5 意外关系发现：排名百分位/无向 bar_p/模式 A/B/C；只读高维向量，不写 DB/不加载模型 | config、db、highdim、representations、numpy |
| 领域层 | `nebula/audio_dense.py` | v0.5-beta 冻结音频编码器（MERT）：ffmpeg 解码/重采样、5s 窗口编码、0.5s bin 聚合、audio.npz+manifest | config、errors、numpy；torch/transformers 懒加载 |
| 领域层/编排 | `nebula/v05_beta.py` | v0.5-beta 局部音画配对与双塔 InfoNCE 训练/评估/消融；不注册 DB/地图 | torch、numpy、config、db、errors |
| 入口层 | `ingest.py` / `pipeline.py` / `map.py` / `qa.py` / `tools/clip_stability.py` / `tools/desync_experiment.py` / `tools/preflight_works.py` / `tools/build_representation.py` / `tools/compare_spaces.py` / `tools/events_probe.py` / `tools/visual_pipeline.py` / `tools/dense_preprocess_compare.py` / `tools/embedding_diagnostics.py` / `tools/build_v05_tokens.py` / `tools/train_v05.py` / `tools/explore_relations.py` / `tools/audio_dense.py` / `tools/train_v05beta.py` | 参数解析、调用库代码、打印结果、返回退出码 | 对应库模块 |
| 入口层 | `ingest.py` / `pipeline.py` / `map.py` / `qa.py` / `tools/clip_stability.py` / `tools/desync_experiment.py` / `tools/preflight_works.py` / `tools/build_representation.py` / `tools/compare_spaces.py` / `tools/events_probe.py` / `tools/visual_pipeline.py` / `tools/dense_preprocess_compare.py` / `tools/embedding_diagnostics.py` | 参数解析、调用库代码、打印结果、返回退出码 | 对应库模块 |

依赖方向只有一条：入口层 → 展示层 / 编排层 → 领域层 → 基础层。
几个必须保持的边界（由 `tests/test_invariants.py` 静态检查）：

- `nebula.embed` 不导入 `pandas`、`nebula.db`、`nebula.appdata`、`nebula.mapplot`：
  向量提取必须能脱离数据库与界面独立运行、独立测试；
- `nebula.mapplot` 不导入 `streamlit` / `plotly`：降维是纯计算，可在无界面环境运行；
- `nebula.appdata` 不导入 `streamlit` / `plotly`：数据准备与渲染分离，便于单元测试与
  `streamlit.testing.v1.AppTest` 之外的直接断言；
- `nebula.similarity` 不导入 `umap` / `streamlit` / `plotly`：近邻、相似度、
  近邻密度必须来自高维余弦，不能退化成 UMAP 平面距离；
- `nebula.stability` 不导入模型/torch、UMAP 或界面库：A3 只读已保存向量，不重新编码、
  不参与坐标训练；`tools/clip_stability.py` 同样不加载模型；
- `nebula.identity` 只做本地数据与编排，Bilibili 网络客户端由调用方注入（默认测试离线）；
- `nebula.avsync` 不导入 UMAP/界面库；导入模块本身不加载 torch/transformers（模型在实验脚本内按需使用），
  实验只写 `logs/`，不改主 `data/embeddings` 与 `embedding_version`；
- `nebula.events` / `nebula.temporal` / `nebula.representations` 不导入模型/UMAP/Streamlit/Plotly；
  事件与表征层只做数值计算与文件/DB 登记，渲染在 `app.py`；
- `nebula.dense_visual` 导入模块本身不加载 torch/transformers（编码器在首次编码时才懒加载）；
  它只接收视频路径与参数、返回 numpy 特征，不导入 db/UMAP/Streamlit/Plotly；
  `nebula.visual_pipeline` 只做编排与状态登记，不导入界面库；
- 片段级探索的纯数据接口（`appdata.clip_heatmap_data / clip_pair_rows`）只返回 dict/list，
  不接收 Plotly/Streamlit 对象；渲染在 `app.py`；
- 近邻解释字段（`pair_percentile / is_mutual / hubness`）只影响展示，不改变近邻排序、
  坐标与向量；其统计集合是当前 `embedding_version` 的 canonical 作品，界面筛选不改变它；
- `app.py` 只依赖 `appdata` / `similarity` 与库层，不反向被库代码导入。

## 3. 数据模型

### 3.1 SQLite 表关系

~~~text
works 1────N work_events           （处理事件：created / status / linked / reopen / ...）
works 1────N appearances N────1 lists      （一部作品可出现在多个榜单/年份）
works 1────N embeddings  N────1 embedding_versions
works 1────N clips                         （片段计划：idx/start_s/duration_s）
works 1────N map_coords N────1 map_runs
map_runs N────1 embedding_versions         （星云必须记录基于哪个向量版本）
works 1────N work_links N────1 works       （canonical ← duplicate 单层身份关联，v0.2）
works 1────N work_timings N────1 pipeline_runs   （单部分阶段耗时 ← 批次，v0.2）
works 1────N sequence_states N────1 embedding_versions  （全片编码进度，v0.4）
works 1────N representation_vectors N────1 representations
map_runs N────1 representations            （地图必须绑定一个表征，v0.4）
works 1────N visual_sequences N────1 visual_versions   （密集视觉通道，v0.4.3）
visual_versions 1────N visual_window_sequences N────1 visual_window_plans （窗口计划层，v0.4.3.1）
representations.sequence_version ∈ embedding_versions（source_kind=pe_av）或 visual_versions（source_kind=visual）
map_runs.source_kind ∈ {pe_av, visual}；视觉地图的 embedding_version 存视觉版本 slug
~~~

| 表 | 关键列 | 说明 |
| --- | --- | --- |
| `schema_meta` | `key`, `value` | schema 版本；不匹配时拒绝启动，避免旧库被新代码破坏 |
| `works` | `id`, `work_key` UNIQUE, `source`, `bvid`, `part`, `status`, `error`, `attempts` | 作品唯一记录；`bvid, part` 有部分唯一索引 |
| `work_events` | `work_id`, `ts`, `event`, `detail` | 追加式事件日志，便于追溯失败与重试 |
| `lists` | `list_key` UNIQUE, `kind` (`votes/weekly/manual`), `vote_year` | 榜单/年份元数据 |
| `appearances` | `work_id`, `list_id`, `rank`, `votes` | 作品与榜单的出现关系；跨年重复只增行 |
| `embedding_versions` | `version` PK, `model_checkpoint`, `model_revision`, `frames`, `clip_seconds`, `clips_per_work`, `audio_sample_rate`, `params_json` | 向量版本登记；同版本参数不一致直接报错 |
| `embeddings` | `work_id`, `version`, `kind`, `path`, `dim`, `n_clips` | 只登记向量文件位置与形状，不存元数据 |
| `clips` | `work_id`, `version`, `idx`, `start_s`, `duration_s`, `valid_duration_s`, `weight` | 采样计划落库；v0.4 全片格式为 5 元组，旧 3 元组写入时后两列为 NULL |
| `map_runs` | `version` PK, `embedding_version`, `representation_id`, `source_kind`, `params_json`, `filter_json`, `coords_path`, `model_2d_path`, `model_3d_path`, `n_works`, `status` | 星云版本；`building/ready/failed`；`filter_json` 记录筛选条件（NULL = 全量）；`representation_id` 绑定表征（v0.4）；v0.4.3 起 `source_kind ∈ pe_av/visual`，视觉地图的 `embedding_version` 为视觉版本字符串 |
| `map_coords` | `map_version`, `work_id`, `x2`, `y2`, `x3`, `y3`, `z3` | 主键 `(map_version, work_id)`，一个版本一套坐标 |
| `work_links` | `canonical_work_id`, `duplicate_work_id` UNIQUE, `method`, `confidence`, `note` | 身份关联（v0.2）：duplicate 唯一、单层、默认不进星云；CHECK 禁止自关联 |
| `pipeline_runs` | `id`, `started_at/finished_at`, `embedding_version`, `model_load_s`, `n_selected/n_completed/n_failed`, `total_s`, `params_json` | A4 批次级指标（模型加载属于批次级） |
| `work_timings` | `run_id`, `work_id`, `embedding_version`, `download_s/probe_s/extract_s/preprocess_s/forward_s/save_s/events_s/total_s` | A4 单部作品分阶段耗时；v0.4 新增 `events_s`（事件扫描）；`UNIQUE(run_id, work_id)` 防止重复登记 |
| `representations` | `id`, `name`, `sequence_version`, `kind`, `source_kind`, `dim`, `path_prefix`, `params_json`, `status`, `note` | 表征版本（v0.4/v0.4.3）；`UNIQUE(name, sequence_version)`；kind ∈ legacy/full_mean/temporal_stats/transformer/visual；source_kind ∈ pe_av/visual 决定 sequence_version 的命名空间；聚合器变化只新增表征，不重编码 |
| `representation_vectors` | `representation_id`, `work_id`, `kind`, `path`, `dim` | 派生表征的作品级向量（joint/audio/video）；legacy 不写此表，直接复用 embeddings |
| `sequence_states` | `work_id`, `version`, `duration_s`, `n_clips`, `n_clips_done`, `partial_relpath`, `status` | 全片编码进度（v0.4）；status ∈ in_progress/assembled/failed；支持片段级断点恢复 |
| `visual_versions` | `version` PK, `model_id`, `revision`, `preprocessing`, `fps`, `size`, `dim`, `params_json` | 冻结视觉编码器版本（v0.4.3）；同一版本参数不一致报错；encoder/revision/fps/size/preprocess 任一变化必须新版本 |
| `visual_sequences` | `work_id`, `version`, `status`, `frames_path`, `windows_path`, `n_frames`, `n_windows`, `dim`, `decode_s/encode_s/model_load_s/total_s` | 密集视觉进度与工件（v0.4.3）；主键 `(work_id, version)`；status ∈ pending/in_progress/completed/failed；支持断点续跑与清理 |
| `visual_window_plans` | `version` PK, `clip_seconds`, `overlap_seconds`, `weight_scheme`, `params_json` | 窗口计划版本（v0.4.3.1）：窗口时长/重叠/权重方案；同版本参数不一致报错；与帧编码版本正交 |
| `visual_window_sequences` | `work_id`, `frame_version`, `window_plan_version`, `status`, `windows_path`, `window_std_path`, `window_weights_path`, `n_windows`, `dim`, `total_s` | 窗口层任务（v0.4.3.1）；主键 `(work_id, frame_version, window_plan_version)`；引用 visual_versions + visual_window_plans；支持从已保存帧重建/断点 |

连接设置：WAL、`foreign_keys=ON`、`BEGIN IMMEDIATE` 事务、`busy_timeout`；
所有写操作在事务内完成，读方法返回普通 `dict` 便于上层使用。

**当前数据规模（2026-10-26 v0.4.3 密集视觉全量后；真实运行数据不随仓库分发）**：

- `works` 122 行：33 local + 87 bilibili completed + 2 bilibili failed
  （8.71h 直播回放、API -404 已删除视频），pending = 0；canonical = 110（10 条身份关联）；
- `lists` 1 个（`votes_2025`，89 部）；`appearances` 89 条；
- 序列层：legacy `peav-small-16f__f16_clip8s_x5_48k` 720 条 npy（120 部 × 6）；
  full `peav-small-16f__f16_full8s_ov0_48k` 660 条（110 部 × 6，4130 个窗口）；
  336p 对照 `_h336` 60 条（10 部）；事件层 110 部（约 131MB）；
- 表征层：`legacy_5clip_mean`(id=1)、`full_mean_v1`(id=2, 1024 维)、
  `temporal_stats_v1`(id=3, 6356 维)；后两者均为 110 部 × joint/audio/video；
- 星云：`map_v001`(33, P0)、v002(29, votes_2025)、v003(56, 全量)、
  v004–v006(29, n_neighbors 变体)、`map_v007`(87, 2025 全量)、
  `map_v008`(110, legacy 总星云)、`map_v011`(110, full_mean_v1)、
  `map_v012`(110, temporal_stats_v1，当前默认)、
  `map_v013`(110, CLIP-only 密集视觉)、`map_v014`(110, DINOv2-only 密集视觉)。
  所有版本永久保留，坐标跨版本/跨表征不可直接比较；
- 视觉层（v0.4.3）：4 个 `visual_versions`（CLIP/DINOv2 × 8fps/30fps，letterbox/224，
  固定 revision，DB/目录用 `visual_version_slug`）；**252/252 任务 completed、0 failed**；
  8fps 每个编码器 110 部 / **260,885 帧 / 4,130 窗口**，30fps 子集每编码器 16 部 /
  140,149 帧 / 593 窗口；总存储 **673.9MB**；
- 视觉表征：`dense_visual_clip_v1`(id=4, 512 维) / `dense_visual_dino_v1`(id=5, 384 维)，
  各 110 部 joint 窗口向量；对应地图 `map_v013` / `map_v014`（`source_kind=visual`）。
  v0.4.3.1 重建 v2（`dense_visual_clip_v2`/`dense_visual_dino_v2`）后视觉 v1/v2 并存，
  v1 只读保留；默认地图仍为 PE-AV 优先（`map_v012`），视觉空间需在侧栏手动切换；
- v0.5-alpha 研究产物（**不注册 DB/地图**）：`data/v05_tokens/v05_alpha_v1/`
  110 部 / PE-AV 窗口 4,130（max_segments=128 截断后 token 段 3,904，6 部截断、丢 226 段）/
  141.1MB；checkpoint `data/models/v05_alpha_v1`、
  `data/models/v05_alpha_v2`；指标 `logs/v05_alpha_v1_metrics.json`、
  `logs/v05_alpha_v2_metrics.json`；研究向量
  `data/representations/transformer_alpha_v1/<sequence_version>/vectors.npy`；
  两次训练 **可行性验证未通过（负结果）**，详见 `docs/reports/v05-alpha-validation.md`；
- v0.4.5：`logs/unexpected_relations.json`（约 297KB）+ CSV（85KB）；6 空间/110 部/
  skipped=0；模式 A 950 候选、模式 B 864 候选、common 99 对（>=3 空间 70）、visual_vs_peav 19；
- v0.5-beta：`data/audio_dense/m-a-p-MERT-v1-95M-.../` 110/110 部、约3.87GiB
  （pilot 20/314.1MiB PASS；严格容器口径 108/110，work50/99 tail-mask 排除）；
  `data/v05_tokens/v05_beta_v1/` 110 部/4,130 段/0 截断/142.3MiB；
  `data/models/v05_beta_{42,43,44}/` + `logs/v05_beta_*`（3 seeds，test Top-1 0.301535，
  weak positive / not passed）。

### 3.2 文件系统工件

```text
data/
├── catalog.sqlite                     # 索引与状态
├── embeddings/<sequence_version>/     # 序列层：PE-AV 片段/事件（只编码一次）
│   └── <work_id>/
│       ├── work_joint.npy             # (dim,) float32，L2 范数 ≈ 1（全片/片段均值）
│       ├── work_audio.npy / work_video.npy      # 单模态作品向量（若模型提供）
│       ├── clip_joint.npy             # (n_clips, dim) float32，每行 L2 范数 ≈ 1
│       ├── clip_audio.npy / clip_video.npy
│       ├── clip_times.npy             # (T,4) start_s, duration_s, valid_duration_s, weight（v0.4）
│       ├── event_scan.npz             # v0.4.1 高频事件原始序列（音频/视频/音画关系 + 速率/版本）
│       ├── event_features.npy         # (T,32) 每个窗口的事件特征
│       ├── event_manifest.json
│       ├── manifest.json              # n_clips / duration / sampling 参数 / 代码版本标签
│       └── .partial/                  # 编码中：batch_0000.npz ... 原子写入，完成后删除
├── representations/<name>/<sequence_version>/<work_id>/
│   └── {joint,audio,video}.npy        # 派生表征向量（full_mean/temporal_stats/dense visual）
├── visual_dense/<frame_version>/<work_id>/   # v0.4.3 帧级密集视觉通道
│   ├── dense_visual.npz               # 帧级 fp16 向量 + 时间戳/版本/维度元数据
│   └── manifest.json                  # version/encoder/revision/fps/size/preprocess/duration/n_frames/dim
├── visual_windows/<window_plan_version>/<frame_version>/<work_id>/  # v0.4.3.1 窗口层
│   ├── windows.npy                    # (T,d) float32 窗口池化（与 PE-AV 8s 窗口对齐）
│   ├── window_std.npy                 # (T,d) float32 窗口内变化幅度（可选）
│   ├── window_weights.npy             # (T,) float32 覆盖权重（Σ=1；coverage-v1）
│   └── manifest.json                  # frame/window plan/fps/clip_seconds/overlap/weight_scheme/n_windows/window_times
├── sources/<bvid>/<bvid>_p<part>.*    # v0.4.3.1 Bilibili 源视频持久缓存（不随 temp 清理）
└── maps/<map_version>/
    ├── coords.parquet                 # work_id, x2, y2, x3, y3, z3
    ├── map_2d.joblib                  # 已训练的 2D UMAP 模型（供 transform）
    ├── map_3d.joblib
    └── meta.json                      # 参数、向量版本、表征、作品数、生成时间
```

- 数据库里 `embeddings.path` 存的是相对 `data/` 的 POSIX 路径
  （`config.data_relpath`），保证换机器/换盘符后仍可解析；
- `work_joint` 等作品级向量的 `dim` 在同一个 `sequence_version` 内必须一致；
- 片段级向量形状 `(n_clips, dim)`，`n_clips` 与 `clips` 表的段数一致；
  全片模式没有片段数上限，最后一个窗口结束于视频结尾且任意时间点至少被 1 个窗口覆盖；
- 序列层（片段/事件）编码一次后长期复用；表征层只做聚合（Full Mean / Temporal Stats），
  聚合器变化写新 `representations` 行，绝不重新编码 PE-AV；
- legacy 序列版本目录保持原样、只读；新序列版本另建目录，不覆盖旧向量；
- v0.4.3 视觉层是**独立命名空间**：`visual_version` 形如
  `encoder@revision__fps{fps}_s{size}_{preprocess}`（例 `openai/clip-vit-base-patch16@<rev>__fps8_s224_letterbox`），
  与 PE-AV `sequence_version` 互不混用；帧级 fp16、窗口级 fp32，bundle/manifest 原子写；
- v0.4.3.1 拆出**窗口层**：帧级特征只依赖 Frame Encoding Version；窗口层由
  Window Plan Version（`clip_seconds + overlap_seconds + weight_scheme`）决定，存
  `data/visual_windows/<window_plan_version>/<frame_version>/<work_id>/`；
  修改窗口/权重只重建窗口层与表征，不重新解码/编码帧；
- 源视频持久缓存在 `data/sources/`，`temp/` 只放切片/中间文件；
  `clean-temp` 不得删除 sources 缓存；
- 所有 npy / joblib / parquet 写入都走“临时文件 + 原子替换”，避免半文件；
  全片编码的 `.partial/batch_*.npz` 逐批原子写入，重启只补缺失批。

`logs/` 下的可重建报告（v0.2）：

~~~text
logs/
├── perf_<run_id>.csv            # A4：每部作品的分阶段耗时（pipeline 自动生成）
├── sync_report.json             # 音画同步抽查
├── ablation_report.json        # 静音/遮蔽画面消融
├── clip_stability.json/.csv    # A3：片段稳定性报告（tools/clip_stability.py 生成）
├── av_desync_experiment.json/.csv  # A6：音画错位实验（tools/desync_experiment.py 生成）
├── preflight_89.json           # 2025 选票扩量预检（tools/preflight_works.py 生成）
└── space_compare_*.json/.csv   # v0.4：两个表征空间的邻域迁移对比（tools/compare_spaces.py 生成）
~~~

这些文件只用于观察与证据，**不进入**向量库与星云；删除后可按原命令重新生成。

## 4. 关键不变量

这些不变量是系统正确性的核心，改动相关代码时必须同时考虑：

1. **元数据不参与编码，也不参与 UMAP。**
   `works` 表的标题、时长、发布日期、作者、排名、得票等只用于展示/染色/筛选。
   编码输入只有音视频文件；UMAP 输入只有 `work_joint` 向量矩阵。
   向量版本号只由模型与采样参数派生，不受元数据影响。
2. **向量版本只增不改。**
   `embedding_version` = 模型标签 + 帧数 + 片段长度/数量 + 音频采样率。
   同版本参数必须完全一致（`register_embedding_version` 会拒绝覆盖）；
   模型或采样参数升级 ⇒ 新版本目录，旧向量原样保留，由流水线为旧作品补算新版本。
3. **星云版本只增不改，且必须记录来源。**
   版本号形如 `map_vNNN`，单调递增；`map_runs` 记录 `embedding_version`
   与 UMAP 参数；`map_runs` 与 `map_coords` 在同一事务内登记。
   UMAP 重新训练会移动所有点，因此不同星云版本的 2D/3D 坐标**不可直接比较绝对位置**。
   P3 的 `transform()` 只向固定空间追加新点，旧点坐标保持不变（并整体写回）。
4. **处理状态机严格单调。**
   `pending → downloading → encoding → completed`；任一环节可进入 `failed`；
   `completed` 是终态、没有常规出边；`failed` 允许重试（回到 `pending/downloading/encoding`）。
   把 `completed/failed` 重置回 `pending` 只有一个显式入口 `Catalog.reopen()`（新向量版本、
   人工重试、进程中断恢复），它会写入 `reopen` 事件；其他任何迁移都被拒绝。
   非法迁移抛 `DatabaseError`；每次尝试与状态变化写入 `work_events`。
5. **去重基于稳定标识。**
   本地作品：`local:<相对 input/local 的 POSIX 路径>`；Bilibili：
   `bilibili:<BV号>:p<分P>`。同一作品跨年份/榜单只计算一次向量，
   `appearances` 保留全部榜单、年份、排名记录。
6. **采样确定性、无越界。**
   `plan_clips` 是纯函数：无随机数、同一时长在任何机器得到同一计划；
   片段全部落在 $[0, duration]$ 内（短视频长度按毫秒四舍五入）；短视频整段作为唯一片段；
   短于 `media.min_duration_seconds` 的视频记失败而不是产出空向量。
7. **先写文件、再登记、后清理，顺序保证可自愈。**
   单部作品的登记顺序固定为 `replace_clips -> register_embeddings -> complete`：
   向量文件先落盘，再登记片段计划和向量，最后迁移到 completed；登记成功后才删除该作品
   临时文件。失败时保留 `temp/` 供排查，绝不删除 `input/local/` 中的原始本地视频。
   若进程在登记与 `complete` 之间崩溃，下次运行会根据“片段记录 + 当前版本向量是否齐全”
   自愈（见第 5 节），而不是依赖进程退出状态。
8. **可复现。**
   模型 `revision` 固定、采样确定、UMAP `random_state` 固定且 `n_jobs=1`；
   所有版本参数落库/落盘，便于复算与对比。
9. **展示去重：canonical / duplicate 单层关联（v0.2）。**
   `work_links` 中每个 duplicate 只能属于一个 canonical，canonical 不能是任何人的
   duplicate（禁止链式）；duplicate 的作品记录、本地视频与向量保留，但**默认**不进入
   星云、近邻索引与界面；需要对照时可显式 `include_duplicates=True` 并记录在 filter_json。
   星云是构建时的快照：在建立身份关联**之前**生成的旧版本（如 P0 的 `map_v001`）
   保留当时的实体集合，不回溯改写；v0.2 起新构建的版本默认排除 duplicate，近邻索引
   与界面按当前 canonical 集合计算。
10. **筛选条件落库，筛选星云坐标不可比（v0.2）。**
    每个筛选星云是独立 `map_vNNN`，`filter_json` 记录 dataset/vote_year/list_key/
    include_duplicates；UMAP 在筛选后的 canonical 子集上重新训练。
    筛选版与全量版、不同筛选版之间的坐标**不可直接比较**；跨版本对比只作趋势观察。
11. **近邻、相似度与近邻密度只使用高维余弦（v0.2）。**
    `SimilarityIndex` 用 `work_joint/audio/video` 向量计算真实余弦；
    不允许用 UMAP 平面距离替代（UMAP 只近似保邻域，不保持距离度量）。
    近邻索引排除 duplicate，矩阵进程内缓存，缺失模态显式提示。
12. **分阶段计时只观测，不改变语义（v0.2）。**
    `pipeline_runs` / `work_timings` 记录 download/probe/extract/preprocess/forward/
    save/total 与批次模型加载耗时；负值被数据库层拒绝；计时字段不参与
    `embedding_version`、不进入向量与星云，HashTestEmbedder 允许为 0。
13. **片段稳定性只读、不参与坐标与质量评价（v0.2/A3）。**
    只读取 `clip_joint` 与 `work_joint` npy 做留一重聚合与 Top-k 对比，
    不加载模型、不重新编码、不写数据库；报告输出到 `logs/`，仅用于评估
    “作品近邻结论对单个 8 秒片段的依赖程度”，不是作品优劣指标。
14. **近邻解释仅用于展示（v0.2.x）。**
    `pair_percentile / is_mutual / hubness` 不改变近邻排序、坐标与向量；分位是
    “当前 `embedding_version` 的 canonical 集合”内的相对量，界面筛选/不同星云版本
    之间不可横向比较；小样本下 hubness 不稳定，不能据此评价作品质量或热度。
15. **片段级探索保持纯数据层（v0.2.x）。**
    `clip_heatmap_data / clip_pair_rows` 位于 `appdata`，只返回 dict/list；
    不得导入 Plotly/Streamlit 或接收 UI 对象；`similarity` 的片段矩阵按需加载并缓存。
16. **错位实验与主数据隔离（v0.2.x/A6）。**
    `nebula/avsync` 与 `tools/desync_experiment.py` 只在 `temp/` 生成变体并写
    `logs/` 报告；不写 `data/embeddings`、不登记新 `embedding_version`、
    不参与星云；真实运行需要模型，必须标 `model` marker，默认测试不加载权重。
17. **序列层与表征层分离，版本只增不改（v0.4）。**
    序列层 `data/embeddings/<sequence_version>/` 保存 PE-AV 片段/事件，编码一次后复用；
    表征层 `data/representations/<name>/<sequence_version>/` 只保存聚合结果，聚合器变化只新增
    `representations` 行。不同 sequence/representation 的向量**禁止混入同一 UMAP 输入或近邻矩阵**；
    legacy 目录只读，旧 `embedding_version` 与旧地图不迁移、不覆盖。
18. **地图必须绑定同序列代表征（v0.4）。**
    `map_runs.representation_id` 必须存在，且其 `sequence_version` 必须与地图的
    `embedding_version` 一致；旧地图允许不绑定（历史兼容），但新地图必须绑定；
    版本号 `map_vNNN` 只增不改，重复建图直接拒绝。
19. **全片覆盖与恢复语义（v0.4）。**
    `plan_full_coverage` 生成无间隙、无越界、末尾锚定窗口，`Σweight=1`；
    编码按批写入 `.partial/batch_*.npz`（原子），失败重启用 `sequence_states` 只补缺失批；
    完成后 `clips` 5 元组与 `clip_*` 行数一致，状态置 `assembled`。
20. **密集视觉层是独立命名空间与状态机（v0.4.3）。**
    `visual_version` 必须携带 encoder/revision + fps + size + preprocess
    （如 `encoder@revision__fps8_s224_letterbox`）；同版本参数不一致直接报错，
    模型 revision/fps/size/preprocess 任一变化都新建版本，禁止覆盖；
    `visual_versions`/`visual_sequences` 与 PE-AV `embedding_versions` 完全独立，
    视觉表征/地图的 `source_kind=visual` 只允许引用 `visual_versions`；
    `visual_sequences` 主键 `(work_id, version)`，状态 ∈ pending/in_progress/completed/failed，
    只处理 canonical + completed 作品、每部只下载一次、全部任务成功后清理 temp；
    **视觉向量/地图禁止与 PE-AV 序列/地图混入同一矩阵**；帧级冻结特征不是逐帧语义理解。
21. **三层版本正交与缓存隔离（v0.4.3.1）。**
    Frame Encoding Version（model+revision+fps+size+preprocess）只决定帧级特征；
    Window Plan Version（clip_seconds+overlap_seconds+weight_scheme，配置由
    `config.visual_window_plan_version` 派生）只决定窗口池化/权重；
    Representation Version 只决定聚合器；三者任一变化必须新建对应层版本，禁止覆盖；
    `visual_window_sequences` 主键 `(work_id, frame_version, window_plan_version)`，
    不同窗口计划不共享窗口/权重缓存；同一帧版本可配多个窗口计划，同一计划可跨多个帧版本；
    窗口覆盖权重写入 `window_weights.npy`/`manifest.json` 且 Σ=1（coverage-v1），
    作品向量必须按权重聚合；`data/sources/` 位于 data 下、不进入 temp 清理。
22. **v0.5-alpha 研究产物不进入正式层（负结果边界）。**
    token 数据集/checkpoint/metrics/研究向量只落盘到 `data/v05_tokens/`、`data/models/`、
    `logs/`、`data/representations/transformer_alpha_v1/`；**不注册 representations 表、
    不建 map、不覆盖 v0.4.x 表征/地图**；`nebula/v05_data.py` 只做数据构建，不 import torch/
    transformers；训练工具只读已有冻结特征，不重新编码 PE-AV/CLIP/DINOv2；v0.5-alpha 未通过
    可行性标准（负结果），不得在文档/界面宣称学到可靠时序结构、逐帧语义理解或风格分类。

## 5. 处理流程与失败语义

单部作品（`Pipeline._process_work`）的步骤与失败处理：

1. 检查 `embeddings` 是否已有当前版本 `work_joint` → 有则 `skipped`（断点续跑）；
2. 解析来源：本地文件校验存在性；Bilibili 先取元数据再下载到 `temp/`；
3. `ffprobe` 探测：时长、音视频流；无视频流/时长非法 → `MediaError`；
4. `plan_clips` 生成片段计划；过短 → `SamplingError`；
5. `extract_clip` 生成同步的 `clip_XX.mp4` + `clip_XX.wav`，校验时长；
6. `Embedder.encode_pairs` 批量推理（懒加载模型）；失败 → `EmbeddingError`；
7. `aggregate_work_vector` 聚合为作品级向量；
8. 原子写 npy，登记 `embeddings` / `clips`，更新元数据（时长、has_audio/has_video）；
9. `complete()`；删除该作品的临时目录。任一步失败 → `fail()` 记录可读原因并保留临时文件。
10. 各阶段耗时写入 `work_timings`（批次 `run_id` 由 `pipeline_runs` 登记），
    批次结束生成 `logs/perf_<run_id>.csv`；计时在任何失败路径上也尽量记录已发生阶段（v0.2/A4）。

批处理层面：继续处理其他作品，最终汇总“完成 / 失败 / 跳过”；网络下载失败或视频失效
只影响自身；`failed` 作品下次运行默认重试（可关闭），也可 `pipeline.py retry <work_id>` 单独重试。

**断点续跑与崩溃自愈**：选择待处理作品时，若当前 `embedding_version` 的 `work_joint`
已经存在，则不再重算；若同时状态残留为 `downloading` / `encoding`（上次进程在登记向量后、
`complete` 前中断），会按“片段记录是否完整”自愈：片段记录完整则补记 `completed`
（不重算、不增加尝试次数、写入状态事件；`downloading` 残留会先补一步合法的
`downloading -> encoding` 迁移）；片段记录缺失则 `reopen` 重新排队编码补全。
自愈过程自身失败时只记日志、保留原状态并把作品交回常规处理流程，不允许拖垮整批选择。
因此正常退出与崩溃恢复都以“向量 + 片段记录”为准，不依赖进程状态。

异常类型集中在 `nebula/errors.py`：可预期错误都继承 `NebulaError`，
CLI 顶层捕获后返回非 0 并给出中文提示；库代码用 `logging` 记录，不 `print`。

## 6. 可复现性

| 因素 | 控制方式 |
| --- | --- |
| 模型权重 | `model.checkpoint` + 固定 `model.revision`；缓存于 `model.cache_dir` |
| 采样 | `nebula.sampling.plan_clips` 纯函数；参数写进向量版本名与 DB |
| 推理 | CPU float32 默认；设备/精度策略固定；批大小不影响单个向量（聚合按片段） |
| UMAP | `metric=cosine`、`random_state=42`、`n_jobs=1`；2D/3D 独立训练；参数写入 meta 与 DB |
| 版本隔离 | 模型/采样参数变化 ⇒ 新 `embedding_version`；旧向量、旧星云保留 |
| 测试 | 默认测试离线且不加载权重；`HashTestEmbedder` 仅测试/演示，版本 `test-hash-v1`、`is_test_only=True` |

已知限制：UMAP 数值结果可能随 umap-learn / numpy 版本、平台浮点实现有微小差异；
跨版本比较应使用“同一环境内重新生成”的结果，且不同星云版本本身不可比绝对坐标。
PE-AV 的语义质量不在自动化测试范围内，P0 只做“静音/遮蔽画面后联合向量应变化”的消融抽查。

真实数据运行层的已知边界：

- 8.7 小时的活动直播回放（BV1b6ivBzEpb）全量下载代价过高：本轮记录为 `failed` 并保留
  可读原因，不阻塞其他作品；后续可改用分段下载/只取片段；
- 竖屏视频曾因下载格式的宽高筛选条件失败（height 限制不适用于竖屏），
  `config.yt_dlp_format` 已增加 `width<=480` 与 `best` 兜底并重试成功；
- 真实 P0/P1 的耗时主要来自下载与推理，用 `pipeline.py timings` 与
  `logs/perf_<run_id>.csv` 定位瓶颈，避免凭感觉调参。



### 6.1 数值一致性边界（device / dtype）

PE-AV 向量存在**硬件相关的数值差异**，这决定了什么可以比较、什么不可以：

- 同一设备、同一 dtype、同一模型 revision、同一采样参数下重复运行：向量完全一致
  （实测 max abs diff = 0），这是 `embedding_version` 可复现性的基础；
- CUDA bfloat16 autocast 与 CPU float32 之间：向量方向高度一致但并非 bit 级一致，
  实测 joint 向量余弦相似度约 0.9962（约 1% 差异），单模态向量同理；
- 因此：
  - `embedding_version` 只编码模型 + 采样参数，**不编码 device/dtype**——
    device/dtype 是运行时条件，不改变“这一版向量语义”的定义；同一版本在不同硬件上
    生成的数据可以共存，但对比时要知道它们不是 bit 级可复现的；
  - 跨硬件 / 跨 dtype 的星云坐标、距离、密度只作**趋势比较**，不承诺数值复现；
  - 需要严格复现实验（例如对比两次训练、两个参数或两个模型版本）时，应在同一硬件、
    同一 dtype 下重算，并保留版本与参数记录；
  - P0 的消融实验（静音/遮蔽画面）只回答“联合向量是否变化”，不依赖跨硬件可比性。

### 6.2 A5 低风险提速实测（v0.2）

提速原则：**不改变 `embedding_version` 的行为才可直接采纳**；任何会改变向量的优化必须
发布新向量版本并整批重跑。以下数据来自 T1 的基准工具（`tools/bench_peav.py` /
`tools/verify_peav.py`）在真实视频上的实测：

| 优化项 | 实测结果 | 结论 |
| --- | --- | --- |
| 切片 mp4 去掉 AAC 音轨 | 真实 3 部作品 4 类向量 sha1 全等，max abs diff = 0；joint cos = 0.99999994 | 零漂移，v0.2 起默认采纳 |
| 重复 ffprobe | 作品级探测结果通过 `extract_clip(info=...)` 复用，重复探测已消除 | 默认采纳 |
| ffmpeg preset veryfast → ultrafast | 5 片段 11.07s → 5.88s（1.88×）；临时体积约 2.35×；抽样 joint cos 最低 0.99918 | 本轮不改变默认（体积换时间，收益不足以改方案） |
| 片段分辨率缩放到高 336 | extract 12.88s → 4.43s（2.91×）；preprocess 3.52s → 1.03s（3.43×）；单作品 19.53s → 8.26s（2.36×）；work joint cos 最低 0.99812 | 会改变向量，需新 `embedding_version` + 整批重跑，留作 v0.3 后续评估 |

结论：v0.2 只采纳“零向量漂移”的两项（去 AAC 音轨、复用 ffprobe），其余作为 v0.3 的候选，
通过新向量版本对照验证后再切换。

### 6.3 A6 音画错位实验（v0.2.x）

在真实作品上量化 PE-AV 联合向量对“音画时间对齐”的敏感度：同一片段分别构造
A 原始、A-quality 轻画质退化基线、B 音频 +0.5s、C 音频 +1.0s、D 同作品换其他片段音频，
比较 `Δ = 1 - cos`（joint 为主，audio/video 用于定位模态）。方法、选项与解读局限见
`docs/adr/0008-av-sync-desync-experiment.md`。

- 实验只写 `logs/av_desync_experiment.json/.csv` 与中文摘要，不写主向量库、不生成新
  `embedding_version`、不参与星云；真实运行需要 GPU/权重并标 `model` marker；
- A-quality 提供“重编码漂移”基线，只有超过它的 Δ 才能归因于音画错位；
- 局限：仅约 10 部作品、单次实验、周期性音乐可能把错位落回相似小节，结论只作趋势说明，
  不作为作品质量或模型优劣判断；完整数据与逐作品明细见 ADR-0008 与
  `logs/av_desync_experiment.json`。

**2026-10-25 实测（10 部 / 48 片段，CUDA bf16 autocast）**：

| 变体 | joint Δ mean | joint Δ median | 相对 quality | 机制 |
| --- | --- | --- | --- | --- |
| Q 轻画质退化（CRF28） | 0.000964 | 0.000695 | 1×（噪声基线） | video 分支有小幅变化，audio 不变 |
| B 音频 +0.5s | 0.004222 | 0.003260 | 4.38× | audio 分支敏感，video=0，joint 随音频变化 |
| C 音频 +1.0s | 0.002364 | 0.001249 | 2.45× | 周期性音乐整拍偏移，Δ 反而更小 |
| D 换同作品其他片段音频 | 0.087575 | 0.079961 | 90.85× | audio 大幅变化，video=0，joint 明显变化 |

逐片段高于 quality 的比例：B 47/48、C 36/48、D 48/48；10 部逐部均值排序均为
`swap >> shift0.5s > shift1.0s > quality`，无反例。结论：联合向量确实融合音画两路
（纯音频变体 video Δ 严格为 0），轻微错位 Δ 显著高于重编码噪声但绝对值很小；
`单点 Δ 不能证明同步敏感度`，需结合聚合分布与 swap 对照解读。

### 6.4 v0.4 全片覆盖实测（2026-10-25，真实数据 110/110 canonical）

完整分析见 `docs/reports/full-coverage-analysis.md` 与 `logs/v04_data_analysis.json`；
原始对比 JSON 在 `logs/space_compare_*.json/.csv`。核心数字：

| 指标 | legacy | full（full_mean/temporal） | 336p 对照（10 部） |
| --- | --- | --- | --- |
| canonical 作品 | 110 | 110 | 10 |
| 片段/窗口总数 | 536（≤5/部） | **4130**（min 3 / mean 37.55 / max 214） | 164 |
| 覆盖率 / 末尾锚定 / 权重和 | — | 110/110；Σweight∈[0.99999992, 1.00000003] | 10/10 |
| 事件三件套 | 无 | 110/110（`events_v1`，约 131MB） | 10/10 |
| 单部耗时（均值） | total 16.57s | total 70.35s（extract 24.96 / events 7.53 / forward 22.45） | total 31.85s（1.60× 加速） |
| 存储 | legacy 8.4MB | 序列层约 187MB + representations 9.7MB | h336 6.8MB |

- 顺序敏感性（20 部反转顺序）：Full Mean cos_min=0.99999994（距离≤6e-8）；
  Temporal Stats cos_min=0.67095 / mean 0.67777（1-cos 0.287–0.329）→ 时间结构确实进入表征；
- 336p：work_joint cos min=0.997156 / mean=0.998590；video min=0.977239 / mean=0.986168；
  extract 25.08→11.09s、total 51.08→31.85s → 可加速但视频分支变化不可忽略，**不转正**；
- 邻域迁移（110 部）：legacy×full_mean Top5 平均 Jaccard=0.4068、完全稳定 8/110、
  作品对余弦 Pearson=0.9569；full_mean×temporal Jaccard=0.2049、稳定 0/110、Pearson=0.8294；
  legacy×temporal Jaccard=0.1734。全片覆盖与 Temporal Stats 都构成明显不同的邻域空间；
  具体被替换作品（如 work 8 旧 Top1=104→新 Top1=35，仅保留 1/5 邻居）见报告 §3；
- 默认 `sampling.mode` 已切到 `full_coverage`；地图默认 `map_v012`（Temporal Stats），
  Legacy 空间仍保留 `map_v008`。

### 6.5 能力边界与命名（v0.4）

v0.4 的准确名称是**「全片覆盖与多尺度音画特征表征系统」**，
不是「逐帧音MAD语义理解系统」。系统记录两条互补但信息量有限的视觉/音画通道：

1. **PE-AV 稀疏联合语义通道**：每 8s 窗口只输入 16 帧（约 **2fps**）给 PE-AV，
   输出 1024 维 joint/audio/video embedding。它捕捉的是窗口级、稀疏采样的联合语义，
   **不是逐帧画面、动作、歌词或风格理解**；
2. **高频事件统计通道**：视频按 **30fps** 缩放到 **96×54 灰度**计算帧差/运动/亮度/cut 等统计，
   音频按 **100Hz** 计算能量/频谱/onset 等；只记录节奏、运动与突变结构，**不含视觉语义**。

| 能力层 | v0.4 状态 | 说明 |
| --- | --- | --- |
| 完整时间覆盖 | ✅ 完成 | 任意时刻 ≥1 窗口、末尾锚定、权重归一（110/110；见 §6.4 与报告） |
| 高频音画变化记录 | ✅ 完成 | 100Hz 音频 / 30fps 96×54 灰度统计；事件层只是节奏/运动信息 |
| 时间结构统计 | ✅ 完成（固定统计） | Full Mean 等权平均；Temporal Stats 是人工设计分组与权重的统计聚合基线，**不是学习模型** |
| 逐帧视觉语义编码 | ❌ 未完成 | PE-AV 16 帧/8s ≈2fps，无法逐帧理解 |
| 可学习音画时序关系 | ❌ 未完成 | 本轮无训练目标/可学习融合，只有固定统计 |
| 自监督全曲表示 | ❌ 预留（v0.5） | 仅保留数据接口与 shim 字段，未训练、不做质量结论 |

因此：**完整覆盖 ≠ 逐帧语义理解**。文档、界面文案与汇报中不得把 v0.4 描述为
“逐帧音MAD语义理解系统”；若需要逐帧/可学习时序能力，属于 v0.5 及以后的研究范围。
v0.4.3 新增的紧凑视觉通道（§6.6）用冻结 CLIP/DINOv2 提供 8fps 帧级特征，只是把视觉时间密度
从 PE-AV 的 ≈2fps 提高到 8fps（30fps 仅子集），**仍然不是逐帧语义理解**，也没有可学习时序模型。

### 6.6 v0.4.3 密集视觉语义通道（Dense Visual）

在不改变 PE-AV 联合表征的前提下，新增一个**独立、冻结编码器**的密集视觉通道：

| 编码器 | 维度 | revision | 定位 |
| --- | --- | --- | --- |
| `openai/clip-vit-base-patch16` | 512 | `57c216476eefef5ab752ec549e440a49ae4ae5f3` | 图文对齐语义（主通道） |
| `facebook/dinov2-small` | 384 | `ed25f3a31f01632728cabb09d1542f84ab7b0056` | 自监督视觉结构（对照） |

- **处理**：单次 ffmpeg 流式解码（不整片进内存）→ 按 `visual_dense.fps`（默认 8fps）、
  `size=224`、`preprocess=letterbox|stretch` 取帧 → 冻结编码器逐帧输出
  L2 归一化向量（帧级 **fp16** 落盘）→ 按 v0.4 的 8s 全片窗口做重叠时长加权池化，
  得到窗口级 `windows`（fp32，与 PE-AV 窗口一一对应）与可选 `window_std`；
- **版本命名**：`encoder@revision__fps{fps}_s{size}_{preprocess}`，
  例 `openai/clip-vit-base-patch16@<rev>__fps8_s224_letterbox`；encoder/revision/fps/size/
  preprocess 任一变化必须新版本；与 PE-AV `sequence_version` 完全独立；
- **存储**：`data/visual_dense/<visual_version>/<work_id>/` 下
  `dense_visual.npz` + `windows.npy` + `window_std.npy` + `manifest.json`；
- **状态与批处理**：`visual_versions`/`visual_sequences` 记录版本与作品×版本状态；
  `tools/visual_pipeline.py run` 只处理 canonical + completed 作品、每部只下载一次、
  支持断点续跑，全部任务成功后清理 temp，失败保留 temp；30fps 只对配置子集
  （`subset_fps/subset_count`）运行；
- **表征与地图**：窗口级向量登记为 `kind=visual` 表征
  （`dense_visual_clip_v1` / `dense_visual_dino_v1`），可在星云中构建
  CLIP-only / DINOv2-only 对照空间；视觉地图 `source_kind=visual`、坐标与 PE-AV 地图不可比；
- **诊断**：`tools/embedding_diagnostics.py` 对既有表征做 PCA / UMAP(15/30/50)、
  Top-10 保持率与距离相关，判断低维投影是否保真；
- **pilot 成本**（3 部本地真实作品，RTX 3060 12GB，8fps/224px/batch32/bf16）：
  单部墙钟 CLIP 2.46/7.14/17.98s、DINOv2 1.57/5.64/14.89s；GPU 峰值 782MB/208MB；
  全量 110 部估算 CLIP 约 39min、DINOv2 约 32min（不含 Bilibili 重新下载）；
  详见 `docs/reports/dense-visual-pilot.md`；
- **全量结果（2026-10-26，110/110 canonical）**：4 个 visual_versions、**252/252 任务 completed、0 failed**；
  8fps 每个编码器 110 部 / **260,885 帧 / 4,130 窗口**，30fps 子集每编码器 16 部 /
  140,149 帧 / 593 窗口；存储 **673.9MB**；`dense_visual_clip_v1`(id=4) /
  `dense_visual_dino_v1`(id=5) 各 110 向量；`map_v013`（CLIP-only）/
  `map_v014`（DINOv2-only）各 110 部；宽高比对照采用 letterbox（CLIP 窗口余弦 mean 0.9425、
  DINO 0.8725；帧数一致）；视觉 vs PE-AV 窗口 RSA：CLIP-video mean 0.6949、DINO-video 0.7246；
  30fps 窗口级饱和（8fps↔30fps 余弦 CLIP 0.9998、DINO 0.9994）；v0.4.3 有 5 个表征空间可切换，
  v0.4.3.1 重建后视觉 v2 空间加入并与 v1 并存（以实际登记表征/地图为准）；
  默认仍为 PE-AV 优先（`map_v012`）。详见
  `docs/reports/visual-dense-full-run.md`（数字已回填、t4 复核）；
- **边界**：帧级冻结特征提高时间密度，但仍是“特征”而非“逐帧语义理解”，也没有可学习的
  音画时序关系；30fps 只用于帧级轨迹研究，不默认全量。

### 6.7 窗口计划层与跨模型共识（v0.4.3.1 / v0.4.4）

**v0.4.3.1：覆盖权重修复与三层版本分离**

- 问题：v0.4.3 的窗口视觉向量未保存 `plan_full_coverage` 的覆盖权重，构建 v1 表征时
  退化为等权；且视觉版本号不含窗口时长/overlap/权重方案，改窗口计划会复用旧版本；
- 三层版本：
  | 层 | 内容 | 示例 |
  | --- | --- | --- |
  | Frame Encoding Version | model+revision+fps+size+preprocess | `openai-clip-vit-base-patch16-<rev>__fps8_s224_letterbox` |
  | Window Plan Version | clip_seconds+overlap_seconds+weight_scheme | `win8s_ov0__coverage-v1` |
  | Representation Version | 聚合器 + 引用的窗口计划 | `dense_visual_clip_v2` / `dense_visual_dino_v2` |
- 窗口层：`tools/build_visual_windows.py` 从已保存帧级特征重建
  `data/visual_windows/<window_plan_version>/<frame_version>/<work_id>/`
  （`windows.npy`/`window_std.npy`/`window_weights.npy`/`manifest.json`），
  不下载/不解码/不跑模型；DB 用 `visual_window_plans`/`visual_window_sequences`（schema v5）；
- 权重：`nebula.sampling.coverage_weights` 与 `plan_full_coverage` 同源，
  写入窗口层；作品向量按 `Σ w_i * window_i` 聚合（`weight_scheme=coverage-v1`）；
  `weight_scheme=equal` 仅用于复现旧行为/对照；
- 源缓存：`data/sources/` 持久保存 Bilibili 源视频，`temp/` 只放切片；
  重建不重复下载，`clean-temp` 不影响 sources；
- v2 表征/地图：v1（`dense_visual_clip_v1`/`dense_visual_dino_v1`、map_v013/v014）
  保留只读；v2 另建表征与地图（`dense_visual_clip_v2` id=6 → map_v015；
  `dense_visual_dino_v2` id=7 → map_v016），不覆盖旧版本，坐标跨版本不可比；
- **真实结果（2026-10-26）**：window plan=win8s_ov0__coverage-v1；252/252 窗口任务 completed、
  0 failed，共 9446 窗口（8fps 4130×2、30fps 593×2）；weight_sum 0.999999847..1.000000209；
  v2 表征各 110 joint；v1→v2 逐作品余弦 CLIP min 0.999972 / DINO min 0.999863；
  Top-5 Jaccard CLIP 0.9939（稳定 108/110）/ DINO 0.9970（稳定 109/110）——
  **本数据 8s/ov0 下影响很小，但修复了权重正确性，overlap>0/短窗口下差异会放大**；
- 边界：这是同一帧级特征上的**权重修复**，不改变模型能力，不宣称新的语义理解。

**v0.4.4：跨模型近邻共识与轻量融合**

- 目标：用现有高维表征回答“多个模型共同认可的相似作品是否更有解释力”，不训练新模型；
- 方法：对 `full_mean_v1` + `dense_visual_clip_v2` + `dense_visual_dino_v2`
  （可配）计算 Top-k 有向/互为近邻，统计跨空间保留率、>=2/>=3 空间共识边、覆盖与强度；
  另做 rank-average 与 concat 轻量融合基线（不写 DB）；
- 工具：`tools/build_consensus_graph.py` → `logs/neighbor_consensus.json/.csv`；
- 界面：侧栏“显示高维互为近邻连线”（默认关，只画 mutual，边数截断提示）与
  “仅显示跨模型共识边”（共识 JSON 缺失时回退并提示）；详情显示共识近邻摘要；
  PCA 3D 仅辅助观察，不改变默认 UMAP 坐标语义；
- 边界：共识是观察性邻域一致性证据，不是风格/流派分类；任何正式融合表征需新建版本并重新验证。

### 6.8 v0.5-alpha 分层音画时序模型（负结果，研究产物）

- 目标：在冻结 PE-AV/CLIP/DINOv2/事件特征上验证“局部音画对齐 → 全曲时序聚合”的可学习时序表征；
- 数据：`tools/build_v05_tokens.py` 生成 `data/v05_tokens/v05_alpha_v1/`
  （110 部；PE-AV 窗口 4,130；max_segments=128 截断后 token 段 3,904（6 部截断、丢 226 段）；
  141.1MB；0.5s×16bin/段；fp16 npz+manifest），不重新编码；
- 模型：`nebula/v05_model.py` 局部多模态 Transformer（256d/4heads/2 层）+ 全曲 2 层；
  三头：masked segment（预测 PE-AV joint）、AV mismatch（batch 内音频互换负例）、order recovery；
- 训练/验证：`tools/train_v05.py` 确定性 90/20 split（seed=42）、变长 padding+mask、
  AdamW lr=1e-3、batch=4、mask_ratio=0.2；产物 checkpoint/metrics/研究向量；
- 两次真实训练结果（v1 初始；v2 加 VICReg + mask 残差头）：
  - v1：val mask cosine 0.8149（基线 0.8249）、order 0.05、AV 0.80/AUC 0.729、
    有效秩 1.03、Top-5 Jaccard 0.0447、reverse drop -0.0053；
  - v2：val mask cosine **0.8131**（基线 0.8249）、mask MSE 0.000365、order **0.10**、
    AV **0.40**/AUC 0.635、有效秩 **4.66**、Top-5 Jaccard **0.0468**、
    reverse drop **-0.0048**、段序 work cosine 0.999963/0.999929；
  - **结论：未通过 alpha 可行性标准**；工程链路完整、v2 把 rank 从 1.03 改善到 4.66，
    但 temporal 表征没有学到可用顺序/跨模态结构，mask 不如常量均值基线；
- 原因：8s PE-AV 目标被全局风格主导、段序几乎不影响 work vector、AV 负例可走捷径、
  数据/独立性不足、音频只有事件统计；详见 `docs/reports/v05-alpha-validation.md`；
- 下一步：补密集音频语义、局部 visual↔audio InfoNCE、相对时间 order、whitening/duration 去偏、
  更多数据/30fps 子集；达标前不注册主表征/地图。

### 6.9 v0.4.5 意外关系发现 / v0.5-beta 密集音频与局部双塔

**v0.4.5 高维意外关系发现**

- 目标：用高维余弦的**排名百分位**发现“多模型下意外关系”（不拿原始余弦跨模型相减）；
- 空间：full_mean_v1（joint/audio/video）、temporal_stats_v1（joint）、dense_visual_clip_v2、
  dense_visual_dino_v2；canonical 共同作品按 work_id 对齐；
- 分数：p=1-(r-1)/(N-2)，r 为不含自身的近邻排名（第 1=1），无向对取 bar_p=(p(i,j)+p(j,i))/2；
  模式 A joint vs audio/video 平均、模式 B temporal vs full_mean、模式 C PE-AV/CLIP/DINOv2
  mutual top-k 对比（common/clip_only/dino_only/peav_only/visual_vs_peav）；只用 p 差/组合；
- 模块：`nebula/discovery.py`、`tools/explore_relations.py` →
  `logs/unexpected_relations.json` + CSV；界面“意外关系探索”（模式切换、作品对表、
  选中高亮、Top-k、互邻筛选、CSV 导出）；
- **真实结果（2026-10-26）**：6 空间 / 110 部 / skipped=0；模式 A 950 候选、模式 B 864 候选；
  模式 C common 99 对（其中 >=3 空间 70 对）、visual_vs_peav 19 对；百分位公式逐项复核；
- 边界：染色/标题/外部元数据不进入分数；不写 DB/不建 map/不加载模型/不联网；
  不改变默认地图/7 空间行为。

**v0.5-beta 密集音频与局部音画双塔**

- 目的：针对 alpha 失败原因补密集音频语义与真正的局部 AV 配对目标；结果不预设为正；
- 音频：`m-a-p/MERT-v1-95M` @ `12af15fe...`，第 9 层、24kHz、5s ctx/hop、
  75Hz/768d、0.5s bin、mean_std、l2；版本
  `...__sr24000_ctx5_hop5_l9_poolmean_std_bin0.5_norml2`；
  **Pilot 20/20 PASS**（2582.753s、193,179 帧/5,177 bins、RTF 0.00947、显存 1,320.4MB、
  314.1MiB）；**全量 110/110 ok、failed=0**（32,619.14s、2,439,577 帧/65,291 bins、
  RTF 0.00812、显存 1,320.4MB、4,158,970,616B≈3.87GiB）；`summary.pass=true`
  （帧映射 110/110），**strict_container_alignment=false 108/110**：work 50/99 音轨尾部
  短 1.95s/1.50s、末尾 3 个 0.5s bin mask=False（无伪造值），已从 beta 排除；
- 局部双塔：`nebula/v05_beta.py` + `tools/train_v05beta.py`；v05_beta_v1 全量 token
  （110 部/4,130 段、0 截断/142.3MiB）；音频可用 107 部（排除 50/99 tail-mask、122
  no_anchor_with_negative），按 canonical work_id 划分 **74/21/12**（test 12 部/约 4,822 anchors）；
  视觉 CLIP + 音频 MERT 双塔 MLP（128d/L2），InfoNCE t=0.07，3 seeds×30 epochs；
- **真实结果（3 seeds，test）**：Top-1 0.296557/0.302779/0.305268（均值
  **0.301535±0.003663**）；vs random **+0.03108**、vs frozen **+0.02779**、
  vs simple_sync **+0.00912**；消融 seed42：normal 0.296557、audio_zero/visual_zero 0.272916、
  audio_replacement 0.268768、time_shift 0.280796；
- **结论：weak positive / not passed**——方向一致但相对 simple_sync 仅 +0.91pp、CI 宽、
  best epoch 1–6，证据不足以宣布学会可靠局部音画对应；产物仅
  `data/models/v05_beta_<seed>/` 与 `logs/`；**不注册 DB/不建 map、
  不进入全曲编码、不改 v0.5-alpha 负结果**。

## 7. 扩展点

- **升级模型**：改 `config.yaml` 的 `model.checkpoint/version/frames`（Base 版本同理）
  → 自动得到新 `embedding_version`，旧向量保留；重跑流水线后 `map.py build` 生成新星云。
- **调整采样**：改 `media.clip_seconds/clips_per_work/audio_sample_rate` → 新向量版本；
  不要手改已发布版本的数据。
- **新增染色维度**：在 `nebula/appdata.py` 的字段列表与 `apply_color` 中实现；
  若数据来自已有列则无需数据库迁移，否则需新增 schema 迁移（`schema_meta` 版本）。
- **新增榜单类型**：`lists.kind` 的 CHECK 约束目前是 `votes/weekly/manual`，
  扩展需要 schema 迁移并更新 `ensure_list` 校验。
- **P3 增量**：`mapplot.transform_new` 使用已保存的 UMAP 模型；累计新增较多后可以
  `build_map` 重新训练并保留旧版本对比。
- **多模态消融**：`media.mute_audio_clip` / `media.black_video_clip` 生成对照素材，
  由流水线消融实验使用；不应把消融向量写入主向量版本。

## 8. 目录与职责速查

| 路径 | 说明 |
| --- | --- |
| `nebula/config.py` | 唯一配置来源；所有相对路径相对 config.yaml 解析 |
| `nebula/db.py` | Catalog：作品、榜单、向量、状态、星云版本 |
| `nebula/sampling.py` | 时间轴采样规则：legacy `plan_clips`（≤5×8s）与 v0.4 全覆盖 `plan_full_coverage`（无间隙/末尾锚定/权重和=1） |
| `nebula/media.py` | ffprobe/ffmpeg 封装 |
| `nebula/embed.py` | PE-AV / 测试嵌入器、聚合、npy 读写 |
| `nebula/pipeline.py` | 单部处理与批处理编排 |
| `nebula/ingest.py` | 本地扫描、选票 CSV、周报导入 |
| `nebula/download.py` | Bilibili 元数据 API 与 yt-dlp 下载 |
| `nebula/similarity.py` | 高维余弦近邻索引、三模态 Top-k、分位/互为/hubness、片段级相似（A1/A2/P2） |
| `nebula/identity.py` | 标题规范化、本地↔BV 身份候选/关联、向量复制（B1） |
| `nebula/stability.py` | 片段留一重聚合与 Top-k 稳定性报告（A3） |
| `nebula/avsync.py` | 音画错位实验：WAV 偏移/交换、画质退化基线、Δ 计算与报告（A6） |
| `nebula/events.py` | 高频音画事件扫描（音频 100Hz/视频 30fps、音画互相关、窗口 32 维特征，v0.4.1） |
| `nebula/temporal.py` | Full Mean / Temporal Stats 聚合（v0.4.2） |
| `nebula/representations.py` | 表征注册/解析/保存/加载；legacy 与派生表征隔离；pe_av/visual 来源 |
| `nebula/dense_visual.py` | v0.4.3 冻结视觉编码器（CLIP/DINOv2）、流式解码、帧级特征、窗口池化、bundle 保存 |
| `nebula/visual_pipeline.py` | v0.4.3 密集视觉批处理：版本/任务状态、单次下载、断点续跑与清理 |
| `nebula/mapplot.py` | UMAP 训练、筛选/变体版本落盘、transform、按表征建图（含视觉空间） |
| `nebula/appdata.py` | 可视化数据准备（纯函数，含近邻密度与筛选） |
| `app.py` | Streamlit 应用（近邻面板、密度、筛选） |
| `ingest.py/pipeline.py/map.py/qa.py` | 顶层 CLI（含 link-local、variants） |
| `tools/clip_stability.py` | A3 稳定性报告 CLI（只读、离线） |
| `tools/desync_experiment.py` | A6 音画错位实验 CLI（真实模型，输出 logs） |
| `tools/preflight_works.py` | 2025 选票 89 部扩量预检 CLI（只读 DB，真实 API） |
| `tools/build_representation.py` | v0.4 表征构建 CLI（full_mean / temporal_stats，只读序列、不重编码） |
| `tools/compare_spaces.py` | v0.4 两个表征空间的邻域迁移对比（Jaccard/稳定对/余弦相关/hubness） |
| `tools/events_probe.py` | v0.4.1 事件扫描探针/预览 CLI（离线合成或单部作品） |
| `tools/visual_pipeline.py` | v0.4.3 密集视觉批处理 CLI（run/status/list/clean-temp） |
| `tools/build_visual_windows.py` | v0.4.3.1 从已保存帧级特征重建窗口层（窗口计划/覆盖权重，不下载/不解码/不跑模型） |
| `tools/build_consensus_graph.py` | v0.4.4 跨模型互为近邻/共识边与轻量融合评估（只读表征，不写 DB） |
| `tools/build_v05_tokens.py` | v0.5-alpha 局部音画 token 数据集构建（只读冻结特征，不重新编码） |
| `tools/train_v05.py` | v0.5-alpha 研究模型训练/验证（checkpoint/metrics/研究向量；不注册 DB/地图） |
| `tools/explore_relations.py` | v0.4.5 意外关系发现 CLI（排名百分位模式 A/B/C；不写 DB/不建 map） |
| `tools/audio_dense.py` | v0.5-beta MERT 密集音频特征 pilot（本地作品，不写 DB/不改事件特征） |
| `tools/train_v05beta.py` | v0.5-beta 局部音画双塔训练/验证/消融（不注册 DB/地图） |
| `nebula/v05_data.py` / `v05_model.py` / `v05_train.py` | v0.5-alpha token 构建 / 模型 / 训练验证 |
| `nebula/discovery.py` / `audio_dense.py` / `v05_beta.py` | v0.4.5 发现 / v0.5-beta 音频编码 / 局部双塔 |
| `tools/dense_preprocess_compare.py` | v0.4.3 宽高比预处理对照（stretch vs letterbox） |
| `tools/embedding_diagnostics.py` | v0.4.3 PCA/UMAP 低维投影诊断（Top-10 保持率、距离相关） |
| `tools/bench_ffmpeg.py` / `bench_preprocess.py` / `bench_peav.py` / `verify_peav.py` | 提速/推理基准与真实模型验证证据脚本 |
| `tests/` | 单元、不变量、E2E 测试；`tests/fixtures/` 合成视频与临时配置 |
| `docs/` | 契约（V1/V2）、ADR、质量规范、执行方案 |

## 9. 验证策略

- **唯一门禁**：`python qa.py`（ruff + pytest 快速离线 + 文档/入口/配置检查）；
  `--full` 额外运行 `model` / `network` 标记测试；
- **测试分层**：单元（config/db/sampling/media/embed/ingest/mapplot/appdata/similarity/
  identity/stability/avsync/events/temporal/representations/dense_visual/visual_pipeline/
  embedding_diagnostics）、不变量（`tests/test_invariants.py`：
  模块边界、版本、状态机、去重、schema v4/身份关联/计时/filter_json/采样/gitignore/CLI/
  近邻解释字段/片段数据层/avsync 导入边界/模式切换/表征隔离/地图绑定/迁移幂等/
  视觉版本与序列状态/视觉表征地图绑定）、
  端到端（`tests/test_e2e.py`：合成视频 → 流水线 → 向量 → 近邻解释/稳定性 → 筛选星云 →
  appdata → Streamlit AppTest；v0.4 全片编码 → Full Mean 表征 → 表征地图 → 空间对比）；
- **v0.2.x 独立验收**：全库分位/互为 Top-k/hubness 与手算全库成对相似度矩阵对照；片段热图/最高配对
  schema 与手算片段余弦对照；AppTest 检查解释列、枢纽表、片段热图与模态切换；
  A6 实验检查 Δ 数学与“不写主向量库”；扩量预检 JSON 与 DB 状态一致性核对；
- **v0.4 独立验收**：模式切换版本公式/格式、不同表征与序列隔离、地图代表征绑定与旧地图坐标不可覆盖、
  schema v3 迁移幂等；全链路离线（HashTestEmbedder + 4s/12s/17s/40s 合成视频）覆盖中断恢复、
  Full Mean 构建、表征地图、appdata 数据帧与 `tools/compare_spaces.py` 输出结构；
  真实数据验收（2026-10-25，110/110 canonical）：覆盖率 110/110、4130 窗口、权重和
  ∈[0.99999992,1.00000003]、事件三件套 110/110；三组 compare（legacy×full_mean Jaccard=0.4068/
  Pearson=0.9569；full_mean×temporal Jaccard=0.2049/Pearson=0.8294；legacy×temporal Jaccard=0.1734）；
  336p 对照（10 部：joint cos min=0.9972、video min=0.9772、total 51.08→31.85s）；
  真实 AppTest 默认 `map_v012`（Temporal Stats），三空间切换 map_v011/map_v008/map_v012，
  时间轴与 full 序列片段面板正常（8 PASS / 0 FAIL）；报告见
  `docs/reports/full-coverage-analysis.md`；
- **v0.4.3.1 独立验收**：`coverage_weights` 与 `plan_full_coverage` 逐项一致且 Σ=1；
  `save_bundle` 的 `window_weights` roundtrip/旧 manifest None 兼容；
  `visual_window_plans`/`visual_window_sequences` 主键/参数冲突/状态；
  帧编码版本与窗口计划正交（同一帧可配多计划、同一计划可跨帧版本）；
  不同窗口计划不共享缓存；`data/sources` 在 data 下且不被 `clean-temp` 清理；
  schema v4→v5 迁移幂等；v1 表征/地图只读保留；v2 权重重建不重跑视觉模型；
- **v0.4.3.1 真实数据验收（2026-10-26）**：window plan=win8s_ov0__coverage-v1（DB 仅 1 个）；
  252/252 窗口任务 completed、0 failed、9446 窗口；weight_sum 0.999999847..1.000000209；
  coverage-vs-equal max 0.022221/mean 0.001713（78/252 差>1e-3）；v2 表征 id=6/7（512/384）、
  各 110 joint；v2 地图 map_v015/map_v016=110；逐作品余弦 CLIP 0.999999/0.999972、
  DINO 0.999996/0.999863；Top-5 Jaccard CLIP 0.9939（稳定 108/110）/DINO 0.9970（109/110）；
  FullMean×CLIP v2 0.3355/0.7342，FullMean×DINO v2 0.3065/0.7596；详见
  `docs/reports/v0431-window-weights-and-versioning.md`；
- **v0.4.4 独立验收**：`tools/build_consensus_graph.py` 离线导入（不加载模型/UMAP/界面/写 data）；
  mutual/共识/rank-average/concat 指标与 JSON schema 由合成数据单测；界面互为近邻/共识叠加、
  截断、缺失回退、3D 正常；
  **真实结果（2026-10-26）**：full_mean_v1 / CLIP v2 / DINO v2 的 mutual 边 61/75/73；
  >=2 共识边 53（>=3 全共识 22），覆盖 44/110（40.00%），平均强度 0.8050；
  跨空间 mutual Jaccard 0.3204/0.2407/0.3455；rank-average 融合 Jaccard(vs spaces)=0.5331、
  vs 共识=0.4303；concat(equal) 0.5358/0.4106、concat(inv_sqrt) 0.5385/0.4121；
  真实 AppTest 默认 map_v012、7 空间切换（v008/v011/v012/v013/v014/v015/v016）、互邻 k=5、
  共识阈值 2/3 均通过；详见 `docs/reports/v044-consensus-fusion.md`；
- **v0.4.3 独立验收**：视觉版本参数冲突/命名含 revision+preprocess（DB/目录 slug）、
  visual_sequences 状态/主键/断点、`kind=visual` 表征登记与视觉地图绑定
  （`source_kind=visual`）、不同视觉版本矩阵隔离；
  真实数据复核（2026-10-26）：4 visual_versions / 252/252 completed / 8fps 260,885 帧×2 /
  30fps 140,149 帧×2 / 存储 673.9MB / 表征 id4-5 / map_v013-v014；真实 AppTest 默认
  `map_v012`（PE-AV 优先）、5 空间切换、视觉版本措辞与时间轴提示、
  视觉空间片段面板优雅降级；pilot 与全量证据见 `docs/reports/dense-visual-pilot.md`、
  `docs/reports/visual-dense-full-run.md`；
- **v0.5-alpha 独立验收（负结果）**：token 数据集 110 部/PE-AV 窗口 4130（截断后 3904 段）/141.1MB、manifest 记录
  6 部截断；v1/v2 checkpoint/metrics 与报告数字一致（v1 mask 0.8149/rank 1.03；
  v2 mask 0.8131/rank 4.66/order 0.10/reverse -0.0048/Top-5 0.0468）；
  研究产物未注册 representations 表、未建 map；v05_data 不 import torch、训练工具不写 DB；
  测试 15+19+15 条 CPU 离线用例；报告保留“未通过”负结论，不改成成功；
- **v0.4.5 独立验收**：discovery/工具不 import 模型/UMAP/UI、不写 DB/建 map；
  unexpected_relations JSON/CSV 真实运行（6 空间/110 部/skipped=0）；百分位公式
  p=1-(r-1)/(N-2) 逐项核对（JSON 6 位小数）；模式 A 950、B 864、common 99（>=3 空间 70）、
  visual_vs_peav 19；界面“意外关系探索”真实 AppTest 通过（由 t2 覆盖）；
- **v0.5-beta 独立验收**：MERT pilot 20/20 PASS（2582.753s、193179 帧/5177 bins、RTF 0.00947、
  显存 1320.4MB、产物 314.1MiB、漂移 0.01898s）；audio_version 隔离（model/revision/sr/ctx/
  hop/layer/pool/bin/norm）单测；v05_beta_v1 全量 110 部/4130 段、无截断；3 seeds test
  Top-1 0.296557/0.302779/0.305268（均值 0.301535±0.003663），
  高于 random(0.270455)/frozen(0.273745)/simple_sync(0.292410) 且消融双模态均掉点，
  但相对 simple_sync 仅 +0.91pp、CI 宽、best epoch 1–6，**weak positive / not passed**；
  研究产物不注册 DB/地图；
- **独立预言机**：E2E 使用 `tests/fixtures/video_factory.py` 现场合成的素材与独立断言，
  不复用实现方测试的断言逻辑；`qa.py` 的入口检查与 pytest 独立失败；
- **默认离线**：网络/模型测试必须打 marker，默认测试不触网、不加载权重；
- **人工验收**：真实视频跑 P0 后检查切片音画同步、界面交互与失败重试路径；
  证据与限制按 `docs/agents/QUALITY.md` 的证据式完成报告记录。
