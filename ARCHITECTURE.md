# 架构设计（ARCHITECTURE）

本文描述音MAD星云计划的系统结构、数据模型、模块边界与必须保持的不变量。
接口细节以 `docs/CONTRACTS.md` 为准；用户使用说明见 `README.md`；
关键取舍的理由见 `docs/adr/`。

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
| 数据层 | `nebula/ingest.py` / `nebula/download.py` | 本地扫描、选票/周报解析、Bilibili 元数据与下载 | db、config、errors |
| 数据层 | `nebula/mapplot.py` | 向量矩阵加载、UMAP 2D/3D、版本落盘、transform 增量 | db、config、umap-learn、joblib、numpy |
| 展示层 | `nebula/appdata.py` | 纯数据准备：DataFrame、染色 `ColorSpec`、密度、搜索、hover、链接 | pandas、numpy、db、config；**不导入 streamlit/plotly** |
| 展示层 | `app.py` | Streamlit 页面：侧栏控件、Plotly 图、详情展示 | streamlit、plotly、appdata |
| 入口层 | `ingest.py` / `pipeline.py` / `map.py` / `qa.py` | 参数解析、调用库代码、打印结果、返回退出码 | 对应库模块 |

依赖方向只有一条：入口层 → 展示层 / 编排层 → 领域层 → 基础层。
几个必须保持的边界（由 `tests/test_invariants.py` 静态检查）：

- `nebula.embed` 不导入 `pandas`、`nebula.db`、`nebula.appdata`、`nebula.mapplot`：
  向量提取必须能脱离数据库与界面独立运行、独立测试；
- `nebula.mapplot` 不导入 `streamlit` / `plotly`：降维是纯计算，可在无界面环境运行；
- `nebula.appdata` 不导入 `streamlit` / `plotly`：数据准备与渲染分离，便于单元测试与
  `streamlit.testing.v1.AppTest` 之外的直接断言；
- `app.py` 只依赖 `appdata` 与库层，不反向被库代码导入。

## 3. 数据模型

### 3.1 SQLite 表关系

~~~text
works 1────N work_events           （处理事件：created / status / ...）
works 1────N appearances N────1 lists      （一部作品可出现在多个榜单/年份）
works 1────N embeddings  N────1 embedding_versions
works 1────N clips                         （片段计划：idx/start_s/duration_s）
works 1────N map_coords N────1 map_runs
map_runs N────1 embedding_versions         （星云必须记录基于哪个向量版本）
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
| `clips` | `work_id`, `version`, `idx`, `start_s`, `duration_s` | 采样计划落库，用于复核音画同步 |
| `map_runs` | `version` PK, `embedding_version`, `params_json`, `coords_path`, `model_2d_path`, `model_3d_path`, `n_works`, `status` | 星云版本；`building/ready/failed` |
| `map_coords` | `map_version`, `work_id`, `x2`, `y2`, `x3`, `y3`, `z3` | 主键 `(map_version, work_id)`，一个版本一套坐标 |

连接设置：WAL、`foreign_keys=ON`、`BEGIN IMMEDIATE` 事务、`busy_timeout`；
所有写操作在事务内完成，读方法返回普通 `dict` 便于上层使用。

### 3.2 文件系统工件

```text
data/
├── catalog.sqlite                     # 索引与状态
├── embeddings/<embedding_version>/
│   └── <work_id>/
│       ├── work_joint.npy             # (dim,) float32，L2 范数 ≈ 1
│       ├── work_audio.npy / work_video.npy      # 单模态作品向量（若模型提供）
│       ├── clip_joint.npy             # (n_clips, dim) float32，每行 L2 范数 ≈ 1
│       └── clip_audio.npy / clip_video.npy
└── maps/<map_version>/
    ├── coords.parquet                 # work_id, x2, y2, x3, y3, z3
    ├── map_2d.joblib                  # 已训练的 2D UMAP 模型（供 transform）
    ├── map_3d.joblib
    └── meta.json                      # 参数、向量版本、作品数、生成时间
```

- 数据库里 `embeddings.path` 存的是相对 `data/` 的 POSIX 路径
  （`config.data_relpath`），保证换机器/换盘符后仍可解析；
- `work_joint` 等作品级向量的 `dim` 在同一个 `embedding_version` 内必须一致；
- 片段级向量形状 `(n_clips, dim)`，`n_clips` 与 `clips` 表的段数一致；
- 所有 npy / joblib / parquet 写入都走“临时文件 + 原子替换”，避免半文件。

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
| `nebula/sampling.py` | 时间轴采样规则 |
| `nebula/media.py` | ffprobe/ffmpeg 封装 |
| `nebula/embed.py` | PE-AV / 测试嵌入器、聚合、npy 读写 |
| `nebula/pipeline.py` | 单部处理与批处理编排 |
| `nebula/ingest.py` | 本地扫描、选票 CSV、周报导入 |
| `nebula/download.py` | Bilibili 元数据 API 与 yt-dlp 下载 |
| `nebula/mapplot.py` | UMAP 训练、版本落盘、transform |
| `nebula/appdata.py` | 可视化数据准备（纯函数） |
| `app.py` | Streamlit 应用 |
| `ingest.py/pipeline.py/map.py/qa.py` | 顶层 CLI |
| `tests/` | 单元、不变量、E2E 测试；`tests/fixtures/` 合成视频与临时配置 |
| `docs/` | 契约、ADR、质量规范、执行方案 |

## 9. 验证策略

- **唯一门禁**：`python qa.py`（ruff + pytest 快速离线 + 文档/入口/配置检查）；
  `--full` 额外运行 `model` / `network` 标记测试；
- **测试分层**：单元（config/db/sampling/media/embed/ingest/mapplot/appdata）、
  不变量（`tests/test_invariants.py`：模块边界、版本、状态机、去重、采样、gitignore、CLI）、
  端到端（`tests/test_e2e.py`：合成视频 → 流水线 → UMAP → appdata → Streamlit AppTest）；
- **独立预言机**：E2E 使用 `tests/fixtures/video_factory.py` 现场合成的素材与独立断言，
  不复用实现方测试的断言逻辑；`qa.py` 的入口检查与 pytest 独立失败；
- **默认离线**：网络/模型测试必须打 marker，默认测试不触网、不加载权重；
- **人工验收**：真实视频跑 P0 后检查切片音画同步、界面交互与失败重试路径；
  证据与限制按 `docs/agents/QUALITY.md` 的证据式完成报告记录。
