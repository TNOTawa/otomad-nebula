# 音MAD 星云计划（otomad-nebula）

把音MAD作品变成一片可以交互探索的“星云”。

每部作品用 PE-AV 多模态模型提取音画联合向量，再用 UMAP 降维成 2D/3D 坐标；
在本地 Streamlit 页面里可以缩放、旋转、悬浮查看、按外部元数据染色、搜索作品，
并一键跳转 Bilibili 原视频。**本项目的核心资产不是视频文件，而是可扩充的
音MAD 高维向量数据库**——所有星云、染色、年代对比都从同一套向量生成。

> 项目目标、分阶段计划与验收标准见 `音MAD星云计划 · 完整执行方案.md`；
> 模块接口契约见 `docs/CONTRACTS.md`；架构与不变量见 `ARCHITECTURE.md`。

## 功能一览

- **本地/在线作品统一编目**：扫描 `input/local/` 下的视频，或按 BV 号导入选票清单、周报；
- **音画同步切片**：按固定规则在时间轴上均匀截取 8 秒片段（最多 5 段，短视频自动减少）；
- **PE-AV 联合向量**：保存作品级与片段级的 joint / audio / video 向量；
- **断点续跑与失败隔离**：SQLite 状态机记录 `pending → downloading → encoding → completed`，
  失败作品记录原因、不阻塞其他作品，可单独重试；
- **2D/3D 星云**：UMAP cosine 分别训练二维与三维投影，每次生成保留版本号；
- **多维染色**：视频时长、发布日期、选票排名/得票、作者 UID、选票年份、榜单来源、空间密度；
- **增量更新**：新作品可用已保存的 UMAP 模型 `transform()` 加入固定坐标空间，
  旧点位置不变；积累较多后可重新生成新版本星云作对比；
- **质量门禁**：`python qa.py` 一个入口完成 ruff、pytest、文档/入口/配置检查。

**v0.2 新增（迭代 A/B，现有向量无需重算）**：

- **高维近邻探索（A1）**：选中作品查看 joint / audio / video 三模态 Top5 近邻（真实高维余弦），
  2D 高亮并连线，3D 提供连线开关与“搜索 + 下拉选择”等价入口；
- **高维近邻密度（A2）**：按第 5 近邻的**高维余弦距离**染色（越大越稀疏），不使用 UMAP 平面距离；
- **筛选星云（A2）**：按来源 / 榜单 / 年份构建独立星云版本，筛选条件落库；跨版本坐标不可比；
- **分阶段耗时（A4）**：批次与单部作品的 download/probe/extract/preprocess/forward/save 耗时入库，
  自动输出 `logs/perf_<run_id>.csv`，可用 `pipeline.py timings` 查看；
- **低风险提速（A5）**：切片去 AAC 音轨（零向量漂移）、复用作品级 ffprobe；preset / 分辨率
  对照实验有实测数据（见 ARCHITECTURE §6.2），会改变向量的方案留待 v0.3；
- **身份关联去重（B1）**：本地视频与 Bilibili 条目自动（标题+时长双校验）/ 手动（CSV）关联；
  duplicate 默认不进星云与近邻索引，已有本地向量可复制给 canonical，避免重复下载编码；
- **片段稳定性报告（A3）**：对片段级向量做留一重聚合，量化单个片段对作品 Top-k 近邻结论的影响，
  输出 JSON/CSV，只用于风险评估，不参与坐标与质量评价；
- **近邻解释（v0.2.x）**：近邻表带“全库分位 / 双向”解释列，并展示枢纽作品（hubness），
  避免把“小样本 Top5”误读为强相似；
- **片段级对应探索（v0.2.x）**：选两部作品查看片段对余弦热图与最高片段配对表，
  区分“整体平均向量相近”与“局部片段相近”；
- **音画错位实验（A6，v0.2.x）**：离线证据工具，对比原始 / 轻画质退化 / 音频偏移 / 换音频后的
  `Δ=1-cos`；只写日志，不改主向量与星云。
- **全片覆盖与多尺度音画特征表征（v0.4）**：序列层覆盖整条时间轴（无片段数上限，
  末尾锚定、权重归一），事件层记录音频 100Hz/视频 30fps 的高频变化统计；表征层用
  Full Mean / Temporal Stats 聚合，聚合器变化不重新编码 PE-AV。**不是逐帧语义理解系统**：
  PE-AV 仍是每 8s 16 帧（≈2fps）的稀疏语义，事件层是灰度/频谱统计；
- **三种表征空间切换（v0.4）**：Legacy（旧 5 片段平均）/ Full Mean（全片平均）/
  Temporal Stats（时间结构），各自独立地图，坐标不可跨空间比较；
- **邻域迁移分析（v0.4）**：`tools/compare_spaces.py` 比较两个表征空间的 Top-k 重合率、
  稳定对、被替换作品、余弦矩阵相关与 hubness 变化，输出 JSON/CSV；
- **密集视觉语义通道（v0.4.3）**：冻结 CLIP ViT-B/16（语义）与 DINOv2-small（结构）按 8fps
  提取帧级视觉特征（fp16）并按 8s 窗口池化，版本固定 revision/fps/size/preprocess；
  30fps 仅小样本；可构建 CLIP-only / DINOv2-only 对照星云。仍是特征表征，不是逐帧语义理解。
- **窗口覆盖权重与三层版本（v0.4.3.1）**：修复窗口权重未保存导致的等权聚合；拆成
  Frame Encoding / Window Plan / Representation 三层版本，窗口层独立重建（不重跑视觉模型），
  视觉 v2 表征按 coverage 权重加权；源视频持久缓存 `data/sources`，重建不再重复下载。
- **跨模型近邻共识（v0.4.4）**：比较 PE-AV / CLIP / DINOv2 高维空间的互为近邻与 >=2/>=3
  空间共识边，评估 rank-average / concat 轻量融合；界面可叠加互为近邻/共识连线。不做风格标注。

## 环境要求

| 依赖 | 要求 | 说明 |
| --- | --- | --- |
| 操作系统 | Windows 10/11（也支持 macOS/Linux） | 仓库内命令以 PowerShell 为例 |
| Python | 3.10（`pyproject.toml` 要求 >=3.10,<3.13） | 使用项目内 `.venv`，不要动全局 Python |
| FFmpeg / FFprobe | 需在 PATH 中 | 切片、探测、消融实验都依赖 |
| GPU | 可选 | 有 NVIDIA GPU 时 PE-AV 推理明显更快；纯 CPU 也可运行 |

基准依赖（numpy / pandas / plotly / streamlit / umap-learn / yt-dlp 等）在
`pyproject.toml` 的 `dependencies` 中；PE-AV 推理相关的
torch / transformers / av 在可选组 `ml` 中。**两类依赖分开安装**，
没有 GPU 或只做数据处理时也可以先只装基准依赖。

## 安装

### 1. 创建虚拟环境并安装依赖（推荐 uv）

```powershell
# 在项目根目录执行
uv venv .venv --python 3.10
.\.venv\Scripts\Activate.ps1

# 基准依赖（数据、星云、界面、下载）
uv pip install -e .

# 可选：PE-AV 推理依赖（torch/transformers/av）+ 开发测试依赖
uv pip install -e ".[ml,dev]"
```

如果不用 uv，用 `python -m venv .venv` 与
`pip install -e .[ml,dev]` 效果相同（Python 3.10）。

**可选：NVIDIA GPU 使用 CUDA 版 PyTorch。** 默认从 PyPI 安装的 torch 是 CPU 版；
本机（RTX 3060 / 驱动支持 CUDA 12.6）实际使用 cu126 轮子，命令如下（版本可按
PyTorch 官网选择，注意 Python 版本为 cp310）：

```powershell
uv pip install --python .\.venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu126 ^
  "torch==2.14.1+cu126" "torchvision==0.29.1+cu126"

# 确认 CUDA 可用
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`config.yaml` 的 `model.device: auto` 会自动选择 CUDA；有 GPU 时 5 个片段/部约
3.5 秒（RTX 3060，bf16 autocast），纯 CPU 则慢一两个数量级。GPU/CPU 向量存在约 1%
方向差异，详见 `ARCHITECTURE.md` §6.1。

### 2. 安装 FFmpeg

```powershell
# 任选其一
winget install Gyan.FFmpeg
# 或从 https://www.gyan.dev/ffmpeg/builds/ 下载后把 bin 目录加入 PATH

ffmpeg -version
ffprobe -version
```

### 3. 预下载 PE-AV 权重（首次运行前）

模型默认使用 `facebook/pe-av-small-16-frame`（16 帧输入），权重缓存在项目内
`.cache/hf`（对应 `config.yaml` 的 `model.cache_dir`）。手动预热：

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/pe-av-small-16-frame', revision='9f888eea95c83622212bb742e91bf01d3b46fe96', cache_dir='.cache/hf')"
```

需要联网；内网/离线机器请在可联网环境下载好 `.cache/hf` 后整体拷贝，
并把 `config.yaml` 的 `model.local_files_only` 设为 `true`。

### 4. 自检

```powershell
.\.venv\Scripts\python.exe qa.py
```

快速模式不需要模型权重与网络；全部检查通过即表示环境与代码就绪。

## 目录结构

```text
otomad-nebula/
├── input/
│   ├── local/                 # P0 本地测试视频（原有视频不会被删除）
│   └── lists/                 # 选票 CSV、周刊 BV 清单
├── data/                      # 永久数据，勿随意删除
│   ├── catalog.sqlite         # 作品/榜单/向量/状态/星云 索引
│   ├── embeddings/<sequence_version>/<work_id>/   # PE-AV 序列层（npy/事件/manifest）
│   ├── representations/<name>/<sequence_version>/ # 派生表征向量
│   ├── visual_dense/<frame_version>/<work_id>/    # v0.4.3 帧级密集视觉特征（fp16）
│   ├── visual_windows/<window_plan_version>/<frame_version>/<work_id>/  # v0.4.3.1 窗口层（权重）
│   ├── sources/<bvid>/        # v0.4.3.1 Bilibili 源视频持久缓存（不随 temp 清理）
│   └── maps/<map_vNNN>/       # UMAP 坐标、模型、meta.json
├── temp/                      # 流水线临时目录（成功自动清理，失败保留排查）
├── logs/                      # 运行日志
├── nebula/                    # 核心库：config/db/sampling/media/embed/pipeline/
│                              #   similarity(近邻/解释/片段) identity(身份关联) stability(稳定性)
│                              #   avsync(错位实验) events(事件扫描) temporal(表征聚合)
│                              #   dense_visual(密集视觉编码) visual_pipeline(视觉批处理)
│                              #   representations(表征注册/加载) mapplot/appdata
├── tools/                     # 证据/报告工具：clip_stability.py、desync_experiment.py、
│                              #   preflight_works.py、build_representation.py、compare_spaces.py、
│                              #   events_probe.py、visual_pipeline.py、dense_preprocess_compare.py、
│                              #   embedding_diagnostics.py、build_visual_windows.py、
│                              #   build_consensus_graph.py、bench_ffmpeg/preprocess/peav、verify_peav
├── tests/                     # pytest 测试（含 tests/fixtures 合成视频工厂）
├── docs/                      # 契约（CONTRACTS/CONTRACTS_V2）、ADR、质量规范、
│                              #   执行方案、reports（全片表征分析等）
├── ingest.py                  # 导入本地视频 / 选票 / 周报 / link-local 身份关联
├── pipeline.py                # 下载 → 切片 → 推理 → 保存 → 清理（含分阶段耗时）
├── map.py                     # 生成/更新 2D/3D 星云（build / list / transform / variants）
├── app.py                     # Streamlit 交互界面（近邻、密度、筛选）
├── qa.py                      # 唯一质量入口
└── config.yaml                # 所有运行参数的唯一来源
```

## 配置说明

所有相对路径都相对于 `config.yaml` 所在目录解析；改配置后无需改代码。
主要配置段：

| 配置段 | 关键字段 | 说明 |
| --- | --- | --- |
| `project` | `name` | 项目名，仅用于展示 |
| `paths` | `input_local` / `input_lists` / `data` / `temp` / `logs` / `sources` | 数据目录布局；`sources` 是 Bilibili 源视频持久缓存（默认 `data/sources`，不随 temp 清理） |
| `database` | `filename` | SQLite 文件名（位于 data 目录内，不能带子目录） |
| `media` | `clip_seconds` / `clips_per_work` / `min_duration_seconds` / `audio_sample_rate` / `clip_height` / `long_video_seconds` | 采样与切片参数；改动会改变向量版本；`clip_height: 336` 生成 `_h336` 实验版本 |
| `sampling` | `mode` / `overlap_seconds` | v0.4：`full_coverage`（默认，全片覆盖）或 `legacy`（旧 5 片段，用于旧数据/对照）；overlap 必须小于片段长度 |
| `events` | `enabled` / `audio_rate_hz` / `video_fps` / `version` | v0.4.1 高频事件扫描参数（音频 100Hz、视频 30fps） |
| `temporal` | `stages` | v0.4.2 Temporal Stats 的相对时间阶段数（默认 4） |
| `visual_dense` | `fps` / `size` / `preprocess` / `batch_size` / `encoders` / `subset_fps` / `subset_count` / `weight_scheme` | v0.4.3 冻结视觉编码器（固定 model+revision）、帧率/输入尺寸/预处理；30fps 只做子集；v0.4.3.1 的 `weight_scheme`：`coverage-v1`（默认）或 `equal`（复现旧行为） |
| `model` | `checkpoint` / `revision` / `version` / `frames` / `device` / `local_files_only` 等 | PE-AV 模型；`revision` 固定版本便于复现 |
| `umap` | `metric` / `n_neighbors` / `min_dist` / `random_state` / `n_jobs` / `dimensions` | 降维参数；固定 `random_state`、`n_jobs: 1` 保证可复现 |
| `download` | `yt_dlp_format` / `max_height` / `cookies_file` / `proxy` / `retries` | Bilibili 下载；登录可见视频需 cookies |
| `bilibili` | `api_timeout_seconds` / `user_agent` | 元数据接口请求参数 |
| `app` | `title` / `point_size` / `opacity` / `density_grid` | 界面默认值 |
| `logging` | `level` | 日志级别（`INFO` 建议保留） |

两个自动派生的版本号**不要手工修改**：

- `embedding_version`（当前激活的**序列版本**）：
  - legacy（默认）：由模型版本 + 帧数 + 片段长度/数量 + 音频采样率组成，
    例如 `peav-small-16f__f16_clip8s_x5_48k`；
  - full_coverage：例如 `peav-small-16f__f16_full8s_ov0_48k`（`_h336` 表示 336p 实验）。
  模型或采样参数变化会自动得到新版本，旧向量文件保留、不覆盖；
- `representation`：`legacy_5clip_mean` / `full_mean_v1` / `temporal_stats_v1`；
  聚合器变化只新增表征（`data/representations/...`），**不会重新编码 PE-AV**；
- `map_version`：例如 `map_v001`，每次生成/更新星云自动递增；地图绑定表征，
  不同表征/不同序列的地图坐标不可直接比较。

## 使用流程

### P0：本地视频测试

1. 把测试视频放进 `input/local/`（支持子目录；`.mp4/.mkv/.webm/.avi/.mov/.flv/.m4v`）。
2. 扫描并处理本地视频：

   ```powershell
   .\.venv\Scripts\python.exe pipeline.py process-local --verify-sync 3
   ```

   该命令会先登记 `input/local` 下的所有视频，再处理 pending 作品；
   `--verify-sync` 会对前 N 部作品抽查切片音画时长差（0 表示关闭）。
   目前无 GPU 时首次推理较慢，可用 `--limit 3` 先试跑。

   > 没有模型权重、只想先验证整条流程是否通时，可以用
   > `pipeline.py process-local --test-embedder --limit 3` 走通“切片 → 向量 → 星云 → 界面”。
   > 它生成的是测试用向量（版本 `test-hash-v1`，无语义意义），**不能用于真实观察**。
3. 生成星云坐标：

   ```powershell
   .\.venv\Scripts\python.exe map.py build
   ```
4. 启动界面：

   ```powershell
   .\.venv\Scripts\streamlit.exe run app.py
   ```

   浏览器打开终端提示的本地地址（默认 http://localhost:8501）。

也可以用等价的两步流程（导入与处理分离）：

```powershell
.\.venv\Scripts\python.exe ingest.py scan-local
.\.venv\Scripts\python.exe pipeline.py run-pending --source local
```

### P1：2025 年选票作品试运行

1. 准备选票 CSV，例如 `input/lists/votes_2025.csv`（示例见
   `input/lists/votes_2025.example.csv`）。必需列 `bvid`，
   可选 `part` / `rank` / `votes` / `author_uid` / `note`，
   也兼容 `BV号` / `分P` / `排名` / `得票数` / `UID` / `备注` 等中文列名。
2. 导入清单：

   ```powershell
   .\.venv\Scripts\python.exe ingest.py import-votes input/lists/votes_2025.csv --year 2025
   ```
3. 下载并处理未完成作品（逐部串行，下载完即删原视频，只保留向量与元数据）：

   ```powershell
   .\.venv\Scripts\python.exe pipeline.py run-pending --source bilibili
   ```
4. 生成星云并查看：`python map.py build` 后 `streamlit run app.py`。
5. 失败作品可查看并单独重试：

   ```powershell
   .\.venv\Scripts\python.exe pipeline.py status
   .\.venv\Scripts\python.exe pipeline.py retry <work_id>
   ```

### P2：扩展历年代表作品

1. 每年一份 CSV，例如 `votes_2024.csv`，用同样命令导入：
   `ingest.py import-votes input/lists/votes_2024.csv --year 2024`。
   同一作品跨年份/榜单重复出现时按 BV 号 + 分P 去重，只算一次向量，
   但保留全部榜单、年份、排名记录。
2. `pipeline.py run-pending` 处理新增作品。
3. `map.py build` 重新训练 UMAP，生成 `map_v002` 等新版本；
   旧星云版本与它们的坐标都会保留，界面侧栏可切换版本对比。
   **注意：UMAP 重新训练会改变旧点位置，不同版本的 2D 坐标不能直接作绝对位置比较。**
4. 升级模型或采样参数时：修改 `config.yaml` 的 `model.version` /
   `model.frames` / `media.*`，重新运行流水线。程序会为旧作品补算新向量
   （新版本目录），旧向量与旧星云不受影响。

### P3：接入社群周报，持续更新

1. 导入新一期周报（三种入口任选）：

   ```powershell
   # 直接给 BV 列表
   .\.venv\Scripts\python.exe ingest.py import-weekly --list-key weekly_2025_42 --bvids BV1xx411c7mD,BV1yy411c7mE
   # 或给周报页面 URL（自动抓取并提取 BV 号）
   .\.venv\Scripts\python.exe ingest.py import-weekly --list-key weekly_2025_42 --url "<周报链接>"
   ```

   导入只补充“出现记录”；已有作品不会重复计算向量。
2. `pipeline.py run-pending --source bilibili` 处理新增作品。
3. 优先用增量方式更新星云（旧点位置基本不变）：

   ```powershell
   .\.venv\Scripts\python.exe map.py transform
   ```

   新增作品较多时，也可以 `map.py build` 重新生成一个完整新版本并保留旧版本。
4. 建议每天检查一次周报是否更新，只在发现新内容时运行，不需要常驻高负载程序。

### v0.2：观察、筛选与运维（迭代 A/B）

v0.2 起所有观察功能都直接复用已保存向量，不需要重新编码。当前数据状态（2026-10-25，v0.4 全片升级完成）：

- 89 部 2025 选票全部导入；累计 **120 completed**（33 本地 + 87 Bilibili）+ 2 failed
  （8.71h 直播回放、API -404 已删除视频），pending=0；canonical = **110**（10 条身份关联去重）；
- **v0.4 三种空间**（均 110 部）：legacy `map_v008`；Full Mean `map_v011`；Temporal Stats `map_v012`；
  **当前界面默认 `map_v012`**（Temporal Stats）；侧栏可切换三空间；
- 全片序列：110/110 canonical 覆盖，4130 个 8s 窗口（legacy 仅 536 个），
  事件扫描 110/110（`events_v1`）；详细数字见 `docs/reports/full-coverage-analysis.md`；
- 旧版本 `map_v001`–`map_v007` 保留，用于历史/筛选对比（不同版本坐标不可直接比较）；
- **v0.4.3 密集视觉通道**（全量完成，2026-10-26）：CLIP ViT-B/16（512d）与 DINOv2-small（384d）
  @8fps/30fps、224、letterbox；4 个 visual_versions、**252/252 任务 completed**；
  8fps 每个编码器 110 部 / **260,885 帧 / 4,130 窗口**，30fps 子集每编码器 16 部 / 140,149 帧 / 593 窗口；
  存储 **673.9MB**；表征 `dense_visual_clip_v1`(id=4)/`dense_visual_dino_v1`(id=5) 各 110 向量；
  新地图 `map_v013`（CLIP-only）/ `map_v014`（DINOv2-only）；
  v0.4.3 为 5 个表征空间；v0.4.3.1 后 **7 个空间**（视觉 v1/v2 并存），
 默认仍优先 PE-AV 无筛选最新地图（`map_v012`，Temporal Stats）。

#### A1 高维近邻探索（界面）

启动界面后点击任意点（或通过侧栏搜索/下拉选择作品），详情面板显示 joint / audio / video
三个模态的 Top5 近邻与余弦值；缺失模态显示“无该模态向量”。2D 视图默认高亮 Top5 并画出
选中点到近邻的连线，3D 视图提供“显示近邻连线”开关（默认关闭）。
近邻与相似度全部来自高维余弦（nebula.similarity），与 UMAP 平面距离无关。

近邻表还带两个解释列（v0.2.x）：

- **全库分位（pair_percentile）**：该余弦值高于当前 canonical 集合中多少比例的作品对；
  分位接近 100 表示这种相似度在库里很罕见，而不是用绝对阈值判断“强相似”；
- **双向（is_mutual）**：目标作品是否也把当前作品列入同一 Top-k；单向命中可能是 hub 效应，
  双向证据更强；
- 分位/入度按当前向量版本的 canonical 集合计算，不受界面筛选影响，且跨集合不可横向比较；
- 近邻面板下方有“枢纽作品（被最多作品列为 Top5）”折叠区：展示 hubness 入度最高的作品，
  这是高维嵌入的统计现象，不代表作品质量或重要性。

真实数据样例（110 部 canonical，2026-10-25）：work 1 的 joint Top5 相似度
0.8625/0.8582/0.8562/0.8538/0.8481，对应全库分位 82.5/80.7/79.6/78.5/76.1；
hubness 最高为 work 104 被 40 部作品列为 Top5。可见“Top5 也只处于中高分位”，
需要结合分位与双向解释阅读。

#### P2 片段级相似探索（整体 vs 局部）

在近邻面板的“片段级对应（整体平均向量 vs 局部片段）”折叠区：

1. 勾选“显示片段级对应热图（需要加载片段向量）”；
2. 选择对比的近邻作品与片段模态（联合片段 / 音频片段）；
3. 查看片段对余弦热图、最高片段配对表（含两侧片段序号与起始秒）以及最高/平均余弦统计。

用途：作品级平均向量接近，不代表局部片段也接近；热图能定位具体相似/不相似的片段。
片段级接口是纯数据层（`appdata.clip_heatmap_data / clip_pair_rows`），不依赖 Plotly。

真实数据样例（2026-10-25）：work 1 × work 93 的 5×5 联合片段热图最高 cos=0.8095
（A 片段 26.4s ↔ B 片段 97.1s），25 个片段对全部 >0.5；这类信息无法从作品级 Top5 直接看出。

#### A2 高维近邻密度与筛选星云

- 染色维度选择“高维近邻密度”：值为 1 - 第 5 近邻余弦相似度，越大表示局部越稀疏；
- 侧栏可按来源（全部 / 本地 / Bilibili）、榜单、选票年份筛选当前显示的作品；
- 需要让“筛选后的子集重新布局”时，构建独立星云版本：

```powershell
# 只含 Bilibili 来源、2025 选票年份的作品
python map.py build --dataset bilibili --vote-year 2025

# 用不同 n_neighbors 生成对照版本（每个都是独立 map_vNNN）
python map.py variants --n-neighbors 8 15 30

python map.py list
```

- 筛选条件写入 map_runs.filter_json，并记录在该版本的 meta.json 中；
- **筛选星云与全量星云、以及不同筛选版本之间的坐标不可直接比较**（界面会显示版本与筛选条件）；
- 默认排除已做身份关联的 duplicate；需要对照“包含重复实体”的版本时加 `--include-duplicates`。

#### A3 片段稳定性报告

```powershell
python tools/clip_stability.py --k 5
python tools/clip_stability.py --version peav-small-16f__f16_clip8s_x5_48k --out-dir logs
```

- 默认读取配置派生的当前 `embedding_version`，输出 `logs/clip_stability.json`（完整明细）与
  `logs/clip_stability.csv`（每部作品一行汇总）；
- 指标：留一重聚合后的 Top-k 重叠率、最大名次变化、平均余弦漂移；`mean_overlap` 越低，
  说明该作品结论越依赖某一个 8 秒片段；
- 只读已保存的 clip_joint / work_joint 向量，不加载模型、不重新编码、不参与 UMAP 坐标，
  也不作为作品质量指标；输出目录可用 `--out-dir` 指定；
- 可用 `--work-ids 1 2 3 ...` 只评估部分作品（例如只复盘 P0 的 33 部），
  近邻空间仍是全部 canonical 作品，不会因筛选而改变比较基准。
- 当前真实数据报告（2026-10-25）：`logs/clip_stability_p0.*`（P0 33 部，mean_overlap=0.8978，
  不稳定 1/33）、`logs/clip_stability_v02.*`（v0.2 canonical 56 部，mean_overlap=0.875，
  不稳定 6/56）、`logs/clip_stability_v03.*`（扩量后 canonical 110 部，mean_overlap=0.8488，
  不稳定 21/110）；三份报告均只读、离线、不加载模型。

#### A4 性能与分阶段计时

- 每次批处理自动登记 `pipeline_runs`（批次级）与 `work_timings`（单部作品的分阶段耗时）；
- 阶段字段：`download_s / probe_s / extract_s / preprocess_s / forward_s / save_s / total_s`；
- 运行后查看 `logs/perf_<run_id>.csv`；失败作品也会记录已经发生的阶段耗时；
- 查看最近批次的单部分阶段耗时：

```powershell
python pipeline.py timings                 # 默认最近一批
python pipeline.py timings --run-id 3      # 指定批次
python pipeline.py timings --limit-runs 5  # 列出最近 5 个批次
```

- 计时只用于定位瓶颈，不进入向量与星云，也不改变 `embedding_version`。

#### A5 低风险提速（行为不变）

- 切片输出 mp4 只保留视频轨（不再编码 AAC）：PE-AV 从 mp4 取画面、从 wav 取音频，
  少一次音频编码并减小临时文件；
- 同一部作品只执行一次 ffprobe，结果复用给全部片段（`extract_clip(info=...)`）；
- 以上优化不改变采样方案、向量内容与 `embedding_version`；对照基准可用
  `tools/bench_peav.py`（模型加载/批大小/设备/精度）与 `tools/verify_peav.py`；
- 实测（详见 ARCHITECTURE §6.2）：去 AAC 音轨向量零漂移（sha1 全等，max abs diff=0）；
  preset veryfast→ultrafast 在 5 片段上 11.07s→5.88s（1.88×）但体积约 2.35×；
  分辨率缩到高 336 可再快约 2.4× 但会改变向量（joint cos 最低 0.99812），需新向量版本，
  留待 v0.3 评估。v0.2 只采纳零漂移优化。

#### B1 身份关联与星云去重

```powershell
# 先 dry-run 看候选（标题规范化后完全相等 + 时长差 <= 3s）
python ingest.py link-local --list-key votes_2025 --dry-run

# 建立关联，并把本地已有向量复制登记给 canonical（避免重复下载/编码）
python ingest.py link-local --list-key votes_2025

# 读取 input/lists/local_bv_links.csv 手动覆盖清单
python ingest.py link-local --list-key votes_2025 --manual
```

- 关联后 duplicate 默认不进入星云、近邻索引与界面，但其作品记录、本地视频与向量保留；
- 若本地作品已有当前版本向量，会复制登记给 canonical 并标记 completed，流水线自动跳过下载/编码；
  `--no-copy` 只建立关联、不复制向量；
- 关联可撤销（数据库 `unlink_work`）；拿不准的候选不会自动合并。

#### A6 音画错位实验（离线证据）

```powershell
python tools/desync_experiment.py --limit 10          # 10 部本地已完成作品
python tools/desync_experiment.py --work-ids 6 12 20  # 指定作品（需为本地已完成）
```

- 每个片段生成 5 类输入：A 原始、A-quality 轻画质退化基线、B 音频 +0.5s、C 音频 +1.0s、
  D 同作品换其他片段音频；用真实 PE-AV 编码后比较 `Δ=1-cos`（joint/audio/video）。
- 只写 `logs/av_desync_experiment.json/.csv` 与中文摘要，不修改主向量库、不生成新
  `embedding_version`、不参与星云；方法与解读边界见 ADR-0008。
- 需要 GPU/模型权重（真实运行标记为 `model`）；默认测试离线不跑。

**实测（2026-10-25，10 部真实作品 / 48 个 8s 片段，CUDA bf16）**：
quality（CRF28 轻画质退化基线）joint Δ mean=0.000964；音频 +0.5s=0.004222（4.38×）；
+1.0s=0.002364（2.45×，周期性音乐整拍偏移导致反而更小）；换音频=0.087575（90.85×）；
逐片段高于 quality 的比例：+0.5s 47/48、+1.0s 36/48、换音频 48/48；纯音频变体的 video 分支 Δ=0。
注意：绝对 Δ 很小且仅 10 部单次实验，单点 Δ 不能证明同步敏感度（详见 ADR-0008）。

#### 数据扩量：2025 年 89 部选票（已完成）

- 选票 CSV 已全部导入（89 条 appearances / 89 个 Bilibili 作品登记），身份关联 10 条，
  duplicate 默认不进入星云；
- 预检 `python tools/preflight_works.py` 输出 `logs/preflight_89.json`：
  89 部中 80 normal / 7 multi（多分P）/ 1 long（8.71h 直播回放，failed）/
  1 unavailable（API -404 已删除，failed），并给出建议 batch/skip/done 清单；
- 扩量批处理（run_id=5）：54 部 **54 completed / 0 failed**，总耗时 730.51s（模型加载 17.05s）；
  累计 120 completed + 2 failed、pending=0；
- 扩量后重建星云：`map_v007`（2025 全量 87 部，filter_json list_key=votes_2025）与
  `map_v008`（legacy 总星云 110 部）；v0.4 全片升级后又新增 `map_v011`（Full Mean）与
  `map_v012`（Temporal Stats），均为 110 部。不同版本/不同表征的坐标不可直接比较；
- 查看最新状态与重试：

```powershell
python pipeline.py status
python pipeline.py retry <work_id>
python map.py list
```

> 实时数据状态以 `pipeline.py status` 输出为准；本仓库不提交实时运行数据，
> 地图版本由 `map.py list` 查看。

### v0.4：全片覆盖与多尺度音画特征表征（多空间观察）

v0.4 的准确名称是**「全片覆盖与多尺度音画特征表征系统」**，不是「逐帧音MAD语义理解系统」。
它把“均匀抽 ≤5×8s 片段再平均”升级为**覆盖整条时间轴**的分层表示，
同时保留旧数据与旧空间（不迁移、不覆盖）：

- **序列层**：`data/embeddings/<sequence_version>/<work_id>/` 保存 PE-AV 片段序列、
  `clip_times.npy`（窗口起止/有效时长/权重）、事件文件与 manifest；全片模式没有片段数上限，
  最后一个窗口结束于视频结尾，任意时间点至少被一个窗口覆盖；
- **表征层**：`data/representations/<name>/<sequence_version>/<work_id>/` 保存聚合结果
  （`full_mean_v1` / `temporal_stats_v1`）；改变聚合器只新增表征，绝不重新编码 PE-AV；
- **三种空间**：Legacy（旧 5 片段平均）、Full Mean（全片平均）、Temporal Stats（时间结构）；
  各自使用独立地图，坐标不可跨空间比较。

#### 能力边界（v0.4）

- ✅ **完整时间覆盖**：任意时刻至少被 1 个 8s 窗口覆盖、末尾锚定、窗口权重归一；
  这是“采样覆盖”的完整，不是“逐帧内容”的完整；
- ✅ **高频音画变化记录**：音频 100Hz 事件、视频 30fps（缩放到 96×54 灰度）的帧差/运动/
  亮度/切换统计，记录节奏、运动与突变，**不是视觉语义**；
- ✅ **时间结构统计**：Temporal Stats 是**人工设计的分组与权重**的固定统计聚合器
  （语义全局/阶段/漂移/动态/自相似/事件），不是训练出来的模型；
- ❌ **逐帧视觉语义编码**：PE-AV 每 8s 只有 16 帧（≈2fps）的稀疏联合语义，
  不能理解为逐帧画面/动作/歌词/风格理解；
- ❌ **可学习音画时序关系**：本轮只有固定统计聚合与手工权重，没有训练目标或可学习融合；
- ❌ **自监督全曲表示**：v0.5 仅预留数据接口，本轮未训练、不做质量结论。

一句话：**完整覆盖 ≠ 逐帧语义理解**；v0.4 完成的是“覆盖 + 多尺度特征记录 + 固定统计表征”，
不宣称“看懂”每一个画面或每一个音MAD梗。

真实数据结果（2026-10-25，110/110 canonical；完整报告见
`docs/reports/full-coverage-analysis.md`）：

| 指标 | legacy | Full Mean / Temporal Stats |
| --- | --- | --- |
| 片段/窗口总数 | 536（≤5/部） | **4130**（min 3 / mean 37.55 / max 214） |
| 覆盖率 / 权重和 | — | 110/110；Σweight∈[0.99999992, 1.00000003] |
| 事件扫描 | 无 | 110/110（`events_v1`） |
| 每部耗时（均值） | 16.57s | 70.35s（extract 24.96 / events 7.53 / forward 22.45） |
| 地图 | map_v008 | map_v011 / map_v012（当前默认） |

- 邻域迁移：legacy×Full Mean 平均 Top5 Jaccard=0.4068（完全稳定 8/110，余弦 Pearson=0.9569）；
  Full Mean×Temporal Jaccard=0.2049（稳定 0/110，Pearson=0.8294）；Legacy×Temporal Jaccard=0.1734；
  即“完整覆盖”和“时间结构”都显著改变邻域，换空间后近邻结论不能直接沿用；
- Temporal Stats 顺序敏感性：20 部反转顺序后 Full Mean cos_min≈1.0，Temporal cos_min=0.671；
- 336p 实验（10 部）：work_joint cos min=0.9972 / mean=0.9986，video min=0.9772；
  extract 2.26×、total 1.60× 加速，但视频分支向量会变，**仅作候选，不转正**；
- 存储：全片序列约 187MB（事件约 131MB）+ representations 9.7MB；legacy 向量 8.4MB。

1. 切换采样模式并编码全片序列（当前 `config.yaml` 默认已是 `full_coverage`，旧空间用
   `--sampling legacy` 或直接加载旧地图）：

   ```powershell
   # 覆盖 config.sampling.mode，只处理本地视频前 3 部
   python pipeline.py process-local --sampling full_coverage --limit 3

   # 336p 对照实验会生成带 _h336 的新序列版本（不覆盖旧版本）
   python pipeline.py process-local --sampling full_coverage --clip-height 336 --limit 3

   # 超长视频（默认超过 media.long_video_seconds 时拒绝）需显式允许
   python pipeline.py run-pending --sampling full_coverage --allow-long
   ```

   - 全片编码按批写入 `.partial/batch_*.npz`；中断后重跑同一命令只补缺失批次；
   - 事件扫描（v0.4.1）由 `events.enabled` 控制，失败会记失败原因而不是静默写零向量。

2. 从已保存序列构建表征（不下载、不解码、不跑 PE-AV）：

   ```powershell
   python tools/build_representation.py full_mean --sequence auto
   python tools/build_representation.py temporal_stats --stages 4 --sequence auto
   python tools/build_representation.py list
   ```

3. 按表征生成独立星云（不覆盖旧地图）：

   ```powershell
   python map.py build --representation full_mean_v1
   python map.py build --representation temporal_stats_v1
   python map.py list
   ```

4. 界面侧栏切换“表征空间”（Legacy / Full Mean / Temporal Stats），选中该空间最新 ready 且无筛选的地图；
   详情面板增加“完整时间轴/片段窗口”信息（事件曲线与窗口边界，数据缺失时显示提示）。

5. 邻域迁移分析（回答“换空间后哪些作品/作品对变了”）：

   ```powershell
   python tools/compare_spaces.py --a legacy_5clip_mean --b full_mean_v1 --k 5
   python tools/compare_spaces.py --a full_mean_v1 --b temporal_stats_v1 --k 5
   ```

   输出 `logs/space_compare_*.json/.csv`：每作品 Top-k Jaccard、完全稳定作品比例、
   稳定对比例、被替换案例（新旧邻居）、作品对余弦 Pearson/Spearman、hubness 前后变化。
   分析报告见 `docs/reports/full-coverage-analysis.md`。

**兼容与边界**：旧 `embedding_version` 目录与旧地图只读保留；legacy 与 full 序列版本号不同，
地图/近邻/密度都按各自空间计算；全片模式的片段更多，片段级对应面板仍可用；
PE-AV 仍是 16 帧稀疏采样，事件层补充节奏/运动信息，但**不宣称逐帧语义理解**。

### v0.4.3：密集视觉语义通道（Dense Visual）

v0.4.3 在 PE-AV 联合表征之外新增一个**独立、冻结编码器**的视觉通道，回答“更高帧率的视觉语义
是否携带 PE-AV 之外的信息”。它仍然是**特征表征**，不是逐帧语义理解系统，也没有训练任何
时序模型：

- **编码器（固定 revision，禁止浮动版本）**：
  - `openai/clip-vit-base-patch16` @ `57c216476eefef5ab752ec549e440a49ae4ae5f3`（512 维，图文对齐语义）；
  - `facebook/dinov2-small` @ `ed25f3a31f01632728cabb09d1542f84ab7b0056`（384 维，自监督视觉结构）；
- **粒度**：默认 8fps 解码整部视频（单次 ffmpeg 流式，不整片进内存），帧级 L2 归一化向量
  **fp16** 保存；再按 v0.4 的 8s 全片窗口（与 PE-AV 窗口一一对应）做重叠时长加权池化，
  得到窗口级 `window` 向量（fp32）与 `window_std`；30fps 只对配置子集
  （`subset_fps=30`、`subset_count=16`）运行，供帧级轨迹研究，不默认全量；
- **版本命名**：逻辑身份 `model_id@revision__fps{fps}_s{size}_{preprocess}`，例如
  `openai/clip-vit-base-patch16@57c216...__fps8_s224_letterbox`；
  **DB / data/visual_dense 目录 / 地图 embedding_version 使用 `visual_version_slug` 后的
  文件系统安全形式**（"/"、"@" 转 "-"，如
  `openai-clip-vit-base-patch16-<rev>__fps8_s224_letterbox`），
  `visual_versions.model_id/revision` 保留原值，slug 化不丢身份；
  encoder/revision/fps/size/preprocess 任一变化都必须新建版本，绝不覆盖；
  **与 PE-AV `sequence_version` 完全独立**，禁止混入同一近邻/UMAP 矩阵；
- **预处理**：默认 `letterbox`（保持宽高比 + 黑边）；`stretch` 仅用于
  宽高比对照实验（`tools/dense_preprocess_compare.py`）；
- **数据布局**：
  `data/visual_dense/<visual_version>/<work_id>/dense_visual.npz`（帧级 fp16 + 时间戳元数据）、
  `windows.npy`（窗口级）、`window_std.npy`（窗口内变化）、`manifest.json`
  （version/encoder/revision/fps/size/preprocess/duration/n_frames/dim/n_windows/window_times/created_at）；
- **DB（schema v4）**：`visual_versions`（版本与 model/revision/preprocessing/fps/size/dim）
  和 `visual_sequences`（作品×版本状态、frames/windows 相对路径、n_frames/n_windows/dim、
  decode/encode/model_load/total 计时）；状态机 `pending → in_progress → completed / failed`，
  支持断点续跑；只处理 canonical 且 completed 的作品，Bilibili 作品每部只下载一次，
  全部任务成功后清理 temp，失败保留 temp 续跑；
- **表征与星云**：窗口级向量可登记为 `kind=visual` 表征
  （`dense_visual_clip_v1` / `dense_visual_dino_v1`），各自建地图
  （CLIP-only / DINOv2-only 对照空间）；视觉地图与 PE-AV 地图坐标不可比较。

配置（`config.yaml` 的 `visual_dense` 段，关键字段）：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `fps` | 8.0 | 主帧率；30fps 只做子集 |
| `size` | 224 | 方形输入边长 |
| `preprocess` | letterbox | stretch / letterbox（宽高比对照） |
| `batch_size` | 32 | 每批编码帧数 |
| `device` / `dtype` | auto | 推理设备/精度（bf16 autocast） |
| `subset_fps` / `subset_count` | 30 / 16 | 30fps 小样本对照 |
| `encoders` | CLIP + DINOv2 | 固定 model + revision（禁止浮动版本） |

命令：

```powershell
# 全量/增量运行（按 config.visual_dense 的编码器、8fps、窗口对齐；支持断点续跑）
python tools/visual_pipeline.py run
python tools/visual_pipeline.py status
python tools/visual_pipeline.py list
python tools/visual_pipeline.py clean-temp

# 宽高比预处理对照（少量作品，stretch vs letterbox；JSON+摘要）
python tools/dense_preprocess_compare.py --help

# 低维投影诊断：PCA / UMAP(15/30/50)、Top-10 保持率、距离相关（输出 JSON/CSV，可选 HTML）
python tools/embedding_diagnostics.py --representation full_mean_v1 temporal_stats_v1
```

全量结果（2026-10-26，真实数据 110/110 canonical）：

| 指标 | 实际值 |
| --- | --- |
| visual_versions | 4 个（CLIP/DINOv2 × 8fps/30fps，固定 revision、letterbox/224） |
| visual 任务 | **252/252 completed，0 failed**（8fps 220 + 30fps 子集 32） |
| 8fps 帧/窗口 | 每个编码器 110 部、**260,885 帧 / 4,130 窗口** |
| 30fps 子集 | 每编码器 16 部、140,149 帧 / 593 窗口 |
| 存储 | **673.9MB**（8fps 447.9 + 30fps 226.0） |
| 密集视觉表征 | `dense_visual_clip_v1` id=4（512d）/ `dense_visual_dino_v1` id=5（384d），各 110 向量 |
| 对照星云 | `map_v013`（CLIP-only，110 部）/ `map_v014`（DINOv2-only，110 部） |
| 宽高比对照（3 部） | stretch vs letterbox 窗口余弦：CLIP mean=0.9425/min=0.8760；DINO mean=0.8725/min=0.7897 → **采用 letterbox** |
| 视觉 vs PE-AV 窗口 RSA（110 部） | CLIP-video mean 0.6949；DINO-video 0.7246；CLIP-joint 0.6226；DINO-joint 0.6482 |
| 30fps 子集窗口饱和 | 8fps↔30fps 窗口 cos：CLIP mean 0.9998/min 0.9997；DINO 0.9994/0.9987 |

**能力边界**：v0.4.3 提供的是**冻结编码器的帧级语义特征 + 固定窗口池化**；
它比 PE-AV 的 ≈2fps 稀疏语义时间密度更高（8fps，可选 30fps 子集），但仍是“特征”，
不是“逐帧语义理解”，也没有可学习音画时序关系；完整覆盖/高频特征/密集视觉特征都不等于语义理解。

> 全量运行与对照分析（数字已回填、t4 复核）：`docs/reports/visual-dense-full-run.md`
> （宽高比对照 / 全量统计 / CLIP-only 与 DINOv2-only 星云 / PCA-UMAP 诊断）；
> pilot 结论见 `docs/reports/dense-visual-pilot.md`。

### v0.4.3.1：窗口覆盖权重修复与三层版本分离

用户复核 v0.4.3 发现两处数据一致性问题，本轮在不重新编码视觉帧的前提下修复：

1. **覆盖权重未保存**：v0.4.3 的窗口级视觉向量没有把 `plan_full_coverage` 的覆盖权重写入
   manifest，构建 `dense_visual_clip_v1`/`dense_visual_dino_v1` 时退化为**等权平均**；
   修复后权重写入窗口层并用于作品向量加权聚合；
2. **视觉版本号缺少窗口身份**：调整窗口时长/overlap/权重方案会复用旧版本目录；修复后拆成三层版本。

| 层 | 版本身份 | 示例 | 变化时重建 |
| --- | --- | --- | --- |
| Frame Encoding Version | model+revision+fps+size+preprocess（沿用 `visual_version`） | `openai-clip-vit-base-patch16-<rev>__fps8_s224_letterbox` | 解码+编码帧（GPU） |
| Window Plan Version | clip_seconds+overlap_seconds+weight_scheme（`config.visual_window_plan_version`） | `win8s_ov0__coverage-v1` | 只读帧级特征重算窗口/权重 |
| Representation Version | 聚合器 + 引用的窗口计划（`dense_visual_clip_v2` / `dense_visual_dino_v2`） | 同一窗口计划下的加权均值 | 只读窗口层重算作品向量 |

- **窗口计划**写入 DB `visual_window_plans` / `visual_window_sequences`（schema v5）；
  同一帧编码版本可配多个窗口计划，同一窗口计划可跨多个帧编码版本（**正交**）；
- **数据布局**：
  - 帧级：`data/visual_dense/<frame_version>/<work_id>/dense_visual.npz`；
  - 窗口层：`data/visual_windows/<window_plan_version>/<frame_version>/<work_id>/`
    下 `windows.npy` / `window_std.npy` / `window_weights.npy` / `manifest.json`；
  - 源缓存：`data/sources/<bvid>/<bvid>_p<part>.*`（持久保存，不随 temp 清理）；
- **配置**：`paths.sources`（默认 `data/sources`）、`visual_dense.weight_scheme`
  （`coverage-v1` 默认 / `equal` 复现旧行为）；窗口计划版本由
  `media.clip_seconds` + `sampling.overlap_seconds` + `weight_scheme` 派生；
- **兼容**：v1 表征与 `map_v013`/`map_v014` 保留只读；v2 是同一帧级特征上的
  权重修复，另建表征/地图，坐标与 v1 不可直接比较。

命令：

```powershell
# 1) 从已保存帧级特征重建窗口层（不下载、不解码、不跑模型）
python tools/build_visual_windows.py --all --frame-version <frame_version> \\
    --window-seconds 8 --overlap-seconds 0 --weight-scheme coverage-v1

# 2) 构建 v2 表征（覆盖权重，--v2 读取窗口计划层的 window_weights）
python tools/build_dense_visual_representation.py --v2

# 3) 建 v2 对照地图 / 前后对比
python map.py build --representation dense_visual_clip_v2
python map.py build --representation dense_visual_dino_v2
python tools/compare_spaces.py --a dense_visual_clip_v1 --b dense_visual_clip_v2
```

实际结果（2026-10-26，真实 110 部/4 个帧编码版本）：

| 指标 | v1（等权，历史） | v2（coverage 权重） |
| --- | --- | --- |
| 窗口计划 | 未记录 | **win8s_ov0__coverage-v1**（8.0s / ov0 / coverage-v1） |
| 窗口任务 / 窗口数 | — | **252 / 252 completed，0 failed；9446 窗口**（8fps 4130×2、30fps 593×2） |
| 权重和范围 | 等权 | **0.999999847..1.000000209**；coverage-vs-equal max 0.022221 / mean 0.001713 |
| 密集视觉表征 | id=4/5（512/384） | **id=6/7（512/384）**，各 110 joint |
| 对照地图 | map_v013/map_v014 | **map_v015/map_v016**（110 部，source_kind=visual） |
| 逐作品 v1→v2 余弦 | — | CLIP mean 0.999999 / min 0.999972；DINO mean 0.999996 / min 0.999863 |
| Top-5 稳定性 | — | CLIP Jaccard 0.9939 / 稳定 108/110；DINO 0.9970 / 稳定 109/110 |
| 帧编码 | — | 未重新编码（窗口层从已保存帧重建） |

**影响量化**：本数据为 8s/不重叠窗口，权重除末尾不足 8s 的窗口外近似均匀，因此 v1→v2
数值变化很小（Top-5 Jaccard 0.99+）；但这是**正确性修复**（权重真实落盘并参与聚合），
在 overlap>0 / 更短窗口/非均匀覆盖计划下差异会放大，不能因当前影响小而不修。

> 修复报告（问题/证据/重建数字/前后近邻变化）：`docs/reports/v0431-window-weights-and-versioning.md`
> （已回填并由 t4 复核）。

### v0.4.4：跨模型近邻共识与轻量融合

在现有高维表征上评估“多个模型共同认可的相似作品是否比单一模型近邻更有解释力”，
不训练新模型、不做人工风格标注、不注册正式主模型：

- **空间**：默认 `full_mean_v1`（PE-AV）+ `dense_visual_clip_v2` +
  `dense_visual_dino_v2`（v0.4.3.1 覆盖权重版）；`--representations` 可配；
- **共识**：每空间高维余弦 Top-k（默认 k=5）有向边与互为近邻（mutual）边；统计跨空间保留率、
  **>=2 空间共识边**、**>=3 空间全共识边**、覆盖作品比例与共识强度；输出
  `logs/neighbor_consensus.json` / `.csv`；
- **轻量融合基线**（不写 DB）：rank-average 融合与 concat 向量融合（等权或 $1/\\sqrt{dim}$），
  与单空间/共识边比较 Jaccard 与重合率；
- **界面**：新增“显示高维互为近邻连线”（默认关，只画 mutual，最大边数截断提示）与
  “仅显示跨模型共识边”（读取共识 JSON，缺失时回退并提示）；详情面板显示共识近邻摘要；
  PCA 3D 仅作辅助观察，不改变默认 UMAP 坐标语义；
- 边界：共识是**观察性邻域一致性证据**，不是风格/流派分类，也不宣称逐帧语义理解；
  正式融合表征必须新建版本、重新验证，不覆盖已有 v0.4.x 表征/地图。

```powershell
python tools/build_consensus_graph.py \\
    --representations full_mean_v1 dense_visual_clip_v2 dense_visual_dino_v2 \\
    --k 5 --min-spaces 2 --fusion both
```

实际结果（2026-10-26，真实 110 部）：

- 空间 mutual 边：full_mean_v1 **61**（比例 0.2218）、dense_visual_clip_v2 **75**（0.2727）、
  dense_visual_dino_v2 **73**（0.2655）；mean Top-k 余弦 0.9146 / 0.9365 / 0.8146；
- 跨空间 mutual Jaccard：FullMean×CLIP **0.3204**、FullMean×DINO **0.2407**、CLIP×DINO **0.3455**；
- 共识边 **53** 条（**全共识 22** 条），覆盖 **44/110（40.00%）** 作品，平均强度 **0.8050**；
- 融合基线：rank-average Jaccard(vs spaces)=**0.5331** / vs 共识=**0.4303**；
  concat(equal) 0.5358/0.4106；concat(inv_sqrt_dim) 0.5385/0.4121；
- 界面：默认仍 `map_v012`；7 空间（v1/v2 视觉并存）；互邻连线 k=5、
  共识阈值 >=2/>=3 均可在侧栏切换（真实 AppTest 通过）；
- 结论：多模型共识提供单模型之外的稳定候选，但轻量融合不能替代共识；
  共识是观察性证据，不做风格分类。

> 共识与融合报告（方法/结果/案例）：`docs/reports/v044-consensus-fusion.md`（已回填并由 t4 复核）。

### 界面使用

- v0.4.4 新增“显示高维互为近邻连线”（默认关；只画 mutual，超过边数上限截断提示）与
  “仅显示跨模型共识边”（读取 `logs/neighbor_consensus.json`，缺失时提示并回退）；
  详情面板可显示共识近邻摘要；PCA 3D 仅作辅助观察；
- 侧栏：**表征空间切换（Legacy / Full Mean / Temporal Stats / Dense Visual·CLIP v1 /
  Dense Visual·DINOv2 v1 / Dense Visual·CLIP v2 / Dense Visual·DINOv2 v2，共 7 个空间）**、
  星云版本、2D/3D 切换、染色维度（含“高维近邻密度”）、
  按文件名/标题/BV 搜索、点大小/透明度、数据集筛选（来源 / 榜单 / 年份）；
  默认优先打开 PE-AV 无筛选最新地图（当前 `map_v012`，Temporal Stats；视觉空间需手动切换）；
- 主图：Plotly 散点图，支持缩放、旋转、悬浮；点击点（或搜索 + 下拉选择）显示详情；
- 详情：标题、BV/URL（可点击跳转 Bilibili）、时长、发布日期、作者、选票排名/得分、本地文件，
  以及三个模态的 Top5 近邻（余弦值 + 全库分位 + 双向）；2D 高亮近邻并连线，3D 可开关连线；
- 近邻面板还含“枢纽作品（被最多作品列为 Top5）”与“片段级对应（整体 vs 局部片段）”两个折叠区；
  片段热图默认关闭，勾选后按需加载片段向量；全片模式下片段更多（按窗口一一对应）；
- 详情面板还有“完整时间轴（音画事件 + 片段窗口）”折叠区：音频/视频/音画关系事件包络、
  窗口起点竖线、覆盖率与事件版本；缺事件数据时显示提示；
- 3D 点击选择在部分浏览器/Plotly 版本下不可用，可用“搜索 + 下拉选择作品”的等价入口；
- 无数据（尚未生成星云）时页面显示中文提示，而不是报错堆栈；筛选星云会标注
  “坐标仅在本版本内可比”。

## 数据目录与安全清理

**永久保存（不要随手删）**：

- `data/catalog.sqlite`：作品、榜单、向量登记、处理状态、星云版本；
- `data/embeddings/<sequence_version>/`：序列层片段/事件/时间轴（核心资产，重新编码成本高）；
- `data/representations/<name>/<sequence_version>/`：派生表征向量（可从序列重建）；
- `data/visual_dense/<frame_version>/<work_id>/`：v0.4.3 帧级密集视觉特征（重编码可重建）；
- `data/visual_windows/<window_plan_version>/<frame_version>/<work_id>/`：v0.4.3.1 窗口层（含权重，只读帧级特征可重建）；
- `data/sources/`：Bilibili 源视频持久缓存（避免重建时重复下载，**不要随手清理**）；
- `data/maps/<map_vNNN>/`：UMAP 坐标、模型与参数（可从表征重建，但版本记录珍贵）。

**可安全清理**：

- `temp/`：切片/临时音频等中间文件。流水线在向量成功写入并登记后自动删除；
  失败时保留便于排查，确认后可运行 `python pipeline.py clean-temp`；
- `logs/`：运行日志，可随时清空；
- Bilibili 下载的原始视频：处理完成后自动删除；`input/local/` 里原本就有的
  测试视频**永远不会**被自动删除。

**安全清理建议**：

1. 清理前先备份数据库与向量：直接复制 `data/catalog.sqlite`、`data/embeddings/`、
   `data/representations/`、`data/visual_dense/`、`data/visual_windows/` 与
   `data/sources/` 目录即可（DB 中路径相对 data 登记）；
2. 不要单独删除 `data/embeddings/` 中的某个版本目录，否则数据库里会出现
   指向不存在文件的记录；确实要删除时先备份，再用 SQLite 工具删掉对应登记行。
3. 建议保留最近 1-2 个星云版本；旧版本可以删 `data/maps/map_vNNN/` 目录与
   `map_runs` / `map_coords` 表中对应行（先备份数据库）。
4. 视频源不可重新获取时，向量就是唯一资产——必要时把 `data/embeddings/`
   归档到其他磁盘。

## 测试与质量门禁

唯一质量入口：

```powershell
.\.venv\Scripts\python.exe qa.py            # 默认：快速离线检查
.\.venv\Scripts\python.exe qa.py --full     # 额外运行 model/network 标记测试
.\.venv\Scripts\python.exe qa.py --only ruff,pytest
.\.venv\Scripts\python.exe qa.py --list
```

默认检查内容：ruff（nebula/tests/顶层脚本）、pytest（排除 `model` / `network` 标记）、
文档（README/ARCHITECTURE/ADR）、CLI 入口 `--help`、配置有效性。全部通过返回 0，
任一失败返回非 0 并汇总失败项。

开发时可直接跑局部检查：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m pytest tests -q -m "not model and not network"
.\.venv\Scripts\ruff.exe check nebula tests
```

- 默认测试离线、快速，不下载模型、不访问网络、不需要 GPU；
- 需要真实 PE-AV 权重或网络的测试分别用 `model` / `network` / `integration` 标记标出；
- E2E 测试使用 `tests/fixtures/video_factory.py` 现场合成的视频（首次会缓存到
  `tests/fixtures/out/`），不依赖任何私人素材；
- v0.2 测试覆盖：`tests/test_similarity.py`（高维近邻/密度/解释字段/片段热图）、
  `tests/test_identity.py`（身份关联与去重）、`tests/test_stability.py`（留一稳定性 + CLI）、
  `tests/test_avsync.py`（错位实验 WAV/Δ 数学与离线 CLI）、`tests/test_app.py`
  （近邻面板、解释列、枢纽、片段探索、筛选）、`tests/test_mapplot.py`（筛选星云与变体）、
  `tests/test_db.py`（schema v2 迁移、work_links、计时）；
- `tests/test_invariants.py` 静态检查 similarity / stability / identity / avsync 的导入边界
  （不依赖 UMAP / 模型 / UI），并覆盖 filter_json 落库、duplicate 默认排除、计时非负、
  近邻解释字段仅展示、片段级数据层纯净等不变量；
- `tests/test_e2e.py` 的 v0.2.x 独立验收：全库分位/双向/hubness 与手算矩阵逐项对照、
  片段热图/最高配对 schema、AppTest 枢纽表与片段热图；
- v0.4 测试覆盖：`tests/test_sampling.py`（plan_full_coverage 覆盖/权重/边界）、
  `tests/test_pipeline.py`（全片片段数/中断恢复/超长视频）、`tests/test_events.py`
  （事件扫描确定性/合成闪帧/音脉冲）、`tests/test_temporal.py`（Full Mean/Temporal Stats 数学性质、
  顺序敏感性）、`tests/test_representations.py`（注册/解析/隔离/legacy 一致性）、
  `tests/test_core_v04.py`（schema v3 迁移/回填/地图绑定）；
- v0.4.3 测试覆盖：`tests/test_dense_visual.py`（预处理 filter/letterbox 黑边/版本含 preprocess/
  bundle roundtrip/旧接口兼容）、`tests/test_dense_visual_av_duration.py`（音轨长于视频轨的
  期望帧数）、`tests/test_dense_visual_representation.py`（视觉表征构建）、
  `tests/test_visual_pipeline.py`（视觉版本/序列状态/断点/清理）、
  `tests/test_embedding_diagnostics.py`（PCA/UMAP 指标、Top-k 保持率、距离相关）；
  `tests/test_invariants.py` 另覆盖 visual_versions/visual_sequences 版本冲突/状态/主键、
  visual 表征登记与地图绑定、不同视觉版本矩阵隔离；
- v0.4.3.1 测试覆盖：`tests/test_sampling.py`（`coverage_weights` 手算/重叠/非整 8s/sum=1）、
  `tests/test_dense_visual.py`（`window_weights` roundtrip/旧 manifest 兼容）、
  `tests/test_build_visual_windows.py`（帧→窗口重建、权重、断点/force、窗口计划隔离）、
  `tests/test_visual_pipeline.py`/`tests/test_download.py`（源缓存复用/不误删）、
  `tests/test_invariants.py`（schema v5、窗口计划主键/状态、帧版本与窗口计划正交、
  不同计划不共享缓存、`sources` 在 data 下且不被 temp 清理）；
- v0.4.4 测试覆盖：`tests/test_consensus_graph.py`（mutual/共识/rank-average/concat 指标、
  确定性、JSON schema）、`tests/test_app.py`/`tests/test_appdata.py`（互为近邻/共识连线开关、
  截断、共识文件缺失回退）、`tests/test_invariants.py`（共识工具离线导入边界）；
- v0.4 全链路 E2E（`tests/test_e2e.py -k v04`）：4 部合成视频 → HashTestEmbedder 全片编码 →
  注入中断 → 只补缺失批 → `tools/build_representation.py full_mean` →
  `map.py build --representation full_mean_v1` → `appdata.load_nebula_frame` →
  `tools/compare_spaces.py legacy×full_mean`，并核对覆盖率/版本隔离/地图代表征/输出结构；
- `tests/test_invariants.py` 的 v0.4 不变量：模式切换与 legacy 公式、full 版本格式、
  不同表征/序列不混矩阵、地图绑定同序列代表征、旧地图坐标不被覆盖、重复 initialize/登记幂等；
- QA 的 CLI 冒烟额外覆盖 `map.py variants`、`ingest.py link-local`、
  `tools/clip_stability.py`、`tools/desync_experiment.py`、`tools/preflight_works.py`，
  v0.4 新 CLI（`build_representation.py` / `compare_spaces.py` / `events_probe.py`）
  也纳入 qa.py 的 `--help` 冒烟；v0.4.3 新增 `visual_pipeline.py`、
  `dense_preprocess_compare.py`、`embedding_diagnostics.py` 同样纳入；
  v0.4.3.1/v0.4.4 的 `build_visual_windows.py` / `build_consensus_graph.py` 也纳入；
- 最终门禁：`python qa.py --full` = **PASS 5/5**（2026-10-26 v0.4.3.1/v0.4.4；pytest
  **901 passed, 1 skipped** 含 model/network，279.04s；ruff 全绿；
  docs 含 v0.2.x/v0.4/v0.4.3/v0.4.3.1/v0.4.4；CLI **16 个入口** --help；
  config 含 visual_dense/window plan/sources 校验，embedding_version=full 序列版本；
  另：真实数据复核 5 PASS / 0 FAIL，真实 AppTest 7 空间/互邻/共识通过）。

## 常见问题

**没有 GPU 能跑吗？**
可以。`model.device: auto` 会自动回退到 CPU；CPU 推理较慢（每部作品数秒到数十秒），
建议先用 `--limit 3` 小批量验证，并保持 `model.batch_size` 较小。

**没有网络 / 无法下载模型权重？**
1. 在可联网环境按上面“预下载 PE-AV 权重”把 `.cache/hf` 准备好并拷入项目；
2. 把 `config.yaml` 中 `model.local_files_only` 设为 `true`；
3. 默认测试不依赖网络；只有 `--full` 或带 `network` 标记的测试会触网。

**下载失败 / 视频已失效怎么办？**
失败会写入 SQLite 的 `failed` 状态并记录可读原因，不影响其他作品。
用 `pipeline.py status` 查看失败列表，修复后
`pipeline.py retry <work_id>` 单独重试。需要登录的视频在
`download.cookies_file` 配置 cookies 文件；网络受限时配置 `download.proxy`。

已知边界：8.7 小时的活动直播回放（BV1b6ivBzEpb）全量下载代价过高，本轮记录为 failed
并保留可读原因（未来可改用分段下载）；竖屏视频的下载格式已通过
`config.yt_dlp_format` 中的 `width<=480` 与 `best` 兜底修复。

**怎么确认音画没有错位？**
`pipeline.py process-local --verify-sync N` 会对前 N 部作品比较切片里
视频与音频的时长差（容差 0.15 秒）；也可用 `nebula.media.clip_duration_report` 手动抽查。

**星云页面打不开 / 提示没有星云数据？**
先确认已成功运行 `map.py build`（`map.py list` 能看到 `map_vNNN` 且状态 ready），
且处理完成的作品数量 >= 3（UMAP 至少需要 3 部作品）。自定义配置路径时设置
`NEBULA_CONFIG` 环境变量指向配置文件。

**为什么两张星云（或筛选版 vs 全量版）的坐标对不上？**
UMAP 是非线性降维，训练集变化会让整个布局重新排布：筛选星云与全量星云、不同筛选星云、
不同 `n_neighbors` 变体之间的坐标都不可直接比较。请只在“同一 map 版本内”比较位置；
需要对比不同数据子集时，分别生成版本，在侧栏切换查看。

**v0.4/v0.4.3 是逐帧音MAD语义理解系统吗？**
不是。v0.4 的准确名称是**「全片覆盖与多尺度音画特征表征系统」**。它记录了两类信息：
(1) PE-AV 每 8s 16 帧（≈2fps）的稀疏联合语义 embedding；
(2) 音频 100Hz / 视频 30fps（96×54 灰度）的高频事件统计（节奏/运动/突变）。
v0.4.3 增加了冻结 CLIP/DINOv2 的**帧级特征通道**（默认 8fps，可选 30fps 小样本），
但仍是固定编码器 + 固定窗口池化，不是逐帧语义理解，也没有训练任何时序模型。
Temporal Stats 是人工设计分组与权重的固定统计聚合器，不是学习模型；本轮没有可学习的
音画时序关系或自监督全曲表示。完整时间覆盖只说明“每个时刻都被窗口覆盖”，
不等于“逐帧内容都被理解”。

**Legacy 和 Full Mean 的星云坐标能直接比较吗？**
不能。两种空间来自不同序列版本与不同聚合器，地图分别训练；坐标绝对位置不可比，
但可用于各自空间内的近邻/密度观察。跨空间比较请用 `tools/compare_spaces.py`
的 Top-k 重合率、稳定对比例与余弦矩阵相关等指标，而不是直接看坐标。

**切换到 full_coverage 会重新下载视频吗？**
`config.yaml` 现在默认已是 `full_coverage`。对尚未做全片编码的作品，需要从视频开头到
结尾的完整时间轴，Bilibili 作品必须重新下载（低分辨率格式）再编码；这是预期的，旧 5 片段向量、
legacy 表征与旧地图（如 `map_v008`）仍保留、不覆盖。只有从已保存序列构建 Full Mean /
Temporal Stats 表征（`tools/build_representation.py`）时不需要重新下载或编码。
若某个 Bilibili 下载接口在部分地区失败，下载层会尝试 playurl 回退（v0.4 新增），失败仍会
写入可读原因而不是静默跳过。

**做了身份关联后，旧的星云会变吗？**
不会。星云是构建时的快照：关联之前生成的旧版本（如 P0 的 `map_v001`）保留当时的作品集合，
不回溯改写；重新运行 `map.py build` 生成的新版本默认排除 duplicate。
同理，失败/未完成的作品没有向量，不会出现在任何星云里（例如 2025 试运行 30 部里 1 部下载失败，
`map_v002` 为 29 部）。

**近邻相似度和图上的远近为什么不一致？**
近邻/相似度/近邻密度都使用高维余弦（模型向量空间），主图是 UMAP 二维/三维投影；
投影只近似保持邻域关系，所以“图上距离”不能当作语义相似度。

**Top5 近邻余弦只有 0.6，算不算“很像”？**
不要只看绝对值：近邻表里的“全库分位”表示该相似度高于当前 canonical 集合多少比例的作品对；
接近 100% 才表示这种相似度在库里很罕见。“双向”表示对方也把当前作品列入 Top5，证据更强；
“枢纽作品”折叠区展示被最多作品列为 Top5 的 hub 点——这是高维嵌入的统计现象，
不代表作品质量。小样本/筛选子集上分位与 hubness 都不稳定，只作观察线索。

**音画错位实验说明什么？**
`tools/desync_experiment.py` 用真实作品对比原始/轻画质退化/音频偏移 +0.5s/+1.0s/换音频的
`Δ=1-cos`，用于评估联合向量对同步破坏的敏感度；方法与局限（仅约 10 部、单次实验、
周期性音乐可能低估错位影响）见 `docs/adr/0008-av-sync-desync-experiment.md`。

**3D 图上点击没有反应？**
Plotly/Streamlit 在部分环境不支持 3D 点击选择，使用侧栏搜索 + 下拉选择作品即可。

**Windows 控制台中文乱码？**
qa.py 已强制 UTF-8 输出；其他命令行建议使用 PowerShell 7，且
`$env:PYTHONUTF8 = "1"`。

## 验收清单

| 阶段 | 完成条件 | 自检命令 |
| --- | --- | --- |
| P0 | 本地数十部视频生成可点击星云；全部正常视频维度一致；坏文件记录原因 | `pipeline.py process-local`、`map.py build`、`app.py`、`qa.py` |
| P1 | 2025 年约 100 部作品自动处理；支持多维染色与跳转；失败可重试 | `ingest.py import-votes ...`、`pipeline.py run-pending --source bilibili`、`pipeline.py status` |
| P2 | 多个历史年份合并；按年份比较；星云版本化保存 | 多次 `ingest.py import-votes --year ...`、`map.py build`、`map.py list` |
| P3 | 新一期周报出现后自动/半自动增量收录 | `ingest.py import-weekly ...`、`pipeline.py run-pending`、`map.py transform` |
| v0.2 A1/A2 | 近邻面板三模态可切换；高维近邻密度染色；筛选星云 filter_json 正确、跨版本坐标不可比提示 | `streamlit run app.py`、`map.py build --dataset/--vote-year`、`map.py variants` |
| v0.2 A3/A4/B1 | 稳定性报告 JSON+CSV 生成；`perf_<run_id>.csv` 生成且耗时非负；link-local 关联后 duplicate 默认不进星云 | `tools/clip_stability.py`、`pipeline.py status`、`ingest.py link-local` |
| v0.2.x 解释/片段 | 近邻表出现“全库分位/双向”；枢纽表可渲染；片段热图与配对表 schema 正确且与手算余弦一致 | `streamlit run app.py`、`pytest -q tests/test_e2e.py -k "p1 or p2"` |
| v0.2.x A6 实验 | 错位实验 JSON/CSV 生成，结论写入 ADR-0008，且未改动主向量/星云 | `tools/desync_experiment.py --limit 10` |
| v0.2.x 扩量 | 89 部选票已导入；预检 JSON 给出建议 batch/skip/done；失败项可读 | `tools/preflight_works.py`、`pipeline.py status` |
| v0.4 全片序列 | 110/110 canonical 完成；任意时刻覆盖、末尾锚定、权重和=1；事件三件套齐全 | `pipeline.py run-pending --sampling full_coverage`、`pipeline.py status` |
| v0.4 表征/地图 | full_mean_v1 / temporal_stats_v1 全量构建；map_v011/map_v012=110，默认 map_v012 | `tools/build_representation.py list`、`map.py list` |
| v0.4 空间对比 | 三组 compare JSON/CSV 生成；Top-k 重合/稳定对/余弦相关/hubness 数字可复现 | `tools/compare_spaces.py`、`docs/reports/full-coverage-analysis.md` |
| v0.4.3 视觉序列 | visual_versions 固定 encoder@revision__fps/size/preprocess；canonical 作品状态/断点/清理可核对 | `tools/visual_pipeline.py status`、`tools/visual_pipeline.py list` |
| v0.4.3 视觉表征/诊断 | CLIP-only / DINOv2-only 视觉表征可建地图；PCA/UMAP 诊断 JSON/CSV 生成 | `tools/visual_pipeline.py run`、`tools/embedding_diagnostics.py`、`map.py list` |
| v0.4.3.1 权重修复 | 窗口层含 `window_weights` 且和为 1；v2 表征按 coverage 权重聚合；v1 只读保留；源缓存可复用 | `tools/build_visual_windows.py`、`tools/build_dense_visual_representation.py`、`tools/compare_spaces.py --a dense_visual_clip_v1 --b dense_visual_clip_v2` |
| v0.4.4 共识 | `logs/neighbor_consensus.json/.csv` 生成（mutual/共识/融合指标）；界面可叠加互为近邻/共识连线 | `tools/build_consensus_graph.py`、`streamlit run app.py` |

交付前至少确认：

- [ ] `.\.venv\Scripts\python.exe qa.py` 全部通过（快速模式离线可跑）；
- [ ] `pytest` 无失败、无意外 skip（并行开发期缺模块的 skip 在交付时应消失）；
- [ ] 同一视频重复处理不会重复编码（数据库状态 completed，向量文件 mtime 不变）；
- [ ] 失败文件中止后不影响其他作品，且错误信息可读；
- [ ] `map.py build` 后 `app.py` 能渲染 2D/3D、染色与搜索；
- [ ] 现有 33 部作品无需重新编码即可使用 A1/A2 近邻、密度与筛选功能；
- [ ] 筛选星云的 filter_json 与界面展示一致，并提示“坐标仅在本版本内可比”；
- [ ] `tools/clip_stability.py` 能生成 `logs/clip_stability.json` 与 `.csv`，且不加载模型；
- [ ] `logs/perf_<run_id>.csv` 与 `work_timings` 字段齐全且非负；
- [ ] `ingest.py link-local` 关联后 duplicate 默认不出现在星云/近邻索引，unlink 可恢复；
- [ ] 近邻表显示“全库分位/双向”，枢纽表可渲染；分位/入度与手算全库成对相似度一致；
- [ ] 片段热图为 5×5（或实际片段数）矩阵，最高配对表含起始秒，`Δ` 统计正确；
- [ ] `tools/desync_experiment.py` 报告有 A/A-quality/B/C/D 五类结果且未修改主向量库；
- [ ] `tools/preflight_works.py` 的 `logs/preflight_89.json` 覆盖 89 行并给出 batch/skip/done 建议；
- [ ] `data/` 已备份，临时目录与下载原视频已清理。

## 文档索引

- `ARCHITECTURE.md`：数据模型、模块边界、关键不变量、可复现性（含 v0.2 近邻/身份/稳定性层）；
- `docs/CONTRACTS.md`：v0.1 模块接口与写权限契约；
- `docs/CONTRACTS_V2.md`：v0.2/v0.3 迭代契约（DB v2、近邻、筛选、身份关联、稳定性、计时）；
- [docs/reports/embedding-pipeline.md](docs/reports/embedding-pipeline.md)：嵌入流程报告（采样 → 切片 → PE-AV → 聚合 → 存储/版本 → 实测耗时与实验局限）；
- `docs/adr/`：关键决策记录——0001-0004（PE-AV 选型、采样规则、星云版本、测试嵌入器边界），
  0005（身份关联与星云去重）、0006（筛选地图与坐标不可比）、0007（高维近邻密度）、
  0008（音画错位实验方法与局限）、0009（近邻解释：相对分位/互为 Top-k/hubness）；
- `docs/agents/QUALITY.md`：质量与验证原则；
- `音MAD星云计划 · 完整执行方案.md`：项目背景与分阶段计划。

## 数据来源与合规

- 下载与处理 Bilibili 视频时请遵守平台使用条款与版权要求，优先处理有权保存用于
  本地分析的作品；不要公开分发下载的原始视频。
- 本项目长期保留的是向量与元数据索引，不是视频文件；分享星云时注意不要泄露
  不适合公开的个人信息（如 cookies、私密清单）。
