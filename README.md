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
  输出 JSON/CSV，只用于风险评估，不参与坐标与质量评价。

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
uv pip install --python ..venvScriptspython.exe --index-url https://download.pytorch.org/whl/cu126 ^
  "torch==2.14.1+cu126" "torchvision==0.29.1+cu126"

# 确认 CUDA 可用
..venvScriptspython.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
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
│   ├── embeddings/<版本>/<work_id>/   # 高维向量（npy）
│   └── maps/<map_vNNN>/       # UMAP 坐标、模型、meta.json
├── temp/                      # 流水线临时目录（成功自动清理，失败保留排查）
├── logs/                      # 运行日志
├── nebula/                    # 核心库：config/db/sampling/media/embed/pipeline/
│                              #   similarity(近邻) identity(身份关联) stability(稳定性) mapplot/appdata
├── tools/                     # 证据/报告工具：bench_peav.py、verify_peav.py、clip_stability.py
├── tests/                     # pytest 测试（含 tests/fixtures 合成视频工厂）
├── docs/                      # 契约（CONTRACTS/CONTRACTS_V2）、ADR、质量规范
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
| `paths` | `input_local` / `input_lists` / `data` / `temp` / `logs` | 数据目录布局 |
| `database` | `filename` | SQLite 文件名（位于 data 目录内，不能带子目录） |
| `media` | `clip_seconds` / `clips_per_work` / `min_duration_seconds` / `audio_sample_rate` 等 | 采样与切片参数；改动会改变向量版本 |
| `model` | `checkpoint` / `revision` / `version` / `frames` / `device` / `local_files_only` 等 | PE-AV 模型；`revision` 固定版本便于复现 |
| `umap` | `metric` / `n_neighbors` / `min_dist` / `random_state` / `n_jobs` / `dimensions` | 降维参数；固定 `random_state`、`n_jobs: 1` 保证可复现 |
| `download` | `yt_dlp_format` / `max_height` / `cookies_file` / `proxy` / `retries` | Bilibili 下载；登录可见视频需 cookies |
| `bilibili` | `api_timeout_seconds` / `user_agent` | 元数据接口请求参数 |
| `app` | `title` / `point_size` / `opacity` / `density_grid` | 界面默认值 |
| `logging` | `level` | 日志级别（`INFO` 建议保留） |

两个自动派生的版本号**不要手工修改**：

- `embedding_version`：由模型版本 + 帧数 + 片段长度/数量 + 音频采样率组成，
  例如 `peav-small-16f__f16_clip8s_x5_48k`。模型或采样参数变化会自动得到新版本，
  旧向量文件保留、不覆盖。
- `map_version`：例如 `map_v001`，每次生成/更新星云自动递增。

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

现有 33 部作品与 map_v001 可以直接使用本节功能，不需要重新编码。

#### A1 高维近邻探索（界面）

启动界面后点击任意点（或通过侧栏搜索/下拉选择作品），详情面板显示 joint / audio / video
三个模态的 Top5 近邻与余弦值；缺失模态显示“无该模态向量”。2D 视图默认高亮 Top5 并画出
选中点到近邻的连线，3D 视图提供“显示近邻连线”开关（默认关闭）。
近邻与相似度全部来自高维余弦（nebula.similarity），与 UMAP 平面距离无关。

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

### 界面使用

- 侧栏：星云版本、2D/3D 切换、染色维度（含“高维近邻密度”）、按文件名/标题/BV 搜索、
  点大小/透明度、数据集筛选（来源 / 榜单 / 年份）；
- 主图：Plotly 散点图，支持缩放、旋转、悬浮；点击点（或搜索 + 下拉选择）显示详情；
- 详情：标题、BV/URL（可点击跳转 Bilibili）、时长、发布日期、作者、选票排名/得分、本地文件，
  以及三个模态的 Top5 近邻（余弦值）；2D 高亮近邻并连线，3D 可开关连线；
- 3D 点击选择在部分浏览器/Plotly 版本下不可用，可用“搜索 + 下拉选择作品”的等价入口；
- 无数据（尚未生成星云）时页面显示中文提示，而不是报错堆栈；筛选星云会标注
  “坐标仅在本版本内可比”。

## 数据目录与安全清理

**永久保存（不要随手删）**：

- `data/catalog.sqlite`：作品、榜单、向量登记、处理状态、星云版本；
- `data/embeddings/<向量版本>/`：作品级与片段级向量（核心资产，重新生成成本高）；
- `data/maps/<map_vNNN>/`：UMAP 坐标、模型与参数（可从向量重建，但版本记录珍贵）。

**可安全清理**：

- `temp/`：切片/临时音频等中间文件。流水线在向量成功写入并登记后自动删除；
  失败时保留便于排查，确认后可运行 `python pipeline.py clean-temp`；
- `logs/`：运行日志，可随时清空；
- Bilibili 下载的原始视频：处理完成后自动删除；`input/local/` 里原本就有的
  测试视频**永远不会**被自动删除。

**安全清理建议**：

1. 清理前先备份数据库与向量：直接复制 `data/catalog.sqlite` 与
   `data/embeddings/` 目录即可（两者要一起备份，路径是相对 data 目录登记的）。
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
- v0.2 测试覆盖：`tests/test_similarity.py`（高维近邻/密度）、`tests/test_identity.py`
  （身份关联与去重）、`tests/test_stability.py`（留一稳定性数学性质 + CLI）、
  `tests/test_app.py`（近邻面板、筛选）、`tests/test_mapplot.py`（筛选星云与变体）、
  `tests/test_db.py`（schema v2 迁移、work_links、计时）；
- `tests/test_invariants.py` 静态检查 similarity / stability / identity 的导入边界
  （不依赖 UMAP / 模型 / UI），并覆盖 filter_json 落库、duplicate 默认排除、计时非负等 v2 不变量；
- QA 的 CLI 冒烟额外覆盖 `map.py variants`、`ingest.py link-local`、
  `tools/clip_stability.py --help`。

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

**做了身份关联后，旧的星云会变吗？**
不会。星云是构建时的快照：关联之前生成的旧版本（如 P0 的 `map_v001`）保留当时的作品集合，
不回溯改写；重新运行 `map.py build` 生成的新版本默认排除 duplicate。
同理，失败/未完成的作品没有向量，不会出现在任何星云里（例如 2025 试运行 30 部里 1 部下载失败，
`map_v002` 为 29 部）。

**近邻相似度和图上的远近为什么不一致？**
近邻/相似度/近邻密度都使用高维余弦（模型向量空间），主图是 UMAP 二维/三维投影；
投影只近似保持邻域关系，所以“图上距离”不能当作语义相似度。

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
- [ ] `data/` 已备份，临时目录与下载原视频已清理。

## 文档索引

- `ARCHITECTURE.md`：数据模型、模块边界、关键不变量、可复现性（含 v0.2 近邻/身份/稳定性层）；
- `docs/CONTRACTS.md`：v0.1 模块接口与写权限契约；
- `docs/CONTRACTS_V2.md`：v0.2/v0.3 迭代契约（DB v2、近邻、筛选、身份关联、稳定性、计时）；
- `docs/adr/`：关键决策记录——0001-0004（PE-AV 选型、采样规则、星云版本、测试嵌入器边界），
  0005（身份关联与星云去重）、0006（筛选地图与坐标不可比）、0007（高维近邻密度）；
- `docs/agents/QUALITY.md`：质量与验证原则；
- `音MAD星云计划 · 完整执行方案.md`：项目背景与分阶段计划。

## 数据来源与合规

- 下载与处理 Bilibili 视频时请遵守平台使用条款与版权要求，优先处理有权保存用于
  本地分析的作品；不要公开分发下载的原始视频。
- 本项目长期保留的是向量与元数据索引，不是视频文件；分享星云时注意不要泄露
  不适合公开的个人信息（如 cookies、私密清单）。
