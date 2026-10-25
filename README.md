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
├── nebula/                    # 核心库：config/db/sampling/media/embed/pipeline/...
├── tests/                     # pytest 测试（含 tests/fixtures 合成视频工厂）
├── docs/                      # 契约、ADR、质量规范
├── ingest.py                  # 导入本地视频 / 选票 / 周报
├── pipeline.py                # 下载 → 切片 → 推理 → 保存 → 清理
├── map.py                     # 生成/更新 2D/3D 星云
├── app.py                     # Streamlit 交互界面
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

### 界面使用

- 侧栏：选择星云版本、2D/3D 切换、染色维度、按文件名/标题/BV 搜索、点大小与透明度；
- 主图：Plotly 散点图，支持缩放、旋转、悬浮；点击点显示详情；
- 详情：标题、BV/URL（可点击跳转 Bilibili）、时长、发布日期、作者、选票排名/得分、本地文件；
- 3D 点击选择在部分浏览器/Plotly 版本下不可用，可用“搜索 + 下拉选择作品”的等价入口；
- 无数据（尚未生成星云）时页面显示中文提示，而不是报错堆栈。

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
  `tests/fixtures/out/`），不依赖任何私人素材。

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

**怎么确认音画没有错位？**
`pipeline.py process-local --verify-sync N` 会对前 N 部作品比较切片里
视频与音频的时长差（容差 0.15 秒）；也可用 `nebula.media.clip_duration_report` 手动抽查。

**星云页面打不开 / 提示没有星云数据？**
先确认已成功运行 `map.py build`（`map.py list` 能看到 `map_vNNN` 且状态 ready），
且处理完成的作品数量 >= 3（UMAP 至少需要 3 部作品）。自定义配置路径时设置
`NEBULA_CONFIG` 环境变量指向配置文件。

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

交付前至少确认：

- [ ] `.\.venv\Scripts\python.exe qa.py` 全部通过（快速模式离线可跑）；
- [ ] `pytest` 无失败、无意外 skip（并行开发期缺模块的 skip 在交付时应消失）；
- [ ] 同一视频重复处理不会重复编码（数据库状态 completed，向量文件 mtime 不变）；
- [ ] 失败文件中止后不影响其他作品，且错误信息可读；
- [ ] `map.py build` 后 `app.py` 能渲染 2D/3D、染色与搜索；
- [ ] `data/` 已备份，临时目录与下载原视频已清理。

## 文档索引

- `ARCHITECTURE.md`：数据模型、模块边界、关键不变量、可复现性；
- `docs/CONTRACTS.md`：模块接口与写权限契约（并行开发约定）；
- `docs/adr/`：关键决策记录（PE-AV 选型、采样规则、星云版本、测试嵌入器边界）；
- `docs/agents/QUALITY.md`：质量与验证原则；
- `音MAD星云计划 · 完整执行方案.md`：项目背景与分阶段计划。

## 数据来源与合规

- 下载与处理 Bilibili 视频时请遵守平台使用条款与版权要求，优先处理有权保存用于
  本地分析的作品；不要公开分发下载的原始视频。
- 本项目长期保留的是向量与元数据索引，不是视频文件；分享星云时注意不要泄露
  不适合公开的个人信息（如 cookies、私密清单）。
