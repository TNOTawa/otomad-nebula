# AGENTS.md

> **This file is a map, not a manual.**
> Keep it short, stable, project-specific, and mechanically actionable.
> Put branch-specific knowledge behind pointers; put enforceable rules into tests, CI, linters, architecture checks, or tool permissions.

## Sources of truth

- The current task/spec describes intended behavior.
- Existing code and tests describe current behavior and compatibility obligations.
- Project docs describe architecture, data semantics, and rationale.
- Local/nested `AGENTS.md` files may add subtree-specific facts and commands.
- If these sources conflict, do not silently pick the easiest one. Follow an explicitly documented authority rule; otherwise surface the conflict as a decision.

## Default work protocol

1. **Investigate** — inspect the relevant implementation, tests, docs, and existing call sites before assuming behavior.
2. **Verify need** — establish whether a code change is actually required. A verified **no-change** result is a valid successful outcome.
3. **Resolve uncertainty** — look up facts yourself. Ask only about consequential decisions that remain unresolved after investigation.
4. **Change narrowly** — make the smallest coherent change; reuse existing abstractions and single sources of truth.
5. **Verify independently** — run focused checks while developing, then the project QA gate before completion.
6. **Report evidence** — state what changed, what was run, what passed, what could not be run, and any remaining risk.

## Engineering invariants

- Preserve user data, backward compatibility, and stable identities unless the task explicitly changes them.
- Confirm unfamiliar APIs, units, encodings, indexes, lifetimes, and boundary semantics from code/docs/tests or a minimal experiment.
- Keep dependency direction and architecture boundaries intact.
- Prefer deterministic behavior for generators, compilers, serializers, exporters, and reports when the domain allows it.
- Treat tests, golden files, quality thresholds, and CI rules as part of the verification system. Changes to them require task intent and explicit justification.
- Prefer rules that can be mechanically checked over rules that merely ask the Agent to “be careful”.
- An inability to run a required check is a reported limitation, not an implicit pass.

## Read on demand

- **Material product/architecture ambiguity remains, or the user asks to stress-test a plan / invokes a grilling workflow**  
  → read `docs/agents/DECISIONS.md`.
- **Adding/changing tests, QA, coverage, mutation testing, golden files, or deciding whether work is complete**  
  → read `docs/agents/QUALITY.md`.
- **Changing CI, permissions, sandboxes, architecture checks, agent infrastructure, or instruction layout**  
  → read `docs/agents/HARNESS.md`.
- **Need the evidence behind these rules**  
  → read `docs/agents/RESEARCH.md`.
- **Working inside a subtree with its own `AGENTS.md`**  
  → read that file and treat it as the local delta; do not duplicate root guidance into it.

## Project map

- **Project:** otomad-nebula（音MAD星云计划）
- **Purpose:** 用 PE-AV 多模态模型提取音MAD作品联合向量，持久化为可扩充的高维向量数据库，并用 UMAP 生成可交互 2D/3D 星云（外部元数据仅用于染色/筛选）。v0.4 的准确命名是**「全片覆盖与多尺度音画特征表征系统」**，不是「逐帧音MAD语义理解系统」：PE-AV 每 8s 仅 16 帧（≈2fps）稀疏语义，事件层是 30fps（96×54 灰度）/100Hz 统计；Temporal Stats 是人工分组/权重统计基线，**不是训练模型**；完整覆盖 ≠ 逐帧语义理解。v0.4.3.1 起视觉通道拆成 Frame Encoding / Window Plan / Representation 三层版本（窗口覆盖权重修复）；v0.4.4 的跨模型近邻共识是观察性证据，不做风格分类。v0.5-alpha 研究模型（分层音画时序）已实现并两次真实训练，但**可行性验证未通过（负结果）**，只落盘研究产物，不注册 DB/地图、不宣称学到可靠时序结构。v0.4.5 意外关系发现已真实运行（排名百分位模式 A/B/C，只作探索）；v0.5-beta 密集音频 pilot 20/20 + 全量 110/110 PASS（MERT），局部双塔 3 seeds 仅 weak positive / not passed，不注册 DB/地图、不宣称成功。
- **Main entry point:** 方案文档要求的四个入口：`ingest.py`（导入/身份关联）、`pipeline.py`（下载/切片/编码/清理/计时；v0.4 用 `--sampling full_coverage` 开启全片覆盖）、`map.py`（UMAP 星云版本/筛选/变体/按表征建图）、`app.py`（Streamlit 可视化：2D/3D、近邻、染色、筛选、表征空间切换）；内部实现在 `nebula/`，新增 `similarity.py`（高维近邻）、`identity.py`（本地/BV 去重）、`stability.py`（片段稳定性）、`avsync.py`（错位实验）、`events.py`（高频事件）、`temporal.py`（Full Mean/Temporal Stats 聚合）、`dense_visual.py`（v0.4.3 冻结视觉编码器/窗口池化）、`visual_pipeline.py`（v0.4.3 视觉批处理）、`v05_data.py`（v0.5 token 构建）、`v05_model.py`/`v05_train.py`（v0.5-alpha 研究模型/训练）、`discovery.py`（v0.4.5 意外关系发现）、`audio_dense.py`（v0.5-beta 密集音频）、`v05_beta.py`（v0.5-beta 局部双塔）、`representations.py`（表征注册/加载）。辅助入口：`tools/build_representation.py`、`tools/compare_spaces.py`、`tools/events_probe.py`、`tools/visual_pipeline.py`、`tools/build_visual_windows.py`、`tools/build_dense_visual_representation.py`、`tools/build_consensus_graph.py`、`tools/build_v05_tokens.py`、`tools/train_v05.py`、`tools/explore_relations.py`、`tools/audio_dense.py`、`tools/train_v05beta.py`、`tools/dense_preprocess_compare.py`、`tools/embedding_diagnostics.py`。
- **Architecture map:** `ARCHITECTURE.md`；接口与边界契约见 `docs/CONTRACTS.md` 与 `docs/CONTRACTS_V2.md`（v0.4 见 §10，v0.4.3 视觉契约见 §10.3）；v0.4 设计契约 `docs/plans/v0.4-full-coverage-design.md`；v0.4.3 设计 `docs/plans/v0.4.3-dense-visual-semantics.md`；pilot `docs/reports/dense-visual-pilot.md`；全片表征分析 `docs/reports/full-coverage-analysis.md`；视觉全量报告（数字已回填、t4 复核）`docs/reports/visual-dense-full-run.md`；
v0.4.3.1 权重修复 `docs/reports/v0431-window-weights-and-versioning.md`；
v0.4.4 共识 `docs/reports/v044-consensus-fusion.md`；
v0.5-alpha 设计 `docs/plans/v0.5-alpha-temporal-model.md`、验证报告
`docs/reports/v05-alpha-validation.md`（负结果）；本轮 v0.4.5/v0.5-beta 交付
`docs/reports/v045-v05beta-delivery.md`、音频 pilot `docs/reports/v05beta-audio-pilot.md`；v0.5-beta v2 协议/局部时序/扩展 `docs/reports/v05beta-v2-protocol-local.md`；
执行计划见 `音MAD星云计划 · 完整执行方案.md`。
- **Build/setup:** Python 3.10 + `.venv`（uv）。`uv venv .venv`；`uv pip install -e ".[ml,dev]"`（ffmpeg/ffprobe 需在 PATH）；配置 `config.yaml`。
- **Fast relevant tests:** `.\\.venv\\Scripts\\python.exe -m pytest tests -q`；`ruff check nebula tests`。
- **Full QA command:** `.\\.venv\\Scripts\\python.exe qa.py`（ruff + pytest + 文档 + CLI + config 五项；`--full` 含 model/network 测试）。
- **Release/build artifact:** 本地 `streamlit run app.py` 网页；核心资产是 `data/catalog.sqlite` + `data/embeddings/` 高维向量（序列层），聚合后的派生表征在 `data/representations/`，v0.4.3 帧级密集视觉特征在 `data/visual_dense/`，v0.4.3.1 窗口层在 `data/visual_windows/`，Bilibili 源视频缓存在 `data/sources/`，v0.5 研究 token/checkpoint 在 `data/v05_tokens/` 与 `data/models/`，
v0.5-beta 密集音频在 `data/audio_dense/`（均不注册 DB），星云版本在 `data/maps/`。
- **Critical data/compatibility notes:**
  - 外部元数据（年份/作者/排名/时长/发布日期）绝不进入模型编码或 UMAP 降维，只用于展示/染色/筛选。
  - 向量版本由模型版本+采样参数决定（`config.embedding_version`）；换模型/采样参数必须生成新版本，禁止覆盖旧版本。
  - 星云版本 `map_vNNN` 每次新生成，2D 与 3D 分别训练；P3 增量用保存的 UMAP `transform()`。
  - 作品去重键：本地相对路径 `local:<relpath>`；Bilibili `bilibili:<bvid>:p<part>`；跨年份/榜单只追加 appearances。
  - 状态机 `pending -> downloading -> encoding -> completed`，失败 `failed` 可重试；`completed` 只能通过显式 `Catalog.reopen()` 重开。
  - 任务状态、向量登记、星云版本以 SQLite 为唯一事实来源；临时文件仅在向量成功登记后清理，本地原始视频永不删除。
  - PE-AV 加载依赖 transformers 5.20.2 的 `TimmWrapperConfig.num_labels` workaround（见 `nebula/embed.py`），升级 transformers 后必须重验。
  - 身份关联：`work_links` 中 canonical 是星云保留实体，duplicate 默认排除（npy/记录保留）；关联依据是规范化标题完全相等 + Bilibili 分P时长差 ≤3s，不确定不自动合并。
  - 筛选星云把条件写入 `map_runs.filter_json`；不同版本/不同筛选的 UMAP 坐标不可直接比较绝对位置。
  - 分阶段耗时写入 `work_timings`/`pipeline_runs`，明细 CSV 在 `logs/perf_<run_id>.csv`（`pipeline.py timings` 查看）。
  - 已知边界：超长活动直播回放（如 8.7h）暂不做全量下载，记录 failed；竖屏视频下载需 width 维度筛选（config.yt_dlp_format 已含）。
  - **v0.4 分层**：序列层 `data/embeddings/<sequence_version>/` 只编码一次；表征层 `data/representations/<name>/<sequence_version>/` 只做聚合。聚合器变化（legacy→full_mean→temporal_stats）**禁止重新编码 PE-AV**，只新增 `representations` 行。当前真实数据：110/110 canonical 全片 assembled；full_mean_v1(id=2)/temporal_stats_v1(id=3) 全量；地图 map_v011/map_v012（legacy=map_v008，默认 map_v012）。
  - **v0.4 全片覆盖**：`sampling.mode=full_coverage`（当前默认）无片段数上限、末尾锚定、窗口权重和为 1；编码按 `.partial/batch_*.npz` 原子分批，重启经 `sequence_states` 只补缺失批；超长视频需 `--allow-long`。真实 110 部共 4130 窗口（legacy 536），事件三件套 110/110。Bilibili 下载失败时优先 playurl 回退（v0.4），仍失败才记 failed。
  - **v0.4 空间隔离**：不同 sequence/representation 的向量禁止混入同一 UMAP 或近邻矩阵；`map_runs.representation_id` 必须与地图的 sequence version 一致；旧 `embedding_version` 目录与旧地图只读、不覆盖。
  - **v0.4 事件/时间结构**：`event_scan.npz`/`event_features.npy`/`clip_times.npy` 与窗口一一对应；事件只是节奏/运动补充，不宣称逐帧语义理解；`clip_times` 的 `weight` 用于加权聚合，时间结构不进入坐标以外的任何元数据。
  - **v0.4.3 视觉命名空间**：逻辑 `visual_version` 形如 `model_id@revision__fps{fps}_s{size}_{preprocess}`（固定 CLIP/DINOv2 revision；默认 8fps/224/letterbox；30fps 仅子集）；**DB `visual_versions.version`、`data/visual_dense/` 目录与 `map_runs.embedding_version` 使用 `visual_version_slug`（slugify："/"、"@"→"-"）**，model_id/revision 原值保留；encoder/revision/fps/size/preprocess 任一变化必须新版本，禁止覆盖；与 PE-AV `sequence_version` 完全独立，禁止混入同一近邻/UMAP 矩阵。
  - **v0.4.3 视觉层**：`visual_versions`/`visual_sequences`（schema v4，状态 pending/in_progress/completed/failed）记录帧编码版本与作品×版本任务；`data/visual_dense/<frame_version>/<work_id>/` 保存帧级 fp16（`dense_visual.npz`）+ manifest（v0.4.3.1 起窗口级移到 `data/visual_windows/`）；只处理 canonical+completed 作品、每部只下载一次、断点续跑、全部任务成功后清理 temp。真实全量：4 个 visual_versions、**252/252 任务 completed**、8fps 每编码器 110 部/260,885 帧/4,130 窗口、30fps 子集每编码器 16 部/140,149 帧/593 窗口、存储 673.9MB。
  - **v0.4.3 表征/地图**：窗口级视觉向量登记为 `kind=visual` 表征（`dense_visual_clip_v1`(id=4,512d)/`dense_visual_dino_v1`(id=5,384d)，各 110 向量），`source_kind=visual`；地图 `source_kind` 由表征推导，视觉地图 `embedding_version` 存视觉版本 slug；当前 `map_v013`（CLIP-only）/ `map_v014`（DINOv2-only）；默认地图仍优先 PE-AV 无筛选最新 `map_v012`；v0.4.3 为 5 空间，v0.4.3.1 后 7 空间（视觉 v1/v2 并存），视觉空间需手动切换。
  - **v0.4.3 能力边界**：冻结编码器帧级特征 + 固定窗口池化是“特征”而非“逐帧语义理解”；没有训练可学习音画时序模型；30fps 只服务帧级轨迹研究，不默认全量。
  - **v0.4.3.1 三层版本与覆盖权重**：Frame Encoding Version（model+revision+fps+size+preprocess）、Window Plan Version（`config.visual_window_plan_version`=`win{clip}s_ov{overlap}__{weight_scheme}`，如 `win8s_ov0__coverage-v1`）、Representation Version（`dense_visual_clip_v2`/`dense_visual_dino_v2`）三者正交；窗口层 `data/visual_windows/<window_plan_version>/<frame_version>/<work_id>/` 存 `windows.npy`/`window_std.npy`/`window_weights.npy`/`manifest.json`；DB 用 `visual_window_plans`/`visual_window_sequences`（schema v5，PK work+frame+window_plan）；`weight_scheme=coverage-v1` 默认，`equal` 仅对照；修复不重新编码帧。真实结果：window plan=win8s_ov0__coverage-v1，252/252 completed、9446 窗口、weight_sum 0.999999847..1.000000209；v2 表征 id=6/7（512/384）各 110、地图 map_v015/v016=110；v1→v2 余弦 min CLIP 0.999972/DINO 0.999863（本数据 8s/ov0 影响很小，但正确性修复）。
  - **v0.4.3.1 源缓存**：`paths.sources` 默认 `data/sources`（位于 data 下，**不随 temp 清理**）；Bilibili 优先复用可 probe 的源文件，未命中才下载；`clean-temp` 不得删除 sources。
  - **v0.4.4 共识**：`tools/build_consensus_graph.py` 读取 `full_mean_v1`+`dense_visual_clip_v2`+`dense_visual_dino_v2` 的 Top-k mutual/共识边与 rank-average/concat 轻量融合，输出 `logs/neighbor_consensus.json/.csv`；界面可叠加互为近邻/共识连线（默认关、截断提示、缺失回退）；共识是观察性证据，不做风格分类/主模型。真实结果：mutual 61/75/73，共识边 53（全共识 22），覆盖 44/110（40.00%），强度 0.8050；融合 rank Jaccard 0.5331/共识 0.4303，concat 0.5358/0.4106 与 0.5385/0.4121；真实 AppTest 7 空间（v008/v011/v012/v013/v014/v015/v016）、默认 map_v012。
  - **v0.5-alpha 研究模型（负结果）**：`nebula/v05_data.py`/`v05_model.py`/`v05_train.py`、`tools/build_v05_tokens.py`/`train_v05.py`；数据 `data/v05_tokens/v05_alpha_v1/`（110 部；PE-AV 窗口 4130，max_segments=128 截断后 token 段 3904（6 部截断、丢 226 段）；141.1MB）；产物 checkpoint/metrics/研究向量在 `data/models/`、`logs/`、`data/representations/transformer_alpha_v1/`；**未注册 representations 表、未建 map**。v1 mask 0.8149（基线 0.8249）/rank 1.03；v2 mask 0.8131、AV 0.40/AUC 0.635、order 0.10、rank 4.66、reverse -0.0048、Top-5 0.0468；**可行性未通过**。下一步补密集音频语义、局部 AV InfoNCE、相对时间 order、whitening/duration 去偏、更多数据/30fps；达标前不注册主表征/地图。
  - **v0.4.5 意外关系发现**：`nebula/discovery.py` + `tools/explore_relations.py`；只用高维余弦排名百分位（p=1-(r-1)/(N-2)，无向对 bar_p），模式 A/B/C；输出 `logs/unexpected_relations.json`（约 297KB）+ CSV（85KB）；真实结果：6 空间/110 部/skipped=0，模式 A 950 候选、B 864 候选、common 99 对（>=3 空间 70）、visual_vs_peav 19；不写 DB/不建 map/不加载模型/不联网；染色/元数据不进入分数；界面“意外关系探索”已接入。
  - **v0.5-beta 密集音频/局部双塔**：`nebula/audio_dense.py` + `tools/audio_dense.py` → `data/audio_dense/<audio_version>/<work_id>/audio.npz` + manifest（model/revision/sample_rate/context/hop/layer/pool/bin/normalize）；pilot 20/20 PASS（2582.753s、193,179 帧/5,177 bins、RTF 0.00947、314.1MiB）；全量 110/110 ok（32,619.14s、2,439,577 帧/65,291 bins、RTF 0.00812、4,158,970,616B≈3.87GiB、pass=true；strict_container 108/110，work50/99 tail mask=False 已从 beta 排除）；`nebula/v05_beta.py` + `tools/train_v05beta.py` 用 v05_beta_v1（110 部/4,130 段/0 截断/142.3MiB）训练视觉/音频双塔；音频可用 107 部、74/21/12 划分，3 seeds test Top-1 0.296557/0.302779/0.305268（均值 0.301535±0.003663）；vs random +0.03108、vs frozen +0.02779、vs simple_sync +0.00912；消融双模态掉点，但相对 simple_sync 仅 +0.91pp、best epoch 1–6 —— **weak positive / not passed**；产物只落 `data/models/v05_beta_<seed>/` 与 `logs/`，**不注册 DB/不建 map、不进入全曲编码、不改 alpha 负结果**。

## Completion

Completion is based on **observable evidence**, not Agent confidence.

A normal completion report should be compact and factual, for example:

```text
Changed: ...
Focused checks: ...
Full QA: ...
Golden/determinism: ...
Not run / remaining risk: ...
```