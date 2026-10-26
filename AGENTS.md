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
- **Purpose:** 用 PE-AV 多模态模型提取音MAD作品联合向量，持久化为可扩充的高维向量数据库，并用 UMAP 生成可交互 2D/3D 星云（外部元数据仅用于染色/筛选）。v0.4 的准确命名是**「全片覆盖与多尺度音画特征表征系统」**，不是「逐帧音MAD语义理解系统」：PE-AV 每 8s 仅 16 帧（≈2fps）稀疏语义，事件层是 30fps（96×54 灰度）/100Hz 统计；Temporal Stats 是人工分组/权重统计基线，**不是训练模型**；完整覆盖 ≠ 逐帧语义理解。
- **Main entry point:** 方案文档要求的四个入口：`ingest.py`（导入/身份关联）、`pipeline.py`（下载/切片/编码/清理/计时；v0.4 用 `--sampling full_coverage` 开启全片覆盖）、`map.py`（UMAP 星云版本/筛选/变体/按表征建图）、`app.py`（Streamlit 可视化：2D/3D、近邻、染色、筛选、表征空间切换）；内部实现在 `nebula/`，新增 `similarity.py`（高维近邻）、`identity.py`（本地/BV 去重）、`stability.py`（片段稳定性）、`avsync.py`（错位实验）、`events.py`（高频事件）、`temporal.py`（Full Mean/Temporal Stats 聚合）、`representations.py`（表征注册/加载）。辅助入口：`tools/build_representation.py`、`tools/compare_spaces.py`、`tools/events_probe.py`。
- **Architecture map:** `ARCHITECTURE.md`；接口与边界契约见 `docs/CONTRACTS.md` 与 `docs/CONTRACTS_V2.md`（v0.4 见 §10）；v0.4 设计契约 `docs/plans/v0.4-full-coverage-design.md`；全片表征分析 `docs/reports/full-coverage-analysis.md`；执行计划见 `音MAD星云计划 · 完整执行方案.md`。
- **Build/setup:** Python 3.10 + `.venv`（uv）。`uv venv .venv`；`uv pip install -e ".[ml,dev]"`（ffmpeg/ffprobe 需在 PATH）；配置 `config.yaml`。
- **Fast relevant tests:** `.\\.venv\\Scripts\\python.exe -m pytest tests -q`；`ruff check nebula tests`。
- **Full QA command:** `.\\.venv\\Scripts\\python.exe qa.py`（ruff + pytest + 文档 + CLI + config 五项；`--full` 含 model/network 测试）。
- **Release/build artifact:** 本地 `streamlit run app.py` 网页；核心资产是 `data/catalog.sqlite` + `data/embeddings/` 高维向量（序列层），聚合后的派生表征在 `data/representations/`，星云版本在 `data/maps/`。
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