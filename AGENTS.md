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
- **Purpose:** 用 PE-AV 多模态模型提取音MAD作品联合向量，持久化为可扩充的高维向量数据库，并用 UMAP 生成可交互 2D/3D 星云（外部元数据仅用于染色/筛选）。
- **Main entry point:** 方案文档要求的四个入口：`ingest.py`（导入）、`pipeline.py`（下载/切片/编码/清理）、`map.py`（UMAP 星云版本）、`app.py`（Streamlit 可视化）；内部实现在 `nebula/`。
- **Architecture map:** `ARCHITECTURE.md`；接口与边界契约见 `docs/CONTRACTS.md`；执行计划见 `音MAD星云计划 · 完整执行方案.md`。
- **Build/setup:** Python 3.10 + `.venv`（uv）。`uv venv .venv`；`uv pip install -e ".[ml,dev]"`（ffmpeg/ffprobe 需在 PATH）；配置 `config.yaml`。
- **Fast relevant tests:** `.\\.venv\\Scripts\\python.exe -m pytest tests -q`；`ruff check nebula tests`。
- **Full QA command:** `.\\.venv\\Scripts\\python.exe qa.py`（ruff + pytest + 文档 + CLI + config 五项；`--full` 含 model/network 测试）。
- **Release/build artifact:** 本地 `streamlit run app.py` 网页；核心资产是 `data/catalog.sqlite` + `data/embeddings/` 高维向量，星云版本在 `data/maps/`。
- **Critical data/compatibility notes:**
  - 外部元数据（年份/作者/排名/时长/发布日期）绝不进入模型编码或 UMAP 降维，只用于展示/染色/筛选。
  - 向量版本由模型版本+采样参数决定（`config.embedding_version`）；换模型/采样参数必须生成新版本，禁止覆盖旧版本。
  - 星云版本 `map_vNNN` 每次新生成，2D 与 3D 分别训练；P3 增量用保存的 UMAP `transform()`。
  - 作品去重键：本地相对路径 `local:<relpath>`；Bilibili `bilibili:<bvid>:p<part>`；跨年份/榜单只追加 appearances。
  - 状态机 `pending -> downloading -> encoding -> completed`，失败 `failed` 可重试；`completed` 只能通过显式 `Catalog.reopen()` 重开。
  - 任务状态、向量登记、星云版本以 SQLite 为唯一事实来源；临时文件仅在向量成功登记后清理，本地原始视频永不删除。
  - PE-AV 加载依赖 transformers 5.20.2 的 `TimmWrapperConfig.num_labels` workaround（见 `nebula/embed.py`），升级 transformers 后必须重验。

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
