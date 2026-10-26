# v0.4.5 / v0.5-beta 交付报告

> 本文件合并记录本轮三阶段（工程收敛 → v0.4.5 → v0.5-beta）的真实结果与限制。
> 状态：Phase 0/1/2/3 完成（2026-10-26）；t4 真实产物/界面复核 4 PASS/0 FAIL，
> qa.py --full PASS 5/5；本文件不改负结果、不复制大段实验日志；未 git commit。

## Phase 0：工程审计与收敛（G0）

### 基准

- 起始提交：`47904d9`（本地与 GitHub main 一致）
- 工作树：干净；无未提交修改
- `python qa.py --full`：**PASS 5/5**，pytest **951 passed, 1 skipped**，总耗时 238.69s
- 真实库：schema v5，无未完成迁移；默认地图 map_v012；7 空间可切换；共识 JSON 可读

### 审计表

| 模块/功能 | 当前用途 | 实际调用入口 | 问题 | 处理决定 |
| --- | --- | --- | --- | --- |
| `nebula/highdim.py` | 新增共享行归一化/余弦矩阵/Top-k 邻域 | consensus、diagnostics、v0.4.5 | 两工具原先各有一份数学实现 | **合并**（行为由原测试锁定） |
| `tools/build_consensus_graph.py` | 跨模型共识/融合 | qa CLI、测试、App 数据 | 本地 normalize/cosine/topk 重复 | 改为 import highdim；旧名保留别名 |
| `tools/embedding_diagnostics.py` | PCA/UMAP 诊断 | qa CLI、测试、App 链接 | 同上（距离升序语义不同） | 改为 import highdim；旧名保留别名 |
| `tools/compare_sequences.py` | 旧序列版本对照脚本 | 无 CLI/测试/文档引用 | 无活跃入口、无历史契约 | **删除**（Git 可恢复） |
| `nebula/v05_*.py`、`tools/*v05*` | v0.5-alpha 实现 | v0.5 报告/产物 | alpha 负结果必须保留 | 保留；beta 复用而非复制 |
| `tools/bench_*.py`、`verify_peav.py` | 历史性能/真实模型证据 | README/ARCH/CONTRACTS 引用 | 有文档契约 | 保留 |
| `data/sources` | Bilibili 源缓存 | 下载流水线 | 当前无缓存视频（仅占位文件） | 保留机制；本轮不重复下载 |
| 配置项 | config.yaml | 全部 Python 代码 | 无未引用叶子配置 | 保留 |

### 收敛结果

- 新增 `nebula/highdim.py`，删除两个工具中的重复数学实现；旧内部名以别名保留，调用点/测试不破坏。
- 删除无引用的 `tools/compare_sequences.py`（**224 行**；git 可恢复）。
- 未进行大范围重构：v05 研究代码、共识/诊断的空间业务逻辑与 UI 数据层职责不同，强行合并会增加风险。

### 兼容风险

- `normalize_vectors` 的错误文案由“无法计算余弦距离”统一为通用文案；仅错误文本差异，无行为差异。
- 旧工具模块属性 `normalize_vectors`/`topk_neighbors`/`cosine_distance_matrix` 仍可通过导入名访问，测试已覆盖。

### Phase 0 复核（t4，2026-10-26）

逐项核对 `ff582c4` 与工作树，以上审计表事实成立：

| 核对项 | 结果 |
| --- | --- |
| `nebula/highdim.py` | 新增共享归一化/余弦/Top-k；`build_consensus_graph`、`embedding_diagnostics` 已改为 import；两个工具的旧内部名别名保留（`topk_neighbors` 分别指向 similarity/distance 版本，语义独立）；`tests/test_highdim.py` 4 项通过 |
| 删除 `tools/compare_sequences.py` | 文件已删除（原 224 行）；`git grep compare_sequences` 仅本报告有记录，无代码/测试/文档/qa 引用 |
| v0.5-alpha 研究代码 | 保留；beta 只复用 `v05_data` 的加载/构建接口，不复制 alpha 全管线（待 beta 实现验证） |
| bench/verify 脚本 | 保留（README/ARCH/CONTRACTS 有历史证据契约） |
| `data/sources` | 仅 .gitkeep 占位，无视频缓存；机制保留、本轮不重复下载 |
| 配置项 | 静态检查未发现未引用叶子配置；本轮不新增配置键 |
| 文档/QA | 基线 `qa.py --full` = PASS 5/5、pytest 951 passed/1 skipped；CLI 18 入口；本报告随阶段推进追加 |

**保留项/风险**：

1. `highdim` 同时提供 similarity 降序与 distance 升序两套 Top-k；调用方必须按语义选择，混用会得到错误邻域（现有测试覆盖两种排序，但仍属调用方责任）。
2. 删除的 `compare_sequences.py` 对仓库外用户脚本可能仍有引用；Git 历史可恢复，本轮不提供替代 CLI（如需可在下一轮按新契约重建）。
3. 高维运算合并只统一底层数学，不统一“业务排序语义”；共识/诊断/发现模块仍各自决定用相似度还是距离，避免把不同任务强行合并。
4. 本轮不改变 v0.5-alpha 负结果、不注册研究模型、不重跑真实训练；beta/音频结果另节追加。
5. `data/sources` 当前无真实缓存，若后续需要复现下载路径，需先跑一次真实流水线生成缓存。

## Phase 1：v0.4.5 高维意外关系发现

- 代码/入口：`nebula/discovery.py`（数据/评分层，复用 `nebula/highdim.py` 与
  `representations.load_representation_matrix`）、`tools/explore_relations.py`；
- 真实运行：`python tools/explore_relations.py --top 50 --k 10` →
  `logs/unexpected_relations.json`（**303,793 B**）+ `logs/unexpected_relations.csv`（**86,604 B**）；
- 数据：6 个空间（full_mean_v1 joint/audio/video、temporal_stats_v1 joint、
  dense_visual_clip_v2、dense_visual_dino_v2），110 部 canonical 共同作品，skipped=0；
  分数只用排名百分位 p=1-(r-1)/(N-2)，无向对取 bar_p；cos 仅展示；
- 结果摘要（官方最终）：
  | 类别 | 候选/结果 | 示例（work_id 对，分数） |
  | --- | --- | --- |
  | 模式 A（joint vs audio/video） | 950 候选，Top 50 | 19–46（0.3287）、13–69（0.3194） |
  | 模式 B（temporal vs full_mean） | 864 候选，Top 50 | 18–101（0.6898）、25–30（0.6713） |
  | 模式 C common（>=2 空间 mutual） | 99 对（all3 70 / peav_clip 22 / peav_dino 7） | 20–22（0.9985）、36–117（0.9969） |
  | 模式 C visual_vs_peav | 19 对 | 55–108（0.1806）、58–79（0.1435） |
  | 模式 C 单模型 | peav 51 / clip 60 / dino 60 对 | — |
- t4 复核：JSON/CSV 存在；110 部/6 空间；百分位公式逐项核对（JSON 6 位小数）通过；
  top 示例与 JSON 一致；不写 DB/不建 map/不加载模型/不联网；染色字段不参与分数；
- 界面：`app.py` 的“意外关系探索”（模式表/选中高亮/Top-k/互邻筛选/CSV）由 t2 接入，
  不改变默认地图/7 空间；真实 AppTest 默认 map_v012 + 7 空间 + 模式 A/B/C + 高亮/邻域/
  CSV/染色不改分数全部通过（详见 §最终 QA）。

## Phase 2：v0.5-beta 密集音频（MERT-v1-95M）

- 模型：`m-a-p/MERT-v1-95M`，固定 revision `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`，
  trust_remote_code，取 encoder 第 9 层（forward hook；transformers 5.20.2 的 custom model
  不返回 hidden_states）；24kHz、context/hop=5s、75Hz/768d、0.5s bin、mean_std、l2；
- 版本：`m-a-p-MERT-v1-95M-12af15fe...__sr24000_ctx5_hop5_l9_poolmean_std_bin0.5_norml2`；
- **Pilot（20 部，PASS）**：20/20 成功（0 failed/skipped），音频 2582.753s；
  193,179 帧/5,177 bins；RTF **0.00947**（约 105.6× 实时）；全流程 36.682s（冷加载 9.85s）；
  显存峰值 **1,320.4MB**；产物 329,401,863B≈314.1MiB；漂移最大 0.01898s（≤1 bin）；
  `logs/audio_dense_pilot.json` 的 `summary.pass`=true；
- **全量（110 部，t1 最终 force 重跑）**：**110/110 成功、failed=0、skipped=0**；
  总音频 **32,619.14s**；**2,439,577 帧 / 65,291 bins**；处理墙钟 264.717s（RTF **0.00812**，
  约 123.2× 实时）；全流程 278.564s；显存峰值 1,320.4MB；产物 **4,158,970,616 B≈3.87GiB**；
- **对齐口径**：`summary.pass=true`（frame_time_mapping：解码音频与最后一帧时间差
  ≤1 bin，110/110 通过）；**strict_container_alignment=false（108/110）**：
  work 50 容器 994.0s/音频流短 1.950667s/末尾 3 个 0.5s bin mask=False；work 99 容器 228.483s/
  短 1.496333s/末尾 3 bin mask=False；这是源音轨比容器短，**不是时间戳漂移/编码错误，没有伪造值**；
  其余检查全部通过（无 NaN/Inf/空序列、bin_count_matches_duration=true）；
  这两部已从 beta 明确排除（`--exclude-work-ids 50 99`），不参与全曲/地图；
- 产物：`data/audio_dense/<audio_version>/<work_id>/audio.npz` + manifest；
  限制：MERT 第 9 层未调优；5s 非重叠窗口首尾有少量 bin_mask=False；冻结音频特征，
  不宣称语义理解；详见 `docs/reports/v05beta-audio-pilot.md`（已补全量/tail mask 节）。

## Phase 3：v0.5-beta 局部音画双塔（官方最终）

- 数据修复：`data/v05_tokens/v05_beta_v1/` 全量 token，**max_segments=None、
  110 部 / 4,130 段、0 截断**（142.3MiB / 149.2MB）；alpha 的 v05_alpha_v1 保持不动；
- 音频适配：可用 **107 部**（MERT audio_dense 排除 work 50/99 的 tail-mask 问题；
  work 122 因 no_anchor_with_negative 排除）；按 canonical work_id 划分 **74/21/12**
  （train/val/test，不伪造 70/20/20），test 12 works；候选负样本仅同作品时间偏移
  Δ={1,2,4,8,16} bins，近重复>0.95 跳过；不使用作品 ID/位置/外部元数据；
- 模型：视觉 CLIP + 音频 MERT → 双塔 MLP（hidden128/embed128，L2）；InfoNCE temperature=0.07，
  AdamW lr1e-3/batch256/epochs30/patience5，3 seeds（42/43/44×30 epochs，best epoch=1/2/6）；
  val 选模型、test 只评一次；
- **真实结果（3 seeds，test 4,822 anchors / 12 部作品）**：

  | seed | test Top-1 | Recall@5 | random | frozen projection | simple_sync | best epoch |
  | --- | --- | --- | --- | --- | --- | --- |
  | 42 | 0.296557 | 0.9290 | 0.270455 | 0.266072 | 0.292410 | 1 |
  | 43 | 0.302779 | 0.9330 | 0.270455 | 0.280589 | 0.292410 | 2 |
  | 44 | 0.305268 | 0.9316 | 0.270455 | 0.274575 | 0.292410 | 6 |
  | 均值 | **0.301535 ± 0.003663** | — | 0.270455 | 0.273745 | 0.292410 | — |

  **delta**：vs random **+0.03108**、vs frozen projection **+0.02779**、vs simple_sync **+0.00912**；
- 消融（seed 42）：normal 0.296557 / audio_zero 0.272916 / visual_zero 0.272916 /
  audio_replacement 0.268768 / time_shift 0.280796 —— 双模态置零/替换/时间偏移均掉点；
- 结论：**weak positive / not passed**。方向一致（高于随机/冻结/简单同步、双模态消融均掉点、
  无坍缩），但相对 simple_sync 仅 **+0.91pp**、bootstrap CI 宽、best epoch 仅 1–6，
  证据不足以宣布学会可靠局部音画对应；**不注册 DB/地图、不进入全曲编码、不改 alpha 负结论**；
  下一轮需扩大 test/作品覆盖、固定更强基线、DINOv2 消融与跨作品负例；
- 产物：`data/models/v05_beta_{42,43,44}/checkpoint.pt`、`logs/v05_beta_{42,43,44}_metrics.json`、
  `logs/v05_beta_summary.json`、候选清单 `logs/v05_beta_candidates_manifest.json`；
- t4 复核：v05_beta_v1 manifest 无截断；summary 与 3 seeds metrics 逐一核对一致
  （aggregate top1 mean 0.3015346）；test 仅在训练/选模型后评一次；未注册 DB/地图。

## 最终 QA（2026-10-26）

- `python qa.py --full` = **PASS 5/5**，总耗时 **241.68s**；
  ruff All checks passed；pytest **1019 passed, 1 skipped**（model/network 已启用），231.52s；
  docs 含 v0.2.x/v0.4/v0.4.3/v0.4.3.1/v0.4.4/v0.5-alpha/v0.4.5/v0.5-beta；
  CLI **21 个入口**（新增 explore_relations / audio_dense / train_v05beta）；config 校验通过；
- t4 真实产物/界面复核 **4 PASS / 0 FAIL**：
  1. v0.4.5 discovery JSON（303,793B）+ CSV（86,604B）：6 空间/110 部，模式 A 950、
     B 864、common 99（all3 70 / peav_clip 22 / peav_dino 7）、visual_vs_peav 19、
     单模型 peav/clip/dino 51/60/60；百分位公式逐项通过；
  2. 全量 MERT audio JSON：110/110 ok、32,619.14s、2,439,577 帧/65,291 bins、RTF .00812、
     pass=true（帧映射）；strict_container 108/110（work 50/99 tail mask=False，excluded）；
  3. v0.5-beta 3 seeds 结果/数据：summary Top-1 mean 0.301535、3 checkpoints/metrics、
     v05_beta_v1 无截断 4,130 段；summary 与 metrics 一致；
  4. 真实 AppTest：默认 map_v012、7 空间不变；“意外关系探索（v0.4.5）”模式 A/B/C
     作品对 50/50/219 行、当前作品对/CSV/只看互为近邻均可用；染色切换不改变候选；
     无异常；星图默认选择不被探索面板改变；
- **最终结论**：v0.4.5 可用作探索层（只读、不写 DB/地图）；v0.5-beta 音频全量 PASS
  （帧映射），beta 局部双塔为 **weak positive / not passed**（+0.91pp vs simple_sync、
  CI 宽、best epoch 1–6），不注册 DB/地图、不进入全曲编码、不改 alpha 负结论；
- 本报告只记录真实结果与限制，不复制大段实验日志；未 git commit。
