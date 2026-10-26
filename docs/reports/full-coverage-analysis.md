# 全片覆盖与多尺度音画特征表征分析（v0.4）

> 状态：已完成（2026-10-25，真实数据 110/110 canonical；v0.4.1 事件扫描 + v0.4.2 三空间全部落地）
> 设计契约：[docs/plans/v0.4-full-coverage-design.md](../plans/v0.4-full-coverage-design.md)；
> 数据/地图/报告为运行时产物（gitignored），本报告记录可复现命令、参数与结果摘要；不随仓库分发原始数据。
>
> **命名边界**：v0.4 的准确名称是**「全片覆盖与多尺度音画特征表征系统」**，
> 不是「逐帧音MAD语义理解系统」。本报告只描述“覆盖 + 多尺度特征记录 + 固定统计表征”的结果，
> 不声称逐帧视觉语义、可学习音画时序或自监督全曲表示能力。

## 0. 能力边界与五层目标

| 层级 | 目标 | v0.4 状态 | 说明 |
| --- | --- | --- | --- |
| 1 | 数据获取：完整时间覆盖 | ✅ 完成 | 8s 窗口覆盖整条时间轴、末尾锚定、权重归一；110/110 canonical（4130 窗口，legacy 536） |
| 2 | 高频变化记录 | ✅ 完成 | 音频 100Hz + 视频 30fps（96×54 灰度）的节奏/运动/突变统计；**不是视觉语义** |
| 3 | 逐帧视觉语义编码 | ❌ 未完成 | PE-AV 每 8s 仅 16 帧（≈2fps）稀疏联合语义，不能逐帧理解画面/动作/歌词/风格 |
| 4 | 可学习音画时序关系 | ❌ 未完成 | 本轮只有固定统计聚合与人工权重（Full Mean / Temporal Stats），无训练目标 |
| 5 | 自监督全曲表示 | ❌ 预留（v0.5） | 仅保留数据接口与 shim 字段，未训练、不做质量结论 |

**结论**：v0.4 完成的是第 1–2 层 + 固定统计表征（Full Mean / Temporal Stats），
第 3–5 层属于后续研究；**完整覆盖 ≠ 逐帧语义理解**。报告下文所有“覆盖/表征/邻域迁移”结论
均受此边界约束。

## 1. 目标与问题

把“均匀抽 ≤5×8s 片段再平均”的 legacy 表征，升级为覆盖整条时间轴的全片序列 + 多种聚合表征，
回答三个问题：

1. **覆盖完整**后，作品邻域结构会不会显著改变？哪些作品/作品对最不稳定？
2. Full Mean（全片平均）与 legacy（5 片段平均）的差异，是“更稳定”还是“只是换了一个空间”？
3. Temporal Stats（固定时间结构）是否带来与 Full Mean 不同的信息，是否值得保留多空间对照？

本报告不评价作品质量、作者风格或模型优劣，只描述**表征空间之间的邻域迁移**。

## 2. 数据与方法

| 项目 | legacy | full_mean_v1 | temporal_stats_v1 |
| --- | --- | --- | --- |
| sequence_version | `peav-small-16f__f16_clip8s_x5_48k` | `peav-small-16f__f16_full8s_ov0_48k` | 同 full_mean |
| 采样 | 均匀 ≤5×8s | 全覆盖 8s 窗口（overlap=0） | 同 full_mean |
| 聚合 | L2(mean(L2(z_i))) | 同公式、全片窗口（等权） | 语义/阶段/漂移/动态/自相似/事件分组拼接（6356 维） |
| canonical 作品数 | 110 | 110 | 110 |
| 地图版本 | map_v008（另有 v001–v007 历史/筛选） | map_v011 | map_v012 |
| 任意时刻 ≥1 窗口覆盖 | — | 110/110 | 110/110 |

- 真实数据：120 completed（33 本地 + 87 Bilibili）+ 2 failed（8.71h 直播回放、API -404 已删除），
  pending=0；10 条身份关联，duplicate 默认排除；canonical = 110；
- 336p 对照：10 部 480p-only 本地作品，另生成 `_h336` 序列版本（不覆盖正式版本）；
- 关键工具：`pipeline.py --sampling full_coverage`、`tools/build_representation.py`、
  `map.py build --representation ...`、`tools/compare_spaces.py`、`tools/events_probe.py`；
- 证据文件：
  - `logs/v04_data_analysis.json`：规模、窗口数、事件、336p、逆序、表征/地图统计；
  - 三组空间对比：`logs/space_compare_legacy_5clip_mean_vs_full_mean_v1__*.json/.csv`、
    `logs/space_compare_full_mean_v1_vs_temporal_stats_v1__*.json/.csv`、
    `logs/space_compare_legacy_5clip_mean_vs_temporal_stats_v1__*.json/.csv`；
  - 336p：`logs/seq_compare_f16_full8s_ov0_48k_f16_full8s_ov0_48k_h336.json`、`logs/seq_h336.csv`；
  - 独立复核：`logs/t4_verify_space_compare_*`（QA 用 compare_spaces 重新生成，数字与上表一致）。

## 3. 实验结果

### 3.1 覆盖率、编码与存储

| 指标 | legacy | full_mean / temporal_stats 序列 | 336p 对照序列 |
| --- | --- | --- | --- |
| 作品数 | 110 | 110 | 10 |
| 片段总数 | 536 | **4130** | 164（10 部） |
| 每部片段数（min/mean/median/max） | 2 / 4.87 / 5 / 5 | **3 / 37.55 / 24 / 214** | 4 / 16.4 / - / 32 |
| 无间隙 / 末尾锚定 / 任意时刻覆盖 | — | 110/110 全部通过 | 10/10 通过 |
| 窗口权重和 | — | 0.99999992 – 1.00000003 | 同左范围 |
| 视频总时长 | 32619.8s（9.06h） | 32619.8s（同一批作品） | 1269.0s |
| 事件文件（scan/features/manifest） | 无 | 110/110，`events_v1`，131MB | 10/10 |
| 每部耗时（work_timings 均值） | download 7.20 / extract 3.18 / preprocess 1.16 / forward 3.20 / total 16.57s | download 5.54 / probe 0.045 / extract 24.96 / preprocess 8.31 / forward 22.45 / events 7.53 / save 0.028 / **total 70.35s** | total 31.85s（extract 11.09、forward 10.08、events 5.94） |
| 存储占用 | 8.4MB（legacy 向量） | 序列层 174.7→约 187MB（其中事件 131MB）；representations 9.7MB | h336 序列 6.8MB |

全片批次 run_id=8：99 部 / 7277s（含 1 部失败）；work 66 通过 Bilibili playurl 回退修复后补跑
（run_id=10），最终 110/110 assembled/completed。

**336p 对照实验（10 部，480p-only 本地作品）**：

| 指标 | mean | min | max |
| --- | --- | --- | --- |
| work_joint cos（full vs h336） | 0.998590 | 0.997156 | 0.999824 |
| work_audio cos | 1.000000 | 1.0 | 1.0 |
| work_video cos | 0.986168 | 0.977239 | 0.999306 |
| 对齐片段 clip_joint cos（164 对） | 0.997427 | 0.987465 | 0.999475 |

耗时：extract 25.08→11.09s（2.26×）、forward 9.85→10.08s、events 5.45→5.94s、
total 51.08→31.85s（1.60×）。结论：336p 可加速但会改变视频分支向量（min 0.977），
**仅作候选，不转正**；需要新 `sequence_version` + 单独地图，不能与正式 full 序列混用。

**片段顺序敏感性（20 部真实作品反转顺序）**：Full Mean 逆序 cos_min=0.99999994（距离≤6e-8）；
Temporal Stats 逆序 cos_min=0.67095 / mean 0.67777（1-cos 0.287–0.329）。
说明 Full Mean 对顺序不敏感（等权平均的预期），Temporal Stats 确实编码了时间结构。

### 3.2 legacy × full_mean 邻域迁移（110 部 canonical）

| 指标 | 数值 |
| --- | --- |
| 共同 canonical 作品 | 110（无缺失） |
| 平均 / 中位 Top-5 Jaccard | **0.4068 / 0.4286** |
| 平均重合邻居数 | 2.72 / 5 |
| 完全稳定作品数 / 比例 | 8 / 110（7.27%） |
| Top1 一致率 | 38.18% |
| 稳定对比例（无序对判定一致率） | 92.99% |
| 有向边保留率 / 交并比 | **54.36% / 0.3733** |
| 互为邻居对交并比 | 0.2952 |
| 作品对余弦 Pearson / Spearman / mean\|Δ\| / max\|Δ\| | **0.9569 / 0.9480 / 0.0265 / 0.1356** |
| hubness 秩相关 | 0.8616 |

被替换案例（来自 `replaced_works`；标题缺失时以 work_id/BV 记录）：

| work_id | 旧 Top5 → 新 Top5 | 保留 | 典型变化 |
| --- | --- | --- | --- |
| 8 | [104, 111, 114, 17, 12] → [35, 115, 17, 121, 82] | 1/5 | old Top1=104 → new Top1=35；邻居大幅换血 |
| 13 | [104, 48, 35, 114, 71] → [115, 82, 83, 35, 34] | 1/5 | old Top1=104 → new Top1=115 |
| 1 | [93, 110, 72, 88, 30] → [115, 30, 110, 48, 82] | 3/5 | 部分保留，新增 hub 115/82 |
| 104（hub） | [121, 120, 53, 103, 48] → [120, 93, 50, 82, 117] | 1/5 | 被列次数 40→33 |
| 93 | [120, 117, 50, 79, 104] → [120, 104, 50, 36, 82] | 3/5 | 被列次数 34→27 |

hubness 变化：旧 Top hub 104(40)、93(34)、50/120(25)、39(23)；新 Top hub 115(37)、82(33)、
104(33)、121(29)、36(27)。增加最多：115 +27（10→37）、82 +25、36 +15、56 +13、121 +13；
减少最多：48 -13、79 -7、93 -7、104 -7、12 -6。

### 3.3 full_mean × temporal_stats 邻域迁移（110 部）

| 指标 | 数值 |
| --- | --- |
| 共同 canonical 作品 | 110 |
| 平均 / 中位 Top-5 Jaccard | **0.2049 / 0.1111** |
| 平均重合邻居数 | 1.55 / 5 |
| 完全稳定作品数 / 比例 | 0 / 110（0%） |
| Top1 一致率 | 17.27% |
| 稳定对比例 | 89.42% |
| 有向边保留率 / 交并比 | **31.09% / 0.1841** |
| 互为邻居对交并比 | 0.1449 |
| 余弦 Pearson / Spearman / mean\|Δ\| / max\|Δ\| | **0.8294 / 0.7979 / 0.1210 / 0.2753** |
| hubness 秩相关 | 0.7415 |

被替换案例：work 1 Jaccard=0（[115,30,110,48,82] → [104,72,74,66,50]）；work 8 Jaccard=0；
work 104 Jaccard=0；work 13 Jaccard=1/9；work 93 Jaccard=1/9。
hubness：115(37)→36(36)、82(33)→21、104(33)→28；36 由 27 升至 36；
增加最多 57 +15、41 +13、36 +9、79 +9、99 +8；减少最多 115 -23、93 -20、56 -14、39 -13、121 -13。

### 3.4 legacy × temporal_stats（对照）

平均 Jaccard=0.1734（中位 0.1111），完全稳定 1/110（0.91%），Top1 一致 19.09%，
稳定对 88.89%，有向边保留 26.55%，余弦矩阵 Pearson=0.7889 / Spearman=0.7475。
说明 Temporal Stats 与 legacy/Full Mean 都构成明显不同的空间，而不是 legacy 的近似副本。

### 3.5 解读

- 全片覆盖显著改变邻域结构：legacy→full_mean 只有 7.27% 作品 Top5 完全不变、平均 Jaccard 0.41；
  这不是“更稳定”，而是**换了一个更细粒度的表示空间**；
- 作品对余弦矩阵仍高度相关（Pearson 0.957），说明整体语义结构大体保留，但局部 Top-k 邻域对
  采样覆盖高度敏感——评估近邻结论时必须说明表征空间；
- Temporal Stats 与 Full Mean 差异更大（Jaccard 0.20、Top1 17%、边保留 31%），且顺序敏感性验证
  证明它编码了时间结构；但本轮不宣称它对下游任务更好，只说明“不可互相替代”；
- hubness 变化显示：少数作品（如 115、36、82）在新空间成为高频邻居，旧 hub（104/93）被列次数下降；
  这是集合相对量，不能解读为作品质量变化；
- 336p 只作为提速候选：视频分支 cos 最低 0.977，不能与正式 full 序列混用。

## 4. 局限与注意事项

- 全片模式对超长视频（>1h）默认跳过，需要 `--allow-long`；本轮 8.71h 直播回放仍记 failed；
- PE-AV 仍是 16 帧稀疏采样；全片窗口覆盖整条时间轴，但 8s 窗口内部细节未被完整建模；
- Full Mean 只是“完整覆盖的单一变量基线”，不代表优于 legacy；Temporal Stats 的组权重是设计选择，
  未做下游任务调参；
- 地图坐标跨表征/跨序列不可比较绝对位置；分位、hubness 都是集合相对量；
- 真实数据只跑一次，不宣称统计显著性；336p 对照仅 10 部 480p-only 本地作品；
- 部分 Bilibili 作品标题未抓取（报告以 work_id 为准），不影响向量与邻域计算；
- 不把邻域迁移解读为作品质量、作者水平或模型优劣。

## 5. 复现与证据

```powershell
# 1) 全片序列（真实数据示例；需要模型/GPU 与 Bilibili 网络）
python pipeline.py run-pending --sampling full_coverage --allow-long

# 2) 从已保存序列构建表征（不下载、不解码、不重编码 PE-AV）
python tools/build_representation.py full_mean --sequence peav-small-16f__f16_full8s_ov0_48k
python tools/build_representation.py temporal_stats --sequence peav-small-16f__f16_full8s_ov0_48k
python tools/build_representation.py list

# 3) 表征地图（不覆盖旧版本）
python map.py build --representation full_mean_v1 --sequence-version peav-small-16f__f16_full8s_ov0_48k
python map.py build --representation temporal_stats_v1 --sequence-version peav-small-16f__f16_full8s_ov0_48k
python map.py list

# 4) 邻域迁移对比（三组）
python tools/compare_spaces.py --a legacy_5clip_mean --b full_mean_v1 --k 5
python tools/compare_spaces.py --a full_mean_v1 --b temporal_stats_v1 --k 5
python tools/compare_spaces.py --a legacy_5clip_mean --b temporal_stats_v1 --k 5
```

- 真实数据验收（t4）：覆盖率/窗口/权重/事件三件套 110/110 通过；三组 compare 数字与
  `logs/space_compare_*` 一致；真实 AppTest 默认 map_v012，三空间切换
  Full Mean→map_v011 / Legacy→map_v008 / Temporal→map_v012，时间轴与片段面板正常（8 PASS / 0 FAIL）；
- 结论数字以本报告与 `logs/v04_data_analysis.json`、三组 compare JSON、336p JSON 为准；
- 数据库/地图/向量为运行产物，不随仓库提交；复现时按 README 准备环境与配置。
