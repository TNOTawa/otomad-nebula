# v0.5-beta 局部模型诊断：修正消融与完整 test

> 状态：**已完成（负结果）**。本轮只做有上限的诊断，不扩大模型、不训练全曲编码器。
> 目标：区分“模型真的使用音频内容/同步”与“模型利用位置、mask、视觉或子集偏差”。
> 产物不注册 DB/地图；12 部 held-out 作品 123–134 本轮严格保留。
> 日期：2026-10-26。

## 1. 上一轮消融的问题

- 上一报告中的 `time_shift` 在 `v05_local._ablate_batch` 中只做
  `audio_time = clip(audio_time + 1.0, -1, 1)`；音频 **内容完全不变**，移动的是
  位置编码输入。它不能证明模型学到了真实音画同步。
- `audio_zero` 虽然把音频内容清零，但保留了音频时间、mask 与模型位置编码；
  “置零后准确率更高”是警讯，但不能直接定位捷径来源。
- 因此本轮先修正消融，再在完整 test 上重评；不在旧子集数字上继续加模型。

## 2. 子集偏差（数据抽取成本造成的选择偏差）

`data/v05_protocol/v2_subset` 是每部作品按 `anchors.npz` 顺序取前 N 条
（train 6/val 2/test 10），不是随机抽样。与完整 test 对比：

| 指标（test） | 子集 112 anchors | 完整 test 3,681 anchors |
| --- | --- | --- |
| anchor 位置（作品时长比例）中位数 | 2.3% | 50.0% |
| anchor 在前 10% 的比例 | 88.4% | 10.6% |
| 左边界截断比例（<1.5s） | 10.7% | 0.33% |
| 右边界截断比例（距结尾<1.5s） | 0% | 0.24% |
| 视觉 token 均值 / 最大 | 23.43 / 24 | 23.96 / 24 |
| 音频有效 token 总均值 / 最大 | 439.0 / 450 | 449.3 / 450 |
| 正例位置分布（0..5） | 26/17/18/18/17/16 | 635/619/600/616/617/594 |

- 候选正例位置是平衡的，问题不在 label 位置；主要是 **anchor 集中在作品开头**，
  以及子集只有 112/3,681 个 anchor。子集结果不能外推。
- 完整统计：`logs/v05_local_subset_bias.json`。

## 3. 修正后的消融与预先门槛

修正后的消融（`nebula/v05_local.py`）：

| 类别 | 模式 | 语义 |
| --- | --- | --- |
| 内容破坏 | `audio_zero` | 音频内容清零，保留 time/mask |
| 内容破坏 | `audio_mask_only` | 内容与 time 清零，只保留 mask |
| 内容破坏 | `audio_mean` | 每个候选替换为自身有效 token 均值 |
| 内容破坏 | `audio_global_mean` | 全 batch 单一音频均值，候选间不可区分 |
| 内容破坏 | `audio_permute_candidates` | 同一 anchor 内随机置换候选音频内容；目标 time/mask/标签不变，正例内容必被换走 |
| 真实错位 | `audio_roll_0p5s/1s/2s` | 内容沿 token 轴平移真实秒数，time/mask 不变 |
| 真实错位 | `audio_center_shift` | 数据集层面以 anchor±delta 重新取音窗；视觉 anchor 不变（±0.5/±1/±2s） |
| 位置/内容分离 | `audio_time_zero` | 保留音频内容，仅去掉音频相对时间 |
| 视觉对照 | `visual_zero` | 视觉内容清零 |
| 历史 | `time_shift` | 仅改 audio_time 位置编码，不应解读为真实错位 |

**预先固定的通过条件**（在完整 test 前写入 `logs/v05_local_diagnostics_criteria.json`）：

1. 内容破坏型消融中最高保留准确率仍比 normal 低 **≥5pp**（audio_zero / global_mean /
   permute_candidates 三者都必须被内容破坏显著影响）；
2. 真实错位（roll 1/2s 与数据集 ±1/2s 平均）比 normal 低 **≥5pp**；
3. normal 与 macro top1 均高于 simple_sync 与旧 beta 均值各 **≥3pp**；
4. 12 部作品里至少 **7 部** normal 高于 simple_sync 和旧 beta；
5. 3 seeds 中至少 **2 个**同时满足，才算该 arch 通过；任一 arch 通过则本轮通过。

若未通过：暂停当前局部配对路线，不再加层/加参数，不再处理剩余 1,846 部；保留全部
特征数据，把项目重心放回已有作品探索工具。

## 4. 完整 test 结果

协议：完整 test **3,681 anchors / 12 部作品**（work 11/19/30/37/43/74/81/82/84/93/111/116），
K=6 主配对，K=32 retrieval；所有 checkpoints 使用同一份 test anchors，未使用 held-out 123–134。
运行约 52 分钟（6 checkpoints × 完整 test + 11 消融 + 6 个音频错位）。每 seed 明细：

| arch | seed | normal Top-1 | macro Top-1 | retrieval K=32 Top-1 |
| --- | --- | --- | --- | --- |
| temporal | 42 | 15.32% | 18.92% | 3.26% |
| temporal | 43 | 14.53% | 19.16% | 3.50% |
| temporal | 44 | 13.01% | 18.11% | 3.15% |
| temporal | mean±std | **14.29 ± 0.96%** | 18.73 ± 0.45% | 3.31 ± 0.15% |
| cross | 42 | 13.07% | 19.55% | 3.18% |
| cross | 43 | 13.45% | 17.96% | 3.12% |
| cross | 44 | 14.18% | 19.47% | 3.61% |
| cross | mean±std | **13.57 ± 0.46%** | 18.99 ± 0.73% | 3.31 ± 0.22% |

参考基线（完整 test，旧模型不重训）：chance **16.67%**；simple_sync **17.55%**
（macro 16.03%）；v05_beta_42/43/44 **17.96/17.90/17.98%**（均值 17.95%，macro 23.33%）；
retrieval chance 3.125%、simple_sync 3.45%、旧 beta 4.56–4.89%。

**两种局部模型在完整 test 上都低于 chance、simple_sync 与旧 beta**；retrieval 也回到
chance 附近。子集上的 26.79%/25.00% 没有外推到完整 test；结合第 2 节的开始位置偏差，
这正是子集选择造成的乐观偏差。

### 4.1 修正消融（3 seeds 均值）

| 模式 | temporal Top-1 | Δ vs normal | cross Top-1 | Δ vs normal |
| --- | --- | --- | --- | --- |
| normal | 14.29% | — | 13.57% | — |
| audio_zero（内容清零，留 time/mask） | **17.39%** | **+3.10pp** | **17.50%** | **+3.93pp** |
| audio_mask_only（只留 mask） | 16.98% | +2.69pp | 17.11% | +3.54pp |
| audio_mean（候选内均值） | 14.22% | -0.07pp | 13.37% | -0.20pp |
| audio_global_mean（全 batch 均值） | **17.38%** | **+3.09pp** | **17.47%** | **+3.90pp** |
| audio_permute_candidates（候选内容互换，time/mask 不变） | **17.21%** | **+2.92pp** | **17.89%** | **+4.32pp** |
| audio_roll_0.5s（内容真实平移） | 14.58% | +0.29pp | 14.04% | +0.47pp |
| audio_roll_1s | 14.59% | +0.30pp | 14.53% | +0.96pp |
| audio_roll_2s | 15.78% | +1.49pp | 15.55% | +1.98pp |
| audio_time_zero（去音频时间） | 14.15% | -0.14pp | 14.49% | +0.92pp |
| visual_zero | 13.51% | -0.78pp | 13.91% | +0.34pp |
| time_shift（仅位置编码；历史消融） | 13.76% | -0.53pp | 13.79% | +0.22pp |
| audio_center_shift ±1s（数据集层面取音窗错位） | 13.56% | -0.73pp | 12.90% | -0.67pp |
| audio_center_shift ±2s | 13.53% | -0.76pp | 12.62% | -0.95pp |

- **内容破坏没有让模型变差，反而稳定提高约 3–4pp**：audio_zero、audio_global_mean、
  audio_permute_candidates 三种都把准确率抬到 17–18%，高于 normal 与 chance。这说明正常
  预测受到了音频内容/候选内容的负面影响，模型没有利用“正确的音画内容对应”。
- **真实内容错位（roll 1s/2s、数据集 ±1/±2s）没有显著掉点**；roll 甚至略高，dataset
  center shift 只掉 0.7–1.0pp。无法证明模型依赖音画同步。
- audio_mean（候选内平均）与 visual_zero 略低于 normal；但综合内容置换/清零结果，
  仍不能解释为“音频内容被正确使用”。
- `time_shift` 仅改位置编码，结果与 normal 接近；再次证明上一轮把它解释为“错位掉点”
  是不成立的。

### 4.2 预先门槛逐项判定

| 门槛 | temporal | cross | 结果 |
| --- | --- | --- | --- |
| 内容破坏消融最高保留率比 normal 低 ≥5pp | 内容保留 17.39%（比 normal 高 3.10pp） | 17.89%（高 4.32pp） | **失败** |
| 真实错位平均比 normal 低 ≥5pp | roll 平均保留 15.18%（比 normal 高） | roll 平均保留 15.04%（比 normal 高） | **失败** |
| normal 比 simple_sync 高 ≥3pp | 14.29 vs 17.55%（低 3.26pp） | 13.57 vs 17.55%（低 3.98pp） | **失败** |
| normal 比旧 beta 高 ≥3pp | 14.29 vs 17.95%（低 3.66pp） | 13.57 vs 17.95%（低 4.38pp） | **失败** |
| 12 部中至少 7 部高于 simple_sync | 3/4/3（3 seeds） | 3/3/3 | **失败** |
| 12 部中至少 7 部高于旧 beta | 1/2/0 | 1/0/2 | **失败** |
| 3 seeds 中 ≥2 个同时满足 | 0/3 | 0/3 | **失败** |

- macro Top-1 表面上高于 simple_sync macro（18.7–19.0% vs 16.03%），但主要来自
  work 30（16 anchors）与 work 111（2 anchors）的小样本；对旧 beta macro（23.33%）仍低
  约 3.8–5.4pp。按作品逐项比较，模型只在 3–4/12 部作品上高于 simple_sync，不满足“作品级可信”。
- 结论：**0/6 seeds、0/2 arch 通过；本轮诊断未通过。**

## 5. 结论与止损

1. 修正消融后，上一轮“time_shift 证明同步”的解释被否定；真实内容置换/平移没有让模型
   变差，内容清零反而提高准确率，**没有可信的音频内容/同步贡献证据**。
2. 完整 test 上局部模型低于 chance、simple_sync 与旧 beta；子集上的正向结果由“前若干
   anchors”选择偏差造成，**优势消失**。
3. 按本轮预先固定门槛：**未通过**。因此：
   - **暂停当前局部配对训练路线**，不加层/加参数，不训练全曲统一编码器；
   - 不处理剩余 1,846 部作品，不新增基础编码器，不建正式星云/地图；
   - 保留全部 CLIP/DINOv2/MERT/PE-AV 特征、旧 checkpoint 和诊断产物；旧模型/旧地图不动；
   - 项目重心回到已可用的作品探索工具（高维近邻、跨模型共识、意外关系），数据积累
     等有更合适的预训练基础/更多独立作品后再重启统一编码器研究。
4. 12 部 held-out（123–134）本轮未参与训练、调参、诊断或表征/地图；继续保持独立，
   留到未来方案冻结后的最终测试。

## 6. 证据与复现

- 完整诊断：`logs/v05_local_diagnostics_full.json`（逐 checkpoint/消融/per-work/CI）
- 汇总表：`logs/v05_local_diagnostics_summary.csv`
- 门槛：`logs/v05_local_diagnostics_criteria.json`
- 子集偏差：`logs/v05_local_subset_bias.json`
- 代码：`nebula/v05_local.py`、`tools/train_v05_local.py`
- 复现命令：

```powershell
python tools/train_v05_local.py --eval-checkpoints `
  data/models/v05_local_subset/v05_local_temporal/seed_42/checkpoint.pt `
  data/models/v05_local_subset/v05_local_temporal/seed_43/checkpoint.pt `
  data/models/v05_local_subset/v05_local_temporal/seed_44/checkpoint.pt `
  data/models/v05_local_subset/v05_local_cross/seed_42/checkpoint.pt `
  data/models/v05_local_subset/v05_local_cross/seed_43/checkpoint.pt `
  data/models/v05_local_subset/v05_local_cross/seed_44/checkpoint.pt `
  --batch-size 24 --bootstrap 200 --cache-size 16 `
  --eval-shifts -2 -1 -0.5 0.5 1 2 `
  --eval-out logs/v05_local_diagnostics_full.json
```

- held-out 123–134：本轮不参与诊断/调参/表征/地图；等方案冻结后只做一次最终测试。
