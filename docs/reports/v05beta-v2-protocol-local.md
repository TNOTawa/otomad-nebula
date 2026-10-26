# v0.5-beta v2：配对协议修正、局部时序消融与作品库扩展

> 状态：协议修正完成；旧模型在新协议下重评；局部时序实验为**compute-limited 固定子集**，
> 结果**未通过停止条件**：模型在子集上高于简单同步/随机，但 audio 消融反而更好，
> 且仅 112 个 test anchors/12 部作品，不足以宣布学到稳健局部音画对应。
> 不注册 DB/map，不训练全曲编码器。日期：2026-10-26。

## 1. 协议 v2（修正 beta 的测试协议）

- 固定 **K=6** 主配对（1 正 + 5 负），负样本从同作品**双向**偏移 ±1,±2,±4,±8,±16 bins
  （0.5s/bin）确定性采样；不再使用可变 K；无足够非近重复负样本时跳过并记录，不伪造。
- 近重复（视觉或音频余弦 >0.95）候选**不删除**，全部进入独立 periodic 集；
  主协议 K=6 排除它们；periodic anchor/candidate 单独评估。
- 固定 split：canonical work_id **74/21/12**（107 部，与 beta 可比，保持 test 独立）；
  main K=6 + test K=32 retrieval（1 正+31 负）；所有模型使用同一 test anchors。
- 旧 beta 的 Recall@5=93% 已明确标注：K≤6 时 Recall@5 期望下限为 5/6=83.3%，
  **该指标无信息量，不再作为证据**。

真实规模：107 部共 **62,613 anchors**；主 K=6 kept **44,513（71.1%）**，
periodic anchors 18,092，periodic candidates 207,260；test 主集 **3,681 anchors/12 部**；
single-model insufficient 仅 8 个。

## 2. 旧 checkpoint 在新协议下的真实能力

test main K=6（chance 16.67%；per-work bootstrap CI 宽）：

| model | main K=6 Top1 | retrieval K=32 Top1 / R@5 / R@10 |
| --- | --- | --- |
| random | 16.67% | 3.125 / 15.625 / 31.25% |
| frozen projection | 16.27% | 2.20 / 14.32 / 30.13% |
| simple_sync | 17.55% | 3.45 / 16.98 / 33.22% |
| v05_beta_42 | 17.96% | 4.56 / 20.78 / 39.91% |
| v05_beta_43 | 17.90% | 4.73 / 20.32 / 38.41% |
| v05_beta_44 | 17.98% | 4.89 / 20.35 / 38.60% |

- 主 K=6 仅比 simple_sync 高约 **0.4–0.5pp**，CI 宽，不能宣称显著优于简单基线。
- retrieval K=32 优于 chance/简单基线，但旧模型训练协议不同，只作参考。
- periodic candidate 约 50%（chance），periodic anchor 13.16–14.27%（chance 14.70%），
  旧模型无法区分近重复候选。

## 3. 局部帧级时序模型（temporal / cross-attn）

实现：直接读已有 CLIP 8fps 帧向量与 MERT 75Hz 帧特征；3s 上下文、音频下采样 25Hz、
视觉 8fps、相对时间正弦编码与边界 mask；temporal=各自 self-attn + pooled cosine；
cross=各自 self-attn + 双向 cross-attention；仅 InfoNCE，128d/4heads，无额外头。

**compute-limited 固定子集**：按作品取前若干 anchor（train 6/作品=441，val 2=42，
test 10=112，retrieval 10=112）；test 12 部/112 anchors；所有基线在同一子集重算。
该子集是为了在现有 CPU 数据抽取成本下完成最小消融，不是完整 test 协议。

| model | main K=6 Top1（3 seeds mean±std） | retrieval K=32 Top1 |
| --- | --- | --- |
| random / chance | 16.67% | 3.125% |
| frozen projection | 18.75% | — |
| simple_sync | 16.07% | — |
| old beta_42/43/44 | 21.43 / 33.93 / 23.21% | — |
| **temporal (ours)** | **26.79 ± 0.73%** | **20.54 ± 3.34%** |
| **cross-attn (ours)** | **25.00 ± 0.73%** | **25.30 ± 3.67%** |

- temporal 在子集上高于 simple_sync（+10.7pp）与 frozen（+8.0pp），但 old beta 同子集均值
  约 26.2%（old_43 33.9%），**没有清晰的架构增益**；cross 主 K=6 更低。
- **消融红旗**：temporal 3 seeds 的 audio_zero Top1 分别 28.6/29.5/30.4%，
  **高于 normal 27.7/26.8/25.9%**；visual_zero 在部分 seed 也更高；time_shift 降到
  8.0–12.5%。说明模型主要利用时间位置/视觉捷径，而不是稳定的音频-视觉对应。
- 按用户停止条件：**未通过**；不训练全曲 Transformer/不注册表征。
- 此外，完整 test（3,681 anchors）上的新模型未跑（数据抽取成本）；上面的子集结果不能
  替代完整协议结论。

## 4. 作品库扩展（独立 held-out）

- 解析 otopick 2022/2023/2024 页面：467/680/712 行，去重 **1,858 个新 BV**；
  仅小批量导入 **12 部**（work_id 123–134，每年 4 部）作为 held-out；其余 1,846 个 BV
  已解析为本地 CSV，未入库（避免 pending 污染现有队列）。
- 12/12 完成 PE-AV+事件、CLIP/DINOv2 8fps 帧+窗口层、MERT 密集音频；0 失败；
  新增磁盘约 709.76MB（源缓存新增 244.79MB，总 2.455GB，未删除/未重复下载）。
- manifest：logs/v05_expansion_manifest.json；明确 held_out=true、
  exclude_from_training/tuning/representation/map；已核对与 v05_alpha/v05_beta token
  和 protocol split 交集为空，DB representation_vectors/map_coords 均为 0。
- 发现的视觉流水线 slug bug 已修复：compose_visual_version 现在统一返回 visual_version_slug；
  新增回归测试，避免创建含 /、@ 的重复版本。

## 5. 结论与停止条件

1. 旧 beta 的“93% Recall@5”确实不可用；新协议下旧模型仅 +0.4–0.5pp vs simple_sync。
2. 双向候选/固定 K/周期集/固定 test 已实现；旧模型在 periodic 集上无区分能力。
3. 最小局部时序模型在 compute-limited 子集上有高于简单基线的数值，但 audio 消融反而更好，
   且缺少完整 test 验证；按停止条件**不扩大模型/不训练全曲编码器**。
4. 数据扩展的 12 部 held-out 已完成特征，可留待下一轮严格测试；本轮不参与训练。

## 6. 复现

```powershell
# 协议 v2 + 旧 checkpoint 重评
python tools/build_v05_protocol.py --quiet
# 固定子集局部消融
python tools/train_v05_local.py --protocol-dir data/v05_protocol/v2_subset --arch both --seeds 42 43 44 --epochs 5 --batch-size 8 --device cuda --cache-size 4 --num-workers 0 --out-dir data/models/v05_local_subset
# held-out 特征（本轮已完成，manifest 见 logs/v05_expansion_manifest.json）
```

## 7. 证据

- 协议：data/v05_protocol/v2/manifest.json、anchors.npz、logs/v05_protocol_v2_metrics.json/.csv
- 子集：data/v05_protocol/v2_subset/；基线 logs/v05_protocol_v2_subset_baselines.json
- 局部模型：data/models/v05_local_temporal/、v05_local_cross/（subset 产物）；logs/v05_local_subset_train3.log
- 扩展：logs/v05_expansion_manifest.json、input/lists/otopick_*.csv（本地）
- 旧 recall 修正说明在 logs/v05_protocol_v2_metrics.json notes
