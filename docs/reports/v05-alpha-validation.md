# v0.5-alpha 验证报告：分层音画时序模型（首次可行性）

> 状态：**实现完成，可行性验证未通过**。本报告如实记录负结果；未注册正式表征/地图，
> 不宣称学到可靠音MAD时序结构或风格。设计见 `docs/plans/v0.5-alpha-temporal-model.md`。
> 日期：2026-10-26；数据：110 部 canonical 作品；GPU：RTX 3060。

## 1. 实现内容

| 层 | 文件 | 说明 |
| --- | --- | --- |
| 数据管道 | `nebula/v05_data.py`、`tools/build_v05_tokens.py` | 从 PE-AV 窗口、CLIP/DINOv2 帧、100Hz/30fps 事件构建局部 token；0.5s×16 bin/段；产物 `data/v05_tokens/v05_alpha_v1/` |
| 模型 | `nebula/v05_model.py` | 局部多模态 Transformer（256d/4heads/2层）+ 全曲 2 层 Transformer；3 头（masked segment、AV mismatch、order recovery）；4.31M 参数 |
| 训练/验证 | `nebula/v05_train.py`、`tools/train_v05.py` | 确定性 90/20 split、变长 padding+mask、三目标训练、验证指标、checkpoint/metrics/研究向量落盘 |
| 测试 | `tests/test_v05_data.py`/`_model.py`/`_train.py` | 15+19+15 条 CPU 离线测试，均通过 |

数据规模：110 部作品、4,130 段、141.1MB（6 部 >128 段截断：work 36/38/42/56/57/117）。
训练超参：hidden=256、4 heads、local/temporal 各 2 层、dropout=0.1、batch=4、AdamW lr=1e-3、
epochs=40、patience=12、mask_ratio=0.2、seed=42；训练/验证按 work_id hash 固定（90/20 部）。

## 2. 两次迭代与验证结果

### 2.1 v05_alpha_v1（初始实现）

| 指标 | 值 | 判定 |
| --- | --- | --- |
| val mask cosine | 0.8149 | 低于常量 Full Mean 基线 0.8249 ❌ |
| AV mismatch accuracy / AUC | 0.80 / 0.729 | 高于 chance，但负例来自同 batch 音频互换，含作品身份泄漏风险 |
| order recovery accuracy | 0.05 | 远低于 chance=0.167 ❌ |
| 时间敏感性（mask cosine reverse/shuffle drop） | -0.0053 / -0.0018 | 反向变好，无敏感性 ❌ |
| work vector pairwise cosine mean / 有效秩 | 0.99901 / **1.03** | 严重坍缩 ❌ |
| Top-5 Jaccard vs Full Mean | 0.045 | 几乎无关（坍缩后的伪差异）❌ |

### 2.2 v05_alpha_v2（修复坍缩与 mask 基线后的重训）

改动：work_vector 加 VICReg variance/covariance 正则（0.5/0.05 权重）；mask head 改为
“数据集 PE-AV 均值 + 零初始化残差”，初始即等于常量基线；训练 37 epoch（best epoch 25）。

| 指标 | 值 | 判定 |
| --- | --- | --- |
| val mask cosine | 0.8131 | 基线 0.8249，仍低 1.2pp ❌ |
| mask MSE | 0.000365 | 基线 0.000342，略差 ❌ |
| AV accuracy / AUC / 负例准确率 | 0.40 / 0.635 / 0.75 | accuracy 低于 0.5，不稳定 ❌ |
| order accuracy（20 val 样本） | 0.10 | chance 0.167 ❌ |
| 时间敏感性 mask reverse/shuffle drop | -0.0048 / -0.0015 | 无敏感性 ❌ |
| work vector pairwise cosine mean / 有效秩 | 0.9806 / **4.66** | 秩从 1.03 改善，仍远低于 >8 目标 ❌ |
| Top-5 Jaccard vs Full Mean | 0.047 | 远低于 0.3–0.8 目标 ❌ |
| duration R² / author balanced acc | 0.769 / 0.027 | duration 泄漏明显；author 近随机 |
| 段序敏感性（工作向量 cosine 正常 vs 反转/打乱） | 0.999963 / 0.999929 | 近似完全顺序不变 ❌ |

**结论：未通过 v0.5-alpha 可行性标准。** 唯一明确的是工程链路完整（数据→模型→训练→验证→产物），
并在第二次迭代把 work vector 从严重坍缩（rank 1.03）改善到 rank 4.66；但 temporal 表征没有
学到可用的顺序/跨模态结构，masked 重建不如常量均值基线。

## 3. 失败原因分析（按证据）

1. **PE-AV 8s 目标几乎被全局风格主导**：110 部作品的 PE-AV joint 段向量均值 cosine 基线已达
   0.825；相邻窗口重叠且 stride≈7.8s，8s 内语义高度冗余，上下文对“被 mask 段”的可预测增量很小，
   模型只能在均值附近拟合噪声。v2 的 mask 残差头零初始化后仍没学到正增益。
2. **work vector 对段序几乎不变**：normal/reversed/shuffled 的 work cosine 均 >0.9999，说明
   temporal encoder + masked mean 没有被目标函数迫使使用时间顺序；VICReg 只改善了分散度，没有
   创造时序信号。order head 使用绝对段位置编码，3 段排列任务在 20 个 val 样本上低于 chance，
   不能作为可靠能力证据。
3. **局部音画没有真正的配对目标**：当前 AV mismatch 是整段/整作品的二分类，负例来自 batch
   内音频互换，模型可借作品身份或全局统计走捷径；没有 “0.5s 级视觉 token ↔ 音频 token” 的
   InfoNCE/对齐损失，所以“先局部对齐再全曲聚合”只停留在结构上。
4. **数据量/独立性不足**：110 部作品、帧级高度相关，26 万帧不是 26 万独立样本；4.3M 参数模型在
   89 部训练作品上容易记忆 duration（R²=0.769）而不是学可迁移时序结构。
5. **音频侧信息不足**：100Hz 事件只有能量/通量等声学统计，没有 0.5s 级音频语义向量；这与用户
   “若只学到节奏峰值，应优先补密集音频语义”的判断一致。

## 4. 未做/边界

- 未把 `transformer_alpha_v1` 注册到 `representations` 表、未建星云地图；产物只落盘研究目录。
- 未做全量 30fps 训练、未做多 seed/超参搜索、未做下游检索任务增益评估。
- v05_alpha_v1/v2 的 checkpoint、metrics、vectors 均保留（不覆盖）。
- 未重跑视觉编码；所有输入均为已有 PE-AV/CLIP/DINOv2 冻结向量。

## 5. 下一步建议（优先顺序）

1. **补密集音频语义**：在 0.5s 级加入音频编码器语义向量（而非仅事件统计），再验证局部 AV 对齐；
   这是当前最大信息缺口。
2. **重写自监督目标**：局部 visual↔audio InfoNCE（时间平移负例）+ 局部 token/片段 CPC 预测，
   替代“预测整段 PE-AV 均值”；masked 目标改为相对时间差分/残差，并报告相对基线的增量。
3. **显式时间敏感目标**：段序判别/局部 bin 排列，使用相对时间编码而非绝对位置；扩大 order 样本数
   （当前 val 仅 20 部，指标噪声大）。
4. **表征头**：work vector 加 whitening/对比学习 + duration 去偏（对抗或正交化），再评估 Top-5
   邻域与 Full Mean 的互补性；目标先达到有效秩 >8、order 判别显著高于 chance。
5. **数据扩展**：纳入 30fps 子集做快速闪帧验证；增加作品数后再评估是否能支撑 4M 参数模型。
6. **达标前不注册主表征/地图**；达标后再讨论 `transformer_alpha_v1` 的 DB 登记与星云。

## 6. 复现与证据

```powershell
# 1) 数据（已生成，141MB）
.\.venv\Scripts\python.exe tools\build_v05_tokens.py --version v05_alpha_v1 --force
# 2) 训练（v2 为报告主结果；v1 保留对照）
.\.venv\Scripts\python.exe tools\train_v05.py --version v05_alpha_v2 --dataset-version v05_alpha_v1 --epochs 40 --patience 12 --batch-size 4 --device auto
```

证据：
- 数据：`data/v05_tokens/v05_alpha_v1/manifest.json`（110 works / 4130 segments / 6 truncated）
- v1：`data/models/v05_alpha_v1/checkpoint.pt`、`logs/v05_alpha_v1_metrics.json`
- v2：`data/models/v05_alpha_v2/checkpoint.pt`、`logs/v05_alpha_v2_metrics.json`
- 研究向量（未注册）：`data/representations/transformer_alpha_v1/<sequence_version>/vectors.npy`
- 训练日志：`logs/train_v05_alpha_v1.log`、`logs/train_v05_alpha_v2.log`
