# 全片时序表征升级分析（v0.4）

> 状态：进行中（阶段1 骨架；真实全片编码与多空间地图完成后回填数字）
> 基准：设计契约 [docs/plans/v0.4-full-coverage-design.md](../plans/v0.4-full-coverage-design.md)、
> 基础层提交 `55b01f7`；工具 `tools/compare_spaces.py`、`tools/clip_stability.py`。
> 数据/地图均为运行时产物（gitignored），本报告只记录可复现的命令、参数与结果摘要。

## 1. 目标与问题

把“均匀抽 ≤5×8s 片段再平均”的 legacy 表征，升级为覆盖整条时间轴的全片序列 + 多种聚合表征，
回答三个问题：

1. **覆盖完整**后，作品邻域结构会不会显著改变？哪些作品/作品对最不稳定？
2. Full Mean（全片平均）与 legacy（5 片段平均）的差异，是“更稳定”还是“只是换了一个空间”？
3. 后续 v0.4.2 Temporal Stats 与 v0.5 Transformer 是否值得保留旧空间作为对照？

本报告不评价作品质量、作者风格或模型优劣，只描述**表征空间之间的邻域迁移**。

## 2. 数据与方法（待阶段2 回填）

| 项目 | legacy | full_mean_v1 | temporal_stats_v1 |
| --- | --- | --- | --- |
| sequence_version | `peav-small-16f__f16_clip8s_x5_48k` | `待回填` | `待回填` |
| 采样 | 均匀 ≤5×8s | 全覆盖 8s 窗口（overlap 待定） | 同 full_mean |
| 聚合 | L2(mean(L2(z_i))) | 同公式、全片窗口 | 分组拼接（semantic/stages/drift/dynamics/self-sim/events） |
| canonical 作品数 | 待回填 | 待回填 | 待回填 |
| 地图版本 | 旧 map_v00x | 新 map_v0xx | 新 map_v0xx |
| 覆盖率（任意时刻 ≥1 窗口） | — | 待验证 | 待验证 |

分析工具：

```powershell
python tools/compare_spaces.py --a legacy_5clip_mean --b full_mean_v1 --k 5
python tools/compare_spaces.py --a full_mean_v1 --b temporal_stats_v1 --k 5
python tools/clip_stability.py --version <sequence_version> --k 5
```

输出指标：每作品 Top-k Jaccard、完全稳定作品比例、稳定对一个比例、被替换案例（含新旧邻居）、
作品对余弦矩阵 Pearson/Spearman、hubness 前后变化与秩相关。

## 3. 实验结果（占位，待阶段2）

### 3.1 覆盖率与编码统计

- 待回填：作品数 / 片段数分布 / 总时长 / 任取时间点覆盖情况 / 编码耗时 / 中断恢复验证。

### 3.2 legacy × full_mean 邻域迁移

| 指标 | 数值 |
| --- | --- |
| 共同 canonical 作品 | 待回填 |
| 平均 Top-5 Jaccard | 待回填 |
| 完全稳定作品数/比例 | 待回填 |
| 稳定对比例 | 待回填 |
| 作品对余弦 Pearson / Spearman | 待回填 |
| hubness 秩相关 | 待回填 |
| 替换最大案例（Top3） | 待回填 |

### 3.3 full_mean × temporal_stats

- 待回填：同表 + 顺序敏感案例（反转片段顺序后 Full Mean 距离≈0、Temporal Stats 距离显著>0）。

### 3.4 hubness 与稳定对的变化

- 待回填：Top hub 前后变化、是否出现新的“万金油”作品、hub 与片段数/时长的相关性（仅描述，不作因果）。

## 4. 局限与注意事项

- 全片模式对超长视频（>1h）默认跳过或需要 `--allow-long`；样本覆盖不等于均匀语义覆盖；
- PE-AV 仍是 16 帧稀疏采样，8s 窗口内部细节未被完整建模；
- full_mean 只是“完整覆盖的单一变量基线”，不代表优于 legacy；
- 地图坐标跨表征/跨序列不可比较绝对位置；分位、hubness 都是集合相对量；
- 真实数据只跑一次，不宣称统计显著性；
- 不把邻域迁移解读为作品质量或作者水平变化。

## 5. 复现与证据

- 命令：见第 2 节；原始 JSON/CSV：`logs/space_compare_*`、`logs/clip_stability_*`；
- 数据库/地图为运行产物，不随仓库提交；需要复现时按 README 的“环境要求 + 配置说明”准备；
- 待阶段2 回填后，在 README / ARCHITECTURE 引用本报告结论与数字。
