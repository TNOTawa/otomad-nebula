# v0.4.4 跨模型近邻共识与轻量融合评估

> 状态：完成（2026-10-26；真实表征/真实共识产物，t4 复核）
> 相关：v0.4.3.1 窗口权重修复 `docs/reports/v0431-window-weights-and-versioning.md`；
> 契约 `docs/CONTRACTS_V2.md` §10.7.5；界面 `app.py` 高维邻域连线。

## 1. 目标与边界

- 回答：**多个模型共同认可的相似作品，是否比单一模型的近邻更有解释力？**
- 输入：`full_mean_v1`（PE-AV 全片语义）+ `dense_visual_clip_v2`（CLIP 密集视觉语义，
  coverage 权重）+ `dense_visual_dino_v2`（DINOv2 密集视觉结构）；110 部 canonical 共同作品；
- 不做：人工风格/流派标注、正式注册“主模型”、以共识结果定义分类；不训练新模型；
- 输出是**观察性与可复现的邻域一致性证据**，不是因果或审美判断。

## 2. 方法

### 2.1 高维近邻与共识边

- 对每个空间用 canonical 作品的高维余弦矩阵计算 Top-k（k=5）：
  - 有向边：i 的 Top-k 中包含 j；
  - 互为近邻（mutual）：i∈Top-k(j) 且 j∈Top-k(i)；
- 跨空间统计：每空间 mutual 边数、跨空间有向/互邻保留率、共识边（>=2 个空间互为 Top-k）、
  全共识边（>=3 空间）、覆盖作品比例、共识强度；
- 缺失空间显式报告，不静默补零。

### 2.2 轻量融合基线（不写 DB、不注册正式表征）

1. **rank-average 融合**：各空间 Top-k 排名平均后重取 Top-k；
2. **concat 向量融合**：各空间向量 L2 后按块权重（equal / $1/\sqrt{dim}$）拼接再 L2；
   在融合空间计算 Top-5 及与单空间/共识边的重合；只作评估，不写 `data/representations/`。

### 2.3 输出与命令

```text
logs/neighbor_consensus.json   # 空间元数据、mutual/共识统计、融合指标、节点/边表
logs/neighbor_consensus.csv    # 边表（per-space rank/cos、mutual、consensus_count 等）
logs/neighbor_consensus_nodes.csv  # 节点表（mutual/consensus 度、强度）
```

```powershell
python tools/build_consensus_graph.py \
    --representations full_mean_v1 dense_visual_clip_v2 dense_visual_dino_v2 \
    --k 5 --min-spaces 2 --fusion both
```

- JSON 实际字段：`tool/version/generated_at/meta`、`representations`、`params`、
  `spaces`、`mutual_edges`、`consensus`（直方图/覆盖/强度/跨空间统计）、
  `nodes`、`edges`、`consensus_edges`、`fusion`
  （rank_average 与 concat equal/inv_sqrt_dim）；确定性输出，无随机。

## 3. 结果（已回填）

### 3.1 各空间 mutual 结构

| 空间 | 作品数 | 维度 | mutual 边数 | mutual 比例 | mean Top-k 余弦 |
| --- | --- | --- | --- | --- | --- |
| `full_mean_v1`（PE-AV） | 110 | 1024 | 61 | 0.2218 | 0.9146 |
| `dense_visual_clip_v2` | 110 | 512 | 75 | 0.2727 | 0.9365 |
| `dense_visual_dino_v2` | 110 | 384 | 73 | 0.2655 | 0.8146 |

### 3.2 跨空间一致性

| 空间对 | 有向 Jaccard | 有向保留率（A→B） | mutual Jaccard | mutual 保留率（A→B / B→A） |
| --- | --- | --- | --- | --- |
| Full Mean × CLIP v2 | 0.3018 | 0.4636 | 0.3204 | 0.5410 / 0.4400 |
| Full Mean × DINOv2 v2 | 0.2761 | 0.4327 | 0.2407 | 0.4262 / 0.3562 |
| CLIP v2 × DINOv2 v2 | 0.3514 | 0.5200 | 0.3455 | 0.5067 / 0.5205 |

### 3.3 共识边与融合基线

| 指标 | 值 |
| --- | --- |
| 有向边总数 | 913（3 空间 × 550，去重后边集） |
| >=2 空间共识边 | **53** |
| >=3 空间全共识边 | **22** |
| 共识直方图（0/1/2/3 空间） | 779 / 81 / 31 / 22 |
| 共识边覆盖作品比例 | **44/110（40.00%）**；全共识覆盖 23/110（20.91%） |
| 共识强度 mean/std/min/max | 0.8050 / 0.1642 / 0.6667 / 1.0000 |
| 平均/最大共识度 | 0.9636 / 5 |
| rank-average 融合 | Jaccard(vs spaces)=**0.5331**；vs 共识=**0.4303**；edge Jaccard=0.1054；mean rank score=3.6285 |
| concat(equal) 融合（dim 1920） | Jaccard(vs spaces)=**0.5358**；vs 共识=**0.4106**；edge Jaccard=0.1061；mean neighbor cos=0.8828 |
| concat(inv_sqrt_dim)（dim 1920） | Jaccard(vs spaces)=**0.5385**；vs 共识=**0.4121**；edge Jaccard=0.1088；mean neighbor cos=0.8704 |

说明：3 个空间的直接两两 mutual Jaccard 仅 0.24–0.35，说明“哪个模型算近邻”差异很大；
共识边只占全部有向的约 5.8%，但覆盖 40% 作品。融合基线相对单空间有较高邻域重合
（Jaccard≈0.53），但并没有完全复现共识边（vs 共识≈0.41–0.43）。

## 4. 界面：高维互为近邻连线与共识边叠加

- 侧栏“显示高维互为近邻连线”（默认关）：用当前表征空间 `SimilarityIndex(representation_id)`
  计算可见作品的互为 Top-k（默认 k=5），Plotly line traces 叠加到 2D/3D；只画 mutual；
  最大边数上限 300，超过截断并提示；
- 可选“仅显示跨模型共识边”：读取 `logs/neighbor_consensus.json` 的 consensus_count，
  阈值 >=2 / >=3；文件缺失时提示并回退普通 mutual；
- 详情面板：共识 JSON 可用时显示“跨空间 / 跨模型共识近邻”摘要（最多 5 条）；
- PCA 3D 仅作辅助观察（链接 `logs/dim_diagnostics_v2_html` 或说明），不改变默认 UMAP 坐标语义；
- 真实 AppTest（t4 复核）：默认 `map_v012`；7 空间（Legacy→map_v008、Full Mean→map_v011、
  Temporal→map_v012、CLIP v1→map_v013、DINO v1→map_v014、CLIP v2→map_v015、DINO v2→map_v016）；
  互邻开关 k=5、共识阈值 2/3、边数 caption 正常；视觉空间显示“视觉版本”与时间轴提示。

## 5. 结论与下一步

1. **多模型共识提供单模型之外的信息**：三个空间的 mutual 结构差异明显（两两 Jaccard 0.24–0.35），
   40% 作品至少有一条跨 2 空间共识边、23 部作品有全共识边；共识边可作为跨模型稳定相似候选。
2. **融合不等于共识**：rank-average / concat 融合与各单空间重合约 0.53，但只覆盖共识边的
   约 41–43% Jaccard；融合更适合作为“综合近邻浏览”，不能替代共识作为跨模型一致性证据。
3. **concat 权重影响有限**：equal vs inv_sqrt_dim 的 Jaccard 仅差约 0.003；inv_sqrt_dim 的
   mean neighbor cos 更低（0.870 vs 0.883），说明块权重会改变融合几何，但共识覆盖变化不大。
4. **使用建议**：共识边用于“多模型都认可的相似对”观察与界面高亮；单模型近邻仍用于各自空间；
   两者都只作交互探索，不能作为风格/流派分类依据。
5. **下一步（不在本轮）**：如进入 v0.5 可学习时序模型，需新建表征版本并重新训练/验证；
   任何正式融合表征不得覆盖 v0.4.3.1 的 v2 结果；本轮不宣称逐帧语义理解或风格结论。

## 6. 复现与证据

- 工具：`tools/build_consensus_graph.py`；界面：`app.py`、`nebula/appdata.py`、
  `nebula/similarity.py`；测试：`tests/test_consensus_graph.py`、`tests/test_app.py`、
  `tests/test_appdata.py`、`tests/test_similarity.py`、`tests/test_invariants.py`；
- 真实产物：`logs/neighbor_consensus.json`、`logs/neighbor_consensus.csv`、
  `logs/neighbor_consensus_nodes.csv`（命令：`python tools/build_consensus_graph.py --k 5 --fusion both`）；
- 输入表征：full_mean_v1（id=2）、dense_visual_clip_v2（id=6）、dense_visual_dino_v2（id=7）；
- AppTest/最终 QA 数字见交付报告；共识为观察性证据，不是分类模型。
