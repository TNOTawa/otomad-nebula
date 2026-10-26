# 密集视觉语义通道全量运行报告（v0.4.3）

> 状态：全量运行完成（2026-10-26；数字由 Lead 回填，t4 独立复核）
> 相关：设计 `docs/plans/v0.4.3-dense-visual-semantics.md`；pilot `docs/reports/dense-visual-pilot.md`；
> 契约 `docs/CONTRACTS_V2.md` §10.3；架构 `ARCHITECTURE.md` §6.6。
> 本报告只记录可复现命令、参数与结果摘要；原始数据（data/visual_dense、logs）为运行时产物，不随仓库提交。

## 1. 目标与能力边界

在 PE-AV 联合表征（每 8s 仅 16 帧≈2fps 稀疏语义）之外，新增一个**独立、冻结编码器**的密集视觉通道：

- 按配置帧率（默认 8fps）解码整部视频，帧级 L2 归一化特征（fp16）；
- 按 v0.4 全片 8s 窗口池化为窗口级向量（fp32，与 PE-AV 窗口一一对应）；
- 用 CLIP-only / DINOv2-only 两个空间做对照星云，检查“更高频视觉语义是否携带 PE-AV 之外的信息”；
- 30fps 只在小样本子集（默认 16 部）上测量帧级轨迹，不默认全量。

**边界**：这是冻结编码器的**帧级特征 + 固定窗口池化**，提高的是视觉时间密度，**不是逐帧语义理解**；
本轮不训练任何可学习音画时序模型，也不宣称“像人一样观看”。完整覆盖/密集特征都不等于语义理解。

## 2. 数据与方法

### 2.1 编码器与固定 revision

| 通道 | 模型 | revision | 维度 | 说明 |
| --- | --- | --- | --- | --- |
| CLIP | `openai/clip-vit-base-patch16` | `57c216476eefef5ab752ec549e440a49ae4ae5f3` | 512 | 图文对齐语义（主通道） |
| DINOv2 | `facebook/dinov2-small` | `ed25f3a31f01632728cabb09d1542f84ab7b0056` | 384 | 自监督视觉结构（对照） |

revision 固定写入配置与 `visual_versions` 表，禁止浮动主干；任一 model/revision/fps/size/preprocess
变化都必须新建 visual version。

### 2.2 参数与版本命名

- 主运行：`fps=8.0, size=224, preprocess=letterbox, batch_size=32`；帧级 fp16 + 窗口级 fp32；
- 30fps 子集：`subset_fps=30.0, subset_count=16`（按动态分数覆盖短/中/长时长桶，确定性排序）；
- 版本命名：逻辑身份 `encoder@revision__fps{fps}_s{size}_{preprocess}`，例如
  `openai/clip-vit-base-patch16@57c216...__fps8_s224_letterbox`；
  DB/目录使用 `visual_version_slug` 后的文件系统安全形式
  （`/`、`@` 转为 `-`，例如
  `openai-clip-vit-base-patch16-57c216...__fps8_s224_letterbox`）；
  `visual_versions.model_id/revision` 保留原始身份；
- 窗口：复用 v0.4 `plan_full_coverage` 的 8s 窗口（无间隙、末尾锚定、权重和=1），
  按重叠时长加权池化；窗口时间戳与 PE-AV `clip_times.npy` 对齐。

### 2.3 语料与流水线

- canonical 作品：**110 部**（真实库共 122 条 works，10 条本地/Bilibili 重复身份合并后：
  本地 23 + Bilibili 87）；30fps 子集 **16 部**；任务总数 **252**（110×2 主版本 + 16×2 子集版本）；
- 只处理 canonical 且 `status=completed` 的作品；Bilibili 作品每部只下载一次，
  优先复用 temp 中可 probe 的源文件，否则下载；所有视觉任务成功后清理 temp，失败保留 temp 续跑；
- 每部作品任务数：主 8fps 版本 × 2 编码器；30fps 子集作品再加 ×2；状态写入
  `visual_sequences(work_id, version)`，支持断点续跑。

### 2.4 命令

~~~powershell
python tools/visual_pipeline.py run
python tools/visual_pipeline.py status
python tools/visual_pipeline.py list
python tools/visual_pipeline.py clean-temp
python tools/dense_preprocess_compare.py --help
python tools/embedding_diagnostics.py --representation dense_visual_clip_v1 dense_visual_dino_v1 \
    --umap-neighbors 15 30 50 --components 2 3
~~~

## 3. 宽高比预处理对照（stretch vs letterbox）

少量作品、同一编码器/帧率/尺寸下比较两种预处理；窗口池化后在高维空间比较余弦。

样本：3 部本地 canonical 作品（work 7=29.87s、work 1=101.45s、work 8=252.65s，均为 16:9）。
条件：同一编码器、8fps、224px、batch=32、bf16；两种预处理逐作品各跑一次。
指标为 stretch 与 letterbox 的**窗口池化向量逐窗口余弦**（1.0 表示完全一致），
帧数在两种预处理下完全相同。

| 编码器 | 窗口余弦 mean | 窗口余弦 min | 帧数一致 | decode 比值（letterbox/stretch） | encode 比值（letterbox/stretch） |
| --- | --- | --- | --- | --- | --- |
| CLIP ViT-B/16 | 0.942454 | 0.875964 | 是（239/811/2021） | 0.9784 | 0.7015（首作品含预热，参考稳态） |
| DINOv2-small | 0.872513 | 0.789650 | 是（239/811/2021） | 1.0776 | 0.7440（同上） |

结论：**采用 letterbox**。拉伸会改变窗口语义（CLIP 平均差 5.8%、DINOv2 平均差 12.7%，
最小余弦 0.79），而 letterbox 保持宽高比、黑边补齐，编码吞吐无实质损失
（首作品的 encode 值含模型预热，其他作品两者接近）。

原始证据：`logs/dense_preprocess_compare_clip.json`、`logs/dense_preprocess_compare_dino.json`。

## 4. 全量运行统计

| 指标 | 实际值 |
| --- | --- |
| 运行日期 / 代码版本 | 2026-10-26 / v0.4.3（schema v4；提交号见 git log） |
| canonical 作品数 | 110（本地 23 + Bilibili 87；原 122 条 works 中 10 条重复身份合并） |
| visual_versions | 4 个：CLIP/DINOv2 × {8fps, 30fps}，均 letterbox/224 |
| visual_sequences 总任务数 | **252**（8fps 主版本 220 + 30fps 子集 32） |
| 状态计数 | pending 0 / in_progress 0 / **completed 252** / failed 0 |
| 帧级向量总数（8fps） | 每个编码器 **260,885** 帧（110 部合计），窗口 **4,130**（=PE-AV 全片窗口数） |
| 30fps 子集 | 16 部 × 2 编码器 = 32 任务；每个编码器 **140,149** 帧 / **593** 窗口 |
| 平均帧数/部、窗口数/部 | 8fps：2,371.7 帧 / 37.55 窗口；30fps：8,759.3 帧 / 37.06 窗口 |
| 耗时均值/部（decode / encode / total） | CLIP 8fps：2.22 / 10.23 / 12.64s；DINOv2 8fps：2.26 / 7.42 / 9.84s；CLIP 30fps：2.76 / 36.70 / 39.98s；DINOv2 30fps：2.61 / 24.78 / 27.79s |
| 存储 | 8fps 合计 **447.9MB**（CLIP 256.1 + DINO 191.8）；30fps 子集合计 226.0MB；总 **673.9MB**（按 visual_full_run_stats.json 实际 storage_bytes 合计） |
| 失败与续跑记录 | 最终 0 失败；开发中两次真实失败均为“音频轨长于视频轨导致按容器时长估算帧数偏多”（work 32 / work 60），改为按**视频流时长**估算期望帧数后重跑通过（`temp` 复用、只补缺失任务）；87 部 Bilibili 作品每部仅下载一次 |

运行日志：`logs/visual_run_full.log`、`logs/visual_run_full2.log`、`logs/visual_run_full3.log`；
离线统计快照：`logs/visual_full_run_stats.json`；命令：`tools/visual_pipeline.py status`。

产物规模：`data/visual_dense/<4 个 visual_version>/<work_id>/`（帧级 fp16 + 窗口级 fp32 + manifest），
每个 8s 窗口与 PE-AV `clip_times.npy` 对齐；临时视频在每部作品全部任务完成后自动清理。

## 5. CLIP-only 与 DINOv2-only 对照星云

| 指标 | CLIP-only（dense_visual_clip_v1） | DINOv2-only（dense_visual_dino_v1） |
| --- | --- | --- |
| visual_version | `openai-clip-vit-base-patch16-...__fps8_s224_letterbox` | `facebook-dinov2-small-...__fps8_s224_letterbox` |
| 表征 id / dim | 4 / 512 | 5 / 384 |
| 地图版本 | `map_v013`（110 部） | `map_v014`（110 部） |
| 与 PE-AV clip_video 的窗口结构 RSA（110 部，mean / std / min / p10） | 0.6949 / 0.1577 / 0.3689 / 0.4865 | 0.7246 / 0.1483 / 0.1722 / 0.5429 |
| 与 PE-AV clip_joint 的窗口结构 RSA | 0.6226 / 0.2224 / -0.5282 / 0.3970 | 0.6482 / 0.2093 / -0.6025 / 0.4592 |
| CLIP × DINO 空间（Top-5 Jaccard / 余弦矩阵 Pearson / mean\|Δcos\|） | 0.3874 / 0.7684 / 0.2627 | 同左（两空间互比） |
| 与 Full Mean 的邻域迁移（Jaccard / Pearson / 完全稳定作品 / Top1 一致率） | 0.3333 / 0.7344 / 1/110 / 24.55% | 0.3065 / 0.7594 / 3/110 / 20.91% |
| hubness Top（本空间） | 36（27 部把它列入 Top5） | 117（25 部） |
| 8fps vs 30fps 窗口向量余弦（16 部子集） | mean 0.9998 / min 0.9997 | mean 0.9994 / min 0.9987 |

说明：窗口级 8fps 与 30fps 在 pilot 中几乎一致（work 7 余弦 mean/min=0.9999），30fps 的价值主要在
帧级轨迹；本轮不把 30fps 作为全量默认。

## 6. PCA / UMAP 低维投影诊断

对 `dense_visual_clip_v1` / `dense_visual_dino_v1`（必要时加 full_mean_v1 / temporal_stats_v1 对照）
计算 PCA 2D/3D 与 UMAP（n_neighbors=15/30/50，metric=cosine，random_state=42，n_jobs=1）：

每个单元格为 **Top-10 保持率 / 高维余弦距离与低维欧氏距离的 Pearson**（110 部，k=10，random_state=42）。
Top-1 与 Spearman 见 `logs/dim_diagnostics_v2.json`。

| 表征（维度） | PCA-2D | PCA-3D | UMAP-15/3D | UMAP-30/3D | UMAP-50/3D |
| --- | --- | --- | --- | --- | --- |
| Full Mean v1（1024） | 0.297 / 0.416 | 0.392 / 0.503 | 0.412 / 0.313 | 0.400 / 0.327 | 0.405 / 0.333 |
| Temporal Stats v1（6356） | 0.213 / 0.302 | 0.272 / 0.349 | 0.386 / 0.354 | 0.390 / 0.351 | 0.369 / 0.364 |
| Dense Visual·CLIP（512） | 0.369 / 0.574 | 0.474 / 0.686 | 0.436 / 0.411 | 0.446 / 0.402 | 0.435 / 0.392 |
| Dense Visual·DINOv2（384） | 0.348 / 0.601 | 0.373 / 0.757 | 0.453 / 0.520 | 0.453 / 0.548 | 0.447 / 0.549 |

UMAP n_neighbors 两两 Top-10 保持率（3D）：Full Mean 0.75–0.76；Temporal Stats 0.60–0.63；
CLIP 0.73–0.76；DINOv2 0.73–0.79（2D 明显更低，Temporal Stats 2D 仅 0.36–0.49）。

证据：`logs/dim_diagnostics_v2.json/.csv`、`logs/dim_diagnostics_v2_html/`；v0.4 旧版
`logs/dim_diagnostics.json` 保留。

**判读（3D 结构杂乱的来源）**：

1. **低维投影的信息损失是主因**：多数表征的 3D 投影只保留约 37–47% 的高维 Top-10 邻居
   （Temporal Stats 更低，见下）；2D 更低。投影不是“高维结构的忠实缩略图”，只能用于浏览。
2. **Temporal Stats 的损失更大**：6356 维由人工分组/权重拼接，3D 只保留 27–39%，
   距离相关也最低（0.28–0.36）；说明其高低维几何更不一致，地图上“看起来远/近”更容易误导。
3. **UMAP 超参敏感**：n_neighbors 从 15 改到 30/50 会替换约 24–40% 的 3D 近邻
   （Temporal Stats 最多），所以“地图上相邻”不能当作稳定结论。
4. **不是单一 bug**：DINOv2 的 PCA-3D 距离相关最高（0.757）却仍只保留 37% 近邻，
   说明问题不是某次训练/随机种子，而是 110 部 × 高维空间的固有稀疏 + 投影限制。
   定量比较必须回到高维余弦近邻（SimilarityIndex / compare_spaces），地图只作交互浏览；
   跨表征/跨投影坐标不可比较。

## 7. 结论与下一步

- **工程可行性**：252/252 任务完成、最终 0 失败；8fps 双编码器共 521,770 帧 / 8,260 窗口，
  存储 447.9MB（30fps 子集另 226.0MB，**合计 673.9MB**）；任务计时累计约 59 分钟，三次运行墙钟累计约 55 分钟
  （含一次帧数校验中断与 87 部 B 站作品每部一次重新下载）。分层、下载一次、断点续跑与
  完成后清理均按设计工作。
- **信息价值**：dense visual 窗口结构与 PE-AV 相关但不等价（CLIP RSA 0.69、DINOv2 0.72，
  最差作品接近 0.17–0.37）；CLIP-only 与 DINOv2-only 两个空间本身也只有
  Top-5 Jaccard=0.387、余弦矩阵 Pearson=0.768、平均 |Δcos|=0.263，说明“语义”与
  “视觉结构”是两条可测的不同信号；DINOv2 对画面变化更敏感（PCA-3D 距离相关 0.757 最高）。
- **30fps 子集**：窗口级 8fps 与 30fps 几乎饱和（CLIP cosine 0.9998，DINOv2 0.9994），
  30fps 的增量只在帧级轨迹（pilot：窗口内每秒变化 1.59×）；因此 30fps 不作为全量默认，
  保留 16 部供 v0.5 帧级时序研究。
- **地图与近邻**：低维投影只保留 37–47% 的 Top-10 高维邻居，Temporal Stats 更低；
  3D 星云只能用于浏览，定量结论必须用高维近邻。建议后续 App 默认优先 PE-AV 空间
  （已实现：默认仍打开 Temporal Stats 的 map_v012，Dense Visual 为可选对照空间）。
- **下一步（不在本轮执行）**：把 dense visual 窗口/帧级序列作为新输入，构建
  `temporal_stats_v2` 或 v0.5 分层时序模型（局部音画交叉注意力 → 全曲 Transformer）；
  任何融合/训练表征都必须新建版本，不覆盖 v0.4/v0.4.3 已有表征与地图；仍不宣称逐帧语义理解。

## 8. 复现与证据

- 代码：`nebula/dense_visual.py`、`nebula/visual_pipeline.py`、
  `tools/visual_pipeline.py`、`tools/build_dense_visual_representation.py`、
  `tools/dense_preprocess_compare.py`、`tools/embedding_diagnostics.py`；
- 测试：`tests/test_dense_visual.py`、`tests/test_dense_visual_av_duration.py`、
  `tests/test_dense_visual_representation.py`、`tests/test_visual_pipeline.py`、
  `tests/test_embedding_diagnostics.py`、`tests/test_invariants.py`；
- 运行时数据：`data/visual_dense/<visual_version>/`（不随仓库提交）；
- 报告数字以本文件 + `logs/` 下 JSON/CSV 为准，由 Lead 回填、t4 复核；
- 关键证据文件：
  - 全量统计：`logs/visual_full_run_stats.json`；
  - 宽高比对照：`logs/dense_preprocess_compare_clip.json`、`logs/dense_preprocess_compare_dino.json`；
  - 视觉通道分析：`logs/dense_visual_full_analysis.json`；
  - 空间对比：`logs/space_compare_*.json/.csv`（Full Mean×CLIP、Full Mean×DINOv2、CLIP×DINOv2）；
  - 低维诊断：`logs/dim_diagnostics_v2.json/.csv`、`logs/dim_diagnostics_v2_html/`；
  - 视觉表征构建：`tools/build_dense_visual_representation.py --all-main`；
  - 新地图：`map_v013`（CLIP-only）、`map_v014`（DINOv2-only），旧 `map_v008/v011/v012` 保留。
