# v0.4.3.1 窗口覆盖权重修复与三层版本分离

> 状态：修复与 v2 重建完成（2026-10-26；真实库/真实产物，t4 复核）
> 相关：v0.4.3 全量报告 `docs/reports/visual-dense-full-run.md`；契约 `docs/CONTRACTS_V2.md` §10.3/§10.7；
> 架构 `ARCHITECTURE.md` §6.6/§6.7；设计 `docs/plans/v0.4.3-dense-visual-semantics.md`。

## 1. 背景与问题

用户复核 v0.4.3 交付时确认两处数据一致性问题（均不影响 PE-AV 主序列，但影响视觉表征与地图解释）：

1. **窗口覆盖权重未保存，作品向量退化为等权平均**
   - `plan_full_coverage` 为每个 8s 窗口计算了覆盖权重（重叠时长积分归一，Σweight=1），
     PE-AV 全片序列把权重写入 `clip_times.npy` 并用于加权聚合；
   - v0.4.3 视觉流水线保存 `windows.npy` 时没有把同样窗口计划下的权重写入
     `manifest.json`，`tools/build_dense_visual_representation.py` 的
     `_manifest_weights` 读不到权重后退化为等权：
     `work_vector = mean(windows)`，而正确行为是 `Σ weight_i * window_i`（再 L2 归一）；
   - 影响：`dense_visual_clip_v1`/`dense_visual_dino_v1` 与 `map_v013`/`map_v014`
     的作品向量/近邻/坐标与“按真实覆盖加权”的表征不一致；v0.4.4 共识若基于 v1 也会继承该偏差。
2. **视觉版本号没有窗口计划身份**
   - v0.4.3 的 `visual_version` 只含 encoder/revision/fps/size/preprocess，
     不含窗口时长、overlap 与权重方案；
   - 一旦调整窗口计划（例如 8s→4s、overlap 0→2s、coverage→equal），
     旧版本号会复用同一 `data/visual_dense/<version>/` 或 DB 记录，**破坏版本隔离**，
     无法并存 A/B 结果；变更窗口也不需要重新编码帧，但旧结构迫使重新编码或覆盖。

## 2. 证据（已回填）

| 证据 | 实际值 |
| --- | --- |
| 旧 manifest 无 `window_weights` | v0.4.3 的 `manifest.json` 有 `window_times` 但无权重字段；v2 窗口层 manifest 均含 `window_weights` |
| 权重审计（252 个序列） | `logs/v0431_weight_audit.json`：weight_sum 0.999999847..1.000000209；coverage-vs-equal 最大绝对差 max=0.022221、mean=0.001713；78/252 序列差>1e-3；最大案例 work 27（3 窗口，weights 0.344436/0.311112/0.344451） |
| 等权 vs coverage 向量差异 | 逐作品 v1→v2 余弦：CLIP mean=0.999999 / min=0.999972；DINOv2 mean=0.999996 / min=0.999863（110 部均无 >1e-3 的差） |
| 近邻变化 | CLIP v1→v2 Top-5 Jaccard=0.9939、稳定 108/110（98.18%）；DINOv2 Jaccard=0.9970、稳定 109/110（99.09%）；详见 §5 |
| 帧级特征不变 | 重建用 `tools/build_visual_windows.py` 只读已保存帧级特征，未下载/解码/跑模型；帧 bundle 未改写、未重新编码（`logs/build_visual_windows_full.log`） |

## 3. 修复方式

### 3.1 三层版本分离

| 层 | 版本内容 | 示例 | 变化时重建范围 |
| --- | --- | --- | --- |
| Frame Encoding Version | encoder/model+revision+fps+size+preprocess | `openai-clip-vit-base-patch16-<rev>__fps8_s224_letterbox` | 帧级特征（需解码+编码） |
| Window Plan Version | clip_seconds + overlap_seconds + weight_scheme | `win8s_ov0__coverage-v1` | 窗口池化/权重（只读帧级特征） |
| Representation Version | 聚合器 + 使用哪个窗口计划/帧版本 | `dense_visual_clip_v2` / `dense_visual_dino_v2` | 作品向量（只读窗口层） |

- 帧编码版本继续使用现有 `visual_versions`（schema v4）；
- 窗口计划版本写入新表 `visual_window_plans`，窗口任务写入
  `visual_window_sequences(work_id, frame_version, window_plan_version)`（schema v5）；
- 表征表通过 `params_json` 记录所用 frame_version/window_plan_version/weight_scheme/aggregation；
- **正交**：同一帧编码版本可配多个窗口计划；同一窗口计划可用于多个帧编码版本。

### 3.2 覆盖权重落到窗口层

- `nebula/sampling.py` 公开 `coverage_weights(windows, duration_s)`：
  与 `plan_full_coverage` 相同的覆盖计数积分并归一（Σ=1），供通用复用；
- `nebula/dense_visual.py` 的 `save_bundle`/`load_bundle` 支持
  `window_weights`（长度=$n_windows$、有限、非负、和>0；旧 manifest 无字段时 None）；
- `tools/build_visual_windows.py` 从已保存帧级特征
  （`data/visual_dense/<frame_version>/<work_id>/dense_visual.npz`）重建窗口：
  按窗口计划池化 `windows.npy`、`window_std.npy`，保存
  `window_weights.npy`（权重）与 `manifest.json`，**不下载、不解码、不跑模型**；
- 数据布局：
  `data/visual_windows/<window_plan_version>/<frame_version>/<work_id>/`。

### 3.3 源视频持久缓存（避免重建时重复下载）

- 新增 `config.paths.sources`（默认 `data/sources`，位于 `data/` 下，**不随 temp 清理**）；
- Bilibili 下载优先复用 `data/sources/<bvid>/<bvid>_p<part>.*` 中可 probe 的源文件；
  未命中才下载，先写缓存临时文件再原子 rename；失败清理临时文件、保留成功缓存；
- 帧/窗口重建直接使用已保存帧级特征；temp 仅存放切片/中间文件，清理 temp 不影响 sources。

### 3.4 v2 表征与地图

- `tools/build_dense_visual_representation.py --v2` 从窗口计划层读取
  `windows.npy` + `window_weights.npy`，按权重聚合为
  `dense_visual_clip_v2`(id=6, 512d) / `dense_visual_dino_v2`(id=7, 384d)；
  `params_json` 记录 window_plan_version=win8s_ov0__coverage-v1、weight_scheme=coverage-v1、
  clip=8.0/overlap=0.0、aggregation=coverage-weighted-window-mean-l2；
- 旧 `dense_visual_clip_v1`/`dense_visual_dino_v1` 与 `map_v013`/`map_v014`
  保持只读、不覆盖；v2 另建地图 `map_v015`（CLIP v2）/ `map_v016`（DINOv2 v2），
  均 110 部、source_kind=visual；
- 旧 v1 与 v2 坐标不可直接比较；跨空间比较用 `tools/compare_spaces.py` 高维指标。

### 3.5 命令

```powershell
# 1) 从已保存帧级特征重建窗口层（不重新编码帧）
python tools/build_visual_windows.py --all --frame-version <frame_version> \
    --window-seconds 8 --overlap-seconds 0 --weight-scheme coverage-v1

# 2) 构建 v2 视觉表征（覆盖权重；--v2 读取窗口层 window_weights.npy）
python tools/build_dense_visual_representation.py --v2

# 3) 建 v2 对照地图
python map.py build --representation dense_visual_clip_v2
python map.py build --representation dense_visual_dino_v2

# 4) 前后对比
python tools/compare_spaces.py --a dense_visual_clip_v1 --b dense_visual_clip_v2
python tools/compare_spaces.py --a dense_visual_dino_v1 --b dense_visual_dino_v2
```

## 4. 重建 v2 统计（已回填）

| 指标 | v1（等权，历史） | v2（coverage 权重） |
| --- | --- | --- |
| 窗口计划版本 | 无（未记录） | **win8s_ov0__coverage-v1**（clip 8.0 / overlap 0.0 / coverage-v1；DB 仅此 1 个计划） |
| 帧编码版本 | CLIP/DINOv2 × 8fps/30fps 共 4 个 | 同左（未重新编码帧） |
| 任务数 / completed / failed | 未单独登记窗口层 | **252 / 252 / 0** |
| 窗口数 | — | **9446**（8fps：4130×2=8260；30fps：593×2=1186） |
| 权重和范围 | 未保存（等权） | **0.999999847..1.000000209**（252 序列） |
| 窗口层存储新增 | — | 约 **33MB**（windows + std + weights） |
| 作品向量数（CLIP/DINOv2） | 各 110 | 各 110（joint） |
| 表征状态/维度 | v1 id=4/5，512/384 ready | **id=6/7，512/384 ready** |
| 地图版本（CLIP/DINOv2） | map_v013/map_v014 | **map_v015/map_v016**（各 110 部，source_kind=visual） |

## 5. 修复前后对比（已回填）

| 指标 | CLIP v1→v2 | DINOv2 v1→v2 |
| --- | --- | --- |
| 作品向量余弦 mean / min | 0.999999 / 0.999972 | 0.999996 / 0.999863 |
| Top-5 Jaccard / 中位 / 平均重合 | 0.9939 / 1.0 / 4.98/5 | 0.9970 / 1.0 / 4.98/5 |
| 稳定作品 / 完全稳定比例 | 108/110（98.18%） | 109/110（99.09%） |
| Top1 一致率 | 99.09% | 100.00% |
| 稳定对比例 | 0.999666 | 0.999833 |
| 高维距离 Pearson / Spearman | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| 平均绝对差 / 最大绝对差 | 0.000165 / 0.003153 | 0.000297（均值） |
| 被替换案例 | work 22（丢 25、得 24，Jaccard 0.667）；work 80（丢 79、得 106） | work 59（丢 44、得 120，Jaccard 0.667） |
| hubness Top / 秩相关 | 36（27）/ 0.9995 | 117（25）/ 0.9975 |

**影响量化与结论（如实）**：本数据集为 8s/不重叠窗口，除末尾不足 8s 的窗口外权重近似均匀，
因此 v1→v2 的作品向量/近邻变化**非常小**（Top-5 Jaccard 0.99+，稳定作品 108/109 部）。
但这是**正确性修复**：coverage 权重现在真实写入窗口层并用于聚合（weight_sum 审计通过），
未来在 overlap>0、更短窗口或非均匀覆盖的计划下差异会显著放大；不能因当前影响小而不修。
与 Full Mean 的邻域迁移也几乎不变：CLIP v2 Jaccard=0.3355/Pearson=0.7342/稳定 1/110/Top1 24.55%；
DINOv2 v2 0.3065/0.7596/稳定 3/110/Top1 20.91%。

## 6. 兼容、限制与边界

- 旧 v1 表征/地图保留只读；v2 为**同一帧级特征**上的权重修复，不改变模型能力，不宣称新的“理解”；
- 权重修复只影响作品级聚合与由它派生的近邻/地图，不改变帧级特征与窗口向量本身；
- 不同窗口计划/不同帧编码版本的窗口层与表征矩阵禁止混用；
- 旧 manifest 无 `window_weights` 时读取为 None（显式区分“等权”与“未知”），
  不得静默假设等权；
- 本轮未重跑/未新增 30fps 全量（仍为 16 部子集）；未重新编码任何帧；
- 仍不宣称逐帧语义理解；30fps 只服务帧级轨迹研究。

## 7. 复现与证据

- 代码：`nebula/sampling.py`、`nebula/dense_visual.py`、
  `nebula/visual_pipeline.py`、`nebula/download.py`、`nebula/db.py`、
  `nebula/config.py`、`nebula/representations.py`；
- 工具：`tools/build_visual_windows.py`、`tools/build_dense_visual_representation.py`；
- 测试：`tests/test_sampling.py`、`tests/test_dense_visual.py`、
  `tests/test_build_visual_windows.py`、`tests/test_visual_pipeline.py`、
  `tests/test_download.py`、`tests/test_invariants.py`；
- 证据：
  - 窗口重建日志 `logs/build_visual_windows_full.log`（252 个窗口序列，9446 窗口，0 失败，16.77s）；
  - 权重审计 `logs/v0431_weight_audit.json`；
  - 重建统计 `logs/v0431_rebuild_stats.json`（本报告 §4 的独立复核）；
  - 空间对比：
    `logs/space_compare_dense_visual_clip_v1_vs_dense_visual_clip_v2__*.json`、
    `logs/space_compare_dense_visual_dino_v1_vs_dense_visual_dino_v2__*.json`、
    `logs/space_compare_full_mean_v1_vs_dense_visual_clip_v2__*.json`、
    `logs/space_compare_full_mean_v1_vs_dense_visual_dino_v2__*.json`（文件名含 hash，glob 匹配）；
  - DB：schema v5、visual_window_plans/visual_window_sequences；v2 表征 id 6/7、地图 map_v015/map_v016。
- 由 t4 独立复核并回填 §2/§4/§5/§7 数字；全量 QA 见最终交付报告。
