# ADR-0010：全片覆盖序列与表征分层（v0.4）

- 状态：已接受（v0.4 开发中，2026-10-25）
- 相关：ADR 0002（片段采样规则）、ADR 0001（向量版本）、`docs/plans/v0.4-full-coverage-design.md`
- 相关代码：`nebula/sampling.py`、`nebula/pipeline.py`、`nebula/events.py`、`nebula/representations.py`、`nebula/temporal.py`、`tools/build_representation.py`

## Decision（决策）

### 1. 新增独立的 `plan_full_coverage()`，不修改 `plan_clips()`

旧 `plan_clips()` 的语义是“在整条时间轴均匀选最多 N 段”，不是“连续覆盖”；直接把
`clips_per_work` 调大既不能保证尾段，也不能保证时间间隙被覆盖。全片计划采用几何均匀步长：

- `n = ceil((duration - clip) / (clip - overlap)) + 1`，`step = (duration - clip) / (n - 1)`；
- 第 i 个窗口起点为 `i * step`，最后一个窗口精确锚定 `duration - clip`；
- 所有窗口长度 = clip（整片短于 clip 时退化为单窗口），因此无未覆盖间隙、无越界、无短尾填充；
- 每个窗口权重 = 对 `1 / 覆盖计数(t)` 的积分并归一，保证 ∑weight = 1 且时间轴上每一秒的
  总权重为 1，避免尾段重叠被重复加权。

### 2. 序列层与表征层分离

- **序列层**（`data/embeddings/<sequence_version>/`）：PE-AV 有序片段向量、片段窗口时间与
  权重、高频事件文件。编码一次，长期复用。
- **表征层**（`data/representations/<name>/<sequence_version>/`）：不同聚合器生成的作品
  向量（Full Mean / Temporal Stats / 未来 Transformer）。聚合器变化只新增表征，绝不重新
  下载/解码/编码 PE-AV。
- 版本号：`Config.embedding_version` 按 `sampling.mode` 返回激活的序列版本；
  `Config.legacy_embedding_version` 保留旧公式；旧数据不迁移、不覆盖。

### 3. 片段级持久化与恢复

每个 batch 编码完成后原子写 `.partial/batch_{k}.npz` 并更新 `sequence_states`；进程中断后
只补编码缺失批次。只有全部批次完成才写最终 npy、登记 embedding、标记 completed、清理临时
文件；部分失败不污染正式向量。

### 4. 高频事件版本独立于序列版本

事件扫描（`events_v1`）的算法版本记录在事件文件、`event_manifest.json` 与表征参数中。
改进事件算法时可以只重扫事件/重建表征，而不改变 PE-AV 序列版本；代价是若原始视频已删除，
重扫需要重新下载（本地作品始终保留原文件）。

### 5. 地图绑定表征

`map_runs` 增加 `representation_id`，新地图只能由某一个表征构建；不同表征/不同序列的向量
禁止混入同一 UMAP 输入，地图坐标不跨表征比较。旧地图在 schema v3 迁移时回填到对应
`legacy_5clip_mean` 表征。

### 6. 事件扫描不把“高同步”定义为更好

音画事件关系只记录时间相关、偏移、重合率等结构，不评分、不排序；刻意错位、反差与延迟呼应
同样是音MAD结构的一部分。

### 7. 336p 预处理作为新序列版本实验

`media.clip_height=336` 会改变向量，因此必须体现在序列版本号（`_h336`）中，先在 10 部
作品上与原生分辨率对比耗时与余弦差异，不自动转正。

## Why（理由）

1. 现有 5×8s 表示最多覆盖 40 秒，无法回答“整部作品的展开方式是否形成新邻域”；
2. 只扩大片段数会引入尾部/间隙等边界错误，必须由独立算法定义“完整覆盖”；
3. 编码一次的成本远高于聚合；把聚合器与编码解耦，后续 Transformer/统计实验不必重新下载解码；
4. 事件扫描能补充 PE-AV 16 帧（约 2 FPS）看不到的快速剪辑、闪帧与瞬态，但必须版本化、
   可复现，且不能宣称逐帧语义理解；
5. 无标签发现要求元数据不进入向量；序列/表征/地图三层版本隔离可以机械检查，避免混用。

## Options（备选方案）

| 方案 | 优点 | 缺点 |
| --- | --- | --- |
| A. 新增 plan_full_coverage + 序列/表征分层（选中） | 覆盖完整、可恢复、聚合器可迭代、旧数据不变 | 改动多，需要新 schema/目录与更多测试 |
| B. 直接把 clips_per_work 从 5 调大 | 改动最小 | 语义仍是“均匀抽样”，不能保证尾部与间隙，且片段分布随 N 变化不可控 |
| C. 全片逐帧特征 + 聚类 | 时间分辨率最高 | 存储/计算成本高，且与现有 PE-AV 语义向量不可比 |
| D. 先在旧 5 片段上训练 Transformer | 无需改流水线 | 输入只覆盖 40s，时序信息本身不完整，结论不可用 |

## Chosen（选择）

方案 A：

1. v0.4 先实现全片覆盖编码与 Full Mean 表征，验证“完整覆盖”单一变量的影响；
2. v0.4.1 增加 `events_v1` 全片高频扫描，并把事件特征按 8s 窗口压缩；
3. v0.4.2 增加不训练的时间结构聚合（Full Mean / Temporal Stats 双空间 + 邻域迁移分析）；
4. v0.5 自监督 Transformer 只读取已保存的序列/事件特征，待全片数据可靠后再训练；
5. 旧版本向量、旧片段与旧地图保持只读，app 通过地图绑定的表征切换空间。

## Consequences（后果）

- 全片编码成本约为旧的 4–5 倍（片段数约 23 vs 5），需要后台批次运行与片段级恢复；
- 事件文件约 0.5–1MB/部作品；表征向量约几十 KB/部作品；
- Bilibili 作品若原视频已删除，全片重编码需要重新下载；本地作品始终保留原文件；
- schema v3 迁移需要同时维护旧地图与 legacy 表征的兼容读取；
- 不同表征/序列的 UMAP 坐标不可直接比较，跨空间分析必须用高维近邻而不是二维位置；
- v0.5 模型训练质量在数据规模扩大前不作为结论。

## How verified（如何验证）

- `tests/test_sampling.py`：1s–600s 参数化覆盖完整性、末尾锚定、无间隙/越界、权重和为 1、
  每秒总权重为 1、确定性、非法参数；
- `tests/test_pipeline.py`：full 模式片段数/时间、批中断后续跑只补缺失批、无音轨、超长视频
  显式跳过/允许、legacy 回归；
- `tests/test_events.py`：受控闪帧/音脉冲检出、窗口对齐、确定性、无音轨、保存加载 roundtrip；
- `tests/test_temporal.py`：Full Mean 与旧聚合公式一致；片段顺序反转时 Full Mean 距离≈0，
  Temporal Stats 距离显著 > 0；
- `tests/test_core_v04.py` / `tests/test_invariants.py`：schema v2→v3 迁移、旧地图回填、
  版本/表征/地图隔离；
- `tests/test_e2e.py`：离线全链路（全片编码→恢复→Full Mean 表征→地图→appdata→
  compare_spaces）；
- 真实数据：120 部作品全片编码后检查覆盖率/恢复/耗时，构建 Full Mean 与 Temporal Stats
  地图，并用 `tools/compare_spaces.py` 输出 Top-k 邻域迁移；最终 `python qa.py --full`。
