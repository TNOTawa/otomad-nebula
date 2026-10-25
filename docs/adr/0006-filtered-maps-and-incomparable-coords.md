# ADR-0006：筛选星云独立成版本，坐标只在同一版本内可比

- 状态：已采纳（迭代 v0.2）
- 日期：2026-10-25
- 相关代码：`nebula/mapplot.py`（build_map / build_map_variants / filter_spec）、`map.py`
- 相关契约：`docs/CONTRACTS_V2.md` 第 1、4 节

## Decision（决策）

星云支持按数据集筛选构建独立版本：`dataset`（all/local/bilibili）、`vote_year`、
`list_key`、`include_duplicates`（默认 false）。筛选条件在构建时下推到训练集：

- 只用筛选后的 canonical 作品重新训练 UMAP（2D/3D 各自独立），生成新的 `map_vNNN`；
- 筛选条件以 `filter_json` 落到 `map_runs`，同时写入该版本 `meta.json` 与 params；
- 每个版本保存自己的坐标、UMAP 模型与参数；界面按版本切换，并明确显示该版本的筛选条件；
- **筛选星云与全量星云的坐标不可直接比较**；不同筛选星云之间同样不可比绝对位置。
  `map.py build --n-neighbors` / `map.py variants` 生成的 UMAP 参数变体也遵守同一规则。

## Why（理由）

1. UMAP 是非线性降维，训练集变化会整体改变嵌入布局；把筛选后的点“投影”进旧布局
   会得到看似可比、实际语义不一致的坐标；
2. 只在前端隐藏点、沿用全量坐标，会让密度/邻域观察仍然基于被筛掉的数据，结论错误；
3. 版本化独立训练保留可审计性：每个图都能回答“用了哪些作品、哪些参数、哪版向量”；
4. 迭代 A 只需要“观察不同数据子集的空间形态”，不要求跨图绝对坐标对齐，独立版本足够。

## Options（备选方案）

| 方案 | 优点 | 缺点 |
| --- | --- | --- |
| A. 筛选子集独立重训新版本（选中） | 语义正确、可审计、实现简单 | 每个筛选组合需要一次 UMAP 构建 |
| B. 前端隐藏点，坐标沿用全量图 | 无需重建 | 密度/邻域被隐藏数据污染，误导观察 |
| C. 用保存模型 transform 子集到全量空间 | 坐标与全量图同空间 | 子集布局不反映子集结构，且与训练集定义不符 |
| D. 固定全局坐标后叠加筛选 | 单图复用 | 无法观察子集内部结构 |

## Chosen（选择）

方案 A，并显式记录 filter_json；默认排除 duplicate。`map.py variants` 可一次生成
`n_neighbors=8/15/30` 等参数变体作对照，每个变体都是独立版本。

## Consequences（后果）

- 版本数量会随筛选组合增长；需要 `map.py list` 与界面版本选择器区分，定期清理时须同时删 DB 行；
- 文档、界面与 CLI 输出都必须提示“跨版本坐标不可比”，避免用户误读；
- 筛选后作品数 < 3 时无法训练 UMAP，直接给出中文 MapError 而不是生成空图；
- 全量星云仍是默认最新版本；筛选版本是并列的独立资产，不覆盖全量图。

## How verified（如何验证）

- `tests/test_mapplot.py`：filter_spec 下推、n_works/roundtrip、作品不足报错、变体版本递增；
- `tests/test_invariants.py`：filter_json 原样落库、无筛选为 null；
- `tests/test_e2e.py`：同一作品在全量与筛选版本中的坐标不同、filter_json 正确、
  AppTest 能渲染筛选版本并显示对应作品集合。
