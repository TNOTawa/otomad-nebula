# ADR-0003：星云版本化（map_vNNN）与 UMAP transform 增量更新

- 状态：已采纳
- 日期：2026-10-25
- 相关代码：`nebula/mapplot.py`、`nebula/db.py` 的 `map_runs` / `map_coords`
- 相关契约：`docs/CONTRACTS.md` 第 4.2、9 节；方案文档第五、六节

## Decision（决策）

1. 每次完整降维训练生成新的星云版本号 `map_vNNN`（单调递增），坐标文件、
   2D/3D UMAP 模型与参数元数据分别落盘到 `data/maps/<map_version>/`，
   并在 `map_runs` / `map_coords` 中原子登记；旧版本永久保留、可切换对比。
2. P3 持续更新时，优先用 `transform_new`：加载最新版本的 `map_2d.joblib` /
   `map_3d.joblib` 对缺失坐标的新作品调用 UMAP `transform()`，把新点放进固定空间；
   旧作品坐标保持不变，完成后整体写回 `map_coords` 与 parquet。
3. 只有累计新增较多、或主动需要重构空间时，才 `build_map` 重新训练生成下一个版本。
4. 失败时清理本次半成品目录与 DB 登记，不留下“有版本无坐标”的脏状态。

## Why（理由）

1. UMAP 不是可交换的坐标变换：加入数据重新训练会让所有历史作品的位置移动。
   方案文档明确要求“保留版本以便比较，不同版本坐标不能直接作绝对位置比较”，
   因此不能用一个固定目录反复覆盖。
2. 增量更新（transform）让网站可以频繁加入周报新作，同时让用户熟悉的旧点保持稳定；
   但 transform 的质量依赖已训练模型，数据量大到分布明显变化时需要重训。
3. 版本号落库 + meta.json 记录 `embedding_version` 与 UMAP 参数，才能回答
   “这张图基于哪版向量、哪些参数生成”，也便于复算。
4. 每个版本保存模型文件，使 transform 与历史对比成为可能；文件只增不改，天然支持回滚。

## Options（备选方案）

| 方案 | 优点 | 缺点 |
| --- | --- | --- |
| A. 每次更新都重新训练 | 空间始终反映最新整体分布 | 旧点移动，跨版本不可比；频繁重训成本高 |
| B. 只用 transform，永不重训 | 位置稳定、增量便宜 | 新数据分布漂移后投影失真；模型冻结 |
| C. 版本化重训 + P3 transform（选中） | 兼顾稳定与可比较；保留历史 | 需要版本管理、模型文件与清理策略 |
| D. 固定随机种子使重训可复现 | 同输入可复现 | 输入变化后仍会移动旧点，无法解决跨版本比较 |

## Chosen（选择）

方案 C：完整重训生成新版本；日常增量先用 transform；版本只增不改，界面默认最新版本。
2D 与 3D 分别训练独立模型（各自 `n_neighbors=min(cfg值, n-1)`、`random_state`、
`n_jobs=1`），因为同一模型不能输出两种维度。

## Consequences（后果）

- 磁盘上每个星云版本都保存 parquet + 两个 joblib 模型；版本多时需要手动清理旧目录
  （须同时删除 DB 中对应 `map_runs` 行，先备份）。
- 用户必须理解不同版本坐标不可直接比较；界面侧栏显示版本号并默认最新。
- transform 只对“已有模型”有效：若没有历史版本或模型损坏，只能 `build_map` 重训。
- 失败清理与原子登记要求先写文件后写 DB；半成品目录必须删除，避免 latest_map 指向坏版本。

## How verified（如何验证）

- 单元测试：`next_map_version` 单调递增；`create_map_run` 拒绝重复版本；
  `replace_map_coords` 整体替换并更新 `n_works`；失败路径不留下 ready 记录；
- E2E：`build_map` 后 `appdata.load_nebula_frame` 能得到与作品数一致的坐标，
  `AppTest` 可渲染；`transform_new` 后旧作品坐标不变、新作品获得坐标；
- 不变量测试：星云版本必须记录 `embedding_version`；地图登记与坐标在同一事务。

## 备注

P3 的“是否 transform 还是重训”是产品策略而不仅是技术选择：本项目默认 transform，
当新增量达到一定规模（或用户主动要求）再重训并保留旧图，避免无谓的旧点漂移。
