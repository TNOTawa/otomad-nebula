# ADR-0005：本地视频与 Bilibili 条目的身份关联，以及星云去重

- 状态：已采纳（迭代 v0.2）
- 日期：2026-10-25
- 相关代码：`nebula/identity.py`、`nebula/db.py` 的 work_links / `nebula/mapplot.py` 的 duplicate 排除、`ingest.py link-local`
- 相关契约：`docs/CONTRACTS_V2.md` 第 1、5 节

## Decision（决策）

同一音MAD作品可能既有本地视频文件，又有 Bilibili 条目（BV 号），它们是同一实体的两个身份。
系统用 `work_links` 表做**单层身份关联**：

- canonical（规范实体）是星云/榜单/界面默认展示的实体；
- duplicate（重复实体）保留作品记录、原始文件与已算好的向量，但默认**不进入星云、近邻索引与界面**；
- 一个 duplicate 只能关联一个 canonical；canonical 自身不能是任何人的 duplicate（禁止链式关联）；
- 自动关联条件：标题规范化（去空白/统一全半角括号引号冒号感叹号/casefold）后**完全相等**，
  且本地时长与 Bilibili 元数据时长差 ≤ 3 秒；拿不准一律不合并；
- 关联时可把 local duplicate 已有的当前版本向量/片段计划**复制登记**到 canonical，
  避免为同一作品重复下载/重复编码；不删除本地视频与 local 的向量；
- 支持 `input/lists/local_bv_links.csv` 手动覆盖清单（`ingest.py link-local --manual`）。

## Why（理由）

1. 同一个作品出现两次会让星云出现“双胞胎点”，扭曲近邻统计、密度与版本对比；
2. 直接删除/硬合并会丢失来源信息（本地文件路径、BV 号、榜单出现记录），且难以撤销；
3. 软关联 + canonical 集合既保证展示层去重，又保留全部原始数据与可追溯性；
4. 标题完全相等 + 时长双重校验是保守策略：自动错并会污染整个星云，宁可留给人工清单；
5. 复制向量能让“先有本地视频、后导入选票”的常见路径省掉一次下载与推理。

## Options（备选方案）

| 方案 | 优点 | 缺点 |
| --- | --- | --- |
| A. 单层 canonical/duplicate 软关联（选中） | 展示去重、数据不丢、可撤销、可审计 | 代码需要处处区分 canonical 集合 |
| B. 直接删除 duplicate 记录 | 实现最简单 | 丢失来源/文件/榜单记录，不可恢复 |
| C. 完全保留两个实体 | 无额外逻辑 | 星云双点、近邻统计失真、必须人工处理 |
| D. 模糊标题自动合并 | 覆盖率高 | 误合并风险高，污染不可逆 |

## Chosen（选择）

方案 A。数据库层用 `work_links` 唯一定义 duplicate；`get_duplicate_work_ids()`、
`list_works_for_list(only_canonical=True)`、mapplot 的 `include_duplicates=False`、
SimilarityIndex 与 appdata 的 `is_canonical` 默认排除 duplicate；过滤星云可显式
`include_duplicates=true` 生成对照版本（filter_json 记录该选择）。

## Consequences（后果）

- 所有“默认星云”相关的代码路径必须使用 canonical 集合；新增查询接口时需要显式说明是否排除 duplicate；
- duplicate 的 `embeddings` 与文件保留，会占用磁盘但可被 unlink 后再利用；
- 关联是单层的，跨榜单/跨年份出现仍走 `appearances`，不会形成链；
- 自动关联只覆盖“标题完全相等 + 时长接近”的高置信场景，剩余候选留给人工清单，属于产品折衷。

## How verified（如何验证）

- `tests/test_db.py`：work_links 唯一性、canonical 不能是 duplicate、unlink 语义；
- `tests/test_identity.py`：标题规范化边界、时长不符拒绝、手动清单、复制向量后
  `has_embedding` 为真、不删除原文件；
- `tests/test_invariants.py`：单层关联不变量、duplicate 默认不出现在榜单查询但数据保留；
- 独立验收（tests/test_e2e.py）：关联后默认地图/近邻索引不包含 duplicate，`include_duplicates`
  对照地图才包含；AppTest 界面默认不展示 duplicate。
