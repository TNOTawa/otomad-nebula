# ADR-0009：近邻解释——相对分位、互为 Top-k 与 hubness

- 状态：已采纳并实现（v0.2.x，2026-10-25）
- 日期：2026-10-25
- 相关代码：`nebula/similarity.py`、`nebula/appdata.py`、`app.py`（T2 负责）
- 相关契约：`docs/CONTRACTS_V2.md` 第 2、3 节；`docs/adr/0007-high-dim-neighbor-density.md`

## Decision（决策）

Top-k 近邻列表不再只显示裸余弦值，而是同时给出三种“解释性”信息：

1. **相对分位 `pair_percentile`**：该近邻相似度在“当前 canonical 集合全部成对相似度”
   分布中的百分位（0–100）。例如 cos=0.62 若位于第 85 百分位，说明它比全库 85% 的作品对更相似，
   而不是用绝对阈值判断“强相似”；
2. **互为 Top-k（`mutual`）**：A 的 Top-k 包含 B，且 B 的 Top-k 也包含 A 时标记为互为近邻；
   单向命中比互为命中更弱，展示层要区分；
3. **Hubness（`hubness`）**：统计每个作品在“其他作品 Top-k 列表”中出现的次数（入度）。
   高维空间里少数点会成为很多作品的近邻（hub），这是嵌入空间的普遍现象，不是作品问题；
   展示层给出 Top hub 列表/入度分布（最大值、中位数），帮助理解“很多作品都挨着同几个点”。

实现上复用 `SimilarityIndex` 已加载的每个模态矩阵：成对相似度矩阵只计算一次并缓存，
分位/入度均从同一矩阵推导；所有解释字段只影响展示，不改变排序、星云坐标与任何向量。

## Why（理由）

1. 小样本下 Top5 很容易被误读为“强相似”：在 56 部作品里，Top5 的余弦可能只有 0.5–0.6；
   用全库成对相似度分布做分位，能把“相对更近”与“绝对相似”区分开；
2. 单向邻居可能只是 hub 效应或阈值噪声，互为 Top-k 提供更强的双向证据；
3. hubness 是高维嵌入的已知现象（hub 点吸引大量近邻），如果不解释，用户会误以为
   “某作品和谁都像”是内容结论；把它显式展示为统计现象更诚实；
4. 这些解释字段比“绝对相似度阈值”更稳健：模型/数据规模变化时不需要重新调阈值。

## Options（备选方案）

| 方案 | 优点 | 缺点 |
| --- | --- | --- |
| A. 只显示裸余弦（现状） | 简单 | 小样本下易被误读为强相似 |
| B. 固定绝对阈值（如 cos>=0.8 才算相似） | 直观 | 阈值依赖模型/数据，跨版本失效 |
| C. 相对分位 + 互为 Top-k + hubness（选中） | 与数据规模自适应、信息完整、不改变排序 | 计算 O(n²)；需解释字段与文档 |
| D. 只做相对分位 | 实现最简 | 缺少双向证据与 hub 现象说明 |

## Chosen（选择）

方案 C，已实现：

- `similarity.Neighbor` 新增带默认值的字段 `pair_percentile: float = 0.0`、
  `is_mutual: bool = False`，旧的三参数构造保持兼容；
- `SimilarityIndex.pair_percentile(a, b, kind)`：定义
  `100 * count(其他无序对相似度 < 本对相似度) / 总无序对数`（严格小于；并列按“不低于”处理，
  因此精确并列时是该分位的下界）；
- `SimilarityIndex.neighbors(...)` 为每条近邻附带分位与
  `is_mutual`（目标作品是否也把当前作品列入同一 Top-k）；
- `SimilarityIndex.hubness(k, kind)` 返回每部作品被列入 Top-k 的次数（入度），
  `hub_top(k, n, kind)` 按 (次数降序, work_id 升序) 返回 Top hub；
- `appdata.neighbor_rows` 输出 `pair_percentile / is_mutual / in_map`；
  `appdata.hub_rows` 输出枢纽展示行；
- `app.py` 近邻表显示“相似度 + 全库分位 + 双向”，枢纽作品放在可折叠区域；
  分位/入度的计算集合是该 `embedding_version` 的全部 canonical 作品（不受界面筛选影响，
  筛选只改变当前视图），界面文案会说明这一点。

## Consequences（后果）

- 首次计算需要 O(n²) 相似度矩阵（当前 n≈60，开销可忽略；上千部需分块/近似，记为扩展点）；
- 相对分位是**数据集相对量**：不同 canonical 集合（全量 vs 2025 子集、不同筛选星云）之间
  分位不可横向比较，展示时必须带上当前集合上下文；
- hubness 只是描述性统计，不表示作品“质量差”或“太普通”，文档与界面措辞需保持中性；
- 互为 Top-k 依赖 k 的取值（默认 5）；k 变化时互为命中数会变化，报告需说明 k；
- 不改变任何坐标/向量/相似度排序，只增加展示字段，因此不影响既有测试与星云版本。

## How verified（如何验证）

- 单元测试（T2 的 `tests/test_similarity.py`）：用合成向量手算全库成对相似度矩阵，
  逐项校验 pair_percentile、互为 Top-k 标记、hubness 入度与 Top hub 排序；
- `tests/test_invariants.py`：解释字段不得改变近邻排序与前 k 个 work_id；字段仅作展示；
- 独立验收（`tests/test_e2e.py`）：临时 DB + 合成向量，用 numpy 独立计算分位/入度，
  与实现输出逐项对比；AppTest 检查近邻面板出现相对分位/互为/hubness 文案且不影响高亮与连线；
- 真实数据复核：在 map_v003 的 56 部 canonical 集合上抽查分位与入度，确认与手算一致。

## 实现细节（v0.2.x 已回填）

- 缓存：`_full_similarity_cache`（n×n 余弦矩阵）、`_pair_values_cache`
  （上三角无序对排序数组，用于 `np.searchsorted` 分位）、`_topk_cache`
  （按 kind,k 缓存 Top-k 集合）、`_clip_matrix_cache`（按 kind,work_id 缓存片段矩阵）；
  `close()` 释放这些缓存。
- 片段级探索（P2）：`clip_matrix(work_id, kind)`、`clip_times(work_id)`、
  `clip_similarity(a, b, kind)`、`best_clip_pairs(..., k)`、
  `explain_pair(..., threshold=0.5)`；纯数据层函数
  `appdata.clip_heatmap_data`（矩阵 + 含起始秒的行列标签 + 标题）与
  `appdata.clip_pair_rows`（含两侧片段序号/起始秒/余弦的展示行）。
- 界面：近邻面板三模态表（列：排名/作品/余弦相似度/全库分位/双向）、枢纽作品折叠区、
  片段级对应折叠区（勾选后渲染 Plotly heatmap + 最高片段配对表 + 解释统计，
  默认关闭以避免每次加载片段矩阵）。
- 边界：分位是相对当前 canonical 集合的量，跨集合不可比较；小样本下 hubness 不稳定；
  解释字段只读展示，不参与排序/坐标/向量写入（由 tests/test_invariants.py 约束）。

## 真实数据观察（2026-10-25，扩量后 110 部 canonical）

- work 1 的 joint Top5 相似度为 0.8625 / 0.8582 / 0.8562 / 0.8538 / 0.8481，对应全库分位
  82.5 / 80.7 / 79.6 / 78.5 / 76.1——即使库里最相似的几对也只处于“中高分位”，
  说明单纯看 Top5 余弦很容易高估相似强度；
- hubness Top：work 104 被 40 部作品列为 Top5、work 93 被 34 次、work 50/120 各 25 次；
  少数 hub 点会稳定出现在很多作品的近邻列表里，解释近邻时需要意识到这一点；
- 片段级：work 1 × work 93 的 5×5 联合片段热图最高 cos = 0.8095
  （A 片段 26.4s ↔ B 片段 97.1s），25 个片段对全部 > 0.5，可用于区分
  “作品整体相近”与“局部片段对应”。
- 以上数值只用于说明语义，会随数据规模与 canonical 集合变化，不作为固定阈值。
