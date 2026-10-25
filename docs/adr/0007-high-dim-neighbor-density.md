# ADR-0007：高维近邻密度，禁止用 UMAP 平面距离近似

- 状态：已采纳（迭代 v0.2）
- 日期：2026-10-25
- 相关代码：`nebula/similarity.py`、`nebula/appdata.py` 的 density_knn / `app.py`
- 相关契约：`docs/CONTRACTS_V2.md` 第 2、3 节

## Decision（决策）

“高维近邻密度”（density_knn）定义为：对每部作品，取其**高维联合向量**（work_joint，
必要时可切换 audio/video 模态）与其它 canonical 作品的余弦相似度，排序后取第 5 近邻的
余弦距离 `1 - cos` 作为该作品的局部稀疏度（值越大越稀疏）。

- 只使用 `SimilarityIndex` 的高维余弦，**禁止使用 UMAP 2D/3D 平面距离**；
- 原有“不染色”的二维网格密度保留为独立选项（`none`），两者语义不同；
- 近邻索引排除 duplicate 作品，按 canonical 集合计算；
- 矩阵在进程内缓存；界面用 `st.cache_resource` 按 DB 文件 mtime + 向量版本缓存；
- 模态缺失时给出“无该模态向量”，不参与密度/近邻计算。

## Why（理由）

1. UMAP 只近似保持邻域关系，不保持距离度量；平面上的远近不能解释为语义相似度；
2. 高维余弦距离直接对应模型嵌入空间的真实关系，与 Top-k 近邻列表、相似度值一致；
3. 第 5 近邻距离对局部密度敏感，比“到所有点的平均距离”更抗全局尺度影响；
4. 二维网格密度会受投影形变与画布范围影响，只适合观察平面聚集形态，不能作为语义密度。

## Options（备选方案）

| 方案 | 优点 | 缺点 |
| --- | --- | --- |
| A. 高维余弦 kNN 距离（选中） | 语义正确、与近邻一致、O(n²) 对当前规模可接受 | 需要读高维矩阵；单模态缺失需处理 |
| B. UMAP 平面 kNN 距离 | 实现简单、无需高维数据 | 距离不可解释，误差来自降维 |
| C. 2D 直方图/网格密度 | 快速、直观 | 受画布与投影形变影响，非语义密度 |
| D. Local Outlier Factor 等 | 更复杂 | 参数多、解释成本高，超出本轮需求 |

## Chosen（选择）

方案 A：SimilarityIndex.kth_neighbor_distance(k=5)，appdata apply_color(field="density_knn",
similarity_index=...) 生成连续染色；二维网格密度保留为 `none`。

## Consequences（后果）

- 打开近邻密度染色需要加载高维矩阵（首次点击有一次读 npy 成本，之后缓存）；
- 作品数增长到数千时 O(n²) 距离开销需要优化（分块/近似索引），本轮 33–100 部足够；
- 密度是相对当前 canonical 集合的统计量：不同筛选/星云版本的数据集不同，密度值不可直接横向比较；
- 缺失模态必须显式提示，不能静默用 0 向量或平面距离代替。

## How verified（如何验证）

- `tests/test_similarity.py`：近邻排序、排除自身、k 截断、缺失模态、duplicate 排除、
  缓存不重复读文件、第 k 近邻距离与手算余弦一致；
- `tests/test_invariants.py`：similarity 模块不导入 UMAP/界面库（保证不用平面距离）；
- 独立验收（tests/test_e2e.py）：用合成向量手算每部作品的第 5 近邻余弦距离，与
  density_knn 的染色值/接口输出逐项对比；AppTest 中密度染色可渲染。
