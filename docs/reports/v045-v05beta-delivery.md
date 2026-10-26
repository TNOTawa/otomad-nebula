# v0.4.5 / v0.5-beta 交付报告

> 本文件合并记录本轮三阶段（工程收敛 → v0.4.5 → v0.5-beta）的真实结果与限制。
> 状态：进行中（Phase 0 完成；后续阶段追加）。

## Phase 0：工程审计与收敛（G0）

### 基准

- 起始提交：`47904d9`（本地与 GitHub main 一致）
- 工作树：干净；无未提交修改
- `python qa.py --full`：**PASS 5/5**，pytest **951 passed, 1 skipped**，总耗时 238.69s
- 真实库：schema v5，无未完成迁移；默认地图 map_v012；7 空间可切换；共识 JSON 可读

### 审计表

| 模块/功能 | 当前用途 | 实际调用入口 | 问题 | 处理决定 |
| --- | --- | --- | --- | --- |
| `nebula/highdim.py` | 新增共享行归一化/余弦矩阵/Top-k 邻域 | consensus、diagnostics、v0.4.5 | 两工具原先各有一份数学实现 | **合并**（行为由原测试锁定） |
| `tools/build_consensus_graph.py` | 跨模型共识/融合 | qa CLI、测试、App 数据 | 本地 normalize/cosine/topk 重复 | 改为 import highdim；旧名保留别名 |
| `tools/embedding_diagnostics.py` | PCA/UMAP 诊断 | qa CLI、测试、App 链接 | 同上（距离升序语义不同） | 改为 import highdim；旧名保留别名 |
| `tools/compare_sequences.py` | 旧序列版本对照脚本 | 无 CLI/测试/文档引用 | 无活跃入口、无历史契约 | **删除**（Git 可恢复） |
| `nebula/v05_*.py`、`tools/*v05*` | v0.5-alpha 实现 | v0.5 报告/产物 | alpha 负结果必须保留 | 保留；beta 复用而非复制 |
| `tools/bench_*.py`、`verify_peav.py` | 历史性能/真实模型证据 | README/ARCH/CONTRACTS 引用 | 有文档契约 | 保留 |
| `data/sources` | Bilibili 源缓存 | 下载流水线 | 当前无缓存视频（仅占位文件） | 保留机制；本轮不重复下载 |
| 配置项 | config.yaml | 全部 Python 代码 | 无未引用叶子配置 | 保留 |

### 收敛结果

- 新增 `nebula/highdim.py`，删除两个工具中的重复数学实现；旧内部名以别名保留，调用点/测试不破坏。
- 删除无引用的 `tools/compare_sequences.py`（221 行）。
- 未进行大范围重构：v05 研究代码、共识/诊断的空间业务逻辑与 UI 数据层职责不同，强行合并会增加风险。

### 兼容风险

- `normalize_vectors` 的错误文案由“无法计算余弦距离”统一为通用文案；仅错误文本差异，无行为差异。
- 旧工具模块属性 `normalize_vectors`/`topk_neighbors`/`cosine_distance_matrix` 仍可通过导入名访问，测试已覆盖。
