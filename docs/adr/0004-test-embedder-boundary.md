# ADR-0004：测试用 HashTestEmbedder 的边界

- 状态：已采纳
- 日期：2026-10-25
- 相关代码：`nebula/embed.py`（`HashTestEmbedder`）、`nebula/pipeline.py`
  （`allow_test_embedder`）、`pipeline.py --test-embedder`
- 相关契约：`docs/CONTRACTS.md` 第 3、6、11 节

## Decision（决策）

提供一个仅用于测试/演示的确定性嵌入器 `HashTestEmbedder`：

- `version = "test-hash-v1"`、`is_test_only = True`，不依赖 torch / transformers / 网络；
- 输入 `(视频路径, 音频路径)` 对，基于文件内容的哈希生成固定维度、逐行 L2 归一化的向量，
  同一输入永远得到同一输出，不同内容得到不同向量；
- 只有显式允许时才可进入流水线：`Pipeline(allow_test_embedder=True)` 或
  `pipeline.py --test-embedder`；默认配置下使用它会报错并给出中文提示；
- 其向量**永远不能**用于真实观察/发布数据；E2E 与离线演示使用临时目录与临时数据库。

## Why（理由）

1. 默认测试必须离线、快速、确定性：真实 PE-AV 权重约 3.1GB、CPU 推理慢，
   不适合每次 QA 都加载；
2. E2E 要验证的是“流水线、数据库、文件、星云、界面”之间的连接与不变量，
   而不是模型语义质量；用确定性哈希嵌入器可以独立断言形状、范数、去重、断点续跑、
   再次运行不重编码等行为；
3. 若用 mock 返回随机向量，测试不可复现；若用真实模型，CI 与无网机器无法运行；
4. “仅测试”必须通过接口显式暴露（`is_test_only`、`allow_test_embedder`），
   避免误把测试向量写进正式数据库。

## Options（备选方案）

| 方案 | 优点 | 缺点 |
| --- | --- | --- |
| A. HashTestEmbedder（选中） | 无依赖、确定性、快、可离线 | 向量无语义意义 |
| B. mock 返回固定/随机数组 | 实现简单 | 随机不可复现；固定数组无法区分输入 |
| C. 真实 PE-AV 小模型做测试 | 语义真实 | 慢、需权重、CI 不可行 |
| D. 预生成 golden 向量文件 | 真实输出可复用 | 素材/版本绑定，换环境易碎，且需要真实模型生成 |

## Chosen（选择）

方案 A：哈希嵌入器 + 显式开关。E2E 使用 `Pipeline(allow_test_embedder=True)` 并注入
`HashTestEmbedder`；默认测试中它是唯一不触网/不加载权重的嵌入器。
真实模型测试单独用 `pytest -m model` 标记，在具备权重的环境运行。

## Consequences（后果）

- E2E 只能断言结构与行为不变量（维度一致、范数、数量、确定性、状态机、渲染成功），
  不能断言“相似作品距离更近”之类的语义；
- 使用测试嵌入器时 CLI 会打印警告；若有人把测试向量写进正式 data 目录，
  版本号 `test-hash-v1` 可被识别与清理；
- 真实模型路径仍需 `pytest -m model` 与人工 P0 验收覆盖，测试嵌入器不能替代；
- 新增嵌入器实现必须继续满足 `Embedder` 协议（`version`、`is_test_only`、
  `encode_pairs`、`close`），以便流水线统一处理。

## How verified（如何验证）

- `tests/test_embed.py`：HashTestEmbedder 形状/范数/确定性/不同输入不同向量；
- `tests/test_e2e.py`：注入测试嵌入器跑通“合成视频 → 流水线 → UMAP → appdata →
  Streamlit AppTest”，并验证重复处理不重新编码；
- `tests/test_invariants.py`：`encode_pairs` 只接收路径对（元数据不进入编码）、
  `is_test_only` 语义与版本号可识别；
- 真实模型测试用 `pytest -m model` 独立验证，与测试嵌入器互不替代。
