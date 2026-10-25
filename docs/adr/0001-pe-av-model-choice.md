# ADR-0001：采用 PE-AV 作为第一版音画联合编码模型

- 状态：已采纳（P0 起固定）
- 日期：2026-10-25
- 相关代码：`nebula/config.py`、`nebula/embed.py`、`config.yaml` 的 `model` 段
- 相关契约：`docs/CONTRACTS.md` 第 4.1、6 节

## Decision（决策）

第一版只使用一个多模态模型 `facebook/pe-av-small-16-frame`（PE-AV，16 帧输入）
提取音画联合向量；模型 revision 固定为 `9f888eea95c83622212bb742e91bf01d3b46fe96`，
权重缓存在项目内 `model.cache_dir`（默认 `.cache/hf`）。
模型标识、帧数与采样参数共同派生 `embedding_version`；升级模型只新增向量版本，
不覆盖旧版本数据。

## Why（理由）

1. 需求是“音画联合”向量：必须让同一次推理同时看到画面与声音，才能观察音MAD的
   音画组合相似性；分别用图像模型和音频模型再拼接无法表达跨模态关系。
2. PE-AV 是小规模（small）版本、16 帧输入，本地 CPU 也能在可接受时间内完成
   几十到上百部作品的编码，符合 P0/P1 的计算预算。
3. 使用 HuggingFace/transformers 生态，加载、缓存、离线复用成熟，便于固定 revision
   与缓存目录，满足可复现要求。
4. 方案文档明确“推荐起始检查点 `facebook/pe-av-small-16-frame`，后续可升级 Base”。

## Options（备选方案）

| 方案 | 优点 | 缺点 |
| --- | --- | --- |
| A. PE-AV small 16f（选中） | 原生音画联合嵌入；算力可控；官方权重 | CPU 仍偏慢；16 帧分辨率有限 |
| B. PE-AV Base / 更大检查点 | 语义质量更好 | 显存/时间成本高，P0 不合适 |
| C. 图像模型 + 音频模型分别编码后拼接 | 可用成熟单模态模型 | 缺乏跨模态交互；拼接方式任意；两套推理 |
| D. 自训练/微调模型 | 可针对音MAD优化 | 需要标注与训练资源，超出第一版范围 |

## Chosen（选择）

方案 A：`facebook/pe-av-small-16-frame`，revision 固定，16 帧，
joint/audio/video 三种嵌入同时保存（模型提供时），联合向量作为主坐标来源。
设备策略 `model.device: auto`（CPU float32，CUDA 可用时 bfloat16 autocast）。

## Consequences（后果）

- 推理依赖 torch/transformers/av，放在可选依赖组 `ml`，没有 GPU 或只做数据处理时
  可以不安装；默认测试不得加载模型。
- CPU 推理慢，P1 的 100 部作品需要串行跑较长时间；可在 P2 再优化批处理/GPU。
- 模型 fixed 16 帧意味着 `model.frames` 改变必须视为新向量版本。
- 升级模型（small → base）不会覆盖旧向量：新版本目录 + 新 `embedding_version`，
  旧星云与旧向量均可保留对比。
- 权重约 3.1GB，随仓库外缓存分发；离线机器需预先拷贝 `.cache/hf` 并开启
  `model.local_files_only`。
- **数值一致性边界**：同一设备/同一 dtype/同一 revision 下重复运行向量完全一致
  （实测 max abs diff = 0）；但 CUDA bfloat16 autocast 与 CPU float32 之间存在约 1%
  方向差异（实测 joint 余弦相似度约 0.9962）。因此 `embedding_version` 只编码
  模型 + 采样参数，**不编码 device/dtype**；跨硬件/跨 dtype 的坐标与距离只作趋势比较，
  不承诺 bit 级一致，严格对比需在同一硬件与 dtype 下重算。

## How verified（如何验证）

- `pytest -m model`：真实权重下编码 12 秒视频的 2 个片段，检查 joint/audio/video
  形状一致、逐行 L2 范数 ≈ 1、两次运行结果逐元素接近；
- 数值边界实测（P0 落机）：同设备同配置重复编码 max abs diff = 0；CPU fp32 与 CUDA bf16
  autocast 的 joint 余弦相似度约 0.9962（只记录趋势差异，不作为通过/失败断言）；
- P0 消融实验 `pipeline.py ablate`：静音/遮蔽画面后联合向量应发生变化，
  用于排查“模型事实上只依赖单一模态”的明显问题；
- `tests/test_invariants.py`：模型/采样参数变化必须产生新的 `embedding_version`；
- 失败与离线路径：权重缺失时给出中文错误提示，默认测试不依赖权重。
