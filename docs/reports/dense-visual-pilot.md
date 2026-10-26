# 密集视觉语义编码 pilot 报告（v0.4.3 第一步）

- 日期：2026-10-26
- 状态：pilot 完成，等待是否全量运行的决定
- 相关设计：docs/plans/v0.4.3-dense-visual-semantics.md
- 原始数据：logs/dense_visual_pilot.json、logs/dense_visual_pilot_analysis.json、
  logs/dense_visual_decode_probe.json、logs/dense_visual_pilot_run.log
- 代码：nebula/dense_visual.py、tools/dense_visual_pilot.py、tests/test_dense_visual.py

## 1. 目的

v0.4 审查确认：当前只有 PE-AV 每 8s/16 帧（约 2fps）的稀疏联合语义，加上 30fps 96x54 灰度事件统计，
没有逐帧视觉语义。本 pilot 回答两个问题：

1. 用冻结视觉编码器按更高帧率提取帧级语义，工程上是否可行（吞吐/显存/存储）？
2. dense visual 是否携带 PE-AV 之外的信息（窗口结构、帧级动态）？

## 2. 方法

- 样本：3 部本地真实作品（原始视频保留，无需重新下载）：work 7（29.87s）、work 1（101.45s）、
  work 8（252.65s，含较多静态/LIVE 画面）。
- 编码器（冻结）：openai/clip-vit-base-patch16（512d，图文对齐语义）、
  facebook/dinov2-small（384d，自监督视觉结构）。
- 帧率：8fps 为主；对 work 7 加测 30fps。
- 流程：单条 ffmpeg 流式解码（fps,scale=224:224,format=rgb24）→ 冻结编码器 batch=32、bf16 →
  逐帧 L2 归一化 → 按 v0.4 全覆盖 8s 窗口做重叠时长加权池化 → 保存 dense_visual.npz + manifest。
- 对照：v0.4 全片序列的 PE-AV clip_video / clip_joint 窗口向量（同窗口对齐）。

## 3. 结果

### 3.1 工程

| 指标 | CLIP ViT-B/16 @8fps | DINOv2-small @8fps |
| --- | --- | --- |
| 单部墙钟（30s/101s/253s 视频） | 2.46 / 7.14 / 17.98s | 1.57 / 5.64 / 14.89s |
| 编码吞吐（work 8） | 约 145 帧/s | 约 184 帧/s |
| GPU 峰值（batch 32 bf16） | 782MB | 208MB |
| 全量 110 部（9.06h 视频）估算 | 约 39min | 约 32min |
| 存储（253s 作品，npz fp16 压缩） | 447KB | 278KB |
| 30fps 成本（work 7） | 帧数 896（3.75x）、墙钟 4.91s、存储 810KB | 未测 |

注：全量还包含 87 部 Bilibili 视频重新下载（v0.4 实测 download 均值约 5.5s/部，长视频另计）。

### 3.2 信息价值

- 8fps 窗口级与 30fps 几乎一致：work 7 的窗口池化向量 8fps vs 30fps 余弦 mean=0.9999、min=0.9999。
  窗口级表征无需 30fps；30fps 的价值在帧级轨迹。
- 帧级动态有增量：窗口内每秒相邻帧变化（1-cos）：8fps 0.170，30fps 0.271（1.59x）。
- 与 PE-AV 部分重合但不同：窗口自相似矩阵 RSA（dense vs PE-AV video）：
  work 7 0.985（4 窗口）、work 1 0.661（13 窗口）、work 8 0.947（32 窗口）；
  vs joint：0.958 / 0.579 / 0.927；95 个相邻窗口变化相关 Pearson=0.70。
- 编码器互补：CLIP 窗口间自相似高（work 1 mean 0.853），偏全局语义；
  DINOv2-small 更低（0.648，std 0.215），对画面/构图变化更敏感。

## 4. 结论

1. 工程可行：两个编码器都可在 RTX 3060 上全量运行（<1.5h，显存 <0.8GB，存储 <0.5GB）；
2. 8fps 足够支撑窗口级对照；30fps 仅作为帧级时序研究的小样本选项；
3. dense visual 与 PE-AV 的时序结构有可测差异，DINOv2 的变化敏感度明显高于 PE-AV video；
   值得作为独立通道进入全量，并作为 v0.5 分层时序模型的输入；
4. 仍不宣称逐帧语义理解：本 pilot 只提供帧级冻结语义向量，未训练任何时序模型。

## 5. 下一步（待确认）

- 全量 110 部 x {CLIP, DINOv2} @8fps → data/visual_dense/<visual_version>/；
- 30fps 小样本（10–20 部）用于帧级对照；
- 构建 dense_visual_v1 窗口级表征与星云对照；如需融合 PE-AV 语义，另建新表征版本；
- 人工抽查若干窗口的语义合理性后，再进入 v0.5 训练。
