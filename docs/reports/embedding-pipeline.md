# 音MAD 星云：嵌入流程报告（v0.3 数据快照）

> 本文说明当前架构**如何把一部视频变成一个向量**，以及这些向量如何保存、聚合与版本化。
> 面向：想理解/复用/审阅嵌入流程的人。代码入口见文末。
> 数据快照：2026-10-25，120 部有向量作品、586 个片段、720 个向量文件、embedding_version = peav-small-16f__f16_clip8s_x5_48k。

## 0. 一句话总结

对每部视频在时间轴上均匀截取最多 5 个 8 秒音画同步片段；每个片段分别把 16 帧画面与 48kHz 音频喂给 PE-AV small 模型，得到 1024 维的联合/音频/视频向量；片段向量逐行 L2 归一化后求平均再归一化，得到作品级向量；全部向量以版本化目录 + SQLite 登记的形式保存为可扩充的高维数据库，UMAP 只负责从这些向量投影出可看的星云。

## 1. 总览

~~~text
视频文件 / Bilibili 下载
      │
      ▼
ffprobe 探测（整部作品只探测一次）
      │  duration / has_audio / has_video
      ▼
时间轴采样 plan_clips            <-- 纯函数、无随机
      │  最多 5 段 × 8 秒；短视频自动减少段数
      ▼
ffmpeg 切片（每片段一条命令同时产出）
      │  clip_XX.mp4 : 仅视频轨 H.264/yuv420p
      │  clip_XX.wav : 48kHz / 2ch / pcm_s16le
      ▼
PE-AV Processor（16 帧 + 音频波形）
      │  画面 336×336，均匀取 16 帧（8 秒 ≈ 2 FPS）
      ▼
PE-AV Small 前向（batch=5，CUDA bf16 autocast / CPU fp32）
      │  outputs: audio_video_embeds / audio_embeds / video_embeds
      │  每类 1024 维，原始未归一化
      ▼
后处理：转 float32、逐行 L2 归一化
      │
      ├─ 片段级：clip_joint / clip_audio / clip_video  (n_clips, 1024)
      │
      └─ 作品级聚合：归一化 → 平均 → 归一化
            work_joint / work_audio / work_video      (1024,)
                  │
                  ▼
       data/embeddings/<embedding_version>/<work_id>/*.npy
                  + SQLite：embeddings / clips / embedding_versions
                  │
                  ▼
       UMAP 2D/3D（map_vNNN）→ Streamlit 星云（只读向量）
~~~

## 2. 模型与运行配置

| 项目 | 当前值 | 说明 |
|---|---|---|
| 模型 | facebook/pe-av-small-16-frame | 方案文档推荐的起始检查点；847M 参数 |
| Revision | 9f888eea95c83622212bb742e91bf01d3b46fe96 | 固定 revision，避免上游悄悄变化 |
| 输入帧数 | 16 帧 | 由该检查点决定，均匀采样 |
| 视觉输入 | 336×336，RGB，mean/std=0.5 | processor 内部缩放，不要求源视频分辨率 |
| 音频输入 | 48000 Hz | processor 读取 wav；不用 mp4 内音轨 |
| 向量维度 | 1024（joint / audio / video 同维） | 每次前向同时返回三类 |
| 设备 | auto：本机 CUDA（RTX 3060 12GB） | 无 CUDA 时回退 CPU fp32 |
| 精度 | CUDA：bf16 autocast；CPU：float32 | 数值有轻微设备差异，见第 7 节 |
| 批大小 | 5 片段/批 | 每部最多 5 段，正好一批 |
| 模型缓存 | .cache/hf（HF_HOME 布局，实际 .cache/hf/hub） | 代码自动识别 hub 子目录 |
| 加载 workaround | 修正嵌套 vision_config.num_labels=1024 | transformers 5.20.2 的配置解析坑，见 embed.py |

模型只做前向推理，不训练、不微调；第一版只有这一个多模态模型。

## 3. 第一步：时间轴采样（nebula/sampling.py）

输入整部作品时长 D，规则完全确定、无随机：

1. D ≤ 8s：整段作为唯一片段；
2. 否则片段数 n = min(5, floor(D / 8))，每段长度固定 8s；
3. 第 i 段中心位于 D × (i+0.5) / n，起始时间 = 中心 − 4s，再夹回 [0, D−8s]；
4. 起始时间四舍五入到毫秒，片段完全落在视频内。

结果：8 秒短片 1 段、20 秒 2 段、40 秒及以上 5 段；短视频不会硬凑 5 段。
当前 120 部作品的 586 个片段分布：5 段 × 113 部、4 段 × 1、3 段 × 5、2 段 × 1。

## 4. 第二步：音画同步切片（nebula/media.py）

每个片段用**一条 ffmpeg 命令**同时产出 mp4 与 wav：

- -ss 放在 -i 之前：音视频在同一个精确 seek 偏移上取出，保证时间对应；
- 视频：-map 0:v:0，-t 8s，libx264 / veryfast / crf18 / yuv420p / +faststart，且 -an（不编码 AAC）；
- 音频：-map 0:a:0，-t 8s，pcm_s16le / 48kHz / 2ch；源无音轨时用 anullsrc 补等长静音；
- 切片失败抛 MediaError 并带 ffmpeg stderr 摘要；源视频永不删除、不改动。

为什么还要重新编码而不是直接 -c copy：精确 seek + 统一解码输入 + 统一音频格式，能保证不同容器/编码的源在进入模型前一致；去掉 AAC 是因为模型只从 mp4 读画面、从独立 wav 读音频，这一步实测**零向量漂移**（见 8 节）。

## 5. 第三步：PE-AV 预处理（processor）

~~~python
inputs = processor(
    videos=[clip_00.mp4, ...],     # 每段一个视频文件
    audio=[clip_00.wav, ...],      # 与视频一一对应
    sampling_rate=48000,
    return_tensors="pt",
    padding=True,
)
~~~

- 视频侧：按模型配置从 8 秒内**均匀取 16 帧**（约 2 FPS），缩放到 336×336，归一化；
- 音频侧：读取 48kHz 波形并按 batch padding；
- 一部作品最多 5 段，正好一个 batch；padding=True 让不同长度片段同批处理。

这一步的输入是“重新编码后的 8 秒片段”，不是原始长视频；因此后续所有相似性都是“这 5 个采样片段”的性质，而不是整部作品的逐帧性质。

## 6. 第四步：前向与后处理（nebula/embed.py）

模型一次返回三类向量：

| 输出字段 | 含义 | 是否主坐标 |
|---|---|---|
| audio_video_embeds | 联合音画向量 | ✅ 星云主坐标 |
| audio_embeds | 纯音频向量 | 仅分析/近邻对照 |
| video_embeds | 纯视频向量 | 仅分析/近邻对照 |

原始输出未归一化（实测联合向量范数约 3.4），因此统一做：

~~~text
z_i_norm = z_i / ||z_i||_2                    # 每个片段逐行单位化
z_work   = normalize( (1/N) * Σ z_i_norm )    # 平均后再单位化
~~~

三类模态各自独立执行同样的聚合；片段级向量按原样（已归一化）保存，不丢弃。

**为什么平均**：一个作品只能有一个主坐标，平均是最简单、确定、可解释的聚合；代价是局部结构被平滑（一部作品里某一段特别像另一个作品，整体向量未必靠近）。这正是后来补充“片段级相似探索”的原因：整体 Top5 不适合解释局部相似，片段级 5×5 热图可以。

## 7. 保存、版本与可复现

### 7.1 保存结构

~~~text
data/embeddings/<embedding_version>/<work_id:06d>/
    work_joint.npy   (1024,)        作品级联合向量（星云用）
    work_audio.npy   (1024,)
    work_video.npy   (1024,)
    clip_joint.npy   (n_clips,1024) 片段级，后续研究/解释用
    clip_audio.npy   (n_clips,1024)
    clip_video.npy   (n_clips,1024)

data/catalog.sqlite
    embedding_versions : 模型/帧数/采样参数/聚合说明
    embeddings         : work_id, version, kind, path, dim, n_clips
    clips              : work_id, version, idx, start_s, duration_s
~~~

写入顺序：先原子写 npy（临时文件 + os.replace）→ 再登记 clips → 再登记 embeddings → 最后把作品状态置 completed；只有登记成功后才清理 temp。断点续跑时按 has_embedding 跳过；崩溃窗口有自愈逻辑。

### 7.2 向量版本

~~~text
embedding_version = <模型版本> + __f<帧数>_clip<片段秒>s_x<片段数>_<采样率>k
例：peav-small-16f__f16_clip8s_x5_48k
~~~

- 改模型、改帧数、改片段长度/数量/采样率 → 自动生成新版本目录，旧向量原样保留；
- 同一版本号登记不同参数会被拒绝，防止静默覆盖；
- device/dtype **不进入版本号**：同设备同配置重复编码实测 max abs diff = 0；CUDA bf16 与 CPU fp32 的联合向量余弦约 0.996，因此跨硬件只作趋势比较（ADR-0001）。

### 7.3 实测耗时（真实 54 部扩量批次，run_id=5）

| 阶段 | 单部平均 | 说明 |
|---|---|---|
| download_s | 6.21s | 仅 Bilibili 作品；本批次全部为 Bilibili |
| probe_s | 0.045s | 整部只探测一次 |
| extract_s | 2.80s | 5 段 × 约 0.56s |
| preprocess_s | 1.01s | processor 解码/采样/缩放/波形 |
| forward_s | 3.05s | batch=5，CUDA bf16 |
| save_s | 0.02s | 6 个 npy + DB 登记 |
| total_s | 13.53s | 单部累计；整批 730.5s / 54 部（含首个作品触发的模型加载摊销） |
| 模型加载 | 17.0s | 批次级一次（冷启动/磁盘争抢时可达 25–50s） |

明细见 logs/perf_005.csv，可用 pipeline.py timings 查看。

## 8. 多模态与同步敏感性（实验结论摘要）

- **去 AAC 音轨**：对真实 3 部作品的 4 类向量与旧实现逐文件 sha1 比较，**完全一致（max abs diff=0）**，因此“mp4 去音轨 + 独立 wav”是零漂移优化，已默认采纳。
- **单探测**：作品级 ffprobe 结果传入切片函数，片段不再重复探测；同样零漂移。
- **音画错位实验（10 部 / 48 片段，Δ=1−cos）**：
  - 普通再编码基线 Δ≈0.00096；
  - 音频 +0.5s：Δ≈0.00422（约 4.4× 基线）；
  - 音频 +1.0s：Δ≈0.00236（约 2.5×；周期性音乐整拍偏移反而更小）；
  - 换成同作品其他片段音频：Δ≈0.0876（约 91× 基线）；
  - 纯音频变体下 video 向量 Δ 严格为 0，joint 明显变化 → 联合向量确实融合两路，且对配对关系有可测敏感性，但绝对 Δ 很小，不能夸大为“理解同步”。
- **未采纳的候选提速**：preset veryfast→ultrafast 约 1.88×，但向量抽样 cos 0.99918；缩放到 336p 端到端约 2.36×，但 work joint 最低 cos 0.99812 → 都会改变向量，需新 embedding_version + 整批重跑，留给后续版本。

## 9. 当前数据规模（v0.3 快照）

- 作品：122 条记录 = 120 completed（33 本地 + 87 Bilibili）+ 2 failed（8.7h 活动直播回放、已删除视频）；
- 片段：586 条 clips；向量文件：720 个（120 × 6 类）；作品向量维度 1024、范数 0.99999994–1.0；
- 身份关联：10 条 work_links（本地与 B 站作品去重，duplicate 默认不进星云）；
- 星云：map_v001…map_v008；最近的总星云 map_v008 含 110 部 canonical 作品；
- 核心资产是 data/ 下的 SQLite + 向量，星云只是可再生的投影。

## 10. 已知局限与下一步方向

1. **时间分辨率**：8 秒 16 帧 ≈ 2 FPS，音MAD 的逐拍切片/闪帧/瞬态同步远快于此；模型没有看到的帧无法被后续恢复。
2. **片段平均**：作品级向量会平滑局部结构；片段级向量已保留，可做局部解释，但不适合直接当主坐标。
3. **预训练目标**：PE-AV 是通用音视频对比学习，不专门优化音MAD手法；实验显示它保留部分同步关系，但能力边界仍未知。
4. **样本量**：110 部时 Top5 只占约 4.6%，近邻解释需要结合全库分位/双向/hubness，而不是只看排名。
5. 后续候选（未决定）：新的 embedding_version（更高时间分辨率/不同片段策略）、336p 版本并行对比、音画结构自监督表征实验、超长视频分段下载。

## 11. 代码入口

| 环节 | 文件 |
|---|---|
| 采样规则 | nebula/sampling.py |
| 切片/探测/同步检查 | nebula/media.py |
| processor/模型/归一化/聚合/计时 | nebula/embed.py |
| 编排、存储、状态机、断点续跑 | nebula/pipeline.py |
| 配置与 embedding_version | nebula/config.py、config.yaml |
| 模型选型与版本策略 | docs/adr/0001-pe-av-model-choice.md |
| 音画错位实验 | docs/adr/0008-av-sync-desync-experiment.md、nebula/avsync.py |
| 真实耗时明细 | logs/perf_005.csv（pipeline.py timings） |
