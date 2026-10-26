# v0.5-beta 密集音频编码 Pilot 报告（MERT-v1-95M，20 部本地作品）

日期：2026-10-26
任务：task-39（t1）。产物为研究数据，**不写 DB、不建地图、不覆盖事件特征**。

## 1. 模型选择依据（事实核查）

| 候选 | 结论 | 证据 |
| --- | --- | --- |
| transformers 5.20.2 内置 MERTModel | 不可用 | `from transformers import MERTModel` -> ImportError |
| m-a-p/MERT-v1-95M（trust_remote_code） | **采用** | 仓库自带 configuration_MERT.py / modeling_MERT.py；AutoConfig/AutoModel 在 5.20.2 下加载成功；MERTModel 继承 HubertModel，94,371,712 参数；24kHz 输入，conv stride 320 -> 75Hz 帧；GPU 5s batch2 前向 RTF≈0.005–0.03 |
| facebook/wav2vec2-base / hubert-base-ls960 | 备选，未引入 | MERT 可跑通，按任务“只选一个模型”要求不混用 |
| CLAP 10s 滑窗 | 备选，未引入 | 同上；且 10s 粒度与 0.5s bin 对齐目标不符 |

固定 revision：`12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`（写入版本与 manifest）。
MERT 的 custom modeling 在 5.20.2 下 `output_hidden_states=True` 不返回 hidden_states，
因此实现用 forward hook 取 encoder 第 9 层（共 12 层）输出；layer 未做调优。

## 2. 编码版本与参数

版本号（同时是 `data/audio_dense/` 目录名）：

~~~
m-a-p-MERT-v1-95M-12af15fef9d0ac838c3f475bfbbf26d2060dd4f5__sr24000_ctx5_hop5_l9_poolmean_std_bin0.5_norml2
~~~

| 参数 | 值 |
| --- | --- |
| encoder / revision | m-a-p/MERT-v1-95M ` 12af15fe... |
| sample_rate | 24000 Hz（单声道，ffmpeg 重采样） |
| context / hop | 5.0s / 5.0s（MERT 典型 5s 片段，不重叠） |
| frame_rate / dim | 75 Hz / 768 |
| layer | 第 9 层 encoder 输出（forward hook） |
| bin_seconds | 0.5s 全局 bin |
| pool | mean_std（bin 内均值 + 总体标准差） |
| normalize | l2（逐帧 L2 归一化后再聚合） |
| 帧时间映射 | frame_time = window_start + (i*stride + receptive/2)/sample_rate；stride=320、receptive=400 采样点 |

## 3. Pilot 设置

- 数据源：`data/catalog.sqlite` 中 source=local、status=completed、canonical（排除
  work_links duplicate）且源文件存在的作品共 33 部；按时长分位选 20 部（21.2s–295.4s），
  work_id = 27, 25, 7, 30, 22, 21, 24, 18, 20, 23, 1, 13, 28, 17, 11, 12, 33, 29, 32, 19。
- 源文件：`input/local/` 下原始音视频（只读，未下载网络视频、未删除）。
- 设备：RTX 3060 12GB；模型权重从 hf-mirror 缓存到 `.cache/hf/hub`（`HF_ENDPOINT=https://hf-mirror.com`；
  本机已缓存，重跑不重复下载）。
- 输出：`data/audio_dense/<version>/<work_id>/audio.npz` + `manifest.json`；
  pilot JSON：`logs/audio_dense_pilot.json`。

## 4. 结果（通过）

| 指标 | 值 |
| --- | --- |
| 成功 / 选中 | **20 / 20**（failed 0，skipped 0） |
| 音频总时长 | **2582.753s**（43m02.8s） |
| 总帧数 / 总 bin 数 | **193,179 帧 / 5,177 bin** |
| 处理墙钟（20 部求和） | **24.457s** |
| 全流程墙钟（含模型加载/DB/保存） | **36.682s** |
| RTF（processing_wall / audio_s） | **0.00947**（约 **105.6× 实时**） |
| 模型加载 | 9.85s（冷加载，含 custom code + 94.37M 权重） |
| 显存峰值 | **1320.4 MB**（≤8GB 要求） |
| 产物磁盘 | **329,401,863 B ≈ 314.1 MiB** |
| 每部墙钟 / 编解码 | 0.384–2.533s / encode 0.134–1.102s / decode 0.084–0.356s |
| 时长漂移（decoded vs ffprobe） | 最大 **0.01898s**（≤1 bin=0.5s） |
| bin_mask 覆盖率 | 最小 0.990476（3 部 <1.0，见 §5） |

通过条件逐项：

| 检查 | 结果 |
| --- | --- |
| 20 部全部成功 | PASS |
| bin 数与时长一致（n_bins == ceil(duration/0.5)） | PASS（20/20） |
| 无 NaN/Inf / 空序列 | PASS |
| 对齐误差 ≤ 1 bin（decoded duration drift ≤ 0.5s） | PASS（max 19ms） |
| 显存 ≤ 8GB | PASS（1.32GB） |
| 断点复用 | PASS（第二次运行 20/20 skipped，纯 JSON 输出） |

整体结论：**PASS**，pilot JSON 中 `summary.pass = true`。

## 5. 产物与格式

~~~
data/audio_dense/m-a-p-MERT-v1-95M-12af15fe...__sr24000_ctx5_hop5_l9_poolmean_std_bin0.5_norml2/
  000001/audio.npz
  000001/manifest.json
  ...（20 个 work_id 目录，共 314.1MiB）
logs/audio_dense_pilot.json
~~~

`audio.npz` 数组：

| key | dtype | shape | 说明 |
| --- | --- | --- | --- |
| frame_features | float16 | (N, 768) | MERT 第 9 层帧特征，逐帧 L2 归一化 |
| frame_times | float32 | (N,) | 帧中心秒数（全局时间轴） |
| bin_features | float32 | (n_bins, 768) | 0.5s bin 内均值 |
| bin_std | float32 | (n_bins, 768) | 0.5s bin 内总体标准差 |
| bin_times | float32 | (n_bins, 2) | bin 起止秒 |
| bin_mask | bool | (n_bins,) | 该 bin 是否有帧 |
| duration_s | float64 | () | ffprobe 时长 |

`manifest.json` 包含 version/encoder/revision/sample_rate/context_s/hop_s/layer/
pool/bin_seconds/normalize/frame_rate/dim/n_frames/n_bins/duration_s/
decoded_duration_s/decode_s/encode_s/model_load_s/wall_s/created_at/work_id。

每部作品只读源文件、原子写产物；`--force` 覆盖、否则 manifest 完整且版本匹配时跳过。
部分作品 bin_mask_ratio <1.0（0.990/0.991/0.998）是因为 5s 非重叠窗口在作品首尾
存在不足一个 bin 的空隙（conv 感受野边界），这些 bin 的向量置零并 mask=False；
bin 总数、时间映射与对齐检查均通过。

## 6. 复现命令

~~~
cd F:\Coding\otomad_nebula
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = "F:\Coding\otomad_nebula\.cache\hf"
.\.venv\Scripts\python.exe tools/audio_dense.py --pilot --limit 20
.\.venv\Scripts\python.exe tools/audio_dense.py --pilot --limit 20 --json   # 机器可读
.\.venv\Scripts\python.exe -m pytest tests/test_audio_dense.py -q           # 离线单元测试
~~~

## 7. 局限与风险

1. **trust_remote_code**：MERT 仓库 custom code 会被执行；已固定 revision 且版本/manifest
   记录，但 transformers 升级（当前 5.20.2）后必须重验（custom modeling 依赖
   transformers.models.hubert 内部实现）。
2. **层选择未调优**：第 9 层是折中（12 层中的中后段），不宣称最优；v0.5-beta 训练时
   可作超参/换层实验，但换层要新版本。
3. **5s 非重叠窗口**：MERT 典型 5s 输入；窗口边界有少量 bin 无帧（bin_mask=False），
   长作品的局部音画学习可能受边界影响；需要时可改 context/hop（新版本）做对照。
4. **特征能力边界**：这是冻结 MERT 帧特征，不含人工语义标签，不宣称“逐帧语义理解”；
   本 pilot 只验证时间对齐/质量/性能，不注册 DB/地图、不改 v0.5-alpha 负结果。
5. **覆盖范围**：只处理 20/33 部本地 canonical 作品（约 43 分钟音频）；全量 110 部中
   含 Bilibili 来源，本次按任务要求不下载网络视频，因此不覆盖。
6. **单模型约束**：按要求只接入 MERT；未同时接入 wav2vec2/hubert/CLAP，跨模型对照
   留给后续实验（若 MERT 语义不足再评估）。
7. **RTF 口径**：RTF = 处理墙钟 / 音频总时长，不含模型加载与 DB 查询；总墙钟 36.7s
   含 9.85s 冷加载。不同 GPU/驱动会有差异。

## 8. 第二阶段：全量 110 部 canonical 扩展（最终结果）

Pilot PASS 后，用同一冻结 MERT、同一版本/参数扩展到全部 canonical + completed
作品：local 23 + Bilibili 87 = **110 部**。Bilibili 源用 `resolve_bilibili_source`
每部下载一次并持久缓存到 `data/sources/<bvid>/`（87 个目录，**2108.2 MB**；不删除、
不重复下载；已有缓存直接复用）。由于修复了个别源音频比容器短的计时问题（解码最多截到
ffprobe 容器时长），最终对 110 部做了同版本 force 全量重跑以保证算法一致；pilot 的
20 部产物被同一版本重写（参数未变，旧 pilot 指标保留在 §4 与 `logs/audio_dense_pilot.json`）。

| 指标 | 全量 110 部 |
| --- | --- |
| 覆盖 / 成功 / 失败 | **110 / 110 / 0**（local 23 + Bilibili 87） |
| 失败列表 | 无 |
| 音频总时长 | **32619.14s**（9h03m39s） |
| 总帧数 / 总 bin 数 | **2,439,577 帧 / 65,291 bin** |
| 处理墙钟（110 部求和） | **264.717s** |
| 全流程墙钟（含模型加载/缓存复用/保存） | **278.564s** |
| RTF / 实时倍速 | **0.00812** / **123.2×** |
| 模型加载 | 9.3s（权重/源均已缓存） |
| 显存峰值 | **1320.4 MB**（≤8GB） |
| 音频产物磁盘 | **4,158,970,616 B ≈ 3.87 GiB** |
| 每部墙钟 / encode / decode | 0.397–11.812s / 0.138–6.382s / 0.082–1.535s |
| 作品时长范围 | 21.2s–1705.2s |
| bin 数与时长一致 | PASS（110/110） |
| 帧时间对齐 ≤1 bin（最后一帧 vs 已解码音频结尾） | PASS（110/110） |
| 无 NaN/Inf / 空序列 | PASS |
| 显存 ≤8GB | PASS |

Pilot vs 全量：

| 指标 | Pilot（20 部本地） | 全量（110 部 canonical） |
| --- | --- | --- |
| 音频时长 | 2582.753s | 32619.14s |
| 帧 / bin | 193,179 / 5,177 | 2,439,577 / 65,291 |
| RTF / 实时倍速 | 0.00947 / 105.6× | 0.00812 / 123.2× |
| 处理墙钟 / 全流程 | 24.457s / 36.682s | 264.717s / 278.564s |
| 显存峰值 | 1320.4 MB | 1320.4 MB |
| 产物 | 314.1 MiB | 3.87 GiB（另 sources 2.1GB） |
| 失败 | 0 | 0 |

### 8.1 严格容器时长对齐：仅 work 50 / 99，原因是源音视频时长差（非时间戳漂移）

- **严格口径**：`abs(decoded_audio_duration - ffprobe_container_duration) <= 1 bin (0.5s)`。
  110 部中 **108 部通过**，仅 2 部失败：
  - **work_id=50**：容器 994.0s，音频流实际可解码 **短 1.950667s**，末尾
    **3 个 0.5s bin mask=False**（bin_mask_ratio 0.998491）。
  - **work_id=99**：容器 228.483s，音频流实际可解码 **短 1.496333s**，末尾
    **3 个 0.5s bin mask=False**（bin_mask_ratio 0.993435）。
- **结论**：这是**源视频音轨比容器/视频短**导致的尾部音轨缺失，帧时间戳映射本身
  没有漂移/错位；不是编码错误、不是 NaN/Inf、也不是 bin 计数错误。若按严格容器口径，
  full 的 `pass=false 仅由这 2 部引起`；按产物实际帧时间映射口径 **110/110 PASS**，
  pilot 为 **PASS**。这两部已从 v0.5-beta 的 77/22/11 划分中**排除**（可用 108 部），
  并由 `summary.strict_container_alignment` 与 `summary.excluded_from_beta` 显式记录。
- 全量 JSON 额外字段：`alignment_definition`、`strict_container_alignment`
  （metric/passed/failed_works/tail_missing_bins/cause/note）、
  `container_duration_drift_s`、`data_quality_notes`、`excluded_from_beta`；
  works 记录含 `bin_mask_missing`、`last_frame_s`、`decoded_duration_s`。

所有 110 部产物使用同一版本/参数，可直接供 v0.5-beta 局部双塔按 77/22/11 划分训练
（work 50/99 排除）；本任务未注册 DB、未建地图、未覆盖事件特征。
全量 JSON：`logs/audio_dense_full.json`。

