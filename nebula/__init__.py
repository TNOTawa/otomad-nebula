"""音MAD星云计划核心包。

模块划分（与方案文档一致）：

- nebula.config    配置与路径
- nebula.db        SQLite 编目：作品、榜单、向量登记、任务状态、星云版本
- nebula.sampling  时间轴均匀采样规则（纯函数，便于测试）
- nebula.media     ffprobe/ffmpeg 音画同步切片
- nebula.embed     PE-AV 向量提取与片段聚合（含仅测试用的确定性嵌入器）
- nebula.pipeline  下载 -> 切片 -> 推理 -> 保存 -> 清理 的流水线
- nebula.ingest    本地扫描、选票 CSV、周报导入
- nebula.download  yt-dlp 下载与 Bilibili 元数据
- nebula.mapplot   UMAP 二维/三维星云与版本管理
- nebula.appdata   可视化用纯数据准备（染色、搜索、跳转链接）
"""

__version__ = "0.2.0"

# 向量种类：作品级与片段级各保存联合/音频/视频三类
WORK_VECTOR_KINDS = ("work_joint", "work_audio", "work_video")
CLIP_VECTOR_KINDS = ("clip_joint", "clip_audio", "clip_video")
EMBEDDING_KINDS = WORK_VECTOR_KINDS + CLIP_VECTOR_KINDS
