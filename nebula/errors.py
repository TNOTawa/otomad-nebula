"""项目内统一的异常类型。"""

from __future__ import annotations


class NebulaError(Exception):
    """项目内所有可预期错误的基类。"""


class ConfigError(NebulaError):
    """配置文件缺失、字段非法或路径非法。"""


class DatabaseError(NebulaError):
    """SQLite 编目操作失败（例如非法的状态迁移）。"""


class MediaError(NebulaError):
    """ffprobe / ffmpeg 探测或切片失败。"""


class SamplingError(NebulaError):
    """时间轴采样参数非法（例如视频过短）。"""


class EmbeddingError(NebulaError):
    """向量提取或聚合失败。"""


class DownloadError(NebulaError):
    """视频下载或元数据获取失败。"""


class IngestError(NebulaError):
    """导入清单数据失败。"""


class MapError(NebulaError):
    """星云坐标生成/更新失败（例如作品数量不足、向量维度不一致）。"""


class PipelineError(NebulaError):
    """流水线编排失败。"""
