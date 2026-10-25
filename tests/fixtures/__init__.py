"""测试夹具（fixtures）：合成媒体工厂与临时配置辅助。

本目录下的模块被 tests/ 使用，不是 pytest fixture 的 conftest.git 正式模块；
通过 "from tests.fixtures.video_factory import ..." 导入。
"""

from __future__ import annotations

__all__ = [
    "ALL_VIDEO_NAMES",
    "BAD_VIDEO_NAMES",
    "FACTORY_VERSION",
    "GOOD_VIDEO_NAMES",
    "OUT_DIR",
    "REPO_ROOT",
    "VIDEO_SPECS",
    "VideoSpec",
    "build_video_library",
    "create_temp_catalog",
    "create_temp_config",
    "install_video_library",
    "media_tools_available",
    "probe_duration_s",
]
