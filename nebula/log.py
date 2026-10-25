"""统一日志配置。

库代码只使用 logging.getLogger(...)；CLI 在入口调用 configure_logging。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(
    level: str = "INFO",
    *,
    log_file: str | Path | None = None,
    force: bool = False,
) -> None:
    """配置根日志：控制台 + 可选文件。

    重复调用默认是幂等的；需要重新配置时传 force=True。
    """

    handlers: list[logging.Handler] = [logging.StreamHandler(stream=sys.stderr)]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=force,
    )
