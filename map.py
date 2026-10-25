"""星云坐标命令行入口（build / list / transform）。

用法（在项目根目录执行，使用 .venv 中的 python）：

    python map.py build
    python map.py list
    python map.py transform

配置文件可用 --config 指定，也可用环境变量 NEBULA_CONFIG。
"""

from __future__ import annotations

from nebula.mapplot import main as _main


def main(argv: list[str] | None = None) -> int:
    """转发到 nebula.mapplot.main，便于脚本与测试统一入口。"""

    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
