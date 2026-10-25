"""音MAD星云计划 · 数据导入命令行入口。

子命令与用法见：python ingest.py --help

示例：
    python ingest.py scan-local
    python ingest.py --config config.yaml import-votes input/lists/votes_2025.csv --year 2025
    python ingest.py import-votes input/lists/zuozhe_votes_2025.csv --year 2025 --limit 30
    python ingest.py link-local --list-key votes_2025 --dry-run
    python ingest.py link-local --list-key votes_2025 --manual
    python ingest.py import-weekly --list-key weekly_2025_w01 --url https://www.bilibili.com/read/...
    python ingest.py import-weekly --list-key weekly_2025_w01 --bvids BV1xx411c7mD BV1yy411c7mE
"""

from __future__ import annotations

import sys

from nebula.ingest import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
