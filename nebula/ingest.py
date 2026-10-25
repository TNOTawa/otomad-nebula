"""数据导入：本地视频扫描、选票 CSV、周报 BV 列表。

接口见 docs/CONTRACTS.md 第 8 节。设计要点：

- 本地作品以"相对 input_local 的 POSIX 路径"作为稳定 work_key，重复扫描幂等；
- 选票 CSV 使用 utf-8-sig 编码，列名中英文兼容，非法行报错时带文件行号；
- 周报导入只负责"提取 BV 号 + 建立榜单/出现记录"，元数据与视频由流水线补齐；
- 所有可预期错误抛 IngestError（中文提示），CLI 顶层统一转换为退出码。

库代码只用 logging，不 print；只有 CLI 的 main() 打印中文摘要。
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .config import Config, load_config
from .db import Catalog
from .download import BVID_PATTERN, BVID_RE, http_get_text
from .errors import DownloadError, IngestError, NebulaError

LOGGER = logging.getLogger(__name__)

# 可识别的视频扩展名（小写比较），契约见 docs/CONTRACTS.md 第 8 节。
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".m4v"}

# 选票 CSV 的列名兼容表：规范字段 -> 可接受的表头（比较前统一 strip + casefold）。
_VOTES_COLUMNS: dict[str, tuple[str, ...]] = {
    "bvid": ("bvid", "bv号", "bv"),
    "part": ("part", "分p"),
    "rank": ("rank", "排名"),
    "votes": ("votes", "得票数", "票数"),
    "author_uid": ("author_uid", "uid", "up主uid"),
    "note": ("note", "备注"),
}

# 从任意文本中提取 BV 号：前后不能紧邻数字/字母，避免匹配更长的字母数字串。
_BVID_IN_TEXT_RE = re.compile(rf"(?<![0-9A-Za-z]){BVID_PATTERN}(?![0-9A-Za-z])")

WeeklyFetcher = Callable[[str], str]


# ---------------------------------------------------------------- 本地视频


@dataclass(frozen=True)
class VideoFile:
    """扫描到的本地视频文件。"""

    rel_path: str  # 相对 input_local 的 POSIX 路径
    abs_path: Path


@dataclass(frozen=True)
class LocalScanSummary:
    """一次本地扫描/登记的统计结果。"""

    files: int
    created: int
    existing: int


def scan_local_videos(input_local: Path) -> list[VideoFile]:
    """递归扫描本地视频目录，返回按 rel_path 排序的视频文件列表。

    只按扩展名识别视频，不读取文件内容；隐藏文件也会被收录，除非扩展名不在
    VIDEO_EXTENSIONS 里。目录不存在时抛 IngestError 并给出中文提示。
    """

    root = Path(input_local)
    if not root.is_dir():
        raise IngestError(f"本地视频目录不存在: {root}")
    resolved_root = root.resolve()
    try:
        paths = sorted(resolved_root.rglob("*"))
    except OSError as exc:
        raise IngestError(f"扫描本地视频目录失败: {resolved_root}: {exc}") from exc

    videos: list[VideoFile] = []
    for path in paths:
        try:
            if not path.is_file():
                continue
        except OSError as exc:
            raise IngestError(f"读取本地文件信息失败: {path}: {exc}") from exc
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        rel_path = path.relative_to(resolved_root).as_posix()
        videos.append(VideoFile(rel_path=rel_path, abs_path=path))
    videos.sort(key=lambda video: video.rel_path)
    LOGGER.debug("扫描到 %d 个本地视频（目录 %s）", len(videos), resolved_root)
    return videos


def register_local_videos(cat: Catalog, cfg: Config) -> LocalScanSummary:
    """扫描并登记 cfg.paths.input_local 下的视频为 source=local 的 pending 作品。

    幂等：已登记的文件不会重复创建作品，重复调用只更新统计。
    """

    videos = scan_local_videos(cfg.paths.input_local)
    created = 0
    existing = 0
    for video in videos:
        _, is_new = cat.upsert_local_work(video.rel_path)
        if is_new:
            created += 1
        else:
            existing += 1
    summary = LocalScanSummary(files=len(videos), created=created, existing=existing)
    LOGGER.info(
        "本地视频登记完成: 文件 %d 个, 新建作品 %d 个, 已存在 %d 个",
        summary.files,
        summary.created,
        summary.existing,
    )
    return summary


# ---------------------------------------------------------------- 选票 CSV


@dataclass(frozen=True)
class VotesRow:
    """选票 CSV 中的一行（一个作品）。"""

    bvid: str
    part: int = 1
    rank: int | None = None
    votes: int | None = None
    author_uid: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class ImportSummary:
    """一次导入操作的统计结果。"""

    list_id: int
    works_created: int
    works_existing: int
    appearances: int
    bvid_count: int


def _read_votes_records(path: Path) -> tuple[list[str], list[tuple[int, dict[str, str]]]]:
    """读取 CSV 表头与数据行，跳过空行和以 # 开头的注释行。

    返回 (表头列表, [(物理行号, 行字典)])。手工逐行读取而不是 csv.DictReader，
    是为了在存在注释行时仍能报告真实文件行号（表头为第 1 行起算的物理行号）。
    局限：多行引号字段的首字段若以 # 开头会被当作注释，选票 CSV 不需要该场景。
    """

    header: list[str] | None = None
    records: list[tuple[int, dict[str, str]]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for cells in reader:
            line_no = reader.line_num
            if not cells:
                continue
            if all(not str(cell).strip() for cell in cells):
                continue
            if str(cells[0]).lstrip().startswith("#"):
                continue
            if header is None:
                header = [str(cell).strip() for cell in cells]
                continue
            if len(cells) > len(header):
                raise IngestError(
                    f"第 {line_no} 行: 列数({len(cells)})多于表头({len(header)})，"
                    "请检查是否有多余逗号"
                )
            row: dict[str, str] = {}
            for index, name in enumerate(header):
                if not name:
                    continue
                row[name] = cells[index] if index < len(cells) else ""
            records.append((line_no, row))
    return header or [], records


def _resolve_columns(fieldnames: Sequence[str] | None) -> dict[str, str]:
    """把 CSV 表头映射到规范字段名，返回 规范字段 -> 实际表头 的字典。"""

    if not fieldnames:
        raise IngestError("选票 CSV 为空或缺少表头")
    normalized: dict[str, str] = {}
    for name in fieldnames:
        if name is None:
            continue
        key = str(name).strip().casefold()
        if key:
            normalized.setdefault(key, str(name))
    resolved: dict[str, str] = {}
    for canonical, aliases in _VOTES_COLUMNS.items():
        for alias in aliases:
            if alias in normalized:
                resolved[canonical] = normalized[alias]
                break
    if "bvid" not in resolved:
        accepted = " / ".join(_VOTES_COLUMNS["bvid"])
        found = ", ".join(str(name) for name in fieldnames if name)
        raise IngestError(f"选票 CSV 缺少 bvid 列（可用列名: {accepted}）；实际表头: {found}")
    return resolved


def _cell(row: Mapping[str, Any], columns: Mapping[str, str], field: str) -> str:
    header = columns.get(field)
    if header is None:
        return ""
    value = row.get(header)
    if value is None:
        return ""
    return str(value).strip()


def _parse_int_cell(value: str, *, field: str, line_no: int) -> int | None:
    if not value:
        return None
    try:
        return int(value, 10)
    except ValueError as exc:
        raise IngestError(f"第 {line_no} 行: {field} 必须是整数，收到 {value!r}") from exc


def _optional_text(value: str) -> str | None:
    return value or None


def parse_votes_csv(path: str | Path, *, vote_year: int) -> list[VotesRow]:
    """解析选票 CSV，返回去重后的 VotesRow 列表。

    规则
    ----
    - 编码 utf-8-sig（自动去掉 BOM），逗号分隔，以 # 开头的行是注释；
    - 必须列 bvid（兼容 BV号 / BV）；可选列 part/rank/votes/author_uid/note，
      中文列名兼容 分P、排名、得票数/票数、UID/UP主UID、备注；
    - bvid 必须匹配 BV + 10 位数字/字母；数字列必须是整数；
    - part >= 1，rank/votes >= 0；同 (bvid, part) 重复行由后者覆盖前者；
    - 非法数据抛 IngestError，消息带文件行号（表头为第 1 行起算的物理行号）。
    """

    csv_path = Path(path)
    if not isinstance(vote_year, int) or isinstance(vote_year, bool) or vote_year < 1:
        raise IngestError(f"选票年份非法: {vote_year!r}")
    if not csv_path.is_file():
        raise IngestError(f"选票 CSV 不存在: {csv_path}")
    LOGGER.debug("解析 %d 年选票 CSV: %s", vote_year, csv_path)

    rows: dict[tuple[str, int], VotesRow] = {}
    try:
        fieldnames, records = _read_votes_records(csv_path)
        columns = _resolve_columns(fieldnames)
        for line_no, raw in records:
            cells = {field: _cell(raw, columns, field) for field in columns}
            if not any(cells.values()):
                continue
            bvid = cells.get("bvid", "")
            if not bvid:
                raise IngestError(f"第 {line_no} 行: bvid 不能为空")
            if not BVID_RE.fullmatch(bvid):
                raise IngestError(f"第 {line_no} 行: BV 号格式非法: {bvid!r}")

            part_value = _parse_int_cell(cells.get("part", ""), field="part", line_no=line_no)
            part = 1 if part_value is None else part_value
            if part < 1:
                raise IngestError(f"第 {line_no} 行: part 必须 >= 1，收到 {part}")
            rank = _parse_int_cell(cells.get("rank", ""), field="rank", line_no=line_no)
            if rank is not None and rank < 0:
                raise IngestError(f"第 {line_no} 行: rank 不能为负数，收到 {rank}")
            votes = _parse_int_cell(cells.get("votes", ""), field="votes", line_no=line_no)
            if votes is not None and votes < 0:
                raise IngestError(f"第 {line_no} 行: votes 不能为负数，收到 {votes}")

            rows[(bvid, part)] = VotesRow(
                bvid=bvid,
                part=part,
                rank=rank,
                votes=votes,
                author_uid=_optional_text(cells.get("author_uid", "")),
                note=_optional_text(cells.get("note", "")),
            )
    except UnicodeDecodeError as exc:
        raise IngestError(
            f"选票 CSV 不是合法的 UTF-8 文本: {csv_path}；"
            "请用 Excel 另存为“CSV UTF-8”后重试"
        ) from exc
    except csv.Error as exc:
        raise IngestError(f"选票 CSV 解析失败: {csv_path}: {exc}") from exc
    except OSError as exc:
        raise IngestError(f"无法读取选票 CSV: {csv_path}: {exc}") from exc
    LOGGER.debug("选票 CSV 解析完成: %d 个作品", len(rows))
    return list(rows.values())


def _describe_source(cfg: Config, path: Path) -> str:
    """把导入源路径尽量记录成相对项目根的 POSIX 路径。"""

    resolved = path.resolve()
    try:
        return resolved.relative_to(cfg.root).as_posix()
    except ValueError:
        return resolved.as_posix()


def import_votes_csv(
    cat: Catalog,
    cfg: Config,
    path: str | Path,
    *,
    vote_year: int,
    list_name: str | None = None,
) -> ImportSummary:
    """导入某一年选票 CSV：登记作品、生成 votes_<年份> 榜单与出现记录。

    同一 CSV 重复导入幂等（作品按 (bvid, part) 去重、出现记录按榜单 upsert）。
    """

    csv_path = Path(path)
    rows = parse_votes_csv(csv_path, vote_year=vote_year)
    if not rows:
        raise IngestError(f"选票 CSV 没有有效数据行: {csv_path}")

    list_key = f"votes_{vote_year}"
    name = (list_name or f"{vote_year}年音MAD选票").strip()
    list_id = cat.ensure_list(
        list_key,
        name=name,
        kind="votes",
        vote_year=vote_year,
        source_path=_describe_source(cfg, csv_path),
    )

    created = 0
    existing = 0
    bvids: set[str] = set()
    for row in rows:
        work_id, is_new = cat.upsert_bilibili_work(
            row.bvid,
            part=row.part,
            author_uid=row.author_uid,
        )
        if is_new:
            created += 1
        else:
            existing += 1
        cat.add_appearance(
            work_id,
            list_id,
            rank=row.rank,
            votes=row.votes,
            note=row.note,
        )
        bvids.add(row.bvid)

    summary = ImportSummary(
        list_id=list_id,
        works_created=created,
        works_existing=existing,
        appearances=len(rows),
        bvid_count=len(bvids),
    )
    LOGGER.info(
        "选票导入完成: %d 年, 榜单 %s, 作品 %d 个(新建 %d / 已存在 %d), 出现记录 %d 条",
        vote_year,
        list_key,
        len(rows),
        summary.works_created,
        summary.works_existing,
        summary.appearances,
    )
    return summary


# ---------------------------------------------------------------- BV 号与周报


def extract_bvids(text: str) -> list[str]:
    """从任意文本（周报正文、HTML、BV 列表文本）中提取 BV 号，去重保序。"""

    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for match in _BVID_IN_TEXT_RE.finditer(str(text)):
        bvid = match.group(0)
        if bvid not in seen:
            seen.add(bvid)
            found.append(bvid)
    return found


def _normalize_bvids(values: Sequence[str], *, source: str) -> list[str]:
    """校验并去重直接给出的 BV 号列表（保序）。"""

    if isinstance(values, (str, bytes)):
        raise IngestError(f"{source} 需要 BV 号列表，收到单个字符串: {values!r}")
    cleaned: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(values, start=1):
        text = str(raw or "").strip()
        if not BVID_RE.fullmatch(text):
            raise IngestError(f"{source} 中第 {index} 个 BV 号格式非法: {raw!r}")
        if text not in seen:
            seen.add(text)
            cleaned.append(text)
    return cleaned


def _fetch_weekly_text(url: str, fetcher: WeeklyFetcher | None, cfg: Config) -> str:
    """抓取周报页面文本：优先用注入的 fetcher，否则走 http_get_text。"""

    clean = str(url or "").strip()
    if not clean:
        raise IngestError("周报 URL 不能为空")
    if fetcher is not None:
        try:
            text = fetcher(clean)
        except NebulaError:
            raise
        except Exception as exc:  # 注入 fetcher 的异常类型不限，统一为导入错误
            raise IngestError(f"抓取周报页面失败: {clean}: {exc}") from exc
    else:
        try:
            text = http_get_text(
                clean,
                user_agent=cfg.bilibili.user_agent,
                timeout_s=cfg.bilibili.api_timeout_seconds,
                proxy=cfg.download.proxy,
            )
        except DownloadError as exc:
            raise IngestError(f"抓取周报页面失败: {clean}: {exc}") from exc
    if not isinstance(text, str) or not text.strip():
        raise IngestError(f"周报页面内容为空: {clean}")
    return text


def import_weekly(
    cat: Catalog,
    cfg: Config,
    *,
    list_key: str,
    name: str | None = None,
    url: str | None = None,
    html: str | None = None,
    bvids: Sequence[str] | None = None,
    fetcher: WeeklyFetcher | None = None,
) -> ImportSummary:
    """导入一期周报，生成 kind=weekly 的榜单与全部 BV 的出现记录。

    三种输入任选其一（同时给出时优先级 bvids > html > url）：

    - bvids: 直接给出 BV 号列表；
    - html: 周报页面文本/HTML，用 extract_bvids 提取；
    - url: 用 fetcher（或默认 http_get_text）抓取页面后提取。

    异常
    ----
    IngestError:
        缺少输入、BV 号非法、页面抓取失败或未提取到任何 BV 号。
    """

    key = str(list_key or "").strip()
    if not key:
        raise IngestError("周报 list_key 不能为空")
    if bvids is None and html is None and url is None:
        raise IngestError("import_weekly 需要提供 bvids、html 或 url 三者之一")

    if bvids is not None:
        extracted = _normalize_bvids(bvids, source="bvids 参数")
        source_label = "bvids 参数"
    elif html is not None:
        extracted = extract_bvids(html)
        source_label = "html 内容"
    else:
        assert url is not None
        text = _fetch_weekly_text(url, fetcher, cfg)
        extracted = extract_bvids(text)
        source_label = f"URL {url.strip()}"
    if not extracted:
        raise IngestError(f"周报 {key} 未能从 {source_label} 中提取到任何 BV 号")

    display_name = (name or f"周报 {key}").strip()
    list_id = cat.ensure_list(
        key,
        name=display_name,
        kind="weekly",
        url=url.strip() if isinstance(url, str) and url.strip() else None,
        source_path=None,
    )

    created = 0
    existing = 0
    for bvid in extracted:
        work_id, is_new = cat.upsert_bilibili_work(bvid)
        if is_new:
            created += 1
        else:
            existing += 1
        cat.add_appearance(work_id, list_id)

    summary = ImportSummary(
        list_id=list_id,
        works_created=created,
        works_existing=existing,
        appearances=len(extracted),
        bvid_count=len(extracted),
    )
    LOGGER.info(
        "周报导入完成: %s, 作品 %d 个(新建 %d / 已存在 %d)",
        key,
        len(extracted),
        summary.works_created,
        summary.works_existing,
    )
    return summary


# ---------------------------------------------------------------- CLI


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        metavar="PATH",
        default=argparse.SUPPRESS,
        help="配置文件路径（默认当前目录 config.yaml）",
    )
    common.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=argparse.SUPPRESS,
        help="覆盖配置文件里的日志级别",
    )
    parser = argparse.ArgumentParser(
        prog="ingest.py",
        description="音MAD星云计划 · 数据导入（本地视频 / 选票 CSV / 周报）",
        parents=[common],
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="子命令")

    scan = subparsers.add_parser(
        "scan-local",
        parents=[common],
        help="递归扫描本地视频目录并登记为待处理作品",
        description="递归扫描 paths.input_local，登记 source=local 的 pending 作品。",
    )
    scan.set_defaults(handler=_cmd_scan_local)

    votes = subparsers.add_parser(
        "import-votes",
        parents=[common],
        help="导入某一年的选票 CSV",
        description="导入选票 CSV，生成 votes_<年份> 榜单与出现记录。",
    )
    votes.add_argument("csv", help="选票 CSV 路径（utf-8-sig，列名中英文兼容）")
    votes.add_argument("--year", type=int, required=True, help="选票年份，例如 2025")
    votes.add_argument(
        "--name",
        default=None,
        help="榜单显示名（默认“<年份>年音MAD选票”）",
    )
    votes.set_defaults(handler=_cmd_import_votes)

    weekly = subparsers.add_parser(
        "import-weekly",
        parents=[common],
        help="导入一期周报（URL / HTML 文件 / BV 列表）",
        description="从周报页面或 BV 列表中提取 BV 号，生成 weekly 榜单。",
    )
    weekly.add_argument("--list-key", required=True, help="周报期次键，例如 weekly_2025_w01")
    weekly.add_argument("--name", default=None, help="榜单显示名（默认“周报 <list_key>”）")
    source = weekly.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="周报页面 URL（联网抓取）")
    source.add_argument("--html", metavar="FILE", help="本地保存的周报 HTML 文件")
    source.add_argument("--bvids", nargs="+", metavar="BV", help="直接给出 BV 号列表")
    weekly.set_defaults(handler=_cmd_import_weekly)
    return parser


def _cmd_scan_local(cat: Catalog, cfg: Config, args: argparse.Namespace) -> int:
    summary = register_local_videos(cat, cfg)
    print(f"本地视频扫描完成：{cfg.paths.input_local}")
    print(
        f"  视频文件 {summary.files} 个，新建作品 {summary.created} 个，"
        f"已存在 {summary.existing} 个"
    )
    return 0


def _cmd_import_votes(cat: Catalog, cfg: Config, args: argparse.Namespace) -> int:
    summary = import_votes_csv(
        cat,
        cfg,
        args.csv,
        vote_year=args.year,
        list_name=args.name,
    )
    print(f"选票导入完成：{args.year} 年，榜单 ID {summary.list_id}")
    print(
        f"  作品 {summary.works_created + summary.works_existing} 个"
        f"（新建 {summary.works_created}，已存在 {summary.works_existing}），"
        f"出现记录 {summary.appearances} 条，BV 号 {summary.bvid_count} 个"
    )
    return 0


def _cmd_import_weekly(cat: Catalog, cfg: Config, args: argparse.Namespace) -> int:
    html: str | None = None
    if args.html is not None:
        html_path = Path(args.html)
        try:
            html = html_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise IngestError(f"周报 HTML 文件不是合法的 UTF-8 文本: {html_path}") from exc
        except OSError as exc:
            raise IngestError(f"无法读取周报 HTML 文件: {html_path}: {exc}") from exc
    summary = import_weekly(
        cat,
        cfg,
        list_key=args.list_key,
        name=args.name,
        url=args.url,
        html=html,
        bvids=args.bvids,
    )
    print(f"周报导入完成：{args.list_key}，榜单 ID {summary.list_id}")
    print(
        f"  作品 {summary.works_created + summary.works_existing} 个"
        f"（新建 {summary.works_created}，已存在 {summary.works_existing}），"
        f"出现记录 {summary.appearances} 条"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """ingest.py 的命令行入口；返回进程退出码（0 成功，1 可预期错误）。"""

    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    config_path = getattr(args, "config", None)
    log_level = getattr(args, "log_level", None)
    try:
        cfg = load_config(config_path) if config_path else load_config()
        _configure_logging(log_level or cfg.logging.level)
        cfg.ensure_directories()
        with Catalog(cfg.paths.db_path) as cat:
            cat.initialize()
            return int(args.handler(cat, cfg, args))
    except NebulaError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("操作已取消", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover - 便于 python -m nebula.ingest
    raise SystemExit(main())
