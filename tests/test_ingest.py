"""nebula.ingest 与 ingest.py CLI 的离线测试。"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from nebula.db import Catalog
from nebula.download import BVID_RE
from nebula.errors import IngestError
from nebula.ingest import (
    VIDEO_EXTENSIONS,
    LocalScanSummary,
    VotesRow,
    extract_bvids,
    import_votes_csv,
    import_weekly,
    parse_votes_csv,
    register_local_videos,
    scan_local_videos,
)
from nebula.ingest import main as ingest_main

# 形如 BV + 10 位数字/字母，但均为虚构数据。
BVIDS = ("BV1demo00001", "BV1demo00002", "BV1demo00003", "BV1demo00004", "BV1demo00005")


def _write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8", newline="")
    return path


def _votes_csv(path: Path, rows: str) -> Path:
    return _write_csv(path, "bvid,part,rank,votes,author_uid,note\n" + rows)


# ---------------------------------------------------------------- 本地扫描


def test_video_extensions_match_contract() -> None:
    assert {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".m4v"} == VIDEO_EXTENSIONS


def test_scan_local_videos_recursive_sorted(config) -> None:
    root = config.paths.input_local
    (root / "b").mkdir()
    (root / "b" / "second.MKV").write_bytes(b"x")
    (root / "a.mp4").write_bytes(b"x")
    (root / "notes.txt").write_bytes(b"not a video")
    (root / "sub").mkdir()
    (root / "sub" / "c.webm").write_bytes(b"x")

    videos = scan_local_videos(root)

    assert [video.rel_path for video in videos] == ["a.mp4", "b/second.MKV", "sub/c.webm"]
    assert all(video.abs_path.is_file() for video in videos)
    assert all("\\" not in video.rel_path for video in videos)


def test_scan_local_videos_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="不存在"):
        scan_local_videos(tmp_path / "missing")


def test_register_local_videos_idempotent(config, catalog: Catalog) -> None:
    root = config.paths.input_local
    (root / "a.mp4").write_bytes(b"x")
    (root / "sub").mkdir()
    (root / "sub" / "b.mkv").write_bytes(b"x")

    first = register_local_videos(catalog, config)
    assert first == LocalScanSummary(files=2, created=2, existing=0)

    second = register_local_videos(catalog, config)
    assert second == LocalScanSummary(files=2, created=0, existing=2)

    works = catalog.list_works(source="local")
    assert len(works) == 2
    assert {work["local_path"] for work in works} == {"a.mp4", "sub/b.mkv"}
    assert all(work["status"] == "pending" for work in works)
    assert catalog.get_work_by_key("local:sub/b.mkv") is not None
    # 已登记作品出现在榜单里时，重复扫描不应重置状态
    work_id = works[0]["id"]
    catalog.set_status(work_id, "encoding")
    register_local_videos(catalog, config)
    assert catalog.get_work(work_id)["status"] == "encoding"


# ---------------------------------------------------------------- 选票 CSV 解析


def test_parse_votes_csv_basic(tmp_path: Path) -> None:
    path = _votes_csv(
        tmp_path / "votes.csv",
        "BV1demo00001,1,1,100,10001,第一\n"
        "BV1demo00002,2,3,50,10002,\n",
    )
    rows = parse_votes_csv(path, vote_year=2025)
    assert rows == [
        VotesRow(
            bvid="BV1demo00001",
            part=1,
            rank=1,
            votes=100,
            author_uid="10001",
            note="第一",
        ),
        VotesRow(
            bvid="BV1demo00002",
            part=2,
            rank=3,
            votes=50,
            author_uid="10002",
            note=None,
        ),
    ]


@pytest.mark.parametrize(
    "header",
    [
        "bvid,part,rank,votes,author_uid,note",
        "BV号,分P,排名,得票数,UP主UID,备注",
        "BV,分P,排名,票数,UID,备注",
    ],
)
def test_parse_votes_csv_column_variants(tmp_path: Path, header: str) -> None:
    path = _write_csv(
        tmp_path / "votes.csv",
        header + "\nBV1demo00001,1,2,66,10001,备注内容\n",
    )
    row = parse_votes_csv(path, vote_year=2025)[0]
    assert row == VotesRow(
        bvid="BV1demo00001",
        part=1,
        rank=2,
        votes=66,
        author_uid="10001",
        note="备注内容",
    )


@pytest.mark.parametrize("header", ["作品标题", "title", "标题"])
def test_parse_votes_csv_title_aliases(tmp_path: Path, header: str) -> None:
    path = _write_csv(
        tmp_path / "votes.csv",
        f"bvid,{header}\n"
        "BV1demo00001,作品甲\n",
    )
    row = parse_votes_csv(path, vote_year=2025)[0]
    assert row.title == "作品甲"


def test_parse_votes_csv_multiline_title_stripped(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "votes.csv",
        "BV号,排名,票数,作品标题\n"
        'BV1demo00001,1,10,"多行标题\n\n"\n'
        "BV1demo00002,2,9,  普通标题  \n",
    )
    rows = parse_votes_csv(path, vote_year=2025)
    assert rows[0].title == "多行标题"
    assert rows[1].title == "普通标题"


def test_parse_votes_csv_bom_encoding(tmp_path: Path) -> None:
    text = "BV号,排名\nBV1demo00001,1\n"
    path = tmp_path / "bom.csv"
    path.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    rows = parse_votes_csv(path, vote_year=2025)
    assert rows == [VotesRow(bvid="BV1demo00001", rank=1)]


def test_parse_votes_csv_minimal_columns_defaults(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "min.csv", "bvid\nBV1demo00001\n")
    row = parse_votes_csv(path, vote_year=2025)[0]
    assert row == VotesRow(bvid="BV1demo00001")


def test_parse_votes_csv_comments_and_blank_lines(tmp_path: Path) -> None:
    text = (
        "# 这是注释\n"
        "\n"
        "bvid,rank\n"
        "BV1demo00001,1\n"
        "\n"
        "# 中间注释\n"
        "BV1demo00002,2\n"
    )
    rows = parse_votes_csv(_write_csv(tmp_path / "votes.csv", text), vote_year=2025)
    assert [row.bvid for row in rows] == ["BV1demo00001", "BV1demo00002"]


def test_parse_votes_csv_invalid_bvid_reports_line(tmp_path: Path) -> None:
    path = _votes_csv(tmp_path / "votes.csv", "BV1demo00001,1\nnot-a-bv,1\n")
    with pytest.raises(IngestError) as info:
        parse_votes_csv(path, vote_year=2025)
    message = str(info.value)
    assert "第 3 行" in message
    assert "not-a-bv" in message


def test_parse_votes_csv_line_number_with_comments(tmp_path: Path) -> None:
    text = "# 注释一\n# 注释二\nbvid\nBV1demo00001\nbad_bv\n"
    with pytest.raises(IngestError) as info:
        parse_votes_csv(_write_csv(tmp_path / "votes.csv", text), vote_year=2025)
    assert "第 5 行" in str(info.value)


def test_parse_votes_csv_missing_bvid_column(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "votes.csv", "rank,votes\n1,10\n")
    with pytest.raises(IngestError, match="bvid"):
        parse_votes_csv(path, vote_year=2025)


def test_parse_votes_csv_empty_file(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "empty.csv", "")
    with pytest.raises(IngestError, match="表头"):
        parse_votes_csv(path, vote_year=2025)


def test_parse_votes_csv_only_comments(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "comments.csv", "# 只有注释\n\n")
    with pytest.raises(IngestError, match="表头"):
        parse_votes_csv(path, vote_year=2025)


def test_parse_votes_csv_duplicate_last_wins(tmp_path: Path) -> None:
    path = _votes_csv(
        tmp_path / "votes.csv",
        "BV1demo00001,1,1,10\n"
        "BV1demo00001,1,2,20\n"
        "BV1demo00001,2,3,30\n",
    )
    rows = parse_votes_csv(path, vote_year=2025)
    assert len(rows) == 2
    part1 = next(row for row in rows if row.part == 1)
    part2 = next(row for row in rows if row.part == 2)
    assert (part1.rank, part1.votes) == (2, 20)
    assert (part2.rank, part2.votes) == (3, 30)


@pytest.mark.parametrize(
    ("row", "keyword"),
    [
        ("BV1demo00001,0,,", "part"),
        ("BV1demo00001,-1,,", "part"),
        ("BV1demo00001,1,-2,", "rank"),
        ("BV1demo00001,1,2,-3", "votes"),
        ("BV1demo00001,abc,1,5", "part"),
        ("BV1demo00001,1,1.5,5", "rank"),
    ],
)
def test_parse_votes_csv_rejects_bad_numbers(tmp_path: Path, row: str, keyword: str) -> None:
    path = _votes_csv(tmp_path / "votes.csv", row + "\n")
    with pytest.raises(IngestError) as info:
        parse_votes_csv(path, vote_year=2025)
    assert keyword in str(info.value)
    assert "第 2 行" in str(info.value)


def test_parse_votes_csv_missing_file(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="不存在"):
        parse_votes_csv(tmp_path / "missing.csv", vote_year=2025)


def test_parse_votes_csv_non_utf8_raises(tmp_path: Path) -> None:
    path = tmp_path / "gbk.csv"
    path.write_bytes("bvid,备注\nBV1demo00001,中文备注\n".encode("gbk"))
    with pytest.raises(IngestError, match="UTF-8"):
        parse_votes_csv(path, vote_year=2025)


@pytest.mark.parametrize("year", [0, -1, "2025", True])
def test_parse_votes_csv_rejects_bad_year(tmp_path: Path, year) -> None:
    path = _votes_csv(tmp_path / "votes.csv", "BV1demo00001,1\n")
    with pytest.raises(IngestError, match="年份"):
        parse_votes_csv(path, vote_year=year)


# ---------------------------------------------------------------- 选票导入


def test_import_votes_csv_creates_works_lists_appearances(config, catalog: Catalog, tmp_path: Path) -> None:
    path = _votes_csv(
        tmp_path / "votes.csv",
        "BV1demo00001,1,1,100,10001,第一名\n"
        "BV1demo00002,1,2,80,10002,第二名\n"
        "BV1demo00003,2,3,60,10003,\n",
    )
    summary = import_votes_csv(catalog, config, path, vote_year=2025)

    assert summary.list_id > 0
    assert summary.works_created == 3
    assert summary.works_existing == 0
    assert summary.appearances == 3
    assert summary.bvid_count == 3

    lists = catalog.list_lists()
    assert len(lists) == 1
    assert lists[0]["list_key"] == "votes_2025"
    assert lists[0]["name"] == "2025年音MAD选票"
    assert lists[0]["kind"] == "votes"
    assert lists[0]["vote_year"] == 2025

    work = catalog.get_work_by_bvid("BV1demo00001")
    assert work is not None
    assert work["source"] == "bilibili"
    assert work["status"] == "pending"
    assert work["url"] == "https://www.bilibili.com/video/BV1demo00001"
    appearance = catalog.get_appearances(work["id"])[0]
    assert appearance["rank"] == 1
    assert appearance["votes"] == 100
    assert appearance["note"] == "第一名"

    part2 = catalog.get_work_by_bvid("BV1demo00003", 2)
    assert part2 is not None
    assert part2["work_key"] == "bilibili:BV1demo00003:p2"


def test_import_votes_csv_idempotent_and_updates(config, catalog: Catalog, tmp_path: Path) -> None:
    path = tmp_path / "votes.csv"
    _votes_csv(path, "BV1demo00001,1,1,100,10001,第一版\n")
    first = import_votes_csv(catalog, config, path, vote_year=2025)
    assert first.works_created == 1

    second = import_votes_csv(catalog, config, path, vote_year=2025)
    assert second.works_created == 0
    assert second.works_existing == 1
    assert len(catalog.list_lists()) == 1
    assert len(catalog.appearance_rows()) == 1

    # 修改 CSV 后重新导入：出现记录的 rank/votes/note 被更新
    _votes_csv(path, "BV1demo00001,1,9,999,10001,第二版\n")
    import_votes_csv(catalog, config, path, vote_year=2025)
    work = catalog.get_work_by_bvid("BV1demo00001")
    appearance = catalog.get_appearances(work["id"])[0]
    assert (appearance["rank"], appearance["votes"], appearance["note"]) == (9, 999, "第二版")


def test_import_votes_csv_same_bvid_different_part(config, catalog: Catalog, tmp_path: Path) -> None:
    path = _votes_csv(
        tmp_path / "votes.csv",
        "BV1demo00001,1,1,100\n"
        "BV1demo00001,2,2,90\n",
    )
    summary = import_votes_csv(catalog, config, path, vote_year=2025)
    assert summary.works_created == 2
    assert summary.appearances == 2
    assert summary.bvid_count == 1


def test_import_votes_csv_empty_data_raises(config, catalog: Catalog, tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "votes.csv", "# 只有注释\nbvid,rank\n")
    with pytest.raises(IngestError, match="没有有效数据行"):
        import_votes_csv(catalog, config, path, vote_year=2025)
    assert catalog.list_lists() == []


def test_import_votes_csv_writes_title_and_honours_limit(
    config, catalog: Catalog, tmp_path: Path
) -> None:
    path = _write_csv(
        tmp_path / "votes.csv",
        "BV号,排名,票数,作品标题\n"
        "BV1demo00001,1,10,作品甲\n"
        "BV1demo00002,2,9,作品乙\n"
        "BV1demo00003,3,8,作品丙\n",
    )
    summary = import_votes_csv(catalog, config, path, vote_year=2025, limit=2)
    assert summary.works_created == 2
    assert summary.appearances == 2
    assert catalog.get_work_by_bvid("BV1demo00001")["title"] == "作品甲"
    assert catalog.get_work_by_bvid("BV1demo00003") is None

    again = import_votes_csv(catalog, config, path, vote_year=2025)
    assert again.works_created == 1
    assert again.works_existing == 2
    assert again.appearances == 3
    assert catalog.get_work_by_bvid("BV1demo00003")["title"] == "作品丙"


@pytest.mark.parametrize("limit", [0, -1, True])
def test_import_votes_csv_invalid_limit(
    config, catalog: Catalog, tmp_path: Path, limit
) -> None:
    path = _votes_csv(tmp_path / "votes.csv", "BV1demo00001,1\n")
    with pytest.raises(IngestError, match="数量上限"):
        import_votes_csv(catalog, config, path, vote_year=2025, limit=limit)


def test_import_votes_csv_custom_name(config, catalog: Catalog, tmp_path: Path) -> None:
    path = _votes_csv(tmp_path / "votes.csv", "BV1demo00001,1\n")
    import_votes_csv(catalog, config, path, vote_year=2024, list_name="2024 年度精选")
    list_row = catalog.list_lists()[0]
    assert list_row["list_key"] == "votes_2024"
    assert list_row["name"] == "2024 年度精选"
    assert list_row["source_path"].endswith("votes.csv")


# ---------------------------------------------------------------- BV 号提取


def test_extract_bvids_dedup_and_order() -> None:
    text = (
        "链接 https://www.bilibili.com/video/BV1demo00001 和 BV1demo00002，"
        "再提一次 BV1demo00001。"
        'HTML: <a href="https://www.bilibili.com/video/BV1demo00003?p=2">x</a>'
    )
    assert extract_bvids(text) == ["BV1demo00001", "BV1demo00002", "BV1demo00003"]


def test_extract_bvids_ignores_noise() -> None:
    text = "BV号：BV1demo001 av12345 BW1demo00001 ABV1demo00001 BV1demo00001x"
    assert extract_bvids(text) == []
    assert extract_bvids("") == []
    assert extract_bvids("没有视频") == []


def test_extract_bvids_accepts_html_and_plain_text() -> None:
    html = '<div class="item"><a href="//www.bilibili.com/video/BV1demo00001/">A</a></div>'
    assert extract_bvids(html) == ["BV1demo00001"]
    assert extract_bvids("BV1demo00001 BV1demo00001") == ["BV1demo00001"]


# ---------------------------------------------------------------- 周报导入


def test_import_weekly_bvids(config, catalog: Catalog) -> None:
    summary = import_weekly(
        catalog,
        config,
        list_key="weekly_2025_w01",
        bvids=["BV1demo00001", "BV1demo00002", "BV1demo00001"],
    )
    assert summary.works_created == 2
    assert summary.appearances == 2
    assert summary.bvid_count == 2

    lists = catalog.list_lists()
    assert len(lists) == 1
    assert lists[0]["list_key"] == "weekly_2025_w01"
    assert lists[0]["kind"] == "weekly"
    assert lists[0]["name"] == "周报 weekly_2025_w01"
    assert lists[0]["url"] is None

    again = import_weekly(
        catalog,
        config,
        list_key="weekly_2025_w01",
        bvids=["BV1demo00001", "BV1demo00002"],
    )
    assert again.works_created == 0
    assert again.works_existing == 2
    assert len(catalog.list_lists()) == 1
    assert len(catalog.appearance_rows()) == 2


def test_import_weekly_html(config, catalog: Catalog) -> None:
    html = (
        "<html><body>"
        "<div>本期周报推荐：<a href='/video/BV1demo00001'>作品A</a></div>"
        "<div>以及 BV1demo00002 和 BV1demo00003</div>"
        "</body></html>"
    )
    summary = import_weekly(
        catalog,
        config,
        list_key="weekly_2025_w02",
        html=html,
        name="2025 第 2 期周报",
    )
    assert summary.bvid_count == 3
    list_row = catalog.list_lists()[0]
    assert list_row["name"] == "2025 第 2 期周报"


def test_import_weekly_url_with_injected_fetcher(config, catalog: Catalog) -> None:
    calls: list[str] = []

    def fetcher(url: str) -> str:
        calls.append(url)
        return "推荐 BV1demo00001 以及 BV1demo00002"

    summary = import_weekly(
        catalog,
        config,
        list_key="weekly_2025_w03",
        url="https://example.com/weekly/2025-w03",
        fetcher=fetcher,
    )
    assert calls == ["https://example.com/weekly/2025-w03"]
    assert summary.bvid_count == 2
    list_row = catalog.list_lists()[0]
    assert list_row["url"] == "https://example.com/weekly/2025-w03"


def test_import_weekly_url_fetcher_empty(config, catalog: Catalog) -> None:
    def fetcher(url: str) -> str:
        return "   "

    with pytest.raises(IngestError, match="内容为空"):
        import_weekly(
            catalog,
            config,
            list_key="weekly_2025_w04",
            url="https://example.com/empty",
            fetcher=fetcher,
        )


def test_import_weekly_requires_input(config, catalog: Catalog) -> None:
    with pytest.raises(IngestError, match="bvids、html 或 url"):
        import_weekly(catalog, config, list_key="weekly_2025_w05")


def test_import_weekly_invalid_bvids(config, catalog: Catalog) -> None:
    with pytest.raises(IngestError, match="格式非法"):
        import_weekly(
            catalog,
            config,
            list_key="weekly_2025_w06",
            bvids=["BV1demo00001", "not-a-bv"],
        )
    with pytest.raises(IngestError, match="BV 号列表"):
        import_weekly(
            catalog,
            config,
            list_key="weekly_2025_w06",
            bvids="BV1demo00001",  # type: ignore[arg-type]
        )
    with pytest.raises(IngestError, match="未能从"):
        import_weekly(catalog, config, list_key="weekly_2025_w06", html="没有 BV")


def test_import_weekly_requires_list_key(config, catalog: Catalog) -> None:
    with pytest.raises(IngestError, match="list_key"):
        import_weekly(catalog, config, list_key="  ", bvids=["BV1demo00001"])


# ---------------------------------------------------------------- 示例文件


def test_example_votes_file_is_parseable(repo_root: Path) -> None:
    path = repo_root / "input" / "lists" / "votes_2025.example.csv"
    assert path.is_file()
    rows = parse_votes_csv(path, vote_year=2025)
    assert len(rows) == 5
    assert len({row.bvid for row in rows}) == 5
    for row in rows:
        assert BVID_RE.fullmatch(row.bvid)
        assert row.part >= 1
        assert row.rank is not None and row.rank >= 1
        assert row.votes is not None and row.votes >= 0


def test_parse_real_votes_csv_89_rows(repo_root: Path) -> None:
    """迭代 B1 真实选票 CSV：89 行全部解析、无丢行、标题 strip 正确。"""

    path = repo_root / "input" / "lists" / "zuozhe_votes_2025.csv"
    if not path.is_file():
        pytest.skip("真实选票 CSV 不存在（属于本地运行数据，不随仓库分发）")
    rows = parse_votes_csv(path, vote_year=2025)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))

    assert len(raw_rows) == 89
    assert len(rows) == 89
    assert len({row.bvid for row in rows}) == 89
    for parsed, source in zip(rows, raw_rows, strict=True):
        assert parsed.bvid == source["BV号"].strip()
        assert BVID_RE.fullmatch(parsed.bvid)
        assert parsed.part == 1
        assert parsed.rank is not None and parsed.rank >= 0
        assert parsed.votes is not None and parsed.votes >= 0
        assert parsed.title == (source["作品标题"] or "").strip()
        assert parsed.title  # 真实数据每行都有标题
        assert parsed.title == parsed.title.strip()
    assert rows[0].title == "Slow Dance w/u"
    assert rows[-1].title == "孩又"


def test_example_votes_file_can_be_imported(config, catalog: Catalog, repo_root: Path) -> None:
    path = repo_root / "input" / "lists" / "votes_2025.example.csv"
    summary = import_votes_csv(catalog, config, path, vote_year=2025)
    assert summary.works_created == 5
    assert summary.appearances == 5
    assert len(catalog.list_lists()) == 1


# ---------------------------------------------------------------- CLI


def test_cli_scan_local(config, capsys: pytest.CaptureFixture[str]) -> None:
    (config.paths.input_local / "a.mp4").write_bytes(b"x")
    rc = ingest_main(["--config", str(config.path), "scan-local"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "本地视频扫描完成" in captured.out
    assert "新建作品 1 个" in captured.out


def test_cli_config_after_subcommand(config, capsys: pytest.CaptureFixture[str]) -> None:
    rc = ingest_main(["scan-local", "--config", str(config.path)])
    assert rc == 0
    assert "本地视频扫描完成" in capsys.readouterr().out


def test_cli_import_votes(config, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _votes_csv(tmp_path / "votes.csv", "BV1demo00001,1,1,10\n")
    rc = ingest_main(
        ["--config", str(config.path), "import-votes", str(path), "--year", "2025"]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "选票导入完成" in captured.out
    assert "新建 1" in captured.out
    with Catalog(config.paths.db_path) as cat:
        cat.initialize()
        assert len(cat.list_lists()) == 1
        assert len(cat.list_works(source="bilibili")) == 1


def test_cli_import_votes_limit(
    config, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_csv(
        tmp_path / "votes.csv",
        "BV号,排名,票数,作品标题\n"
        "BV1demo00001,1,10,作品甲\n"
        "BV1demo00002,2,9,作品乙\n"
        "BV1demo00003,3,8,作品丙\n",
    )
    rc = ingest_main(
        [
            "--config",
            str(config.path),
            "import-votes",
            str(path),
            "--year",
            "2025",
            "--limit",
            "2",
        ]
    )
    assert rc == 0
    assert "作品 2 个" in capsys.readouterr().out
    with Catalog(config.paths.db_path) as cat:
        cat.initialize()
        assert len(cat.list_works(source="bilibili")) == 2


def test_cli_import_weekly_bvids(config, capsys: pytest.CaptureFixture[str]) -> None:
    rc = ingest_main(
        [
            "--config",
            str(config.path),
            "import-weekly",
            "--list-key",
            "weekly_2025_w01",
            "--bvids",
            "BV1demo00001",
            "BV1demo00002",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "周报导入完成" in captured.out
    with Catalog(config.paths.db_path) as cat:
        cat.initialize()
        assert cat.list_lists()[0]["kind"] == "weekly"


def test_cli_import_weekly_html_file(
    config, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    html_path = tmp_path / "weekly.html"
    html_path.write_text("<div>推荐 BV1demo00001 与 BV1demo00002</div>", encoding="utf-8")
    rc = ingest_main(
        [
            "--config",
            str(config.path),
            "import-weekly",
            "--list-key",
            "weekly_2025_w09",
            "--html",
            str(html_path),
        ]
    )
    assert rc == 0
    assert "周报导入完成" in capsys.readouterr().out


def test_cli_import_weekly_bad_html_encoding(
    config, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    html_path = tmp_path / "weekly_gbk.html"
    html_path.write_bytes("<div>推荐 BV1demo00001</div>".encode("gbk"))
    rc = ingest_main(
        [
            "--config",
            str(config.path),
            "import-weekly",
            "--list-key",
            "weekly_2025_w10",
            "--html",
            str(html_path),
        ]
    )
    assert rc == 1
    assert "UTF-8" in capsys.readouterr().err


def test_cli_error_returns_exit_code(
    config, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "missing.csv"
    rc = ingest_main(
        ["--config", str(config.path), "import-votes", str(missing), "--year", "2025"]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "错误" in captured.err
    assert "不存在" in captured.err


def test_cli_rejects_bad_subcommand(config) -> None:
    with pytest.raises(SystemExit) as info:
        ingest_main(["--config", str(config.path), "no-such-command"])
    assert info.value.code == 2
