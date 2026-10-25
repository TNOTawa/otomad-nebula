"""nebula.identity（B1 身份关联）的离线单元测试。

全部测试使用假 BilibiliClient，不访问网络、不加载模型。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nebula.config import Config
from nebula.db import Catalog
from nebula.download import BilibiliClient, BilibiliMeta
from nebula.errors import DownloadError, IngestError
from nebula.identity import (
    MANUAL_LINKS_FILENAME,
    IdentityCandidate,
    candidates_from_manual_links,
    find_candidates,
    link_candidates,
    load_manual_links,
    normalize_title,
)
from nebula.ingest import VIDEO_EXTENSIONS, import_votes_csv, parse_votes_csv
from nebula.ingest import main as ingest_main

BVID_CHROME = "BV1GvYZznES3"
BVID_NICE = "BV1dAVYziE5X"
BVID_OTHER = "BV1AAAAAAAA1"
BVID_SECOND = "BV1BBBBBBBB2"


# ---------------------------------------------------------------- 假对象与夹具


class _StaticResponse:
    """假 requests.Response：只实现 BilibiliClient 用到的最小接口。"""

    status_code = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class _StaticSession:
    """假 requests.Session：固定返回同一份 API 响应（离线）。"""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> _StaticResponse:
        self.calls.append((url, dict(kwargs)))
        return _StaticResponse(self.payload)


class FakeBilibiliClient:
    """假 Bilibili 客户端：按 bvid 返回预设元数据，并记录调用。"""

    def __init__(self, metas: dict[str, Any]) -> None:
        self.metas = dict(metas)
        self.calls: list[tuple[str, int]] = []
        self.closed = False

    def fetch_meta(self, bvid: str, *, part: int = 1):
        self.calls.append((bvid, part))
        value = self.metas.get(bvid)
        if value is None:
            raise DownloadError(f"fake 未配置 bvid: {bvid}")
        if isinstance(value, Exception):
            raise value
        return value

    def close(self) -> None:
        self.closed = True


def _meta(
    bvid: str,
    title: str,
    duration_s: float,
    *,
    author_uid: str | None = "10001",
    author_name: str | None = "测试UP主",
    publish_date: str | None = "2025-01-01",
) -> BilibiliMeta:
    return BilibiliMeta(
        bvid=bvid,
        title=title,
        author_uid=author_uid,
        author_name=author_name,
        duration_s=duration_s,
        publish_date=publish_date,
        cover_url=None,
        page_count=1,
    )


def _write_votes_csv(path: Path, entries: list[tuple[str, str]]) -> Path:
    """写一个中文列名、UTF-8-BOM 的选票 CSV（标题支持多行）。"""

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["BV号", "年份", "排名", "票数", "作品标题"])
        for index, (bvid, title) in enumerate(entries, start=1):
            writer.writerow([bvid, 2025, index, 10 + index, title])
    return path


def _setup_list(
    catalog: Catalog, config: Config, tmp_path: Path, entries: list[tuple[str, str]]
) -> dict[str, int]:
    """导入选票 CSV，返回 bvid -> bilibili work_id。"""

    csv_path = _write_votes_csv(tmp_path / "votes.csv", entries)
    import_votes_csv(catalog, config, csv_path, vote_year=2025)
    result: dict[str, int] = {}
    for bvid, _ in entries:
        work = catalog.get_work_by_bvid(bvid)
        assert work is not None
        result[bvid] = int(work["id"])
    return result


def _add_local_work(
    catalog: Catalog,
    config: Config,
    rel_path: str = "【音EDM合作】Chromeme.mp4",
    *,
    duration_s: float | None = 1517.7,
) -> int:
    video_path = config.paths.input_local / Path(rel_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"fake-local-video")
    work_id, _ = catalog.upsert_local_work(rel_path)
    if duration_s is not None:
        catalog.update_work_metadata(work_id, duration_s=duration_s, has_video=1, has_audio=1)
    return work_id


def _register_version(catalog: Catalog, config: Config) -> str:
    version = config.embedding_version
    if catalog.get_embedding_version(version) is None:
        catalog.register_embedding_version(
            version,
            model_checkpoint="test-checkpoint",
            model_revision=None,
            frames=16,
            clip_seconds=8.0,
            clips_per_work=5,
            audio_sample_rate=48000,
            params={"test": True},
        )
    return version


def _register_local_vectors(
    catalog: Catalog, config: Config, work_id: int, version: str
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "work_joint": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "work_audio": np.array([0.10, 0.20, 0.30, 0.40], dtype=np.float32),
        "work_video": np.array([0.50, 0.60, 0.70, 0.80], dtype=np.float32),
        "clip_joint": np.arange(8, dtype=np.float32).reshape(2, 4),
        "clip_audio": np.arange(8, 16, dtype=np.float32).reshape(2, 4),
        "clip_video": np.arange(16, 24, dtype=np.float32).reshape(2, 4),
    }
    work_dir = config.paths.embeddings / version / f"{work_id:06d}"
    work_dir.mkdir(parents=True, exist_ok=True)
    for kind, array in arrays.items():
        path = work_dir / f"{kind}.npy"
        np.save(path, array)
        catalog.register_embedding(
            work_id,
            version=version,
            kind=kind,
            path=config.data_relpath(path),
            dim=4,
            n_clips=2 if kind.startswith("clip") else None,
        )
    catalog.replace_clips(work_id, version=version, clips=[(0, 0.0, 8.0), (1, 10.0, 8.0)])
    return arrays


def _manual_candidate(
    local_id: int,
    bilibili_id: int,
    *,
    accepted: bool = True,
    local_duration: float | None = 100.0,
    bilibili_duration: float | None = 100.0,
    reason: str = "测试候选",
) -> IdentityCandidate:
    return IdentityCandidate(
        local_work_id=local_id,
        bilibili_work_id=bilibili_id,
        local_title="本地标题",
        bilibili_title="B站标题",
        local_duration_s=local_duration,
        bilibili_duration_s=bilibili_duration,
        method="manual",
        confidence=1.0,
        accepted=accepted,
        reason=reason,
    )


# ---------------------------------------------------------------- 标题规范化


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Hello World", "helloworld"),
        ("  A B\tC\nD  ", "abcd"),
        ("ＡＢＣ（１２３）", "abc(123)"),
        ("【括号】", "[括号]"),
        ("全角：！？，；", "全角:!?,;"),
        ("“双引号”", '"双引号"'),
        ("‘单引号’", "'单引号'"),
        ("「书名」", '"书名"'),
        ("～", "~"),
        ("ＡＢＣ", "abc"),
        ("Straße", "strasse"),
        ("A\u200bB", "AB"),
        ("♿️", "♿"),
    ],
)
def test_normalize_title_equivalences(left: str, right: str) -> None:
    assert normalize_title(left) == normalize_title(right)


def test_normalize_title_keeps_different_titles_apart() -> None:
    assert normalize_title("作品一") != normalize_title("作品二")
    assert normalize_title("【A】B") != normalize_title("[A]C")


@pytest.mark.parametrize("value", ["", "   ", "\n\t ", "\u200b"])
def test_normalize_title_rejects_empty(value: str) -> None:
    with pytest.raises(IngestError):
        normalize_title(value)


def test_normalize_title_rejects_non_string() -> None:
    with pytest.raises(IngestError):
        normalize_title(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------- find_candidates


def _chromeme_env(config: Config, catalog: Catalog, tmp_path: Path):
    """一部标题与时长都匹配的本地作品 + 一部时长不匹配的作品。"""

    bili_ids = _setup_list(
        catalog,
        config,
        tmp_path,
        [
            (BVID_CHROME, "【音EDM合作】Chromeme"),
            (BVID_NICE, "【NICE爷爷】My story"),
            (BVID_OTHER, "完全不同的作品"),
        ],
    )
    local_chrome = _add_local_work(
        catalog, config, "【音EDM合作】Chromeme.mp4", duration_s=1517.7
    )
    local_nice = _add_local_work(catalog, config, "【NICE爷爷】My story.mp4", duration_s=177.5)
    client = FakeBilibiliClient(
        {
            BVID_CHROME: _meta(BVID_CHROME, "【音EDM合作】Chromeme", 1518),
            BVID_NICE: _meta(BVID_NICE, "【NICE爷爷】My story", 500),
            BVID_OTHER: _meta(BVID_OTHER, "完全不同的作品", 200),
        }
    )
    return bili_ids, local_chrome, local_nice, client


def test_find_candidates_accepts_title_and_duration_match(
    config: Config, catalog: Catalog, tmp_path: Path
) -> None:
    bili_ids, local_chrome, _, client = _chromeme_env(config, catalog, tmp_path)

    candidates = find_candidates(catalog, config, client=client)

    accepted = [c for c in candidates if c.accepted]
    assert len(accepted) == 1
    candidate = accepted[0]
    assert candidate.local_work_id == local_chrome
    assert candidate.bilibili_work_id == bili_ids[BVID_CHROME]
    assert candidate.local_title == "【音EDM合作】Chromeme"
    assert candidate.bilibili_title == "【音EDM合作】Chromeme"
    assert candidate.local_duration_s == pytest.approx(1517.7)
    assert candidate.bilibili_duration_s == pytest.approx(1518.0)
    assert candidate.method == "title+duration"
    assert candidate.confidence == pytest.approx(0.99)
    assert "时长差" in candidate.reason
    assert client.closed is False  # 注入的 client 由调用方负责关闭

    # 元数据补全：所有榜单作品（含未匹配的）都被真实元数据刷新
    chrome_work = catalog.get_work(bili_ids[BVID_CHROME])
    assert chrome_work["title"] == "【音EDM合作】Chromeme"
    assert chrome_work["duration_s"] == pytest.approx(1518.0)
    assert chrome_work["author_uid"] == "10001"
    assert chrome_work["author_name"] == "测试UP主"
    assert chrome_work["publish_date"] == "2025-01-01"
    other_work = catalog.get_work(bili_ids[BVID_OTHER])
    assert other_work["title"] == "完全不同的作品"
    assert other_work["duration_s"] == pytest.approx(200.0)
    # 每个作品只请求一次
    assert sorted(bvid for bvid, _ in client.calls) == sorted(
        [BVID_CHROME, BVID_NICE, BVID_OTHER]
    )


def test_find_candidates_rejects_duration_mismatch(
    config: Config, catalog: Catalog, tmp_path: Path
) -> None:
    _, _, local_nice, client = _chromeme_env(config, catalog, tmp_path)

    candidates = find_candidates(catalog, config, client=client)

    rejected = [c for c in candidates if c.local_work_id == local_nice]
    assert len(rejected) == 1
    assert rejected[0].accepted is False
    assert rejected[0].bilibili_duration_s == pytest.approx(500.0)
    assert "时长差" in rejected[0].reason
    assert "容差" in rejected[0].reason
    assert rejected[0].confidence == pytest.approx(0.0)


def test_find_candidates_title_mismatch_produces_no_pair(
    config: Config, catalog: Catalog, tmp_path: Path
) -> None:
    bili_ids = _setup_list(
        catalog, config, tmp_path, [(BVID_CHROME, "【音EDM合作】Chromeme")]
    )
    local_id = _add_local_work(catalog, config, "完全无关的本地视频.mp4", duration_s=100.0)
    client = FakeBilibiliClient(
        {BVID_CHROME: _meta(BVID_CHROME, "【音EDM合作】Chromeme", 100)}
    )
    candidates = find_candidates(catalog, config, client=client)
    assert candidates == []
    assert catalog.get_work(bili_ids[BVID_CHROME])["title"] == "【音EDM合作】Chromeme"
    assert catalog.get_work(local_id)["duration_s"] == pytest.approx(100.0)


def test_find_candidates_meta_failure_rejected(
    config: Config, catalog: Catalog, tmp_path: Path
) -> None:
    bili_ids = _setup_list(
        catalog, config, tmp_path, [(BVID_CHROME, "【音EDM合作】Chromeme")]
    )
    local_id = _add_local_work(catalog, config, "【音EDM合作】Chromeme.mp4", duration_s=1517.7)
    client = FakeBilibiliClient({BVID_CHROME: DownloadError("网络不可用")})

    candidates = find_candidates(catalog, config, client=client)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.local_work_id == local_id
    assert candidate.bilibili_work_id == bili_ids[BVID_CHROME]
    assert candidate.accepted is False
    assert "无法获取 Bilibili 元数据" in candidate.reason
    assert candidate.bilibili_duration_s is None


def test_find_candidates_missing_local_duration_rejected(
    config: Config, catalog: Catalog, tmp_path: Path
) -> None:
    _setup_list(catalog, config, tmp_path, [(BVID_CHROME, "【音EDM合作】Chromeme")])
    local_id = _add_local_work(
        catalog, config, "【音EDM合作】Chromeme.mp4", duration_s=None
    )
    client = FakeBilibiliClient(
        {BVID_CHROME: _meta(BVID_CHROME, "【音EDM合作】Chromeme", 1518)}
    )
    candidates = find_candidates(catalog, config, client=client)
    assert len(candidates) == 1
    assert candidates[0].local_work_id == local_id
    assert candidates[0].accepted is False
    assert "缺少时长" in candidates[0].reason


def test_find_candidates_prefers_closest_duration(
    config: Config, catalog: Catalog, tmp_path: Path
) -> None:
    _setup_list(catalog, config, tmp_path, [(BVID_CHROME, "同名作品")])
    local_far = _add_local_work(catalog, config, "同名作品-远.mp4", duration_s=100.0)
    local_near = _add_local_work(catalog, config, "同名作品.mp4", duration_s=103.0)
    # 两段本地文件名不同，但把第二部作品的标题改成完全相同的"同名作品"
    catalog.update_work_metadata(local_far, title="同名作品")
    catalog.update_work_metadata(local_near, title="同名作品")
    client = FakeBilibiliClient({BVID_CHROME: _meta(BVID_CHROME, "同名作品", 102)})

    candidates = find_candidates(catalog, config, client=client)

    accepted = [c for c in candidates if c.accepted]
    assert len(accepted) == 1
    assert accepted[0].local_work_id == local_near  # 103 vs 102 差 1s
    rejected = [c for c in candidates if not c.accepted]
    assert len(rejected) == 1
    assert rejected[0].local_work_id == local_far
    assert "多部本地候选" in rejected[0].reason


def test_find_candidates_matches_meta_title_when_csv_title_differs(
    config: Config, catalog: Catalog, tmp_path: Path
) -> None:
    bili_ids = _setup_list(catalog, config, tmp_path, [(BVID_CHROME, "旧标题")])
    local_id = _add_local_work(catalog, config, "真实标题.mp4", duration_s=100.0)
    client = FakeBilibiliClient({BVID_CHROME: _meta(BVID_CHROME, "真实标题", 100.5)})

    candidates = find_candidates(catalog, config, client=client)

    assert len(candidates) == 1
    assert candidates[0].local_work_id == local_id
    assert candidates[0].bilibili_work_id == bili_ids[BVID_CHROME]
    assert candidates[0].accepted is True
    assert "B站标题" in candidates[0].reason
    assert catalog.get_work(bili_ids[BVID_CHROME])["title"] == "真实标题"


def test_find_candidates_missing_list_raises(config: Config, catalog: Catalog) -> None:
    with pytest.raises(IngestError, match="榜单不存在"):
        find_candidates(catalog, config, list_key="votes_2099")


def test_find_candidates_skips_already_linked(
    config: Config, catalog: Catalog, tmp_path: Path
) -> None:
    bili_ids, local_chrome, _, client = _chromeme_env(config, catalog, tmp_path)
    candidates = find_candidates(catalog, config, client=client)
    accepted = [c for c in candidates if c.accepted]
    link_candidates(catalog, config, candidates=accepted, copy_vectors=False)

    again = find_candidates(catalog, config, client=client)
    assert all(c.local_work_id != local_chrome for c in again)
    assert catalog.get_canonical_work_id(local_chrome) == bili_ids[BVID_CHROME]


def _multipart_api_payload() -> dict[str, Any]:
    """多分P视频 API 响应：整部 356s，两个分P各 178s（B1 真实 bug 数据形态）。"""

    return {
        "code": 0,
        "message": "0",
        "data": {
            "bvid": BVID_NICE,
            "title": "【NICE爷爷】My story",
            "duration": 356,
            "videos": 2,
            "pubdate": 1735761600,
            "pic": "https://i0.hdslb.com/bfs/archive/cover.jpg",
            "owner": {"mid": 10086, "name": "测试UP主"},
            "pages": [
                {"cid": 1, "page": 1, "duration": 178, "part": "my story V2-"},
                {"cid": 2, "page": 2, "duration": 178, "part": "老版本"},
            ],
        },
    }


def test_find_candidates_multipart_uses_page_duration(
    config: Config, catalog: Catalog, tmp_path: Path
) -> None:
    """回归 BV1dAVYziE5X 类多分P误拒：整部 356s、part1 178s，应与本地 177.5s 匹配。"""

    bili_ids = _setup_list(catalog, config, tmp_path, [(BVID_NICE, "【NICE爷爷】My story")])
    local_id = _add_local_work(
        catalog, config, "【NICE爷爷】My story.mp4", duration_s=177.5
    )
    session = _StaticSession(_multipart_api_payload())
    client = BilibiliClient(config.bilibili, session=session)

    candidates = find_candidates(catalog, config, client=client)

    assert session.calls  # 走真实 BilibiliClient 的解析路径（注入假 session）
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.local_work_id == local_id
    assert candidate.bilibili_work_id == bili_ids[BVID_NICE]
    assert candidate.accepted is True
    assert candidate.local_duration_s == pytest.approx(177.5)
    assert candidate.bilibili_duration_s == pytest.approx(178.0)
    assert "时长差" in candidate.reason
    # 元数据补全落库的是分P时长 178s，而不是整部总时长 356s
    assert catalog.get_work(bili_ids[BVID_NICE])["duration_s"] == pytest.approx(178.0)


# ---------------------------------------------------------------- link_candidates


def test_link_candidates_copies_vectors_and_keeps_local(
    config: Config, catalog: Catalog, tmp_path: Path
) -> None:
    bili_ids, local_chrome, _, client = _chromeme_env(config, catalog, tmp_path)
    candidates = [c for c in find_candidates(catalog, config, client=client) if c.accepted]
    assert len(candidates) == 1
    version = _register_version(catalog, config)
    arrays = _register_local_vectors(catalog, config, local_chrome, version)
    canonical_id = bili_ids[BVID_CHROME]
    local_work_dir = config.paths.embeddings / version / f"{local_chrome:06d}"

    result = link_candidates(catalog, config, candidates=candidates)

    assert result["linked"] == 1
    assert result["copied"] == 1
    assert result["skipped"] == []
    assert result["version"] == version

    # 身份关联与 duplicate 集合
    assert catalog.get_canonical_work_id(local_chrome) == canonical_id
    assert catalog.get_duplicate_work_ids() == {local_chrome}
    links = catalog.get_work_links()
    assert len(links) == 1
    assert links[0]["method"] == "title+duration"
    assert links[0]["confidence"] == pytest.approx(0.99)

    # canonical 侧向量文件内容与本地一致，路径在 canonical 目录下
    canonical_dir = config.paths.embeddings / version / f"{canonical_id:06d}"
    for kind, array in arrays.items():
        embedding = catalog.get_embedding(canonical_id, version=version, kind=kind)
        assert embedding is not None
        assert embedding["path"] == config.data_relpath(canonical_dir / f"{kind}.npy")
        np.testing.assert_array_equal(np.load(canonical_dir / f"{kind}.npy"), array)

    # 片段计划复制（比较 idx/start/duration；work_id 各自不同）
    canonical_clips = catalog.get_clips(canonical_id, version=version)
    assert [(c["idx"], c["start_s"], c["duration_s"]) for c in canonical_clips] == [
        (0, 0.0, 8.0),
        (1, 10.0, 8.0),
    ]
    local_clips = catalog.get_clips(local_chrome, version=version)
    assert [(c["idx"], c["start_s"], c["duration_s"]) for c in local_clips] == [
        (c["idx"], c["start_s"], c["duration_s"]) for c in canonical_clips
    ]

    # canonical 置 completed 并记录 identity 事件
    assert catalog.get_work(canonical_id)["status"] == "completed"
    events = catalog.get_events(canonical_id)
    assert any(event["event"] == "identity" for event in events)

    # 本地视频/本地向量/本地作品记录全部保留
    assert (config.paths.input_local / "【音EDM合作】Chromeme.mp4").is_file()
    assert local_work_dir.is_dir()
    for kind in arrays:
        assert (local_work_dir / f"{kind}.npy").is_file()
        assert catalog.get_embedding(local_chrome, version=version, kind=kind) is not None
    assert catalog.get_work(local_chrome)["source"] == "local"


def test_link_candidates_without_copy_keeps_canonical_pending(
    config: Config, catalog: Catalog, tmp_path: Path
) -> None:
    bili_ids, local_chrome, _, client = _chromeme_env(config, catalog, tmp_path)
    candidates = [c for c in find_candidates(catalog, config, client=client) if c.accepted]
    version = _register_version(catalog, config)
    _register_local_vectors(catalog, config, local_chrome, version)

    result = link_candidates(catalog, config, candidates=candidates, copy_vectors=False)

    assert result["linked"] == 1
    assert result["copied"] == 0
    canonical_id = bili_ids[BVID_CHROME]
    assert catalog.has_embedding(canonical_id, version=version) is False
    assert catalog.get_work(canonical_id)["status"] == "pending"


def test_link_candidates_without_local_vectors(
    config: Config, catalog: Catalog, tmp_path: Path
) -> None:
    bili_ids = _setup_list(catalog, config, tmp_path, [(BVID_CHROME, "作品")])
    local_id = _add_local_work(catalog, config, "作品.mp4", duration_s=100.0)
    candidate = _manual_candidate(local_id, bili_ids[BVID_CHROME])

    result = link_candidates(catalog, config, candidates=[candidate])

    assert result["linked"] == 1
    assert result["copied"] == 0
    assert len(result["skipped"]) == 1
    assert "没有当前版本向量" in result["skipped"][0]["reason"]
    canonical_id = bili_ids[BVID_CHROME]
    assert catalog.get_duplicate_work_ids() == {local_id}
    assert catalog.get_work(canonical_id)["status"] == "pending"


def test_link_candidates_idempotent(
    config: Config, catalog: Catalog, tmp_path: Path
) -> None:
    bili_ids = _setup_list(catalog, config, tmp_path, [(BVID_CHROME, "作品")])
    local_id = _add_local_work(catalog, config, "作品.mp4", duration_s=100.0)
    candidate = _manual_candidate(local_id, bili_ids[BVID_CHROME])

    first = link_candidates(catalog, config, candidates=[candidate], copy_vectors=False)
    second = link_candidates(catalog, config, candidates=[candidate], copy_vectors=False)

    assert first["linked"] == 1
    assert second["linked"] == 0
    assert len(catalog.get_work_links()) == 1
    assert any("建立关联失败" in item["reason"] for item in second["skipped"])


def test_link_candidates_skips_rejected(
    config: Config, catalog: Catalog, tmp_path: Path
) -> None:
    bili_ids = _setup_list(catalog, config, tmp_path, [(BVID_CHROME, "作品")])
    local_id = _add_local_work(catalog, config, "作品.mp4", duration_s=100.0)
    candidate = _manual_candidate(
        local_id, bili_ids[BVID_CHROME], accepted=False, reason="时长差 10s > 容差"
    )

    result = link_candidates(catalog, config, candidates=[candidate])

    assert result["linked"] == 0
    assert len(result["skipped"]) == 1
    assert "未接受候选" in result["skipped"][0]["reason"]
    assert catalog.get_work_links() == []


# ---------------------------------------------------------------- 手动关联清单


def _write_manual_links(config: Config, text: str) -> Path:
    path = config.paths.input_lists / MANUAL_LINKS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")
    return path


def test_load_manual_links_missing_file_returns_empty(
    config: Config, catalog: Catalog
) -> None:
    assert load_manual_links(catalog, config) == []


def test_load_manual_links_maps_paths_and_parts(
    config: Config, catalog: Catalog, tmp_path: Path
) -> None:
    bili_ids = _setup_list(
        catalog, config, tmp_path, [(BVID_CHROME, "作品一"), (BVID_SECOND, "作品二")]
    )
    local_one = _add_local_work(catalog, config, "作品一.mp4", duration_s=100.0)
    local_two = _add_local_work(catalog, config, "子目录\\作品二.mp4", duration_s=200.0)
    # 为 BVID_SECOND 建一个分P 2 记录
    part2_id, _ = catalog.upsert_bilibili_work(BVID_SECOND, part=2, title="作品二 p2")
    _write_manual_links(
        config,
        "# 注释行\n"
        "local_path,bvid,part\n"
        "\n"
        "作品一.mp4,BV1GvYZznES3,1\n"
        "子目录\\\\作品二.mp4,BV1BBBBBBBB2,2\n",
    )

    links = load_manual_links(catalog, config)

    assert links == [(local_one, bili_ids[BVID_CHROME]), (local_two, part2_id)]


def test_load_manual_links_last_row_wins(
    config: Config, catalog: Catalog, tmp_path: Path
) -> None:
    bili_ids = _setup_list(
        catalog, config, tmp_path, [(BVID_CHROME, "作品一"), (BVID_SECOND, "作品二")]
    )
    local_id = _add_local_work(catalog, config, "作品一.mp4", duration_s=100.0)
    _write_manual_links(
        config,
        "local_path,bvid\n"
        "作品一.mp4,BV1GvYZznES3\n"
        "作品一.mp4,BV1BBBBBBBB2\n",
    )
    assert load_manual_links(catalog, config) == [(local_id, bili_ids[BVID_SECOND])]


def test_load_manual_links_invalid_data_raises(
    config: Config, catalog: Catalog, tmp_path: Path
) -> None:
    _setup_list(catalog, config, tmp_path, [(BVID_CHROME, "作品一")])
    _add_local_work(catalog, config, "作品一.mp4", duration_s=100.0)
    _write_manual_links(
        config,
        "local_path,bvid\n"
        "作品一.mp4,not-a-bv\n",
    )
    with pytest.raises(IngestError, match="第 2 行"):
        load_manual_links(catalog, config)


def test_load_manual_links_unknown_work_raises(
    config: Config, catalog: Catalog, tmp_path: Path
) -> None:
    _setup_list(catalog, config, tmp_path, [(BVID_CHROME, "作品一")])
    _add_local_work(catalog, config, "作品一.mp4", duration_s=100.0)
    _write_manual_links(
        config,
        "local_path,bvid\n"
        "不存在的本地.mp4,BV1GvYZznES3\n",
    )
    with pytest.raises(IngestError, match="本地作品未登记"):
        load_manual_links(catalog, config)
    _write_manual_links(
        config,
        "local_path,bvid\n"
        "作品一.mp4,BV1AAAAAAAA1\n",
    )
    with pytest.raises(IngestError, match="Bilibili 作品未登记"):
        load_manual_links(catalog, config)


def test_load_manual_links_missing_column_raises(
    config: Config, catalog: Catalog
) -> None:
    _write_manual_links(config, "bvid\nBV1GvYZznES3\n")
    with pytest.raises(IngestError, match="缺少 local_path"):
        load_manual_links(catalog, config)


def test_candidates_from_manual_links(
    config: Config, catalog: Catalog, tmp_path: Path
) -> None:
    bili_ids = _setup_list(catalog, config, tmp_path, [(BVID_CHROME, "作品一")])
    local_id = _add_local_work(catalog, config, "作品一.mp4", duration_s=100.0)

    candidates = candidates_from_manual_links(catalog, [(local_id, bili_ids[BVID_CHROME])])

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.accepted is True
    assert candidate.method == "manual"
    assert candidate.confidence == pytest.approx(1.0)
    assert candidate.local_work_id == local_id
    assert candidate.bilibili_work_id == bili_ids[BVID_CHROME]
    assert candidate.local_title == "作品一"
    assert candidate.bilibili_title == "作品一"
    assert candidate.local_duration_s == pytest.approx(100.0)


# ---------------------------------------------------------------- 真实数据（离线）


def test_real_votes_csv_title_prefilter(repo_root: Path) -> None:
    """真实 2025 选票 CSV 与 33 部本地视频：规范化标题恰好 10 部完全一致。

    不联网、不依赖数据库，只验证 B1 的标题预筛在真实数据上的结果。
    """

    votes_path = repo_root / "input" / "lists" / "zuozhe_votes_2025.csv"
    local_dir = repo_root / "input" / "local"
    if not votes_path.is_file() or not local_dir.is_dir():
        pytest.skip("真实选票 CSV 或本地视频目录不存在（属于本地运行数据）")
    local_videos = [
        path for path in local_dir.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not local_videos:
        pytest.skip("本地视频目录为空")
    local_titles = {normalize_title(path.stem) for path in local_videos}
    rows = parse_votes_csv(votes_path, vote_year=2025)
    matched = [
        row for row in rows if row.title and normalize_title(row.title) in local_titles
    ]
    assert len(matched) == 10
    matched_bvids = {row.bvid for row in matched}
    # 抽查 Lead 提到的几部真实匹配作品
    assert any(row.title and "Chromeme" in row.title for row in matched)
    assert any(row.title and "NICE爷爷" in row.title for row in matched)
    assert len(matched_bvids) == 10


# ---------------------------------------------------------------- CLI link-local


def _patch_empty_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nebula.ingest.find_candidates", lambda *args, **kwargs: [])


def test_cli_link_local_manual_dry_run(
    config: Config,
    catalog: Catalog,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bili_ids = _setup_list(catalog, config, tmp_path, [(BVID_CHROME, "作品一")])
    local_id = _add_local_work(catalog, config, "作品一.mp4", duration_s=100.0)
    _write_manual_links(config, "local_path,bvid\n作品一.mp4,BV1GvYZznES3\n")
    _patch_empty_candidates(monkeypatch)

    code = ingest_main(
        ["--config", str(config.path), "link-local", "--list-key", "votes_2025", "--manual", "--dry-run"]
    )

    assert code == 0
    output = capsys.readouterr().out
    assert "候选 1 条（手动 1 条）" in output
    assert "dry-run" in output
    assert "手动关联清单" in output
    assert catalog.get_work_links() == []
    assert catalog.get_work(bili_ids[BVID_CHROME]) is not None
    assert catalog.get_canonical_work_id(local_id) == local_id  # 未写入


def test_cli_link_local_manual_applies(
    config: Config,
    catalog: Catalog,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bili_ids = _setup_list(catalog, config, tmp_path, [(BVID_CHROME, "作品一")])
    local_id = _add_local_work(catalog, config, "作品一.mp4", duration_s=100.0)
    _write_manual_links(config, "local_path,bvid\n作品一.mp4,BV1GvYZznES3\n")
    _patch_empty_candidates(monkeypatch)

    code = ingest_main(
        ["--config", str(config.path), "link-local", "--manual", "--no-copy"]
    )

    assert code == 0
    output = capsys.readouterr().out
    assert "身份关联完成" in output
    assert "关联 1 条" in output
    assert catalog.get_canonical_work_id(local_id) == bili_ids[BVID_CHROME]
    assert catalog.get_duplicate_work_ids() == {local_id}


def test_cli_link_local_uses_auto_candidates(
    config: Config,
    catalog: Catalog,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bili_ids = _setup_list(catalog, config, tmp_path, [(BVID_CHROME, "作品一")])
    local_id = _add_local_work(catalog, config, "作品一.mp4", duration_s=100.0)
    candidate = _manual_candidate(local_id, bili_ids[BVID_CHROME])
    monkeypatch.setattr("nebula.ingest.find_candidates", lambda *args, **kwargs: [candidate])

    code = ingest_main(["--config", str(config.path), "link-local", "--no-copy"])

    assert code == 0
    output = capsys.readouterr().out
    assert "候选 1 条" in output
    assert "身份关联完成" in output
    assert catalog.get_canonical_work_id(local_id) == bili_ids[BVID_CHROME]


def test_cli_link_local_missing_manual_file(
    config: Config,
    catalog: Catalog,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_empty_candidates(monkeypatch)
    code = ingest_main(["--config", str(config.path), "link-local", "--manual"])
    assert code == 1
    assert "手动关联清单不存在" in capsys.readouterr().err


def test_cli_link_local_no_candidates(
    config: Config,
    catalog: Catalog,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_empty_candidates(monkeypatch)
    code = ingest_main(["--config", str(config.path), "link-local", "--dry-run"])
    assert code == 0
    assert "未找到标题规范化后完全相等的候选" in capsys.readouterr().out
