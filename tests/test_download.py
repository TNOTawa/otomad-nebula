"""nebula.download 的离线单元测试 + 默认跳过的真实网络测试。"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import pytest
import requests

from nebula.download import (
    BILIBILI_VIEW_API,
    BilibiliClient,
    BilibiliMeta,
    download_bilibili,
    http_get_text,
    is_valid_bvid,
)
from nebula.errors import DownloadError

# 真实网络测试默认跳过；设置 NEBULA_NETWORK_TESTS=1 才运行。
NETWORK_ENABLED = os.environ.get("NEBULA_NETWORK_TESTS") == "1"

VALID_BVID = "BV1xx411c7mD"  # 形状合法；解析测试用，不联网时不会真的访问
OTHER_BVID = "BV1234567890"
REAL_BVID = "BV1xx411c7mD"  # 真实网络冒烟测试用的公开 BV 号（仅获取元数据）


# ---------------------------------------------------------------- 假对象


class FakeResponse:
    """模拟 requests.Response 的最小接口。"""

    def __init__(
        self,
        *,
        payload: Any = None,
        text: str = "",
        status_code: int = 200,
        json_error: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.text = text
        self.status_code = status_code
        self.json_error = json_error

    def json(self) -> Any:
        if self.json_error is not None:
            raise self.json_error
        return self.payload


class FakeSession:
    """假 requests.Session：记录调用参数，返回预设响应或抛出预设异常。"""

    def __init__(self, response: FakeResponse | Exception) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, dict(kwargs)))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def close(self) -> None:
        pass


class FakeYDL:
    """假 yt-dlp 对象：按 outtmpl 写出假文件并返回 info 字典。"""

    def __init__(
        self,
        options: dict[str, Any],
        *,
        info: dict[str, Any] | None = None,
        write_file: bool = True,
        error: Exception | None = None,
        with_requested_downloads: bool = True,
    ) -> None:
        self.options = dict(options)
        self.info = dict(info or {})
        self.write_file = write_file
        self.error = error
        self.with_requested_downloads = with_requested_downloads
        self.urls: list[str] = []

    def _target(self) -> Path:
        return Path(str(self.options["outtmpl"]).replace("%(ext)s", "mp4"))

    def extract_info(self, url: str, *, download: bool = True) -> dict[str, Any]:
        self.urls.append(url)
        if self.error is not None:
            raise self.error
        if self.write_file:
            target = self._target()
            target.write_bytes(b"fake-video-bytes")
            if self.with_requested_downloads:
                self.info = {
                    **self.info,
                    "ext": "mp4",
                    "requested_downloads": [{"filepath": str(target)}],
                }
        return self.info

    def prepare_filename(self, info: dict[str, Any]) -> str:
        return str(self._target())


def _make_factory(captured: list[FakeYDL], **ydl_kwargs: Any):
    def factory(options: dict[str, Any]) -> FakeYDL:
        ydl = FakeYDL(options, **ydl_kwargs)
        captured.append(ydl)
        return ydl

    return factory


def _meta_payload(**data_updates: Any) -> dict[str, Any]:
    """构造一个典型的 view 接口响应（发布时间 2025-01-01 20:00 UTC = 北京 01-02 04:00）。"""

    data: dict[str, Any] = {
        "bvid": VALID_BVID,
        "title": "测试视频标题",
        "duration": 100,
        "pubdate": 1735761600,
        "pic": "https://i0.hdslb.com/bfs/archive/cover.jpg",
        "owner": {"mid": 123456, "name": "测试UP主"},
        "pages": [
            {"cid": 111, "page": 1, "duration": 100},
            {"cid": 222, "page": 2, "duration": 42},
        ],
    }
    data.update(data_updates)
    return {"code": 0, "message": "0", "data": data}


# ---------------------------------------------------------------- BV 号校验


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("BV1xx411c7mD", True),
        ("BV1234567890", True),
        (" BV1xx411c7mD ", True),
        ("bv1xx411c7mD", False),
        ("BV1xx411c7m", False),
        ("BV1xx411c7mDD", False),
        ("", False),
        (None, False),
        (123, False),
    ],
)
def test_is_valid_bvid(value: Any, expected: bool) -> None:
    assert is_valid_bvid(value) is expected


# ---------------------------------------------------------------- fetch_meta


def test_fetch_meta_parses_payload(config) -> None:
    session = FakeSession(FakeResponse(payload=_meta_payload()))
    client = BilibiliClient(config.bilibili, session=session)

    meta = client.fetch_meta(VALID_BVID)

    assert meta == BilibiliMeta(
        bvid=VALID_BVID,
        title="测试视频标题",
        author_uid="123456",
        author_name="测试UP主",
        duration_s=100.0,
        publish_date="2025-01-02",  # 北京时间（UTC+8）
        cover_url="https://i0.hdslb.com/bfs/archive/cover.jpg",
        page_count=2,
    )
    url, kwargs = session.calls[0]
    assert url == BILIBILI_VIEW_API
    assert kwargs["params"] == {"bvid": VALID_BVID}
    assert kwargs["headers"]["User-Agent"] == config.bilibili.user_agent
    assert kwargs["timeout"] == config.bilibili.api_timeout_seconds


def test_fetch_meta_second_part_uses_page_duration(config) -> None:
    session = FakeSession(FakeResponse(payload=_meta_payload()))
    meta = BilibiliClient(config.bilibili, session=session).fetch_meta(VALID_BVID, part=2)
    assert meta.duration_s == 42.0
    assert meta.page_count == 2


def test_fetch_meta_api_code_error(config) -> None:
    payload = {"code": -400, "message": "请求错误", "data": None}
    session = FakeSession(FakeResponse(payload=payload))
    with pytest.raises(DownloadError) as info:
        BilibiliClient(config.bilibili, session=session).fetch_meta(VALID_BVID)
    message = str(info.value)
    assert "code=-400" in message
    assert "请求错误" in message


def test_fetch_meta_invalid_json(config) -> None:
    session = FakeSession(FakeResponse(json_error=ValueError("bad json")))
    with pytest.raises(DownloadError, match="JSON"):
        BilibiliClient(config.bilibili, session=session).fetch_meta(VALID_BVID)


def test_fetch_meta_network_error(config) -> None:
    session = FakeSession(requests.ConnectionError("connection refused"))
    with pytest.raises(DownloadError) as info:
        BilibiliClient(config.bilibili, session=session).fetch_meta(VALID_BVID)
    assert "网络错误" in str(info.value)


def test_fetch_meta_http_error(config) -> None:
    session = FakeSession(FakeResponse(payload={}, status_code=502))
    with pytest.raises(DownloadError, match="HTTP 502"):
        BilibiliClient(config.bilibili, session=session).fetch_meta(VALID_BVID)


def test_fetch_meta_missing_data(config) -> None:
    session = FakeSession(FakeResponse(payload={"code": 0, "message": "0"}))
    with pytest.raises(DownloadError, match="缺少 data"):
        BilibiliClient(config.bilibili, session=session).fetch_meta(VALID_BVID)


def test_fetch_meta_missing_title(config) -> None:
    session = FakeSession(FakeResponse(payload=_meta_payload(title="")))
    with pytest.raises(DownloadError, match="缺少标题"):
        BilibiliClient(config.bilibili, session=session).fetch_meta(VALID_BVID)


def test_fetch_meta_invalid_duration(config) -> None:
    payload = _meta_payload()
    payload["data"].pop("duration")
    payload["data"]["pages"] = []
    session = FakeSession(FakeResponse(payload=payload))
    with pytest.raises(DownloadError, match="duration"):
        BilibiliClient(config.bilibili, session=session).fetch_meta(VALID_BVID)


def test_fetch_meta_part_out_of_range(config) -> None:
    session = FakeSession(FakeResponse(payload=_meta_payload()))
    with pytest.raises(DownloadError, match="分P 超出范围"):
        BilibiliClient(config.bilibili, session=session).fetch_meta(VALID_BVID, part=3)


def test_fetch_meta_rejects_invalid_bvid(config) -> None:
    session = FakeSession(FakeResponse(payload=_meta_payload()))
    client = BilibiliClient(config.bilibili, session=session)
    with pytest.raises(DownloadError, match="BV 号格式非法"):
        client.fetch_meta("av12345")
    with pytest.raises(DownloadError, match="分P必须"):
        client.fetch_meta(VALID_BVID, part=0)
    assert session.calls == []


def test_fetch_meta_without_optional_fields(config) -> None:
    payload = _meta_payload(
        owner={"mid": None, "name": None},
        pubdate=None,
        pic=None,
        pages=[],
    )
    session = FakeSession(FakeResponse(payload=payload))
    meta = BilibiliClient(config.bilibili, session=session).fetch_meta(VALID_BVID)
    assert meta.author_uid is None
    assert meta.author_name is None
    assert meta.publish_date is None
    assert meta.cover_url is None
    assert meta.page_count == 1


# ---------------------------------------------------------------- http_get_text


def test_http_get_text_passes_ua_timeout_proxy() -> None:
    session = FakeSession(FakeResponse(text="<html>周报正文</html>"))
    text = http_get_text(
        "https://example.com/weekly",
        user_agent="UA-test",
        timeout_s=5,
        session=session,
    )
    assert text == "<html>周报正文</html>"
    url, kwargs = session.calls[0]
    assert url == "https://example.com/weekly"
    assert kwargs["headers"]["User-Agent"] == "UA-test"
    assert kwargs["timeout"] == 5.0
    assert kwargs["proxies"] is None


def test_http_get_text_forwards_proxy() -> None:
    session = FakeSession(FakeResponse(text="ok"))
    http_get_text(
        "https://example.com/x",
        user_agent="ua",
        timeout_s=3.5,
        proxy="http://127.0.0.1:7890",
        session=session,
    )
    _, kwargs = session.calls[0]
    assert kwargs["proxies"] == {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}


def test_http_get_text_network_error() -> None:
    session = FakeSession(requests.Timeout("timeout"))
    with pytest.raises(DownloadError, match="网络错误"):
        http_get_text("https://example.com", user_agent="ua", timeout_s=1, session=session)


def test_http_get_text_http_error() -> None:
    session = FakeSession(FakeResponse(text="", status_code=503))
    with pytest.raises(DownloadError, match="HTTP 503"):
        http_get_text("https://example.com", user_agent="ua", timeout_s=1, session=session)


@pytest.mark.parametrize(
    ("url", "timeout_s"),
    [("", 1.0), ("https://example.com", 0), ("https://example.com", -1)],
)
def test_http_get_text_validates_arguments(url: str, timeout_s: float) -> None:
    with pytest.raises(DownloadError):
        http_get_text(url, user_agent="ua", timeout_s=timeout_s)


# ---------------------------------------------------------------- download_bilibili


def test_download_bilibili_returns_file_and_options(config, tmp_path: Path) -> None:
    captured: list[FakeYDL] = []
    out_dir = tmp_path / "downloads"
    path = download_bilibili(
        VALID_BVID,
        out_dir=out_dir,
        download=config.download,
        ydl_factory=_make_factory(captured),
    )

    assert path == (out_dir / f"{VALID_BVID}_p1.mp4").resolve()
    assert path.is_file()
    ydl = captured[0]
    assert ydl.urls == [f"https://www.bilibili.com/video/{VALID_BVID}"]
    options = ydl.options
    assert options["format"] == config.download.yt_dlp_format  # 配置自带高度上限时原样使用
    assert "height<=480" in options["format"]
    assert options["retries"] == config.download.retries
    assert options["fragment_retries"] == config.download.retries
    assert options["noplaylist"] is True
    assert options["noprogress"] is True
    assert "cookiefile" not in options
    assert "proxy" not in options


def test_download_bilibili_part_url(config, tmp_path: Path) -> None:
    captured: list[FakeYDL] = []
    path = download_bilibili(
        VALID_BVID,
        part=2,
        out_dir=tmp_path / "dl",
        download=config.download,
        ydl_factory=_make_factory(captured),
    )
    assert captured[0].urls == [f"https://www.bilibili.com/video/{VALID_BVID}?p=2"]
    assert path.name == f"{VALID_BVID}_p2.mp4"


def test_download_bilibili_progress_hook(config, tmp_path: Path) -> None:
    captured: list[FakeYDL] = []

    def on_progress(payload: dict[str, Any]) -> None:
        raise AssertionError("假对象不会真的回调进度")

    download_bilibili(
        VALID_BVID,
        out_dir=tmp_path / "dl",
        download=config.download,
        on_progress=on_progress,
        ydl_factory=_make_factory(captured),
    )
    assert captured[0].options["progress_hooks"] == [on_progress]
    assert captured[0].options["noprogress"] is False


def test_download_bilibili_cookies_missing(config, tmp_path: Path) -> None:
    download = dataclasses.replace(
        config.download, cookies_file=tmp_path / "cookies.txt"
    )
    with pytest.raises(DownloadError, match="cookies 文件不存在"):
        download_bilibili(
            VALID_BVID,
            out_dir=tmp_path / "dl",
            download=download,
            ydl_factory=_make_factory([]),
        )


def test_download_bilibili_cookies_and_proxy_forwarded(config, tmp_path: Path) -> None:
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    download = dataclasses.replace(
        config.download,
        cookies_file=cookies,
        proxy="http://127.0.0.1:7890",
    )
    captured: list[FakeYDL] = []
    download_bilibili(
        VALID_BVID,
        out_dir=tmp_path / "dl",
        download=download,
        ydl_factory=_make_factory(captured),
    )
    options = captured[0].options
    assert options["cookiefile"] == str(cookies.resolve())
    assert options["proxy"] == "http://127.0.0.1:7890"


def test_download_bilibili_format_fallbacks(config, tmp_path: Path) -> None:
    captured: list[FakeYDL] = []

    empty_format = dataclasses.replace(config.download, yt_dlp_format="", max_height=360)
    download_bilibili(
        VALID_BVID,
        out_dir=tmp_path / "a",
        download=empty_format,
        ydl_factory=_make_factory(captured),
    )
    assert captured[0].options["format"] == "bv*[height<=360]+ba/b[height<=360]/b"

    no_height = dataclasses.replace(
        config.download, yt_dlp_format="bv*+ba/b", max_height=240
    )
    captured.clear()
    download_bilibili(
        VALID_BVID,
        out_dir=tmp_path / "b",
        download=no_height,
        ydl_factory=_make_factory(captured),
    )
    assert captured[0].options["format"] == "bv*[height<=240]+ba/b[height<=240]/b"


def test_download_bilibili_factory_error(config, tmp_path: Path) -> None:
    def factory(options: dict[str, Any]) -> FakeYDL:
        raise RuntimeError("factory boom")

    with pytest.raises(DownloadError, match="初始化 yt-dlp 失败"):
        download_bilibili(
            VALID_BVID,
            out_dir=tmp_path / "dl",
            download=config.download,
            ydl_factory=factory,
        )


def test_download_bilibili_extract_error(config, tmp_path: Path) -> None:
    captured: list[FakeYDL] = []
    factory = _make_factory(captured, error=RuntimeError("network boom"))
    with pytest.raises(DownloadError, match="yt-dlp 下载失败"):
        download_bilibili(
            VALID_BVID,
            out_dir=tmp_path / "dl",
            download=config.download,
            ydl_factory=factory,
        )


def test_download_bilibili_no_file(config, tmp_path: Path) -> None:
    captured: list[FakeYDL] = []
    factory = _make_factory(captured, write_file=False, info={"ext": "mp4"})
    with pytest.raises(DownloadError, match="找不到视频文件"):
        download_bilibili(
            VALID_BVID,
            out_dir=tmp_path / "dl",
            download=config.download,
            ydl_factory=factory,
        )


def test_download_bilibili_glob_fallback(config, tmp_path: Path) -> None:
    """yt-dlp 只返回 info、不给出 filepath 时，按固定模板在 out_dir 兜底查找。"""

    captured: list[FakeYDL] = []
    factory = _make_factory(
        captured,
        write_file=True,
        with_requested_downloads=False,
        info={"ext": "mp4"},
    )
    path = download_bilibili(
        VALID_BVID,
        out_dir=tmp_path / "dl",
        download=config.download,
        ydl_factory=factory,
    )
    assert path.name == f"{VALID_BVID}_p1.mp4"
    assert path.is_file()


def test_download_bilibili_ignores_partial_files(config, tmp_path: Path) -> None:
    out_dir = tmp_path / "dl"
    out_dir.mkdir()
    (out_dir / f"{VALID_BVID}_p1.mp4.part").write_bytes(b"partial")
    captured: list[FakeYDL] = []
    factory = _make_factory(captured, write_file=False, info={"ext": "mp4"})
    with pytest.raises(DownloadError, match="找不到视频文件"):
        download_bilibili(
            VALID_BVID,
            out_dir=out_dir,
            download=config.download,
            ydl_factory=factory,
        )


@pytest.mark.parametrize("bvid, part", [("av12345", 1), (VALID_BVID, 0)])
def test_download_bilibili_rejects_invalid_arguments(
    config, tmp_path: Path, bvid: str, part: int
) -> None:
    with pytest.raises(DownloadError):
        download_bilibili(
            bvid,
            part=part,
            out_dir=tmp_path / "dl",
            download=config.download,
            ydl_factory=_make_factory([]),
        )


def test_download_bilibili_out_dir_is_file(config, tmp_path: Path) -> None:
    target = tmp_path / "not-a-dir"
    target.write_bytes(b"x")
    with pytest.raises(DownloadError, match="下载目录不是文件夹"):
        download_bilibili(
            VALID_BVID,
            out_dir=target,
            download=config.download,
            ydl_factory=_make_factory([]),
        )


# ---------------------------------------------------------------- 真实网络（默认跳过）


@pytest.mark.network
@pytest.mark.skipif(
    not NETWORK_ENABLED,
    reason="网络测试默认跳过；设置 NEBULA_NETWORK_TESTS=1 后运行",
)
def test_fetch_meta_real_bilibili(config) -> None:
    """真实 Bilibili view 接口冒烟测试：只取元数据，不下载视频。"""

    with BilibiliClient(config.bilibili) as client:
        meta = client.fetch_meta(REAL_BVID)
    assert meta.bvid == REAL_BVID
    assert meta.title.strip()
    assert meta.duration_s > 0
    assert meta.page_count >= 1
    assert meta.author_name is not None
