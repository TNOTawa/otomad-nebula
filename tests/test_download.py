"""nebula.download 的离线单元测试 + 默认跳过的真实网络测试。"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import requests

from nebula import download as download_module
from nebula.download import (
    BILIBILI_VIEW_API,
    BilibiliClient,
    BilibiliMeta,
    _playurl_qn,
    download_bilibili,
    download_bilibili_playurl,
    http_get_text,
    is_valid_bvid,
)
from nebula.errors import DownloadError

# 真实网络测试默认跳过；设置 NEBULA_NETWORK_TESTS=1 才运行。
NETWORK_ENABLED = os.environ.get("NEBULA_NETWORK_TESTS") == "1"

VALID_BVID = "BV1xx411c7mD"  # 形状合法；解析测试用，不联网时不会真的访问
OTHER_BVID = "BV1234567890"
REAL_BVID = "BV1xx411c7mD"  # 真实网络冒烟测试用的公开 BV 号（仅获取元数据）
# B1 回归真实案例：【NICE爷爷】My story 是多分P视频（整部 356s，P1/P2 各 178s）。
REAL_MULTIPART_BVID = "BV1dAVYziE5X"


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


def _multipart_payload() -> dict[str, Any]:
    """多分P视频响应：整部总时长 356s，两个分P各 178s（B1 真实 bug 形态）。"""

    payload = _meta_payload()
    data = payload["data"]
    data["title"] = "多分P作品"
    data["duration"] = 356
    data["videos"] = 2
    data["pages"] = [
        {"cid": 111, "page": 1, "duration": 178, "part": "my story V2-"},
        {"cid": 222, "page": 2, "duration": 178, "part": "老版本"},
    ]
    return payload


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


@pytest.mark.parametrize(
    ("part", "expected_part_title"),
    [(1, "my story V2-"), (2, "老版本")],
)
def test_fetch_meta_multipart_uses_page_duration(
    config, part: int, expected_part_title: str
) -> None:
    """回归：多分P视频必须取该分P时长，而不是 data["duration"] 整部总时长。"""

    session = FakeSession(FakeResponse(payload=_multipart_payload()))
    meta = BilibiliClient(config.bilibili, session=session).fetch_meta(VALID_BVID, part=part)

    assert meta.duration_s == pytest.approx(178.0)
    assert meta.page_count == 2
    assert meta.title == "多分P作品"  # 整部标题保留
    assert meta.part_title == expected_part_title


def test_fetch_meta_multipart_without_pages_falls_back_to_total(config) -> None:
    payload = _multipart_payload()
    payload["data"]["pages"] = []
    session = FakeSession(FakeResponse(payload=payload))
    meta = BilibiliClient(config.bilibili, session=session).fetch_meta(VALID_BVID)
    assert meta.duration_s == pytest.approx(356.0)
    assert meta.page_count == 1
    assert meta.part_title is None


def test_fetch_meta_multipart_invalid_page_duration_falls_back(config) -> None:
    payload = _multipart_payload()
    payload["data"]["pages"][0]["duration"] = 0
    session = FakeSession(FakeResponse(payload=payload))
    meta = BilibiliClient(config.bilibili, session=session).fetch_meta(VALID_BVID, part=1)
    assert meta.duration_s == pytest.approx(356.0)


def test_fetch_meta_multipart_part_out_of_range(config) -> None:
    session = FakeSession(FakeResponse(payload=_multipart_payload()))
    with pytest.raises(DownloadError, match="分P 超出范围"):
        BilibiliClient(config.bilibili, session=session).fetch_meta(VALID_BVID, part=3)


def test_fetch_meta_single_page_uses_page_duration(config) -> None:
    """pages 只有一个且时长与总时长不同时，也以该分P时长为准。"""

    payload = _meta_payload()
    payload["data"]["duration"] = 999
    payload["data"]["pages"] = [
        {"cid": 1, "page": 1, "duration": 321, "part": "正片"},
    ]
    session = FakeSession(FakeResponse(payload=payload))
    meta = BilibiliClient(config.bilibili, session=session).fetch_meta(VALID_BVID)
    assert meta.duration_s == pytest.approx(321.0)
    assert meta.page_count == 1
    assert meta.part_title == "正片"


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
            api_fallback=False,
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
            api_fallback=False,
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
            api_fallback=False,
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
            api_fallback=False,
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


# ---------------------------------------------------------------- playurl 回退（离线）


REAL_FALLBACK_BVID = "BV1LufpYGEb5"  # 全片批次里 yt-dlp 解析失败的真实案例


class _JsonResponse:
    """假 JSON API 响应。"""

    status_code = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def close(self) -> None:
        pass


class _BytesStreamResponse:
    """假流式媒体响应。"""

    status_code = 200

    def __init__(self, content: bytes) -> None:
        self._content = content

    def iter_content(self, chunk_size: int = 1):
        for index in range(0, len(self._content), chunk_size):
            yield self._content[index : index + chunk_size]

    def close(self) -> None:
        pass


class _RoutedSession:
    """按 URL 前缀路由的假 requests.Session，记录调用参数。"""

    def __init__(self, routes: list[tuple[str, Any]]) -> None:
        self.routes = list(routes)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> Any:
        self.calls.append((url, dict(kwargs)))
        for prefix, value in self.routes:
            if not url.startswith(prefix):
                continue
            if callable(value) and not isinstance(value, (_JsonResponse, _BytesStreamResponse)):
                return value()
            return value
        raise requests.ConnectionError(f"未配置的 URL: {url}")

    def close(self) -> None:
        pass


def _view_payload() -> dict[str, Any]:
    return {
        "code": 0,
        "message": "OK",
        "data": {"pages": [{"cid": 28112061734, "page": 1, "duration": 208}]},
    }


def _dash_payload() -> dict[str, Any]:
    return {
        "code": 0,
        "message": "OK",
        "data": {
            "quality": 32,
            "format": "flv480",
            "dash": {
                "video": [
                    {
                        "id": 32,
                        "height": 480,
                        "codec": "hev1.1.6.L120.90",
                        "bandwidth": 900,
                        "baseUrl": "https://media.example/hevc480.m4s",
                    },
                    {
                        "id": 32,
                        "height": 480,
                        "codec": "avc1.64001F",
                        "bandwidth": 500,
                        "baseUrl": "https://media.example/avc480.m4s",
                        "backupUrl": ["https://media.example/avc480_backup.m4s"],
                    },
                    {
                        "id": 16,
                        "height": 360,
                        "codec": "avc1.64001E",
                        "bandwidth": 100,
                        "baseUrl": "https://media.example/avc360.m4s",
                    },
                ],
                "audio": [
                    {
                        "id": 30216,
                        "bandwidth": 50,
                        "baseUrl": "https://media.example/audio_low.m4s",
                    },
                    {
                        "id": 30280,
                        "bandwidth": 200,
                        "baseUrl": "https://media.example/audio_high.m4s",
                    },
                ],
            },
        },
    }


def _fake_ffmpeg_runner(captured: list[list[str]]):
    def runner(command, *, capture_output: bool = False, timeout: float | None = None):
        command_list = [str(item) for item in command]
        captured.append(command_list)
        target = Path(command_list[-1])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"MUXED")
        return subprocess.CompletedProcess(command_list, 0, b"", b"")

    return runner


@pytest.mark.parametrize(
    ("max_height", "expected_qn"),
    [(360, 16), (480, 32), (720, 64), (1080, 80), (0, 80)],
)
def test_playurl_qn_mapping(max_height: int, expected_qn: int) -> None:
    assert _playurl_qn(max_height) == expected_qn


def test_playurl_dash_download_offline(
    config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _RoutedSession(
        [
            ("https://api.bilibili.com/x/web-interface/view", _JsonResponse(_view_payload())),
            ("https://api.bilibili.com/x/player/playurl", _JsonResponse(_dash_payload())),
            ("https://media.example/avc480.m4s", _BytesStreamResponse(b"VIDEO-DATA")),
            ("https://media.example/audio_high.m4s", _BytesStreamResponse(b"AUDIO-DATA")),
        ]
    )
    ffmpeg_commands: list[list[str]] = []
    monkeypatch.setattr(
        download_module.subprocess, "run", _fake_ffmpeg_runner(ffmpeg_commands)
    )
    out_dir = tmp_path / "dl"

    path = download_bilibili_playurl(
        VALID_BVID,
        out_dir=out_dir,
        download=config.download,
        session=session,
        ffmpeg="ffmpeg",
    )

    assert path == (out_dir / f"{VALID_BVID}_p1.mp4").resolve()
    assert path.read_bytes() == b"MUXED"
    # 临时 m4s / .part 文件全部清理，只剩最终 mp4
    assert sorted(item.name for item in out_dir.iterdir()) == [f"{VALID_BVID}_p1.mp4"]

    media_urls = [url for url, _ in session.calls if url.startswith("https://media.example/")]
    assert media_urls == [
        "https://media.example/avc480.m4s",
        "https://media.example/audio_high.m4s",
    ]
    api_calls = {url: kwargs for url, kwargs in session.calls if "api.bilibili.com" in url}
    playurl_params = api_calls[
        "https://api.bilibili.com/x/player/playurl"
    ]["params"]
    assert playurl_params["qn"] == 32  # max_height=480 -> 480P
    assert playurl_params["fnval"] == 4048
    headers = api_calls["https://api.bilibili.com/x/player/playurl"]["headers"]
    assert headers["Referer"] == f"https://www.bilibili.com/video/{VALID_BVID}/"
    assert headers["User-Agent"]

    assert len(ffmpeg_commands) == 1
    command = ffmpeg_commands[0]
    assert "-c" in command and "copy" in command
    assert command.count("-i") == 2
    # ffmpeg 先写 .mp4.part 再原子替换为最终文件
    assert command[-1] == str(out_dir / f"{VALID_BVID}_p1.mp4.part")


def test_playurl_progressive_download_offline(
    config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "code": 0,
        "message": "OK",
        "data": {
            "quality": 16,
            "format": "mp4",
            "durl": [
                {"url": "https://media.example/progressive.mp4", "size": 123},
            ],
        },
    }
    session = _RoutedSession(
        [
            ("https://api.bilibili.com/x/web-interface/view", _JsonResponse(_view_payload())),
            ("https://api.bilibili.com/x/player/playurl", _JsonResponse(payload)),
            ("https://media.example/progressive.mp4", _BytesStreamResponse(b"PROGRESSIVE")),
        ]
    )

    def ffmpeg_must_not_run(*args, **kwargs):
        raise AssertionError("durl 渐进流不需要 ffmpeg 合并")

    monkeypatch.setattr(download_module.subprocess, "run", ffmpeg_must_not_run)
    out_dir = tmp_path / "dl"
    path = download_bilibili_playurl(
        VALID_BVID, out_dir=out_dir, download=config.download, session=session
    )

    assert path == (out_dir / f"{VALID_BVID}_p1.mp4").resolve()
    assert path.read_bytes() == b"PROGRESSIVE"
    assert sorted(item.name for item in out_dir.iterdir()) == [f"{VALID_BVID}_p1.mp4"]


def test_playurl_api_error_raises(config, tmp_path: Path) -> None:
    payload = {"code": -404, "message": "啥都木有"}
    session = _RoutedSession(
        [
            ("https://api.bilibili.com/x/web-interface/view", _JsonResponse(_view_payload())),
            ("https://api.bilibili.com/x/player/playurl", _JsonResponse(payload)),
        ]
    )
    out_dir = tmp_path / "dl"
    with pytest.raises(DownloadError) as info:
        download_bilibili_playurl(
            VALID_BVID, out_dir=out_dir, download=config.download, session=session
        )
    message = str(info.value)
    assert "playurl API 不可用" in message
    assert "-404" in message
    assert "需要登录" in message
    assert list(out_dir.iterdir()) == []


def test_playurl_no_stream_raises(config, tmp_path: Path) -> None:
    payload = {"code": 0, "message": "OK", "data": {}}
    session = _RoutedSession(
        [
            ("https://api.bilibili.com/x/web-interface/view", _JsonResponse(_view_payload())),
            ("https://api.bilibili.com/x/player/playurl", _JsonResponse(payload)),
        ]
    )
    with pytest.raises(DownloadError, match="DASH 或 durl"):
        download_bilibili_playurl(
            VALID_BVID, out_dir=tmp_path / "dl", download=config.download, session=session
        )


def test_playurl_stream_failure_cleans_temp_files(config, tmp_path: Path) -> None:
    session = _RoutedSession(
        [
            ("https://api.bilibili.com/x/web-interface/view", _JsonResponse(_view_payload())),
            ("https://api.bilibili.com/x/player/playurl", _JsonResponse(_dash_payload())),
            (
                "https://media.example/avc480.m4s",
                lambda: (_ for _ in ()).throw(requests.ConnectionError("stream boom")),
            ),
        ]
    )
    download = dataclasses.replace(config.download, retries=0)
    out_dir = tmp_path / "dl"
    with pytest.raises(DownloadError, match="视频流"):
        download_bilibili_playurl(
            VALID_BVID, out_dir=out_dir, download=download, session=session
        )
    assert list(out_dir.iterdir()) == []


def test_download_bilibili_falls_back_and_cleans_partial(
    config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "dl"
    out_dir.mkdir()
    partial = out_dir / f"{VALID_BVID}_p1.mp4.part"
    partial.write_bytes(b"half-downloaded")
    fallback_path = out_dir / f"{VALID_BVID}_p1.mp4"

    def fake_fallback(bvid, *, part=1, out_dir, download, **kwargs):
        fallback_path.write_bytes(b"from-playurl")
        return fallback_path

    monkeypatch.setattr(
        "nebula.download.download_bilibili_playurl", fake_fallback
    )

    def broken_factory(options: dict[str, Any]) -> Any:
        raise RuntimeError("yt-dlp broken")

    path = download_bilibili(
        VALID_BVID,
        out_dir=out_dir,
        download=config.download,
        ydl_factory=broken_factory,
    )
    assert path == fallback_path
    assert not partial.exists()
    assert fallback_path.read_bytes() == b"from-playurl"


def test_download_bilibili_fallback_failure_reports_both(
    config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_fallback(bvid, **kwargs):
        raise DownloadError("playurl boom")

    monkeypatch.setattr(
        "nebula.download.download_bilibili_playurl", broken_fallback
    )

    def broken_factory(options: dict[str, Any]) -> Any:
        raise RuntimeError("yt boom")

    with pytest.raises(DownloadError) as info:
        download_bilibili(
            VALID_BVID,
            out_dir=tmp_path / "dl",
            download=config.download,
            ydl_factory=broken_factory,
        )
    message = str(info.value)
    assert "yt-dlp 与 playurl 回退均失败" in message
    assert "yt boom" in message
    assert "playurl boom" in message


# ---------------------------------------------------------------- 真实网络（默认跳过）


@pytest.mark.network
@pytest.mark.skipif(
    not NETWORK_ENABLED,
    reason="网络测试默认跳过；设置 NEBULA_NETWORK_TESTS=1 后运行",
)
def test_download_bilibili_playurl_real_network(config, tmp_path: Path) -> None:
    """真实网络验证 playurl 回退：BV1LufpYGEb5（208s）能下到含音视频的 mp4。"""

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        pytest.skip("缺少 ffprobe")
    out_dir = tmp_path / "dl"
    path = download_bilibili_playurl(
        REAL_FALLBACK_BVID,
        out_dir=out_dir,
        download=config.download,
        timeout_s=300.0,
    )
    assert path.is_file()
    assert path.stat().st_size > 1_000_000
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    payload = json.loads(probe.stdout or "{}")
    kinds = {stream.get("codec_type") for stream in payload.get("streams", [])}
    assert {"video", "audio"} <= kinds
    duration = float((payload.get("format") or {}).get("duration") or 0.0)
    assert duration > 150.0
    assert path.parent == out_dir.resolve()


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


@pytest.mark.network
@pytest.mark.skipif(
    not NETWORK_ENABLED,
    reason="网络测试默认跳过；设置 NEBULA_NETWORK_TESTS=1 后运行",
)
def test_fetch_meta_real_multipart_uses_part_duration(config) -> None:
    """B1 真实回归：多分P视频的每个分P时长必须远小于整部总时长。

    【NICE爷爷】My story 整部 356s、两个分P各约 178s；修复前 part=1 会错误返回 356s。
    """

    with BilibiliClient(config.bilibili) as client:
        meta = client.fetch_meta(REAL_MULTIPART_BVID, part=1)
    assert meta.bvid == REAL_MULTIPART_BVID
    assert meta.page_count >= 2
    assert meta.duration_s < 250  # 分P时长，而不是整部总时长（约 356）
    assert meta.part_title is not None
