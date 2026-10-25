"""Bilibili 元数据获取、页面抓取与 yt-dlp 视频下载。

本模块是"数据导入"与"流水线"的下载边界（接口见 docs/CONTRACTS.md 第 7 节）：

- BilibiliClient 调用 Bilibili 公开的 view 接口解析作品元数据；
- http_get_text 抓取周报等页面文本（UA / 超时 / 代理可配置）；
- download_bilibili 通过 yt-dlp Python API 下载限高视频并返回真实文件路径；
  yt-dlp 解析失败时回退到 Bilibili player/playurl API（DASH 音视频分流 + ffmpeg -c copy，
  无 DASH 时回退 durl 渐进 mp4），回退只使用匿名可获取的清晰度，不绕过登录/收费限制。

默认测试必须完全离线，因此网络会话与 yt-dlp 对象都支持注入；真实网络测试使用
pytest.mark.network 标记，默认跳过。

若真实下载遇到 Bilibili 风控（例如 yt-dlp 报 HTTP 412/403、要求登录态），
不要绕过风控：在 config.yaml 里把 download.cookies_file 指向浏览器导出的
Netscape 格式 cookies 文件，本模块会把它透传给 yt-dlp。
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import requests

from .config import BilibiliConfig, DownloadConfig
from .errors import DownloadError, NebulaError

LOGGER = logging.getLogger(__name__)

# Bilibili 视频号：BV + 10 位数字/字母（契约规定）。这里是 BV 号校验的唯一来源，
# nebula.ingest 复用同一个正则，避免两处规则漂移。
BVID_PATTERN = r"BV[0-9A-Za-z]{10}"
BVID_RE = re.compile(rf"^{BVID_PATTERN}$")

BILIBILI_VIEW_API = "https://api.bilibili.com/x/web-interface/view"
BILIBILI_PLAYURL_API = "https://api.bilibili.com/x/player/playurl"
BILIBILI_VIDEO_URL = "https://www.bilibili.com/video/{bvid}"

# playurl 回退使用的默认 UA / 头。yt-dlp 首选路径不需要它；
# 回退路径没有 BilibiliConfig 可传（pipeline 只把 DownloadConfig 交给 download_bilibili），
# 因此提供与项目 config.yaml 一致的默认值，并允许调用方覆盖。
DEFAULT_BILIBILI_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
# fnval=4048 打开 DASH + 4K 能力位；匿名账号只会返回其可用清晰度（通常 <=480P）。
_PLAYURL_FNVAL = 4048
_STREAM_CHUNK_BYTES = 1024 * 1024

# Bilibili 发布时间按北京时间（UTC+8）取日期，与网页上显示的日期一致。
_BILIBILI_TZ = timezone(timedelta(hours=8))

# 下载完成后不会出现在 out_dir 里的临时后缀（yt-dlp 未完成的中间文件）。
_INCOMPLETE_SUFFIXES = frozenset({".part", ".ytdl", ".temp", ".tmp", ".download"})

ProgressCallback = Callable[[dict[str, Any]], None]
YdlFactory = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class BilibiliMeta:
    """一部 Bilibili 视频（或其中一个分P）的元数据。

    duration_s 在指定分P时是"该分P的时长"：多分P视频不会用整部总时长，
    否则本地视频与 Bilibili 分P的时长校验会被误判（回归 bug 见 B1）。
    part_title 是该分P的标题（pages[part-1]["part"]；单P或缺少该字段时为 None），
    整部标题保留在 title 中。
    """

    bvid: str
    title: str
    author_uid: str | None
    author_name: str | None
    duration_s: float
    publish_date: str | None  # YYYY-MM-DD
    cover_url: str | None
    page_count: int
    part_title: str | None = None


def is_valid_bvid(value: object) -> bool:
    """判断给定值是否是合法 BV 号（BV + 10 位数字/字母）。"""

    return isinstance(value, str) and BVID_RE.fullmatch(value.strip()) is not None


# ---------------------------------------------------------------- HTTP 基础


def _make_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    return session


def _proxies(proxy: str | None) -> dict[str, str] | None:
    if not proxy:
        return None
    return {"http": proxy, "https": proxy}


def _status_code(response: Any) -> int | None:
    status = getattr(response, "status_code", None)
    if isinstance(status, bool) or not isinstance(status, int):
        return None
    return status


def http_get_text(
    url: str,
    *,
    user_agent: str,
    timeout_s: float,
    proxy: str | None = None,
    session: requests.Session | None = None,
) -> str:
    """抓取一个页面的文本内容（周报抓取用）。

    参数
    ----
    url:
        页面地址。
    user_agent:
        请求 UA，Bilibili 对默认 UA 可能返回风控页。
    timeout_s:
        超时秒数，必须大于 0。
    proxy:
        可选 HTTP(S) 代理，例如 http://127.0.0.1:7890。
    session:
        可注入的 requests.Session（默认测试用假对象，不触网）。

    异常
    ----
    DownloadError:
        网络失败、HTTP 状态码 >= 400 或响应没有文本内容。
    """

    url = str(url or "").strip()
    if not url:
        raise DownloadError("抓取地址不能为空")
    if not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise DownloadError(f"抓取超时时间必须大于 0: {timeout_s!r}")
    sess = session if session is not None else _make_session(user_agent)
    try:
        response = sess.get(
            url,
            headers={"User-Agent": user_agent},
            timeout=float(timeout_s),
            proxies=_proxies(proxy),
        )
    except requests.RequestException as exc:
        raise DownloadError(f"抓取页面失败（网络错误）: {url}: {exc}") from exc
    status = _status_code(response)
    if status is not None and status >= 400:
        raise DownloadError(f"抓取页面失败: {url}: HTTP {status}")
    text = getattr(response, "text", None)
    if text is None:
        raise DownloadError(f"抓取页面失败: {url}: 响应没有文本内容")
    return str(text)


# ---------------------------------------------------------------- 元数据接口


class BilibiliClient:
    """Bilibili 元数据客户端。

    session 可注入（例如测试中的假 session），此时由调用方负责关闭；
    未注入时本对象内部创建并持有 requests.Session，可用 close() 关闭。
    """

    def __init__(self, config: BilibiliConfig, *, session: requests.Session | None = None) -> None:
        self.config = config
        self._session = session
        self._owns_session = session is None

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = _make_session(self.config.user_agent)
        return self._session

    def close(self) -> None:
        if self._owns_session and self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self) -> BilibiliClient:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def fetch_meta(self, bvid: str, *, part: int = 1) -> BilibiliMeta:
        """获取一个 Bilibili 视频（或其中一个分P）的元数据。

        调用 https://api.bilibili.com/x/web-interface/view?bvid=...，
        解析标题、时长、UP 主、发布日期、封面与分P数量。指定分P时 duration_s
        是该分P的时长（多分P视频不能用整部总时长），标题仍保留整部标题。

        异常
        ----
        DownloadError:
            BV 号/分P非法、网络失败、HTTP 错误、JSON 无法解析、
            code != 0 或返回结构缺字段。
        """

        clean = str(bvid or "").strip()
        if not is_valid_bvid(clean):
            raise DownloadError(f"BV 号格式非法: {bvid!r}（应形如 BV1xx411c7mD）")
        if part < 1:
            raise DownloadError(f"分P必须 >= 1: {part}")

        try:
            response = self.session.get(
                BILIBILI_VIEW_API,
                params={"bvid": clean},
                headers={"User-Agent": self.config.user_agent},
                timeout=self.config.api_timeout_seconds,
            )
        except requests.RequestException as exc:
            raise DownloadError(
                f"获取 Bilibili 元数据失败（网络错误）: bvid={clean}: {exc}"
            ) from exc

        status = _status_code(response)
        if status is not None and status >= 400:
            raise DownloadError(f"获取 Bilibili 元数据失败: bvid={clean}: HTTP {status}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise DownloadError(f"Bilibili API 返回的不是合法 JSON: bvid={clean}") from exc
        if not isinstance(payload, Mapping):
            raise DownloadError(f"Bilibili API 返回结构异常（顶层不是对象）: bvid={clean}")
        code = payload.get("code")
        if code != 0:
            message = payload.get("message")
            raise DownloadError(
                f"Bilibili API 返回错误: bvid={clean}, code={code!r}, message={message!r}"
            )
        data = payload.get("data")
        if not isinstance(data, Mapping):
            raise DownloadError(f"Bilibili API 缺少 data 字段: bvid={clean}")
        return _meta_from_payload(clean, part, data)


def _meta_from_payload(bvid: str, part: int, data: Mapping[str, Any]) -> BilibiliMeta:
    """把 view 接口的 data 字段解析为 BilibiliMeta。"""

    title = str(data.get("title") or "").strip()
    if not title:
        raise DownloadError(f"Bilibili 元数据缺少标题: bvid={bvid}")

    pages = data.get("pages")
    page_count = len(pages) if isinstance(pages, list) and pages else 1
    if part > page_count:
        raise DownloadError(
            f"分P 超出范围: bvid={bvid}, part={part}, 该视频只有 {page_count} 个分P"
        )

    page: Mapping[str, Any] | None = None
    if isinstance(pages, list) and 1 <= part <= len(pages):
        candidate = pages[part - 1]
        if isinstance(candidate, Mapping):
            page = candidate

    duration_value: Any = data.get("duration")
    part_title: str | None = None
    if page is not None:
        # 多分P视频的 data["duration"] 是整部总时长，必须优先采用该分P的时长；
        # 只有分P时长缺失/非法时才回退到总时长（并保证最终时长 > 0）。
        page_duration = _optional_positive_float(page.get("duration"))
        if page_duration is not None:
            duration_value = page_duration
        raw_part_title = page.get("part")
        if raw_part_title is not None:
            part_title = str(raw_part_title).strip() or None
    duration_s = _positive_float(duration_value, field="duration", bvid=bvid)

    owner = data.get("owner")
    author_uid: str | None = None
    author_name: str | None = None
    if isinstance(owner, Mapping):
        mid = owner.get("mid")
        if mid is not None:
            author_uid = str(mid).strip() or None
        owner_name = owner.get("name")
        if owner_name is not None:
            author_name = str(owner_name).strip() or None

    publish_date = _publish_date(data.get("pubdate"), bvid=bvid)
    cover = data.get("pic")
    cover_url = (str(cover).strip() or None) if cover is not None else None

    return BilibiliMeta(
        bvid=bvid,
        title=title,
        author_uid=author_uid,
        author_name=author_name,
        duration_s=duration_s,
        publish_date=publish_date,
        cover_url=cover_url,
        page_count=page_count,
        part_title=part_title,
    )


def _optional_positive_float(value: Any) -> float | None:
    """解析可选正浮点数；None/非法/非正数时返回 None（分P时长回退用）。"""

    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _positive_float(value: Any, *, field: str, bvid: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DownloadError(
            f"Bilibili 元数据字段非法: bvid={bvid}, {field}={value!r}（需要正数）"
        ) from exc
    if not math.isfinite(number) or number <= 0:
        raise DownloadError(
            f"Bilibili 元数据字段非法: bvid={bvid}, {field}={value!r}（需要正数）"
        )
    return number


def _publish_date(value: Any, *, bvid: str) -> str | None:
    """把 pubdate Unix 时间戳转换为北京时间日期（YYYY-MM-DD）。"""

    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        try:
            timestamp = int(str(value).strip())
        except ValueError as exc:
            raise DownloadError(
                f"Bilibili 元数据 pubdate 非法: bvid={bvid}, value={value!r}"
            ) from exc
    else:
        timestamp = int(value)
    if timestamp <= 0:
        return None
    try:
        moment = datetime.fromtimestamp(timestamp, tz=_BILIBILI_TZ)
    except (OverflowError, OSError, ValueError) as exc:
        raise DownloadError(
            f"Bilibili 元数据 pubdate 非法: bvid={bvid}, value={value!r}"
        ) from exc
    return moment.strftime("%Y-%m-%d")


# ---------------------------------------------------------------- 视频下载


def _format_selector(download: DownloadConfig) -> str:
    """给出 yt-dlp 的格式选择器。

    优先使用 download.yt_dlp_format（约定其中自带 [height<=N] 高度上限）；
    若它为空或不含高度约束，则用 download.max_height 生成带高度上限的选择器，
    保证"限制高度"这一契约要求不会因为配置写法而失效。
    """

    fmt = (download.yt_dlp_format or "").strip()
    if fmt and re.search(r"height\s*<", fmt):
        return fmt
    cap = int(download.max_height)
    if cap > 0:
        height = f"[height<={cap}]"
        if fmt:
            LOGGER.warning(
                "配置的 download.yt_dlp_format 未包含高度约束，改用 max_height=%s 生成的选择器",
                cap,
            )
        return f"bv*{height}+ba/b{height}/b"
    return fmt or "bv*+ba/b"


def _default_ydl_factory(options: dict[str, Any]) -> Any:
    """默认的 yt-dlp 对象工厂（延迟导入 yt-dlp，避免默认测试加载重型模块）。"""

    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - 环境已安装 yt-dlp，防御性分支
        raise DownloadError("未安装 yt-dlp，无法下载视频；请先安装 yt-dlp") from exc
    try:
        return yt_dlp.YoutubeDL(options)
    except Exception as exc:  # yt-dlp 初始化失败的异常类型随版本变化
        raise DownloadError(f"初始化 yt-dlp 失败: {exc}") from exc


def _stderr_tail_text(stderr: str | bytes | None, limit: int = 800) -> str:
    if not stderr:
        return ""
    text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else str(stderr)
    text = text.strip()
    if len(text) > limit:
        return "..." + text[-limit:]
    return text


def _playurl_qn(max_height: int) -> int:
    """把 config.download.max_height 映射为 Bilibili qn（16/32/64/80）。"""

    if not max_height or int(max_height) <= 0:
        return 80
    cap = int(max_height)
    if cap <= 360:
        return 16
    if cap <= 480:
        return 32
    if cap <= 720:
        return 64
    return 80


def _bilibili_headers(bvid: str, user_agent: str | None = None) -> dict[str, str]:
    return {
        "User-Agent": user_agent or DEFAULT_BILIBILI_USER_AGENT,
        "Referer": f"https://www.bilibili.com/video/{bvid}/",
    }


def _get_json(
    session: requests.Session,
    url: str,
    *,
    params: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout_s: float,
    retries: int,
    proxy: str | None,
    what: str,
) -> dict[str, Any]:
    """请求并解析 Bilibili JSON API（带重试；网络/HTTP/JSON 失败转 DownloadError）。"""

    attempts = max(0, int(retries)) + 1
    last_error = "未知错误"
    for attempt in range(1, attempts + 1):
        response: Any = None
        try:
            response = session.get(
                url,
                params=dict(params),
                headers=dict(headers),
                timeout=timeout_s,
                proxies=_proxies(proxy),
            )
            status = getattr(response, "status_code", None)
            if isinstance(status, int) and not isinstance(status, bool) and status >= 400:
                last_error = f"HTTP {status}"
            else:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise DownloadError(f"{what} 返回的不是合法 JSON: {exc}") from exc
                if not isinstance(payload, Mapping):
                    raise DownloadError(f"{what} 返回结构异常（顶层不是对象）")
                return dict(payload)
        except requests.RequestException as exc:
            last_error = f"网络错误: {exc}"
        finally:
            if response is not None:
                with contextlib.suppress(Exception):  # 假对象可能没有 close
                    response.close()
        if attempt < attempts:
            time.sleep(min(2.0, 0.5 * attempt))
    raise DownloadError(f"{what} 请求失败: {last_error}")


def _stream_urls(stream: Mapping[str, Any]) -> list[str]:
    """从 DASH 流对象里收集主地址 + 备用地址（去重保序）。"""

    urls: list[str] = []
    for key in ("baseUrl", "base_url"):
        value = stream.get(key)
        if isinstance(value, str) and value:
            urls.append(value)
    for key in ("backupUrl", "backup_url"):
        value = stream.get(key)
        if isinstance(value, list):
            urls.extend(str(item) for item in value if item)
    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def _download_stream(
    session: requests.Session,
    urls: Sequence[str],
    destination: Path,
    *,
    headers: Mapping[str, str],
    proxy: str | None,
    timeout_s: float,
    retries: int,
    what: str,
) -> int:
    """把媒体流（m4s/mp4）流式写入 destination；失败抛 DownloadError 并删掉半成品。"""

    if not urls:
        raise DownloadError(f"{what}没有可用下载地址")
    attempts = max(1, int(retries) + 1)
    last_error = "未知错误"
    for url in urls:
        for attempt in range(1, attempts + 1):
            response: Any = None
            written = 0
            try:
                response = session.get(
                    url,
                    headers=dict(headers),
                    timeout=timeout_s,
                    stream=True,
                    proxies=_proxies(proxy),
                )
                status = getattr(response, "status_code", None)
                if isinstance(status, int) and not isinstance(status, bool) and status >= 400:
                    raise DownloadError(f"HTTP {status}")
                with destination.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=_STREAM_CHUNK_BYTES):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        written += len(chunk)
                if written <= 0:
                    raise DownloadError("下载内容为空")
                return written
            except (requests.RequestException, DownloadError, OSError) as exc:
                last_error = str(exc)
                destination.unlink(missing_ok=True)
            finally:
                if response is not None:
                    with contextlib.suppress(Exception):  # 假对象可能没有 close
                        response.close()
            if attempt < attempts:
                time.sleep(min(2.0, 0.5 * attempt))
    raise DownloadError(f"{what}下载失败: {last_error}")


def _mux_streams(
    ffmpeg: str,
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    timeout_s: float,
) -> None:
    """用 ffmpeg -c copy 把 DASH 视频/音频合并成 mp4（不重编码）。"""

    command = [
        str(ffmpeg),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(output_path),
    ]
    try:
        proc = subprocess.run(command, capture_output=True, timeout=timeout_s)
    except FileNotFoundError as exc:
        raise DownloadError(
            f"找不到 ffmpeg 可执行文件（{ffmpeg}），无法合并 playurl 音视频"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise DownloadError(f"ffmpeg 合并 playurl 音视频超时（>{timeout_s:g} 秒）") from exc
    except OSError as exc:
        raise DownloadError(f"无法启动 ffmpeg 合并 playurl 音视频（{ffmpeg}）: {exc}") from exc
    if proc.returncode != 0:
        tail = _stderr_tail_text(proc.stderr)
        detail = f"；stderr 摘要: {tail}" if tail else ""
        raise DownloadError(
            f"ffmpeg 合并 playurl 音视频失败（退出码 {proc.returncode}）{detail}"
        )


def _select_dash_streams(
    body: Mapping[str, Any], max_height: int
) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    """从 playurl data 里挑选视频/音频 DASH 流。

    规则：视频优先 h264(avc1)、优先 <= max_height 的最高清晰度，再按码率；
    音频取最高码率。没有 DASH 时返回 None（调用方回退 durl）。
    """

    dash = body.get("dash")
    if not isinstance(dash, Mapping):
        return None
    videos = [
        stream
        for stream in dash.get("video") or []
        if isinstance(stream, Mapping) and _stream_urls(stream)
    ]
    audios = [
        stream
        for stream in dash.get("audio") or []
        if isinstance(stream, Mapping) and _stream_urls(stream)
    ]
    if not videos or not audios:
        return None

    cap = int(max_height) if max_height and int(max_height) > 0 else 0
    within: list[Mapping[str, Any]] = []
    for stream in videos:
        height = stream.get("height")
        if cap <= 0 or not isinstance(height, (int, float)) or float(height) <= cap:
            within.append(stream)
    candidates = within or videos
    avc_streams = [
        stream
        for stream in candidates
        if str(stream.get("codec") or "").lower().startswith("avc1")
    ]
    pool = avc_streams or candidates

    def video_key(stream: Mapping[str, Any]) -> tuple[float, float]:
        height = stream.get("height")
        bandwidth = stream.get("bandwidth")
        return (
            float(height) if isinstance(height, (int, float)) else 0.0,
            float(bandwidth) if isinstance(bandwidth, (int, float)) else 0.0,
        )

    video = max(pool, key=video_key)
    audio = max(
        audios,
        key=lambda stream: (
            float(stream["bandwidth"])
            if isinstance(stream.get("bandwidth"), (int, float))
            else 0.0
        ),
    )
    return video, audio


def _cleanup_incomplete_fragments(out_dir: Path, bvid: str, part: int) -> int:
    """清理该作品残留的未完成下载文件（.part/.ytdl 等），返回删除数量。"""

    removed = 0
    for path in out_dir.glob(f"{bvid}_p{part}.*"):
        if not path.is_file() or path.suffix.lower() not in _INCOMPLETE_SUFFIXES:
            continue
        try:
            path.unlink()
            removed += 1
        except OSError as exc:  # pragma: no cover - 磁盘异常
            LOGGER.warning("清理未完成下载文件失败: %s: %s", path, exc)
    return removed


def _download_dash_streams(
    session: requests.Session,
    bvid: str,
    part: int,
    streams: tuple[Mapping[str, Any], Mapping[str, Any]],
    target_dir: Path,
    *,
    headers: Mapping[str, str],
    download: DownloadConfig,
    ffmpeg: str,
    timeout_s: float,
) -> Path:
    video_stream, audio_stream = streams
    video_urls = _stream_urls(video_stream)
    audio_urls = _stream_urls(audio_stream)
    video_tmp = target_dir / f"{bvid}_p{part}.video.m4s.part"
    audio_tmp = target_dir / f"{bvid}_p{part}.audio.m4s.part"
    mux_tmp = target_dir / f"{bvid}_p{part}.mp4.part"
    final_path = target_dir / f"{bvid}_p{part}.mp4"
    LOGGER.info(
        "playurl 回退选择 DASH: bvid=%s 视频=%s 音频=%s",
        bvid,
        video_stream.get("id"),
        audio_stream.get("id"),
    )
    try:
        _download_stream(
            session,
            video_urls,
            video_tmp,
            headers=headers,
            proxy=download.proxy,
            timeout_s=timeout_s,
            retries=int(download.retries),
            what=f"playurl 视频流（{bvid}）",
        )
        _download_stream(
            session,
            audio_urls,
            audio_tmp,
            headers=headers,
            proxy=download.proxy,
            timeout_s=timeout_s,
            retries=int(download.retries),
            what=f"playurl 音频流（{bvid}）",
        )
        _mux_streams(ffmpeg, video_tmp, audio_tmp, mux_tmp, timeout_s=timeout_s)
        if not mux_tmp.is_file() or mux_tmp.stat().st_size <= 0:
            raise DownloadError(f"playurl 回退：ffmpeg 合并后文件为空（bvid={bvid}）")
        _require_inside(mux_tmp, target_dir)
        os.replace(mux_tmp, final_path)
        _require_inside(final_path, target_dir)
        LOGGER.info("playurl 回退下载完成: %s", final_path)
        return final_path
    finally:
        for temporary in (video_tmp, audio_tmp, mux_tmp):
            temporary.unlink(missing_ok=True)


def _download_progressive_stream(
    session: requests.Session,
    bvid: str,
    part: int,
    body: Mapping[str, Any],
    target_dir: Path,
    *,
    headers: Mapping[str, str],
    download: DownloadConfig,
    timeout_s: float,
) -> Path:
    durl = [
        entry
        for entry in body.get("durl") or []
        if isinstance(entry, Mapping) and isinstance(entry.get("url"), str)
    ]
    if not durl:
        raise DownloadError(f"playurl 未返回可用的 DASH 或 durl 视频流: bvid={bvid}")
    if len(durl) > 1:
        raise DownloadError(
            f"playurl 返回分段 durl（{len(durl)} 段），暂不支持自动合并: bvid={bvid}"
        )
    fmt = str(body.get("format") or "mp4").strip().lower()
    extension = "mp4" if fmt not in ("flv", "mp4") else fmt
    final_path = target_dir / f"{bvid}_p{part}.{extension}"
    tmp_path = target_dir / f"{bvid}_p{part}.{extension}.part"
    urls = [str(durl[0]["url"])]
    backup = durl[0].get("backup_url") or durl[0].get("backupUrl")
    if isinstance(backup, list):
        urls.extend(str(item) for item in backup if item)
    LOGGER.info("playurl 回退使用 durl 渐进流: bvid=%s format=%s", bvid, fmt)
    try:
        _download_stream(
            session,
            urls,
            tmp_path,
            headers=headers,
            proxy=download.proxy,
            timeout_s=timeout_s,
            retries=int(download.retries),
            what=f"playurl durl（{bvid}）",
        )
        _require_inside(tmp_path, target_dir)
        os.replace(tmp_path, final_path)
        _require_inside(final_path, target_dir)
        LOGGER.info("playurl 回退下载完成（durl）: %s", final_path)
        return final_path
    finally:
        tmp_path.unlink(missing_ok=True)


def download_bilibili_playurl(
    bvid: str,
    *,
    part: int = 1,
    out_dir: Path,
    download: DownloadConfig,
    session: requests.Session | None = None,
    user_agent: str | None = None,
    ffmpeg: str = "ffmpeg",
    timeout_s: float = 600.0,
) -> Path:
    """Bilibili playurl API 回退下载（yt-dlp 解析失败时使用）。

    流程：view API 取 cid -> player/playurl API（fnval=4048 DASH）-> 匿名可用清晰度
    （qn 由 max_height 映射，通常 <=480P）-> requests 流式下载 video/audio m4s ->
    ffmpeg -c copy 合并 mp4；没有 DASH 时回退 durl 渐进 mp4。

    只使用匿名可获取的清晰度，不绕过登录/收费/版权限制；API 返回 -404/需登录时抛
    DownloadError 并附中文说明。失败清理所有临时文件，结果路径保证在 out_dir 内。
    """

    clean = str(bvid or "").strip()
    if not is_valid_bvid(clean):
        raise DownloadError(f"BV 号格式非法: {bvid!r}（应形如 BV1xx411c7mD）")
    if part < 1:
        raise DownloadError(f"分P必须 >= 1: {part}")
    target_dir = Path(out_dir).expanduser().resolve()
    if target_dir.exists() and not target_dir.is_dir():
        raise DownloadError(f"下载目录不是文件夹: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    headers = _bilibili_headers(clean, user_agent)
    owns_session = session is None
    active = session if session is not None else requests.Session()
    try:
        view = _get_json(
            active,
            BILIBILI_VIEW_API,
            params={"bvid": clean},
            headers=headers,
            timeout_s=timeout_s,
            retries=int(download.retries),
            proxy=download.proxy,
            what=f"Bilibili view API（{clean}）",
        )
        view_code = view.get("code")
        if view_code != 0:
            raise DownloadError(
                f"Bilibili view API 返回错误: bvid={clean}, code={view_code!r}, "
                f"message={view.get('message')!r}"
            )
        view_data = view.get("data")
        if not isinstance(view_data, Mapping):
            raise DownloadError(f"Bilibili view API 缺少 data: bvid={clean}")
        pages = view_data.get("pages")
        if not isinstance(pages, list) or not pages:
            raise DownloadError(f"Bilibili view API 没有分P信息: bvid={clean}")
        if part > len(pages):
            raise DownloadError(
                f"分P超出范围: bvid={clean}, part={part}, 该视频只有 {len(pages)} 个分P"
            )
        page = pages[part - 1]
        cid = page.get("cid") if isinstance(page, Mapping) else None
        if not isinstance(cid, int) or cid <= 0:
            raise DownloadError(f"Bilibili view API 缺少 cid: bvid={clean}, part={part}")
        qn = _playurl_qn(int(download.max_height))
        playurl = _get_json(
            active,
            BILIBILI_PLAYURL_API,
            params={
                "bvid": clean,
                "cid": cid,
                "qn": qn,
                "fnval": _PLAYURL_FNVAL,
                "fourk": 1,
                "fnver": 0,
            },
            headers=headers,
            timeout_s=timeout_s,
            retries=int(download.retries),
            proxy=download.proxy,
            what=f"Bilibili playurl API（{clean} cid={cid}）",
        )
        playurl_code = playurl.get("code")
        if playurl_code != 0:
            raise DownloadError(
                f"Bilibili playurl API 不可用: bvid={clean}, cid={cid}, "
                f"code={playurl_code!r}, message={playurl.get('message')!r}"
                "（可能视频已删除、需要登录或受版权限制，不尝试绕过）"
            )
        body = playurl.get("data")
        if not isinstance(body, Mapping):
            raise DownloadError(f"Bilibili playurl API 缺少 data: bvid={clean}")
        LOGGER.info(
            "playurl 回退: bvid=%s cid=%s 请求 qn=%s 实际清晰度=%s format=%s",
            clean,
            cid,
            qn,
            body.get("quality"),
            body.get("format"),
        )
        dash_streams = _select_dash_streams(body, int(download.max_height))
        if dash_streams is not None:
            return _download_dash_streams(
                active,
                clean,
                part,
                dash_streams,
                target_dir,
                headers=headers,
                download=download,
                ffmpeg=ffmpeg,
                timeout_s=timeout_s,
            )
        return _download_progressive_stream(
            active,
            clean,
            part,
            body,
            target_dir,
            headers=headers,
            download=download,
            timeout_s=timeout_s,
        )
    finally:
        if owns_session:
            active.close()


def download_bilibili(
    bvid: str,
    *,
    part: int = 1,
    out_dir: Path,
    download: DownloadConfig,
    on_progress: ProgressCallback | None = None,
    ydl_factory: YdlFactory | None = None,
    api_fallback: bool = True,
    session: requests.Session | None = None,
    user_agent: str | None = None,
    ffmpeg: str = "ffmpeg",
    timeout_s: float = 600.0,
) -> Path:
    """下载一个 Bilibili 视频，返回 out_dir 内的真实文件路径。

    首选 yt-dlp；yt-dlp 解析/下载失败时（例如 B 站页面改版导致
    "Unable to extract initial state"），若 api_fallback=True 自动回退到
    Bilibili player/playurl API：view API 取 cid，DASH 音视频分流后 requests
    流式下载并用 ffmpeg -c copy 合并为 mp4；无 DASH 时回退 durl 渐进 mp4。
    回退只使用匿名可获取的清晰度，不绕过登录/收费/版权限制。

    参数
    ----
    bvid:
        BV 号（BV + 10 位数字/字母）。
    part:
        分P 序号，从 1 开始。
    out_dir:
        下载输出目录，不存在会自动创建。
    download:
        下载配置（格式、高度上限、cookies、代理、重试次数）。
    on_progress:
        可选的 yt-dlp 进度回调，接收 yt-dlp 的 progress hook 字典。
    ydl_factory:
        可注入的 yt-dlp 对象工厂，签名 factory(options: dict) -> ydl；
        默认工厂调用 yt_dlp.YoutubeDL。默认测试通过注入假对象完全离线。
    api_fallback:
        是否在 yt-dlp 失败后尝试 playurl API 回退（默认 True）。
    session:
        可注入的 requests.Session（playurl 回退用；默认自建。测试注入假对象保持离线）。
    user_agent / ffmpeg / timeout_s:
        回退路径的 UA、ffmpeg 可执行文件与单步超时。
    """

    clean = str(bvid or "").strip()
    if not is_valid_bvid(clean):
        raise DownloadError(f"BV 号格式非法: {bvid!r}（应形如 BV1xx411c7mD）")
    if part < 1:
        raise DownloadError(f"分P必须 >= 1: {part}")

    target_dir = Path(out_dir).expanduser().resolve()
    if target_dir.exists() and not target_dir.is_dir():
        raise DownloadError(f"下载目录不是文件夹: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    if download.cookies_file is not None:
        cookies = Path(download.cookies_file).expanduser()
        if not cookies.is_file():
            raise DownloadError(
                f"cookies 文件不存在: {cookies}；请更新 download.cookies_file 或留空"
            )

    try:
        return _download_bilibili_ytdlp(
            clean,
            part=part,
            target_dir=target_dir,
            download=download,
            on_progress=on_progress,
            ydl_factory=ydl_factory,
        )
    except DownloadError as yt_error:
        if not api_fallback:
            raise
        removed = _cleanup_incomplete_fragments(target_dir, clean, part)
        LOGGER.warning(
            "yt-dlp 下载失败，回退 playurl API: bvid=%s part=%s（清理未完成文件 %d 个）: %s",
            clean,
            part,
            removed,
            yt_error,
        )
        try:
            return download_bilibili_playurl(
                clean,
                part=part,
                out_dir=target_dir,
                download=download,
                session=session,
                user_agent=user_agent,
                ffmpeg=ffmpeg,
                timeout_s=timeout_s,
            )
        except DownloadError as api_error:
            _cleanup_incomplete_fragments(target_dir, clean, part)
            raise DownloadError(
                "Bilibili 下载失败（yt-dlp 与 playurl 回退均失败）: "
                f"bvid={clean}, part={part}；yt-dlp: {yt_error}；playurl: {api_error}"
            ) from api_error


def _download_bilibili_ytdlp(
    bvid: str,
    *,
    part: int,
    target_dir: Path,
    download: DownloadConfig,
    on_progress: ProgressCallback | None,
    ydl_factory: YdlFactory | None,
) -> Path:
    """yt-dlp 首选下载路径（失败时由 download_bilibili 触发 playurl 回退）。"""

    url = BILIBILI_VIDEO_URL.format(bvid=bvid)
    if part > 1:
        url = f"{url}?p={part}"
    outtmpl = str(target_dir / f"{bvid}_p{part}.%(ext)s")
    options: dict[str, Any] = {
        "format": _format_selector(download),
        "outtmpl": outtmpl,
        "retries": int(download.retries),
        "fragment_retries": int(download.retries),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": on_progress is None,
        "merge_output_format": "mp4",
    }
    if download.cookies_file is not None:
        options["cookiefile"] = str(Path(download.cookies_file).expanduser().resolve())
    if download.proxy:
        options["proxy"] = download.proxy
    if on_progress is not None:
        options["progress_hooks"] = [on_progress]

    factory = ydl_factory if ydl_factory is not None else _default_ydl_factory
    try:
        ydl = factory(options)
    except NebulaError:
        raise
    except Exception as exc:  # 注入工厂/yt-dlp 的异常统一转换
        raise DownloadError(f"初始化 yt-dlp 失败: bvid={bvid}, part={part}: {exc}") from exc

    LOGGER.info("开始下载 Bilibili 视频（yt-dlp）: bvid=%s part=%s -> %s", bvid, part, target_dir)
    try:
        info = ydl.extract_info(url, download=True)
    except NebulaError:
        raise
    except Exception as exc:  # yt-dlp 的 DownloadError 类型随版本变化，统一转换
        raise DownloadError(f"yt-dlp 下载失败: bvid={bvid}, part={part}: {exc}") from exc
    if not isinstance(info, Mapping):
        raise DownloadError(f"yt-dlp 未返回视频信息: bvid={bvid}, part={part}")

    path = _resolve_downloaded_path(ydl, info, target_dir, bvid, part)
    LOGGER.info("下载完成（yt-dlp）: bvid=%s part=%s -> %s", bvid, part, path)
    return path


def _iter_download_entries(info: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """遍历顶层 info 与可能的播放列表条目，便于收集已下载文件路径。"""

    yield info
    entries = info.get("entries")
    if entries is None or isinstance(entries, (str, bytes, Mapping)):
        return
    try:
        for entry in entries:
            if isinstance(entry, Mapping):
                yield entry
    except TypeError:  # entries 不是可迭代对象（防御性分支）
        return


def _resolve_downloaded_path(
    ydl: Any,
    info: Mapping[str, Any],
    out_dir: Path,
    bvid: str,
    part: int,
) -> Path:
    """在 out_dir 内定位 yt-dlp 实际写出的视频文件。"""

    candidates: list[Path] = []
    for entry in _iter_download_entries(info):
        requested = entry.get("requested_downloads")
        if isinstance(requested, list):
            for item in requested:
                if isinstance(item, Mapping):
                    filepath = item.get("filepath") or item.get("_filename")
                    if filepath:
                        candidates.append(Path(str(filepath)))
    if info.get("_type") != "playlist":
        try:
            prepared = ydl.prepare_filename(info)
        except Exception:  # 假对象或异常 info 时退回候选/glob 查找
            prepared = None
        if prepared:
            candidates.append(Path(str(prepared)))
    ext = str(info.get("ext") or "").strip() or "mp4"
    candidates.append(out_dir / f"{bvid}_p{part}.{ext}")

    for candidate in candidates:
        path = candidate if candidate.is_absolute() else out_dir / candidate
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not resolved.is_file() or resolved.stat().st_size <= 0:
            continue
        _require_inside(resolved, out_dir)
        return resolved

    # 某些 yt-dlp 版本只返回 info、不给出 filepath：按固定模板在目录内兜底查找。
    matches = sorted(
        path.resolve()
        for path in out_dir.glob(f"{bvid}_p{part}.*")
        if path.is_file()
        and path.stat().st_size > 0
        and path.suffix.lower() not in _INCOMPLETE_SUFFIXES
    )
    for match in matches:
        _require_inside(match, out_dir)
        return match
    raise DownloadError(
        f"下载结束后在 {out_dir} 内找不到视频文件: bvid={bvid}, part={part}；"
        "请检查 yt-dlp 输出或磁盘空间"
    )


def _require_inside(path: Path, out_dir: Path) -> None:
    try:
        path.resolve().relative_to(out_dir.resolve())
    except ValueError as exc:
        raise DownloadError(f"下载结果不在目标目录内: {path}（目录: {out_dir}）") from exc
