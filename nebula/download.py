"""Bilibili 元数据获取、页面抓取与 yt-dlp 视频下载。

本模块是"数据导入"与"流水线"的下载边界（接口见 docs/CONTRACTS.md 第 7 节）：

- BilibiliClient 调用 Bilibili 公开的 view 接口解析作品元数据；
- http_get_text 抓取周报等页面文本（UA / 超时 / 代理可配置）；
- download_bilibili 通过 yt-dlp Python API 下载限高视频并返回真实文件路径。

默认测试必须完全离线，因此网络会话与 yt-dlp 对象都支持注入；真实网络测试使用
pytest.mark.network 标记，默认跳过。

若真实下载遇到 Bilibili 风控（例如 yt-dlp 报 HTTP 412/403、要求登录态），
不要绕过风控：在 config.yaml 里把 download.cookies_file 指向浏览器导出的
Netscape 格式 cookies 文件，本模块会把它透传给 yt-dlp。
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import requests

from .config import BilibiliConfig, DownloadConfig
from .errors import DownloadError, NebulaError

LOGGER = logging.getLogger(__name__)

# Bilibili 视频号：BV + 10 位数字/字母（契约规定）。这里是 BV 号校验的唯一来源，
# nebula.ingest 复用同一个正则，避免两处规则漂移。
BVID_PATTERN = r"BV[0-9A-Za-z]{10}"
BVID_RE = re.compile(rf"^{BVID_PATTERN}$")

BILIBILI_VIEW_API = "https://api.bilibili.com/x/web-interface/view"
BILIBILI_VIDEO_URL = "https://www.bilibili.com/video/{bvid}"

# Bilibili 发布时间按北京时间（UTC+8）取日期，与网页上显示的日期一致。
_BILIBILI_TZ = timezone(timedelta(hours=8))

# 下载完成后不会出现在 out_dir 里的临时后缀（yt-dlp 未完成的中间文件）。
_INCOMPLETE_SUFFIXES = frozenset({".part", ".ytdl", ".temp", ".tmp", ".download"})

ProgressCallback = Callable[[dict[str, Any]], None]
YdlFactory = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class BilibiliMeta:
    """一部 Bilibili 视频（或其中一个分P）的元数据。"""

    bvid: str
    title: str
    author_uid: str | None
    author_name: str | None
    duration_s: float
    publish_date: str | None  # YYYY-MM-DD
    cover_url: str | None
    page_count: int


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
        解析标题、时长、UP 主、发布日期、封面与分P数量。

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
    if (
        page is not None
        and (part > 1 or duration_value is None)
        and page.get("duration") is not None
    ):
        duration_value = page["duration"]
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
    )


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


def download_bilibili(
    bvid: str,
    *,
    part: int = 1,
    out_dir: Path,
    download: DownloadConfig,
    on_progress: ProgressCallback | None = None,
    ydl_factory: YdlFactory | None = None,
) -> Path:
    """用 yt-dlp 下载一个 Bilibili 视频，返回 out_dir 内的真实文件路径。

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

    异常
    ----
    DownloadError:
        参数非法、yt-dlp 初始化/下载失败、或下载结束后找不到有效视频文件。
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

    url = BILIBILI_VIDEO_URL.format(bvid=clean)
    if part > 1:
        url = f"{url}?p={part}"
    outtmpl = str(target_dir / f"{clean}_p{part}.%(ext)s")
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
        raise DownloadError(f"初始化 yt-dlp 失败: bvid={clean}, part={part}: {exc}") from exc

    LOGGER.info("开始下载 Bilibili 视频: bvid=%s part=%s -> %s", clean, part, target_dir)
    try:
        info = ydl.extract_info(url, download=True)
    except NebulaError:
        raise
    except Exception as exc:  # yt-dlp 的 DownloadError 类型随版本变化，统一转换
        raise DownloadError(f"yt-dlp 下载失败: bvid={clean}, part={part}: {exc}") from exc
    if not isinstance(info, Mapping):
        raise DownloadError(f"yt-dlp 未返回视频信息: bvid={clean}, part={part}")

    path = _resolve_downloaded_path(ydl, info, target_dir, clean, part)
    LOGGER.info("下载完成: bvid=%s part=%s -> %s", clean, part, path)
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
