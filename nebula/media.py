"""ffprobe 探测与 ffmpeg 音画同步切片。

本模块只负责“媒体文件 -> 片段文件”的确定性转换，不涉及数据库与流水线编排：

- probe_media：解析 ffprobe 的 JSON，容忍只有视频 / 只有音频的文件；
- extract_clip：用一条 ffmpeg 命令同时截取音画（-ss 放在 -i 前，
  音视频同源同偏移），源文件无音轨时用 anullsrc 补等长静音；
- mute_audio_clip / black_video_clip：为“静音 / 黑屏”消融实验生成等长素材；
- clip_duration_report：抽查抽出的视频与音频时长是否一致（P0 音画同步自检）。

所有 ffmpeg / ffprobe 失败的路径都抛 nebula.errors.MediaError，
并在异常信息中附带 stderr 摘要，便于定位坏文件或参数错误。
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import MediaConfig
from .errors import MediaError
from .sampling import ClipPlan

logger = logging.getLogger(__name__)

# 异常信息里保留的 stderr 尾部长度：足够看清报错原因，又不会淹没中文提示
_STDERR_TAIL_CHARS = 1200

_CHANNEL_LAYOUTS = {1: "mono", 2: "stereo"}


@dataclass(frozen=True)
class MediaInfo:
    """一次 ffprobe 探测的结果。

    只有视频轨时 has_audio 为 False、audio_codec 为 None；
    只有音频轨时 has_video 为 False，画面相关字段均为 None。
    """

    path: Path
    duration_s: float
    has_video: bool
    has_audio: bool
    width: int | None
    height: int | None
    fps: float | None
    video_codec: str | None
    audio_codec: str | None


@dataclass(frozen=True)
class ExtractedClip:
    """一个已截取到磁盘的音画同步片段。"""

    index: int
    start_s: float
    duration_s: float
    video_path: Path
    audio_path: Path


def _stderr_tail(stderr: str | bytes | None, limit: int = _STDERR_TAIL_CHARS) -> str:
    """截取 stderr 尾部作为异常摘要；没有输出时返回空字符串。"""

    if not stderr:
        return ""
    text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else stderr
    text = text.strip()
    if len(text) > limit:
        return "..." + text[-limit:]
    return text


def _run_tool(
    command: list[str], *, tool: str, timeout_s: float, what: str
) -> subprocess.CompletedProcess[str]:
    """执行 ffmpeg / ffprobe，把启动、超时等失败统一转换为 MediaError。"""

    logger.debug("%s命令: %s", what, " ".join(command))
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        raise MediaError(
            f"找不到 {tool} 可执行文件（{command[0]}）：请检查 PATH 或 config.yaml 中的 media 配置"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"{what}超时（>{timeout_s:g} 秒）：{command[0]}") from exc
    except OSError as exc:
        raise MediaError(f"{what}无法启动 {tool}（{command[0]}）：{exc}") from exc


def _check_result(
    proc: subprocess.CompletedProcess[str], *, what: str, target: str | Path
) -> None:
    """检查子进程退出码，非 0 时抛带 stderr 摘要的 MediaError。"""

    if proc.returncode == 0:
        return
    tail = _stderr_tail(proc.stderr)
    detail = f"；stderr 摘要: {tail}" if tail else ""
    raise MediaError(f"{what}失败（退出码 {proc.returncode}）：{target}{detail}")


def _check_timeout(timeout_s: float, *, what: str) -> float:
    """校验超时参数为正的有限数。"""

    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise MediaError(f"{what}超时参数非法: {timeout_s!r}")
    value = float(timeout_s)
    if not math.isfinite(value) or value <= 0:
        raise MediaError(f"{what}超时参数必须为正数，收到: {timeout_s!r}")
    return value


def _parse_positive_float(value: object) -> float | None:
    """把 ffprobe 的数值字段解析成正的有限 float；无法解析时返回 None。"""

    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _parse_positive_int(value: object) -> int | None:
    """把 ffprobe 的尺寸字段解析成正整数；无法解析时返回 None。"""

    if isinstance(value, bool):
        return None
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _parse_rate(value: object) -> float | None:
    """解析帧率字段（形如 30000/1001 或 25）；非法时返回 None。"""

    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            den = float(denominator)
            if den == 0:
                return None
            rate = float(numerator) / den
        else:
            rate = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(rate) or rate <= 0:
        return None
    return rate


def _is_attached_pic(stream: dict[str, object]) -> bool:
    """判断视频流是否是封面图（音频文件内嵌封面不应算作视频轨）。"""

    disposition = stream.get("disposition")
    if not isinstance(disposition, dict):
        return False
    try:
        return int(disposition.get("attached_pic", 0) or 0) == 1
    except (TypeError, ValueError):
        return False


def _probe_duration(payload: dict[str, object], streams: list[dict[str, object]]) -> float | None:
    """按 format -> 视频流 -> 音频流的顺序找第一个正时长。"""

    fmt = payload.get("format")
    if isinstance(fmt, dict):
        duration = _parse_positive_float(fmt.get("duration"))
        if duration is not None:
            return duration
    for stream in streams:
        duration = _parse_positive_float(stream.get("duration"))
        if duration is not None:
            return duration
    return None


def probe_media(
    path: str | Path, *, ffprobe: str = "ffprobe", timeout_s: float = 60.0
) -> MediaInfo:
    """用 ffprobe 探测媒体文件，返回时长、轨道与编码信息。

    参数
    ----
    path:
        媒体文件路径。
    ffprobe:
        ffprobe 可执行文件（默认从 PATH 查找，流水线传 cfg.media.ffprobe）。
    timeout_s:
        探测超时（秒）。

    异常
    ----
    MediaError:
        文件不存在、ffprobe 失败、输出无法解析或时长 <= 0 时抛出。
    """

    source = Path(path)
    if not source.is_file():
        raise MediaError(f"媒体文件不存在或不是普通文件: {source}")
    timeout = _check_timeout(timeout_s, what="ffprobe 探测")

    command = [
        str(ffprobe),
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(source),
    ]
    proc = _run_tool(command, tool="ffprobe", timeout_s=timeout, what=f"探测媒体 {source.name} ")
    _check_result(proc, what=f"ffprobe 探测媒体 {source.name} ", target=source)

    stdout = proc.stdout or ""
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise MediaError(f"无法解析 ffprobe 输出 JSON（{source}）：{exc}") from exc
    if not isinstance(payload, dict):
        raise MediaError(f"ffprobe 输出结构异常（{source}）：顶层不是 JSON 对象")

    raw_streams = payload.get("streams")
    if raw_streams is not None and not isinstance(raw_streams, list):
        raise MediaError(f"ffprobe 输出结构异常（{source}）：streams 不是数组")
    streams: list[dict[str, object]] = [
        stream for stream in (raw_streams or []) if isinstance(stream, dict)
    ]

    video_stream = next(
        (
            stream
            for stream in streams
            if stream.get("codec_type") == "video" and not _is_attached_pic(stream)
        ),
        None,
    )
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if video_stream is None and audio_stream is None:
        raise MediaError(f"未能从文件中解析出音视频流: {source}")

    duration_s = _probe_duration(payload, streams)
    if duration_s is None:
        raise MediaError(f"无法确定媒体时长（时长缺失或 <= 0）: {source}")

    video_codec = video_stream.get("codec_name") if video_stream is not None else None
    audio_codec = audio_stream.get("codec_name") if audio_stream is not None else None
    fps = None
    if video_stream is not None:
        fps = _parse_rate(video_stream.get("avg_frame_rate")) or _parse_rate(
            video_stream.get("r_frame_rate")
        )

    info = MediaInfo(
        path=source.resolve(),
        duration_s=duration_s,
        has_video=video_stream is not None,
        has_audio=audio_stream is not None,
        width=_parse_positive_int(video_stream.get("width")) if video_stream is not None else None,
        height=_parse_positive_int(video_stream.get("height")) if video_stream is not None else None,
        fps=fps,
        video_codec=video_codec if isinstance(video_codec, str) else None,
        audio_codec=audio_codec if isinstance(audio_codec, str) else None,
    )
    logger.debug(
        "探测完成 %s: duration=%.3fs video=%s audio=%s %sx%s @%s",
        source.name,
        info.duration_s,
        info.has_video,
        info.has_audio,
        info.width,
        info.height,
        info.fps,
    )
    return info


def _channel_layout(channels: int) -> str:
    """把声道数映射为 ffmpeg 的 channel layout 名称。"""

    try:
        return _CHANNEL_LAYOUTS[int(channels)]
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaError(f"不支持的音频声道数: {channels!r}（只支持 1 或 2）") from exc


def _validate_media_config(media: MediaConfig) -> None:
    """在构造 ffmpeg 命令前做最基本的配置校验，避免晦涩的编码器报错。"""

    if not str(media.ffmpeg).strip():
        raise MediaError("media.ffmpeg 不能为空")
    if not str(media.ffprobe).strip():
        raise MediaError("media.ffprobe 不能为空")
    if int(media.audio_sample_rate) <= 0:
        raise MediaError(f"media.audio_sample_rate 非法: {media.audio_sample_rate!r}")
    _channel_layout(media.audio_channels)


def _validate_duration(duration_s: float, *, what: str) -> float:
    """校验片段 / 素材时长为正的有限数。"""

    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)):
        raise MediaError(f"{what}非法: {duration_s!r}")
    value = float(duration_s)
    if not math.isfinite(value) or value <= 0:
        raise MediaError(f"{what}必须为正数，收到: {duration_s!r}")
    return value


def _cleanup_outputs(*paths: Path) -> None:
    """删除失败留下的半成品文件（尽力而为，不掩盖原始异常）。"""

    for path in paths:
        try:
            if path.is_file():
                path.unlink()
        except OSError:  # pragma: no cover - 清理失败不改变主错误
            logger.warning("无法删除未完成的输出文件: %s", path)


def _require_output(path: Path, *, what: str) -> None:
    """确认 ffmpeg 真的产出了非空文件。"""

    if not path.is_file() or path.stat().st_size <= 0:
        raise MediaError(f"ffmpeg 未生成有效的{what}: {path}")


def _extract_command(
    source: Path,
    *,
    start_s: float,
    duration_s: float,
    video_path: Path,
    audio_path: Path,
    media: MediaConfig,
    has_audio: bool,
) -> list[str]:
    """构造“一条命令同时输出 mp4 + wav”的 ffmpeg 参数。

    -ss 放在 -i 之前：同一条命令里音视频输入使用同一个精确 seek 偏移，
    保证抽出的音频与画面来自同一时间段（P0 音画同步的关键）。
    源文件无音轨时，用 lavfi 的 anullsrc 作为第二路输入补等长静音。
    """

    layout = _channel_layout(media.audio_channels)
    rate = int(media.audio_sample_rate)
    channels = int(media.audio_channels)
    command = [
        str(media.ffmpeg),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(source),
    ]
    if has_audio:
        command += [
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-t",
            f"{duration_s:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            str(media.video_preset),
            "-crf",
            str(int(media.video_crf)),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(video_path),
            "-map",
            "0:a:0",
            "-t",
            f"{duration_s:.3f}",
            "-vn",
            "-c:a",
            "pcm_s16le",
            "-ar",
            str(rate),
            "-ac",
            str(channels),
            str(audio_path),
        ]
    else:
        command += [
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={rate}:cl={layout}",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-t",
            f"{duration_s:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            str(media.video_preset),
            "-crf",
            str(int(media.video_crf)),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(video_path),
            "-map",
            "1:a:0",
            "-t",
            f"{duration_s:.3f}",
            "-vn",
            "-c:a",
            "pcm_s16le",
            "-ar",
            str(rate),
            "-ac",
            str(channels),
            str(audio_path),
        ]
    return command


def extract_clip(
    source: Path, plan: ClipPlan, dst_dir: Path, *, media: MediaConfig
) -> ExtractedClip:
    """按采样计划截取一个音画同步片段，输出 mp4 + wav 两个文件。

    输出文件名固定为 clip_{index:02d}.mp4 与 clip_{index:02d}.wav。
    源文件没有音轨时，用 anullsrc 补等长静音 wav（是否补静音由
    probe_media 的 has_audio 决定），因此音频文件总是存在且可直接送模型。

    异常
    ----
    MediaError:
        源文件不存在 / 没有视频轨、片段越界、ffmpeg 失败或未产出有效文件时抛出。
    """

    src = Path(source)
    if not isinstance(plan, ClipPlan):
        raise MediaError(f"片段计划类型非法: {type(plan).__name__}（需要 ClipPlan）")
    if not src.is_file():
        raise MediaError(f"源视频不存在或不是普通文件: {src}")
    index = plan.index
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise MediaError(f"片段序号非法: {index!r}")
    duration_s = _validate_duration(plan.duration_s, what="片段时长")
    if (
        isinstance(plan.start_s, bool)
        or not isinstance(plan.start_s, (int, float))
        or not math.isfinite(float(plan.start_s))
        or float(plan.start_s) < 0
    ):
        raise MediaError(f"片段起始时间非法: {plan.start_s!r}")
    start_s = float(plan.start_s)
    _validate_media_config(media)

    info = probe_media(src, ffprobe=media.ffprobe, timeout_s=media.probe_timeout_seconds)
    if not info.has_video:
        raise MediaError(f"源文件没有视频轨，无法截取音画片段: {src}")
    if start_s + duration_s > info.duration_s + 0.05:
        raise MediaError(
            f"片段超出视频时长: start={start_s:.3f}s + duration={duration_s:.3f}s "
            f"> {info.duration_s:.3f}s（{src}）"
        )

    dst = Path(dst_dir)
    try:
        dst.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MediaError(f"无法创建片段输出目录 {dst}: {exc}") from exc
    video_path = dst / f"clip_{index:02d}.mp4"
    audio_path = dst / f"clip_{index:02d}.wav"

    command = _extract_command(
        src,
        start_s=start_s,
        duration_s=duration_s,
        video_path=video_path,
        audio_path=audio_path,
        media=media,
        has_audio=info.has_audio,
    )
    try:
        proc = _run_tool(
            command,
            tool="ffmpeg",
            timeout_s=media.extract_timeout_seconds,
            what=f"截取片段 {index} ",
        )
        _check_result(proc, what=f"截取片段 {index} ", target=src)
        _require_output(video_path, what="视频片段")
        _require_output(audio_path, what="音频片段")
    except MediaError:
        _cleanup_outputs(video_path, audio_path)
        raise

    clip = ExtractedClip(
        index=index,
        start_s=start_s,
        duration_s=duration_s,
        video_path=video_path,
        audio_path=audio_path,
    )
    logger.info(
        "已截取片段 %s: %.3fs 起，时长 %.3fs（音轨=%s）",
        index,
        start_s,
        duration_s,
        "有" if info.has_audio else "无，已补静音",
    )
    return clip


def mute_audio_clip(duration_s: float, out_wav: Path, *, media: MediaConfig) -> Path:
    """生成与指定时长等长的静音 wav（消融实验：只移除声音）。"""

    duration = _validate_duration(duration_s, what="静音时长")
    _validate_media_config(media)
    target = Path(out_wav)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MediaError(f"无法创建静音 wav 输出目录 {target.parent}: {exc}") from exc
    layout = _channel_layout(media.audio_channels)
    command = [
        str(media.ffmpeg),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={int(media.audio_sample_rate)}:cl={layout}",
        "-t",
        f"{duration:.3f}",
        "-c:a",
        "pcm_s16le",
        "-ar",
        str(int(media.audio_sample_rate)),
        "-ac",
        str(int(media.audio_channels)),
        str(target),
    ]
    try:
        proc = _run_tool(
            command, tool="ffmpeg", timeout_s=media.extract_timeout_seconds, what="生成静音 wav "
        )
        _check_result(proc, what="生成静音 wav ", target=target)
        _require_output(target, what="静音 wav")
    except MediaError:
        _cleanup_outputs(target)
        raise
    logger.debug("已生成静音 wav: %s（%.3fs）", target, duration)
    return target


def black_video_clip(
    duration_s: float,
    out_mp4: Path,
    *,
    media: MediaConfig,
    width: int = 336,
    height: int = 336,
    fps: float = 25,
) -> Path:
    """生成与指定时长等长的纯黑视频（消融实验：只移除画面）。"""

    duration = _validate_duration(duration_s, what="黑屏视频时长")
    _validate_media_config(media)
    for name, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise MediaError(f"黑屏视频 {name} 非法: {value!r}")
        if value % 2 != 0:
            raise MediaError(f"黑屏视频 {name} 必须为偶数（yuv420p 编码要求）: {value}")
    fps_value = _validate_duration(fps, what="黑屏视频帧率")
    target = Path(out_mp4)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MediaError(f"无法创建黑屏视频输出目录 {target.parent}: {exc}") from exc
    command = [
        str(media.ffmpeg),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={width}x{height}:r={fps_value:g}:d={duration:.3f}",
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        str(media.video_preset),
        "-crf",
        str(int(media.video_crf)),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target),
    ]
    try:
        proc = _run_tool(
            command, tool="ffmpeg", timeout_s=media.extract_timeout_seconds, what="生成黑屏视频 "
        )
        _check_result(proc, what="生成黑屏视频 ", target=target)
        _require_output(target, what="黑屏视频")
    except MediaError:
        _cleanup_outputs(target)
        raise
    logger.debug("已生成黑屏视频: %s（%.3fs）", target, duration)
    return target


def clip_duration_report(
    video_path: Path,
    audio_path: Path,
    *,
    ffprobe: str = "ffprobe",
    tolerance_s: float = 0.15,
) -> dict[str, float | bool]:
    """对比片段视频与音频的时长，返回 P0 音画同步抽查报告。

    返回 {"video_duration_s", "audio_duration_s", "delta_s", "ok"}；
    ok 表示两者时长差不超过 tolerance_s 秒。
    """

    if isinstance(tolerance_s, bool) or not isinstance(tolerance_s, (int, float)):
        raise MediaError(f"时长容差非法: {tolerance_s!r}")
    tolerance = float(tolerance_s)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise MediaError(f"时长容差必须为非负有限数，收到: {tolerance_s!r}")

    video_info = probe_media(video_path, ffprobe=ffprobe)
    audio_info = probe_media(audio_path, ffprobe=ffprobe)
    video_duration_s = float(video_info.duration_s)
    audio_duration_s = float(audio_info.duration_s)
    delta_s = abs(video_duration_s - audio_duration_s)
    return {
        "video_duration_s": video_duration_s,
        "audio_duration_s": audio_duration_s,
        "delta_s": float(round(delta_s, 6)),
        "ok": bool(delta_s <= tolerance),
    }
