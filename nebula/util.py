"""与业务无关的小工具：时间戳、路径规范化、版本号标签。"""

from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path

_SLUG_RE = re.compile(r"[^0-9A-Za-z._-]+")


def now_utc_iso() -> str:
    """返回确定格式的 UTC 时间戳，例如 2026-10-25T07:00:00Z。"""

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_relpath(value: str | os.PathLike[str]) -> str:
    """把路径规范化为 POSIX 风格的相对路径字符串。

    用于本地作品的稳定 work_key。绝对路径、.. 逃逸都会被拒绝。
    """

    raw = str(value).replace("\\", "/")
    path = Path(raw)
    if path.is_absolute():
        raise ValueError(f"需要相对路径，收到绝对路径: {raw}")
    parts = [p for p in raw.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise ValueError(f"相对路径不能包含 '..': {raw}")
    if not parts:
        raise ValueError("相对路径不能为空")
    return "/".join(parts)


def slugify(value: str) -> str:
    """把任意字符串变成可用于目录名的安全标签。"""

    slug = _SLUG_RE.sub("-", value.strip()).strip("-._")
    return slug or "unnamed"


def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def file_sha1(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_iso_date(value: str) -> str:
    """把常见日期字符串规范化为 YYYY-MM-DD；无法解析时抛 ValueError。"""

    text = value.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"无法解析日期: {value!r}") from exc
