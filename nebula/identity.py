"""本地作品与 Bilibili 条目的身份关联（迭代 B1）。

场景：同一部音MAD作品可能既有本地视频（P0 已提取向量），又在选票 CSV 中
对应一条 Bilibili 作品。身份关联的目标是：

1. 用"规范化标题完全相等 + 时长差 <= 容差"双重校验确认两者是同一作品；
2. 建立 canonical（Bilibili 条目）与 duplicate（本地视频）的关联；
3. 把本地已提取的向量/片段计划复制登记到 canonical 作品，避免重复下载与编码；
4. 本地原视频、本地向量一律保留，只追加 canonical 侧副本。

拿不准就不合并：标题必须规范化后完全相等，且必须拿到真实 Bilibili 时长并
通过容差校验；任一步缺失都只生成"拒绝"候选，交给 --manual 清单人工确认。

接口见 docs/CONTRACTS_V2.md 第 5 节。
"""

from __future__ import annotations

import csv
import logging
import os
import re
import shutil
import tempfile
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from . import CLIP_VECTOR_KINDS, WORK_VECTOR_KINDS
from .config import Config
from .db import Catalog, work_key_for_local
from .download import BVID_RE, BilibiliClient, BilibiliMeta
from .errors import DatabaseError, IngestError, NebulaError
from .util import normalize_relpath

LOGGER = logging.getLogger(__name__)

MANUAL_LINKS_FILENAME = "local_bv_links.csv"
TITLE_DURATION_METHOD = "title+duration"
MANUAL_METHOD = "manual"
ALL_VECTOR_KINDS = tuple(WORK_VECTOR_KINDS) + tuple(CLIP_VECTOR_KINDS)

# 标题规范化：全角/半角标点统一到 ASCII 形式（NFKC 已处理大部分全角字符，
# 这里补 NFKC 不转换的 CJK 括号/引号/顿号等）。
_PUNCT_MAP = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "「": '"',
        "」": '"',
        "『": '"',
        "』": '"',
        "【": "[",
        "】": "]",
        "〔": "(",
        "〕": ")",
        "：": ":",
        "！": "!",
        "？": "?",
        "，": ",",
        "、": ",",
        "；": ";",
        "～": "~",
    }
)
# 零宽字符/变体选择符不参与比较。
_INVISIBLE_CHARS = ("\ufeff", "\u200b", "\u200c", "\u200d", "\u2060", "\ufe0f")
_WHITESPACE_RE = re.compile(r"\s+")

_MANUAL_COLUMNS: dict[str, tuple[str, ...]] = {
    "local_path": ("local_path", "本地路径"),
    "bvid": ("bvid", "bv号", "bv"),
    "part": ("part", "分p"),
}


@dataclass(frozen=True)
class IdentityCandidate:
    """一条"本地作品 <-> Bilibili 作品"候选关联。"""

    local_work_id: int
    bilibili_work_id: int
    local_title: str
    bilibili_title: str
    local_duration_s: float | None
    bilibili_duration_s: float | None
    method: str
    confidence: float
    accepted: bool
    reason: str


def normalize_title(value: str) -> str:
    """把作品标题规范化为可比较的键。

    规则：NFKC（全角转半角）、统一括号/引号/冒号/感叹号等标点、去掉所有空白
    （含换行）、去掉零宽字符、casefold。空串或只有空白/标点时抛 IngestError。
    """

    if not isinstance(value, str):
        raise IngestError(f"标题必须是字符串，收到 {type(value).__name__}: {value!r}")
    text = unicodedata.normalize("NFKC", value)
    for char in _INVISIBLE_CHARS:
        text = text.replace(char, "")
    text = text.translate(_PUNCT_MAP)
    text = _WHITESPACE_RE.sub("", text)
    text = text.casefold()
    if not text:
        raise IngestError(f"标题规范化后为空，无法用于身份关联: {value!r}")
    return text


# ---------------------------------------------------------------- 候选查找


def _local_display_title(work: dict[str, Any]) -> str:
    """本地作品的比较标题：优先作品 title，否则用文件名（不含扩展名）。"""

    title = work.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    local_path = str(work.get("local_path") or "").replace("\\", "/")
    return Path(local_path).stem if local_path else str(work.get("work_key") or "")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _list_bilibili_works(cat: Catalog, list_key: str) -> list[dict[str, Any]]:
    lists = cat.list_lists()
    if not any(str(row["list_key"]) == list_key for row in lists):
        raise IngestError(f"榜单不存在: {list_key}（请先运行 import-votes 导入选票 CSV）")
    work_ids = [
        int(row["work_id"])
        for row in cat.appearance_rows()
        if str(row["list_key"]) == list_key
    ]
    works: list[dict[str, Any]] = []
    for work_id in sorted(set(work_ids)):
        work = cat.get_work(work_id)
        if work is not None and str(work["source"]) == "bilibili":
            works.append(work)
    return works


def _evaluate_candidate(
    *,
    source_label: str,
    meta: BilibiliMeta | None,
    fetch_error: str | None,
    local_duration_s: float | None,
    tolerance_s: float,
) -> tuple[bool, float, float | None, str]:
    """根据标题匹配来源与时长差给出接受/拒绝结论。"""

    title_part = f"标题一致（匹配来源：{source_label}）"
    if meta is None:
        detail = (fetch_error or "未知原因")[:120]
        return False, 0.0, None, f"{title_part}；无法获取 Bilibili 元数据，未校验时长: {detail}"
    bili_duration = float(meta.duration_s)
    if local_duration_s is None:
        return False, 0.0, bili_duration, f"{title_part}；本地作品缺少时长，无法校验"
    diff = abs(local_duration_s - bili_duration)
    if diff <= tolerance_s:
        return (
            True,
            0.99,
            bili_duration,
            f"{title_part}；时长差 {diff:.2f}s <= 容差 {tolerance_s:.2f}s",
        )
    return (
        False,
        0.0,
        bili_duration,
        f"{title_part}；时长差 {diff:.2f}s > 容差 {tolerance_s:.2f}s",
    )


def _candidate_diff(candidate: IdentityCandidate) -> float:
    if candidate.local_duration_s is None or candidate.bilibili_duration_s is None:
        return float("inf")
    return abs(candidate.local_duration_s - candidate.bilibili_duration_s)


def _resolve_multi_matches(candidates: list[IdentityCandidate]) -> list[IdentityCandidate]:
    """同一 Bilibili 作品/同一本地作品出现多个接受候选时，只保留时长最接近的一条。"""

    accepted = [candidate for candidate in candidates if candidate.accepted]
    best_by_bilibili: dict[int, IdentityCandidate] = {}
    for candidate in accepted:
        current = best_by_bilibili.get(candidate.bilibili_work_id)
        if current is None or _candidate_diff(candidate) < _candidate_diff(current):
            best_by_bilibili[candidate.bilibili_work_id] = candidate
    best_by_local: dict[int, IdentityCandidate] = {}
    for candidate in accepted:
        current = best_by_local.get(candidate.local_work_id)
        if current is None or _candidate_diff(candidate) < _candidate_diff(current):
            best_by_local[candidate.local_work_id] = candidate

    resolved: list[IdentityCandidate] = []
    for candidate in candidates:
        if not candidate.accepted:
            resolved.append(candidate)
            continue
        if best_by_bilibili[candidate.bilibili_work_id] is not candidate:
            resolved.append(
                replace(
                    candidate,
                    accepted=False,
                    confidence=0.0,
                    reason="同一 Bilibili 作品存在多部本地候选，保留时长差最小的一部",
                )
            )
        elif best_by_local[candidate.local_work_id] is not candidate:
            resolved.append(
                replace(
                    candidate,
                    accepted=False,
                    confidence=0.0,
                    reason="同一本地作品存在多部 Bilibili 候选，保留时长差最小的一部",
                )
            )
        else:
            resolved.append(candidate)
    return resolved


def find_candidates(
    cat: Catalog,
    config: Config,
    *,
    list_key: str = "votes_2025",
    client: Any = None,
    duration_tolerance_s: float = 3.0,
) -> list[IdentityCandidate]:
    """在指定榜单的 Bilibili 作品与全部本地作品之间查找身份关联候选。

    流程
    ----
    1. 取出该榜单的 Bilibili 作品与全部 source=local 的作品；
    2. 用 BilibiliClient 拉取每个 Bilibili 作品的真实标题/时长，并补全作品元数据
       （title/author_uid/author_name/duration_s/publish_date）；
    3. 本地标题（作品 title 或文件名）与 Bilibili 标题（选票 CSV 标题或真实标题）
       规范化后完全相等才进入候选；
    4. 双方都有时长且 |差值| <= duration_tolerance_s 时 accepted=True。

    参数
    ----
    cat, config:
        编目与配置。
    list_key:
        榜单键，默认 votes_2025。
    client:
        可注入的 Bilibili 客户端（离线测试用假对象）；缺省创建 BilibiliClient。
    duration_tolerance_s:
        时长容差（秒），默认 3.0。

    异常
    ----
    IngestError:
        榜单不存在。
    """

    if duration_tolerance_s < 0:
        raise IngestError(f"时长容差必须 >= 0: {duration_tolerance_s}")
    bilibili_works = _list_bilibili_works(cat, list_key)
    local_works = sorted(cat.list_works(source="local"), key=lambda work: int(work["id"]))
    duplicate_ids = cat.get_duplicate_work_ids()
    local_works = [work for work in local_works if int(work["id"]) not in duplicate_ids]
    bilibili_works = [work for work in bilibili_works if int(work["id"]) not in duplicate_ids]
    if not bilibili_works or not local_works:
        LOGGER.info(
            "身份关联候选: 榜单作品 %d 部, 本地作品 %d 部, 无可比较对象",
            len(bilibili_works),
            len(local_works),
        )
        return []

    # 本地标题 -> [(work, 显示标题)]
    local_index: dict[str, list[tuple[dict[str, Any], str]]] = {}
    for work in local_works:
        display_title = _local_display_title(work)
        try:
            key = normalize_title(display_title)
        except IngestError:
            LOGGER.warning("本地作品标题无法规范化，跳过: work_id=%s", work["id"])
            continue
        local_index.setdefault(key, []).append((work, display_title))

    owns_client = client is None
    active_client = client
    if active_client is None:
        active_client = BilibiliClient(config.bilibili)
    metas: dict[int, BilibiliMeta] = {}
    fetch_errors: dict[int, str] = {}
    try:
        for work in bilibili_works:
            work_id = int(work["id"])
            bvid = str(work.get("bvid") or "").strip()
            part = int(work.get("part") or 1)
            if not bvid:
                fetch_errors[work_id] = "作品缺少 bvid"
                LOGGER.warning("Bilibili 作品缺少 bvid: work_id=%s", work_id)
                continue
            try:
                meta = active_client.fetch_meta(bvid, part=part)
            except NebulaError as exc:
                fetch_errors[work_id] = str(exc)
                LOGGER.warning(
                    "获取 Bilibili 元数据失败: work_id=%s bvid=%s part=%s: %s",
                    work_id,
                    bvid,
                    part,
                    exc,
                )
                continue
            metas[work_id] = meta
            cat.update_work_metadata(
                work_id,
                title=meta.title,
                author_uid=meta.author_uid,
                author_name=meta.author_name,
                duration_s=meta.duration_s,
                publish_date=meta.publish_date,
            )
    finally:
        if owns_client and active_client is not None:
            active_client.close()

    candidates: list[IdentityCandidate] = []
    for work in bilibili_works:
        work_id = int(work["id"])
        meta = metas.get(work_id)
        stored_title = str(work.get("title") or "").strip()
        title_sources: list[tuple[str, str]] = []
        if stored_title:
            title_sources.append(("选票标题", stored_title))
        if meta is not None:
            meta_title = str(meta.title or "").strip()
            if meta_title and normalize_safe(meta_title) != normalize_safe(stored_title):
                title_sources.append(("B站标题", meta_title))

        matched: dict[int, dict[str, Any]] = {}
        for label, title in title_sources:
            try:
                key = normalize_title(title)
            except IngestError:
                continue
            for local_work, display_title in local_index.get(key, []):
                local_work_id = int(local_work["id"])
                entry = matched.setdefault(
                    local_work_id,
                    {"work": local_work, "display_title": display_title, "labels": set()},
                )
                entry["labels"].add(label)

        for local_work_id in sorted(matched):
            entry = matched[local_work_id]
            labels = sorted(entry["labels"])
            source_label = "与".join(labels) if labels else "标题"
            local_work = entry["work"]
            local_duration = _as_float(local_work.get("duration_s"))
            accepted, confidence, bili_duration, reason = _evaluate_candidate(
                source_label=source_label,
                meta=meta,
                fetch_error=fetch_errors.get(work_id),
                local_duration_s=local_duration,
                tolerance_s=duration_tolerance_s,
            )
            if meta is not None:
                bili_title = str(meta.title or "").strip() or stored_title
                bili_duration = float(meta.duration_s)
            else:
                bili_title = stored_title
                bili_duration = _as_float(work.get("duration_s"))
            candidates.append(
                IdentityCandidate(
                    local_work_id=local_work_id,
                    bilibili_work_id=work_id,
                    local_title=str(entry["display_title"]),
                    bilibili_title=bili_title,
                    local_duration_s=local_duration,
                    bilibili_duration_s=bili_duration,
                    method=TITLE_DURATION_METHOD,
                    confidence=confidence,
                    accepted=accepted,
                    reason=reason,
                )
            )

    candidates.sort(key=lambda item: (item.bilibili_work_id, item.local_work_id))
    resolved = _resolve_multi_matches(candidates)
    accepted_count = sum(1 for candidate in resolved if candidate.accepted)
    LOGGER.info(
        "身份关联候选: 榜单 %s, 候选 %d 条（接受 %d, 拒绝 %d）",
        list_key,
        len(resolved),
        accepted_count,
        len(resolved) - accepted_count,
    )
    return resolved


def normalize_safe(value: Any) -> str:
    """normalize_title 的宽松版本：无法规范化时返回空串（用于去重比较）。"""

    try:
        return normalize_title(str(value or ""))
    except IngestError:
        return ""


# ---------------------------------------------------------------- 建立关联与复制向量


def _ensure_completed(cat: Catalog, work_id: int, *, reason: str) -> None:
    """把作品安全地置为 completed（必要时先 reopen），并记录 identity 事件。"""

    work = cat.get_work(work_id)
    if work is None:
        raise DatabaseError(f"作品不存在: {work_id}")
    if str(work["status"]) != "completed":
        try:
            cat.begin_attempt(work_id, "encoding")
        except DatabaseError:
            cat.reopen(work_id, reason=reason)
            cat.begin_attempt(work_id, "encoding")
        cat.complete(work_id)
    cat.record_event(work_id, "identity", reason)


def _atomic_copy_file(source: Path, destination: Path) -> None:
    """把文件原子复制到目标路径（临时文件 + os.replace）。"""

    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(handle, "wb") as out_handle, source.open("rb") as in_handle:
            shutil.copyfileobj(in_handle, out_handle)
        os.replace(tmp_name, destination)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _copy_vectors_for_candidate(
    cat: Catalog,
    config: Config,
    candidate: IdentityCandidate,
    version: str,
) -> tuple[bool, str]:
    """把 local 作品当前版本的向量/片段复制登记到 canonical 作品。

    返回 (是否复制成功, 说明)。不删除任何本地文件或本地向量。
    """

    local_id = int(candidate.local_work_id)
    canonical_id = int(candidate.bilibili_work_id)
    if cat.has_embedding(canonical_id, version=version, kind="work_joint"):
        _ensure_completed(cat, canonical_id, reason=f"canonical 已有向量，补记为已完成（版本 {version}）")
        return False, "canonical 已有当前版本向量，跳过复制"

    rows: list[dict[str, Any]] = []
    for kind in ALL_VECTOR_KINDS:
        row = cat.get_embedding(local_id, version=version, kind=kind)
        if row is not None:
            rows.append(row)
    if not rows:
        return False, f"本地作品没有当前版本向量（{version}），未复制；关联仍已建立"
    work_joint = next((row for row in rows if row["kind"] == "work_joint"), None)
    if work_joint is None:
        return False, "本地作品缺少联合向量 work_joint，未复制；关联仍已建立"
    work_joint_source = config.data_path(str(work_joint["path"]))
    if not work_joint_source.is_file():
        return False, f"本地联合向量文件不存在: {work_joint_source}，未复制；关联仍已建立"

    target_dir = config.paths.embeddings / version / f"{canonical_id:06d}"
    copied_files = 0
    for row in rows:
        source = config.data_path(str(row["path"]))
        kind = str(row["kind"])
        if not source.is_file():
            LOGGER.warning("本地向量文件不存在，跳过该模态: kind=%s path=%s", kind, source)
            continue
        destination = target_dir / f"{kind}.npy"
        _atomic_copy_file(source, destination)
        cat.register_embedding(
            canonical_id,
            version=version,
            kind=kind,
            path=config.data_relpath(destination),
            dim=int(row["dim"]),
            n_clips=row.get("n_clips"),
        )
        copied_files += 1

    clips = cat.get_clips(local_id, version=version)
    if clips:
        cat.replace_clips(
            canonical_id,
            version=version,
            clips=[
                (int(clip["idx"]), float(clip["start_s"]), float(clip["duration_s"]))
                for clip in clips
            ],
        )

    _ensure_completed(
        cat,
        canonical_id,
        reason=(
            f"从本地作品 {local_id} 复制向量（版本 {version}, "
            f"{copied_files} 个向量文件, {len(clips)} 个片段计划）"
        ),
    )
    return True, f"复制 {copied_files} 个向量文件与 {len(clips)} 个片段计划"


def link_candidates(
    cat: Catalog,
    config: Config,
    *,
    candidates: Sequence[IdentityCandidate],
    copy_vectors: bool = True,
    version: str | None = None,
) -> dict[str, Any]:
    """只对 accepted 候选建立身份关联；默认同时复制本地向量到 canonical。

    返回 {"linked": 关联数, "copied": 复制向量数, "skipped": [...], "version": 向量版本}。
    已建立关联的作品、非 accepted 候选、缺少本地向量的候选都会进入 skipped 并给出原因。
    """

    target_version = str(version or config.embedding_version)
    linked = 0
    copied = 0
    skipped: list[dict[str, Any]] = []
    for candidate in candidates:
        if not candidate.accepted:
            skipped.append(
                {
                    "local_work_id": int(candidate.local_work_id),
                    "bilibili_work_id": int(candidate.bilibili_work_id),
                    "reason": f"未接受候选：{candidate.reason}",
                }
            )
            continue
        try:
            cat.link_works(
                int(candidate.bilibili_work_id),
                int(candidate.local_work_id),
                method=str(candidate.method),
                confidence=float(candidate.confidence),
                note=candidate.reason,
            )
        except DatabaseError as exc:
            skipped.append(
                {
                    "local_work_id": int(candidate.local_work_id),
                    "bilibili_work_id": int(candidate.bilibili_work_id),
                    "reason": f"建立关联失败：{exc}",
                }
            )
            continue
        linked += 1
        LOGGER.info(
            "已建立身份关联: bilibili work_id=%s <- local work_id=%s (%s)",
            candidate.bilibili_work_id,
            candidate.local_work_id,
            candidate.method,
        )
        if not copy_vectors:
            continue
        try:
            copied_ok, detail = _copy_vectors_for_candidate(cat, config, candidate, target_version)
        except (DatabaseError, OSError) as exc:
            LOGGER.exception(
                "复制向量失败: local=%s canonical=%s",
                candidate.local_work_id,
                candidate.bilibili_work_id,
            )
            skipped.append(
                {
                    "local_work_id": int(candidate.local_work_id),
                    "bilibili_work_id": int(candidate.bilibili_work_id),
                    "reason": f"关联已建立，但复制向量失败：{exc}",
                }
            )
            continue
        if copied_ok:
            copied += 1
        else:
            skipped.append(
                {
                    "local_work_id": int(candidate.local_work_id),
                    "bilibili_work_id": int(candidate.bilibili_work_id),
                    "reason": f"关联已建立；{detail}",
                }
            )

    result = {
        "linked": linked,
        "copied": copied,
        "skipped": skipped,
        "version": target_version,
    }
    LOGGER.info(
        "身份关联完成: 关联 %d 条, 复制向量 %d 条, 跳过 %d 条（版本 %s）",
        linked,
        copied,
        len(skipped),
        target_version,
    )
    return result


def candidates_from_manual_links(
    cat: Catalog,
    links: Sequence[tuple[int, int]],
) -> list[IdentityCandidate]:
    """把 load_manual_links 的结果转换为 accepted 的手动 IdentityCandidate。"""

    candidates: list[IdentityCandidate] = []
    for local_id, bilibili_id in links:
        local_work = cat.get_work(int(local_id))
        bilibili_work = cat.get_work(int(bilibili_id))
        if local_work is None:
            raise IngestError(f"手动关联指向的本地作品不存在: work_id={local_id}")
        if bilibili_work is None:
            raise IngestError(f"手动关联指向的 Bilibili 作品不存在: work_id={bilibili_id}")
        candidates.append(
            IdentityCandidate(
                local_work_id=int(local_id),
                bilibili_work_id=int(bilibili_id),
                local_title=_local_display_title(local_work),
                bilibili_title=str(bilibili_work.get("title") or ""),
                local_duration_s=_as_float(local_work.get("duration_s")),
                bilibili_duration_s=_as_float(bilibili_work.get("duration_s")),
                method=MANUAL_METHOD,
                confidence=1.0,
                accepted=True,
                reason="手动关联清单（人工确认）",
            )
        )
    return candidates


# ---------------------------------------------------------------- 手动关联清单


def _read_manual_rows(path: Path) -> tuple[list[str], list[tuple[int, dict[str, str]]]]:
    """读取手动关联 CSV：跳过空行与 # 注释行，返回 (规范化表头列表, [(行号, 行字典)])。"""

    header: list[str] = []
    records: list[tuple[int, dict[str, str]]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        seen_header = False
        for cells in reader:
            line_no = reader.line_num
            if not cells:
                continue
            if all(not str(cell).strip() for cell in cells):
                continue
            if str(cells[0]).lstrip().startswith("#"):
                continue
            if not seen_header:
                header = [str(cell).strip().casefold() for cell in cells]
                seen_header = True
                continue
            if len(cells) > len(header):
                raise IngestError(
                    f"第 {line_no} 行: 列数({len(cells)})多于表头({len(header)})，"
                    "请检查是否有多余逗号"
                )
            row: dict[str, str] = {}
            for index, name in enumerate(header):
                if name:
                    row[name] = str(cells[index]).strip() if index < len(cells) else ""
            records.append((line_no, row))
    return header, records


def _manual_cell(row: dict[str, str], field: str) -> str:
    for alias in _MANUAL_COLUMNS[field]:
        if alias in row:
            return row[alias]
    return ""


def load_manual_links(cat: Catalog, config: Config) -> list[tuple[int, int]]:
    """读取 input/lists/local_bv_links.csv 手动覆盖清单。

    列：local_path（相对 input/local 的路径，必填）、bvid（必填）、part（可选，默认 1）。
    返回 [(local_work_id, bilibili_work_id)]，同一 local_path 出现多次时最后一行为准。
    文件不存在时返回空列表；文件存在但内容非法时抛 IngestError（带行号）。
    """

    path = config.paths.input_lists / MANUAL_LINKS_FILENAME
    if not path.is_file():
        return []
    header, records = _read_manual_rows(path)
    header_set = set(header)
    for field in ("local_path", "bvid"):
        if not any(alias in header_set for alias in _MANUAL_COLUMNS[field]):
            accepted = " / ".join(_MANUAL_COLUMNS[field])
            raise IngestError(f"手动关联清单缺少 {field} 列（可用列名: {accepted}）: {path}")

    pairs: dict[int, int] = {}
    for line_no, row in records:
        if not any(row.values()):
            continue
        raw_local_path = _manual_cell(row, "local_path")
        raw_bvid = _manual_cell(row, "bvid")
        raw_part = _manual_cell(row, "part")
        if not raw_local_path:
            raise IngestError(f"第 {line_no} 行: local_path 不能为空")
        if not raw_bvid:
            raise IngestError(f"第 {line_no} 行: bvid 不能为空")
        try:
            rel_path = normalize_relpath(raw_local_path)
        except ValueError as exc:
            raise IngestError(f"第 {line_no} 行: local_path 非法: {raw_local_path!r}（{exc}）") from exc
        if not BVID_RE.fullmatch(raw_bvid):
            raise IngestError(f"第 {line_no} 行: BV 号格式非法: {raw_bvid!r}")
        if raw_part:
            try:
                part = int(raw_part, 10)
            except ValueError as exc:
                raise IngestError(f"第 {line_no} 行: part 必须是整数，收到 {raw_part!r}") from exc
        else:
            part = 1
        if part < 1:
            raise IngestError(f"第 {line_no} 行: part 必须 >= 1，收到 {part}")

        local_work = cat.get_work_by_key(work_key_for_local(rel_path))
        if local_work is None:
            raise IngestError(
                f"第 {line_no} 行: 本地作品未登记: {rel_path}（请先运行 ingest.py scan-local）"
            )
        bilibili_work = cat.get_work_by_bvid(raw_bvid, part=part)
        if bilibili_work is None:
            raise IngestError(
                f"第 {line_no} 行: Bilibili 作品未登记: {raw_bvid} p{part}"
                "（请先运行 ingest.py import-votes）"
            )
        pairs[int(local_work["id"])] = int(bilibili_work["id"])
    LOGGER.info("手动关联清单: %d 条（%s）", len(pairs), path)
    return list(pairs.items())
