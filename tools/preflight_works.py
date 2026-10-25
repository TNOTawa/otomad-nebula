"""全部 89 部 2025 选票作品的批量处理预检（只读，不下载、不写生产 DB）。

在把选票从已处理的前 30 部扩到全部 89 部之前，用真实 Bilibili API 对每部作品
做元数据预检，回答三个问题：

1. 哪些作品适合批量下载/编码（时长、分P、可用性）；
2. 哪些应该暂缓/跳过（超长、不可用、已处理）；
3. 未处理作品中是否有可以复用本地已有向量的身份关联候选（免下载/免编码）。

约束：
- 只调用 Bilibili 元数据 API；不下载视频、不加载模型、不占用 GPU。
- 不写 data/catalog.sqlite：生产库只用 sqlite3 只读 URI 查询；身份关联预演在
  data/catalog.sqlite 的临时快照上执行，所有写操作只发生在系统临时目录。
- 自动重试 + 串行请求间隔，避免触发风控；失败记录原因不阻塞其余作品。
- 输出 logs/preflight_89.json（结构化明细）+ 控制台中文摘要（按风险分组，不刷屏）。

用法：
    .\\.venv\\Scripts\\python.exe tools/preflight_works.py
    .\\.venv\\Scripts\\python.exe tools/preflight_works.py --interval 0.2 --out logs/preflight_89.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# tools/ 脚本直接运行时，把项目根目录加入 import 路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nebula.config import Config, load_config  # noqa: E402
from nebula.db import Catalog  # noqa: E402
from nebula.download import BilibiliClient, BilibiliMeta  # noqa: E402
from nebula.errors import DownloadError, NebulaError  # noqa: E402
from nebula.identity import find_candidates  # noqa: E402
from nebula.ingest import import_votes_csv, parse_votes_csv  # noqa: E402
from nebula.util import now_utc_iso  # noqa: E402

LOGGER = logging.getLogger("preflight_works")

CSV_FILENAME = "zuozhe_votes_2025.csv"
VOTE_YEAR = 2025
LIST_KEY = "votes_2025"
DEFAULT_OUT = "logs/preflight_89.json"

# 风险阈值（秒）：> 1h 提示留意，> 2h 建议暂缓（全量下载/编码代价过大）
DURATION_WARN_S = 3600.0
DURATION_LONG_S = 7200.0
DEFAULT_INTERVAL_S = 0.15
DEFAULT_RETRIES = 2
RETRY_BACKOFF_S = 1.5

RISK_ORDER = ("normal", "multi", "warn", "long", "unavailable")
RISK_LABELS = {
    "normal": "普通",
    "multi": "多分P",
    "warn": "较长(>1h)",
    "long": "超长(>2h)建议暂缓",
    "unavailable": "不可用",
}
RECOMMENDATION_LABELS = {
    "batch": "可批量处理",
    "batch_reuse_local": "可复用本地向量",
    "retry": "失败可重试",
    "skip": "建议跳过",
    "done": "已处理完成",
}


@dataclass(frozen=True)
class DbState:
    """生产库（只读快照）里与预检相关的状态。"""

    works_by_key: dict[tuple[str, int], dict[str, Any]]
    vector_work_ids: set[int]
    linked_duplicates: dict[int, list[int]]  # canonical work_id -> duplicate work_ids
    list_exists: bool


# ---------------------------------------------------------------- 只读数据库


def _readonly_connect(db_path: Path) -> sqlite3.Connection:
    """以只读方式打开生产 SQLite（URI mode=ro，绝不写入）。"""

    uri = db_path.resolve().as_uri() + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=30.0)
    except sqlite3.Error as exc:
        raise RuntimeError(f"无法只读打开数据库 {db_path}: {exc}") from exc
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def load_db_state(db_path: Path, version: str) -> DbState:
    """只读查询：作品、当前版本向量、身份关联。"""

    conn = _readonly_connect(db_path)
    try:
        tables = {
            str(row["name"])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if "works" not in tables:
            raise RuntimeError(f"数据库缺少 works 表，无法预检: {db_path}")
        works_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for row in conn.execute(
            "SELECT id, bvid, part, title, duration_s, status, source, error FROM works"
        ):
            record = dict(row)
            bvid = record.get("bvid")
            if not bvid:
                continue
            works_by_key[(str(bvid), int(record.get("part") or 1))] = record

        vector_work_ids: set[int] = set()
        if "embeddings" in tables:
            for row in conn.execute(
                "SELECT DISTINCT work_id FROM embeddings WHERE version = ? AND kind = 'work_joint'",
                (version,),
            ):
                vector_work_ids.add(int(row["work_id"]))

        linked_duplicates: dict[int, list[int]] = {}
        if "work_links" in tables:
            for row in conn.execute(
                "SELECT canonical_work_id, duplicate_work_id FROM work_links ORDER BY id"
            ):
                linked_duplicates.setdefault(int(row["canonical_work_id"]), []).append(
                    int(row["duplicate_work_id"])
                )

        list_exists = False
        if "lists" in tables:
            list_exists = (
                conn.execute("SELECT 1 FROM lists WHERE list_key = ? LIMIT 1", (LIST_KEY,)).fetchone()
                is not None
            )
    finally:
        conn.close()
    return DbState(
        works_by_key=works_by_key,
        vector_work_ids=vector_work_ids,
        linked_duplicates=linked_duplicates,
        list_exists=list_exists,
    )


def _snapshot_db(source: Path, destination: Path, *, retries: int = 3) -> None:
    """用 SQLite backup API 把生产库一致性快照到临时文件（含 WAL 已提交内容）。"""

    for attempt in range(1, retries + 1):
        src: sqlite3.Connection | None = None
        dst: sqlite3.Connection | None = None
        try:
            src = _readonly_connect(source)
            dst = sqlite3.connect(str(destination))
            src.backup(dst)
            return
        except sqlite3.Error as exc:
            if attempt >= retries:
                raise RuntimeError(f"快照数据库失败（重试 {retries} 次后放弃）: {exc}") from exc
            LOGGER.warning("快照数据库失败（第 %d 次），%.1fs 后重试: %s", attempt, RETRY_BACKOFF_S, exc)
            time.sleep(RETRY_BACKOFF_S)
        finally:
            if dst is not None:
                dst.close()
            if src is not None:
                src.close()


# ---------------------------------------------------------------- 元数据预检


def fetch_all_metas(
    rows: Iterable[dict[str, Any]],
    config: Config,
    *,
    interval_s: float = DEFAULT_INTERVAL_S,
    retries: int = DEFAULT_RETRIES,
) -> tuple[dict[tuple[str, int], BilibiliMeta], dict[tuple[str, int], str], int]:
    """串行调用真实 Bilibili API（带间隔与重试），返回 (成功, 失败原因, 请求次数)。"""

    row_list = list(rows)
    metas: dict[tuple[str, int], BilibiliMeta] = {}
    errors: dict[tuple[str, int], str] = {}
    requests = 0
    with BilibiliClient(config.bilibili) as client:
        for index, row in enumerate(row_list):
            key = (str(row["bvid"]), int(row["part"]))
            last_error = ""
            for attempt in range(1, retries + 2):
                requests += 1
                try:
                    metas[key] = client.fetch_meta(key[0], part=key[1])
                    break
                except DownloadError as exc:
                    last_error = str(exc)
                    if attempt <= retries:
                        LOGGER.warning(
                            "API 失败（%s p%s，第 %d 次），%.1fs 后重试: %s",
                            key[0],
                            key[1],
                            attempt,
                            RETRY_BACKOFF_S,
                            exc,
                        )
                        time.sleep(RETRY_BACKOFF_S)
                except NebulaError as exc:  # 其它可预期错误也记录，不阻塞
                    last_error = str(exc)
                    break
            if key not in metas:
                errors[key] = last_error or "未知错误"
                LOGGER.error("API 预检失败: %s p%s -> %s", key[0], key[1], errors[key])
            done = index + 1
            if done % 10 == 0 or done == len(row_list):
                LOGGER.info("元数据预检进度 %d/%d（成功 %d，失败 %d）", done, len(row_list), len(metas), len(errors))
            if index < len(row_list) - 1:
                time.sleep(max(0.0, interval_s))
    return metas, errors, requests


# ---------------------------------------------------------------- 分类与建议


def _duration_hours(duration_s: float | None) -> str:
    if duration_s is None:
        return "-"
    return f"{duration_s / 3600.0:.2f}h"


def _classify_risk(meta: BilibiliMeta | None) -> tuple[str, list[str]]:
    """返回 (主风险分类, 全部风险标记)。主分类按 unavailable > long > warn > multi > normal。"""

    if meta is None:
        return "unavailable", ["unavailable"]
    flags: list[str] = []
    if meta.page_count > 1:
        flags.append("multi")
    if meta.duration_s > DURATION_LONG_S:
        flags.append("long")
    elif meta.duration_s > DURATION_WARN_S:
        flags.append("warn")
    if "long" in flags:
        primary = "long"
    elif "warn" in flags:
        primary = "warn"
    elif "multi" in flags:
        primary = "multi"
    else:
        primary = "normal"
    return primary, flags


def build_record(
    index: int,
    row: dict[str, Any],
    meta: BilibiliMeta | None,
    error: str | None,
    db_state: DbState,
    version: str,
) -> dict[str, Any]:
    """把一行选票 + 元数据 + DB 状态整理成预检记录。"""

    key = (str(row["bvid"]), int(row["part"]))
    work = db_state.works_by_key.get(key)
    has_vector = bool(work is not None and int(work["id"]) in db_state.vector_work_ids)
    linked_duplicates = (
        list(db_state.linked_duplicates.get(int(work["id"]), [])) if work is not None else []
    )
    risk, flags = _classify_risk(meta)
    duration_s = meta.duration_s if meta is not None else None

    if meta is None:
        recommendation = "skip"
        reason = f"Bilibili API 不可用: {error or '未知错误'}"
    elif has_vector:
        recommendation = "done"
        reason = "已入库且已有当前版本向量"
    elif meta.duration_s > DURATION_LONG_S:
        recommendation = "skip"
        reason = f"超长视频（{_duration_hours(meta.duration_s)}，>2h），全量下载/编码代价过大"
    elif work is not None and str(work.get("status")) == "failed":
        recommendation = "retry"
        reason = f"已入库但处理失败（{work.get('error') or '未知原因'}），可重试"
    else:
        recommendation = "batch"
        reason = "可批量下载/编码"
    if recommendation in ("batch", "retry") and meta is not None and flags:
        if "warn" in flags:
            reason += "；时长超过 1h，注意下载/编码耗时"
        if "multi" in flags:
            reason += "；多分P，使用 part=1 的分P时长"

    return {
        "index": index,
        "bvid": key[0],
        "part": key[1],
        "csv_title": row.get("title"),
        "bilibili_title": meta.title if meta is not None else None,
        "rank": row.get("rank"),
        "votes": row.get("votes"),
        "duration_s": duration_s,
        "duration_h": round(duration_s / 3600.0, 3) if duration_s is not None else None,
        "page_count": meta.page_count if meta is not None else None,
        "part_title": meta.part_title if meta is not None else None,
        "publish_date": meta.publish_date if meta is not None else None,
        "author_uid": meta.author_uid if meta is not None else None,
        "author_name": meta.author_name if meta is not None else None,
        "risk": risk,
        "flags": flags,
        "in_db": work is not None,
        "work_id": int(work["id"]) if work is not None else None,
        "status": str(work.get("status")) if work is not None else None,
        "has_vector": has_vector,
        "linked_duplicate_work_ids": linked_duplicates,
        "recommendation": recommendation,
        "reason": reason,
        "error": error,
    }


def build_records(
    rows: list[dict[str, Any]],
    metas: dict[tuple[str, int], BilibiliMeta],
    errors: dict[tuple[str, int], str],
    db_state: DbState,
    version: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        key = (str(row["bvid"]), int(row["part"]))
        records.append(
            build_record(index, row, metas.get(key), errors.get(key), db_state, version)
        )
    return records


# ---------------------------------------------------------------- 身份关联预演


class _CachedBilibiliClient:
    """把已抓取的元数据包装成 BilibiliClient 形状，避免身份预演重复请求 API。"""

    def __init__(
        self,
        metas: dict[tuple[str, int], BilibiliMeta],
        errors: dict[tuple[str, int], str],
    ) -> None:
        self.metas = dict(metas)
        self.errors = dict(errors)
        self.calls = 0

    def fetch_meta(self, bvid: str, *, part: int = 1) -> BilibiliMeta:
        self.calls += 1
        key = (bvid, part)
        if key in self.metas:
            return self.metas[key]
        raise DownloadError(self.errors.get(key, f"预检未获取到元数据: {bvid} p{part}"))

    def close(self) -> None:
        pass


def run_identity_preview(
    config: Config,
    csv_path: Path,
    rows: list[dict[str, Any]],
    metas: dict[tuple[str, int], BilibiliMeta],
    errors: dict[tuple[str, int], str],
    remaining_keys: set[tuple[str, int]],
) -> dict[str, Any]:
    """在数据库临时快照上用真实 identity.find_candidates 逻辑预演身份关联。

    生产库只读；临时快照内会补齐全部 89 行（幂等 import）后再计算。
    """

    if not remaining_keys:
        return {
            "available": True,
            "note": "没有未处理作品，跳过身份预演",
            "total_candidates": 0,
            "remaining_candidates": [],
            "accepted": [],
            "rejected": [],
        }
    work_dir = Path(tempfile.mkdtemp(prefix="nebula_preflight_"))
    snapshot_path = work_dir / "catalog_snapshot.sqlite"
    temp_cat: Catalog | None = None
    try:
        _snapshot_db(config.paths.db_path, snapshot_path)
        temp_cat = Catalog(snapshot_path)
        temp_cat.initialize()
        import_votes_csv(temp_cat, config, csv_path, vote_year=VOTE_YEAR)
        client = _CachedBilibiliClient(metas, errors)
        candidates = find_candidates(temp_cat, config, list_key=LIST_KEY, client=client)
        work_ids = sorted({int(c.bilibili_work_id) for c in candidates})
        work_keys: dict[int, tuple[str, int]] = {}
        for work_id in work_ids:
            work = temp_cat.get_work(work_id)
            if work is not None and work.get("bvid"):
                work_keys[work_id] = (str(work["bvid"]), int(work.get("part") or 1))
        selected = [
            c for c in candidates if work_keys.get(int(c.bilibili_work_id)) in remaining_keys
        ]
        candidate_rows = []
        for candidate in selected:
            key = work_keys[int(candidate.bilibili_work_id)]
            candidate_rows.append(
                {
                    "bvid": key[0],
                    "part": key[1],
                    "local_work_id": int(candidate.local_work_id),
                    "local_title": candidate.local_title,
                    "bilibili_title": candidate.bilibili_title,
                    "local_duration_s": candidate.local_duration_s,
                    "bilibili_duration_s": candidate.bilibili_duration_s,
                    "accepted": bool(candidate.accepted),
                    "confidence": float(candidate.confidence),
                    "method": candidate.method,
                    "reason": candidate.reason,
                }
            )
        return {
            "available": True,
            "total_candidates": len(candidates),
            "remaining_candidates": candidate_rows,
            "accepted": [row for row in candidate_rows if row["accepted"]],
            "rejected": [row for row in candidate_rows if not row["accepted"]],
            "cached_client_calls": client.calls,
        }
    except (NebulaError, RuntimeError, sqlite3.Error) as exc:
        LOGGER.error("身份关联预演失败（不影响预检主体）: %s", exc)
        return {
            "available": False,
            "note": f"身份关联预演失败: {exc}",
            "total_candidates": 0,
            "remaining_candidates": [],
            "accepted": [],
            "rejected": [],
        }
    finally:
        if temp_cat is not None:
            temp_cat.close()
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------- 汇总与输出


def _remaining_keys(records: list[dict[str, Any]]) -> set[tuple[str, int]]:
    """未处理 = 没有当前版本向量的作品（pending 或 failed）。"""

    return {(r["bvid"], int(r["part"])) for r in records if not r["has_vector"]}


def apply_identity_results(
    records: list[dict[str, Any]], identity: dict[str, Any]
) -> None:
    """把身份预演结果写回记录：可复用本地向量的作品改推荐为 batch_reuse_local。"""

    accepted_keys = {
        (row["bvid"], int(row["part"])): row for row in identity.get("accepted", [])
    }
    for record in records:
        key = (record["bvid"], int(record["part"]))
        candidate = accepted_keys.get(key)
        if candidate is None:
            continue
        record["identity_candidate"] = candidate
        if record["recommendation"] in ("batch", "retry"):
            record["recommendation"] = "batch_reuse_local"
            record["reason"] = (
                "标题+分P时长与本地作品一致，可先运行 link-local 复用已有向量（免下载/编码）"
            )
        elif record["recommendation"] == "skip" and record.get("risk") == "long":
            record["identity_candidate_note"] = "即使有本地向量，超长视频仍建议暂缓"


def build_summary(records: list[dict[str, Any]], identity: dict[str, Any]) -> dict[str, Any]:
    by_risk = {risk: 0 for risk in RISK_ORDER}
    for record in records:
        by_risk[record["risk"]] = by_risk.get(record["risk"], 0) + 1
    flag_counts: dict[str, int] = {}
    for record in records:
        for flag in record["flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    in_db = sum(1 for record in records if record["in_db"])
    with_vector = sum(1 for record in records if record["has_vector"])
    linked = sum(1 for record in records if record["linked_duplicate_work_ids"])
    failed = sum(1 for record in records if record["status"] == "failed")
    remaining = len(_remaining_keys(records))

    recommendations: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        recommendations.setdefault(record["recommendation"], []).append(record)
    batch = recommendations.get("batch", [])
    reuse = recommendations.get("batch_reuse_local", [])
    retry = recommendations.get("retry", [])
    skip = recommendations.get("skip", [])
    done = recommendations.get("done", [])

    return {
        "total": len(records),
        "by_risk": by_risk,
        "flag_counts": flag_counts,
        "in_db": in_db,
        "with_vector": with_vector,
        "linked_duplicate_works": linked,
        "failed": failed,
        "remaining_no_vector": remaining,
        "recommendation_counts": {
            "batch": len(batch) + len(reuse) + len(retry),
            "batch_reuse_local": len(reuse),
            "retry": len(retry),
            "skip": len(skip),
            "done": len(done),
        },
        "identity_candidates": {
            "available": bool(identity.get("available")),
            "total_candidates": int(identity.get("total_candidates", 0)),
            "accepted": len(identity.get("accepted", [])),
            "rejected": len(identity.get("rejected", [])),
            "note": identity.get("note"),
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(payload, out, ensure_ascii=False, indent=2)
            out.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _short_title(value: str | None, limit: int = 42) -> str:
    text = str(value or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _format_seconds(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.0f}s"


def print_summary(
    payload: dict[str, Any],
    out_path: Path,
) -> None:
    summary = payload["summary"]
    records: list[dict[str, Any]] = payload["works"]
    print("=" * 78)
    print("2025 选票 89 部作品 · 批量处理预检")
    print("=" * 78)
    print(f"共 {summary['total']} 部作品，按风险分类：")
    for risk in RISK_ORDER:
        count = summary["by_risk"].get(risk, 0)
        if count:
            print(f"  {RISK_LABELS[risk]:<16} {count:>3} 部")
    print()
    print(
        f"数据库状态：已入库 {summary['in_db']}，已有向量 {summary['with_vector']}，"
        f"已关联本地去重 {summary['linked_duplicate_works']}，失败 {summary['failed']}，"
        f"未处理/无向量 {summary['remaining_no_vector']}"
    )
    ids = payload["identity_candidates"]
    if ids.get("available"):
        print(
            f"身份候选补充：接受 {len(ids.get('accepted', []))} 条 / "
            f"拒绝 {len(ids.get('rejected', []))} 条"
            f"（仅统计未处理作品；明细见 JSON）"
        )
        for candidate in ids.get("accepted", []):
            print(
                f"  + local#{candidate['local_work_id']} {_short_title(candidate['local_title'])}"
                f" <-> {candidate['bvid']} p{candidate['part']} "
                f"Δ{abs(float(candidate['local_duration_s']) - float(candidate['bilibili_duration_s'])):.2f}s"
            )
        for candidate in ids.get("rejected", [])[:5]:
            print(
                f"  - {candidate['bvid']} p{candidate['part']} "
                f"{_short_title(candidate['bilibili_title'], 30)}: {candidate['reason']}"
            )
    else:
        print(f"身份候选补充：未执行（{ids.get('note') or '原因未知'}）")
    print()
    rec = summary["recommendation_counts"]
    print(
        f"建议：可批量处理 {rec['batch']} 部"
        f"（其中可复用本地向量 {rec['batch_reuse_local']} 部，失败可重试 {rec['retry']} 部），"
        f"建议跳过 {rec['skip']} 部，已处理完成 {rec['done']} 部"
    )
    skip_records = [record for record in records if record["recommendation"] == "skip"]
    if skip_records:
        print("建议跳过清单：")
        for record in skip_records:
            print(
                f"  [{record['risk']}] {record['bvid']} p{record['part']} "
                f"rank={record['rank']} {_format_seconds(record['duration_s'])} "
                f"{_short_title(record['bilibili_title'] or record['csv_title'], 34)}"
                f" —— {record['reason']}"
            )
    multi_records = [record for record in records if "multi" in record["flags"]]
    if multi_records:
        print("多分P作品（按 part=1 的分P时长处理）：")
        for record in multi_records:
            print(
                f"  {record['bvid']} p{record['part']} pages={record['page_count']} "
                f"part1={_format_seconds(record['duration_s'])} "
                f"part_title={_short_title(record['part_title'], 24)} "
                f"推荐={RECOMMENDATION_LABELS.get(record['recommendation'], record['recommendation'])}"
            )
    warn_records = [record for record in records if record["risk"] == "warn"]
    if warn_records:
        print("较长但仍建议处理（>1h）：")
        for record in warn_records:
            print(
                f"  {record['bvid']} p{record['part']} rank={record['rank']} "
                f"{_format_seconds(record['duration_s'])} "
                f"{_short_title(record['bilibili_title'] or record['csv_title'], 34)}"
            )
    batch_records = [
        record
        for record in records
        if record["recommendation"] in ("batch", "batch_reuse_local", "retry")
    ]
    if batch_records:
        print(f"可批量处理清单：{len(batch_records)} 部（bvid 列表见 JSON 明细）")
        preview = ", ".join(record["bvid"] for record in batch_records[:8])
        print(f"  前 8 部：{preview}{' …' if len(batch_records) > 8 else ''}")
    print()
    print(f"JSON 明细：{out_path}")


# ---------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preflight_works.py",
        description="2025 选票 89 部作品批量处理预检（只读 DB，真实 Bilibili API，不下载）",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.yaml"),
        help="配置文件路径（默认项目根 config.yaml）",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help=f"选票 CSV 路径（默认 <input_lists>/{CSV_FILENAME}）",
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / DEFAULT_OUT),
        help=f"JSON 输出路径（默认 {DEFAULT_OUT}）",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_S,
        help=f"API 请求间隔秒数（默认 {DEFAULT_INTERVAL_S}，避免风控）",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"单个作品 API 重试次数（默认 {DEFAULT_RETRIES}）",
    )
    parser.add_argument(
        "--skip-identity",
        action="store_true",
        help="跳过身份关联预演（只做元数据与风险分类）",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="日志级别（默认 INFO）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = load_config(args.config)
        csv_path = Path(args.csv) if args.csv else config.paths.input_lists / CSV_FILENAME
        out_path = Path(args.out)
        if not csv_path.is_file():
            print(f"错误：选票 CSV 不存在: {csv_path}", file=sys.stderr)
            return 1
        rows = parse_votes_csv(csv_path, vote_year=VOTE_YEAR)
        if len(rows) != 89:
            LOGGER.warning("选票 CSV 应为 89 行，实际解析到 %d 行（继续执行）", len(rows))
        row_dicts = [
            {
                "bvid": row.bvid,
                "part": row.part,
                "title": row.title,
                "rank": row.rank,
                "votes": row.votes,
            }
            for row in rows
        ]
        db_state = load_db_state(config.paths.db_path, config.embedding_version)
        LOGGER.info("数据库只读状态：作品 %d，已有当前版本向量 %d", len(db_state.works_by_key), len(db_state.vector_work_ids))
        LOGGER.info("开始真实 Bilibili API 元数据预检（%d 部，间隔 %.2fs）...", len(row_dicts), args.interval)
        metas, errors, request_count = fetch_all_metas(
            row_dicts,
            config,
            interval_s=max(0.0, args.interval),
            retries=max(0, args.retries),
        )
        records = build_records(row_dicts, metas, errors, db_state, config.embedding_version)
        remaining = _remaining_keys(records)
        if args.skip_identity:
            identity: dict[str, Any] = {
                "available": False,
                "note": "已按 --skip-identity 跳过",
                "total_candidates": 0,
                "remaining_candidates": [],
                "accepted": [],
                "rejected": [],
            }
        else:
            LOGGER.info("身份关联预演：未处理作品 %d 部（临时 DB 快照，不写生产库）...", len(remaining))
            identity = run_identity_preview(
                config, csv_path, row_dicts, metas, errors, remaining
            )
        apply_identity_results(records, identity)
        summary = build_summary(records, identity)
        payload: dict[str, Any] = {
            "generated_at": now_utc_iso(),
            "inputs": {
                "csv": str(csv_path),
                "db_path": str(config.paths.db_path),
                "embedding_version": config.embedding_version,
                "list_key": LIST_KEY,
                "vote_year": VOTE_YEAR,
            },
            "thresholds": {
                "duration_warn_s": DURATION_WARN_S,
                "duration_long_s": DURATION_LONG_S,
            },
            "api": {
                "requests": request_count,
                "success": len(metas),
                "failed": len(errors),
                "interval_s": float(args.interval),
                "retries": int(args.retries),
            },
            "summary": summary,
            "identity_candidates": identity,
            "recommendations": {
                "batch": [
                    {
                        "bvid": record["bvid"],
                        "part": record["part"],
                        "rank": record["rank"],
                        "duration_s": record["duration_s"],
                        "risk": record["risk"],
                    }
                    for record in records
                    if record["recommendation"] in ("batch", "batch_reuse_local", "retry")
                ],
                "skip": [
                    {
                        "bvid": record["bvid"],
                        "part": record["part"],
                        "rank": record["rank"],
                        "duration_s": record["duration_s"],
                        "risk": record["risk"],
                        "reason": record["reason"],
                    }
                    for record in records
                    if record["recommendation"] == "skip"
                ],
                "done": [
                    {
                        "bvid": record["bvid"],
                        "part": record["part"],
                        "duration_s": record["duration_s"],
                    }
                    for record in records
                    if record["recommendation"] == "done"
                ],
            },
            "works": records,
        }
        _write_json(out_path, payload)
        print_summary(payload, out_path)
        return 0
    except (NebulaError, RuntimeError, sqlite3.Error) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        LOGGER.exception("预检失败")
        return 1
    except KeyboardInterrupt:
        print("预检已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
