"""SQLite 编目：作品、榜单、向量登记、处理状态与星云版本。

设计要点：

- 作品（works）只存一份；同一作品出现在多个榜单/年份时，只在
  appearances 表新增出现记录（P2 去重原则）。
- 向量文件本身不存进数据库，只在 embeddings 表登记路径、维度、
  向量版本与片段数量；向量版本由“模型版本 + 采样参数”决定。
- 任务状态严格按 pending -> downloading -> encoding -> completed
  迁移，任一环节失败进入 failed，允许重试。
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from . import EMBEDDING_KINDS
from .errors import DatabaseError
from .util import now_utc_iso, normalize_relpath

SCHEMA_VERSION = "1"

STATUSES = ("pending", "downloading", "encoding", "completed", "failed")

TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset({"downloading", "encoding", "failed"}),
    "downloading": frozenset({"encoding", "failed"}),
    "encoding": frozenset({"completed", "failed"}),
    "completed": frozenset(),
    "failed": frozenset({"pending", "downloading", "encoding"}),
}

WORK_FIELDS = (
    "url",
    "title",
    "author_uid",
    "author_name",
    "duration_s",
    "publish_date",
    "bvid",
    "part",
    "has_audio",
    "has_video",
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS works (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    work_key TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL CHECK (source IN ('local', 'bilibili')),
    bvid TEXT,
    part INTEGER NOT NULL DEFAULT 1,
    url TEXT,
    title TEXT,
    author_uid TEXT,
    author_name TEXT,
    duration_s REAL,
    publish_date TEXT,
    local_path TEXT,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'downloading', 'encoding', 'completed', 'failed')),
    error TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    has_audio INTEGER,
    has_video INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_works_bvid_part
    ON works(bvid, part) WHERE bvid IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_works_status ON works(status);
CREATE INDEX IF NOT EXISTS ix_works_source ON works(source);

CREATE TABLE IF NOT EXISTS work_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    work_id INTEGER NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    ts TEXT NOT NULL,
    event TEXT NOT NULL,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS ix_work_events_work ON work_events(work_id);

CREATE TABLE IF NOT EXISTS lists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    list_key TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('votes', 'weekly', 'manual')),
    vote_year INTEGER,
    url TEXT,
    source_path TEXT,
    imported_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS appearances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    work_id INTEGER NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    list_id INTEGER NOT NULL REFERENCES lists(id) ON DELETE CASCADE,
    rank INTEGER,
    votes INTEGER,
    note TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (work_id, list_id)
);
CREATE INDEX IF NOT EXISTS ix_appearances_list ON appearances(list_id);

CREATE TABLE IF NOT EXISTS embedding_versions (
    version TEXT PRIMARY KEY,
    model_checkpoint TEXT NOT NULL,
    model_revision TEXT,
    frames INTEGER NOT NULL,
    clip_seconds REAL NOT NULL,
    clips_per_work INTEGER NOT NULL,
    audio_sample_rate INTEGER NOT NULL,
    params_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    work_id INTEGER NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    version TEXT NOT NULL REFERENCES embedding_versions(version),
    kind TEXT NOT NULL CHECK (kind IN (
        'work_joint', 'work_audio', 'work_video',
        'clip_joint', 'clip_audio', 'clip_video'
    )),
    path TEXT NOT NULL,
    dim INTEGER NOT NULL,
    n_clips INTEGER,
    created_at TEXT NOT NULL,
    UNIQUE (work_id, version, kind)
);
CREATE INDEX IF NOT EXISTS ix_embeddings_version_kind ON embeddings(version, kind);

CREATE TABLE IF NOT EXISTS clips (
    work_id INTEGER NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    version TEXT NOT NULL,
    idx INTEGER NOT NULL,
    start_s REAL NOT NULL,
    duration_s REAL NOT NULL,
    PRIMARY KEY (work_id, version, idx)
);

CREATE TABLE IF NOT EXISTS map_runs (
    version TEXT PRIMARY KEY,
    embedding_version TEXT NOT NULL REFERENCES embedding_versions(version),
    params_json TEXT NOT NULL,
    coords_path TEXT,
    model_2d_path TEXT,
    model_3d_path TEXT,
    n_works INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'building'
        CHECK (status IN ('building', 'ready', 'failed')),
    note TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS map_coords (
    map_version TEXT NOT NULL REFERENCES map_runs(version) ON DELETE CASCADE,
    work_id INTEGER NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    x2 REAL,
    y2 REAL,
    x3 REAL,
    y3 REAL,
    z3 REAL,
    PRIMARY KEY (map_version, work_id)
);
"""


def _as_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _as_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def work_key_for_local(relative_path: str) -> str:
    return "local:" + normalize_relpath(relative_path)


def work_key_for_bilibili(bvid: str, part: int = 1) -> str:
    bvid = (bvid or "").strip()
    if not bvid:
        raise DatabaseError("bvid 不能为空")
    if part < 1:
        raise DatabaseError(f"分P必须 >= 1: {part}")
    return f"bilibili:{bvid}:p{part}"


class Catalog:
    """一个 SQLite 编目连接。对象创建后需先调用 initialize()。"""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------ 基础
    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA busy_timeout = 30000")
            self._conn = conn
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "Catalog":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    def initialize(self) -> None:
        """建表并校验 schema 版本；可重复调用。"""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self.conn
        conn.executescript(SCHEMA_SQL)
        with self.transaction() as tx:
            row = tx.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
            if row is None:
                tx.execute(
                    "INSERT INTO schema_meta(key, value) VALUES ('schema_version', ?)",
                    (SCHEMA_VERSION,),
                )
            elif str(row["value"]) != SCHEMA_VERSION:
                raise DatabaseError(
                    f"数据库 schema 版本不匹配: 文件为 {row['value']}, 程序需要 {SCHEMA_VERSION}; "
                    "请备份后重建或执行迁移"
                )

    # ------------------------------------------------------------------ 作品
    def upsert_work(
        self,
        *,
        source: str,
        work_key: str,
        bvid: str | None = None,
        part: int = 1,
        url: str | None = None,
        title: str | None = None,
        author_uid: str | None = None,
        author_name: str | None = None,
        duration_s: float | None = None,
        publish_date: str | None = None,
        local_path: str | None = None,
    ) -> tuple[int, bool]:
        """按 work_key 幂等登记作品。

        返回 (work_id, created)；已存在时只补全缺失字段，不重置状态。
        """

        if source not in ("local", "bilibili"):
            raise DatabaseError(f"未知来源: {source}")
        if source == "local" and not local_path:
            raise DatabaseError("本地作品必须提供 local_path")
        if source == "bilibili" and not bvid:
            raise DatabaseError("Bilibili 作品必须提供 bvid")
        now = now_utc_iso()
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM works WHERE work_key = ?", (work_key,)).fetchone()
            if row is None:
                cur = conn.execute(
                    """
                    INSERT INTO works (
                        work_key, source, bvid, part, url, title, author_uid, author_name,
                        duration_s, publish_date, local_path, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                    """,
                    (
                        work_key,
                        source,
                        bvid,
                        part,
                        url,
                        title,
                        author_uid,
                        author_name,
                        duration_s,
                        publish_date,
                        local_path,
                        now,
                        now,
                    ),
                )
                work_id = int(cur.lastrowid)
                self._event(
                    conn,
                    work_id,
                    "created",
                    json.dumps({"source": source, "work_key": work_key}, ensure_ascii=False),
                )
                return work_id, True
            work_id = int(row["id"])
            conn.execute(
                """
                UPDATE works SET
                    url = COALESCE(?, url),
                    title = COALESCE(?, title),
                    author_uid = COALESCE(?, author_uid),
                    author_name = COALESCE(?, author_name),
                    duration_s = COALESCE(?, duration_s),
                    publish_date = COALESCE(?, publish_date),
                    local_path = COALESCE(?, local_path),
                    bvid = COALESCE(?, bvid),
                    part = COALESCE(?, part),
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    url,
                    title,
                    author_uid,
                    author_name,
                    duration_s,
                    publish_date,
                    local_path,
                    bvid,
                    part,
                    now,
                    work_id,
                ),
            )
            return work_id, False

    def upsert_local_work(self, relative_path: str, *, title: str | None = None) -> tuple[int, bool]:
        rel = normalize_relpath(relative_path)
        return self.upsert_work(
            source="local",
            work_key=work_key_for_local(rel),
            local_path=rel,
            title=title,
        )

    def upsert_bilibili_work(
        self,
        bvid: str,
        *,
        part: int = 1,
        title: str | None = None,
        author_uid: str | None = None,
        author_name: str | None = None,
        duration_s: float | None = None,
        publish_date: str | None = None,
        url: str | None = None,
    ) -> tuple[int, bool]:
        key = work_key_for_bilibili(bvid, part)
        if url is None:
            url = f"https://www.bilibili.com/video/{bvid}"
            if part > 1:
                url += f"?p={part}"
        return self.upsert_work(
            source="bilibili",
            work_key=key,
            bvid=bvid,
            part=part,
            url=url,
            title=title,
            author_uid=author_uid,
            author_name=author_name,
            duration_s=duration_s,
            publish_date=publish_date,
        )

    def get_work(self, work_id: int) -> dict[str, Any] | None:
        return _as_dict(self.conn.execute("SELECT * FROM works WHERE id = ?", (work_id,)).fetchone())

    def get_work_by_key(self, work_key: str) -> dict[str, Any] | None:
        return _as_dict(
            self.conn.execute("SELECT * FROM works WHERE work_key = ?", (work_key,)).fetchone()
        )

    def get_work_by_bvid(self, bvid: str, part: int = 1) -> dict[str, Any] | None:
        return _as_dict(
            self.conn.execute(
                "SELECT * FROM works WHERE bvid = ? AND part = ?", (bvid, part)
            ).fetchone()
        )

    def list_works(
        self,
        *,
        status: str | None = None,
        source: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM works"
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            if status not in STATUSES:
                raise DatabaseError(f"未知状态: {status}")
            clauses.append("status = ?")
            params.append(status)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id"
        if limit is not None:
            if limit < 1:
                raise DatabaseError("limit 必须 >= 1")
            sql += " LIMIT ?"
            params.append(limit)
        return _as_dicts(self.conn.execute(sql, params).fetchall())

    def count_works_by_status(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM works GROUP BY status"
        ).fetchall()
        counts = {status: 0 for status in STATUSES}
        for row in rows:
            counts[str(row["status"])] = int(row["n"])
        return counts

    def update_work_metadata(self, work_id: int, **fields: Any) -> None:
        unknown = set(fields) - set(WORK_FIELDS)
        if unknown:
            raise DatabaseError(f"不允许更新的字段: {', '.join(sorted(unknown))}")
        if not fields:
            return
        assignments = ", ".join(f"{name} = ?" for name in fields)
        params = list(fields.values()) + [now_utc_iso(), work_id]
        with self.transaction() as conn:
            cur = conn.execute(
                f"UPDATE works SET {assignments}, updated_at = ? WHERE id = ?", params
            )
            if cur.rowcount == 0:
                raise DatabaseError(f"作品不存在: {work_id}")

    # ------------------------------------------------------------------ 状态
    def set_status(
        self,
        work_id: int,
        status: str,
        *,
        error: str | None = None,
        note: str | None = None,
    ) -> str:
        """执行一次状态迁移，返回旧状态。非法迁移会抛 DatabaseError。"""

        if status not in STATUSES:
            raise DatabaseError(f"未知状态: {status}")
        now = now_utc_iso()
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT status, attempts FROM works WHERE id = ?", (work_id,)
            ).fetchone()
            if row is None:
                raise DatabaseError(f"作品不存在: {work_id}")
            old = str(row["status"])
            if old == status:
                if error is not None:
                    conn.execute(
                        "UPDATE works SET error = ?, updated_at = ? WHERE id = ?",
                        (error, now, work_id),
                    )
                return old
            if status not in TRANSITIONS[old]:
                raise DatabaseError(f"非法状态迁移: {old} -> {status} (work {work_id})")
            attempts = int(row["attempts"])
            # 只在真正开始一次处理时计数；failed -> pending 只是重置，不算一次尝试
            if old in ("pending", "failed") and status in ("downloading", "encoding"):
                attempts += 1
            conn.execute(
                "UPDATE works SET status = ?, error = ?, attempts = ?, updated_at = ? WHERE id = ?",
                (status, error, attempts, now, work_id),
            )
            self._event(conn, work_id, "status", f"{old}->{status}" + (f" | {note}" if note else ""))
            return old

    def begin_attempt(self, work_id: int, stage: str) -> str:
        if stage not in ("downloading", "encoding"):
            raise DatabaseError(f"未知处理阶段: {stage}")
        return self.set_status(work_id, stage)

    def fail(self, work_id: int, message: str) -> str:
        text = (message or "未知错误").strip()
        if len(text) > 4000:
            text = text[:4000] + "..."
        return self.set_status(work_id, "failed", error=text)

    def complete(self, work_id: int) -> str:
        return self.set_status(work_id, "completed", error=None)

    def record_event(self, work_id: int, event: str, detail: str | None = None) -> None:
        with self.transaction() as conn:
            self._event(conn, work_id, event, detail)

    def get_events(self, work_id: int, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM work_events WHERE work_id = ? ORDER BY id DESC LIMIT ?",
            (work_id, limit),
        ).fetchall()
        return _as_dicts(rows)

    def _event(
        self,
        conn: sqlite3.Connection,
        work_id: int,
        event: str,
        detail: str | None,
    ) -> None:
        conn.execute(
            "INSERT INTO work_events(work_id, ts, event, detail) VALUES (?, ?, ?, ?)",
            (work_id, now_utc_iso(), event, detail),
        )

    # ------------------------------------------------------------------ 向量
    def register_embedding_version(
        self,
        version: str,
        *,
        model_checkpoint: str,
        model_revision: str | None,
        frames: int,
        clip_seconds: float,
        clips_per_work: int,
        audio_sample_rate: int,
        params: Mapping[str, Any],
    ) -> None:
        """登记向量版本；同版本必须参数一致，否则报错（防止静默覆盖）。"""

        if not version or not str(version).strip():
            raise DatabaseError("向量版本号不能为空")
        payload = dict(params or {})
        params_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        row_values = (
            model_checkpoint,
            model_revision,
            int(frames),
            float(clip_seconds),
            int(clips_per_work),
            int(audio_sample_rate),
            params_json,
        )
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM embedding_versions WHERE version = ?", (version,)
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO embedding_versions (
                        version, model_checkpoint, model_revision, frames, clip_seconds,
                        clips_per_work, audio_sample_rate, params_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (version, *row_values, now_utc_iso()),
                )
                return
            actual = (
                row["model_checkpoint"],
                row["model_revision"],
                int(row["frames"]),
                float(row["clip_seconds"]),
                int(row["clips_per_work"]),
                int(row["audio_sample_rate"]),
                row["params_json"],
            )
            if actual != row_values:
                raise DatabaseError(
                    f"向量版本 {version} 已存在但参数不同；"
                    "请修改 model.version / 采样参数以生成新版本，而不是覆盖旧版本"
                )

    def get_embedding_version(self, version: str) -> dict[str, Any] | None:
        return _as_dict(
            self.conn.execute(
                "SELECT * FROM embedding_versions WHERE version = ?", (version,)
            ).fetchone()
        )

    def register_embedding(
        self,
        work_id: int,
        *,
        version: str,
        kind: str,
        path: str,
        dim: int,
        n_clips: int | None = None,
    ) -> None:
        if kind not in EMBEDDING_KINDS:
            raise DatabaseError(f"未知向量种类: {kind}")
        if dim < 1:
            raise DatabaseError(f"向量维度非法: {dim}")
        if not path:
            raise DatabaseError("向量路径不能为空")
        now = now_utc_iso()
        with self.transaction() as conn:
            work = conn.execute("SELECT 1 FROM works WHERE id = ?", (work_id,)).fetchone()
            if work is None:
                raise DatabaseError(f"作品不存在: {work_id}")
            version_row = conn.execute(
                "SELECT 1 FROM embedding_versions WHERE version = ?", (version,)
            ).fetchone()
            if version_row is None:
                raise DatabaseError(f"向量版本未登记: {version}")
            conn.execute(
                """
                INSERT INTO embeddings (work_id, version, kind, path, dim, n_clips, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(work_id, version, kind) DO UPDATE SET
                    path = excluded.path,
                    dim = excluded.dim,
                    n_clips = excluded.n_clips,
                    created_at = excluded.created_at
                """,
                (work_id, version, kind, path, dim, n_clips, now),
            )

    def get_embedding(self, work_id: int, *, version: str, kind: str = "work_joint") -> dict[str, Any] | None:
        return _as_dict(
            self.conn.execute(
                "SELECT * FROM embeddings WHERE work_id = ? AND version = ? AND kind = ?",
                (work_id, version, kind),
            ).fetchone()
        )

    def has_embedding(self, work_id: int, *, version: str, kind: str = "work_joint") -> bool:
        return self.get_embedding(work_id, version=version, kind=kind) is not None

    def list_embeddings(self, version: str, *, kind: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM embeddings WHERE version = ?"
        params: list[Any] = [version]
        if kind is not None:
            if kind not in EMBEDDING_KINDS:
                raise DatabaseError(f"未知向量种类: {kind}")
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY work_id"
        return _as_dicts(self.conn.execute(sql, params).fetchall())

    def list_works_with_embedding(
        self, version: str, *, kind: str = "work_joint"
    ) -> list[dict[str, Any]]:
        """返回已登记该种类向量的作品（含路径与维度），按 work_id 排序。"""

        rows = self.conn.execute(
            """
            SELECT w.*, e.path AS embedding_path, e.dim AS embedding_dim,
                   e.n_clips AS embedding_n_clips
            FROM works w
            JOIN embeddings e ON e.work_id = w.id
            WHERE e.version = ? AND e.kind = ?
            ORDER BY w.id
            """,
            (version, kind),
        ).fetchall()
        return _as_dicts(rows)

    def replace_clips(
        self,
        work_id: int,
        *,
        version: str,
        clips: Sequence[tuple[int, float, float]],
    ) -> None:
        with self.transaction() as conn:
            conn.execute("DELETE FROM clips WHERE work_id = ? AND version = ?", (work_id, version))
            conn.executemany(
                "INSERT INTO clips(work_id, version, idx, start_s, duration_s) VALUES (?, ?, ?, ?, ?)",
                [(work_id, version, int(idx), float(start), float(dur)) for idx, start, dur in clips],
            )

    def get_clips(self, work_id: int, *, version: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM clips WHERE work_id = ? AND version = ? ORDER BY idx",
            (work_id, version),
        ).fetchall()
        return _as_dicts(rows)

    # ------------------------------------------------------------------ 榜单
    def ensure_list(
        self,
        list_key: str,
        *,
        name: str,
        kind: str,
        vote_year: int | None = None,
        url: str | None = None,
        source_path: str | None = None,
    ) -> int:
        if kind not in ("votes", "weekly", "manual"):
            raise DatabaseError(f"未知榜单类型: {kind}")
        now = now_utc_iso()
        with self.transaction() as conn:
            row = conn.execute("SELECT id FROM lists WHERE list_key = ?", (list_key,)).fetchone()
            if row is not None:
                return int(row["id"])
            cur = conn.execute(
                """
                INSERT INTO lists(list_key, name, kind, vote_year, url, source_path, imported_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (list_key, name, kind, vote_year, url, source_path, now),
            )
            return int(cur.lastrowid)

    def add_appearance(
        self,
        work_id: int,
        list_id: int,
        *,
        rank: int | None = None,
        votes: int | None = None,
        note: str | None = None,
    ) -> None:
        if rank is not None and rank < 0:
            raise DatabaseError("rank 不能为负")
        if votes is not None and votes < 0:
            raise DatabaseError("votes 不能为负")
        now = now_utc_iso()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO appearances(work_id, list_id, rank, votes, note, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(work_id, list_id) DO UPDATE SET
                    rank = COALESCE(excluded.rank, appearances.rank),
                    votes = COALESCE(excluded.votes, appearances.votes),
                    note = COALESCE(excluded.note, appearances.note)
                """,
                (work_id, list_id, rank, votes, note, now),
            )

    def get_appearances(self, work_id: int) -> list[dict[str, Any]]:
        return _as_dicts(
            self.conn.execute(
                """
                SELECT a.*, l.list_key, l.name AS list_name, l.kind AS list_kind,
                       l.vote_year
                FROM appearances a
                JOIN lists l ON l.id = a.list_id
                WHERE a.work_id = ?
                ORDER BY a.id
                """,
                (work_id,),
            ).fetchall()
        )

    def appearance_rows(self) -> list[dict[str, Any]]:
        return _as_dicts(
            self.conn.execute(
                """
                SELECT a.work_id, a.rank, a.votes, l.list_key, l.name AS list_name,
                       l.kind AS list_kind, l.vote_year
                FROM appearances a
                JOIN lists l ON l.id = a.list_id
                ORDER BY a.work_id, l.vote_year, a.rank
                """
            ).fetchall()
        )

    def list_lists(self) -> list[dict[str, Any]]:
        return _as_dicts(self.conn.execute("SELECT * FROM lists ORDER BY id").fetchall())

    # ------------------------------------------------------------------ 星云
    def next_map_version(self) -> str:
        rows = self.conn.execute(
            "SELECT version FROM map_runs WHERE version LIKE 'map_v%'"
        ).fetchall()
        highest = 0
        for row in rows:
            try:
                highest = max(highest, int(str(row["version"]).split("map_v", 1)[1]))
            except (ValueError, IndexError):
                continue
        return f"map_v{highest + 1:03d}"

    def create_map_run(
        self,
        version: str,
        *,
        embedding_version: str,
        params: Mapping[str, Any],
        coords: Sequence[tuple[int, float, float, float, float, float]],
        coords_path: str | None = None,
        model_2d_path: str | None = None,
        model_3d_path: str | None = None,
        note: str | None = None,
    ) -> None:
        """原子地登记一次星云生成：版本信息 + 全部坐标。"""

        if not version.startswith("map_v"):
            raise DatabaseError(f"星云版本号必须以 map_v 开头: {version}")
        rows = list(coords)
        now = now_utc_iso()
        with self.transaction() as conn:
            exists = conn.execute(
                "SELECT 1 FROM map_runs WHERE version = ?", (version,)
            ).fetchone()
            if exists is not None:
                raise DatabaseError(f"星云版本已存在: {version}")
            conn.execute(
                """
                INSERT INTO map_runs (
                    version, embedding_version, params_json, coords_path,
                    model_2d_path, model_3d_path, n_works, status, note, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'ready', ?, ?)
                """,
                (
                    version,
                    embedding_version,
                    json.dumps(dict(params), sort_keys=True, ensure_ascii=False),
                    coords_path,
                    model_2d_path,
                    model_3d_path,
                    len(rows),
                    note,
                    now,
                ),
            )
            conn.executemany(
                """
                INSERT INTO map_coords(map_version, work_id, x2, y2, x3, y3, z3)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [(version, *row) for row in rows],
            )

    def get_map(self, version: str) -> dict[str, Any] | None:
        return _as_dict(
            self.conn.execute("SELECT * FROM map_runs WHERE version = ?", (version,)).fetchone()
        )

    def latest_map(self, *, ready_only: bool = True) -> dict[str, Any] | None:
        sql = "SELECT * FROM map_runs"
        if ready_only:
            sql += " WHERE status = 'ready'"
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT 1"
        return _as_dict(self.conn.execute(sql).fetchone())

    def list_maps(self) -> list[dict[str, Any]]:
        return _as_dicts(
            self.conn.execute("SELECT * FROM map_runs ORDER BY created_at, rowid").fetchall()
        )

    def load_map_rows(self, version: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT w.*, c.x2, c.y2, c.x3, c.y3, c.z3
            FROM map_coords c
            JOIN works w ON w.id = c.work_id
            WHERE c.map_version = ?
            ORDER BY w.id
            """,
            (version,),
        ).fetchall()
        return _as_dicts(rows)

    def replace_map_coords(
        self,
        version: str,
        coords: Sequence[tuple[int, float, float, float, float, float]],
    ) -> None:
        with self.transaction() as conn:
            row = conn.execute("SELECT 1 FROM map_runs WHERE version = ?", (version,)).fetchone()
            if row is None:
                raise DatabaseError(f"星云版本不存在: {version}")
            conn.execute("DELETE FROM map_coords WHERE map_version = ?", (version,))
            conn.executemany(
                """
                INSERT INTO map_coords(map_version, work_id, x2, y2, x3, y3, z3)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [(version, *row) for row in coords],
            )
            conn.execute(
                "UPDATE map_runs SET n_works = ? WHERE version = ?", (len(coords), version)
            )

    # ------------------------------------------------------------------ 统计
    def stats(self) -> dict[str, Any]:
        try:
            status = self.count_works_by_status()
            version_row = self.conn.execute(
                "SELECT COUNT(*) AS n FROM embedding_versions"
            ).fetchone()
            embedding_rows = self.conn.execute(
                "SELECT version, kind, COUNT(*) AS n FROM embeddings GROUP BY version, kind"
            ).fetchall()
            map_rows = self.conn.execute("SELECT COUNT(*) AS n FROM map_runs").fetchone()
            list_rows = self.conn.execute("SELECT COUNT(*) AS n FROM lists").fetchone()
        except sqlite3.OperationalError:
            # 尚未 initialize
            return {"status": status, "tables_ready": False}
        return {
            "tables_ready": True,
            "status": status,
            "works_total": sum(status.values()),
            "embedding_versions": int(version_row["n"]) if version_row else 0,
            "embeddings": {
                f"{row['version']}|{row['kind']}": int(row["n"]) for row in embedding_rows
            },
            "maps": int(map_rows["n"]) if map_rows else 0,
            "lists": int(list_rows["n"]) if list_rows else 0,
        }
