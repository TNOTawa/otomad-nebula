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
from .util import normalize_relpath, now_utc_iso

SCHEMA_VERSION = "4"

STATUSES = ("pending", "downloading", "encoding", "completed", "failed")

#: 旧版“均匀抽 ≤N 段再平均”的表征名；迁移时自动为已有 embedding_versions 补登
LEGACY_REPRESENTATION_NAME = "legacy_5clip_mean"

REPRESENTATION_KINDS = ("legacy", "full_mean", "temporal_stats", "transformer", "visual")
REPRESENTATION_SOURCE_KINDS = ("pe_av", "visual")
REPRESENTATION_VECTOR_KINDS = ("joint", "audio", "video")
REPRESENTATION_STATUSES = ("building", "ready", "failed")
SEQUENCE_STATUSES = ("in_progress", "assembled", "failed")
VISUAL_SEQUENCE_STATUSES = ("pending", "in_progress", "completed", "failed")

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
    valid_duration_s REAL,
    weight REAL,
    PRIMARY KEY (work_id, version, idx)
);

-- 全片编码的片段级恢复状态（v0.4）：每成功编码一批就更新 n_clips_done
CREATE TABLE IF NOT EXISTS sequence_states (
    work_id INTEGER NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    version TEXT NOT NULL,
    duration_s REAL NOT NULL,
    n_clips INTEGER NOT NULL,
    n_clips_done INTEGER NOT NULL DEFAULT 0,
    partial_relpath TEXT,
    status TEXT NOT NULL DEFAULT 'in_progress'
        CHECK (status IN ('in_progress', 'assembled', 'failed')),
    updated_at TEXT NOT NULL,
    PRIMARY KEY (work_id, version)
);

-- 密集视觉语义通道（v0.4.3）：冻结视觉编码器版本；帧级+窗口级向量
CREATE TABLE IF NOT EXISTS visual_versions (
    version TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    revision TEXT NOT NULL,
    preprocessing TEXT NOT NULL CHECK (preprocessing IN ('stretch', 'letterbox')),
    fps REAL NOT NULL,
    size INTEGER NOT NULL,
    dim INTEGER,
    params_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS visual_sequences (
    work_id INTEGER NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    version TEXT NOT NULL REFERENCES visual_versions(version),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'in_progress', 'completed', 'failed')),
    frames_path TEXT,
    windows_path TEXT,
    n_frames INTEGER,
    n_windows INTEGER,
    dim INTEGER,
    error TEXT,
    decode_s REAL NOT NULL DEFAULT 0,
    encode_s REAL NOT NULL DEFAULT 0,
    model_load_s REAL NOT NULL DEFAULT 0,
    total_s REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (work_id, version)
);
CREATE INDEX IF NOT EXISTS ix_visual_sequences_version_status
    ON visual_sequences(version, status);

-- 表征层（v0.4）：聚合器版本，独立于 PE-AV 编码；binding 到 sequence_version
CREATE TABLE IF NOT EXISTS representations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    sequence_version TEXT NOT NULL,
    kind TEXT NOT NULL
        CHECK (kind IN ('legacy', 'full_mean', 'temporal_stats', 'transformer', 'visual')),
    source_kind TEXT NOT NULL DEFAULT 'pe_av'
        CHECK (source_kind IN ('pe_av', 'visual')),
    dim INTEGER NOT NULL DEFAULT 0,
    path_prefix TEXT,
    params_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'building'
        CHECK (status IN ('building', 'ready', 'failed')),
    note TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (name, sequence_version)
);

CREATE TABLE IF NOT EXISTS representation_vectors (
    representation_id INTEGER NOT NULL
        REFERENCES representations(id) ON DELETE CASCADE,
    work_id INTEGER NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('joint', 'audio', 'video')),
    path TEXT NOT NULL,
    dim INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (representation_id, work_id, kind)
);
CREATE INDEX IF NOT EXISTS ix_representation_vectors_rep
    ON representation_vectors(representation_id, kind);

CREATE TABLE IF NOT EXISTS map_runs (
    version TEXT PRIMARY KEY,
    embedding_version TEXT NOT NULL,
    representation_id INTEGER REFERENCES representations(id),
    source_kind TEXT NOT NULL DEFAULT 'pe_av'
        CHECK (source_kind IN ('pe_av', 'visual')),
    params_json TEXT NOT NULL,
    filter_json TEXT,
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

-- 身份关联：同一作品可能既有本地视频、又有 Bilibili 条目。
-- canonical_work_id 是保留在星云/展示中的实体；duplicate_work_id 是重复实体，
-- 默认不进入星云（其向量与原始文件仍然保留）。
CREATE TABLE IF NOT EXISTS work_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_work_id INTEGER NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    duplicate_work_id INTEGER NOT NULL UNIQUE REFERENCES works(id) ON DELETE CASCADE,
    method TEXT NOT NULL,
    confidence REAL,
    note TEXT,
    created_at TEXT NOT NULL,
    CHECK (canonical_work_id != duplicate_work_id)
);
CREATE INDEX IF NOT EXISTS ix_work_links_canonical ON work_links(canonical_work_id);

-- 流水线批次级指标（模型加载属于批次级）
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    embedding_version TEXT,
    model_load_s REAL,
    n_selected INTEGER NOT NULL DEFAULT 0,
    n_completed INTEGER NOT NULL DEFAULT 0,
    n_failed INTEGER NOT NULL DEFAULT 0,
    total_s REAL,
    params_json TEXT
);

-- 单部作品的分阶段耗时（A4）
CREATE TABLE IF NOT EXISTS work_timings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES pipeline_runs(id) ON DELETE SET NULL,
    work_id INTEGER NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    embedding_version TEXT NOT NULL,
    download_s REAL NOT NULL DEFAULT 0,
    probe_s REAL NOT NULL DEFAULT 0,
    extract_s REAL NOT NULL DEFAULT 0,
    preprocess_s REAL NOT NULL DEFAULT 0,
    forward_s REAL NOT NULL DEFAULT 0,
    save_s REAL NOT NULL DEFAULT 0,
    events_s REAL NOT NULL DEFAULT 0,
    total_s REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE (run_id, work_id)
);
CREATE INDEX IF NOT EXISTS ix_work_timings_work ON work_timings(work_id, embedding_version);
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
        """建表并校验/迁移 schema 版本；可重复调用。

        v4 迁移需要重建 representations 表以扩展 kind CHECK 约束；重建期间临时关闭
        外键检查（SQLite 的 PRAGMA foreign_keys 在事务内无效，因此必须在 BEGIN 之前设置）。
        """

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self.conn
        conn.executescript(SCHEMA_SQL)
        row = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
        current = None if row is None else str(row["value"])
        needs_migration = current is not None and current != SCHEMA_VERSION
        fk_disabled = needs_migration and current in ("1", "2", "3")
        if fk_disabled:
            conn.execute("PRAGMA foreign_keys = OFF")
        try:
            with self.transaction() as tx:
                if row is None:
                    tx.execute(
                        "INSERT INTO schema_meta(key, value) VALUES ('schema_version', ?)",
                        (SCHEMA_VERSION,),
                    )
                elif current != SCHEMA_VERSION:
                    self._migrate_schema(tx, current)
        finally:
            if fk_disabled:
                conn.execute("PRAGMA foreign_keys = ON")

    def _migrate_schema(self, conn: sqlite3.Connection, current: str) -> None:
        """把旧版本数据库原地迁移到当前 schema 版本。只做向后兼容的加表/加列。"""

        if current == "1":
            # v1 -> v2：筛选星云把条件写入 map_runs.filter_json
            columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(map_runs)")}
            if "filter_json" not in columns:
                conn.execute("ALTER TABLE map_runs ADD COLUMN filter_json TEXT")
            current = "2"
        if current == "2":
            self._migrate_v2_to_v3(conn)
            current = "3"
        if current == "3":
            self._migrate_v3_to_v4(conn)
            return
        raise DatabaseError(
            f"数据库 schema 版本不兼容: 文件为 {current}, 程序需要 {SCHEMA_VERSION}; "
            "请备份后重建或手工迁移"
        )

    def _migrate_v2_to_v3(self, conn: sqlite3.Connection) -> None:
        """v2 -> v3：全片序列列/状态表/表征层/地图绑定表征。

        - clips 增加有效时长与权重（旧行为 NULL）；
        - work_timings 增加事件扫描耗时；
        - map_runs 增加 representation_id；
        - 已有 embedding_versions 补登 legacy 表征，旧地图回填到对应 legacy 表征。
        """

        def columns(table: str) -> set[str]:
            return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})")}

        clip_columns = columns("clips")
        if "valid_duration_s" not in clip_columns:
            conn.execute("ALTER TABLE clips ADD COLUMN valid_duration_s REAL")
        if "weight" not in clip_columns:
            conn.execute("ALTER TABLE clips ADD COLUMN weight REAL")
        timing_columns = columns("work_timings")
        if "events_s" not in timing_columns:
            conn.execute(
                "ALTER TABLE work_timings ADD COLUMN events_s REAL NOT NULL DEFAULT 0"
            )
        map_columns = columns("map_runs")
        if "representation_id" not in map_columns:
            conn.execute(
                "ALTER TABLE map_runs ADD COLUMN representation_id INTEGER "
                "REFERENCES representations(id)"
            )

        now = now_utc_iso()
        params_json = json.dumps(
            {
                "source": "embeddings",
                "embedding_kind": "work_joint",
                "aggregation": "l2-normalize-mean-l2-normalize",
                "note": "v0.3 及以前：旧版均匀采样片段平均",
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        for row in conn.execute(
            "SELECT version FROM embedding_versions ORDER BY created_at, version"
        ).fetchall():
            version = str(row["version"])
            exists = conn.execute(
                "SELECT 1 FROM representations WHERE name = ? AND sequence_version = ?",
                (LEGACY_REPRESENTATION_NAME, version),
            ).fetchone()
            if exists is None:
                conn.execute(
                    """
                    INSERT INTO representations (
                        name, sequence_version, kind, dim, path_prefix, params_json,
                        status, note, created_at, updated_at
                    ) VALUES (?, ?, 'legacy', 0, NULL, ?, 'ready', ?, ?, ?)
                    """,
                    (
                        LEGACY_REPRESENTATION_NAME,
                        version,
                        params_json,
                        "v0.3 及以前的 5 片段平均表征；向量来自 embeddings 表 work_* 文件",
                        now,
                        now,
                    ),
                )
        conn.execute(
            """
            UPDATE map_runs SET representation_id = (
                SELECT r.id FROM representations r
                WHERE r.kind = 'legacy' AND r.sequence_version = map_runs.embedding_version
                LIMIT 1
            )
            WHERE representation_id IS NULL
            """
        )
        conn.execute(
            "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
            ("3",),
        )

    def _migrate_v3_to_v4(self, conn: sqlite3.Connection) -> None:
        """v3 -> v4：密集视觉序列表 + 表征/地图来源扩展（pe_av | visual）。

        - visual_versions / visual_sequences 由 SCHEMA_SQL 在迁移前创建；
        - representations：kind 扩展 visual，sequence_version 不再强制引用
          embedding_versions（视觉表征引用 visual_versions），新增 source_kind；
        - map_runs：embedding_version 不再强制引用 embedding_versions（视觉地图存
          视觉版本字符串，按 source_kind 解释），新增 source_kind；
          旧地图回填 source_kind='pe_av'。
        SQLite 不能直接修改 CHECK / 外键，只能重建表；initialize() 已在 BEGIN 之前
        关闭外键检查，这里复制数据后改名。
        """

        conn.execute(
            """
            CREATE TABLE representations_v4 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                sequence_version TEXT NOT NULL,
                kind TEXT NOT NULL
                    CHECK (kind IN ('legacy', 'full_mean', 'temporal_stats', 'transformer', 'visual')),
                source_kind TEXT NOT NULL DEFAULT 'pe_av'
                    CHECK (source_kind IN ('pe_av', 'visual')),
                dim INTEGER NOT NULL DEFAULT 0,
                path_prefix TEXT,
                params_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'building'
                    CHECK (status IN ('building', 'ready', 'failed')),
                note TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (name, sequence_version)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO representations_v4 (
                id, name, sequence_version, kind, source_kind, dim, path_prefix,
                params_json, status, note, created_at, updated_at
            )
            SELECT id, name, sequence_version, kind, 'pe_av', dim, path_prefix,
                   params_json, status, note, created_at, updated_at
            FROM representations
            """
        )
        conn.execute("DROP TABLE representations")
        conn.execute("ALTER TABLE representations_v4 RENAME TO representations")

        conn.execute(
            """
            CREATE TABLE map_runs_v4 (
                version TEXT PRIMARY KEY,
                embedding_version TEXT NOT NULL,
                representation_id INTEGER REFERENCES representations(id),
                source_kind TEXT NOT NULL DEFAULT 'pe_av'
                    CHECK (source_kind IN ('pe_av', 'visual')),
                params_json TEXT NOT NULL,
                filter_json TEXT,
                coords_path TEXT,
                model_2d_path TEXT,
                model_3d_path TEXT,
                n_works INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'building'
                    CHECK (status IN ('building', 'ready', 'failed')),
                note TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO map_runs_v4 (
                version, embedding_version, representation_id, source_kind, params_json,
                filter_json, coords_path, model_2d_path, model_3d_path, n_works,
                status, note, created_at
            )
            SELECT version, embedding_version, representation_id, 'pe_av', params_json,
                   filter_json, coords_path, model_2d_path, model_3d_path, n_works,
                   status, note, created_at
            FROM map_runs
            """
        )
        conn.execute("DROP TABLE map_runs")
        conn.execute("ALTER TABLE map_runs_v4 RENAME TO map_runs")
        conn.execute(
            "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
            (SCHEMA_VERSION,),
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

    def reopen(self, work_id: int, *, reason: str | None = None) -> str:
        """把已结束/中断的作品重置为 pending。

        用途：1) 模型或采样参数变化后需要为老作品生成新版本向量；
        2) 失败作品重试；3) 上次进程中断留下 downloading/encoding 残留。
        正常的状态迁移仍由 set_status 严格校验；此方法单独记录 reopen 事件。
        """

        now = now_utc_iso()
        with self.transaction() as conn:
            row = conn.execute("SELECT status FROM works WHERE id = ?", (work_id,)).fetchone()
            if row is None:
                raise DatabaseError(f"作品不存在: {work_id}")
            old = str(row["status"])
            if old == "pending":
                return old
            if old not in ("completed", "failed", "downloading", "encoding"):
                raise DatabaseError(f"不能 reopen 状态为 {old} 的作品 (work {work_id})")
            conn.execute(
                "UPDATE works SET status = 'pending', error = NULL, updated_at = ? WHERE id = ?",
                (now, work_id),
            )
            detail = f"{old}->pending" + (f" | {reason}" if reason else "")
            self._event(conn, work_id, "reopen", detail)
            return old

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
        clips: Sequence[tuple[Any, ...]],
    ) -> None:
        """替换某作品的片段计划。

        clips 的每个元素可以是 (idx, start_s, duration_s)（旧格式，valid/weight 记 NULL），
        也可以是 (idx, start_s, duration_s, valid_duration_s, weight)（v0.4 全片格式）。
        """

        normalized: list[tuple[int, str, int, float, float, float | None, float | None]] = []
        for item in clips:
            values = tuple(item)
            if len(values) == 3:
                idx, start, duration = values
                valid = None
                weight = None
            elif len(values) == 5:
                idx, start, duration, valid, weight = values
                valid = None if valid is None else float(valid)
                weight = None if weight is None else float(weight)
            else:
                raise DatabaseError(
                    f"片段格式非法（需要 3 或 5 元组）: {values!r}"
                )
            normalized.append(
                (work_id, version, int(idx), float(start), float(duration), valid, weight)
            )
        with self.transaction() as conn:
            conn.execute("DELETE FROM clips WHERE work_id = ? AND version = ?", (work_id, version))
            conn.executemany(
                "INSERT INTO clips(work_id, version, idx, start_s, duration_s, "
                "valid_duration_s, weight) VALUES (?, ?, ?, ?, ?, ?, ?)",
                normalized,
            )

    def get_clips(self, work_id: int, *, version: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM clips WHERE work_id = ? AND version = ? ORDER BY idx",
            (work_id, version),
        ).fetchall()
        return _as_dicts(rows)

    # ------------------------------------------------------------------ 全片序列状态
    def upsert_sequence_state(
        self,
        work_id: int,
        *,
        version: str,
        duration_s: float,
        n_clips: int,
        n_clips_done: int = 0,
        partial_relpath: str | None = None,
        status: str = "in_progress",
    ) -> None:
        """登记/更新一部作品的全片编码进度（片段级恢复用）。"""

        if status not in SEQUENCE_STATUSES:
            raise DatabaseError(f"未知序列状态: {status}")
        if int(n_clips) < 0:
            raise DatabaseError(f"片段总数非法: {n_clips}")
        if not 0 <= int(n_clips_done) <= int(n_clips):
            raise DatabaseError(f"已完成片段数非法: {n_clips_done} / {n_clips}")
        now = now_utc_iso()
        with self.transaction() as conn:
            if conn.execute("SELECT 1 FROM works WHERE id = ?", (work_id,)).fetchone() is None:
                raise DatabaseError(f"作品不存在: {work_id}")
            conn.execute(
                """
                INSERT INTO sequence_states (
                    work_id, version, duration_s, n_clips, n_clips_done,
                    partial_relpath, status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(work_id, version) DO UPDATE SET
                    duration_s = excluded.duration_s,
                    n_clips = excluded.n_clips,
                    n_clips_done = excluded.n_clips_done,
                    partial_relpath = excluded.partial_relpath,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (
                    work_id,
                    version,
                    float(duration_s),
                    int(n_clips),
                    int(n_clips_done),
                    partial_relpath,
                    status,
                    now,
                ),
            )

    def get_sequence_state(self, work_id: int, *, version: str) -> dict[str, Any] | None:
        return _as_dict(
            self.conn.execute(
                "SELECT * FROM sequence_states WHERE work_id = ? AND version = ?",
                (work_id, version),
            ).fetchone()
        )

    def list_sequence_states(
        self, *, version: str | None = None, status: str | None = None
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM sequence_states WHERE 1 = 1"
        params: list[Any] = []
        if version is not None:
            sql += " AND version = ?"
            params.append(version)
        if status is not None:
            if status not in SEQUENCE_STATUSES:
                raise DatabaseError(f"未知序列状态: {status}")
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY work_id"
        return _as_dicts(self.conn.execute(sql, params).fetchall())

    def delete_sequence_state(self, work_id: int, *, version: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "DELETE FROM sequence_states WHERE work_id = ? AND version = ?",
                (work_id, version),
            )

    # ------------------------------------------------------------------ 表征层
    def register_representation(
        self,
        name: str,
        *,
        sequence_version: str,
        kind: str,
        params: Mapping[str, Any],
        source_kind: str = "pe_av",
        path_prefix: str | None = None,
        note: str | None = None,
    ) -> int:
        """登记表征版本并返回 id；同名 + 同序列已存在时要求 kind/参数一致。

        source_kind 表示序列版本来自哪一层：
        - pe_av：PE-AV sequence version（embedding_versions 表）；
        - visual：密集视觉序列版本（visual_versions 表）。
        """

        display = str(name or "").strip()
        if not display:
            raise DatabaseError("表征名不能为空")
        if kind not in REPRESENTATION_KINDS:
            raise DatabaseError(f"未知表征类型: {kind}")
        if source_kind not in REPRESENTATION_SOURCE_KINDS:
            raise DatabaseError(f"未知表征来源: {source_kind}")
        payload = dict(params or {})
        params_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        now = now_utc_iso()
        with self.transaction() as conn:
            version_table = (
                "embedding_versions" if source_kind == "pe_av" else "visual_versions"
            )
            if (
                conn.execute(
                    f"SELECT 1 FROM {version_table} WHERE version = ?", (sequence_version,)
                ).fetchone()
                is None
            ):
                raise DatabaseError(f"序列版本未登记: {sequence_version}")
            row = conn.execute(
                "SELECT * FROM representations WHERE name = ? AND sequence_version = ?",
                (display, sequence_version),
            ).fetchone()
            if row is not None:
                if (
                    row["kind"] != kind
                    or row["source_kind"] != source_kind
                    or row["params_json"] != params_json
                ):
                    raise DatabaseError(
                        f"表征 {display}@{sequence_version} 已存在但类型/参数不同；"
                        "请使用新表征名，或构建工具 --force 重建"
                    )
                return int(row["id"])
            cursor = conn.execute(
                """
                INSERT INTO representations (
                    name, sequence_version, kind, source_kind, dim, path_prefix,
                    params_json, status, note, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 0, ?, ?, 'building', ?, ?, ?)
                """,
                (
                    display,
                    sequence_version,
                    kind,
                    source_kind,
                    path_prefix,
                    params_json,
                    note,
                    now,
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def get_representation(
        self,
        representation_id: int | None = None,
        *,
        name: str | None = None,
        sequence_version: str | None = None,
    ) -> dict[str, Any] | None:
        if representation_id is not None:
            return _as_dict(
                self.conn.execute(
                    "SELECT * FROM representations WHERE id = ?", (int(representation_id),)
                ).fetchone()
            )
        if name is None:
            raise DatabaseError("get_representation 需要 representation_id 或 name")
        if sequence_version is not None:
            return _as_dict(
                self.conn.execute(
                    "SELECT * FROM representations WHERE name = ? AND sequence_version = ?",
                    (name, sequence_version),
                ).fetchone()
            )
        return _as_dict(
            self.conn.execute(
                "SELECT * FROM representations WHERE name = ? "
                "ORDER BY created_at DESC, id DESC LIMIT 1",
                (name,),
            ).fetchone()
        )

    def list_representations(
        self,
        *,
        name: str | None = None,
        kind: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM representations WHERE 1 = 1"
        params: list[Any] = []
        if name is not None:
            sql += " AND name = ?"
            params.append(name)
        if kind is not None:
            if kind not in REPRESENTATION_KINDS:
                raise DatabaseError(f"未知表征类型: {kind}")
            sql += " AND kind = ?"
            params.append(kind)
        if status is not None:
            if status not in REPRESENTATION_STATUSES:
                raise DatabaseError(f"未知表征状态: {status}")
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY created_at, id"
        return _as_dicts(self.conn.execute(sql, params).fetchall())

    def update_representation_params(
        self,
        representation_id: int,
        params: Mapping[str, Any],
        *,
        note: str | None = None,
    ) -> None:
        """更新表征参数（用于 --force 重建；正常流程应新建表征名）。"""

        params_json = json.dumps(dict(params or {}), sort_keys=True, ensure_ascii=False)
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE representations SET params_json = ?, note = COALESCE(?, note), "
                "updated_at = ? WHERE id = ?",
                (params_json, note, now_utc_iso(), int(representation_id)),
            )
            if cursor.rowcount == 0:
                raise DatabaseError(f"表征不存在: {representation_id}")

    def set_representation_status(
        self,
        representation_id: int,
        status: str,
        *,
        dim: int | None = None,
        note: str | None = None,
    ) -> None:
        if status not in REPRESENTATION_STATUSES:
            raise DatabaseError(f"未知表征状态: {status}")
        with self.transaction() as conn:
            if dim is None:
                cursor = conn.execute(
                    "UPDATE representations SET status = ?, note = COALESCE(?, note), "
                    "updated_at = ? WHERE id = ?",
                    (status, note, now_utc_iso(), int(representation_id)),
                )
            else:
                cursor = conn.execute(
                    "UPDATE representations SET status = ?, dim = ?, note = COALESCE(?, note), "
                    "updated_at = ? WHERE id = ?",
                    (status, int(dim), note, now_utc_iso(), int(representation_id)),
                )
            if cursor.rowcount == 0:
                raise DatabaseError(f"表征不存在: {representation_id}")

    def ensure_legacy_representation(self, sequence_version: str) -> int:
        """确保某个旧序列版本有对应的 legacy 表征登记，返回表征 id。"""

        row = self.get_representation(
            name=LEGACY_REPRESENTATION_NAME, sequence_version=sequence_version
        )
        if row is not None:
            return int(row["id"])
        if self.get_embedding_version(sequence_version) is None:
            raise DatabaseError(f"序列版本未登记: {sequence_version}")
        params = {
            "source": "embeddings",
            "embedding_kind": "work_joint",
            "aggregation": "l2-normalize-mean-l2-normalize",
        }
        return self.register_representation(
            LEGACY_REPRESENTATION_NAME,
            sequence_version=sequence_version,
            kind="legacy",
            params=params,
            note="旧版均匀采样片段平均表征",
        )

    def clear_representation_vectors(self, representation_id: int) -> int:
        with self.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM representation_vectors WHERE representation_id = ?",
                (int(representation_id),),
            )
            return int(cursor.rowcount)

    def register_representation_vector(
        self,
        representation_id: int,
        work_id: int,
        *,
        kind: str,
        path: str,
        dim: int,
    ) -> None:
        if kind not in REPRESENTATION_VECTOR_KINDS:
            raise DatabaseError(f"未知表征向量种类: {kind}")
        if dim < 1:
            raise DatabaseError(f"向量维度非法: {dim}")
        if not path:
            raise DatabaseError("向量路径不能为空")
        with self.transaction() as conn:
            if (
                conn.execute(
                    "SELECT 1 FROM representations WHERE id = ?", (int(representation_id),)
                ).fetchone()
                is None
            ):
                raise DatabaseError(f"表征不存在: {representation_id}")
            if (
                conn.execute("SELECT 1 FROM works WHERE id = ?", (int(work_id),)).fetchone()
                is None
            ):
                raise DatabaseError(f"作品不存在: {work_id}")
            conn.execute(
                """
                INSERT INTO representation_vectors (
                    representation_id, work_id, kind, path, dim, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(representation_id, work_id, kind) DO UPDATE SET
                    path = excluded.path,
                    dim = excluded.dim,
                    created_at = excluded.created_at
                """,
                (int(representation_id), int(work_id), kind, path, int(dim), now_utc_iso()),
            )

    def list_representation_vectors(
        self, representation_id: int, *, kind: str | None = None
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM representation_vectors WHERE representation_id = ?"
        params: list[Any] = [int(representation_id)]
        if kind is not None:
            if kind not in REPRESENTATION_VECTOR_KINDS:
                raise DatabaseError(f"未知表征向量种类: {kind}")
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY work_id"
        return _as_dicts(self.conn.execute(sql, params).fetchall())

    def list_works_with_representation(
        self, representation_id: int, *, kind: str = "joint"
    ) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT w.*, v.path AS representation_path, v.dim AS representation_dim
            FROM works w
            JOIN representation_vectors v ON v.work_id = w.id
            WHERE v.representation_id = ? AND v.kind = ?
            ORDER BY w.id
            """,
            (int(representation_id), kind),
        ).fetchall()
        return _as_dicts(rows)

    # ------------------------------------------------------------------ 密集视觉序列（v0.4.3）
    def register_visual_version(
        self,
        version: str,
        *,
        model_id: str,
        revision: str,
        preprocessing: str,
        fps: float,
        size: int,
        dim: int | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        """登记一个冻结视觉编码器版本；同版本必须参数一致（防静默覆盖）。"""

        display = str(version or "").strip()
        if not display:
            raise DatabaseError("视觉序列版本号不能为空")
        model = str(model_id or "").strip()
        rev = str(revision or "").strip()
        if not model or not rev:
            raise DatabaseError("视觉序列版本必须固定 model_id 与 revision")
        if preprocessing not in ("stretch", "letterbox"):
            raise DatabaseError(f"未知预处理方式: {preprocessing}")
        fps_value = float(fps)
        size_value = int(size)
        if not (fps_value > 0):
            raise DatabaseError(f"帧率非法: {fps}")
        if size_value < 2:
            raise DatabaseError(f"输入尺寸非法: {size}")
        dim_value = None if dim is None else int(dim)
        if dim_value is not None and dim_value < 1:
            raise DatabaseError(f"向量维度非法: {dim}")
        params_json = json.dumps(dict(params or {}), sort_keys=True, ensure_ascii=False)
        row_values = (model, rev, preprocessing, fps_value, size_value, dim_value, params_json)
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM visual_versions WHERE version = ?", (display,)
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO visual_versions (
                        version, model_id, revision, preprocessing, fps, size,
                        dim, params_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (display, *row_values, now_utc_iso()),
                )
                return
            actual = (
                row["model_id"],
                row["revision"],
                row["preprocessing"],
                float(row["fps"]),
                int(row["size"]),
                None if row["dim"] is None else int(row["dim"]),
                row["params_json"],
            )
            if actual != row_values:
                raise DatabaseError(
                    f"视觉序列版本 {display} 已存在但参数不同；请使用新版本号而不是覆盖"
                )

    def get_visual_version(self, version: str) -> dict[str, Any] | None:
        return _as_dict(
            self.conn.execute(
                "SELECT * FROM visual_versions WHERE version = ?", (str(version),)
            ).fetchone()
        )

    def list_visual_versions(self) -> list[dict[str, Any]]:
        return _as_dicts(
            self.conn.execute(
                "SELECT * FROM visual_versions ORDER BY created_at, version"
            ).fetchall()
        )

    def upsert_visual_sequence(
        self,
        work_id: int,
        *,
        version: str,
        status: str = "pending",
        frames_path: str | None = None,
        windows_path: str | None = None,
        n_frames: int | None = None,
        n_windows: int | None = None,
        dim: int | None = None,
        error: str | None = None,
        decode_s: float = 0.0,
        encode_s: float = 0.0,
        model_load_s: float = 0.0,
        total_s: float = 0.0,
    ) -> None:
        """登记/更新一部作品在某个视觉序列版本下的处理状态（断点续跑依据）。"""

        if status not in VISUAL_SEQUENCE_STATUSES:
            raise DatabaseError(f"未知视觉序列状态: {status}")
        for name, value in (
            ("n_frames", n_frames),
            ("n_windows", n_windows),
            ("decode_s", decode_s),
            ("encode_s", encode_s),
            ("model_load_s", model_load_s),
            ("total_s", total_s),
        ):
            if value is not None and float(value) < 0:
                raise DatabaseError(f"{name} 不能为负: {value}")
        if dim is not None and int(dim) < 1:
            raise DatabaseError(f"dim 非法: {dim}")
        now = now_utc_iso()
        with self.transaction() as conn:
            if conn.execute("SELECT 1 FROM works WHERE id = ?", (int(work_id),)).fetchone() is None:
                raise DatabaseError(f"作品不存在: {work_id}")
            if conn.execute(
                "SELECT 1 FROM visual_versions WHERE version = ?", (str(version),)
            ).fetchone() is None:
                raise DatabaseError(f"视觉序列版本未登记: {version}")
            conn.execute(
                """
                INSERT INTO visual_sequences (
                    work_id, version, status, frames_path, windows_path, n_frames,
                    n_windows, dim, error, decode_s, encode_s, model_load_s, total_s,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(work_id, version) DO UPDATE SET
                    status = excluded.status,
                    frames_path = excluded.frames_path,
                    windows_path = excluded.windows_path,
                    n_frames = excluded.n_frames,
                    n_windows = excluded.n_windows,
                    dim = excluded.dim,
                    error = excluded.error,
                    decode_s = excluded.decode_s,
                    encode_s = excluded.encode_s,
                    model_load_s = excluded.model_load_s,
                    total_s = excluded.total_s,
                    updated_at = excluded.updated_at
                """,
                (
                    int(work_id),
                    str(version),
                    status,
                    frames_path,
                    windows_path,
                    None if n_frames is None else int(n_frames),
                    None if n_windows is None else int(n_windows),
                    None if dim is None else int(dim),
                    error,
                    float(decode_s),
                    float(encode_s),
                    float(model_load_s),
                    float(total_s),
                    now,
                    now,
                ),
            )

    def get_visual_sequence(self, work_id: int, *, version: str) -> dict[str, Any] | None:
        return _as_dict(
            self.conn.execute(
                "SELECT * FROM visual_sequences WHERE work_id = ? AND version = ?",
                (int(work_id), str(version)),
            ).fetchone()
        )

    def list_visual_sequences(
        self,
        *,
        version: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM visual_sequences WHERE 1 = 1"
        params: list[Any] = []
        if version is not None:
            sql += " AND version = ?"
            params.append(str(version))
        if status is not None:
            if status not in VISUAL_SEQUENCE_STATUSES:
                raise DatabaseError(f"未知视觉序列状态: {status}")
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY work_id"
        return _as_dicts(self.conn.execute(sql, params).fetchall())

    def visual_status_summary(self, version: str) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM visual_sequences WHERE version = ? "
            "GROUP BY status",
            (str(version),),
        ).fetchall()
        summary = {status: 0 for status in VISUAL_SEQUENCE_STATUSES}
        for row in rows:
            summary[str(row["status"])] = int(row["n"])
        return summary

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
        filter_spec: Mapping[str, Any] | None = None,
        representation_id: int | None = None,
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
            source_kind = "pe_av"
            if representation_id is not None:
                rep = conn.execute(
                    "SELECT * FROM representations WHERE id = ?", (int(representation_id),)
                ).fetchone()
                if rep is None:
                    raise DatabaseError(f"表征不存在: {representation_id}")
                source_kind = str(rep["source_kind"])
                if str(rep["sequence_version"]) != str(embedding_version):
                    raise DatabaseError(
                        f"地图的向量版本与表征序列不一致: {embedding_version} != "
                        f"{rep['sequence_version']}"
                    )
            version_table = (
                "embedding_versions" if source_kind == "pe_av" else "visual_versions"
            )
            if (
                conn.execute(
                    f"SELECT 1 FROM {version_table} WHERE version = ?", (embedding_version,)
                ).fetchone()
                is None
            ):
                raise DatabaseError(f"序列版本未登记: {embedding_version}")
            conn.execute(
                """
                INSERT INTO map_runs (
                    version, embedding_version, representation_id, source_kind,
                    params_json, filter_json, coords_path, model_2d_path, model_3d_path,
                    n_works, status, note, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?, ?)
                """,
                (
                    version,
                    embedding_version,
                    None if representation_id is None else int(representation_id),
                    source_kind,
                    json.dumps(dict(params), sort_keys=True, ensure_ascii=False),
                    (
                        json.dumps(dict(filter_spec), sort_keys=True, ensure_ascii=False)
                        if filter_spec is not None
                        else None
                    ),
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

    def latest_map(
        self, *, ready_only: bool = True, representation_id: int | None = None
    ) -> dict[str, Any] | None:
        clauses: list[str] = []
        params: list[Any] = []
        if ready_only:
            clauses.append("status = 'ready'")
        if representation_id is not None:
            clauses.append("representation_id = ?")
            params.append(int(representation_id))
        sql = "SELECT * FROM map_runs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT 1"
        return _as_dict(self.conn.execute(sql, params).fetchone())

    def list_maps(self, *, representation_id: int | None = None) -> list[dict[str, Any]]:
        if representation_id is None:
            return _as_dicts(
                self.conn.execute("SELECT * FROM map_runs ORDER BY created_at, rowid").fetchall()
            )
        return _as_dicts(
            self.conn.execute(
                "SELECT * FROM map_runs WHERE representation_id = ? "
                "ORDER BY created_at, rowid",
                (int(representation_id),),
            ).fetchall()
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

    # ------------------------------------------------------------------ 榜单作品查询
    def list_works_for_list(
        self, list_key: str, *, only_canonical: bool = True
    ) -> list[dict[str, Any]]:
        """按榜单出现记录返回作品（按排名、票数、work_id 排序）。

        only_canonical=True 时排除 work_links 里的 duplicate 实体。
        """

        sql = """
            SELECT w.*, a.rank AS list_rank, a.votes AS list_votes,
                   a.note AS list_note, l.list_key AS list_key,
                   l.name AS list_name, l.vote_year AS list_vote_year
            FROM works w
            JOIN appearances a ON a.work_id = w.id
            JOIN lists l ON l.id = a.list_id
            WHERE l.list_key = ?
        """
        if only_canonical:
            sql += " AND w.id NOT IN (SELECT duplicate_work_id FROM work_links)"
        sql += " ORDER BY COALESCE(a.rank, 999999), COALESCE(a.votes, -1) DESC, w.id"
        return _as_dicts(self.conn.execute(sql, (str(list_key),)).fetchall())

    def work_ids_for_list(self, list_key: str, *, only_canonical: bool = True) -> set[int]:
        sql = """
            SELECT DISTINCT w.id AS work_id
            FROM works w
            JOIN appearances a ON a.work_id = w.id
            JOIN lists l ON l.id = a.list_id
            WHERE l.list_key = ?
        """
        if only_canonical:
            sql += " AND w.id NOT IN (SELECT duplicate_work_id FROM work_links)"
        return {
            int(row["work_id"]) for row in self.conn.execute(sql, (str(list_key),)).fetchall()
        }

    def work_ids_for_vote_year(self, vote_year: int, *, only_canonical: bool = True) -> set[int]:
        sql = """
            SELECT DISTINCT w.id AS work_id
            FROM works w
            JOIN appearances a ON a.work_id = w.id
            JOIN lists l ON l.id = a.list_id
            WHERE l.vote_year = ?
        """
        if only_canonical:
            sql += " AND w.id NOT IN (SELECT duplicate_work_id FROM work_links)"
        return {
            int(row["work_id"])
            for row in self.conn.execute(sql, (int(vote_year),)).fetchall()
        }

    # ------------------------------------------------------------------ 身份关联
    def link_works(
        self,
        canonical_work_id: int,
        duplicate_work_id: int,
        *,
        method: str,
        confidence: float | None = None,
        note: str | None = None,
    ) -> int:
        """把 duplicate 关联到 canonical（同一作品的双重身份）。

        canonical 是保留在星云/展示中的实体；duplicate 默认不进入星云。
        每个 duplicate 只能属于一个 canonical；canonical 自身不能是 duplicate。
        返回 link id。
        """

        canonical = int(canonical_work_id)
        duplicate = int(duplicate_work_id)
        if canonical == duplicate:
            raise DatabaseError("canonical 与 duplicate 不能是同一作品")
        if not method or not str(method).strip():
            raise DatabaseError("身份关联必须说明 method")
        if confidence is not None and not 0.0 <= float(confidence) <= 1.0:
            raise DatabaseError("confidence 必须在 0..1")
        now = now_utc_iso()
        with self.transaction() as conn:
            for work_id in (canonical, duplicate):
                if conn.execute("SELECT 1 FROM works WHERE id = ?", (work_id,)).fetchone() is None:
                    raise DatabaseError(f"作品不存在: {work_id}")
            if conn.execute(
                "SELECT 1 FROM work_links WHERE duplicate_work_id = ?", (canonical,)
            ).fetchone():
                raise DatabaseError(f"canonical 作品本身已被关联为重复实体: {canonical}")
            if conn.execute(
                "SELECT 1 FROM work_links WHERE duplicate_work_id = ?", (duplicate,)
            ).fetchone():
                raise DatabaseError(f"作品已有身份关联: {duplicate}")
            cur = conn.execute(
                """
                INSERT INTO work_links (
                    canonical_work_id, duplicate_work_id, method, confidence, note, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (canonical, duplicate, str(method), confidence, note, now),
            )
            self._event(
                conn,
                duplicate,
                "linked",
                json.dumps(
                    {"canonical_work_id": canonical, "method": method},
                    ensure_ascii=False,
                ),
            )
            return int(cur.lastrowid)

    def unlink_work(self, duplicate_work_id: int, *, reason: str | None = None) -> str:
        """解除一条身份关联（保留两个作品），返回原 canonical id。"""

        with self.transaction() as conn:
            row = conn.execute(
                "SELECT canonical_work_id FROM work_links WHERE duplicate_work_id = ?",
                (int(duplicate_work_id),),
            ).fetchone()
            if row is None:
                raise DatabaseError(f"作品没有身份关联: {duplicate_work_id}")
            canonical = int(row["canonical_work_id"])
            conn.execute(
                "DELETE FROM work_links WHERE duplicate_work_id = ?", (int(duplicate_work_id),)
            )
            self._event(
                conn,
                int(duplicate_work_id),
                "unlinked",
                f"canonical={canonical}" + (f" | {reason}" if reason else ""),
            )
            return canonical

    def get_work_links(self) -> list[dict[str, Any]]:
        return _as_dicts(
            self.conn.execute("SELECT * FROM work_links ORDER BY id").fetchall()
        )

    def get_work_link(self, work_id: int) -> dict[str, Any] | None:
        """返回 work_id 参与的身份关联（canonical 或 duplicate 均适用）。"""

        return _as_dict(self.conn.execute(
            """
            SELECT * FROM work_links
            WHERE canonical_work_id = ? OR duplicate_work_id = ?
            ORDER BY id LIMIT 1
            """,
            (int(work_id), int(work_id)),
        ).fetchone())

    def get_canonical_work_id(self, work_id: int) -> int:
        """若 work_id 是 duplicate，返回其 canonical id；否则原样返回。"""

        row = self.conn.execute(
            "SELECT canonical_work_id FROM work_links WHERE duplicate_work_id = ?",
            (int(work_id),),
        ).fetchone()
        return int(row["canonical_work_id"]) if row is not None else int(work_id)

    def get_duplicate_work_ids(self) -> set[int]:
        rows = self.conn.execute("SELECT duplicate_work_id FROM work_links").fetchall()
        return {int(row["duplicate_work_id"]) for row in rows}

    def get_linked_duplicates(self, canonical_work_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT w.*, l.method AS link_method, l.confidence AS link_confidence,
                   l.note AS link_note
            FROM work_links l
            JOIN works w ON w.id = l.duplicate_work_id
            WHERE l.canonical_work_id = ?
            ORDER BY l.id
            """,
            (int(canonical_work_id),),
        ).fetchall()
        return _as_dicts(rows)

    # ------------------------------------------------------------------ 流水线批次与耗时
    def start_pipeline_run(
        self,
        *,
        embedding_version: str | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> int:
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO pipeline_runs (
                    started_at, embedding_version, params_json
                ) VALUES (?, ?, ?)
                """,
                (
                    now_utc_iso(),
                    embedding_version,
                    json.dumps(dict(params), sort_keys=True, ensure_ascii=False)
                    if params is not None
                    else None,
                ),
            )
            return int(cur.lastrowid)

    def finish_pipeline_run(
        self,
        run_id: int,
        *,
        n_selected: int,
        n_completed: int,
        n_failed: int,
        total_s: float,
        model_load_s: float | None = None,
    ) -> None:
        with self.transaction() as conn:
            cur = conn.execute(
                """
                UPDATE pipeline_runs SET
                    finished_at = ?, n_selected = ?, n_completed = ?, n_failed = ?,
                    total_s = ?, model_load_s = ?
                WHERE id = ?
                """,
                (
                    now_utc_iso(),
                    int(n_selected),
                    int(n_completed),
                    int(n_failed),
                    float(total_s),
                    None if model_load_s is None else float(model_load_s),
                    int(run_id),
                ),
            )
            if cur.rowcount == 0:
                raise DatabaseError(f"流水线批次不存在: {run_id}")

    def record_work_timing(
        self,
        run_id: int | None,
        work_id: int,
        *,
        embedding_version: str,
        download_s: float = 0.0,
        probe_s: float = 0.0,
        extract_s: float = 0.0,
        preprocess_s: float = 0.0,
        forward_s: float = 0.0,
        save_s: float = 0.0,
        events_s: float = 0.0,
        total_s: float = 0.0,
    ) -> None:
        for name, value in (
            ("download_s", download_s),
            ("probe_s", probe_s),
            ("extract_s", extract_s),
            ("preprocess_s", preprocess_s),
            ("forward_s", forward_s),
            ("save_s", save_s),
            ("events_s", events_s),
            ("total_s", total_s),
        ):
            if not (float(value) >= 0.0):
                raise DatabaseError(f"耗时不能为负: {name}={value}")
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO work_timings (
                    run_id, work_id, embedding_version, download_s, probe_s, extract_s,
                    preprocess_s, forward_s, save_s, events_s, total_s, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, work_id) DO UPDATE SET
                    embedding_version = excluded.embedding_version,
                    download_s = excluded.download_s,
                    probe_s = excluded.probe_s,
                    extract_s = excluded.extract_s,
                    preprocess_s = excluded.preprocess_s,
                    forward_s = excluded.forward_s,
                    save_s = excluded.save_s,
                    events_s = excluded.events_s,
                    total_s = excluded.total_s,
                    created_at = excluded.created_at
                """,
                (
                    None if run_id is None else int(run_id),
                    int(work_id),
                    str(embedding_version),
                    float(download_s),
                    float(probe_s),
                    float(extract_s),
                    float(preprocess_s),
                    float(forward_s),
                    float(save_s),
                    float(events_s),
                    float(total_s),
                    now_utc_iso(),
                ),
            )

    def list_work_timings(
        self,
        *,
        run_id: int | None = None,
        embedding_version: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM work_timings"
        clauses: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(int(run_id))
        if embedding_version is not None:
            clauses.append("embedding_version = ?")
            params.append(str(embedding_version))
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY work_id"
        return _as_dicts(self.conn.execute(sql, params).fetchall())

    def list_pipeline_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        if limit < 1:
            raise DatabaseError("limit 必须 >= 1")
        return _as_dicts(
            self.conn.execute(
                "SELECT * FROM pipeline_runs ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
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
            link_rows = self.conn.execute("SELECT COUNT(*) AS n FROM work_links").fetchone()
            run_rows = self.conn.execute("SELECT COUNT(*) AS n FROM pipeline_runs").fetchone()
            timing_rows = self.conn.execute("SELECT COUNT(*) AS n FROM work_timings").fetchone()
            representation_rows = self.conn.execute(
                "SELECT COUNT(*) AS n FROM representations"
            ).fetchone()
            sequence_rows = self.conn.execute(
                "SELECT COUNT(*) AS n FROM sequence_states"
            ).fetchone()
            visual_version_rows = self.conn.execute(
                "SELECT COUNT(*) AS n FROM visual_versions"
            ).fetchone()
            visual_sequence_rows = self.conn.execute(
                "SELECT COUNT(*) AS n FROM visual_sequences"
            ).fetchone()
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
            "links": int(link_rows["n"]) if link_rows else 0,
            "pipeline_runs": int(run_rows["n"]) if run_rows else 0,
            "timings": int(timing_rows["n"]) if timing_rows else 0,
            "representations": int(representation_rows["n"]) if representation_rows else 0,
            "sequences": int(sequence_rows["n"]) if sequence_rows else 0,
            "visual_versions": int(visual_version_rows["n"]) if visual_version_rows else 0,
            "visual_sequences": int(visual_sequence_rows["n"]) if visual_sequence_rows else 0,
        }
