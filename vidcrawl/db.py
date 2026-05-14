import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from vidcrawl.models import Duplicate, Evidence, Idea, IngestionRun, Keyframe, Moment, Video

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS videos (
    video_id        TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    source          TEXT NOT NULL CHECK(source IN ('youtube', 'local')),
    url             TEXT,
    duration_sec    REAL NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending', 'ingesting', 'ready', 'error')),
    transcript_path TEXT,
    error_message   TEXT,
    metadata        TEXT DEFAULT '{}',
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS moments (
    moment_id        TEXT PRIMARY KEY,
    video_id         TEXT NOT NULL REFERENCES videos(video_id),
    start_sec        REAL NOT NULL,
    end_sec          REAL NOT NULL,
    transcript_text  TEXT NOT NULL DEFAULT '',
    ocr_text         TEXT DEFAULT '',
    ideas            TEXT DEFAULT '[]',
    keyframe_paths   TEXT DEFAULT '[]',
    content_hash     TEXT,
    parent_moment_id TEXT,
    embedding        BLOB,
    metadata         TEXT DEFAULT '{}',
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS modal_evidence (
    evidence_id TEXT PRIMARY KEY,
    moment_id   TEXT NOT NULL REFERENCES moments(moment_id),
    modality    TEXT NOT NULL,
    content     TEXT NOT NULL,
    confidence  REAL DEFAULT 1.0,
    source      TEXT,
    metadata    TEXT DEFAULT '{}',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ideas (
    idea_id     TEXT PRIMARY KEY,
    moment_id   TEXT NOT NULL REFERENCES moments(moment_id),
    type        TEXT NOT NULL
                CHECK(type IN ('claim','step','definition','example','warning','workflow','comparison')),
    text        TEXT NOT NULL,
    confidence  REAL DEFAULT 0.7,
    source      TEXT DEFAULT 'rule' CHECK(source IN ('rule', 'llm')),
    embedding   BLOB
);

CREATE TABLE IF NOT EXISTS keyframes (
    keyframe_id  TEXT PRIMARY KEY,
    moment_id    TEXT REFERENCES moments(moment_id),
    video_id     TEXT NOT NULL REFERENCES videos(video_id),
    timestamp_sec REAL NOT NULL,
    file_path    TEXT NOT NULL,
    width        INTEGER,
    height       INTEGER,
    ocr_text     TEXT,
    metadata     TEXT DEFAULT '{}',
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS duplicates (
    dup_id              TEXT PRIMARY KEY,
    moment_id           TEXT NOT NULL REFERENCES moments(moment_id),
    canonical_moment_id TEXT NOT NULL REFERENCES moments(moment_id),
    similarity_score    REAL DEFAULT 1.0,
    novelty_score       REAL DEFAULT 0.0,
    method              TEXT DEFAULT 'exact_hash',
    duplicate_type      TEXT DEFAULT 'exact',
    item_type           TEXT DEFAULT 'moment',
    reason              TEXT DEFAULT '',
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id       TEXT PRIMARY KEY,
    video_id     TEXT NOT NULL REFERENCES videos(video_id),
    status       TEXT NOT NULL DEFAULT 'running'
                 CHECK(status IN ('running', 'completed', 'failed')),
    pipeline_steps TEXT DEFAULT '[]',
    error_message TEXT,
    started_at   TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS moments_fts USING fts5(
    moment_id UNINDEXED,
    transcript_text,
    ocr_text,
    ideas_text,
    video_title,
    video_description
);

CREATE INDEX IF NOT EXISTS idx_moments_video_id ON moments(video_id);
CREATE INDEX IF NOT EXISTS idx_moments_content_hash ON moments(content_hash);
CREATE INDEX IF NOT EXISTS idx_ideas_moment_id ON ideas(moment_id);
CREATE INDEX IF NOT EXISTS idx_ideas_type ON ideas(type);
CREATE INDEX IF NOT EXISTS idx_evidence_moment_id ON modal_evidence(moment_id);
CREATE INDEX IF NOT EXISTS idx_keyframes_video_id ON keyframes(video_id);
CREATE INDEX IF NOT EXISTS idx_ingestion_runs_video_id ON ingestion_runs(video_id);
"""


def get_db(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    _migrate_duplicates_schema(conn)
    _init_graph_tables(conn)
    _maybe_rebuild_fts(conn)


def _maybe_rebuild_fts(conn: sqlite3.Connection) -> None:
    """Rebuild FTS if moments exist but FTS index is empty (e.g. after a fresh schema deploy)."""
    fts_count = conn.execute("SELECT COUNT(*) FROM moments_fts").fetchone()[0]
    if fts_count == 0:
        moment_count = conn.execute("SELECT COUNT(*) FROM moments").fetchone()[0]
        if moment_count > 0:
            rebuild_fts(conn)


def _init_graph_tables(conn: sqlite3.Connection) -> None:
    from vidcrawl.graph.build import GRAPH_SCHEMA_SQL
    conn.executescript(GRAPH_SCHEMA_SQL)


def _migrate_duplicates_schema(conn: sqlite3.Connection) -> None:
    _try_add_column(conn, "duplicates", "novelty_score", "REAL DEFAULT 0.0")
    _try_add_column(conn, "duplicates", "duplicate_type", "TEXT DEFAULT 'exact'")
    _try_add_column(conn, "duplicates", "item_type", "TEXT DEFAULT 'moment'")
    _try_add_column(conn, "duplicates", "reason", "TEXT DEFAULT ''")


def _try_add_column(conn, table: str, column: str, col_type: str) -> None:
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
    except sqlite3.OperationalError:
        pass


# ---- Video CRUD ----

def insert_video(conn: sqlite3.Connection, video: Video) -> None:
    conn.execute(
        """INSERT INTO videos
           (video_id, title, source, url, duration_sec, status,
            transcript_path, error_message, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (video.video_id, video.title, video.source, video.url,
         video.duration_sec, video.status, video.transcript_path,
         video.error_message, json.dumps(video.metadata)),
    )


def update_video_status(
    conn: sqlite3.Connection,
    video_id: str,
    status: str,
    error_message: str | None = None,
) -> None:
    conn.execute(
        "UPDATE videos SET status = ?, error_message = ? WHERE video_id = ?",
        (status, error_message, video_id),
    )


def get_video(conn: sqlite3.Connection, video_id: str) -> Video | None:
    row = conn.execute(
        "SELECT * FROM videos WHERE video_id = ?", (video_id,)
    ).fetchone()
    if row is None:
        return None
    return _row_to_video(row)


def list_videos(conn: sqlite3.Connection) -> list[Video]:
    rows = conn.execute(
        "SELECT * FROM videos ORDER BY created_at DESC"
    ).fetchall()
    return [_row_to_video(r) for r in rows]


def _row_to_video(row: sqlite3.Row) -> Video:
    return Video(
        video_id=row["video_id"],
        title=row["title"],
        source=row["source"],
        url=row["url"],
        duration_sec=row["duration_sec"],
        status=row["status"],
        transcript_path=row["transcript_path"],
        error_message=row["error_message"],
        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        created_at=row["created_at"],
    )


# ---- Moment CRUD ----

def insert_moment(conn: sqlite3.Connection, moment: Moment) -> None:
    conn.execute(
        """INSERT INTO moments
           (moment_id, video_id, start_sec, end_sec, transcript_text,
            ocr_text, ideas, keyframe_paths, content_hash,
            parent_moment_id, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (moment.moment_id, moment.video_id, moment.start_sec,
         moment.end_sec, moment.transcript_text, moment.ocr_text,
         json.dumps([i.model_dump() for i in moment.ideas]),
         json.dumps(moment.keyframe_paths), moment.content_hash,
         moment.parent_moment_id, json.dumps(moment.metadata)),
    )


def get_moment(conn: sqlite3.Connection, moment_id: str) -> Moment | None:
    row = conn.execute(
        "SELECT * FROM moments WHERE moment_id = ?", (moment_id,)
    ).fetchone()
    if row is None:
        return None
    return _row_to_moment(row)


def get_moments_by_video(
    conn: sqlite3.Connection, video_id: str
) -> list[Moment]:
    rows = conn.execute(
        "SELECT * FROM moments WHERE video_id = ? ORDER BY start_sec",
        (video_id,),
    ).fetchall()
    return [_row_to_moment(r) for r in rows]


def get_moment_count_by_video(
    conn: sqlite3.Connection, video_id: str
) -> int:
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM moments WHERE video_id = ?",
        (video_id,),
    ).fetchone()
    return row["cnt"] if row else 0


def get_all_moment_ids(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT moment_id FROM moments").fetchall()
    return [r["moment_id"] for r in rows]


def _row_to_moment(row: sqlite3.Row) -> Moment:
    return Moment(
        moment_id=row["moment_id"],
        video_id=row["video_id"],
        start_sec=row["start_sec"],
        end_sec=row["end_sec"],
        transcript_text=row["transcript_text"] or "",
        ocr_text=row["ocr_text"] or "",
        ideas=(
            [Idea(**i) for i in json.loads(row["ideas"])]
            if row["ideas"] else []
        ),
        keyframe_paths=(
            json.loads(row["keyframe_paths"])
            if row["keyframe_paths"] else []
        ),
        content_hash=row["content_hash"],
        parent_moment_id=row["parent_moment_id"],
        metadata=(
            json.loads(row["metadata"]) if row["metadata"] else {}
        ),
        created_at=row["created_at"],
    )


# ---- Evidence CRUD ----

def insert_evidence(conn: sqlite3.Connection, evidence: Evidence) -> None:
    conn.execute(
        """INSERT INTO modal_evidence
           (evidence_id, moment_id, modality, content,
            confidence, source, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (evidence.evidence_id, evidence.moment_id, evidence.modality,
         evidence.content, evidence.confidence, evidence.source,
         json.dumps(evidence.metadata)),
    )


# ---- Ingestion Run CRUD ----

def insert_ingestion_run(
    conn: sqlite3.Connection, run: IngestionRun
) -> None:
    conn.execute(
        """INSERT INTO ingestion_runs
           (run_id, video_id, status, pipeline_steps, error_message, started_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (run.run_id, run.video_id, run.status,
         json.dumps(run.pipeline_steps), run.error_message,
         run.started_at or datetime.now(timezone.utc).isoformat()),
    )


# ---- Keyframe CRUD ----

def insert_keyframe(conn: sqlite3.Connection, keyframe: Keyframe) -> None:
    conn.execute(
        """INSERT INTO keyframes
           (keyframe_id, moment_id, video_id, timestamp_sec, file_path,
            width, height, ocr_text, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (keyframe.keyframe_id, keyframe.moment_id, keyframe.video_id,
         keyframe.timestamp_sec, keyframe.file_path,
         keyframe.width, keyframe.height, keyframe.ocr_text,
         json.dumps(keyframe.metadata)),
    )


def get_keyframe_count_by_video(
    conn: sqlite3.Connection, video_id: str
) -> int:
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM keyframes WHERE video_id = ?",
        (video_id,),
    ).fetchone()
    return row["cnt"] if row else 0


# ---- Evidence CRUD ----

def get_evidence_by_moment(
    conn: sqlite3.Connection, moment_id: str
) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM modal_evidence WHERE moment_id = ? ORDER BY modality",
        (moment_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_evidence_count_by_video(
    conn: sqlite3.Connection, video_id: str
) -> int:
    row = conn.execute(
        """SELECT COUNT(*) as cnt FROM modal_evidence e
           JOIN moments m ON e.moment_id = m.moment_id
           WHERE m.video_id = ?""",
        (video_id,),
    ).fetchone()
    return row["cnt"] if row else 0


# ---- Ideas CRUD ----

def insert_idea(conn: sqlite3.Connection, idea: Idea) -> None:
    conn.execute(
        """INSERT INTO ideas
           (idea_id, moment_id, type, text, confidence, source)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (idea.idea_id, idea.moment_id, idea.type, idea.text,
         idea.confidence, idea.source),
    )


def get_idea_count_by_video(
    conn: sqlite3.Connection, video_id: str
) -> int:
    row = conn.execute(
        """SELECT COUNT(*) as cnt FROM ideas i
           JOIN moments m ON i.moment_id = m.moment_id
           WHERE m.video_id = ?""",
        (video_id,),
    ).fetchone()
    return row["cnt"] if row else 0


def complete_ingestion_run(
    conn: sqlite3.Connection,
    run_id: str,
    status: str = "completed",
    error_message: str | None = None,
) -> None:
    conn.execute(
        """UPDATE ingestion_runs
           SET status = ?, completed_at = datetime('now'), error_message = ?
           WHERE run_id = ?""",
        (status, error_message, run_id),
    )


# ---- Duplicates CRUD ----

def insert_duplicate(conn: sqlite3.Connection, dup: Duplicate) -> None:
    conn.execute(
        """INSERT INTO duplicates
           (dup_id, moment_id, canonical_moment_id, similarity_score,
            novelty_score, method, duplicate_type, item_type, reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (dup.dup_id, dup.moment_id, dup.canonical_moment_id,
         dup.similarity_score, dup.novelty_score, dup.method,
         dup.duplicate_type, dup.item_type, dup.reason),
    )


def get_duplicate_stats(conn: sqlite3.Connection) -> dict:
    total = conn.execute("SELECT COUNT(*) as c FROM duplicates").fetchone()["c"]
    by_type = {}
    for row in conn.execute(
        "SELECT duplicate_type, COUNT(*) as c FROM duplicates GROUP BY duplicate_type"
    ).fetchall():
        by_type[row["duplicate_type"]] = row["c"]
    return {"total": total, "by_type": by_type}


def get_duplicates_for_moment(
    conn: sqlite3.Connection, moment_id: str
) -> list[dict]:
    rows = conn.execute(
        """SELECT * FROM duplicates
           WHERE moment_id = ? OR canonical_moment_id = ?
           ORDER BY created_at""",
        (moment_id, moment_id),
    ).fetchall()
    return [dict(r) for r in rows]


def generate_dup_id() -> str:
    return f"dup:{uuid.uuid4().hex[:12]}"


# ---- FTS ----

def rebuild_fts(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM moments_fts")
    rows = conn.execute(
        """SELECT m.rowid, m.moment_id, m.transcript_text, m.ocr_text, m.ideas,
                  v.title as video_title, v.metadata as video_metadata
           FROM moments m
           JOIN videos v ON m.video_id = v.video_id"""
    ).fetchall()
    for row in rows:
        ideas_text = ""
        if row["ideas"]:
            ideas_list = json.loads(row["ideas"])
            ideas_text = " ".join(
                f"{i.get('type', '')}: {i.get('text', '')}"
                for i in ideas_list
            )
        video_description = ""
        if row["video_metadata"]:
            meta = json.loads(row["video_metadata"])
            video_description = meta.get("description", "") or ""
        conn.execute(
            """INSERT INTO moments_fts
               (rowid, moment_id, transcript_text, ocr_text, ideas_text,
                video_title, video_description)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (row["rowid"], row["moment_id"],
             row["transcript_text"] or "", row["ocr_text"] or "",
             ideas_text, row["video_title"] or "", video_description),
        )


# ---- ID generation ----

def generate_run_id() -> str:
    return f"run:{uuid.uuid4().hex[:12]}"


def generate_evidence_id() -> str:
    return f"ev:{uuid.uuid4().hex[:12]}"


def make_moment_id(
    video_id: str, start_sec: float, end_sec: float
) -> str:
    return f"{video_id}:{start_sec:.2f}:{end_sec:.2f}"


def make_video_id(source: str, identifier: str) -> str:
    if source == "youtube":
        match = re.search(
            r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", identifier
        )
        if match:
            return match.group(1)
    from vidcrawl.utils.hashing import sha256_prefix
    return sha256_prefix(identifier)


def make_idea_id(moment_id: str, index: int) -> str:
    return f"idea:{moment_id}:{index}"


def generate_keyframe_id() -> str:
    return f"kf:{uuid.uuid4().hex[:12]}"
