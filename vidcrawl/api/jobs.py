import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

JOBS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS api_jobs (
    job_id        TEXT PRIMARY KEY,
    status        TEXT NOT NULL DEFAULT 'queued'
                  CHECK(status IN (
                      'queued','running','ready','error','skipped',
                      'needs_transcription','skipped_no_transcript'
                  )),
    source_url    TEXT NOT NULL,
    video_id      TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
    error_message TEXT,
    result        TEXT DEFAULT '{}',
    stage         TEXT DEFAULT 'queued',
    progress_message TEXT DEFAULT '',
    started_at    TEXT,
    finished_at   TEXT,
    duration_ms   INTEGER,
    last_heartbeat_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_api_jobs_source_url ON api_jobs(source_url);
CREATE INDEX IF NOT EXISTS idx_api_jobs_status ON api_jobs(status);
"""


def init_jobs_table(conn: sqlite3.Connection) -> None:
    conn.executescript(JOBS_TABLE_SQL)
    _migrate_jobs_table(conn)


def _migrate_jobs_table(conn: sqlite3.Connection) -> None:
    _try_add_column(conn, "api_jobs", "stage", "TEXT DEFAULT 'queued'")
    _try_add_column(conn, "api_jobs", "progress_message", "TEXT DEFAULT ''")
    _try_add_column(conn, "api_jobs", "started_at", "TEXT")
    _try_add_column(conn, "api_jobs", "finished_at", "TEXT")
    _try_add_column(conn, "api_jobs", "duration_ms", "INTEGER")
    _try_add_column(conn, "api_jobs", "last_heartbeat_at", "TEXT")


def _try_add_column(conn, table: str, column: str, col_type: str) -> None:
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
    except sqlite3.OperationalError:
        pass


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def create_job(conn: sqlite3.Connection, source_url: str, video_id: Optional[str] = None) -> str:
    job_id = str(uuid.uuid4())
    now = _now()
    conn.execute(
        """INSERT INTO api_jobs (job_id, status, source_url, video_id, created_at, updated_at)
           VALUES (?, 'queued', ?, ?, ?, ?)""",
        (job_id, source_url, video_id, now, now),
    )
    conn.commit()
    return job_id


def update_job(
    conn: sqlite3.Connection,
    job_id: str,
    status: str,
    error_message: Optional[str] = None,
    video_id: Optional[str] = None,
    result: Optional[dict] = None,
) -> None:
    now = _now()

    current = get_job(conn, job_id)
    started_at = None
    finished_at = None
    duration_ms = None

    if status == "running":
        started_at = current["started_at"] if current and current.get("started_at") else now
    elif status in ("ready", "error", "skipped", "needs_transcription", "skipped_no_transcript"):
        finished_at = now
        if current and current.get("started_at"):
            try:
                start = datetime.fromisoformat(current["started_at"])
                end = datetime.fromisoformat(now)
                duration_ms = int((end - start).total_seconds() * 1000)
            except (ValueError, TypeError):
                pass

    conn.execute(
        """UPDATE api_jobs
           SET status = ?, updated_at = ?,
               error_message = COALESCE(?, error_message),
               video_id = COALESCE(?, video_id),
               result = COALESCE(?, result),
               started_at = COALESCE(?, started_at),
               finished_at = COALESCE(?, finished_at),
               duration_ms = COALESCE(?, duration_ms),
               last_heartbeat_at = COALESCE(?, last_heartbeat_at)
           WHERE job_id = ?""",
        (
            status, now,
            error_message, video_id,
            json.dumps(result) if result is not None else None,
            started_at, finished_at, duration_ms,
            now if status == "running" else (now if status in ("ready", "error", "skipped", "needs_transcription", "skipped_no_transcript") else None),
            job_id,
        ),
    )
    conn.commit()


def update_job_progress(
    conn: sqlite3.Connection,
    job_id: str,
    stage: str,
    progress_message: str = "",
) -> None:
    now = _now()
    conn.execute(
        """UPDATE api_jobs
           SET stage = ?, progress_message = ?, updated_at = ?,
               last_heartbeat_at = ?
           WHERE job_id = ?""",
        (stage, progress_message, now, now, job_id),
    )
    conn.commit()


def update_job_heartbeat(conn: sqlite3.Connection, job_id: str) -> None:
    now = _now()
    conn.execute(
        "UPDATE api_jobs SET last_heartbeat_at = ?, updated_at = ? WHERE job_id = ?",
        (now, now, job_id),
    )
    conn.commit()


def find_stuck_jobs(conn: sqlite3.Connection, timeout_sec: int = 120) -> list[dict]:
    rows = conn.execute(
        """SELECT * FROM api_jobs
           WHERE status = 'running'
           AND (
               last_heartbeat_at IS NULL
               OR (julianday('now') - julianday(last_heartbeat_at)) * 86400 > ?
           )""",
        (timeout_sec,),
    ).fetchall()
    return [_row_to_job(r) for r in rows]


def mark_stuck_jobs(conn: sqlite3.Connection, timeout_sec: int = 120) -> list[dict]:
    stuck = find_stuck_jobs(conn, timeout_sec)
    for job in stuck:
        msg = f"Job stuck - no heartbeat for > {timeout_sec}s"
        conn.execute(
            """UPDATE api_jobs
               SET status = 'error', stage = 'error',
                   progress_message = ?,
                   error_message = COALESCE(error_message, 'Job timed out - no progress detected')
               WHERE job_id = ?""",
            (msg, job["job_id"]),
        )
    if stuck:
        conn.commit()
    return stuck


def get_job(conn: sqlite3.Connection, job_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM api_jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    return _row_to_job(row) if row else None


def find_job_by_url(conn: sqlite3.Connection, source_url: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM api_jobs WHERE source_url = ? ORDER BY created_at DESC LIMIT 1",
        (source_url,),
    ).fetchone()
    return _row_to_job(row) if row else None


def list_jobs(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM api_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [_row_to_job(r) for r in rows]


def _safe_get(row: sqlite3.Row, key: str, default=None):
    try:
        return row[key]
    except (IndexError, KeyError):
        return default


def _row_to_job(row: sqlite3.Row) -> dict:
    return {
        "job_id": row["job_id"],
        "status": row["status"],
        "source_url": row["source_url"],
        "video_id": row["video_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "error_message": row["error_message"],
        "result": json.loads(row["result"]) if row["result"] else {},
        "stage": _safe_get(row, "stage", ""),
        "progress_message": _safe_get(row, "progress_message", ""),
        "started_at": _safe_get(row, "started_at"),
        "finished_at": _safe_get(row, "finished_at"),
        "duration_ms": _safe_get(row, "duration_ms"),
        "last_heartbeat_at": _safe_get(row, "last_heartbeat_at"),
    }
