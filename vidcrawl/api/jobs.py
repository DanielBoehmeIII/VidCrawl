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
                  CHECK(status IN ('queued','running','ready','error','skipped')),
    source_url    TEXT NOT NULL,
    video_id      TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
    error_message TEXT,
    result        TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_api_jobs_source_url ON api_jobs(source_url);
CREATE INDEX IF NOT EXISTS idx_api_jobs_status ON api_jobs(status);
"""


def init_jobs_table(conn: sqlite3.Connection) -> None:
    conn.executescript(JOBS_TABLE_SQL)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


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
    conn.execute(
        """UPDATE api_jobs
           SET status = ?, updated_at = ?,
               error_message = COALESCE(?, error_message),
               video_id = COALESCE(?, video_id),
               result = COALESCE(?, result)
           WHERE job_id = ?""",
        (
            status,
            now,
            error_message,
            video_id,
            json.dumps(result) if result is not None else None,
            job_id,
        ),
    )
    conn.commit()


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
    }
