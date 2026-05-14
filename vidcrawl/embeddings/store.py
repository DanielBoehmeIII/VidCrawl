import json
import sqlite3
import struct
from typing import Any, Optional

from vidcrawl.db import get_db

EMBEDDING_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS embedding_runs (
    run_id       TEXT PRIMARY KEY,
    provider     TEXT NOT NULL,
    dimension    INTEGER NOT NULL,
    item_count   INTEGER DEFAULT 0,
    status       TEXT DEFAULT 'running',
    error_message TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS embedding_vectors (
    vector_id    TEXT PRIMARY KEY,
    item_type    TEXT NOT NULL CHECK(item_type IN ('moment', 'idea')),
    item_id      TEXT NOT NULL,
    run_id       TEXT NOT NULL REFERENCES embedding_runs(run_id),
    vector_blob  BLOB NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_embedding_vectors_item
    ON embedding_vectors(item_type, item_id);
CREATE INDEX IF NOT EXISTS idx_embedding_vectors_run
    ON embedding_vectors(run_id);
"""


def init_embedding_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(EMBEDDING_SCHEMA_SQL)


def pack_vector(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}d", *vec)


def unpack_vector(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 8}d", blob))


def store_vectors(
    conn: sqlite3.Connection,
    run_id: str,
    item_type: str,
    items: list[tuple[str, list[float]]],
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO embedding_runs (run_id, provider, dimension, status) VALUES (?, 'manual', 0, 'running')",
        (run_id,),
    )
    for item_id, vec in items:
        vid = f"vec:{item_type}:{item_id}"
        conn.execute(
            """INSERT OR REPLACE INTO embedding_vectors
               (vector_id, item_type, item_id, run_id, vector_blob)
               VALUES (?, ?, ?, ?, ?)""",
            (vid, item_type, item_id, run_id, pack_vector(vec)),
        )


def get_vector(
    conn: sqlite3.Connection,
    item_type: str,
    item_id: str,
) -> Optional[list[float]]:
    row = conn.execute(
        "SELECT vector_blob FROM embedding_vectors WHERE item_type = ? AND item_id = ?",
        (item_type, item_id),
    ).fetchone()
    if row is None:
        return None
    return unpack_vector(row["vector_blob"])


def get_all_vectors(
    conn: sqlite3.Connection,
    item_type: str = "moment",
) -> dict[str, list[float]]:
    rows = conn.execute(
        "SELECT item_id, vector_blob FROM embedding_vectors WHERE item_type = ?",
        (item_type,),
    ).fetchall()
    return {r["item_id"]: unpack_vector(r["vector_blob"]) for r in rows}


def build_embeddings(
    db_path: str,
    provider_name: str = "hash",
    item_type: str = "moment",
    dimension: int = 64,
) -> dict[str, Any]:
    from vidcrawl.embeddings.provider import get_provider

    provider = get_provider(provider_name, dimension=dimension)
    conn = get_db(db_path)
    try:
        init_embedding_tables(conn)

        run_id = f"embed:{provider.name}:{item_type}"
        conn.execute(
            "INSERT OR REPLACE INTO embedding_runs (run_id, provider, dimension, status) VALUES (?, ?, ?, 'running')",
            (run_id, provider.name, provider.dimension),
        )

        moments = conn.execute(
            "SELECT moment_id, transcript_text, ocr_text FROM moments"
        ).fetchall()
        texts: list[str] = []
        ids: list[str] = []
        for m in moments:
            combined = f"{m['transcript_text'] or ''} {m['ocr_text'] or ''}".strip()
            if combined:
                texts.append(combined)
                ids.append(m["moment_id"])

        if not texts:
            conn.execute(
                "UPDATE embedding_runs SET status = 'completed', completed_at = datetime('now'), item_count = 0 WHERE run_id = ?",
                (run_id,),
            )
            conn.commit()
            return {"run_id": run_id, "vectors_stored": 0, "dimension": provider.dimension, "provider": provider.name}

        vectors = provider.compute(texts)
        items = list(zip(ids, vectors))
        store_vectors(conn, run_id, item_type, items)

        conn.execute(
            "UPDATE embedding_runs SET status = 'completed', completed_at = datetime('now'), item_count = ? WHERE run_id = ?",
            (len(items), run_id),
        )
        conn.commit()

        return {
            "run_id": run_id,
            "vectors_stored": len(items),
            "dimension": provider.dimension,
            "provider": provider.name,
        }
    finally:
        conn.close()


def get_embedding_stats(db_path: str) -> dict[str, Any]:
    conn = get_db(db_path)
    try:
        init_embedding_tables(conn)
        run = conn.execute(
            "SELECT * FROM embedding_runs ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if run is None:
            return {"has_embeddings": False, "vector_count": 0}
        count = conn.execute(
            "SELECT COUNT(*) as c FROM embedding_vectors"
        ).fetchone()["c"]
        return {
            "has_embeddings": count > 0,
            "vector_count": count,
            "provider": run["provider"],
            "dimension": run["dimension"],
            "run_id": run["run_id"],
            "status": run["status"],
            "created_at": run["created_at"],
        }
    finally:
        conn.close()


def has_embeddings(db_path: str) -> bool:
    conn = get_db(db_path)
    try:
        count = conn.execute(
            "SELECT COUNT(*) as c FROM embedding_vectors"
        ).fetchone()["c"]
        return count > 0
    except Exception:
        return False
    finally:
        conn.close()
