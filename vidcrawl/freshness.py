import json
import re
from datetime import datetime
from typing import Any, Optional

from vidcrawl.db import get_db

STALE_KEYWORDS = [
    "deprecated", "deprecation", "outdated", "legacy", "old version",
    "no longer supported", "replaced by", "superseded", "obsolete",
    "not recommended", "you should not use", "avoid using",
]

FRESH_KEYWORDS = [
    "new", "latest", "updated", "modern", "current", "recent",
    "new version", "v2", "v3", "next gen", "next-generation",
]

TOOL_DECAY_KEYWORDS = [
    "API", "SDK", "library", "package", "module", "framework",
    "tool", "language", "version", "release", "npm", "pip",
]


def compute_freshness_score(
    text: str,
    created_at: Optional[str] = None,
    upload_date: Optional[str] = None,
) -> float:
    score = 1.0
    text_lower = text.lower()

    for kw in STALE_KEYWORDS:
        if kw in text_lower:
            score -= 0.3
            break

    for kw in FRESH_KEYWORDS:
        if kw in text_lower:
            score += 0.2
            break

    has_tool_lang = any(kw.lower() in text_lower for kw in TOOL_DECAY_KEYWORDS)
    if has_tool_lang:
        score -= 0.1

    return max(0.0, min(1.0, score))


def run_freshness_scoring(db_path: str) -> dict[str, Any]:
    conn = get_db(db_path)
    try:
        _ensure_freshness_table(conn)

        moments = conn.execute(
            "SELECT moment_id, transcript_text, ocr_text, created_at FROM moments"
        ).fetchall()

        counts = {"scored": 0, "stale": 0, "fresh": 0, "neutral": 0}

        for m in moments:
            combined = f"{m['transcript_text'] or ''} {m['ocr_text'] or ''}"
            score = compute_freshness_score(combined, created_at=m["created_at"])

            conn.execute(
                """INSERT OR REPLACE INTO freshness_scores
                   (item_type, item_id, freshness_score, stale_reason, metadata_json)
                   VALUES ('moment', ?, ?, ?, ?)""",
                (m["moment_id"], score,
                 "potentially stale" if score < 0.5 else "",
                 json.dumps({"computed_from": "text_analysis"})),
            )
            counts["scored"] += 1
            if score >= 0.7:
                counts["fresh"] += 1
            elif score < 0.5:
                counts["stale"] += 1
            else:
                counts["neutral"] += 1

        conn.commit()
        return counts
    finally:
        conn.close()


def _ensure_freshness_table(conn) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS freshness_scores (
        item_type TEXT NOT NULL,
        item_id TEXT NOT NULL,
        freshness_score REAL DEFAULT 1.0,
        stale_reason TEXT DEFAULT '',
        superseded_by TEXT,
        metadata_json TEXT DEFAULT '{}',
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        PRIMARY KEY (item_type, item_id)
    )""")


def get_freshness_stats(db_path: str) -> dict[str, Any]:
    conn = get_db(db_path)
    try:
        _ensure_freshness_table(conn)
        total = conn.execute("SELECT COUNT(*) as c FROM freshness_scores").fetchone()["c"]
        avg_score = conn.execute(
            "SELECT AVG(freshness_score) as a FROM freshness_scores"
        ).fetchone()["a"] or 0.0
        stale = conn.execute(
            "SELECT COUNT(*) as c FROM freshness_scores WHERE freshness_score < 0.5"
        ).fetchone()["c"]
        fresh = conn.execute(
            "SELECT COUNT(*) as c FROM freshness_scores WHERE freshness_score >= 0.7"
        ).fetchone()["c"]
        return {
            "total_scored": total,
            "average_freshness": round(float(avg_score), 3),
            "stale_count": stale,
            "fresh_count": fresh,
        }
    finally:
        conn.close()
