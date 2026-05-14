import uuid
from typing import Any

from vidcrawl.db import get_db, insert_evidence
from vidcrawl.models import Evidence
from vidcrawl.technical.code import extract_file_paths, extract_code_identifiers, extract_imports
from vidcrawl.technical.commands import extract_commands
from vidcrawl.technical.errors import extract_errors
from vidcrawl.technical.equations import extract_equations
from vidcrawl.graph.build import init_graph_tables


NEW_MODALITIES = {"code", "command", "error", "equation"}


def migrate_modalities(db_path: str) -> None:
    conn = get_db(db_path)
    try:
        conn.execute(
            "ALTER TABLE modal_evidence DROP CONSTRAINT IF EXISTS check_modality"
        )
    except Exception:
        pass
    try:
        conn.execute(
            "ALTER TABLE modal_evidence ADD COLUMN modality_new TEXT"
        )
    except Exception:
        pass
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS modal_evidence_new (
                evidence_id TEXT PRIMARY KEY,
                moment_id   TEXT NOT NULL REFERENCES moments(moment_id),
                modality    TEXT NOT NULL,
                content     TEXT NOT NULL,
                confidence  REAL DEFAULT 1.0,
                source      TEXT,
                metadata    TEXT DEFAULT '{}',
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            )"""
        )
        existing = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='modal_evidence_new'"
        ).fetchone()
        if existing:
            conn.execute(
                """INSERT OR IGNORE INTO modal_evidence_new
                   SELECT * FROM modal_evidence"""
            )
            conn.execute("DROP TABLE modal_evidence")
            conn.execute("ALTER TABLE modal_evidence_new RENAME TO modal_evidence")
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        conn.close()


def extract_technical_evidence_for_moment(
    moment_id: str,
    transcript_text: str,
    ocr_text: str,
) -> dict[str, list[str]]:
    combined = f"{transcript_text}\n{ocr_text}"
    return {
        "file_paths": extract_file_paths(combined),
        "commands": extract_commands(combined),
        "errors": extract_errors(combined),
        "equations": extract_equations(combined),
        "code_identifiers": [ci["identifier"] for ci in extract_code_identifiers(combined)],
        "imports": extract_imports(combined),
    }


def _ensure_modal_table(conn) -> None:
    try:
        conn.execute(
            "INSERT INTO modal_evidence (evidence_id, moment_id, modality, content, confidence, source) "
            "VALUES ('_test_check', '_test_check', 'code', 'test', 1.0, 'test')"
        )
        conn.execute("DELETE FROM modal_evidence WHERE evidence_id = '_test_check'")
        return
    except Exception:
        pass

    conn.execute("""CREATE TABLE IF NOT EXISTS modal_evidence_v2 (
        evidence_id TEXT PRIMARY KEY,
        moment_id   TEXT NOT NULL REFERENCES moments(moment_id),
        modality    TEXT NOT NULL,
        content     TEXT NOT NULL,
        confidence  REAL DEFAULT 1.0,
        source      TEXT,
        metadata    TEXT DEFAULT '{}',
        created_at  TEXT NOT NULL DEFAULT (datetime('now'))
    )""")
    conn.execute("""INSERT OR IGNORE INTO modal_evidence_v2 SELECT * FROM modal_evidence""")
    conn.execute("DROP TABLE IF EXISTS modal_evidence")
    conn.execute("ALTER TABLE modal_evidence_v2 RENAME TO modal_evidence")
    conn.execute("""CREATE INDEX IF NOT EXISTS idx_evidence_moment_id ON modal_evidence(moment_id)""")
    conn.commit()


def _modality_for_key(key: str) -> str:
    mapping = {
        "file_paths": "file_path",
        "commands": "command",
        "errors": "error",
        "equations": "equation",
        "code_identifiers": "code",
    }
    return mapping.get(key, key)


def run_technical_extraction(db_path: str) -> dict[str, Any]:
    conn = get_db(db_path)
    try:
        _ensure_modal_table(conn)
        init_graph_tables(conn)
        moments = conn.execute(
            "SELECT moment_id, transcript_text, ocr_text FROM moments"
        ).fetchall()

        counts: dict[str, int] = {
            "file_paths": 0,
            "commands": 0,
            "errors": 0,
            "equations": 0,
            "code_identifiers": 0,
            "moments_with_technical": 0,
        }
        total_evidence_inserted = 0

        for m in moments:
            result = extract_technical_evidence_for_moment(
                m["moment_id"],
                m["transcript_text"] or "",
                m["ocr_text"] or "",
            )

            has_any = False
            for key in ["file_paths", "commands", "errors", "equations", "code_identifiers"]:
                modality = _modality_for_key(key)
                for item in result.get(key, []):
                    ev_id = f"tev:{uuid.uuid4().hex[:12]}"
                    conf = 0.8 if key == "code_identifiers" else 0.9
                    try:
                        conn.execute(
                            """INSERT INTO modal_evidence
                               (evidence_id, moment_id, modality, content, confidence, source)
                               VALUES (?, ?, ?, ?, ?, 'technical')""",
                            (ev_id, m["moment_id"], modality, item[:1000], conf),
                        )
                    except Exception:
                        _ensure_modal_table(conn)
                        conn.execute(
                            """INSERT INTO modal_evidence
                               (evidence_id, moment_id, modality, content, confidence, source)
                               VALUES (?, ?, ?, ?, ?, 'technical')""",
                            (ev_id, m["moment_id"], modality, item[:1000], conf),
                        )
                    counts[key] = counts.get(key, 0) + 1
                    total_evidence_inserted += 1
                    has_any = True

            if has_any:
                counts["moments_with_technical"] += 1

        conn.commit()
        counts["total_evidence_inserted"] = total_evidence_inserted
        return counts
    finally:
        conn.close()


def get_technical_stats(db_path: str) -> dict[str, Any]:
    conn = get_db(db_path)
    try:
        evidence_counts = conn.execute(
            """SELECT modality, COUNT(*) as c
               FROM modal_evidence
               WHERE modality IN ('file_path', 'command', 'error', 'equation', 'code')
               GROUP BY modality"""
        ).fetchall()

        total_moments = conn.execute(
            "SELECT COUNT(*) as c FROM moments"
        ).fetchone()["c"]

        moments_with_tech = conn.execute(
            """SELECT COUNT(DISTINCT moment_id) as c
               FROM modal_evidence
               WHERE modality IN ('file_path', 'command', 'error', 'equation', 'code')"""
        ).fetchone()["c"]

        by_modality = {r["modality"]: r["c"] for r in evidence_counts}
        return {
            "total_technical_evidence": sum(by_modality.values()),
            "moments_with_technical": moments_with_tech,
            "total_moments": total_moments,
            "by_modality": by_modality,
        }
    finally:
        conn.close()
