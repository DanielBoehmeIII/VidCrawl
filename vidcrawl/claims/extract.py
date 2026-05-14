import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from vidcrawl.db import get_db


@dataclass
class Claim:
    claim_id: str
    claim_text: str
    claim_type: str
    normalized_claim: str
    confidence: float
    source_moment_id: str
    evidence_ids: list[str] = field(default_factory=list)
    metadata_json: Optional[dict] = None


CLAIM_TYPE_PRIORITY = [
    ("definition", [
        r"\bis\s+(?:a|an|the)\s+", r"\b(?:means|refers to|defined as|is called|is known as)\b",
    ], 0.8),
    ("warning", [
        r"\b(?:careful|caution|warning|avoid|do not|dont|never|be careful|watch out|beware)\b",
    ], 0.8),
    ("recommendation", [
        r"\b(?:recommend|should|best practice|suggest|ideally|prefer|use this|try using)\b",
    ], 0.8),
    ("comparison", [
        r"\b(?:compare|comparison|versus|vs|unlike|whereas|on the other hand|better|worse|faster|slower)\b",
    ], 0.7),
    ("error_solution", [
        r"\b(?:error|bug|fix|solution|resolve|workaround|patch|hotfix|debug)\b",
    ], 0.7),
    ("step", [
        r"\b(?:first|then|next|after that|finally|step|install|run|create|build|set up|configure)\b",
    ], 0.65),
    ("limitation", [
        r"\b(?:limitation|drawback|downside|caveat|however|but |issue|problem|challenge|difficulty|complexity)\b",
    ], 0.65),
    ("factual", [
        r"\b(?:are |was |were |has |have |contains|consists|includes|supports)\b",
    ], 0.5),
]


def extract_claims_from_text(
    moment_id: str,
    text: str,
    source: str = "transcript",
) -> list[Claim]:
    claims: list[Claim] = []
    seen_texts: set[str] = set()
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 10:
            continue
        if sent.lower() in seen_texts:
            continue
        seen_texts.add(sent.lower())

        best_type = "factual"
        best_confidence = 0.5

        for ctype, patterns, conf in CLAIM_TYPE_PRIORITY:
            for pat in patterns:
                if re.search(pat, sent, re.IGNORECASE) and conf > best_confidence:
                    best_confidence = conf
                    best_type = ctype

        from vidcrawl.claims.normalize import normalize_claim
        norm = normalize_claim(sent)

        cid = f"clm:{uuid.uuid4().hex[:12]}"
        claims.append(Claim(
            claim_id=cid,
            claim_text=sent,
            claim_type=best_type,
            normalized_claim=norm,
            confidence=round(best_confidence, 2),
            source_moment_id=moment_id,
            metadata_json={"source": source},
        ))

    return claims


def run_claim_extraction(db_path: str) -> dict[str, Any]:
    conn = get_db(db_path)
    try:
        from vidcrawl.graph.build import init_graph_tables
        init_graph_tables(conn)

        _ensure_claim_tables(conn)

        moments = conn.execute(
            "SELECT moment_id, transcript_text, ocr_text FROM moments"
        ).fetchall()

        counts = {"total_claims": 0, "by_type": {}}
        claims_stored = 0

        for m in moments:
            combined = f"{m['transcript_text'] or ''}\n{m['ocr_text'] or ''}"
            claims = extract_claims_from_text(m["moment_id"], combined)
            for c in claims:
                conn.execute(
                    """INSERT OR REPLACE INTO claims
                       (claim_id, claim_text, claim_type, normalized_claim,
                        confidence, source_moment_id, metadata_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (c.claim_id, c.claim_text[:2000], c.claim_type,
                     c.normalized_claim[:2000], c.confidence, c.source_moment_id,
                     json.dumps(c.metadata_json or {})),
                )
                claims_stored += 1
                counts["by_type"][c.claim_type] = counts["by_type"].get(c.claim_type, 0) + 1

                _create_claim_graph_edges(conn, c)

        counts["total_claims"] = claims_stored
        conn.commit()
        return counts
    finally:
        conn.close()


def _ensure_claim_tables(conn) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS claims (
        claim_id TEXT PRIMARY KEY,
        claim_text TEXT NOT NULL,
        claim_type TEXT NOT NULL,
        normalized_claim TEXT DEFAULT '',
        confidence REAL DEFAULT 0.5,
        source_moment_id TEXT NOT NULL REFERENCES moments(moment_id),
        evidence_ids TEXT DEFAULT '[]',
        metadata_json TEXT DEFAULT '{}',
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )""")
    conn.execute("""CREATE INDEX IF NOT EXISTS idx_claims_moment ON claims(source_moment_id)""")
    conn.execute("""CREATE INDEX IF NOT EXISTS idx_claims_type ON claims(claim_type)""")


CLAIM_EDGE_TYPES = {
    "statement_moment": "moment_supports_claim",
    "claim_mentions_entity": "claim_mentions_entity",
    "claim_evidence": "claim_supported_by_evidence",
}

CLAIM_GRAPH_EDGE_TYPES_SQL = """
INSERT OR IGNORE INTO graph_nodes (node_id, node_type, ref_id, label, metadata_json)
VALUES (?, 'claim', ?, ?, '{}')
"""


def _create_claim_graph_edges(conn, claim: Claim) -> None:
    from vidcrawl.graph.build import _node_id, _ensure_edge, _ensure_node

    conn.execute("PRAGMA foreign_keys=OFF")

    claim_nid = _node_id("claim", claim.claim_id)
    _ensure_node(conn, "claim", claim.claim_id, label=claim.claim_text[:200],
                 metadata={"claim_type": claim.claim_type, "confidence": claim.confidence})

    existing_moment = conn.execute(
        "SELECT node_id FROM graph_nodes WHERE ref_id = ? AND node_type = 'moment'",
        (claim.source_moment_id,),
    ).fetchone()
    if existing_moment:
        moment_nid = existing_moment["node_id"]
    else:
        moment_nid = _node_id("moment", claim.source_moment_id)
        _ensure_node(conn, "moment", claim.source_moment_id, label=f"Moment {claim.source_moment_id}")

    _ensure_edge(conn, moment_nid, claim_nid, "moment_supports_claim")

    ev_rows = conn.execute(
        "SELECT evidence_id FROM modal_evidence WHERE moment_id = ? LIMIT 3",
        (claim.source_moment_id,),
    ).fetchall()
    for ev in ev_rows:
        ev_nid = _node_id("evidence", ev["evidence_id"])
        existing_ev = conn.execute(
            "SELECT node_id FROM graph_nodes WHERE node_id = ?", (ev_nid,)
        ).fetchone()
        if not existing_ev:
            _ensure_node(conn, "evidence", ev["evidence_id"], label=f"Evidence {ev['evidence_id']}")
        _ensure_edge(conn, claim_nid, ev_nid, "claim_supported_by_evidence")

    from vidcrawl.graph.entities import extract_entities
    entities = extract_entities(claim.claim_text)
    for ent in entities:
        entity_nid = _ensure_node(conn, "entity", ent["label"], label=ent["label"])
        _ensure_edge(conn, claim_nid, entity_nid, "claim_mentions_entity")

    conn.execute("PRAGMA foreign_keys=ON")


def get_claim_stats(db_path: str) -> dict[str, Any]:
    conn = get_db(db_path)
    try:
        _ensure_claim_tables(conn)
        total = conn.execute("SELECT COUNT(*) as c FROM claims").fetchone()["c"]
        by_type = {}
        for r in conn.execute(
            "SELECT claim_type, COUNT(*) as c FROM claims GROUP BY claim_type"
        ).fetchall():
            by_type[r["claim_type"]] = r["c"]
        return {"total_claims": total, "by_type": by_type}
    finally:
        conn.close()
