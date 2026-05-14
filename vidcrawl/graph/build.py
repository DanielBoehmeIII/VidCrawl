import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from vidcrawl.db import get_db
from vidcrawl.graph.entities import extract_entities


@dataclass
class GraphBuildSummary:
    nodes_created: int = 0
    edges_created: int = 0
    video_nodes: int = 0
    moment_nodes: int = 0
    idea_nodes: int = 0
    evidence_nodes: int = 0
    entity_nodes: int = 0
    cluster_nodes: int = 0
    claim_stub_nodes: int = 0


NODE_TYPES = [
    "video", "moment", "idea", "evidence", "entity",
    "duplicate_cluster", "claim_stub", "claim",
]

EDGE_TYPES = [
    "video_contains_moment",
    "moment_has_evidence",
    "moment_expresses_idea",
    "idea_mentioned_in_moment",
    "idea_mentions_entity",
    "evidence_supports_idea",
    "moment_duplicate_of",
    "moment_variant_of",
    "moment_same_idea_as",
    "cluster_has_moment",
    "cluster_canonical_moment",
    "video_from_source",
    "moment_supports_claim",
    "claim_mentions_entity",
    "claim_supported_by_evidence",
    "claim_same_as_claim",
    "claim_variant_of_claim",
    "claim_contradicts_claim_stub",
    "claim_updates_claim_stub",
]

GRAPH_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS graph_nodes (
    node_id       TEXT PRIMARY KEY,
    node_type     TEXT NOT NULL,
    ref_id        TEXT,
    label         TEXT NOT NULL DEFAULT '',
    metadata_json TEXT DEFAULT '{}',
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS graph_edges (
    edge_id       TEXT PRIMARY KEY,
    source_node_id TEXT NOT NULL REFERENCES graph_nodes(node_id),
    target_node_id TEXT NOT NULL REFERENCES graph_nodes(node_id),
    edge_type     TEXT NOT NULL,
    weight        REAL DEFAULT 1.0,
    metadata_json TEXT DEFAULT '{}',
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_ref_id ON graph_nodes(ref_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_node_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_type ON graph_edges(edge_type);
"""


def init_graph_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(GRAPH_SCHEMA_SQL)


def clear_graph(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM graph_edges")
    conn.execute("DELETE FROM graph_nodes")


def _node_id(prefix: str, ref: str) -> str:
    return f"{prefix}:{ref}"


def _edge_id() -> str:
    return f"edge:{uuid.uuid4().hex[:12]}"


def _ensure_node(
    conn: sqlite3.Connection,
    node_type: str,
    ref_id: str,
    label: str = "",
    metadata: Optional[dict] = None,
) -> str:
    nid = _node_id(node_type, ref_id)
    conn.execute(
        """INSERT OR IGNORE INTO graph_nodes
           (node_id, node_type, ref_id, label, metadata_json)
           VALUES (?, ?, ?, ?, ?)""",
        (nid, node_type, ref_id, label, json.dumps(metadata or {})),
    )
    return nid


def _ensure_edge(
    conn: sqlite3.Connection,
    source_node_id: str,
    target_node_id: str,
    edge_type: str,
    weight: float = 1.0,
    metadata: Optional[dict] = None,
) -> str:
    eid = _edge_id()
    conn.execute(
        """INSERT OR IGNORE INTO graph_edges
           (edge_id, source_node_id, target_node_id, edge_type, weight, metadata_json)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (eid, source_node_id, target_node_id, edge_type, weight,
         json.dumps(metadata or {})),
    )
    return eid


def build_graph(
    db_path: Path,
    rebuild: bool = False,
    entity_texts: Optional[dict[str, str]] = None,
) -> GraphBuildSummary:
    conn = get_db(db_path)
    try:
        return _build_graph_inner(conn, rebuild, entity_texts)
    finally:
        conn.close()


def _build_graph_inner(
    conn: sqlite3.Connection,
    rebuild: bool = False,
    entity_texts: Optional[dict[str, str]] = None,
) -> GraphBuildSummary:
    init_graph_tables(conn)

    if rebuild:
        clear_graph(conn)

    existing = conn.execute("SELECT COUNT(*) as c FROM graph_nodes").fetchone()["c"]
    if existing > 0 and not rebuild:
        return GraphBuildSummary()

    summary = GraphBuildSummary()

    _build_video_nodes(conn, summary)
    _build_moment_nodes(conn, summary)
    _build_idea_nodes(conn, summary)
    _build_evidence_nodes(conn, summary)
    _build_entity_nodes(conn, summary, entity_texts)
    _build_cluster_nodes(conn, summary)
    _build_moment_idea_edges(conn, summary)
    _build_moment_evidence_edges(conn, summary)
    _build_video_moment_edges(conn, summary)
    _build_idea_entity_edges(conn, summary)
    _build_duplicate_edges(conn, summary)
    _build_source_edges(conn, summary)

    conn.commit()
    return summary


def _build_video_nodes(conn: sqlite3.Connection, summary: GraphBuildSummary) -> None:
    rows = conn.execute("SELECT video_id, title, source, metadata FROM videos").fetchall()
    for row in rows:
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        _ensure_node(
            conn, "video", row["video_id"],
            label=row["title"],
            metadata={"source": row["source"], **meta},
        )
        summary.video_nodes += 1
        summary.nodes_created += 1


def _build_moment_nodes(conn: sqlite3.Connection, summary: GraphBuildSummary) -> None:
    rows = conn.execute(
        "SELECT moment_id, video_id, start_sec, end_sec, transcript_text, ocr_text, content_hash, metadata FROM moments"
    ).fetchall()
    for row in rows:
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        label = f"{row['video_id']} {row['start_sec']}s-{row['end_sec']}s"
        transcript_preview = (row["transcript_text"] or "")[:100]
        ocr_preview = (row["ocr_text"] or "")[:100]
        _ensure_node(
            conn, "moment", row["moment_id"],
            label=label,
            metadata={
                "video_id": row["video_id"],
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"],
                "transcript_preview": transcript_preview,
                "ocr_preview": ocr_preview,
                "content_hash": row["content_hash"],
                **meta,
            },
        )
        summary.moment_nodes += 1
        summary.nodes_created += 1


def _build_idea_nodes(conn: sqlite3.Connection, summary: GraphBuildSummary) -> None:
    rows = conn.execute(
        "SELECT idea_id, moment_id, type, text, confidence FROM ideas"
    ).fetchall()
    for row in rows:
        label = f"[{row['type']}] {row['text'][:120]}"
        _ensure_node(
            conn, "idea", row["idea_id"],
            label=label,
            metadata={
                "moment_id": row["moment_id"],
                "idea_type": row["type"],
                "text": row["text"],
                "confidence": row["confidence"],
            },
        )
        summary.idea_nodes += 1
        summary.nodes_created += 1


def _build_evidence_nodes(conn: sqlite3.Connection, summary: GraphBuildSummary) -> None:
    rows = conn.execute(
        "SELECT evidence_id, moment_id, modality, content, confidence FROM modal_evidence"
    ).fetchall()
    for row in rows:
        label = f"[{row['modality']}] {(row['content'] or '')[:120]}"
        _ensure_node(
            conn, "evidence", row["evidence_id"],
            label=label,
            metadata={
                "moment_id": row["moment_id"],
                "modality": row["modality"],
                "content_preview": (row["content"] or "")[:200],
                "confidence": row["confidence"],
            },
        )
        summary.evidence_nodes += 1
        summary.nodes_created += 1


def _build_entity_nodes(
    conn: sqlite3.Connection,
    summary: GraphBuildSummary,
    entity_texts: Optional[dict[str, str]] = None,
) -> None:
    texts_to_scan: dict[str, str] = {}

    if entity_texts:
        texts_to_scan.update(entity_texts)
    else:
        rows = conn.execute(
            "SELECT moment_id, transcript_text, ocr_text FROM moments"
        ).fetchall()
        for row in rows:
            texts_to_scan[row["moment_id"]] = (
                (row["transcript_text"] or "") + "\n" + (row["ocr_text"] or "")
            )

        idea_rows = conn.execute(
            "SELECT moment_id, text FROM ideas"
        ).fetchall()
        for row in idea_rows:
            mid = row["moment_id"]
            if mid in texts_to_scan:
                texts_to_scan[mid] += "\n" + (row["text"] or "")

    for moment_id, text in texts_to_scan.items():
        entities = extract_entities(text)
        for ent in entities:
            nid = _ensure_node(
                conn, "entity", ent["label"],
                label=ent["label"],
                metadata={
                    "entity_type": ent["entity_type"],
                    "source_text": ent["source_text"],
                },
            )
            summary.entity_nodes += 1
            summary.nodes_created += 1


def _build_cluster_nodes(conn: sqlite3.Connection, summary: GraphBuildSummary) -> None:
    dup_rows = conn.execute(
        """SELECT DISTINCT canonical_moment_id, duplicate_type
           FROM duplicates"""
    ).fetchall()

    canonical_ids = set()
    for row in dup_rows:
        canonical_ids.add(row["canonical_moment_id"])

    for cid in canonical_ids:
        mom = conn.execute(
            "SELECT moment_id, video_id, start_sec, end_sec FROM moments WHERE moment_id = ?",
            (cid,),
        ).fetchone()
        if mom is None:
            continue
        label = f"Cluster: {mom['video_id']} {mom['start_sec']}s-{mom['end_sec']}s"
        cluster_nid = _ensure_node(
            conn, "duplicate_cluster", cid,
            label=label,
            metadata={"canonical_moment_id": cid},
        )
        summary.cluster_nodes += 1
        summary.nodes_created += 1

        canonical_nid = _node_id("moment", cid)
        _ensure_edge(
            conn, cluster_nid, canonical_nid,
            "cluster_canonical_moment", weight=1.0,
        )
        summary.edges_created += 1

        all_related = conn.execute(
            """SELECT moment_id FROM duplicates
               WHERE canonical_moment_id = ?
               UNION
               SELECT canonical_moment_id FROM duplicates
               WHERE moment_id = ?""",
            (cid, cid),
        ).fetchall()
        for rel in all_related:
            rel_mid = rel["moment_id"]
            if rel_mid == cid:
                continue
            rel_nid = _node_id("moment", rel_mid)
            _ensure_edge(
                conn, cluster_nid, rel_nid,
                "cluster_has_moment", weight=1.0,
            )
            summary.edges_created += 1


def _build_moment_idea_edges(conn: sqlite3.Connection, summary: GraphBuildSummary) -> None:
    rows = conn.execute(
        "SELECT idea_id, moment_id FROM ideas"
    ).fetchall()
    for row in rows:
        idea_nid = _node_id("idea", row["idea_id"])
        moment_nid = _node_id("moment", row["moment_id"])
        _ensure_edge(conn, moment_nid, idea_nid, "moment_expresses_idea")
        summary.edges_created += 1
        _ensure_edge(conn, idea_nid, moment_nid, "idea_mentioned_in_moment")
        summary.edges_created += 1


def _build_moment_evidence_edges(conn: sqlite3.Connection, summary: GraphBuildSummary) -> None:
    rows = conn.execute(
        "SELECT evidence_id, moment_id FROM modal_evidence"
    ).fetchall()
    for row in rows:
        ev_nid = _node_id("evidence", row["evidence_id"])
        moment_nid = _node_id("moment", row["moment_id"])
        _ensure_edge(conn, moment_nid, ev_nid, "moment_has_evidence")
        summary.edges_created += 1
        _ensure_edge(conn, ev_nid, moment_nid, "evidence_supports_idea")
        summary.edges_created += 1


def _build_video_moment_edges(conn: sqlite3.Connection, summary: GraphBuildSummary) -> None:
    rows = conn.execute(
        "SELECT moment_id, video_id FROM moments"
    ).fetchall()
    for row in rows:
        video_nid = _node_id("video", row["video_id"])
        moment_nid = _node_id("moment", row["moment_id"])
        _ensure_edge(conn, video_nid, moment_nid, "video_contains_moment")
        summary.edges_created += 1


def _build_idea_entity_edges(conn: sqlite3.Connection, summary: GraphBuildSummary) -> None:
    rows = conn.execute("SELECT idea_id, moment_id, text FROM ideas").fetchall()
    for row in rows:
        text = row["text"] or ""
        entities = extract_entities(text)
        for ent in entities:
            entity_nid = _node_id("entity", ent["label"])
            idea_nid = _node_id("idea", row["idea_id"])
            _ensure_edge(conn, idea_nid, entity_nid, "idea_mentions_entity")
            summary.edges_created += 1


def _build_duplicate_edges(conn: sqlite3.Connection, summary: GraphBuildSummary) -> None:
    edge_type_map = {
        "exact": "moment_duplicate_of",
        "near_text": "moment_duplicate_of",
        "same_idea": "moment_same_idea_as",
        "variant": "moment_variant_of",
    }

    rows = conn.execute(
        "SELECT moment_id, canonical_moment_id, duplicate_type, similarity_score FROM duplicates"
    ).fetchall()
    for row in rows:
        dup_type = row["duplicate_type"]
        etype = edge_type_map.get(dup_type, "moment_duplicate_of")
        moment_nid = _node_id("moment", row["moment_id"])
        canonical_nid = _node_id("moment", row["canonical_moment_id"])
        _ensure_edge(
            conn, moment_nid, canonical_nid,
            etype, weight=row["similarity_score"],
            metadata={"duplicate_type": dup_type},
        )
        summary.edges_created += 1


def _build_source_edges(conn: sqlite3.Connection, summary: GraphBuildSummary) -> None:
    rows = conn.execute("SELECT video_id, source FROM videos").fetchall()
    for row in rows:
        video_nid = _node_id("video", row["video_id"])
        source_entity = row["source"]
        source_nid = _ensure_node(
            conn, "entity", f"source:{source_entity}",
            label=source_entity,
            metadata={"entity_type": "source"},
        )
        _ensure_edge(conn, video_nid, source_nid, "video_from_source")
        summary.edges_created += 1
