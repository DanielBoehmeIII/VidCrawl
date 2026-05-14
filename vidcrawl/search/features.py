import json
import sqlite3
from typing import Any

from vidcrawl.db import get_db

QUERY_PATTERNS = {
    "warning": ["careful", "caution", "warning", "avoid", "be careful", "do not", "dont", "never", "watch out"],
    "example": ["example", "for instance", "such as", "e.g.", "like", "scenario"],
    "comparison": ["compare", "comparison", "versus", "vs", "unlike", "whereas", "on the other hand"],
    "code_like": ["install", "run ", "npm", "pip", "git ", "import ", "function", "class "],
}


def compute_features(
    db_path: str,
    moment_ids: list[str],
    query: str = "",
) -> dict[str, dict[str, Any]]:
    if not moment_ids:
        return {}

    conn = get_db(db_path)
    try:
        return _compute_features_batch(conn, moment_ids, query)
    finally:
        conn.close()


def _compute_features_batch(
    conn: sqlite3.Connection,
    moment_ids: list[str],
    query: str,
) -> dict[str, dict[str, Any]]:
    placeholders = ",".join("?" * len(moment_ids))
    moment_rows = conn.execute(
        f"""SELECT m.moment_id, m.video_id, m.start_sec, m.end_sec,
                   m.transcript_text, m.ocr_text, m.ideas, m.keyframe_paths,
                   COALESCE(v.title, '') as video_title
            FROM moments m
            JOIN videos v ON m.video_id = v.video_id
            WHERE m.moment_id IN ({placeholders})""",
        moment_ids,
    ).fetchall()
    moment_map = {r["moment_id"]: r for r in moment_rows}

    idea_rows = conn.execute(
        f"""SELECT moment_id, COUNT(*) as cnt, GROUP_CONCAT(type, ',') as types
            FROM ideas WHERE moment_id IN ({placeholders})
            GROUP BY moment_id""",
        moment_ids,
    ).fetchall()
    idea_counts = {r["moment_id"]: r["cnt"] for r in idea_rows}
    idea_types_map: dict[str, list[str]] = {}
    for r in idea_rows:
        if r["types"]:
            idea_types_map[r["moment_id"]] = sorted(set(r["types"].split(",")))

    evidence_rows = conn.execute(
        f"""SELECT moment_id, modality, COUNT(*) as cnt
            FROM modal_evidence WHERE moment_id IN ({placeholders})
            GROUP BY moment_id, modality""",
        moment_ids,
    ).fetchall()
    evidence_by_moment: dict[str, int] = {}
    modality_by_moment: dict[str, set[str]] = {}
    for r in evidence_rows:
        mid = r["moment_id"]
        evidence_by_moment[mid] = evidence_by_moment.get(mid, 0) + r["cnt"]
        modality_by_moment.setdefault(mid, set()).add(r["modality"])

    dup_rows = conn.execute(
        f"""SELECT moment_id, duplicate_type, canonical_moment_id
            FROM duplicates WHERE moment_id IN ({placeholders})""",
        moment_ids,
    ).fetchall()
    dup_map: dict[str, dict] = {}
    for r in dup_rows:
        dup_map[r["moment_id"]] = {
            "type": r["duplicate_type"],
            "canonical": r["canonical_moment_id"],
        }

    canonical_ids = set()
    for r in conn.execute(
        f"""SELECT DISTINCT canonical_moment_id
            FROM duplicates
            WHERE canonical_moment_id IN ({placeholders})
               OR moment_id IN ({placeholders})""",
        (*moment_ids, *moment_ids),
    ).fetchall():
        canonical_ids.add(r["canonical_moment_id"])

    dup_count_map: dict[str, int] = {}
    for r in conn.execute(
        f"""SELECT canonical_moment_id, COUNT(*) as cnt
            FROM duplicates GROUP BY canonical_moment_id
            HAVING canonical_moment_id IN ({placeholders})""",
        moment_ids,
    ).fetchall():
        dup_count_map[r["canonical_moment_id"]] = r["cnt"]

    graph_node_rows = conn.execute(
        f"""SELECT node_id, node_type, ref_id
            FROM graph_nodes WHERE ref_id IN ({placeholders})""",
        moment_ids,
    ).fetchall()
    node_map: dict[str, str] = {}
    for r in graph_node_rows:
        if r["ref_id"]:
            node_map[r["ref_id"]] = r["node_id"]

    edge_counts: dict[str, dict[str, int]] = {}
    for mid in moment_ids:
        node_id = node_map.get(mid)
        if node_id is None:
            edge_counts[mid] = {}
            continue
        rows = conn.execute(
            """SELECT edge_type, COUNT(*) as cnt
               FROM graph_edges
               WHERE source_node_id = ? OR target_node_id = ?
               GROUP BY edge_type""",
            (node_id, node_id),
        ).fetchall()
        edge_counts[mid] = {r["edge_type"]: r["cnt"] for r in rows}

    query_lower = query.lower()
    query_terms = query_lower.split()

    features: dict[str, dict[str, Any]] = {}
    for mid in moment_ids:
        row = moment_map.get(mid)
        if row is None:
            features[mid] = _empty_features()
            continue

        fts_transcript = (row["transcript_text"] or "").lower()
        fts_ocr = (row["ocr_text"] or "").lower()
        ideas_json = row["ideas"] or "[]"
        ideas_data = json.loads(ideas_json) if isinstance(ideas_json, str) else []
        idea_texts = [i.get("text", "") for i in ideas_data]
        embedded_types = [i.get("type", "") for i in ideas_data]
        db_types = idea_types_map.get(mid, [])
        idea_types = sorted(set(embedded_types + db_types))
        combined_ideas = " ".join(idea_texts).lower()
        keyframe_paths = json.loads(row["keyframe_paths"]) if isinstance(row["keyframe_paths"], str) else []

        ec = edge_counts.get(mid, {})
        dup_info = dup_map.get(mid, {})
        is_canonical = mid in canonical_ids
        is_exact_dup = dup_info.get("type") == "exact" if dup_info else False

        feat: dict[str, Any] = {
            "has_transcript": bool(fts_transcript.strip()),
            "has_ocr": bool(fts_ocr.strip()),
            "has_keyframe": bool(keyframe_paths),
            "idea_count": idea_counts.get(mid, 0),
            "evidence_count": evidence_by_moment.get(mid, 0),
            "modality_count": len(modality_by_moment.get(mid, set())),
            "modalities": sorted(modality_by_moment.get(mid, set())),
            "idea_types": sorted(set(idea_types)),
            "is_canonical": is_canonical,
            "is_exact_duplicate": is_exact_dup,
            "duplicate_type": dup_info.get("type", ""),
            "canonical_moment_id": dup_info.get("canonical", ""),
            "cluster_size": dup_count_map.get(mid, 0) + dup_count_map.get(dup_info.get("canonical", ""), 0),
            "degree": sum(ec.values()),
            "idea_degree": ec.get("idea_mentioned_in_moment", 0) + ec.get("moment_expresses_idea", 0),
            "entity_degree": ec.get("idea_mentions_entity", 0),
            "evidence_degree": ec.get("moment_has_evidence", 0) + ec.get("evidence_supports_idea", 0),
            "cluster_degree": ec.get("cluster_has_moment", 0) + ec.get("cluster_canonical_moment", 0),
            "duplicate_degree": ec.get("moment_duplicate_of", 0) + ec.get("moment_variant_of", 0) + ec.get("moment_same_idea_as", 0),
            "support_strength": ec.get("moment_has_evidence", 0) + ec.get("moment_expresses_idea", 0),
            "graph_context_available": len(ec) > 0,
            "query_in_title": query_lower in (row["video_title"] or "").lower(),
            "query_in_transcript": any(t in fts_transcript for t in query_terms),
            "query_in_ocr": any(t in fts_ocr for t in query_terms),
            "query_in_idea": any(t in combined_ideas for t in query_terms),
            "warning_match": any(t in fts_transcript or t in fts_ocr for t in QUERY_PATTERNS["warning"]),
            "example_match": any(t in fts_transcript or t in fts_ocr for t in QUERY_PATTERNS["example"]),
            "comparison_match": any(t in fts_transcript or t in fts_ocr for t in QUERY_PATTERNS["comparison"]),
            "code_like_match": any(t in fts_transcript or t in fts_ocr for t in QUERY_PATTERNS["code_like"]),
        }
        features[mid] = feat

    return features


def _empty_features() -> dict[str, Any]:
    return {
        "has_transcript": False,
        "has_ocr": False,
        "has_keyframe": False,
        "idea_count": 0,
        "evidence_count": 0,
        "modality_count": 0,
        "modalities": [],
        "idea_types": [],
        "is_canonical": False,
        "is_exact_duplicate": False,
        "duplicate_type": "",
        "canonical_moment_id": "",
        "cluster_size": 0,
        "degree": 0,
        "idea_degree": 0,
        "entity_degree": 0,
        "evidence_degree": 0,
        "cluster_degree": 0,
        "duplicate_degree": 0,
        "support_strength": 0,
        "graph_context_available": False,
        "query_in_title": False,
        "query_in_transcript": False,
        "query_in_ocr": False,
        "query_in_idea": False,
        "warning_match": False,
        "example_match": False,
        "comparison_match": False,
        "code_like_match": False,
    }
