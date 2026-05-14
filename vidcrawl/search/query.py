import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from vidcrawl.utils.time import timestamp_range, youtube_timestamp_url


@dataclass
class SearchResult:
    moment_id: str
    video_id: str
    video_title: str
    source_url: Optional[str] = None
    start_sec: float = 0.0
    end_sec: float = 0.0
    timestamp_label: str = ""
    transcript_snippet: str = ""
    ocr_snippet: str = ""
    idea_summary: str = ""
    idea_types: list[str] = field(default_factory=list)
    keyframe_paths: list[str] = field(default_factory=list)
    score: float = 0.0
    raw_score: float = 0.0
    rank: int = 0
    match_reasons: list[str] = field(default_factory=list)
    is_duplicate: bool = False
    is_variant: bool = False
    canonical_moment_id: Optional[str] = None
    collapsed_count: int = 0
    metadata_json: Optional[dict] = None
    graph_score: Optional[float] = None
    final_score: Optional[float] = None
    ranking_reasons: list[str] = field(default_factory=list)
    features_json: Optional[dict] = None
    cluster_id: Optional[str] = None
    collapsed_duplicate_count: int = 0
    related_variant_count: int = 0


def search_moments(
    query: str,
    db_path: Path,
    limit: int = 10,
    video_id: Optional[str] = None,
    include_duplicates: bool = False,
    diverse: bool = False,
    use_rerank: bool = True,
    search_mode: str = "fts",
) -> list[SearchResult]:
    if not query or not query.strip():
        return []

    db_path = Path(db_path)
    if not db_path.exists():
        return []

    db_str = str(db_path)

    if search_mode == "semantic":
        return _semantic_search_only(query, db_str, limit, video_id)

    if search_mode == "hybrid":
        return _hybrid_search(
            query, db_str, limit, video_id,
            include_duplicates, diverse, use_rerank,
        )

    fts_query = _sanitize_fts_query(query)
    if not fts_query:
        if search_mode == "fts_semantic":
            return _semantic_search_only(query, db_str, limit, video_id)
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = _execute_search(conn, fts_query, limit * 3, video_id)
        if rows is None:
            phrase = _make_phrase_query(query)
            rows = _execute_search(conn, phrase, limit * 3, video_id)
        if rows is None:
            if search_mode == "fts_semantic":
                conn.close()
                return _semantic_search_only(query, db_str, limit, video_id)
            return []
        results = _build_results(rows, query)
        results = _filter_duplicates(
            results, conn, include_duplicates, diverse, limit
        )

        if search_mode == "fts_semantic":
            semantic_results = _semantic_search_only(
                query, db_str, limit * 2, video_id,
            )
            existing_ids = {r.moment_id for r in results}
            for sr in semantic_results:
                if sr.moment_id not in existing_ids:
                    results.append(sr)
                    existing_ids.add(sr.moment_id)

        if use_rerank and _graph_has_data(conn):
            results = _apply_reranking(
                results, conn, db_path, query,
                diverse=diverse,
                include_duplicates=include_duplicates,
                limit=limit,
            )
        else:
            results.sort(key=lambda r: -r.score)
            for i, r in enumerate(results):
                r.rank = i + 1
            results = results[:limit]

        return results
    finally:
        conn.close()


def _semantic_search_only(
    query: str,
    db_str: str,
    limit: int = 10,
    video_id: Optional[str] = None,
) -> list[SearchResult]:
    from vidcrawl.embeddings.store import get_all_vectors, get_embedding_stats, has_embeddings
    from vidcrawl.embeddings.similarity import cosine_similarity_matrix
    from vidcrawl.embeddings.provider import get_provider

    if not has_embeddings(db_str):
        return []

    stats = get_embedding_stats(db_str)
    if not stats["has_embeddings"]:
        return []

    provider = get_provider(stats["provider"].split(":")[0] if ":" not in stats["provider"] else stats["provider"].split(":", 1)[0])
    query_vec = provider.compute([query])[0]
    if not query_vec:
        return []

    vectors = get_all_vectors(get_db(db_str), "moment")
    if not vectors:
        return []

    scored = cosine_similarity_matrix(query_vec, vectors, top_k=limit)

    conn = get_db(db_str)
    try:
        from vidcrawl.search.query import _build_single_result
        results = []
        for moment_id, sim in scored:
            row = conn.execute(
                """SELECT m.moment_id, m.video_id, m.start_sec, m.end_sec,
                          m.transcript_text, m.ocr_text, m.ideas, m.keyframe_paths,
                          COALESCE(v.title, '') as video_title, v.url as source_url
                   FROM moments m
                   JOIN videos v ON m.video_id = v.video_id
                   WHERE m.moment_id = ?""",
                (moment_id,),
            ).fetchone()
            if row is None:
                continue
            r = _build_single_result(row, query)
            r.score = round(sim, 4)
            r.match_reasons.append("semantic")
            results.append(r)

        results.sort(key=lambda r: -r.score)
        for i, r in enumerate(results):
            r.rank = i + 1
        return results[:limit]
    finally:
        conn.close()


def _hybrid_search(
    query: str,
    db_str: str,
    limit: int,
    video_id: Optional[str] = None,
    include_duplicates: bool = False,
    diverse: bool = False,
    use_rerank: bool = True,
) -> list[SearchResult]:
    fts_results = search_moments(
        query, Path(db_str), limit=limit * 2, video_id=video_id,
        include_duplicates=True, diverse=False, use_rerank=False,
        search_mode="fts",
    )
    semantic_results = _semantic_search_only(
        query, db_str, limit=limit * 2, video_id=video_id,
    )

    seen: set[str] = set()
    merged: list = []
    for r in fts_results + semantic_results:
        if r.moment_id not in seen:
            seen.add(r.moment_id)
            merged.append(r)

    merged.sort(key=lambda r: -r.score)

    conn = get_db(db_str)
    try:
        merged = _filter_duplicates(merged, conn, include_duplicates, diverse, limit * 2)

        if use_rerank and _graph_has_data(conn):
            merged = _apply_reranking(
                merged, conn, Path(db_str), query,
                diverse=diverse,
                include_duplicates=include_duplicates,
                limit=limit,
            )

        merged = merged[:limit]
        for i, r in enumerate(merged):
            r.rank = i + 1
        return merged
    finally:
        conn.close()


def _graph_has_data(conn: sqlite3.Connection) -> bool:
    try:
        row = conn.execute(
            "SELECT COUNT(*) as c FROM graph_nodes"
        ).fetchone()
        return row is not None and row["c"] > 0
    except Exception:
        return False


def _apply_reranking(
    results: list[SearchResult],
    conn: sqlite3.Connection,
    db_path: Path,
    query: str,
    diverse: bool = False,
    include_duplicates: bool = False,
    limit: int = 10,
) -> list[SearchResult]:
    from vidcrawl.search.features import _compute_features_batch
    from vidcrawl.search.rerank import compute_graph_score
    from vidcrawl.search.diversity import select_diverse_results

    moment_ids = [r.moment_id for r in results]
    features_map = _compute_features_batch(conn, moment_ids, query)

    for r in results:
        feat = features_map.get(r.moment_id, {})
        r.features_json = feat
        gs, reasons = compute_graph_score(
            feat,
            fts_score=r.score,
            diverse_mode=diverse,
            include_duplicates=include_duplicates,
        )
        r.graph_score = round(gs, 4)
        r.final_score = round(gs, 4)
        r.ranking_reasons = reasons
        r.score = round(gs, 4)

    results.sort(key=lambda r: -r.score)
    for i, r in enumerate(results):
        r.rank = i + 1

    if diverse:
        results = select_diverse_results(
            results, features_map, limit, diversity_strength=0.5
        )

    return results[:limit]


def _sanitize_fts_query(query: str) -> str:
    q = query.strip()
    if not q:
        return ""
    q = q.replace("\0", "")
    if len(q) > 500:
        q = q[:500]
    return q


def _make_phrase_query(query: str) -> str:
    escaped = query.strip().replace('"', '""')
    return f'"{escaped}"'


def _execute_search(
    conn: sqlite3.Connection,
    fts_query: str,
    limit: int,
    video_id: Optional[str] = None,
):
    try:
        sql = """
            SELECT
                sub.rank as raw_score,
                m.moment_id,
                m.video_id,
                COALESCE(v.title, '') as video_title,
                v.url as source_url,
                m.start_sec,
                m.end_sec,
                COALESCE(m.transcript_text, '') as transcript_text,
                COALESCE(m.ocr_text, '') as ocr_text,
                COALESCE(m.ideas, '[]') as ideas_json,
                COALESCE(m.keyframe_paths, '[]') as keyframe_paths_json
            FROM (
                SELECT rowid, rank FROM moments_fts WHERE moments_fts MATCH ?
            ) sub
            JOIN moments m ON sub.rowid = m.rowid
            JOIN videos v ON m.video_id = v.video_id
        """
        params: list[Any] = [fts_query]
        if video_id:
            sql += " WHERE m.video_id = ?"
            params.append(video_id)
        sql += " ORDER BY sub.rank LIMIT ?"
        params.append(limit)
        return conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return None


def _build_results(
    rows: list[sqlite3.Row], query: str
) -> list[SearchResult]:
    query_terms = query.lower().split()
    results: list[SearchResult] = []

    for row in rows:
        ideas_data = json.loads(row["ideas_json"]) if row["ideas_json"] else []
        keyframe_paths = (
            json.loads(row["keyframe_paths_json"])
            if row["keyframe_paths_json"]
            else []
        )
        idea_types = sorted(
            set(i.get("type", "") for i in ideas_data if i.get("type"))
        )
        idea_parts = [
            f"[{i.get('type', '')}] {i.get('text', '')[:80]}"
            for i in ideas_data[:3]
        ]
        idea_summary = "; ".join(idea_parts)

        transcript_snippet = _extract_snippet(
            row["transcript_text"], query_terms
        )
        ocr_snippet = _extract_snippet(row["ocr_text"], query_terms)

        raw_score = row["raw_score"] if row["raw_score"] is not None else 0.0

        result = SearchResult(
            moment_id=row["moment_id"],
            video_id=row["video_id"],
            video_title=row["video_title"],
            source_url=row["source_url"],
            start_sec=row["start_sec"],
            end_sec=row["end_sec"],
            timestamp_label=timestamp_range(
                row["start_sec"], row["end_sec"]
            ),
            transcript_snippet=transcript_snippet,
            ocr_snippet=ocr_snippet,
            idea_summary=idea_summary,
            idea_types=idea_types,
            keyframe_paths=keyframe_paths,
            score=0.0,
            raw_score=raw_score,
            rank=0,
        )
        results.append(result)

    results = _apply_boosts(results, query)
    results.sort(key=lambda r: r.score, reverse=True)
    for i, r in enumerate(results):
        r.rank = i + 1
    return results


def _build_single_result(row, query: str) -> SearchResult:
    query_terms = query.lower().split()
    ideas_data = json.loads(row["ideas"]) if isinstance(row["ideas"], str) and row["ideas"] else []
    keyframe_paths = (
        json.loads(row["keyframe_paths"]) if isinstance(row["keyframe_paths"], str) and row["keyframe_paths"]
        else []
    )
    idea_types = sorted(set(i.get("type", "") for i in ideas_data if i.get("type")))
    idea_parts = [f"[{i.get('type', '')}] {i.get('text', '')[:80]}" for i in ideas_data[:3]]
    idea_summary = "; ".join(idea_parts)
    transcript_snippet = _extract_snippet(row["transcript_text"], query_terms)
    ocr_snippet = _extract_snippet(row["ocr_text"], query_terms)
    raw_score = float(row["raw_score"]) if "raw_score" in row and row["raw_score"] is not None else 0.0
    return SearchResult(
        moment_id=row["moment_id"],
        video_id=row["video_id"],
        video_title=row["video_title"],
        source_url=row.get("source_url") or None,
        start_sec=row["start_sec"],
        end_sec=row["end_sec"],
        timestamp_label=timestamp_range(row["start_sec"], row["end_sec"]),
        transcript_snippet=transcript_snippet,
        ocr_snippet=ocr_snippet,
        idea_summary=idea_summary,
        idea_types=idea_types,
        keyframe_paths=keyframe_paths,
        score=0.0,
        raw_score=raw_score,
        rank=0,
    )


def _filter_duplicates(
    results: list[SearchResult],
    conn: sqlite3.Connection,
    include_duplicates: bool,
    diverse: bool,
    limit: int,
) -> list[SearchResult]:
    if not results:
        return results

    moment_ids = [r.moment_id for r in results]

    placeholders = ",".join("?" * len(moment_ids))
    dup_rows = conn.execute(
        f"""SELECT moment_id, canonical_moment_id, duplicate_type
            FROM duplicates
            WHERE moment_id IN ({placeholders})""",
        moment_ids,
    ).fetchall()

    dup_map = {
        r["moment_id"]: {
            "canonical": r["canonical_moment_id"],
            "type": r["duplicate_type"],
        }
        for r in dup_rows
    }

    canonical_ids = set(
        r["canonical_moment_id"] for r in dup_rows
    )
    result_ids = set(moment_ids)
    total_before = len(results)
    filtered = []
    collapsed = 0

    for r in results:
        info = dup_map.get(r.moment_id)
        if info is None:
            filtered.append(r)
            continue

        if info["type"] == "exact" and not include_duplicates:
            canonical_in_results = (
                info["canonical"] in result_ids
                or info["canonical"] in [fr.moment_id for fr in filtered]
            )
            if canonical_in_results:
                collapsed += 1
                continue
            r.is_duplicate = True
            r.canonical_moment_id = info["canonical"]
            filtered.append(r)
        elif info["type"] in ("near_text", "same_idea") and not include_duplicates:
            if not diverse:
                canonical_in_results = (
                    info["canonical"] in result_ids
                    or info["canonical"] in [fr.moment_id for fr in filtered]
                )
                if canonical_in_results:
                    collapsed += 1
                    continue
            r.is_duplicate = True
            r.is_variant = info["type"] == "same_idea"
            r.canonical_moment_id = info["canonical"]
            filtered.append(r)
        elif info["type"] == "variant" and not include_duplicates:
            if not diverse:
                canonical_in_results = (
                    info["canonical"] in result_ids
                    or info["canonical"] in [fr.moment_id for fr in filtered]
                )
                if canonical_in_results:
                    collapsed += 1
                    continue
            r.is_variant = True
            r.canonical_moment_id = info["canonical"]
            filtered.append(r)
        else:
            filtered.append(r)

    if filtered:
        filtered[0].collapsed_count = collapsed

    if len(filtered) > limit:
        filtered = filtered[:limit]

    return filtered


def _extract_snippet(text: str, query_terms: list[str], max_chars: int = 240) -> str:
    if not text:
        return ""
    if not query_terms:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."

    lower_text = text.lower()
    best_pos = -1
    for term in query_terms:
        if len(term) < 2:
            continue
        pos = lower_text.find(term.lower())
        if pos >= 0 and (best_pos < 0 or pos < best_pos):
            best_pos = pos

    if best_pos < 0:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."

    half = max_chars // 2
    start = max(0, best_pos - half)
    end = min(len(text), best_pos + half)
    snippet = text[start:end]
    if start > 0:
        snippet = "... " + snippet
    if end < len(text):
        snippet = snippet + " ..."
    return snippet


def _apply_boosts(
    results: list[SearchResult], query: str
) -> list[SearchResult]:
    if not results:
        return results

    query_lower = query.lower().strip()
    query_terms = query_lower.split()

    for r in results:
        fts_score = -r.raw_score
        if fts_score < 0:
            fts_score = 0.0

        boost = 0.0
        reasons = []

        has_ocr_match = any(t in r.ocr_snippet.lower() for t in query_terms)
        has_transcript_match = any(
            t in r.transcript_snippet.lower() for t in query_terms
        )
        has_idea_match = any(t in r.idea_summary.lower() for t in query_terms)
        has_title_match = any(t in r.video_title.lower() for t in query_terms)
        has_keyframe = bool(r.keyframe_paths)
        has_idea = bool(r.idea_types)

        if has_transcript_match:
            boost += 0.5
            reasons.append("transcript")
        if has_ocr_match:
            boost += 0.3
            reasons.append("OCR")
        if has_idea_match:
            boost += 0.2
            reasons.append("idea")
        if has_title_match:
            boost += 0.5
            reasons.append("title")
        if has_keyframe:
            boost += 0.2
            reasons.append("keyframes")
        if has_idea:
            boost += 0.1
            reasons.append("has_ideas")

        r.raw_score = round(fts_score, 4)
        r.score = round(fts_score + boost, 4)
        r.match_reasons = reasons

    return results
