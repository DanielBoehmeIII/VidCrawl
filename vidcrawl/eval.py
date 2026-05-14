import json
import time
from pathlib import Path
from typing import Any, Optional

from vidcrawl.db import get_db
from vidcrawl.search.query import search_moments


EVAL_QUERIES_DEMO = {
    "queries": [
        {
            "query": "playwright browser",
            "expected_terms": ["playwright", "browser"],
            "expected_video_id": "demo_coding",
            "expected_moment_contains": "MCP",
        },
        {
            "query": "install npm",
            "expected_terms": ["install"],
            "expected_video_id": "demo_coding",
        },
        {
            "query": "warning headless",
            "expected_terms": ["warning", "headless"],
            "expected_video_id": "demo_coding",
        },
        {
            "query": "transformer attention",
            "expected_terms": ["transformer", "attention"],
            "expected_video_id": "demo_ml",
        },
        {
            "query": "user research interview",
            "expected_terms": ["user", "research"],
            "expected_video_id": "demo_ux",
        },
        {
            "query": "comparison",
            "expected_terms": ["comparison"],
        },
        {
            "query": "definition",
            "expected_terms": ["definition"],
        },
        {
            "query": "step",
            "expected_terms": ["step"],
        },
        {
            "query": "model architecture",
            "expected_terms": ["model", "architecture"],
            "expected_video_id": "demo_ml",
        },
        {
            "query": "error handling",
            "expected_terms": ["error"],
        },
    ]
}


def load_queries(path: Optional[str] = None) -> list[dict]:
    if path:
        with open(path) as f:
            data = json.load(f)
            return data.get("queries", [])
    return EVAL_QUERIES_DEMO.get("queries", [])


def compute_graph_metrics(db_path: Path) -> dict[str, Any]:
    try:
        from vidcrawl.graph.stats import compute_graph_stats
        stats = compute_graph_stats(db_path)
        total = stats.total_nodes or 1
        idea_count = stats.nodes_by_type.get("idea", 0)
        evidence_count = stats.nodes_by_type.get("evidence", 0)
        moment_count = stats.nodes_by_type.get("moment", 0)
        entity_count = stats.nodes_by_type.get("entity", 0)
        cluster_count = stats.nodes_by_type.get("duplicate_cluster", 0)

        pct_moments_with_ideas = 0.0
        pct_moments_with_evidence = 0.0
        pct_ideas_with_entities = 0.0

        if moment_count > 0:
            conn = get_db(db_path)
            try:
                moments_with_ideas = conn.execute(
                    "SELECT COUNT(DISTINCT i.moment_id) FROM ideas i "
                    "JOIN moments m ON i.moment_id = m.moment_id"
                ).fetchone()[0] or 0
                pct_moments_with_ideas = round(moments_with_ideas / moment_count * 100, 1)

                moments_with_evidence = conn.execute(
                    "SELECT COUNT(DISTINCT e.moment_id) FROM modal_evidence e "
                    "JOIN moments m ON e.moment_id = m.moment_id"
                ).fetchone()[0] or 0
                pct_moments_with_evidence = round(moments_with_evidence / moment_count * 100, 1)

                if idea_count > 0:
                    ideas_with_entities = conn.execute(
                        "SELECT COUNT(*) as c FROM graph_edges WHERE edge_type = 'idea_mentions_entity'"
                    ).fetchone()["c"]
                    pct_ideas_with_entities = round(ideas_with_entities / idea_count * 100, 1)
            finally:
                conn.close()

        return {
            "graph_node_count": stats.total_nodes,
            "graph_edge_count": stats.total_edges,
            "average_degree": stats.average_degree,
            "entity_nodes": entity_count,
            "duplicate_clusters": cluster_count,
            "pct_moments_with_ideas": pct_moments_with_ideas,
            "pct_moments_with_evidence": pct_moments_with_evidence,
            "pct_ideas_with_entities": pct_ideas_with_entities,
        }
    except Exception:
        return {}


def evaluate_queries(
    db_path: Path,
    queries: list[dict],
    limit: int = 10,
    include_duplicates: bool = False,
    use_rerank: bool = True,
    diverse: bool = False,
) -> dict[str, Any]:
    total_queries = len(queries)
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    video_hits = 0
    term_hits = 0
    total_moment_hits = 0
    total_latency = 0.0
    term_checks = 0
    total_collapsed = 0
    total_unique_videos = 0
    total_unique_clusters = 0
    total_duplicate_in_results = 0
    total_entity_coverage = 0
    total_idea_diversity = 0

    has_rerank_metrics = False

    for q in queries:
        query_text = q["query"]
        expected_terms = q.get("expected_terms", [])
        expected_video = q.get("expected_video_id")
        expected_contains = q.get("expected_moment_contains")

        start = time.time()
        results = search_moments(
            query_text, db_path, limit=limit,
            include_duplicates=include_duplicates,
            use_rerank=use_rerank,
            diverse=diverse,
        )
        elapsed = time.time() - start

        total_latency += elapsed
        total_moment_hits += len(results)
        if results:
            total_collapsed += results[0].collapsed_count
            videos_in_results = set(r.video_id for r in results if r.video_id)
            total_unique_videos += len(videos_in_results)
            clusters_in_results = set(
                r.canonical_moment_id for r in results if r.canonical_moment_id
            )
            total_unique_clusters += len(clusters_in_results)
            dup_in_results = sum(1 for r in results if r.is_duplicate or r.is_variant)
            total_duplicate_in_results += dup_in_results
            entity_counts = []
            for r in results:
                if r.features_json:
                    entity_counts.append(r.features_json.get("entity_degree", 0))
            total_entity_coverage += sum(entity_counts)
            idea_type_sets = []
            for r in results:
                if r.features_json:
                    idea_type_sets.extend(r.features_json.get("idea_types", []))
            total_idea_diversity += len(set(idea_type_sets))
            if r.graph_score is not None:
                has_rerank_metrics = True

        if results:
            top1_hits += 1
            if len(results) >= 3:
                top3_hits += 1
            if len(results) >= 5:
                top5_hits += 1

            if expected_contains:
                for r in results:
                    if expected_contains.lower() in (
                        r.transcript_snippet + r.ocr_snippet + r.idea_summary
                    ).lower():
                        total_moment_hits += 0
                        break

            for r in results:
                if expected_video and r.video_id == expected_video:
                    video_hits += 1
                    break

        for term in expected_terms:
            term_checks += 1
            for r in results:
                combined = (
                    r.transcript_snippet + r.ocr_snippet + r.idea_summary
                ).lower()
                if term.lower() in combined:
                    term_hits += 1
                    break

    avg_latency = total_latency / total_queries if total_queries else 0
    avg_results = total_moment_hits / total_queries if total_queries else 0

    result: dict[str, Any] = {
        "total_queries": total_queries,
        "top1_hits": top1_hits,
        "top1_pct": round(top1_hits / total_queries * 100, 1) if total_queries else 0,
        "top3_hits": top3_hits,
        "top3_pct": round(top3_hits / total_queries * 100, 1) if total_queries else 0,
        "top5_hits": top5_hits,
        "top5_pct": round(top5_hits / total_queries * 100, 1) if total_queries else 0,
        "video_hits": video_hits,
        "term_hits": term_hits,
        "term_checks": term_checks,
        "term_pct": round(term_hits / term_checks * 100, 1) if term_checks else 0,
        "avg_results_returned": round(avg_results, 2),
        "avg_query_latency_ms": round(avg_latency * 1000, 2),
        "total_latency_ms": round(total_latency * 1000, 2),
        "avg_duplicates_collapsed": round(total_collapsed / total_queries, 2) if total_queries else 0,
        "total_collapsed": total_collapsed,
    }

    if has_rerank_metrics and total_queries:
        result.update({
            "avg_unique_videos_in_topk": round(total_unique_videos / total_queries, 2),
            "avg_unique_clusters_in_topk": round(total_unique_clusters / total_queries, 2),
            "avg_duplicate_in_topk": round(total_duplicate_in_results / total_queries, 2),
            "avg_entity_coverage_in_topk": round(total_entity_coverage / total_queries, 2),
            "avg_idea_type_diversity_in_topk": round(total_idea_diversity / total_queries, 2),
        })

    graph_metrics = compute_graph_metrics(db_path)
    result.update(graph_metrics)

    return result


def format_eval_report(metrics: dict[str, Any]) -> str:
    lines = [
        "Evaluation Results",
        "==================",
        f"Total queries:       {metrics['total_queries']}",
        f"Top-1 hits:          {metrics['top1_hits']}/{metrics['total_queries']} ({metrics['top1_pct']}%)",
        f"Top-3 hits:          {metrics['top3_hits']}/{metrics['total_queries']} ({metrics['top3_pct']}%)",
        f"Top-5 hits:          {metrics['top5_hits']}/{metrics['total_queries']} ({metrics['top5_pct']}%)",
        f"Expected video hits: {metrics['video_hits']}",
        f"Expected term hits:  {metrics['term_hits']}/{metrics['term_checks']} ({metrics['term_pct']}%)",
        f"Avg results/query:   {metrics['avg_results_returned']}",
        f"Avg query latency:   {metrics['avg_query_latency_ms']}ms",
        f"Total latency:       {metrics['total_latency_ms']}ms",
    ]
    if "avg_duplicates_collapsed" in metrics:
        lines.extend([
            "",
            "Redundancy Metrics",
            "------------------",
            f"Avg duplicates collapsed/query: {metrics['avg_duplicates_collapsed']}",
            f"Total collapsed:                {metrics['total_collapsed']}",
        ])
    if "graph_node_count" in metrics and metrics["graph_node_count"] > 0:
        lines.extend([
            "",
            "Graph Metrics",
            "-------------",
            f"Graph nodes:            {metrics['graph_node_count']}",
            f"Graph edges:            {metrics['graph_edge_count']}",
            f"Average degree:         {metrics['average_degree']:.2f}",
            f"Entity nodes:           {metrics['entity_nodes']}",
            f"Duplicate clusters:     {metrics['duplicate_clusters']}",
            f"Moments with ideas:     {metrics['pct_moments_with_ideas']}%",
            f"Moments with evidence:  {metrics['pct_moments_with_evidence']}%",
            f"Ideas with entities:    {metrics['pct_ideas_with_entities']}%",
        ])

    if "avg_unique_videos_in_topk" in metrics:
        lines.extend([
            "",
            "Ranking/Diversity Metrics",
            "-------------------------",
            f"Avg unique videos in top-k:      {metrics['avg_unique_videos_in_topk']}",
            f"Avg unique clusters in top-k:    {metrics['avg_unique_clusters_in_topk']}",
            f"Avg duplicates in top-k:         {metrics['avg_duplicate_in_topk']}",
            f"Avg entity coverage in top-k:    {metrics['avg_entity_coverage_in_topk']}",
            f"Avg idea type diversity top-k:   {metrics['avg_idea_type_diversity_in_topk']}",
        ])
    return "\n".join(lines)
