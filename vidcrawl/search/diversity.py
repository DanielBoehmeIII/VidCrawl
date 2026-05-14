from typing import Any, Optional

from vidcrawl.search.query import SearchResult


def select_diverse_results(
    results: list[SearchResult],
    features_map: dict[str, dict[str, Any]],
    limit: int,
    diversity_strength: float = 0.5,
) -> list[SearchResult]:
    if not results or len(results) <= 1:
        return results[:limit]

    scored: list[tuple[float, int, SearchResult]] = []
    for i, r in enumerate(results):
        feat = features_map.get(r.moment_id, {})
        relevance = r.score
        scored.append((relevance, i, r))

    scored.sort(key=lambda x: -x[0])
    selected: list[SearchResult] = []
    selected_ids: set[str] = set()
    selected_clusters: set[str] = set()
    selected_videos: set[str] = set()
    selected_idea_types: set[tuple[str, ...]] = set()

    for _ in range(limit * 2):
        if not scored:
            break

        best_idx = -1
        best_score = -1e9

        for i in range(len(scored)):
            rel, orig_idx, r = scored[i]
            feat = features_map.get(r.moment_id, {})

            penalty = 0.0

            canonical = feat.get("canonical_moment_id", "") or r.canonical_moment_id or ""
            if canonical and canonical in selected_ids:
                penalty += diversity_strength * 0.5
            elif canonical:
                cluster_key = canonical
                if cluster_key in selected_clusters:
                    penalty += diversity_strength * 0.4

            if r.video_id in selected_videos and len(selected_videos) > 0:
                penalty += diversity_strength * 0.1

            idea_types = tuple(sorted(feat.get("idea_types", [])))
            if idea_types in selected_idea_types and len(selected_idea_types) > 0:
                penalty += diversity_strength * 0.15

            adverse_score = rel - penalty

            if adverse_score > best_score:
                best_score = adverse_score
                best_idx = i

        if best_idx < 0:
            break

        _, _, r = scored.pop(best_idx)
        selected.append(r)
        selected_ids.add(r.moment_id)
        if r.video_id:
            selected_videos.add(r.video_id)
        canonical = (features_map.get(r.moment_id, {}) or {}).get("canonical_moment_id", "") or r.canonical_moment_id or ""
        if canonical:
            selected_clusters.add(canonical)
        feat = features_map.get(r.moment_id, {})
        idea_types = tuple(sorted(feat.get("idea_types", [])))
        if idea_types:
            selected_idea_types.add(idea_types)

        if len(selected) >= limit:
            break

    for i, r in enumerate(selected):
        r.rank = i + 1

    return selected
