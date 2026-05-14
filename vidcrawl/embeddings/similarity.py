import math
from typing import Optional


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0
    dot = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = math.sqrt(sum(ai * ai for ai in a))
    norm_b = math.sqrt(sum(bi * bi for bi in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_similarity_matrix(
    query_vec: list[float],
    vectors: dict[str, list[float]],
    top_k: Optional[int] = None,
) -> list[tuple[str, float]]:
    scores: list[tuple[str, float]] = []
    for item_id, vec in vectors.items():
        sim = cosine_similarity(query_vec, vec)
        scores.append((item_id, sim))
    scores.sort(key=lambda x: -x[1])
    if top_k:
        scores = scores[:top_k]
    return scores
