from typing import Any, Optional

DEFAULT_WEIGHTS = {
    "transcript_mult": 1.0,
    "ocr_boost": 0.10,
    "keyframe_boost": 0.10,
    "idea_boost": 0.15,
    "per_idea_boost": 0.05,
    "max_idea_boost": 0.30,
    "evidence_boost": 0.05,
    "per_evidence_boost": 0.02,
    "max_evidence_boost": 0.20,
    "entity_boost": 0.05,
    "per_entity_boost": 0.03,
    "max_entity_boost": 0.15,
    "modality_boost": 0.10,
    "per_modality_boost": 0.05,
    "max_modality_boost": 0.15,
    "canonical_boost": 0.20,
    "exact_duplicate_penalty": 0.50,
    "variant_boost": 0.20,
    "title_boost": 0.30,
    "query_in_idea_boost": 0.10,
    "warning_boost": 0.10,
    "example_boost": 0.05,
    "comparison_boost": 0.05,
    "code_like_boost": 0.10,
    "support_strength_boost": 0.05,
    "max_support_boost": 0.20,
}

WEIGHT_DESCRIPTIONS = {
    "transcript_mult": "Base FTS score multiplier",
    "ocr_boost": "Has OCR text",
    "keyframe_boost": "Has keyframe",
    "idea_boost": "Has extracted ideas",
    "per_idea_boost": "Per extracted idea",
    "max_idea_boost": "Max idea boost cap",
    "evidence_boost": "Has evidence records",
    "per_evidence_boost": "Per evidence record",
    "max_evidence_boost": "Max evidence boost cap",
    "entity_boost": "Connected to entities",
    "per_entity_boost": "Per entity connection",
    "max_entity_boost": "Max entity boost cap",
    "modality_boost": "Multiple evidence modalities",
    "per_modality_boost": "Per modality",
    "max_modality_boost": "Max modality boost cap",
    "canonical_boost": "Canonical moment in duplicate cluster",
    "exact_duplicate_penalty": "Exact duplicate penalty",
    "variant_boost": "Variant preserved for diversity",
    "title_boost": "Query matched video title",
    "query_in_idea_boost": "Query matched idea text",
    "warning_boost": "Contains warning terms",
    "example_boost": "Contains example terms",
    "comparison_boost": "Contains comparison terms",
    "code_like_boost": "Contains code-like terms",
    "support_strength_boost": "Per evidence+idea connection",
    "max_support_boost": "Max support boost cap",
}


def compute_graph_score(
    features: dict[str, Any],
    weights: Optional[dict[str, float]] = None,
    diverse_mode: bool = False,
    include_duplicates: bool = False,
    fts_score: float = 0.0,
) -> tuple[float, list[str]]:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    reasons: list[str] = []

    graph_score = fts_score * w["transcript_mult"]

    if features.get("has_ocr"):
        graph_score += w["ocr_boost"]
        reasons.append("matched OCR")

    if features.get("has_keyframe"):
        graph_score += w["keyframe_boost"]
        reasons.append("has keyframe")

    idea_count = features.get("idea_count", 0)
    if idea_count > 0:
        graph_score += w["idea_boost"]
        idea_bonus = min(idea_count * w["per_idea_boost"], w["max_idea_boost"])
        graph_score += idea_bonus
        reasons.append(f"has {idea_count} extracted idea(s)")

    evidence_count = features.get("evidence_count", 0)
    if evidence_count > 0:
        graph_score += w["evidence_boost"]
        ev_bonus = min(evidence_count * w["per_evidence_boost"], w["max_evidence_boost"])
        graph_score += ev_bonus
        reasons.append(f"has {evidence_count} evidence record(s)")

    entity_degree = features.get("entity_degree", 0)
    if entity_degree > 0:
        ent_bonus = min(entity_degree * w["per_entity_boost"], w["max_entity_boost"])
        graph_score += ent_bonus
        reasons.append(f"connected to {entity_degree} entit(ies)")

    modality_count = features.get("modality_count", 0)
    if modality_count > 1:
        mod_bonus = min(modality_count * w["per_modality_boost"], w["max_modality_boost"])
        graph_score += mod_bonus
        reasons.append(f"{modality_count} evidence modalities")

    if features.get("is_canonical"):
        graph_score += w["canonical_boost"]
        reasons.append("canonical moment in duplicate cluster")

    if features.get("is_exact_duplicate") and not include_duplicates:
        graph_score -= w["exact_duplicate_penalty"]
        reasons.append("exact duplicate (penalized)")

    if features.get("is_exact_duplicate") and include_duplicates:
        pass

    if diverse_mode and features.get("duplicate_type") in ("variant", "same_idea"):
        graph_score += w["variant_boost"]
        reasons.append("variant preserved for diversity")

    if features.get("query_in_title"):
        graph_score += w["title_boost"]
        reasons.append("matched title")

    if features.get("query_in_idea"):
        graph_score += w["query_in_idea_boost"]
        reasons.append("matched idea text")

    if features.get("warning_match"):
        graph_score += w["warning_boost"]
        reasons.append("contains warning terms")

    if features.get("example_match"):
        graph_score += w["example_boost"]
        reasons.append("contains example terms")

    if features.get("comparison_match"):
        graph_score += w["comparison_boost"]
        reasons.append("contains comparison terms")

    if features.get("code_like_match"):
        graph_score += w["code_like_boost"]
        reasons.append("contains code-like terms")

    support = features.get("support_strength", 0)
    if support > 0:
        support_bonus = min(support * w["support_strength_boost"], w["max_support_boost"])
        graph_score += support_bonus
        reasons.append(f"{support} graph support connection(s)")

    return round(graph_score, 4), reasons
