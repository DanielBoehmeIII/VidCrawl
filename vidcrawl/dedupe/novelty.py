import re

from vidcrawl.dedupe.normalize import normalize_text


def score_novelty(
    candidate_text: str,
    canonical_text: str,
    candidate_ocr: str = "",
    canonical_ocr: str = "",
    candidate_idea_types: list[str] = None,
    canonical_idea_types: list[str] = None,
    candidate_ideas_text: str = "",
    canonical_ideas_text: str = "",
) -> dict:
    candidate_idea_types = candidate_idea_types or []
    canonical_idea_types = canonical_idea_types or []

    sim_score = 0.0
    novelty_score = 0.0
    reasons = []

    norm_cand = normalize_text(candidate_text)
    norm_canon = normalize_text(canonical_text)

    cand_tokens = set(norm_cand.split())
    canon_tokens = set(norm_canon.split())
    new_tokens = cand_tokens - canon_tokens

    if cand_tokens and canon_tokens:
        overlap = cand_tokens & canon_tokens
        sim_score = len(overlap) / len(cand_tokens | canon_tokens)

    for token in new_tokens:
        if _is_warning_term(token):
            novelty_score += 0.15
            if "adds warning terms" not in reasons:
                reasons.append("adds warning terms")
        if _is_example_term(token):
            novelty_score += 0.10
            if "adds example terms" not in reasons:
                reasons.append("adds example terms")
        if _is_comparison_term(token):
            novelty_score += 0.10
            if "adds comparison terms" not in reasons:
                reasons.append("adds comparison terms")
        if _is_code_term(token):
            novelty_score += 0.10
            if "adds code/command terms" not in reasons:
                reasons.append("adds code/command terms")

    if candidate_ocr and candidate_ocr != canonical_ocr:
        novelty_score += 0.15
        reasons.append("different OCR text")

    ocr_tokens = set(normalize_text(candidate_ocr).split()) - set(
        normalize_text(canonical_ocr).split()
    )
    for token in ocr_tokens:
        if _is_code_term(token):
            novelty_score += 0.10
            if "OCR has code terms" not in reasons:
                reasons.append("OCR has code terms")
            break

    text_len_ratio = (
        len(norm_cand) / max(len(norm_canon), 1)
        if norm_canon
        else 0
    )
    if text_len_ratio > 1.3:
        novelty_score += 0.10
        if "longer explanation" not in reasons:
            reasons.append("longer explanation")

    cand_idea_set = set(candidate_idea_types)
    canon_idea_set = set(canonical_idea_types)
    new_idea_types = cand_idea_set - canon_idea_set
    if new_idea_types:
        novelty_score += 0.15
        reasons.append(
            f"new idea type: {', '.join(sorted(new_idea_types))}"
        )

    if candidate_ideas_text and candidate_ideas_text != canonical_ideas_text:
        novelty_score += 0.10
        if "different idea text" not in reasons:
            reasons.append("different idea text")

    novelty_score = min(novelty_score, 1.0)
    reason_str = ", ".join(reasons[:3]) if reasons else "similar content"

    return {
        "similarity_score": round(sim_score, 4),
        "novelty_score": round(novelty_score, 4),
        "reason": reason_str,
    }


def _is_warning_term(token: str) -> bool:
    return token in {
        "warning", "careful", "avoid", "error", "fail", "problem",
        "caution", "incorrect", "wrong", "mistake", "dont", "not",
    }


def _is_example_term(token: str) -> bool:
    return token in {
        "example", "instance", "sample", "demo", "illustrate",
    }


def _is_comparison_term(token: str) -> bool:
    return token in {
        "versus", "compared", "better", "worse", "similar",
        "different", "instead", "rather", "faster", "slower",
    }


def _is_code_term(token: str) -> bool:
    if re.match(r"^[a-z_][a-z0-9_]{2,}$", token):
        if token in {
            "install", "config", "setup", "import", "export",
            "return", "class", "function", "method", "define",
            "create", "build", "run", "cmd", "npm", "pip",
        }:
            return True
        if re.match(r"^[a-z_]+\(\)$", token):
            return True
        if re.match(r"^[a-z_]+\.\w+", token):
            return True
    return False
