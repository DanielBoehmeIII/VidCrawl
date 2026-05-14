import re
from typing import Any


def extract_ideas(text: str) -> list[dict[str, Any]]:
    if not text or not text.strip():
        return []

    sentences = _split_sentences(text)
    ideas = []
    seen = set()

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10:
            continue

        idea = _classify_sentence(sentence)
        if idea is None:
            continue

        dedup_key = (idea["idea_type"], sentence[:80])
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        idea["text"] = sentence
        ideas.append(idea)

    return ideas


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _classify_sentence(sentence: str) -> dict[str, Any] | None:
    lower = sentence.lower()

    result = _check_definition(lower, sentence)
    if result:
        return result

    result = _check_warning(lower, sentence)
    if result:
        return result

    result = _check_step(lower, sentence)
    if result:
        return result

    result = _check_example(lower, sentence)
    if result:
        return result

    result = _check_comparison(lower, sentence)
    if result:
        return result

    if _is_high_information(sentence):
        return {
            "idea_type": "claim",
            "confidence": 0.5,
            "source": "rule",
            "metadata_json": {"matched_rule": "high_information_fallback"},
        }

    return None


def _check_definition(lower: str, original: str) -> dict[str, Any] | None:
    patterns = [
        r"\bis\s+(?:a|an|the)\s+\w",
        r"\bmeans?\s+",
        r"\bdefined\s+as\b",
        r"\brefers?\s+to\b",
        r"\b(is|are)\s+known\s+as\b",
        r"\bwhat\s+(is|are)\s+\w+\s+(?:is|are)\b",
    ]
    for pat in patterns:
        if re.search(pat, lower):
            return {
                "idea_type": "definition",
                "confidence": 0.8,
                "source": "rule",
                "metadata_json": {"matched_rule": pat},
            }
    return None


def _check_step(lower: str, original: str) -> dict[str, Any] | None:
    patterns = [
        r"\bfirst(?:ly)?,",
        r"\bsecond(?:ly)?,",
        r"\bthird(?:ly)?,",
        r"\bnext,",
        r"\bthen\b",
        r"\bfinally,",
        r"\bclick\b",
        r"\brun\b",
        r"\binstall\b",
        r"\bcreate\b",
        r"\bopen\b",
        r"\bstart\b",
        r"\bchoose\b",
        r"\bselect\b",
        r"\btype\b",
        r"\bpress\b",
        r"\bstep\s+\d+\b",
    ]
    for pat in patterns:
        if re.search(pat, lower):
            return {
                "idea_type": "step",
                "confidence": 0.7,
                "source": "rule",
                "metadata_json": {"matched_rule": pat},
            }
    return None


def _check_warning(lower: str, original: str) -> dict[str, Any] | None:
    patterns = [
        r"\bbe\s+careful\b",
        r"\b(?:avoid|avoiding)\b",
        r"\bdo\s+not\b",
        r"\bdon't\b",
        r"\berror\b",
        r"\bfails?\b",
        r"\bproblem\b",
        r"\bwarning\b",
        r"\bcaution\b",
        r"\bincorrect\b",
        r"\bwrong\b",
        r"\bmistake\b",
        r"\b(?:make\s+)?sure\s+(?:you\s+)?(?:don't|do\s+not|avoid)\b",
    ]
    for pat in patterns:
        if re.search(pat, lower):
            return {
                "idea_type": "warning",
                "confidence": 0.8,
                "source": "rule",
                "metadata_json": {"matched_rule": pat},
            }
    return None


def _check_example(lower: str, original: str) -> dict[str, Any] | None:
    patterns = [
        r"\bfor\s+example\b",
        r"\be\.g\.",
        r"\bsuch\s+as\b",
        r"\blike\b",
        r"\bfor\s+instance\b",
        r"\bexample\b",
    ]
    for pat in patterns:
        if re.search(pat, lower):
            return {
                "idea_type": "example",
                "confidence": 0.7,
                "source": "rule",
                "metadata_json": {"matched_rule": pat},
            }
    return None


def _check_comparison(lower: str, original: str) -> dict[str, Any] | None:
    patterns = [
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bcompared?\s+to\b",
        r"\bbetter\s+than\b",
        r"\bworse\s+than\b",
        r"\bsimilar\s+to\b",
        r"\bdifferent\s+from\b",
        r"\binstead\s+of\b",
        r"\brather\s+than\b",
        r"\bon\s+the\s+(?:one|other)\s+hand\b",
        r"\b(?:more|less)\s+\w+\s+than\b",
    ]
    for pat in patterns:
        if re.search(pat, lower):
            return {
                "idea_type": "comparison",
                "confidence": 0.7,
                "source": "rule",
                "metadata_json": {"matched_rule": pat},
            }
    return None


def _is_high_information(sentence: str) -> bool:
    words = sentence.split()
    if len(words) < 5:
        return False
    if len(words) > 50:
        words = words[:50]
    unique_ratio = len(set(w.lower() for w in words)) / len(words)
    return unique_ratio > 0.6 and len(words) >= 8
