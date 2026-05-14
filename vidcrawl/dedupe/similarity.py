from difflib import SequenceMatcher

from vidcrawl.dedupe.normalize import normalize_text


def jaccard_similarity(a: str, b: str) -> float:
    tokens_a = set(normalize_text(a).split())
    tokens_b = set(normalize_text(b).split())
    if not tokens_a and not tokens_b:
        return 1.0
    union = tokens_a | tokens_b
    if not union:
        return 1.0
    intersection = tokens_a & tokens_b
    return len(intersection) / len(union)


def ngram_similarity(a: str, b: str, n: int = 3) -> float:
    def _ngrams(s: str) -> set:
        s = normalize_text(s)
        if len(s) < n:
            return {s}
        return set(s[i : i + n] for i in range(len(s) - n + 1))

    grams_a = _ngrams(a)
    grams_b = _ngrams(b)
    if not grams_a and not grams_b:
        return 1.0
    union = grams_a | grams_b
    if not union:
        return 1.0
    intersection = grams_a & grams_b
    return len(intersection) / len(union)


def sequence_match_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def combined_similarity(a: str, b: str) -> float:
    jac = jaccard_similarity(a, b)
    ngram = ngram_similarity(a, b)
    seq = sequence_match_ratio(a, b)
    return (jac + ngram + seq) / 3.0
