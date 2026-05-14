import re
from typing import Optional

KNOWN_TERMS: set[str] = {
    "Playwright", "Selenium", "Tesseract", "Whisper", "SQLite", "FTS5",
    "MCP", "API", "OCR", "ASR", "REST", "JSON", "HTML", "CSS", "HTTP",
    "RNN", "LSTM", "CNN", "BERT", "GPT", "LLM", "MMR",
    "Transformer", "Attention", "Positional Encoding",
    "YouTube", "pytest", "npm", "node",
}

TECHNICAL_PACKAGES: set[str] = {
    "playwright", "pytesseract", "yt-dlp", "ffmpeg", "pydantic", "typer",
}


def extract_entities(
    *texts: str,
    known_terms: Optional[set[str]] = None,
) -> list[dict]:
    seen: set[str] = set()
    entities: list[dict] = []

    if known_terms is None:
        known_terms = KNOWN_TERMS

    combined = "\n".join(t for t in texts if t)

    for match in _find_capitalized_phrases(combined):
        key = match.lower()
        if key not in seen:
            seen.add(key)
            entities.append({
                "label": match,
                "entity_type": "capitalized_phrase",
                "source_text": match,
            })

    for match in _find_acronyms(combined):
        key = match.lower()
        if key not in seen:
            seen.add(key)
            entities.append({
                "label": match,
                "entity_type": "acronym",
                "source_text": match,
            })

    for match in _find_file_paths(combined):
        key = match.lower()
        if key not in seen:
            seen.add(key)
            entities.append({
                "label": match,
                "entity_type": "file_path",
                "source_text": match,
            })

    for match in _find_code_identifiers(combined):
        key = match.lower()
        if key not in seen:
            seen.add(key)
            entities.append({
                "label": match,
                "entity_type": "code_identifier",
                "source_text": match,
            })

    for term in known_terms:
        key = term.lower()
        if key in seen:
            continue
        pattern = re.compile(
            r'(?<![a-zA-Z])' + re.escape(term) + r'(?![a-zA-Z])',
            re.IGNORECASE,
        )
        if pattern.search(combined):
            seen.add(key)
            entities.append({
                "label": term,
                "entity_type": "known_term",
                "source_text": term,
            })

    for pkg in TECHNICAL_PACKAGES:
        key = pkg.lower()
        if key in seen:
            continue
        pattern = re.compile(
            r'(?<![a-zA-Z])' + re.escape(pkg) + r'(?![a-zA-Z])',
            re.IGNORECASE,
        )
        if pattern.search(combined):
            seen.add(key)
            entities.append({
                "label": pkg if pkg[0].isupper() else pkg,
                "entity_type": "package",
                "source_text": pkg,
            })

    return entities


def _find_capitalized_phrases(text: str) -> set[str]:
    phrases: set[str] = set()
    matches = re.findall(r'[A-Z][a-zA-Z]*\s[A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*', text)
    for m in matches:
        m = m.strip()
        if len(m) >= 4 and not m.endswith(('.', ',')):
            phrases.add(m)
    return phrases


def _find_acronyms(text: str) -> set[str]:
    acronyms: set[str] = set()
    matches = re.findall(r'\b[A-Z]{2,6}\b', text)
    for m in matches:
        if len(m) >= 2:
            acronyms.add(m)
    return acronyms


def _find_file_paths(text: str) -> set[str]:
    paths: set[str] = set()
    matches = re.findall(r'(?:/[a-zA-Z0-9_.-]+)+(?:\.[a-zA-Z0-9]+)?', text)
    for m in matches:
        m = m.strip()
        if len(m) >= 3 and not m.endswith(('.', ',', ')')):
            paths.add(m)
    return paths


def _find_code_identifiers(text: str) -> set[str]:
    ids: set[str] = set()
    patterns = [
        r'\b[A-Z][a-zA-Z0-9]*\([^)]*\)',  # function calls like Foo()
        r'\b[a-z][a-zA-Z0-9]*\.[a-z][a-zA-Z0-9]+',  # method calls like page.goto
        r'[a-z][a-zA-Z0-9]*(?:\.[a-zA-Z][a-zA-Z0-9]*)+',  # dotted paths
    ]
    for pat in patterns:
        matches = re.findall(pat, text)
        for m in matches:
            m = m.strip().rstrip(')')
            if len(m) >= 4:
                ids.add(m)
    return ids
