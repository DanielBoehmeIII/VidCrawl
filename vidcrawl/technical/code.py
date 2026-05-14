import re
from typing import Optional


FILE_PATH_PATTERNS = [
    r'(?:/[a-zA-Z0-9_.-]+)+(?:\.[a-zA-Z0-9]+)?',
    r'[a-zA-Z0-9_]+/[a-zA-Z0-9_/]+(?:\.[a-zA-Z0-9]+)?',
    r'\b(?:src|lib|bin|etc|usr|home|var|tmp|opt|app|dist|build|node_modules|packages)\b[/\\][a-zA-Z0-9_./\\-]+',
]

CODE_IDENTIFIER_PATTERNS = [
    r'\b(?:function|class|def|var|let|const|import|export|return|async|await)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
    r'([a-z][a-z0-9]*(?:_[a-z0-9]+)+)',  # snake_case
    r'([a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+)',  # camelCase
    r'(?:[A-Z][a-z0-9]*){2,}',  # PascalCase types
    r'\b(?:npm|pip|git|docker|kubectl|yarn|npx|uvicorn|gunicorn|conda)\s+\S+',
]

IMPORT_PATTERNS = [
    r'(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
    r'require\([\'"]([a-zA-Z_][a-zA-Z0-9_./]*)',
]


def extract_file_paths(text: str) -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    for pat in FILE_PATH_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            p = m.group(0).strip().rstrip(".,;:')]}")
            if len(p) >= 3 and p not in seen:
                seen.add(p)
                paths.append(p)
    return paths


def extract_code_identifiers(text: str) -> list[dict]:
    results: list[dict] = []
    seen: set[str] = set()
    for pat in CODE_IDENTIFIER_PATTERNS:
        for m in re.finditer(pat, text):
            identifier = m.group(1) if m.lastindex else m.group(0)
            if identifier and identifier not in seen and len(identifier) >= 2:
                seen.add(identifier)
                results.append({"identifier": identifier, "pattern": pat[:20]})
    return results


def extract_imports(text: str) -> list[str]:
    seen: set[str] = set()
    imports: list[str] = []
    for pat in IMPORT_PATTERNS:
        for m in re.finditer(pat, text):
            module = m.group(1).strip()
            if module and module not in seen:
                seen.add(module)
                imports.append(module)
    return imports
