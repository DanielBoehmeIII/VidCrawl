import re

EQUATION_PATTERNS = [
    r'[A-Za-z]\s*=\s*[A-Za-z][A-Za-z0-9]*(?:\^-\d+)?(?:\s*[+\-*/]\s*[A-Za-z0-9]+)*',
    r'det\s*\([^)]+\)',
    r'[A-Za-z]+\^T\s*[A-Za-z]',
    r'(?:\b|(?<=\s))(?:sum|integral|limit|log|exp|sqrt|sin|cos|tan)\s*\([^)]+\)',
    r'(?:\b|(?<=\s))lambda\s*\s*[A-Za-z0-9]+',
    r'(?:\b|(?<=\s))[A-Za-z]\s*=\s*PDP\^-1',
    r'Ax\s*=\s*b',
    r'(?:\b|(?<=\s))[A-Za-z]\s*=\s*[A-Za-z]+\s*[+\-*/]\s*[A-Za-z]+',
    r'(?:\b|(?<=\s))\d+\.?\d*\s*[+\-*/]\s*\d+\.?\d*\s*=\s*\d+\.?\d*',
    r'\\\(.*?\\\)',
    r'\\\[.*?\\\]',
]


def extract_equations(text: str) -> list[str]:
    seen: set[str] = set()
    equations: list[str] = []
    for pat in EQUATION_PATTERNS:
        for m in re.finditer(pat, text):
            eq = m.group(0).strip().rstrip(".,;:")
            if eq and eq not in seen and len(eq) >= 3:
                seen.add(eq)
                equations.append(eq)
    return equations
