import re

ERROR_PATTERNS = [
    r'(?:Error|ERROR|error)[:\s]+.*',
    r'(?:Traceback|traceback)[:\s]+.*',
    r'(?:Cannot find module|Module not found|No module named)\s+\S+',
    r'(?:Permission denied|command not found|segmentation fault|core dumped)',
    r'(?:SyntaxError|TypeError|ValueError|KeyError|IndexError|AttributeError|ImportError|RuntimeError|FileNotFoundError|ZeroDivisionError)',
    r'(?:FAILED|failed|FAILURE|failure)[:\s].*',
    r'(?:exit code|returned non-zero|exited with)\s+\d+',
    r'(?:404|403|500|502|503)\s+(?:Not Found|Forbidden|Error)',
    r'(?:Could not|Cannot|Unable to)\s+\w+',
    r'(?:timeout|timed out|Timed out|Timeout)',
    r'(?:warning|WARNING|Warning)[:\s]+.*deprecat',
    r'(?:panic|PANIC)[:\s].*',
    r'(?:AssertionError|assert\s+.*failed)',
]


def extract_errors(text: str) -> list[str]:
    seen: set[str] = set()
    errors: list[str] = []
    for pat in ERROR_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            err = m.group(0).strip().rstrip(".,;:")
            if err and err not in seen and len(err) >= 5:
                seen.add(err)
                errors.append(err)
    return errors
