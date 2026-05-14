import hashlib


def sha256_prefix(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def content_hash(
    transcript_text: str, ocr_text: str = "", max_chars: int = 500
) -> str:
    combined = (
        transcript_text[:max_chars] + ocr_text[:max_chars]
    ).strip()
    if not combined:
        return hashlib.sha256(b"empty").hexdigest()
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()
