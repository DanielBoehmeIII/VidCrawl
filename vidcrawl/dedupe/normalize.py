import hashlib
import re


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def content_hash(text: str) -> str:
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode()).hexdigest()


def get_combined_text(transcript: str, ocr: str, ideas_text: str = "") -> str:
    parts = [transcript or "", ocr or "", ideas_text or ""]
    return " ".join(p.strip() for p in parts if p.strip())
