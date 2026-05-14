from pathlib import Path
from typing import Any


def extract_file_metadata(path: str) -> dict[str, Any]:
    p = Path(path)
    return {
        "file_path": str(p.resolve()),
        "file_size_bytes": p.stat().st_size,
        "file_name": p.name,
    }


def extract_duration(path: str) -> float:
    return 0.0
