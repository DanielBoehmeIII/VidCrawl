from pathlib import Path

VALID_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v"}


def is_valid_video(path: str) -> bool:
    return Path(path).suffix.lower() in VALID_EXTENSIONS


def validate_video_file(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"File not found: {path}")
    if not p.is_file():
        raise ValueError(f"Not a file: {path}")
    if not is_valid_video(str(p)):
        raise ValueError(
            f"Unsupported video format '{p.suffix}'. "
            f"Supported: {', '.join(sorted(VALID_EXTENSIONS))}"
        )
