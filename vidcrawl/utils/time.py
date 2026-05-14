def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


seconds_to_timestamp = format_timestamp


def timestamp_range(start: float, end: float) -> str:
    return f"{format_timestamp(start)}–{format_timestamp(end)}"


def youtube_timestamp_url(url: str | None, start_sec: float) -> str | None:
    if not url:
        return None
    start = int(start_sec)
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}t={start}s"


def parse_timestamp(ts: str) -> float:
    parts = ts.strip().split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    else:
        raise ValueError(f"Cannot parse timestamp: {ts}")
