import json
import shutil
import subprocess
import urllib.parse
from pathlib import Path
from typing import Any, Optional


def is_yt_dlp_available() -> bool:
    return shutil.which("yt-dlp") is not None


def normalize_youtube_url(url: str) -> str:
    """Normalize a YouTube URL to https://www.youtube.com/watch?v=<id>.

    Strips playlist, list, radio, start_radio, and all other extra params.
    Returns url unchanged if it is not a recognizable YouTube video URL.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return url

    netloc = parsed.netloc.lower()

    if netloc in ("youtu.be", "www.youtu.be"):
        video_id = parsed.path.lstrip("/").split("/")[0].split("?")[0]
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
        return url

    if netloc in ("youtube.com", "www.youtube.com", "m.youtube.com"):
        params = urllib.parse.parse_qs(parsed.query, keep_blank_values=False)
        video_ids = params.get("v")
        if video_ids:
            return f"https://www.youtube.com/watch?v={video_ids[0]}"

    return url


def download_youtube(
    url: str,
    output_dir: str,
    video_id: str,
    timeout_sec: float = 600.0,
) -> Optional[str]:
    yt_dlp = shutil.which("yt-dlp")
    if not yt_dlp:
        return None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    output_template = str(out / f"{video_id}.%(ext)s")

    try:
        result = subprocess.run(
            [yt_dlp, "-f", "mp4", "-o", output_template, url],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if result.returncode != 0:
            return None

        for f in out.iterdir():
            if f.stem == video_id and f.is_file():
                return str(f)
        return None
    except Exception:
        return None


def extract_youtube_metadata(
    url: str,
    timeout_sec: float = 30.0,
) -> dict[str, Any]:
    yt_dlp = shutil.which("yt-dlp")
    if not yt_dlp:
        return {}

    try:
        result = subprocess.run(
            [yt_dlp, "--dump-json", "--no-download", url],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if result.returncode != 0:
            return {}
        return json.loads(result.stdout)
    except Exception:
        return {}


def yt_dlp_install_help() -> str:
    return (
        "yt-dlp is not installed. YouTube download requires yt-dlp.\n"
        "  Install: pip install yt-dlp\n"
        "  Or:      apt install yt-dlp\n"
        "Without yt-dlp, YouTube videos are registered as metadata only."
    )


def accept_local(path: str) -> str:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Local file not found: {p}")
    return str(p)
