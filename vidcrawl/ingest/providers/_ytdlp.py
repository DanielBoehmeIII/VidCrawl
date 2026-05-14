"""Shared yt-dlp helpers for VimeoProvider and GenericYtDlpProvider."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def _yt_dlp_bin() -> Optional[str]:
    return shutil.which("yt-dlp")


def ytdlp_metadata(url: str, timeout_sec: float = 30.0) -> dict:
    bin_ = _yt_dlp_bin()
    if not bin_:
        return {}
    try:
        result = subprocess.run(
            [bin_, "--dump-json", "--no-download", url],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if result.returncode != 0:
            return {}
        return json.loads(result.stdout)
    except Exception:
        return {}


def ytdlp_captions(url: str, timeout_sec: float = 60.0) -> Optional[list[dict]]:
    bin_ = _yt_dlp_bin()
    if not bin_:
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        output_template = str(Path(tmpdir) / "caption")
        try:
            subprocess.run(
                [
                    bin_,
                    "--write-auto-sub", "--write-sub",
                    "--sub-lang", "en",
                    "--skip-download",
                    "--convert-subs", "vtt",
                    "-o", output_template,
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except Exception:
            return None

        from vidcrawl.ingest.transcript import _parse_transcript_file
        for f in sorted(Path(tmpdir).iterdir()):
            if f.suffix in (".vtt", ".srt") and f.stat().st_size > 0:
                try:
                    raw = _parse_transcript_file(str(f), f.suffix)
                    if raw:
                        return raw
                except Exception:
                    continue
    return None


def ytdlp_download(
    url: str,
    output_dir: str,
    video_id: str,
    timeout_sec: float = 600.0,
) -> Optional[str]:
    bin_ = _yt_dlp_bin()
    if not bin_:
        return None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    output_template = str(out / f"{video_id}.%(ext)s")

    from vidcrawl.ingest.media import VALID_EXTENSIONS

    for fmt_args in (["-f", "mp4"], []):
        try:
            result = subprocess.run(
                [bin_] + fmt_args + ["-o", output_template, url],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            if result.returncode == 0:
                break
        except Exception:
            return None

    for f in out.iterdir():
        if f.stem == video_id and f.is_file() and f.suffix.lower() in VALID_EXTENSIONS:
            return str(f)
    return None
