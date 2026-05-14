from __future__ import annotations

import urllib.parse
from typing import Optional

from vidcrawl.ingest.providers._ytdlp import ytdlp_captions, ytdlp_download, ytdlp_metadata
from vidcrawl.ingest.providers.base import VideoMetadata, VideoProvider


class GenericYtDlpProvider(VideoProvider):
    """Fallback provider for any yt-dlp-supported video site."""

    source_type = "generic"

    def detect(self, source: str) -> bool:
        try:
            scheme = urllib.parse.urlparse(source).scheme.lower()
        except Exception:
            return False
        return scheme in ("http", "https")

    def extract_metadata(
        self, source: str, timeout_sec: float = 30.0
    ) -> VideoMetadata:
        raw = ytdlp_metadata(source, timeout_sec=timeout_sec)
        return VideoMetadata(
            title=raw.get("title", "Video"),
            duration_sec=float(raw.get("duration", 0)),
            url=source,
            description=raw.get("description", ""),
            uploader=raw.get("uploader", ""),
            source_type="generic",
        )

    def fetch_captions(
        self, source: str, timeout_sec: float = 60.0
    ) -> Optional[list[dict]]:
        return ytdlp_captions(source, timeout_sec=timeout_sec)

    def download_audio(
        self,
        source: str,
        output_dir: str,
        video_id: str,
        timeout_sec: float = 600.0,
    ) -> Optional[str]:
        return ytdlp_download(source, output_dir, video_id, timeout_sec=timeout_sec)
