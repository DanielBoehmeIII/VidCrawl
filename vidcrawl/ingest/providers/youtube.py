from __future__ import annotations

import urllib.parse
from typing import Optional

from vidcrawl.ingest.providers.base import VideoMetadata, VideoProvider


class YouTubeProvider(VideoProvider):
    source_type = "youtube"

    _HOSTS = frozenset({
        "youtube.com", "www.youtube.com", "m.youtube.com",
        "youtu.be", "www.youtu.be",
    })

    def detect(self, source: str) -> bool:
        try:
            host = urllib.parse.urlparse(source).netloc.lower()
        except Exception:
            return False
        return host in self._HOSTS

    def normalize(self, source: str) -> str:
        from vidcrawl.ingest.downloader import normalize_youtube_url
        return normalize_youtube_url(source)

    def extract_metadata(
        self, source: str, timeout_sec: float = 30.0
    ) -> VideoMetadata:
        from vidcrawl.ingest.downloader import (
            extract_youtube_metadata,
            is_yt_dlp_available,
        )
        raw: dict = {}
        if is_yt_dlp_available():
            raw = extract_youtube_metadata(source, timeout_sec=timeout_sec)
        return VideoMetadata(
            title=raw.get("title", "YouTube video"),
            duration_sec=float(raw.get("duration", 0)),
            url=source,
            description=raw.get("description", ""),
            uploader=raw.get("uploader", ""),
            source_type="youtube",
        )

    def fetch_captions(
        self, source: str, timeout_sec: float = 60.0
    ) -> Optional[list[dict]]:
        from vidcrawl.ingest.transcript import fetch_youtube_captions
        result = fetch_youtube_captions(source, timeout_sec=timeout_sec)
        return result or None

    def download_audio(
        self,
        source: str,
        output_dir: str,
        video_id: str,
        timeout_sec: float = 600.0,
    ) -> Optional[str]:
        from vidcrawl.ingest.downloader import download_youtube, is_yt_dlp_available
        if not is_yt_dlp_available():
            return None
        return download_youtube(source, output_dir, video_id, timeout_sec=timeout_sec)
