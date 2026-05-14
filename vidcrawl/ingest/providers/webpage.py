from __future__ import annotations

from typing import Optional

from vidcrawl.ingest.providers.base import VideoMetadata, VideoProvider


class WebPageProvider(VideoProvider):
    """Handles ClipBounce webpage captures (page_text / selected_text).

    This provider never auto-detects — it is selected explicitly when the
    caller knows the source is a plain webpage rather than a hosted video.
    """

    source_type = "webpage"

    def detect(self, source: str) -> bool:
        return False

    def extract_metadata(
        self, source: str, timeout_sec: float = 30.0
    ) -> VideoMetadata:
        return VideoMetadata(title="", duration_sec=0.0, url=source, source_type="webpage")

    def fetch_captions(
        self, source: str, timeout_sec: float = 60.0
    ) -> Optional[list[dict]]:
        return None
