from __future__ import annotations

from pathlib import Path
from typing import Optional

from vidcrawl.ingest.providers.base import VideoMetadata, VideoProvider


class LocalVideoProvider(VideoProvider):
    source_type = "local"

    def detect(self, source: str) -> bool:
        p = Path(source)
        return p.exists() and p.is_file()

    def normalize(self, source: str) -> str:
        return str(Path(source).resolve())

    def extract_metadata(
        self, source: str, timeout_sec: float = 30.0
    ) -> VideoMetadata:
        from vidcrawl.ingest.metadata import extract_duration, extract_file_metadata
        path = Path(source).resolve()
        extract_file_metadata(str(path))
        duration = extract_duration(str(path))
        return VideoMetadata(
            title=path.stem,
            duration_sec=duration,
            url=None,
            source_type="local",
        )

    def fetch_captions(
        self, source: str, timeout_sec: float = 60.0
    ) -> Optional[list[dict]]:
        from vidcrawl.ingest.transcript import load_sidecar_transcript
        return load_sidecar_transcript(source)

    def download_audio(
        self,
        source: str,
        output_dir: str,
        video_id: str,
        timeout_sec: float = 600.0,
    ) -> Optional[str]:
        from vidcrawl.ingest.downloader import accept_local
        try:
            return accept_local(source)
        except FileNotFoundError:
            return None
