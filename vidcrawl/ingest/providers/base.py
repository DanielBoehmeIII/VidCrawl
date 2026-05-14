from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Optional


class VideoMetadata:
    __slots__ = ("title", "duration_sec", "url", "description", "uploader", "source_type")

    def __init__(
        self,
        *,
        title: str = "",
        duration_sec: float = 0.0,
        url: Optional[str] = None,
        description: str = "",
        uploader: str = "",
        source_type: str = "generic",
    ) -> None:
        self.title = title
        self.duration_sec = duration_sec
        self.url = url
        self.description = description
        self.uploader = uploader
        self.source_type = source_type

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}


class VideoProvider(ABC):
    source_type: ClassVar[str] = "generic"

    @abstractmethod
    def detect(self, source: str) -> bool:
        """Return True if this provider can handle the given source."""

    def normalize(self, source: str) -> str:
        return source

    @abstractmethod
    def extract_metadata(
        self, source: str, timeout_sec: float = 30.0
    ) -> VideoMetadata: ...

    def fetch_captions(
        self, source: str, timeout_sec: float = 60.0
    ) -> Optional[list[dict]]:
        return None

    def download_audio(
        self,
        source: str,
        output_dir: str,
        video_id: str,
        timeout_sec: float = 600.0,
    ) -> Optional[str]:
        return None
