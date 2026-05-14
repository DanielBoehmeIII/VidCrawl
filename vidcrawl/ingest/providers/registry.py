from __future__ import annotations

from typing import Optional

from vidcrawl.ingest.providers.base import VideoProvider
from vidcrawl.ingest.providers.generic import GenericYtDlpProvider
from vidcrawl.ingest.providers.local import LocalVideoProvider
from vidcrawl.ingest.providers.vimeo import VimeoProvider
from vidcrawl.ingest.providers.webpage import WebPageProvider
from vidcrawl.ingest.providers.youtube import YouTubeProvider

# Order matters: more-specific detectors must come before the generic fallback.
_AUTO_PROVIDERS: list[VideoProvider] = [
    YouTubeProvider(),
    VimeoProvider(),
    LocalVideoProvider(),
    GenericYtDlpProvider(),
]

_NAMED_PROVIDERS: dict[str, VideoProvider] = {
    "youtube": YouTubeProvider(),
    "vimeo": VimeoProvider(),
    "generic": GenericYtDlpProvider(),
    "local": LocalVideoProvider(),
    "webpage": WebPageProvider(),
}


def get_provider(source: str, hint: str = "auto") -> VideoProvider:
    """Return the best VideoProvider for *source*.

    hint: "auto" | "youtube" | "vimeo" | "generic" | "local" | "webpage"
    When hint is not "auto", the named provider is returned unconditionally.
    """
    if hint != "auto":
        p = _NAMED_PROVIDERS.get(hint)
        if p is not None:
            return p

    for provider in _AUTO_PROVIDERS:
        if provider.detect(source):
            return provider

    # Last-resort fallback for http/https URLs.
    if source.startswith(("http://", "https://")):
        return _NAMED_PROVIDERS["generic"]

    return _NAMED_PROVIDERS["local"]
