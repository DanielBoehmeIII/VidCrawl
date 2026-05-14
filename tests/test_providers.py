"""Tests for the provider abstraction and caption generation pipeline."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vidcrawl.config import Config
from vidcrawl.db import get_db, init_db
from vidcrawl.ingest.providers.generic import GenericYtDlpProvider
from vidcrawl.ingest.providers.local import LocalVideoProvider
from vidcrawl.ingest.providers.registry import get_provider
from vidcrawl.ingest.providers.vimeo import VimeoProvider
from vidcrawl.ingest.providers.youtube import YouTubeProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_config(tmp_path):
    config = Config(data_dir=str(tmp_path / "data"))
    config.ensure_dirs()
    with get_db(config.db_path) as conn:
        init_db(conn)
    return config


# ---------------------------------------------------------------------------
# 1. Provider routing
# ---------------------------------------------------------------------------

class TestProviderRouting:
    def test_youtube_url_routes_to_youtube_provider(self):
        p = get_provider("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert isinstance(p, YouTubeProvider)

    def test_youtu_be_url_routes_to_youtube_provider(self):
        p = get_provider("https://youtu.be/dQw4w9WgXcQ")
        assert isinstance(p, YouTubeProvider)

    def test_vimeo_url_routes_to_vimeo_provider(self):
        p = get_provider("https://vimeo.com/123456789")
        assert isinstance(p, VimeoProvider)

    def test_vimeo_player_url_routes_to_vimeo_provider(self):
        p = get_provider("https://player.vimeo.com/video/123456789")
        assert isinstance(p, VimeoProvider)

    def test_generic_http_url_routes_to_generic_provider(self):
        p = get_provider("https://example-video-site.com/watch/abc123")
        assert isinstance(p, GenericYtDlpProvider)

    def test_local_file_routes_to_local_provider(self, tmp_path):
        video_file = tmp_path / "clip.mp4"
        video_file.write_bytes(b"\x00" * 100)
        p = get_provider(str(video_file))
        assert isinstance(p, LocalVideoProvider)

    def test_explicit_hint_overrides_detection(self):
        # hint="vimeo" on a YouTube URL still returns VimeoProvider
        p = get_provider("https://www.youtube.com/watch?v=abc", hint="vimeo")
        assert isinstance(p, VimeoProvider)

    def test_explicit_youtube_hint(self):
        p = get_provider("https://vimeo.com/123", hint="youtube")
        assert isinstance(p, YouTubeProvider)


# ---------------------------------------------------------------------------
# 2. Provider normalize / detect
# ---------------------------------------------------------------------------

class TestYouTubeProvider:
    def setup_method(self):
        self.p = YouTubeProvider()

    def test_detect_watch_url(self):
        assert self.p.detect("https://www.youtube.com/watch?v=abc") is True

    def test_detect_short_url(self):
        assert self.p.detect("https://youtu.be/abc") is True

    def test_detect_mobile_url(self):
        assert self.p.detect("https://m.youtube.com/watch?v=abc") is True

    def test_not_detect_vimeo(self):
        assert self.p.detect("https://vimeo.com/123") is False

    def test_normalize_strips_playlist(self):
        url = "https://www.youtube.com/watch?v=xyz&list=PLabc&index=2"
        assert self.p.normalize(url) == "https://www.youtube.com/watch?v=xyz"


class TestVimeoProvider:
    def setup_method(self):
        self.p = VimeoProvider()

    def test_detect_vimeo(self):
        assert self.p.detect("https://vimeo.com/12345") is True

    def test_detect_player(self):
        assert self.p.detect("https://player.vimeo.com/video/12345") is True

    def test_not_detect_youtube(self):
        assert self.p.detect("https://youtube.com/watch?v=abc") is False


class TestGenericProvider:
    def setup_method(self):
        self.p = GenericYtDlpProvider()

    def test_detect_http(self):
        assert self.p.detect("http://example.com/video.mp4") is True

    def test_detect_https(self):
        assert self.p.detect("https://example.com/video") is True

    def test_not_detect_local_path(self):
        assert self.p.detect("/home/user/video.mp4") is False


class TestLocalProvider:
    def setup_method(self):
        self.p = LocalVideoProvider()

    def test_detect_existing_file(self, tmp_path):
        f = tmp_path / "video.mp4"
        f.write_bytes(b"\x00")
        assert self.p.detect(str(f)) is True

    def test_not_detect_missing_file(self):
        assert self.p.detect("/nonexistent/path/video.mp4") is False

    def test_normalize_resolves_path(self, tmp_path):
        f = tmp_path / "video.mp4"
        f.write_bytes(b"\x00")
        normalized = self.p.normalize(str(f))
        assert normalized == str(f.resolve())


# ---------------------------------------------------------------------------
# 3. Caption generation — no captions + allow_whisper=False
# ---------------------------------------------------------------------------

class TestNoCaptionsNoWhisper:
    """When captions are unavailable and allow_whisper=False, raise NeedsTranscriptionError."""

    def test_vimeo_no_captions_returns_needs_transcription(self, tmp_config):
        from vidcrawl.process.pipeline import NeedsTranscriptionError, ingest_any_source

        with patch(
            "vidcrawl.ingest.providers.vimeo.VimeoProvider.fetch_captions",
            return_value=None,
        ):
            with pytest.raises(NeedsTranscriptionError):
                ingest_any_source(
                    "https://vimeo.com/999999999",
                    tmp_config,
                    allow_whisper=False,
                )

    def test_generic_no_captions_returns_needs_transcription(self, tmp_config):
        from vidcrawl.process.pipeline import NeedsTranscriptionError, ingest_any_source

        with patch(
            "vidcrawl.ingest.providers.generic.GenericYtDlpProvider.fetch_captions",
            return_value=None,
        ):
            with pytest.raises(NeedsTranscriptionError):
                ingest_any_source(
                    "https://example-video.com/clip/abc",
                    tmp_config,
                    allow_whisper=False,
                )

    def test_youtube_no_captions_returns_needs_transcription(self, tmp_config):
        from vidcrawl.process.pipeline import NeedsTranscriptionError, ingest_any_source

        with patch(
            "vidcrawl.ingest.providers.youtube.YouTubeProvider.fetch_captions",
            return_value=None,
        ):
            with pytest.raises(NeedsTranscriptionError):
                ingest_any_source(
                    "https://www.youtube.com/watch?v=NoCapsTest1",
                    tmp_config,
                    allow_whisper=False,
                )


# ---------------------------------------------------------------------------
# 4. Caption generation — captions found, no Whisper needed
# ---------------------------------------------------------------------------

FAKE_SEGMENTS = [
    {"start_sec": 0.0, "end_sec": 5.0, "text": "Hello world this is a test caption."},
    {"start_sec": 5.0, "end_sec": 10.0, "text": "Second segment with more content here."},
    {"start_sec": 10.0, "end_sec": 15.0, "text": "Third and final test segment content."},
]


class TestCaptionsFound:
    def test_provider_captions_processed_without_whisper(self, tmp_config):
        """When provider returns captions, the pipeline uses them and never calls Whisper."""
        from vidcrawl.process.pipeline import ingest_any_source

        url = "https://www.youtube.com/watch?v=CapsOKTest1"
        with patch(
            "vidcrawl.ingest.providers.youtube.YouTubeProvider.fetch_captions",
            return_value=FAKE_SEGMENTS,
        ), patch(
            "vidcrawl.ingest.providers.youtube.YouTubeProvider.extract_metadata",
            return_value=MagicMock(
                title="Test Video", duration_sec=15.0, url=url,
                description="", uploader="", source_type="youtube",
            ),
        ), patch(
            "vidcrawl.ingest.transcript.transcribe_audio",
            side_effect=AssertionError("Whisper must not be called when captions exist"),
        ):
            video_id = ingest_any_source(url, tmp_config, allow_whisper=False)

        assert video_id  # processed successfully
        with get_db(tmp_config.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM moments WHERE video_id = ?", (video_id,)
            ).fetchone()[0]
        assert count > 0


# ---------------------------------------------------------------------------
# 5. Caption generation — allow_whisper=True calls Whisper
# ---------------------------------------------------------------------------

class TestWhisperFallback:
    def test_no_captions_allow_whisper_calls_transcription(self, tmp_config, tmp_path):
        """When captions are absent and allow_whisper=True, Whisper is called."""
        from vidcrawl.process.pipeline import ingest_any_source

        fake_audio = tmp_path / "audio.mp4"
        fake_audio.write_bytes(b"\x00" * 100)

        whisper_called = []

        def fake_transcribe(audio_path, **kwargs):
            whisper_called.append(audio_path)
            return FAKE_SEGMENTS

        url = "https://www.youtube.com/watch?v=WhisperTest1"
        with patch(
            "vidcrawl.ingest.providers.youtube.YouTubeProvider.fetch_captions",
            return_value=None,
        ), patch(
            "vidcrawl.ingest.providers.youtube.YouTubeProvider.download_audio",
            return_value=str(fake_audio),
        ), patch(
            "vidcrawl.ingest.providers.youtube.YouTubeProvider.extract_metadata",
            return_value=MagicMock(
                title="Whisper Test", duration_sec=15.0, url=url,
                description="", uploader="", source_type="youtube",
            ),
        ), patch(
            "vidcrawl.process.pipeline.transcribe_audio",
            side_effect=fake_transcribe,
        ):
            video_id = ingest_any_source(url, tmp_config, allow_whisper=True, transcribe_model="tiny")

        assert whisper_called, "Whisper was not called"
        with get_db(tmp_config.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM moments WHERE video_id = ?", (video_id,)
            ).fetchone()[0]
        assert count > 0


# ---------------------------------------------------------------------------
# 6. Generated transcript chunked into moments / evidence / ideas
# ---------------------------------------------------------------------------

class TestTranscriptChunking:
    def test_generated_transcript_produces_moments_evidence_ideas(self, tmp_config, tmp_path):
        from vidcrawl.process.pipeline import ingest_any_source

        fake_audio = tmp_path / "v.mp4"
        fake_audio.write_bytes(b"\x00" * 100)

        long_segments = [
            {"start_sec": i * 5.0, "end_sec": (i + 1) * 5.0,
             "text": f"Segment {i} with enough text to generate an idea pattern URL https://x.com."}
            for i in range(20)
        ]

        url = "https://www.youtube.com/watch?v=ChunkTest11"
        with patch(
            "vidcrawl.ingest.providers.youtube.YouTubeProvider.fetch_captions",
            return_value=None,
        ), patch(
            "vidcrawl.ingest.providers.youtube.YouTubeProvider.download_audio",
            return_value=str(fake_audio),
        ), patch(
            "vidcrawl.ingest.providers.youtube.YouTubeProvider.extract_metadata",
            return_value=MagicMock(
                title="Chunk Test", duration_sec=100.0, url=url,
                description="", uploader="", source_type="youtube",
            ),
        ), patch(
            "vidcrawl.process.pipeline.transcribe_audio",
            return_value=long_segments,
        ):
            video_id = ingest_any_source(url, tmp_config, allow_whisper=True)

        with get_db(tmp_config.db_path) as conn:
            moments = conn.execute(
                "SELECT COUNT(*) FROM moments WHERE video_id = ?", (video_id,)
            ).fetchone()[0]
            evidence = conn.execute(
                "SELECT COUNT(*) FROM modal_evidence WHERE moment_id IN "
                "(SELECT moment_id FROM moments WHERE video_id = ?)", (video_id,)
            ).fetchone()[0]

        assert moments > 0, "Expected moments from transcript"
        assert evidence > 0, "Expected evidence records"


# ---------------------------------------------------------------------------
# 7. Duplicate captures do not re-transcribe (Whisper cache)
# ---------------------------------------------------------------------------

class TestWhisperCache:
    def test_duplicate_capture_does_not_retranscribe(self, tmp_config, tmp_path):
        """Second call with allow_whisper=True should use cached transcript."""
        from vidcrawl.process.pipeline import ingest_any_source

        fake_audio = tmp_path / "cached.mp4"
        fake_audio.write_bytes(b"\x00" * 100)

        whisper_call_count = [0]

        def fake_transcribe(audio_path, **kwargs):
            whisper_call_count[0] += 1
            return FAKE_SEGMENTS

        url = "https://www.youtube.com/watch?v=CacheTest011"

        mock_meta = MagicMock(
            title="Cache Test", duration_sec=15.0, url=url,
            description="", uploader="", source_type="youtube",
        )

        patches = [
            patch("vidcrawl.ingest.providers.youtube.YouTubeProvider.fetch_captions", return_value=None),
            patch("vidcrawl.ingest.providers.youtube.YouTubeProvider.download_audio", return_value=str(fake_audio)),
            patch("vidcrawl.ingest.providers.youtube.YouTubeProvider.extract_metadata", return_value=mock_meta),
            patch("vidcrawl.process.pipeline.transcribe_audio", side_effect=fake_transcribe),
        ]

        # First ingest — Whisper should be called once.
        with patches[0], patches[1], patches[2], patches[3]:
            ingest_any_source(url, tmp_config, allow_whisper=True, force=True)

        assert whisper_call_count[0] == 1, "Whisper should run on first ingest"

        # Second ingest with force=True — cache should be used, Whisper not called again.
        with patches[0], patches[1], patches[2], patches[3]:
            ingest_any_source(url, tmp_config, allow_whisper=True, force=True)

        assert whisper_call_count[0] == 1, "Whisper must not run again when cache exists"

    def test_whisper_cache_written_to_transcripts_dir(self, tmp_config, tmp_path):
        from vidcrawl.process.pipeline import ingest_any_source
        from vidcrawl.db import make_video_id

        fake_audio = tmp_path / "write_cache.mp4"
        fake_audio.write_bytes(b"\x00" * 100)

        url = "https://www.youtube.com/watch?v=WriteCache1Test"
        mock_meta = MagicMock(
            title="Write Cache", duration_sec=15.0, url=url,
            description="", uploader="", source_type="youtube",
        )

        with patch("vidcrawl.ingest.providers.youtube.YouTubeProvider.fetch_captions", return_value=None), \
             patch("vidcrawl.ingest.providers.youtube.YouTubeProvider.download_audio", return_value=str(fake_audio)), \
             patch("vidcrawl.ingest.providers.youtube.YouTubeProvider.extract_metadata", return_value=mock_meta), \
             patch("vidcrawl.process.pipeline.transcribe_audio", return_value=FAKE_SEGMENTS):
            ingest_any_source(url, tmp_config, allow_whisper=True)

        video_id = make_video_id("youtube", url)
        cache_path = tmp_config.transcripts_dir / f"{video_id}_whisper.json"
        assert cache_path.exists(), "Whisper cache file was not written"
        cached = json.loads(cache_path.read_text())
        assert cached == FAKE_SEGMENTS


# ---------------------------------------------------------------------------
# 8. API — ClipBounce capture with Vimeo + allow_whisper
# ---------------------------------------------------------------------------

class TestClipbounceCaptureProviders:
    @pytest.fixture
    def client(self, tmp_config):
        from fastapi.testclient import TestClient
        from vidcrawl.api.app import create_app
        api = create_app(tmp_config)
        with TestClient(api) as c:
            yield c

    def test_vimeo_url_accepted(self, client):
        resp = client.post("/clipbounce/capture", json={
            "url": "https://vimeo.com/123456789",
            "title": "Vimeo Test",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "job_id" in body
        assert body["status"] in ("queued", "running", "ready", "skipped", "needs_transcription")

    def test_non_video_url_still_skipped(self, client):
        resp = client.post("/clipbounce/capture", json={
            "url": "https://example.com/some-article",
            "title": "Article",
            "source_type": "article",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "skipped"

    def test_allow_whisper_field_accepted(self, client):
        resp = client.post("/clipbounce/capture", json={
            "url": "https://vimeo.com/987654321",
            "allow_whisper": True,
        })
        assert resp.status_code == 200
        assert "job_id" in resp.json()

    def test_needs_transcription_status_surfaced(self, tmp_config):
        """Worker sets needs_transcription when captions absent and allow_whisper=False."""
        from vidcrawl.api.jobs import get_job, init_jobs_table, create_job, update_job
        from vidcrawl.api.app import _ingest_worker

        with get_db(tmp_config.db_path) as conn:
            init_jobs_table(conn)
            job_id = create_job(conn, "https://vimeo.com/needs_tx_test")

        with patch(
            "vidcrawl.ingest.providers.vimeo.VimeoProvider.fetch_captions",
            return_value=None,
        ):
            _ingest_worker(
                tmp_config, job_id,
                "https://vimeo.com/needs_tx_test",
                process=True, download=False,
                prefer_yt_captions=False, allow_whisper=False,
                transcribe_model="tiny",
            )

        with get_db(tmp_config.db_path) as conn:
            job = get_job(conn, job_id)
        assert job["status"] == "needs_transcription"
