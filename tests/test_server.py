"""Tests for the VidCrawl FastAPI server layer."""
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vidcrawl.api.app import create_app
from vidcrawl.config import Config
from vidcrawl.db import get_db, init_db


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


@pytest.fixture
def client(tmp_config):
    api = create_app(tmp_config)
    with TestClient(api) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body

    def test_version_matches_package(self, client):
        from vidcrawl import __version__
        resp = client.get("/health")
        assert resp.json()["version"] == __version__


# ---------------------------------------------------------------------------
# GET /stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_empty_db_returns_zeros(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["videos"] == 0
        assert body["moments"] == 0
        assert "db_path" in body

    def test_counts_videos_after_insert(self, client, tmp_config):
        from vidcrawl.db import insert_video
        from vidcrawl.models import Video

        video = Video(
            video_id="test_v1",
            title="Test Video",
            source="local",
            url=None,
            duration_sec=60.0,
            status="pending",
        )
        with get_db(tmp_config.db_path) as conn:
            insert_video(conn, video)

        resp = client.get("/stats")
        assert resp.json()["videos"] == 1

    def test_fts_rows_reflects_moments(self, tmp_config):
        """fts_rows must equal moments after ingest when queried via /stats."""
        from vidcrawl.db import insert_video, insert_moment, rebuild_fts
        from vidcrawl.models import Video, Moment

        video = Video(
            video_id="fts_test_v",
            title="FTS Test",
            source="local",
            url=None,
            duration_sec=60.0,
            status="ready",
        )
        moment = Moment(
            moment_id="fts_test_v:0.00:30.00",
            video_id="fts_test_v",
            start_sec=0.0,
            end_sec=30.0,
            transcript_text="hello world FTS test",
        )
        with get_db(tmp_config.db_path) as conn:
            insert_video(conn, video)
            insert_moment(conn, moment)
            rebuild_fts(conn)

        # Create a fresh app (simulates server restart) — FTS must survive
        api = create_app(tmp_config)
        with TestClient(api) as c:
            resp = c.get("/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["fts_rows"] == 1, f"Expected 1 fts row after restart, got {body['fts_rows']}"


# ---------------------------------------------------------------------------
# POST /clipbounce/capture — YouTube URL normalization
# ---------------------------------------------------------------------------

class TestClipbounceCapture:
    def test_youtube_url_accepted(self, client):
        resp = client.post("/clipbounce/capture", json={
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "title": "Test Video",
            "source_type": "youtube_tab",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "job_id" in body
        assert body["status"] in ("queued", "running", "ready", "skipped")
        assert "dQw4w9WgXcQ" in body["source_url"]

    def test_youtu_be_url_normalized(self, client):
        resp = client.post("/clipbounce/capture", json={
            "url": "https://youtu.be/dQw4w9WgXcQ",
            "title": "Short URL",
            "source_type": "youtube_tab",
        })
        assert resp.status_code == 200
        body = resp.json()
        # Should be normalized to watch?v= form
        assert "youtube.com/watch?v=dQw4w9WgXcQ" in body["source_url"]

    def test_playlist_params_stripped(self, client):
        resp = client.post("/clipbounce/capture", json={
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLxxx&index=3",
            "title": "Playlist Video",
        })
        assert resp.status_code == 200
        url = resp.json()["source_url"]
        assert "list=" not in url
        assert "index=" not in url
        assert "dQw4w9WgXcQ" in url

    def test_non_youtube_url_returns_skipped(self, client):
        resp = client.post("/clipbounce/capture", json={
            "url": "https://example.com/some-article",
            "title": "Article",
            "source_type": "article",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "skipped"

    def test_duplicate_url_does_not_create_new_job(self, client):
        payload = {
            "url": "https://www.youtube.com/watch?v=abc123unique",
            "title": "Dupe Test",
        }
        resp1 = client.post("/clipbounce/capture", json=payload)
        resp2 = client.post("/clipbounce/capture", json=payload)
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        # Same job_id returned
        assert resp1.json()["job_id"] == resp2.json()["job_id"]

    def test_duplicate_youtube_video_does_not_duplicate_video_record(self, client, tmp_config):
        """Capturing the same URL twice must not insert two video rows."""
        from vidcrawl.db import list_videos

        payload = {"url": "https://www.youtube.com/watch?v=dedupetest99"}
        client.post("/clipbounce/capture", json=payload)
        client.post("/clipbounce/capture", json=payload)

        with get_db(tmp_config.db_path) as conn:
            videos = list_videos(conn)

        matching = [v for v in videos if "dedupetest99" in (v.url or "")]
        assert len(matching) <= 1

    def test_already_ready_video_returns_skipped_job(self, client, tmp_config):
        """If the video is already in 'ready' status, capture returns skipped."""
        from vidcrawl.db import insert_video, update_video_status, make_video_id
        from vidcrawl.models import Video

        url = "https://www.youtube.com/watch?v=alreadyready1"
        video_id = make_video_id("youtube", url)
        video = Video(
            video_id=video_id,
            title="Ready Video",
            source="youtube",
            url=url,
            duration_sec=120.0,
            status="ready",
        )
        with get_db(tmp_config.db_path) as conn:
            insert_video(conn, video)

        resp = client.post("/clipbounce/capture", json={"url": url})
        assert resp.status_code == 200
        assert resp.json()["status"] == "skipped"

    def test_sources_batch_form_creates_jobs(self, client):
        """sources[] form creates one job per source and returns BatchCaptureResponse."""
        payload = {
            "sources": [
                {"url": "https://www.youtube.com/watch?v=batchsrc001"},
                {"url": "https://www.youtube.com/watch?v=batchsrc002"},
            ],
            "mode": "captions_first",
        }
        resp = client.post("/clipbounce/capture", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["mode"] == "captions_first"
        assert len(body["jobs"]) == 2
        job_ids = {j["job_id"] for j in body["jobs"]}
        assert len(job_ids) == 2

    def test_sources_batch_deduplicates(self, client):
        """Sending the same URL twice in sources[] returns the same job_id both times."""
        url = "https://www.youtube.com/watch?v=batchdedup01"
        payload = {
            "sources": [{"url": url}, {"url": url}],
            "mode": "default",
        }
        resp = client.post("/clipbounce/capture", json=payload)
        assert resp.status_code == 200
        jobs = resp.json()["jobs"]
        assert len(jobs) == 2
        assert jobs[0]["job_id"] == jobs[1]["job_id"]

    def test_missing_both_url_and_sources_returns_422(self, client):
        """Omitting both url and sources must return 422."""
        resp = client.post("/clipbounce/capture", json={"title": "no url"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /videos + GET /videos/{video_id}
# ---------------------------------------------------------------------------

class TestVideos:
    def test_list_empty(self, client):
        resp = client.get("/videos")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_nonexistent_returns_404(self, client):
        resp = client.get("/videos/no_such_video")
        assert resp.status_code == 404

    def test_list_and_get_after_insert(self, client, tmp_config):
        from vidcrawl.db import insert_video
        from vidcrawl.models import Video

        video = Video(
            video_id="vid_list_test",
            title="Listed",
            source="local",
            url=None,
            duration_sec=30.0,
            status="pending",
        )
        with get_db(tmp_config.db_path) as conn:
            insert_video(conn, video)

        list_resp = client.get("/videos")
        assert any(v["video_id"] == "vid_list_test" for v in list_resp.json())

        get_resp = client.get("/videos/vid_list_test")
        assert get_resp.status_code == 200
        assert get_resp.json()["title"] == "Listed"


# ---------------------------------------------------------------------------
# GET /moments/{moment_id}
# ---------------------------------------------------------------------------

class TestMoments:
    def test_nonexistent_returns_404(self, client):
        resp = client.get("/moments/no_such_moment")
        assert resp.status_code == 404
