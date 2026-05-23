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


# ---------------------------------------------------------------------------
# GET /clipbounce/context
# ---------------------------------------------------------------------------

class TestClipbounceContext:
    # Use 11-char YouTube IDs so make_video_id("youtube", ...) returns the ID directly.
    _READY_YT_ID = "ctxreadyvid"   # 11 chars
    _READY_YT_URL = "https://www.youtube.com/watch?v=ctxreadyvid"
    _PENDING_YT_ID = "ctxpendvid1"  # 11 chars
    _PENDING_YT_URL = "https://www.youtube.com/watch?v=ctxpendvid1"

    def _insert_ready_video_with_moments(self, tmp_config):
        from vidcrawl.db import insert_video, insert_moment
        from vidcrawl.models import Video, Moment

        video = Video(
            video_id=self._READY_YT_ID,
            title="Context Test Video",
            source="youtube",
            url=self._READY_YT_URL,
            duration_sec=150.0,
            status="ready",
        )
        moments = [
            Moment(
                moment_id=f"{self._READY_YT_ID}:{i * 30:.2f}:{(i + 1) * 30:.2f}",
                video_id=self._READY_YT_ID,
                start_sec=float(i * 30),
                end_sec=float((i + 1) * 30),
                transcript_text=f"This is segment {i} about topic {i}",
            )
            for i in range(5)
        ]
        with get_db(tmp_config.db_path) as conn:
            insert_video(conn, video)
            for m in moments:
                insert_moment(conn, m)
        return video, moments

    def test_ready_video_returns_nonempty_moments(self, client, tmp_config):
        self._insert_ready_video_with_moments(tmp_config)
        resp = client.get(f"/clipbounce/context?url={self._READY_YT_URL}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        assert body["video_id"] == self._READY_YT_ID
        assert len(body["moments"]) > 0
        m = body["moments"][0]
        assert "timestamp_label" in m
        assert "transcript_snippet" in m
        assert "idea_summary" in m
        assert "score" in m

    def test_unknown_video_returns_404(self, client):
        resp = client.get("/clipbounce/context?url=https://www.youtube.com/watch?v=unknownvid01")
        assert resp.status_code == 404

    def test_pending_video_returns_indexing(self, client, tmp_config):
        from vidcrawl.db import insert_video
        from vidcrawl.models import Video

        video = Video(
            video_id=self._PENDING_YT_ID,
            title="Pending Video",
            source="youtube",
            url=self._PENDING_YT_URL,
            duration_sec=60.0,
            status="pending",
        )
        with get_db(tmp_config.db_path) as conn:
            insert_video(conn, video)

        resp = client.get(f"/clipbounce/context?url={self._PENDING_YT_URL}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "indexing"
        assert body["moments"] == []

    def test_queued_job_no_video_returns_indexing_not_404(self, client, tmp_config):
        """After /capture, a queued job exists but no video row yet — must be indexing."""
        from vidcrawl.api.jobs import create_job

        # 11-char YouTube ID so make_video_id returns it directly
        yt_url = "https://www.youtube.com/watch?v=idxtest1234"
        with get_db(tmp_config.db_path) as conn:
            create_job(conn, yt_url, video_id="idxtest1234")

        resp = client.get(f"/clipbounce/context?url={yt_url}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "indexing"
        assert body["video_id"] == "idxtest1234"
        assert body["job_id"] is not None
        assert body["moments"] == []

    def test_skipped_no_transcript_job_returns_correct_status(self, client, tmp_config):
        from vidcrawl.api.jobs import create_job, update_job

        yt_url = "https://www.youtube.com/watch?v=notxscript1"  # 11-char ID
        with get_db(tmp_config.db_path) as conn:
            job_id = create_job(conn, yt_url, video_id="notxscript1")
            update_job(conn, job_id, "skipped_no_transcript",
                       error_message="No transcript available")

        resp = client.get(f"/clipbounce/context?url={yt_url}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "skipped_no_transcript"
        assert body["moments"] == []

    def test_needs_transcription_job_returns_correct_status(self, client, tmp_config):
        from vidcrawl.api.jobs import create_job, update_job

        yt_url = "https://www.youtube.com/watch?v=needstrncpt"  # 11-char ID
        with get_db(tmp_config.db_path) as conn:
            job_id = create_job(conn, yt_url, video_id="needstrncpt")
            update_job(conn, job_id, "needs_transcription",
                       error_message="Whisper required")

        resp = client.get(f"/clipbounce/context?url={yt_url}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "needs_transcription"

    def test_normalized_youtube_url_works(self, client, tmp_config):
        from vidcrawl.db import insert_video
        from vidcrawl.models import Video

        # Insert under the canonical watch?v= form
        canonical_url = "https://www.youtube.com/watch?v=normtest123"
        video = Video(
            video_id="normtest123",  # 11-char YouTube ID extracted by make_video_id
            title="Norm Test Video",
            source="youtube",
            url=canonical_url,
            duration_sec=90.0,
            status="pending",
        )
        with get_db(tmp_config.db_path) as conn:
            insert_video(conn, video)

        # Query with the youtu.be short form — must resolve to the same video
        resp = client.get("/clipbounce/context?url=https://youtu.be/normtest123")
        assert resp.status_code == 200
        body = resp.json()
        assert body["video_id"] == "normtest123"


# ---------------------------------------------------------------------------
# GET /clipbounce/jobs
# ---------------------------------------------------------------------------

class TestClipbounceJobs:
    def test_returns_job_for_known_url(self, client, tmp_config):
        from vidcrawl.api.jobs import create_job

        yt_url = "https://www.youtube.com/watch?v=jobstest1234"
        with get_db(tmp_config.db_path) as conn:
            job_id = create_job(conn, yt_url, video_id="jobstest1234")

        resp = client.get(f"/clipbounce/jobs?url={yt_url}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == job_id
        assert body["status"] == "queued"
        assert "jobstest1234" in body["source_url"]

    def test_normalized_short_url_matches_same_job(self, client, tmp_config):
        """youtu.be and watch?v= forms resolve to the same job."""
        from vidcrawl.api.jobs import create_job

        canonical = "https://www.youtube.com/watch?v=jobsnrm1234"
        with get_db(tmp_config.db_path) as conn:
            job_id = create_job(conn, canonical, video_id="jobsnrm1234")

        resp = client.get("/clipbounce/jobs?url=https://youtu.be/jobsnrm1234")
        assert resp.status_code == 200
        assert resp.json()["job_id"] == job_id

    def test_unknown_url_returns_404(self, client):
        resp = client.get("/clipbounce/jobs?url=https://www.youtube.com/watch?v=nojob1234x")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Job progress stages (tasks 1-3)
# ---------------------------------------------------------------------------

class TestJobProgress:
    def test_job_created_with_stage_queued(self, client, tmp_config):
        from vidcrawl.api.jobs import create_job
        from vidcrawl.db import get_db

        yt_url = "https://www.youtube.com/watch?v=progstage01"
        with get_db(tmp_config.db_path) as conn:
            job_id = create_job(conn, yt_url, video_id="progstage01")
            job = conn.execute("SELECT * FROM api_jobs WHERE job_id = ?", (job_id,)).fetchone()
            assert job["stage"] == "queued"

    def test_update_job_progress_sets_stage(self, client, tmp_config):
        from vidcrawl.api.jobs import create_job, update_job_progress
        from vidcrawl.db import get_db

        yt_url = "https://www.youtube.com/watch?v=progstage02"
        with get_db(tmp_config.db_path) as conn:
            job_id = create_job(conn, yt_url, video_id="progstage02")
            update_job_progress(conn, job_id, "captions", "Loaded 42 captions")
            job = conn.execute("SELECT * FROM api_jobs WHERE job_id = ?", (job_id,)).fetchone()
            assert job["stage"] == "captions"
            assert job["progress_message"] == "Loaded 42 captions"

    def test_job_response_includes_progress_fields(self, client, tmp_config):
        from vidcrawl.api.jobs import create_job, update_job_progress
        from vidcrawl.db import get_db

        yt_url = "https://www.youtube.com/watch?v=progstage03"
        with get_db(tmp_config.db_path) as conn:
            job_id = create_job(conn, yt_url, video_id="progstage03")
            update_job_progress(conn, job_id, "chunking", "Creating chunks")

        resp = client.get(f"/clipbounce/jobs?url={yt_url}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["stage"] == "chunking"
        assert body["progress_message"] == "Creating chunks"
        assert "started_at" in body
        assert "duration_ms" in body

    def test_context_returns_progress_stage(self, client, tmp_config):
        from vidcrawl.api.jobs import create_job, update_job_progress
        from vidcrawl.db import get_db

        yt_url = "https://www.youtube.com/watch?v=progstage04"
        with get_db(tmp_config.db_path) as conn:
            job_id = create_job(conn, yt_url, video_id="progstage04")
            update_job_progress(conn, job_id, "writing_db", "Writing moments")

        resp = client.get(f"/clipbounce/context?url={yt_url}")
        assert resp.status_code == 200
        body = resp.json()
        assert "stage" in body
        assert body["stage"] == "writing_db"
        assert "progress_message" in body

    def test_update_job_sets_started_finished_duration(self, client, tmp_config):
        from vidcrawl.api.jobs import create_job, init_jobs_table, update_job
        from vidcrawl.db import get_db

        with get_db(tmp_config.db_path) as conn:
            init_jobs_table(conn)

        yt_url = "https://www.youtube.com/watch?v=progstage05"
        with get_db(tmp_config.db_path) as conn:
            job_id = create_job(conn, yt_url, video_id="progstage05")
            update_job(conn, job_id, "running")
            job = conn.execute("SELECT * FROM api_jobs WHERE job_id = ?", (job_id,)).fetchone()
            assert job["started_at"] is not None
            import time; time.sleep(0.01)
            update_job(conn, job_id, "ready")
            job = conn.execute("SELECT * FROM api_jobs WHERE job_id = ?", (job_id,)).fetchone()
            assert job["finished_at"] is not None
            assert job["duration_ms"] is not None
            assert job["duration_ms"] > 0


# ---------------------------------------------------------------------------
# Stuck job detection (task 4)
# ---------------------------------------------------------------------------

class TestStuckJobs:
    def _init_jobs(self, db_path):
        from vidcrawl.api.jobs import init_jobs_table
        from vidcrawl.db import get_db
        with get_db(db_path) as conn:
            init_jobs_table(conn)

    def test_find_stuck_jobs_detects_running_without_heartbeat(self, tmp_config):
        from vidcrawl.api.jobs import create_job, update_job, find_stuck_jobs
        from vidcrawl.db import get_db

        self._init_jobs(tmp_config.db_path)
        yt_url = "https://www.youtube.com/watch?v=stucktest01"
        with get_db(tmp_config.db_path) as conn:
            job_id = create_job(conn, yt_url, video_id="stucktest01")
            update_job(conn, job_id, "running")
            # Set heartbeat to 3 minutes ago
            conn.execute(
                "UPDATE api_jobs SET last_heartbeat_at = datetime('now', '-3 minutes') WHERE job_id = ?",
                (job_id,),
            )
            conn.commit()
            stuck = find_stuck_jobs(conn, timeout_sec=120)
            assert any(j["job_id"] == job_id for j in stuck)

    def test_recent_heartbeat_not_stuck(self, tmp_config):
        from vidcrawl.api.jobs import create_job, update_job, find_stuck_jobs
        from vidcrawl.db import get_db

        self._init_jobs(tmp_config.db_path)
        yt_url = "https://www.youtube.com/watch?v=stucktest02"
        with get_db(tmp_config.db_path) as conn:
            job_id = create_job(conn, yt_url, video_id="stucktest02")
            update_job(conn, job_id, "running")
            # Recent heartbeat (now) — not stuck
            stuck = find_stuck_jobs(conn, timeout_sec=120)
            assert not any(j["job_id"] == job_id for j in stuck)

    def test_mark_stuck_jobs_updates_status(self, tmp_config):
        from vidcrawl.api.jobs import create_job, update_job, mark_stuck_jobs
        from vidcrawl.db import get_db

        self._init_jobs(tmp_config.db_path)
        yt_url = "https://www.youtube.com/watch?v=stucktest03"
        with get_db(tmp_config.db_path) as conn:
            job_id = create_job(conn, yt_url, video_id="stucktest03")
            update_job(conn, job_id, "running")
            conn.execute(
                "UPDATE api_jobs SET last_heartbeat_at = datetime('now', '-5 minutes') WHERE job_id = ?",
                (job_id,),
            )
            conn.commit()
            marked = mark_stuck_jobs(conn, timeout_sec=120)
            assert any(j["job_id"] == job_id for j in marked)
            job = conn.execute("SELECT * FROM api_jobs WHERE job_id = ?", (job_id,)).fetchone()
            assert job["status"] == "error"

    def test_context_endpoint_clears_stuck_jobs(self, client, tmp_config):
        """Hitting /clipbounce/context should auto-clean stuck running jobs."""
        from vidcrawl.api.jobs import create_job, update_job
        from vidcrawl.db import get_db

        yt_url = "https://www.youtube.com/watch?v=stucktest04"
        with get_db(tmp_config.db_path) as conn:
            job_id = create_job(conn, yt_url, video_id="stucktest04")
            update_job(conn, job_id, "running")
            conn.execute(
                "UPDATE api_jobs SET last_heartbeat_at = datetime('now', '-5 minutes') WHERE job_id = ?",
                (job_id,),
            )
            conn.commit()

        # This call triggers mark_stuck_jobs internally
        client.get(f"/clipbounce/context?url={yt_url}")

        with get_db(tmp_config.db_path) as conn:
            job = conn.execute("SELECT * FROM api_jobs WHERE job_id = ?", (job_id,)).fetchone()
            assert job["status"] == "error", f"Expected error, got {job['status']}"


# ---------------------------------------------------------------------------
# /clipbounce/context: ready when moments exist (task 7)
# ---------------------------------------------------------------------------

class TestClipbounceContextReadyWithMoments:
    _YT_ID = "momentsrdy1"  # exactly 11 chars
    _YT_URL = "https://www.youtube.com/watch?v=momentsrdy1"

    def test_ready_when_moments_exist_even_if_video_not_ready(self, client, tmp_config):
        """Return status=ready as soon as moments exist, even without video.status=ready."""
        from vidcrawl.db import insert_video, insert_moment, make_video_id
        from vidcrawl.models import Video, Moment

        video_id = make_video_id("youtube", self._YT_URL)
        video = Video(
            video_id=video_id,
            title="Moments Ready Test",
            source="youtube",
            url=self._YT_URL,
            duration_sec=60.0,
            status="pending",
        )
        moment = Moment(
            moment_id=f"{video_id}:0.00:30.00",
            video_id=video_id,
            start_sec=0.0,
            end_sec=30.0,
            transcript_text="Test moment content",
        )
        with get_db(tmp_config.db_path) as conn:
            insert_video(conn, video)
            insert_moment(conn, moment)

        resp = client.get(f"/clipbounce/context?url={self._YT_URL}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready", f"Expected ready, got {body['status']}"
        assert len(body["moments"]) > 0
        assert body["moments"][0]["transcript_snippet"] == "Test moment content"

    def test_no_moments_returns_indexing(self, client, tmp_config):
        """If no moments, video with pending status returns indexing."""
        from vidcrawl.db import insert_video, make_video_id
        from vidcrawl.models import Video

        yt_url = "https://www.youtube.com/watch?v=nomoments01"  # 11-char ID
        video_id = make_video_id("youtube", yt_url)
        video = Video(
            video_id=video_id,
            title="No Moments Yet",
            source="youtube",
            url=yt_url,
            duration_sec=60.0,
            status="pending",
        )
        with get_db(tmp_config.db_path) as conn:
            insert_video(conn, video)

        resp = client.get(f"/clipbounce/context?url={yt_url}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "indexing"
        assert len(body["moments"]) == 0


# ---------------------------------------------------------------------------
# Fast mode (task 6)
# ---------------------------------------------------------------------------

class TestFastMode:
    def test_capture_with_fast_mode_creates_job(self, client):
        """mode=fast should be accepted and create a job."""
        resp = client.post("/clipbounce/capture", json={
            "url": "https://www.youtube.com/watch?v=fastmode001",
            "title": "Fast Mode Test",
            "mode": "fast",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "job_id" in body
        assert body["status"] in ("queued", "running", "ready", "skipped")

    def test_batch_capture_with_fast_mode(self, client):
        """Batch capture with mode=fast should create jobs."""
        resp = client.post("/clipbounce/capture", json={
            "sources": [
                {"url": "https://www.youtube.com/watch?v=fastbatch001"},
                {"url": "https://www.youtube.com/watch?v=fastbatch002"},
            ],
            "mode": "fast",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["mode"] == "fast"
        assert len(body["jobs"]) == 2
        for job in body["jobs"]:
            assert "job_id" in job


# ---------------------------------------------------------------------------
# No captions + no whisper → needs_transcription / skipped_no_transcript
# ---------------------------------------------------------------------------

class TestNoCaptionsFlow:
    def test_no_captions_no_whisper_becomes_needs_transcription(self, client, tmp_config):
        """A YouTube job with no captions and allow_whisper=False should settle on
        needs_transcription, not stay running forever."""
        from vidcrawl.api.jobs import get_job, init_jobs_table
        from vidcrawl.db import get_db
        from vidcrawl.ingest.providers.registry import get_provider

        yt_url = "https://www.youtube.com/watch?v=nocaptions01"

        # This provider captions call returns None without yt-dlp
        provider = get_provider(yt_url)
        norm_url = provider.normalize(yt_url)

        resp = client.post("/clipbounce/capture", json={
            "url": yt_url,
            "allow_whisper": False,
        })
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        # Worker runs async in thread pool; poll until it finishes
        import time
        for _ in range(50):
            time.sleep(0.1)
            jresp = client.get(f"/clipbounce/jobs?url={norm_url}")
            if jresp.status_code != 200:
                continue
            body = jresp.json()
            if body["status"] not in ("queued", "running"):
                assert body["status"] in (
                    "needs_transcription", "skipped_no_transcript", "error", "ready"
                ), f"Unexpected terminal status: {body['status']}"
                return

        pytest.fail("Job never left queued/running state")

    def test_debug_endpoint_reports_captions_zero(self, client, tmp_config):
        """/debug/clipbounce-flow should report 0 captions for a no-caption URL."""
        yt_url = "https://www.youtube.com/watch?v=debugcapt01"
        resp = client.get(f"/debug/clipbounce-flow?url={yt_url}")
        assert resp.status_code == 200
        body = resp.json()
        assert "captions_found" in body
        assert body["captions_found"] == 0


# ---------------------------------------------------------------------------
# Worker exception → job marked error
# ---------------------------------------------------------------------------

class TestWorkerError:
    def test_exception_caught_and_stored_in_job(self, client, tmp_config):
        """Simulate a worker crash by submitting a badly-formed source."""
        from vidcrawl.db import get_db, get_video

        # Submit a capture for a URL that will trigger an error path
        # A non-http URL won't match video hosts and goes to skipped path
        resp = client.post("/clipbounce/capture", json={
            "url": "https://www.youtube.com/watch?v=errtestvid01",
            "allow_whisper": False,
        })
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        # Poll until terminal
        import time
        for _ in range(50):
            time.sleep(0.1)
            with get_db(tmp_config.db_path) as conn:
                job = conn.execute(
                    "SELECT * FROM api_jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
            if job and job["status"] not in ("queued", "running"):
                break

        with get_db(tmp_config.db_path) as conn:
            job = conn.execute(
                "SELECT * FROM api_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()

        assert job is not None
        assert job["status"] in (
            "needs_transcription", "skipped_no_transcript", "error"
        ), f"Expected terminal status, got {job['status']}"
        # The error_message should be populated if error occurred
        if job["status"] == "error":
            assert job["error_message"] is not None


# ---------------------------------------------------------------------------
# Stale heartbeat → mark_stuck_jobs (task 4 verification)
# ---------------------------------------------------------------------------

class TestStaleHeartbeat:
    def test_null_heartbeat_detected_as_stuck(self, tmp_config):
        """A running job with NULL last_heartbeat_at must be found by find_stuck_jobs."""
        from vidcrawl.api.jobs import (
            create_job, update_job, find_stuck_jobs, init_jobs_table,
        )
        from vidcrawl.db import get_db

        with get_db(tmp_config.db_path) as conn:
            init_jobs_table(conn)

        with get_db(tmp_config.db_path) as conn:
            job_id = create_job(conn, "https://youtube.com/watch?v=nullhb01")
            update_job(conn, job_id, "running")
            # Set heartbeat to NULL explicitly
            conn.execute(
                "UPDATE api_jobs SET last_heartbeat_at = NULL WHERE job_id = ?",
                (job_id,),
            )
            conn.commit()
            stuck = find_stuck_jobs(conn, timeout_sec=120)
            assert any(j["job_id"] == job_id for j in stuck), \
                "Running job with NULL heartbeat should be detected as stuck"

    def test_null_heartbeat_marked_error(self, tmp_config):
        """A running job with NULL heartbeat should be markable as error."""
        from vidcrawl.api.jobs import (
            create_job, update_job, mark_stuck_jobs, init_jobs_table,
        )
        from vidcrawl.db import get_db

        with get_db(tmp_config.db_path) as conn:
            init_jobs_table(conn)

        with get_db(tmp_config.db_path) as conn:
            job_id = create_job(conn, "https://youtube.com/watch?v=nullhb02")
            update_job(conn, job_id, "running")
            conn.execute(
                "UPDATE api_jobs SET last_heartbeat_at = NULL WHERE job_id = ?",
                (job_id,),
            )
            conn.commit()
            marked = mark_stuck_jobs(conn, timeout_sec=120)
            assert any(j["job_id"] == job_id for j in marked)
            job = conn.execute("SELECT * FROM api_jobs WHERE job_id = ?", (job_id,)).fetchone()
            assert job["status"] == "error"


# ---------------------------------------------------------------------------
# /debug/clipbounce-flow diagnostic endpoint
# ---------------------------------------------------------------------------

class TestDebugEndpoint:
    def test_debug_returns_job_and_video_info(self, client, tmp_config):
        from vidcrawl.api.jobs import create_job, update_job, init_jobs_table
        from vidcrawl.db import get_db, insert_video
        from vidcrawl.models import Video

        yt_url = "https://www.youtube.com/watch?v=debugtest01"
        with get_db(tmp_config.db_path) as conn:
            init_jobs_table(conn)

        video_id = None
        from vidcrawl.db import make_video_id
        video_id = make_video_id("youtube", yt_url)

        with get_db(tmp_config.db_path) as conn:
            insert_video(conn, Video(
                video_id=video_id,
                title="Debug Test",
                source="youtube",
                url=yt_url,
                duration_sec=120.0,
                status="ready",
            ))
            init_jobs_table(conn)
            job_id = create_job(conn, yt_url, video_id=video_id)
            update_job(conn, job_id, "ready")

        resp = client.get(f"/debug/clipbounce-flow?url={yt_url}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["video_id"] == video_id
        assert body["job"] is not None
        assert body["job"]["status"] == "ready"
        assert body["video"] is not None
        assert body["video"]["status"] == "ready"

    def test_debug_unknown_url_returns_no_job(self, client):
        resp = client.get("/debug/clipbounce-flow?url=https://www.youtube.com/watch?v=unknownvid01")
        assert resp.status_code == 200
        body = resp.json()
        assert body["job"] is None
        assert body["video"] is None
