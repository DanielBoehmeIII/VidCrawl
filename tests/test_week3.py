import json
import math
import sqlite3
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vidcrawl.cli import app
from vidcrawl.db import (
    get_db,
    get_moment,
    init_db,
    insert_moment,
    insert_video,
    make_moment_id,
    rebuild_fts,
)
from vidcrawl.models import Idea, Moment, Video
from vidcrawl.search.query import (
    SearchResult,
    _extract_snippet,
    _make_phrase_query,
    _sanitize_fts_query,
    search_moments,
)
from vidcrawl.utils.time import (
    format_timestamp,
    seconds_to_timestamp,
    timestamp_range,
    youtube_timestamp_url,
)

runner = CliRunner()


# ============================================================
# Timestamp Utilities
# ============================================================

class TestTimestampUtils:
    def test_seconds_to_timestamp(self):
        assert seconds_to_timestamp(65) == "1:05"
        assert seconds_to_timestamp(3661) == "1:01:01"
        assert seconds_to_timestamp(0) == "0:00"

    def test_timestamp_range(self):
        result = timestamp_range(65, 125)
        assert "1:05" in result
        assert "2:05" in result
        assert "–" in result

    def test_youtube_timestamp_url_with_question(self):
        url = "https://youtube.com/watch?v=abc123"
        result = youtube_timestamp_url(url, 65)
        assert result == "https://youtube.com/watch?v=abc123&t=65s"

    def test_youtube_timestamp_url_with_ampersand(self):
        url = "https://youtube.com/watch?v=abc123&feature=share"
        result = youtube_timestamp_url(url, 30)
        assert "&t=30s" in result

    def test_youtube_timestamp_url_none(self):
        assert youtube_timestamp_url(None, 65) is None

    def test_youtube_timestamp_url_empty(self):
        assert youtube_timestamp_url("", 65) is None

    def test_format_timestamp_still_works(self):
        assert format_timestamp(65) == "1:05"
        assert format_timestamp(0) == "0:00"


# ============================================================
# Snippet Extraction
# ============================================================

class TestSnippetExtraction:
    def test_empty_text(self):
        assert _extract_snippet("", ["test"]) == ""

    def test_no_query_terms(self):
        text = "Hello world this is a test"
        result = _extract_snippet(text, [])
        assert result == text

    def test_long_text_no_match(self):
        text = "x" * 500
        result = _extract_snippet(text, ["hello"])
        assert len(result) <= 243  # 240 + "..."
        assert result.endswith("...")

    def test_snippet_contains_match(self):
        text = "This is a long text about Playwright MCP and browser automation."
        result = _extract_snippet(text, ["playwright"])
        assert "playwright" in result.lower()

    def test_multiple_terms(self):
        text = "The quick brown fox jumps over the lazy dog."
        result = _extract_snippet(text, ["fox", "dog"])
        assert "fox" in result or "dog" in result

    def test_short_text_exact(self):
        text = "Hello world"
        result = _extract_snippet(text, ["hello"])
        assert result == text

    def test_snippet_adds_ellipsis(self):
        text = "A" * 100 + " test " + "B" * 100
        result = _extract_snippet(text, ["test"])
        assert "test" in result
        assert len(result) <= 243


# ============================================================
# FTS Query Sanitization
# ============================================================

class TestFTSSanitization:
    def test_empty_query(self):
        assert _sanitize_fts_query("") == ""
        assert _sanitize_fts_query("   ") == ""

    def test_normal_query(self):
        assert _sanitize_fts_query("hello world") == "hello world"

    def test_query_with_null(self):
        result = _sanitize_fts_query("hello\0world")
        assert "\0" not in result

    def test_long_query_truncated(self):
        long_q = "a" * 1000
        result = _sanitize_fts_query(long_q)
        assert len(result) <= 500

    def test_phrase_query_escaping(self):
        result = _make_phrase_query('hello "world" test')
        assert result.startswith('"')
        assert result.endswith('"')

    def test_phrase_query_strips(self):
        result = _make_phrase_query("  hello  ")
        assert result == '"hello"'


# ============================================================
# Search Functionality (requires DB)
# ============================================================

def _setup_search_db(db_path: str) -> None:
    conn = get_db(db_path)
    init_db(conn)

    v1 = Video(
        video_id="vid1",
        title="Playwright Tutorial",
        source="youtube",
        url="https://youtube.com/watch?v=abc123",
        duration_sec=120.0,
    )
    v2 = Video(
        video_id="vid2",
        title="Python Programming",
        source="local",
        duration_sec=60.0,
    )
    insert_video(conn, v1)
    insert_video(conn, v2)

    m1 = Moment(
        moment_id=make_moment_id("vid1", 0.0, 10.0),
        video_id="vid1",
        start_sec=0.0,
        end_sec=10.0,
        transcript_text="Connect Playwright MCP so Claude can inspect the browser UI",
        ocr_text="Playwright Inspector Browser DOM",
        ideas=[
            Idea(
                idea_id="idea:vid1:0:0",
                moment_id=make_moment_id("vid1", 0.0, 10.0),
                type="step",
                text="Connect Playwright MCP",
            ),
            Idea(
                idea_id="idea:vid1:0:1",
                moment_id=make_moment_id("vid1", 0.0, 10.0),
                type="definition",
                text="MCP is a browser control protocol",
            ),
        ],
        keyframe_paths=["/frames/frame_000000.jpg", "/frames/frame_000005.jpg"],
    )
    m2 = Moment(
        moment_id=make_moment_id("vid1", 10.0, 20.0),
        video_id="vid1",
        start_sec=10.0,
        end_sec=20.0,
        transcript_text="Now we can write tests using the Playwright API",
        ocr_text="",
        ideas=[],
        keyframe_paths=[],
    )
    m3 = Moment(
        moment_id=make_moment_id("vid2", 0.0, 15.0),
        video_id="vid2",
        start_sec=0.0,
        end_sec=15.0,
        transcript_text="Python is a great programming language for beginners",
        ocr_text="Python Code Editor",
        ideas=[
            Idea(
                idea_id="idea:vid2:0:0",
                moment_id=make_moment_id("vid2", 0.0, 15.0),
                type="claim",
                text="Python is great for beginners",
            ),
        ],
        keyframe_paths=["/frames/frame_000000.jpg"],
    )
    m4 = Moment(
        moment_id=make_moment_id("vid2", 15.0, 30.0),
        video_id="vid2",
        start_sec=15.0,
        end_sec=30.0,
        transcript_text="Unlike Java, Python has simpler syntax",
        ocr_text="",
        ideas=[],
        keyframe_paths=[],
    )
    insert_moment(conn, m1)
    insert_moment(conn, m2)
    insert_moment(conn, m3)
    insert_moment(conn, m4)
    conn.commit()

    rebuild_fts(conn)
    conn.commit()
    conn.close()


@pytest.fixture
def search_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    _setup_search_db(db_path)
    yield Path(db_path)
    Path(db_path).unlink(missing_ok=True)


class TestSearch:
    def test_search_returns_relevant_moments(self, search_db):
        results = search_moments("playwright", search_db)
        assert len(results) > 0
        for r in results:
            assert "playwright" in r.transcript_snippet.lower() or "playwright" in r.ocr_snippet.lower()

    def test_search_joins_video_title(self, search_db):
        results = search_moments("playwright", search_db)
        assert len(results) > 0
        assert results[0].video_title == "Playwright Tutorial"
        assert results[0].video_id == "vid1"

    def test_search_supports_limit(self, search_db):
        results = search_moments("test", search_db, limit=1)
        assert len(results) <= 1

    def test_search_supports_video_id_filter(self, search_db):
        results = search_moments("python", search_db, video_id="vid2")
        assert len(results) > 0
        assert all(r.video_id == "vid2" for r in results)

    def test_search_handles_empty_query(self, search_db):
        results = search_moments("", search_db)
        assert results == []

    def test_search_handles_whitespace_query(self, search_db):
        results = search_moments("   ", search_db)
        assert results == []

    def test_search_handles_weird_punctuation(self, search_db):
        results = search_moments("@#$%^&*()", search_db)
        assert results is not None
        assert isinstance(results, list)

    def test_search_returns_ranked_results(self, search_db):
        results = search_moments("playwright browser", search_db)
        assert len(results) > 0
        assert results[0].rank == 1
        assert all(r.rank == i + 1 for i, r in enumerate(results))

    def test_search_has_idea_types(self, search_db):
        results = search_moments("playwright", search_db)
        if results:
            first = results[0]
            assert isinstance(first.idea_types, list)

    def test_search_has_timestamp_label(self, search_db):
        results = search_moments("playwright", search_db)
        if results:
            assert ":" in results[0].timestamp_label

    def test_search_no_results(self, search_db):
        results = search_moments("xyznonexistent12345", search_db)
        assert results == []

    def test_search_nonexistent_db(self):
        results = search_moments(
            "test", Path("/nonexistent/path/db.db")
        )
        assert results == []

    def test_search_keyframe_paths_present(self, search_db):
        results = search_moments("playwright", search_db)
        if results:
            first = results[0]
            assert isinstance(first.keyframe_paths, list)

    def test_search_source_url_present(self, search_db):
        results = search_moments("playwright", search_db)
        if results:
            first = results[0]
            assert first.source_url is not None

    def test_search_with_quotes(self, search_db):
        results = search_moments('"playwright MCP"', search_db)
        assert isinstance(results, list)

    def test_search_ocr_only_query(self, search_db):
        results = search_moments("Inspector", search_db)
        assert len(results) > 0

    def test_search_idea_only_query(self, search_db):
        results = search_moments("beginners", search_db)
        assert len(results) > 0
        assert any("Python" in r.video_title for r in results)

    def test_search_phrase_fallback(self, search_db):
        query = "playwright" + "!" * 50
        results = search_moments(query, search_db)
        assert isinstance(results, list)


# ============================================================
# Rebuild FTS Tests
# ============================================================

class TestRebuildFTS:
    def test_rebuild_fts_makes_moments_searchable(self, search_db):
        conn = sqlite3.connect(str(search_db))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        init_db(conn)

        v3 = Video(
            video_id="vid3", title="New Video", source="local",
            duration_sec=30.0,
        )
        insert_video(conn, v3)
        m5 = Moment(
            moment_id=make_moment_id("vid3", 0.0, 5.0),
            video_id="vid3",
            start_sec=0.0,
            end_sec=5.0,
            transcript_text="This is a brand new moment about Rust programming",
        )
        insert_moment(conn, m5)
        conn.commit()

        rebuild_fts(conn)
        conn.commit()
        conn.close()

        results = search_moments("Rust", search_db)
        assert len(results) > 0
        assert results[0].video_id == "vid3"

    def test_rebuild_fts_without_moments(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        conn = get_db(db_path)
        init_db(conn)
        rebuild_fts(conn)
        conn.commit()
        conn.close()
        Path(db_path).unlink(missing_ok=True)


# ============================================================
# CLI Search
# ============================================================

class TestCLISearch:
    def test_search_no_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app, ["search", "test", "--data-dir", tmpdir]
            )
            assert result.exit_code == 1
            assert "No database found" in result.stderr

    def test_search_empty_results(self, search_db):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["search", "xyznonexistent12345", "--data-dir", tmpdir]
            )
            assert "No results" in result.stdout

    def test_search_human_readable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            db_path = Path(tmpdir) / "vidcrawl.db"
            _setup_search_db(str(db_path))
            result = runner.invoke(
                app, ["search", "playwright", "--data-dir", tmpdir]
            )
            assert result.exit_code == 0
            assert "Playwright Tutorial" in result.stdout
            assert "score" in result.stdout

    def test_search_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            db_path = Path(tmpdir) / "vidcrawl.db"
            _setup_search_db(str(db_path))
            result = runner.invoke(
                app,
                ["search", "playwright", "--data-dir", tmpdir, "--json"],
            )
            assert result.exit_code == 0
            data = json.loads(result.stdout)
            assert isinstance(data, list)
            assert len(data) > 0
            assert "moment_id" in data[0]
            assert "score" in data[0]
            assert "video_title" in data[0]

    def test_search_with_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            db_path = Path(tmpdir) / "vidcrawl.db"
            _setup_search_db(str(db_path))
            result = runner.invoke(
                app,
                ["search", "playwright", "--data-dir", tmpdir, "--limit", "1"],
            )
            assert result.exit_code == 0
            lines = [
                l for l in result.stdout.splitlines()
                if l.strip() and l[0].isdigit()
            ]
            assert len(lines) <= 1

    def test_search_with_video_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            db_path = Path(tmpdir) / "vidcrawl.db"
            _setup_search_db(str(db_path))
            result = runner.invoke(
                app,
                [
                    "search", "playwright", "--data-dir", tmpdir,
                    "--video-id", "vid1",
                ],
            )
            assert result.exit_code == 0
            assert "vid1" in result.stdout


# ============================================================
# CLI Show
# ============================================================

class TestCLIShow:
    def test_show_no_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app, ["show", "test:0:10", "--data-dir", tmpdir]
            )
            assert result.exit_code == 1
            assert "No database found" in result.stderr

    def test_show_nonexistent_moment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["show", "nonexistent", "--data-dir", tmpdir]
            )
            assert result.exit_code == 1
            assert "not found" in result.stderr

    def test_show_moment_details(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            db_path = Path(tmpdir) / "vidcrawl.db"
            _setup_search_db(str(db_path))
            moment_id = "vid1:0.00:10.00"
            result = runner.invoke(
                app, ["show", moment_id, "--data-dir", tmpdir]
            )
            assert result.exit_code == 0
            assert "Moment ID" in result.stdout
            assert moment_id in result.stdout
            assert "Playwright" in result.stdout
            assert "Transcript" in result.stdout

    def test_show_displays_keyframes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            db_path = Path(tmpdir) / "vidcrawl.db"
            _setup_search_db(str(db_path))
            moment_id = "vid1:0.00:10.00"
            result = runner.invoke(
                app, ["show", moment_id, "--data-dir", tmpdir]
            )
            assert result.exit_code == 0
            assert "frame" in result.stdout.lower() or "Keyframes" in result.stdout


# ============================================================
# CLI Stats
# ============================================================

class TestCLIStats:
    def test_stats_no_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app, ["stats", "--data-dir", tmpdir]
            )
            assert result.exit_code == 1
            assert "No database found" in result.stderr

    def test_stats_empty_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["stats", "--data-dir", tmpdir]
            )
            assert result.exit_code == 0
            assert "Videos:" in result.stdout
            assert "0" in result.stdout.splitlines()[0] if "Videos:" in result.stdout.splitlines()[0] else True

    def test_stats_with_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            db_path = Path(tmpdir) / "vidcrawl.db"
            _setup_search_db(str(db_path))
            result = runner.invoke(
                app, ["stats", "--data-dir", tmpdir]
            )
            assert result.exit_code == 0
            assert "Videos:" in result.stdout
            assert "Moments:" in result.stdout
            assert "FTS Rows:" in result.stdout
            assert str(db_path) in result.stdout
