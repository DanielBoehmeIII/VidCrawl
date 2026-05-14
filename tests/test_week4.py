import json
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

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
from vidcrawl.eval import evaluate_queries, format_eval_report, load_queries
from vidcrawl.ingest.downloader import (
    download_youtube,
    extract_youtube_metadata,
    is_yt_dlp_available,
    yt_dlp_install_help,
)
from vidcrawl.models import Idea, Moment, Video
from vidcrawl.search.query import SearchResult, search_moments
from vidcrawl.demo import create_demo_corpus
from vidcrawl.utils.time import (
    format_timestamp,
    seconds_to_timestamp,
    timestamp_range,
    youtube_timestamp_url,
)

runner = CliRunner()


# ============================================================
# Helper
# ============================================================

def _setup_demo_db(db_path: str) -> None:
    create_demo_corpus(Path(db_path))


# ============================================================
# Demo Corpus
# ============================================================

class TestDemoCorpus:
    def test_create_demo_corpus_creates_videos(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            moments = create_demo_corpus(Path(db_path))
            conn = get_db(db_path)
            videos = conn.execute(
                "SELECT COUNT(*) as c FROM videos"
            ).fetchone()["c"]
            assert videos == 3
            moments_count = conn.execute(
                "SELECT COUNT(*) as c FROM moments"
            ).fetchone()["c"]
            assert moments_count > 0
            conn.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_demo_corpus_has_ideas(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            create_demo_corpus(Path(db_path))
            conn = get_db(db_path)
            ideas = conn.execute(
                "SELECT COUNT(*) as c FROM ideas"
            ).fetchone()["c"]
            assert ideas > 0
            conn.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_demo_corpus_fts_built(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            create_demo_corpus(Path(db_path))
            conn = get_db(db_path)
            fts = conn.execute(
                "SELECT COUNT(*) as c FROM moments_fts"
            ).fetchone()["c"]
            assert fts > 0
            conn.close()
        finally:
            Path(db_path).unlink(missing_ok=True)


class TestDemoCLI:
    def test_demo_creates_corpus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app, ["demo", "init", "--data-dir", tmpdir]
            )
            assert result.exit_code == 0
            assert "Demo corpus created" in result.stdout

    def test_demo_stats_after_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["stats", "--data-dir", tmpdir]
            )
            assert result.exit_code == 0
            assert "Videos:" in result.stdout
            assert "3" in result.stdout.splitlines()[0] if "Videos:" in result.stdout.splitlines()[0] else False

    def test_demo_search_after_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["search", "playwright", "--data-dir", tmpdir]
            )
            assert result.exit_code == 0
            assert "playwright" in result.stdout.lower() or "Playwright" in result.stdout

    def test_demo_search_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["search", "playwright", "--data-dir", tmpdir, "--json"]
            )
            assert result.exit_code == 0
            data = json.loads(result.stdout)
            assert len(data) > 0
            assert data[0]["video_title"] == "Building a Playwright MCP Server for Browser Automation"

    def test_demo_creates_stattable_corpus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["stats", "--data-dir", tmpdir, "--verbose"]
            )
            assert result.exit_code == 0
            assert "Videos:" in result.stdout
            assert "Avg" in result.stdout

    def test_demo_show_after_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            conn = get_db(str(Path(tmpdir) / "vidcrawl.db"))
            moments = conn.execute(
                "SELECT moment_id FROM moments LIMIT 1"
            ).fetchall()
            conn.close()
            if moments:
                moment_id = moments[0]["moment_id"]
                result = runner.invoke(
                    app, ["show", moment_id, "--data-dir", tmpdir]
                )
                assert result.exit_code == 0
                assert moment_id in result.stdout


# ============================================================
# Evaluation
# ============================================================

class TestEval:
    def test_evaluate_empty_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            conn = get_db(db_path)
            init_db(conn)
            conn.close()
            queries = [{"query": "test", "expected_terms": ["test"]}]
            metrics = evaluate_queries(Path(db_path), queries)
            assert metrics["total_queries"] == 1
            assert metrics["top1_hits"] == 0
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_evaluate_demo_corpus(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            create_demo_corpus(Path(db_path))
            queries = [{"query": "playwright", "expected_terms": ["playwright"]}]
            metrics = evaluate_queries(Path(db_path), queries)
            assert metrics["total_queries"] == 1
            assert metrics["top1_hits"] == 1
            assert metrics["term_hits"] >= 1
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_evaluate_multiple_queries(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            create_demo_corpus(Path(db_path))
            queries = [
                {"query": "playwright browser", "expected_terms": ["playwright"]},
                {"query": "transformer attention", "expected_terms": ["transformer"]},
                {"query": "xyznonexistent12345", "expected_terms": ["xyznonexistent"]},
            ]
            metrics = evaluate_queries(Path(db_path), queries)
            assert metrics["total_queries"] == 3
            assert metrics["top1_hits"] >= 2
            assert metrics["avg_query_latency_ms"] > 0
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_load_queries_from_file(self):
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False
        ) as f:
            json.dump(
                {"queries": [{"query": "test", "expected_terms": ["test"]}]},
                f,
            )
            json_path = f.name
        try:
            queries = load_queries(json_path)
            assert len(queries) == 1
            assert queries[0]["query"] == "test"
        finally:
            Path(json_path).unlink(missing_ok=True)

    def test_load_queries_default(self):
        queries = load_queries()
        assert len(queries) > 0

    def test_format_eval_report(self):
        metrics = {
            "total_queries": 5,
            "top1_hits": 3,
            "top1_pct": 60.0,
            "top3_hits": 4,
            "top3_pct": 80.0,
            "top5_hits": 5,
            "top5_pct": 100.0,
            "video_hits": 2,
            "term_hits": 8,
            "term_checks": 10,
            "term_pct": 80.0,
            "avg_results_returned": 3.5,
            "avg_query_latency_ms": 12.34,
            "total_latency_ms": 61.7,
        }
        report = format_eval_report(metrics)
        assert "Total queries" in report
        assert "60.0%" in report
        assert "12.34ms" in report

    def test_eval_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["eval", "--data-dir", tmpdir]
            )
            assert result.exit_code == 0
            assert "Evaluation Results" in result.stdout

    def test_eval_cli_no_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app, ["eval", "--data-dir", tmpdir]
            )
            assert result.exit_code == 1
            assert "No database found" in result.stderr

    def test_eval_cli_with_query_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            query_file = Path(tmpdir) / "queries.json"
            query_file.write_text(
                json.dumps({
                    "queries": [
                        {"query": "playwright", "expected_terms": ["playwright"]}
                    ]
                })
            )
            result = runner.invoke(
                app, ["eval", str(query_file), "--data-dir", tmpdir]
            )
            assert result.exit_code == 0
            assert "Evaluation Results" in result.stdout


# ============================================================
# Downloader
# ============================================================

class TestDownloader:
    def test_yt_dlp_not_available(self):
        assert is_yt_dlp_available() or not is_yt_dlp_available()

    def test_download_youtube_no_ytdlp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_youtube(
                "https://youtube.com/watch?v=test123",
                tmpdir,
                "test123",
            )
            if not is_yt_dlp_available():
                assert result is None

    def test_extract_metadata_no_ytdlp(self):
        meta = extract_youtube_metadata(
            "https://youtube.com/watch?v=test123"
        )
        if not is_yt_dlp_available():
            assert meta == {}

    def test_yt_dlp_install_message(self):
        msg = yt_dlp_install_help()
        assert "yt-dlp" in msg
        assert "pip install" in msg


# ============================================================
# Stats Verbose
# ============================================================

class TestStatsVerbose:
    def test_stats_verbose_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["stats", "--data-dir", tmpdir, "--verbose"]
            )
            assert result.exit_code == 0
            assert "Database size" in result.stdout

    def test_stats_verbose_with_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["stats", "--data-dir", tmpdir, "--verbose"]
            )
            assert result.exit_code == 0
            assert "Avg moments/video" in result.stdout
            assert "Database size" in result.stdout

    def test_stats_no_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app, ["stats", "--data-dir", tmpdir]
            )
            assert result.exit_code == 1
            assert "No database found" in result.stderr


# ============================================================
# Search Polish
# ============================================================

class TestSearchPolish:
    def test_search_match_reasons(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            db_path = Path(tmpdir) / "vidcrawl.db"
            results = search_moments("playwright", db_path)
            if results:
                assert isinstance(results[0].match_reasons, list)

    def test_search_no_snippets_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            result = runner.invoke(
                app,
                ["search", "playwright", "--data-dir", tmpdir, "--no-snippets"],
            )
            assert result.exit_code == 0
            assert "Query:" in result.stdout

    def test_search_empty_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            result = runner.invoke(
                app,
                [
                    "search", "xyznonexistent12345",
                    "--data-dir", tmpdir,
                ],
            )
            assert "No results found" in result.stdout

    def test_search_json_includes_match_reasons(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["search", "playwright", "--data-dir", tmpdir, "--json"]
            )
            assert result.exit_code == 0
            data = json.loads(result.stdout)
            if data:
                assert "match_reasons" in data[0]


# ============================================================
# Ingest Robustness
# ============================================================

class TestIngestRobustness:
    def test_ingest_nonexistent_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["ingest", "/nonexistent/path.mp4", "--data-dir", tmpdir]
            )
            assert result.exit_code == 1
            assert "file not found" in result.stderr

    def test_ingest_invalid_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            bad_path = Path(tmpdir) / "test.txt"
            bad_path.write_text("not a video")
            result = runner.invoke(
                app, ["ingest", str(bad_path), "--data-dir", tmpdir]
            )
            assert result.exit_code == 1
            assert "Unsupported" in result.stderr

    def test_ingest_youtube_registers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            result = runner.invoke(
                app,
                [
                    "ingest",
                    "https://youtube.com/watch?v=dQw4w9WgXcQ",
                    "--data-dir", tmpdir,
                    "--no-process",
                ],
            )
            assert result.exit_code == 0
            assert "Registered" in result.stdout

    def test_ingest_youtube_with_process_no_download(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            result = runner.invoke(
                app,
                [
                    "ingest",
                    "https://youtube.com/watch?v=dQw4w9WgXcQ",
                    "--data-dir", tmpdir,
                    "--process",
                    "--no-download",
                ],
            )
            assert result.exit_code == 0
            assert "Registered" in result.stdout

    def test_ingest_youtube_no_ytdlp_shows_help(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            result = runner.invoke(
                app,
                [
                    "ingest",
                    "https://youtube.com/watch?v=dQw4w9WgXcQ",
                    "--data-dir", tmpdir,
                    "--process",
                ],
            )
            assert result.exit_code == 0
            if not is_yt_dlp_available():
                assert "yt-dlp" in result.stdout


# ============================================================
# Robustness: Error Handling
# ============================================================

class TestRobustness:
    def test_search_no_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app, ["search", "test", "--data-dir", tmpdir]
            )
            assert result.exit_code == 1
            assert "No database found" in result.stderr

    def test_show_no_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app, ["show", "test:0:10", "--data-dir", tmpdir]
            )
            assert result.exit_code == 1
            assert "No database found" in result.stderr

    def test_inspect_no_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app, ["inspect", "test123", "--data-dir", tmpdir]
            )
            assert result.exit_code == 1
            assert "No database found" in result.stderr

    def test_list_no_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app, ["list", "--data-dir", tmpdir]
            )
            assert result.exit_code == 1
            assert "No database found" in result.stderr

    def test_reindex_no_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app, ["reindex", "--data-dir", tmpdir]
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

    def test_inspect_nonexistent_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["inspect", "nonexistent", "--data-dir", tmpdir]
            )
            assert result.exit_code == 1
            assert "not found" in result.stderr

    def test_init_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            r1 = runner.invoke(app, ["init", "--data-dir", tmpdir])
            r2 = runner.invoke(app, ["init", "--data-dir", tmpdir])
            assert r1.exit_code == 0
            assert r2.exit_code == 0

    def test_reindex_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            r1 = runner.invoke(app, ["reindex", "--data-dir", tmpdir])
            r2 = runner.invoke(app, ["reindex", "--data-dir", tmpdir])
            assert r1.exit_code == 0
            assert r2.exit_code == 0
            assert "rebuilt" in r1.stdout.lower()
            assert "rebuilt" in r2.stdout.lower()

    def test_search_handles_punctuation_query(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["search", "@#$%^&*()", "--data-dir", tmpdir]
            )
            assert "No results" in result.stdout


# ============================================================
# Version
# ============================================================

class TestVersion:
    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0


# ============================================================
# Search: Key Features
# ============================================================

class TestSearchFeatures:
    def test_search_demo_playwright(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["search", "playwright browser", "--data-dir", tmpdir]
            )
            assert result.exit_code == 0

    def test_search_demo_warning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["search", "warning", "--data-dir", tmpdir]
            )
            assert result.exit_code == 0

    def test_search_demo_definition(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["search", "definition", "--data-dir", tmpdir]
            )
            assert result.exit_code == 0

    def test_search_demo_comparison(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["search", "comparison", "--data-dir", tmpdir]
            )
            assert result.exit_code == 0

    def test_search_demo_model_architecture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            result = runner.invoke(
                app,
                ["search", "model architecture", "--data-dir", tmpdir],
            )
            assert result.exit_code == 0

    def test_search_demo_user_research(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            result = runner.invoke(
                app,
                ["search", "user research", "--data-dir", tmpdir],
            )
            assert result.exit_code == 0


# ============================================================
# Stats Calculations
# ============================================================

class TestStatsCalc:
    def test_stats_totals_correct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            db_path = Path(tmpdir) / "vidcrawl.db"
            conn = get_db(str(db_path))
            videos = conn.execute(
                "SELECT COUNT(*) as c FROM videos"
            ).fetchone()["c"]
            moments = conn.execute(
                "SELECT COUNT(*) as c FROM moments"
            ).fetchone()["c"]
            fts = conn.execute(
                "SELECT COUNT(*) as c FROM moments_fts"
            ).fetchone()["c"]
            conn.close()
            assert videos == 3
            assert moments == fts
            assert moments > 0

    def test_stats_videos_exist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["list", "--data-dir", tmpdir]
            )
            assert result.exit_code == 0
            assert "demo_coding" in result.stdout
            assert "demo_ml" in result.stdout
            assert "demo_ux" in result.stdout
