import tempfile
from pathlib import Path

import pytest

from vidcrawl.db import get_db, init_db
from vidcrawl.freshness import compute_freshness_score, run_freshness_scoring, get_freshness_stats
from vidcrawl.demo import create_demo_corpus


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    conn = get_db(path)
    init_db(conn)
    conn.close()
    yield Path(path)
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def demo_db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    create_demo_corpus(Path(path))
    yield Path(path)
    Path(path).unlink(missing_ok=True)


class TestFreshnessScoring:

    def test_fresh_score_default(self):
        score = compute_freshness_score("This is a normal description")
        assert 0.5 <= score <= 1.0

    def test_stale_keyword_deprecated(self):
        score = compute_freshness_score("This API is deprecated")
        assert score < 0.7

    def test_stale_keyword_outdated(self):
        score = compute_freshness_score("This old library is outdated")
        assert score < 0.7

    def test_fresh_keyword_new_version(self):
        score = compute_freshness_score("This is the new version of the framework")
        assert score > 0.7

    def test_tool_decay(self):
        score = compute_freshness_score("Install the package using npm")
        assert score == 0.9

    def test_run_scoring(self, demo_db_path):
        result = run_freshness_scoring(str(demo_db_path))
        assert result["scored"] > 0
        assert "stale" in result
        assert "fresh" in result

    def test_stats(self, demo_db_path):
        run_freshness_scoring(str(demo_db_path))
        stats = get_freshness_stats(str(demo_db_path))
        assert stats["total_scored"] > 0
        assert stats["average_freshness"] > 0

    def test_cli_freshness(self, demo_db_path, tmp_path):
        from typer.testing import CliRunner
        from vidcrawl.cli import app
        from vidcrawl.db import get_db, init_db
        db_file = tmp_path / "data" / "vidcrawl.db"
        db_file.parent.mkdir(parents=True, exist_ok=True)
        conn = get_db(db_file)
        init_db(conn)
        conn.close()
        create_demo_corpus(db_file)
        runner = CliRunner()
        result = runner.invoke(app, [
            "freshness", "run",
            "--data-dir", str(tmp_path / "data"),
        ])
        assert result.exit_code == 0

    def test_cli_stats(self, demo_db_path, tmp_path):
        from typer.testing import CliRunner
        from vidcrawl.cli import app
        from vidcrawl.db import get_db, init_db
        db_file = tmp_path / "data" / "vidcrawl.db"
        db_file.parent.mkdir(parents=True, exist_ok=True)
        conn = get_db(db_file)
        init_db(conn)
        conn.close()
        create_demo_corpus(db_file)
        run_freshness_scoring(str(db_file))
        runner = CliRunner()
        result = runner.invoke(app, [
            "freshness", "stats",
            "--data-dir", str(tmp_path / "data"),
        ])
        assert result.exit_code == 0
