import json
import tempfile
from pathlib import Path

import pytest

from vidcrawl.db import get_db, init_db
from vidcrawl.claims.extract import (
    extract_claims_from_text,
    run_claim_extraction,
    get_claim_stats,
    Claim,
)
from vidcrawl.claims.normalize import normalize_claim
from vidcrawl.claims.cluster import detect_contradictions
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


# ---- Claim Extraction Tests ----

class TestClaimExtraction:

    def test_extract_definition(self):
        claims = extract_claims_from_text("m1", "MCP is a protocol for AI assistants.")
        assert len(claims) > 0
        types = [c.claim_type for c in claims]
        assert "definition" in types

    def test_extract_warning(self):
        claims = extract_claims_from_text("m1", "Be careful when running headless browsers.")
        assert len(claims) > 0
        types = [c.claim_type for c in claims]
        assert "warning" in types

    def test_extract_step(self):
        claims = extract_claims_from_text("m1", "First, install the Playwright package using npm.")
        assert len(claims) > 0
        types = [c.claim_type for c in claims]
        assert "step" in types

    def test_extract_comparison(self):
        claims = extract_claims_from_text("m1", "Playwright is faster than Selenium.")
        assert len(claims) > 0
        types = [c.claim_type for c in claims]
        assert "comparison" in types

    def test_extract_limitation(self):
        claims = extract_claims_from_text("m1", "However, the attention mechanism has quadratic complexity.")
        assert len(claims) > 0
        types = [c.claim_type for c in claims]
        assert "limitation" in types

    def test_extract_claim_has_normalized_text(self):
        claims = extract_claims_from_text("m1", "MCP is a protocol for AI assistants.")
        assert len(claims) > 0
        assert len(claims[0].normalized_claim) > 0
        assert "protocol" in claims[0].normalized_claim

    def test_run_claim_extraction(self, demo_db_path):
        from vidcrawl.graph.build import build_graph
        build_graph(demo_db_path)
        result = run_claim_extraction(str(demo_db_path))
        assert result["total_claims"] > 0
        assert "by_type" in result

    def test_claim_graph_nodes(self, demo_db_path):
        from vidcrawl.graph.build import build_graph
        conn = get_db(demo_db_path)
        build_graph(demo_db_path)
        run_claim_extraction(str(demo_db_path))
        nodes = conn.execute(
            "SELECT COUNT(*) as c FROM graph_nodes WHERE node_type='claim'"
        ).fetchone()["c"]
        assert nodes > 0
        conn.close()


# ---- Normalization Tests ----

class TestClaimNormalization:

    def test_normalize_lowercase(self):
        assert normalize_claim("MCP IS A PROTOCOL") == "mcp protocol"

    def test_normalize_strip_punctuation(self):
        result = normalize_claim("Hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_normalize_stop_words(self):
        result = normalize_claim("The cat is on the mat")
        assert "the" not in result.split()
        assert "is" not in result.split()


# ---- Contradiction Tests ----

class TestContradictions:

    def test_detect_contradictions(self, demo_db_path):
        from vidcrawl.graph.build import build_graph
        build_graph(demo_db_path)
        conn = get_db(demo_db_path)
        run_claim_extraction(str(demo_db_path))
        conn.close()
        contradictions = detect_contradictions(str(demo_db_path))
        assert isinstance(contradictions, list)

    def test_no_false_contradictions(self, db_path):
        from vidcrawl.graph.build import build_graph
        from vidcrawl.claims.extract import run_claim_extraction
        build_graph(db_path)
        run_claim_extraction(str(db_path))
        contradictions = detect_contradictions(str(db_path))
        assert len(contradictions) == 0


# ---- Claim Stats Tests ----

class TestClaimStats:

    def test_stats(self, demo_db_path):
        from vidcrawl.graph.build import build_graph
        build_graph(demo_db_path)
        run_claim_extraction(str(demo_db_path))
        stats = get_claim_stats(str(demo_db_path))
        assert stats["total_claims"] > 0
        assert len(stats["by_type"]) > 0

    def test_stats_empty(self, db_path):
        stats = get_claim_stats(str(db_path))
        assert stats["total_claims"] == 0


# ---- CLI Tests ----

class TestClaimsCLI:

    def test_claims_extract_cli(self, db_path, tmp_path):
        from typer.testing import CliRunner
        from vidcrawl.cli import app
        from vidcrawl.db import get_db, init_db
        from vidcrawl.graph.build import build_graph
        db_file = tmp_path / "data" / "vidcrawl.db"
        db_file.parent.mkdir(parents=True, exist_ok=True)
        conn = get_db(db_file)
        init_db(conn)
        conn.close()
        create_demo_corpus(db_file)
        build_graph(db_file)
        runner = CliRunner()
        result = runner.invoke(app, [
            "claims", "extract",
            "--data-dir", str(tmp_path / "data"),
        ])
        assert result.exit_code == 0

    def test_claims_stats_cli(self, db_path, tmp_path):
        from typer.testing import CliRunner
        from vidcrawl.cli import app
        from vidcrawl.db import get_db, init_db
        from vidcrawl.graph.build import build_graph
        db_file = tmp_path / "data" / "vidcrawl.db"
        db_file.parent.mkdir(parents=True, exist_ok=True)
        conn = get_db(db_file)
        init_db(conn)
        conn.close()
        create_demo_corpus(db_file)
        build_graph(db_file)
        from vidcrawl.claims.extract import run_claim_extraction
        run_claim_extraction(str(db_file))
        runner = CliRunner()
        result = runner.invoke(app, [
            "claims", "stats",
            "--data-dir", str(tmp_path / "data"),
        ])
        assert result.exit_code == 0
