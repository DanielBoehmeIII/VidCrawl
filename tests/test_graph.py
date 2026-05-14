import json
import tempfile
from pathlib import Path

import pytest

from vidcrawl.db import get_db, init_db, insert_video, insert_moment, insert_idea, insert_evidence, insert_duplicate
from vidcrawl.db import make_moment_id, make_idea_id, generate_evidence_id, generate_dup_id
from vidcrawl.models import Video, Moment, Idea, Evidence, Duplicate
from vidcrawl.graph.entities import extract_entities, KNOWN_TERMS
from vidcrawl.graph.build import build_graph, GraphBuildSummary, init_graph_tables, clear_graph
from vidcrawl.graph.stats import compute_graph_stats
from vidcrawl.graph.query import get_node, get_neighbors, get_graph_context_for_moment
from vidcrawl.graph.export import export_graph


# ---- Fixtures ----

@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    conn = get_db(path)
    init_db(conn)
    conn.close()
    yield Path(path)
    Path(path).unlink(missing_ok=True)


def _populate_minimal_corpus(conn):
    video = Video(
        video_id="test_vid",
        title="Test Video",
        source="local",
        url=None,
        duration_sec=100.0,
        status="ready",
    )
    insert_video(conn, video)

    moment = Moment(
        moment_id=make_moment_id("test_vid", 0.0, 10.0),
        video_id="test_vid",
        start_sec=0.0,
        end_sec=10.0,
        transcript_text="This is a test about Playwright MCP server.",
        ocr_text="Playwright MCP Server",
        ideas=[],
    )
    insert_moment(conn, moment)

    idea = Idea(
        idea_id=make_idea_id(moment.moment_id, 0),
        moment_id=moment.moment_id,
        type="definition",
        text="MCP is a protocol for AI to interact with browsers",
    )
    insert_idea(conn, idea)

    ev = Evidence(
        evidence_id=generate_evidence_id(),
        moment_id=moment.moment_id,
        modality="transcript",
        content="This is a test about Playwright MCP server.",
    )
    insert_evidence(conn, ev)

    conn.commit()
    return video, moment, idea, ev


# ---- Entity Extraction Tests ----

class TestEntityExtraction:

    def test_extract_acronyms(self):
        entities = extract_entities("MCP API OCR ASR SQLite FTS5")
        labels = {e["label"] for e in entities}
        assert "MCP" in labels
        assert "API" in labels
        assert "OCR" in labels
        assert "ASR" in labels

    def test_extract_capitalized_phrases(self):
        entities = extract_entities("Playwright MCP Server and Transformer Architecture")
        labels = {e["label"] for e in entities}
        assert "Playwright MCP Server" in labels or "Playwright" in labels
        assert "Transformer Architecture" in labels

    def test_extract_code_identifiers(self):
        entities = extract_entities("call page.goto() and then page.click()")
        labels = {e["label"] for e in entities}
        assert any("page.goto" in l for l in labels) or any("page.click" in l for l in labels)

    def test_extract_file_paths(self):
        entities = extract_entities("Save to /home/user/file.txt and /tmp/data.json")
        labels = {e["label"] for e in entities}
        assert any("file.txt" in l for l in labels)

    def test_known_terms(self):
        entities = extract_entities("We use Playwright for browser testing and Tesseract for OCR")
        labels = {e["label"] for e in entities}
        assert "Playwright" in labels
        assert "Tesseract" in labels

    def test_no_entities(self):
        entities = extract_entities("this is all lowercase text with nothing special")
        assert len(entities) == 0

    def test_multiple_texts(self):
        entities = extract_entities("MCP is great", "Playwright MCP Server")
        labels = {e["label"] for e in entities}
        assert "MCP" in labels


# ---- Graph Build Tests ----

class TestGraphBuild:

    def test_graph_tables_created(self, db_path):
        conn = get_db(db_path)
        init_graph_tables(conn)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('graph_nodes', 'graph_edges')"
        ).fetchall()
        conn.close()
        assert len(tables) == 2

    def test_build_creates_nodes(self, db_path):
        conn = get_db(db_path)
        _populate_minimal_corpus(conn)
        conn.close()

        summary = build_graph(db_path)
        assert summary.nodes_created >= 5  # video + moment + idea + evidence + entity nodes
        assert summary.video_nodes >= 1
        assert summary.moment_nodes >= 1
        assert summary.idea_nodes >= 1
        assert summary.evidence_nodes >= 1

    def test_build_creates_edges(self, db_path):
        conn = get_db(db_path)
        _populate_minimal_corpus(conn)
        conn.close()

        summary = build_graph(db_path)
        assert summary.edges_created >= 1

    def test_build_idempotent(self, db_path):
        conn = get_db(db_path)
        _populate_minimal_corpus(conn)
        conn.close()

        summary1 = build_graph(db_path)
        summary2 = build_graph(db_path)
        assert summary2.nodes_created == 0
        assert summary2.edges_created == 0

    def test_rebuild_clears_and_remakes(self, db_path):
        conn = get_db(db_path)
        _populate_minimal_corpus(conn)
        conn.close()

        build_graph(db_path)
        conn = get_db(db_path)
        count_before = conn.execute("SELECT COUNT(*) as c FROM graph_nodes").fetchone()["c"]
        conn.close()
        assert count_before > 0

        build_graph(db_path, rebuild=True)
        conn = get_db(db_path)
        count_after = conn.execute("SELECT COUNT(*) as c FROM graph_nodes").fetchone()["c"]
        conn.close()
        assert count_after == count_before

    def test_duplicate_cluster_nodes(self, db_path):
        conn = get_db(db_path)
        v, m1, idea, ev = _populate_minimal_corpus(conn)

        m2 = Moment(
            moment_id=make_moment_id("test_vid", 10.0, 20.0),
            video_id="test_vid",
            start_sec=10.0,
            end_sec=20.0,
            transcript_text="Duplicate content here.",
            ocr_text="",
        )
        insert_moment(conn, m2)

        dup = Duplicate(
            dup_id=generate_dup_id(),
            moment_id=m2.moment_id,
            canonical_moment_id=m1.moment_id,
            similarity_score=1.0,
            novelty_score=0.0,
            method="exact_hash",
            duplicate_type="exact",
            reason="",
        )
        insert_duplicate(conn, dup)
        conn.commit()
        conn.close()

        summary = build_graph(db_path)
        assert summary.cluster_nodes >= 1

        conn = get_db(db_path)
        cluster_count = conn.execute(
            "SELECT COUNT(*) as c FROM graph_nodes WHERE node_type='duplicate_cluster'"
        ).fetchone()["c"]
        conn.close()
        assert cluster_count >= 1


# ---- Graph Stats Tests ----

class TestGraphStats:

    def test_stats_works(self, db_path):
        conn = get_db(db_path)
        _populate_minimal_corpus(conn)
        conn.close()
        build_graph(db_path)

        stats = compute_graph_stats(db_path)
        assert stats.total_nodes > 0
        assert stats.total_edges > 0
        assert len(stats.nodes_by_type) > 0
        assert len(stats.edges_by_type) > 0

    def test_stats_empty_graph(self, db_path):
        stats = compute_graph_stats(db_path)
        assert stats.total_nodes == 0
        assert stats.total_edges == 0


# ---- Graph Show Tests ----

class TestGraphShow:

    def test_show_node_by_id(self, db_path):
        conn = get_db(db_path)
        _populate_minimal_corpus(conn)
        conn.close()
        build_graph(db_path)

        node = get_node(db_path, "moment:test_vid:0.00:10.00")
        assert node is not None
        assert node.node_type == "moment"

    def test_show_node_by_ref_id(self, db_path):
        conn = get_db(db_path)
        _populate_minimal_corpus(conn)
        conn.close()
        build_graph(db_path)

        node = get_node(db_path, "test_vid")
        assert node is not None

    def test_show_nonexistent(self, db_path):
        node = get_node(db_path, "nonexistent")
        assert node is None


# ---- Graph Neighbors Tests ----

class TestGraphNeighbors:

    def test_neighbors_works(self, db_path):
        conn = get_db(db_path)
        _populate_minimal_corpus(conn)
        conn.close()
        build_graph(db_path)

        result = get_neighbors(db_path, "test_vid")
        assert result
        assert "node" in result
        assert "edges" in result
        assert "neighbors" in result
        assert result["degree"] >= 1

    def test_neighbors_nonexistent(self, db_path):
        result = get_neighbors(db_path, "nonexistent")
        assert result == {}


# ---- Graph Export Tests ----

class TestGraphExport:

    def test_export_json(self, db_path):
        conn = get_db(db_path)
        _populate_minimal_corpus(conn)
        conn.close()
        build_graph(db_path)

        result = export_graph(db_path, fmt="json")
        assert "graph" in result
        assert "nodes" in result["graph"]
        assert "edges" in result["graph"]
        assert len(result["graph"]["nodes"]) > 0

    def test_export_to_file(self, db_path, tmp_path):
        conn = get_db(db_path)
        _populate_minimal_corpus(conn)
        conn.close()
        build_graph(db_path)

        out = tmp_path / "graph.json"
        result = export_graph(db_path, output_path=out, fmt="json")
        assert out.exists()
        with open(out) as f:
            data = json.load(f)
        assert "graph" in data


# ---- Graph Context (Search Integration) Tests ----

class TestGraphContext:

    def test_graph_context_basic(self, db_path):
        conn = get_db(db_path)
        _populate_minimal_corpus(conn)
        conn.close()
        build_graph(db_path)

        ctx = get_graph_context_for_moment(db_path, "test_vid:0.00:10.00")
        assert ctx is not None
        assert ctx.idea_count >= 1

    def test_graph_context_no_graph(self, db_path):
        ctx = get_graph_context_for_moment(db_path, "test_vid:0.00:10.00")
        assert ctx is not None
        assert ctx.idea_count == 0

    def test_graph_context_entities(self, db_path):
        conn = get_db(db_path)
        _populate_minimal_corpus(conn)
        conn.close()
        build_graph(db_path)

        ctx = get_graph_context_for_moment(db_path, "test_vid:0.00:10.00")
        assert len(ctx.entities) >= 1 or ctx.idea_count >= 1


# ---- Eval Graph Metrics ----

class TestEvalGraphMetrics:

    def test_compute_graph_metrics(self, db_path):
        from vidcrawl.eval import compute_graph_metrics

        conn = get_db(db_path)
        _populate_minimal_corpus(conn)
        conn.close()
        build_graph(db_path)

        metrics = compute_graph_metrics(db_path)
        assert "graph_node_count" in metrics
        assert "graph_edge_count" in metrics
        assert metrics["graph_node_count"] > 0

    def test_compute_graph_metrics_no_graph(self, db_path):
        from vidcrawl.eval import compute_graph_metrics
        metrics = compute_graph_metrics(db_path)
        assert metrics.get("graph_node_count", -1) == 0
