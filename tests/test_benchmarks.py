import tempfile
from pathlib import Path

import pytest

from vidcrawl.db import get_db, init_db
from vidcrawl.demo import create_demo_corpus
from vidcrawl.eval import evaluate_queries, format_eval_report, load_queries


@pytest.fixture
def demo_db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    create_demo_corpus(Path(path))
    yield Path(path)
    Path(path).unlink(missing_ok=True)


class TestBenchmarkEval:

    def test_load_demo_queries(self):
        queries = load_queries(None)
        assert len(queries) == 10

    def test_load_file_queries(self, tmp_path):
        qfile = tmp_path / "queries.json"
        qfile.write_text('{"queries": [{"query": "test"}]}')
        queries = load_queries(str(qfile))
        assert len(queries) == 1

    def test_eval_all_modes(self, demo_db_path):
        from vidcrawl.graph.build import build_graph
        build_graph(demo_db_path)
        queries = load_queries(None)
        for mode, rerank in [("raw", False), ("rerank", True)]:
            metrics = evaluate_queries(demo_db_path, queries, use_rerank=rerank)
            assert metrics["total_queries"] == 10
            assert "top1_pct" in metrics

    def test_eval_json_output(self, demo_db_path):
        from vidcrawl.graph.build import build_graph
        build_graph(demo_db_path)
        queries = load_queries(None)
        metrics = evaluate_queries(demo_db_path, queries, use_rerank=True)
        assert isinstance(metrics, dict)
        assert "graph_node_count" in metrics

    def test_format_report(self, demo_db_path):
        from vidcrawl.graph.build import build_graph
        build_graph(demo_db_path)
        queries = load_queries(None)
        metrics = evaluate_queries(demo_db_path, queries, use_rerank=True)
        report = format_eval_report(metrics)
        assert "Evaluation Results" in report
        assert "Graph Metrics" in report


class TestReportCLI:

    def test_report_generate(self, demo_db_path, tmp_path):
        from typer.testing import CliRunner
        from vidcrawl.cli import app
        from vidcrawl.graph.build import build_graph
        from vidcrawl.db import get_db, init_db
        from vidcrawl.demo import create_demo_corpus

        db_file = tmp_path / "data" / "vidcrawl.db"
        db_file.parent.mkdir(parents=True, exist_ok=True)
        create_demo_corpus(db_file)
        build_graph(db_file)

        runner = CliRunner()
        out_file = tmp_path / "report.md"
        result = runner.invoke(app, [
            "report", "generate", "--out", str(out_file),
            "--data-dir", str(tmp_path / "data"),
        ])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert out_file.exists()
        content = out_file.read_text()
        assert "VidCrawl" in content
