from pathlib import Path


DOCS_DIR = Path(__file__).parent.parent / "docs"


class TestDocs:

    def test_architecture_doc_exists(self):
        assert (DOCS_DIR / "ARCHITECTURE.md").exists()

    def test_research_proposal_exists(self):
        assert (DOCS_DIR / "RESEARCH_PROPOSAL.md").exists()

    def test_research_proposal_has_abstract(self):
        text = (DOCS_DIR / "RESEARCH_PROPOSAL.md").read_text()
        assert "Abstract" in text

    def test_research_proposal_has_hypothesis(self):
        text = (DOCS_DIR / "RESEARCH_PROPOSAL.md").read_text()
        assert "Hypothesis" in text

    def test_architecture_has_modules(self):
        text = (DOCS_DIR / "ARCHITECTURE.md").read_text()
        assert "vidcrawl/" in text


class TestWalkthrough:

    def test_walkthrough_script_exists(self):
        script = Path(__file__).parent.parent / "scripts" / "demo_walkthrough.sh"
        assert script.exists()

    def test_walkthrough_has_all_commands(self):
        script = Path(__file__).parent.parent / "scripts" / "demo_walkthrough.sh"
        text = script.read_text()
        commands = ["init", "demo init", "dedupe run", "graph build",
                     "embed build", "technical extract", "claims extract",
                     "freshness run", "search", "eval", "report generate"]
        for cmd in commands:
            assert cmd in text


class TestBenchmarks:

    def test_benchmark_files_exist(self):
        bench_dir = Path(__file__).parent.parent / "benchmarks"
        assert (bench_dir / "demo_queries.json").exists()
        assert (bench_dir / "technical_queries.json").exists()
        assert (bench_dir / "diversity_queries.json").exists()
        assert (bench_dir / "freshness_queries.json").exists()

    def test_demo_queries_valid(self):
        import json
        bench_dir = Path(__file__).parent.parent / "benchmarks"
        data = json.loads((bench_dir / "demo_queries.json").read_text())
        assert "queries" in data
        assert len(data["queries"]) == 10
