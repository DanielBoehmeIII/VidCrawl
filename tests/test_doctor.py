import pytest
from vidcrawl.doctor import run_doctor, run_benchmark


class TestDoctor:

    def test_doctor_no_db(self, tmp_path):
        report = run_doctor(str(tmp_path))
        assert report["database"]["exists"] is False

    def test_doctor_with_demo(self, tmp_path):
        from vidcrawl.demo import create_demo_corpus
        db_file = tmp_path / "vidcrawl.db"
        create_demo_corpus(db_file)
        report = run_doctor(str(tmp_path))
        assert report["database"]["exists"] is True
        assert "videos" in report["counts"]
        assert "moments" in report["counts"]

    def test_doctor_reports_tools(self, tmp_path):
        report = run_doctor(str(tmp_path))
        assert "optional_tools" in report
        assert "ffmpeg" in report["optional_tools"]
        assert isinstance(report["optional_tools"]["ffmpeg"]["available"], bool)


class TestBenchmark:

    def test_benchmark_no_db(self, tmp_path):
        results = run_benchmark(str(tmp_path))
        assert "error" in results

    def test_benchmark_with_demo(self, tmp_path):
        from vidcrawl.demo import create_demo_corpus
        db_file = tmp_path / "vidcrawl.db"
        create_demo_corpus(db_file)
        results = run_benchmark(str(tmp_path))
        assert "moment_count" in results
        assert results["moment_count"] > 0

    def test_cli_doctor(self, tmp_path):
        from typer.testing import CliRunner
        from vidcrawl.cli import app
        runner = CliRunner()
        result = runner.invoke(app, [
            "doctor",
            "--data-dir", str(tmp_path),
        ])
        assert result.exit_code == 0

    def test_cli_benchmark(self, tmp_path):
        from typer.testing import CliRunner
        from vidcrawl.cli import app
        runner = CliRunner()
        result = runner.invoke(app, [
            "benchmark",
            "--data-dir", str(tmp_path),
        ])
        assert result.exit_code == 1  # No DB
