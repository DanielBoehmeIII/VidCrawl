import tempfile
from pathlib import Path

from typer.testing import CliRunner

from vidcrawl.cli import app

runner = CliRunner()


class TestCLI:
    def test_init_creates_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(app, ["init", "--data-dir", tmpdir])
            assert result.exit_code == 0
            assert "Initialized" in result.stdout
            assert Path(tmpdir, "vidcrawl.db").exists()

    def test_init_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            r1 = runner.invoke(app, ["init", "--data-dir", tmpdir])
            r2 = runner.invoke(app, ["init", "--data-dir", tmpdir])
            assert r1.exit_code == 0
            assert r2.exit_code == 0

    def test_list_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            result = runner.invoke(app, ["list", "--data-dir", tmpdir])
            assert result.exit_code == 0
            assert "No videos" in result.stdout

    def test_list_no_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(app, ["list", "--data-dir", tmpdir])
            assert result.exit_code == 1
            assert "No database found" in result.stderr

    def test_inspect_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["inspect", "nonexistent", "--data-dir", tmpdir]
            )
            assert result.exit_code == 1
            assert "not found" in result.stderr

    def test_ingest_errors_on_invalid_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["ingest", "/nonexistent/path.mp4", "--data-dir", tmpdir]
            )
            assert result.exit_code == 1

    def test_ingest_local_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir, "test_video.mp4")
            video_path.write_text("fake video content")
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["ingest", str(video_path), "--data-dir", tmpdir, "--no-process"]
            )
            assert result.exit_code == 0
            assert "Registered" in result.stdout

    def test_ingest_youtube_url(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            result = runner.invoke(
                app,
                ["ingest", "https://youtube.com/watch?v=dQw4w9WgXcQ", "--data-dir", tmpdir],
            )
            assert result.exit_code == 0
            assert "Registered" in result.stdout

    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0

    def test_reindex(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["reindex", "--data-dir", tmpdir]
            )
            assert result.exit_code == 0
            assert "rebuilt" in result.stdout.lower()
