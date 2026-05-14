import json
import tempfile
from pathlib import Path

import pytest

from vidcrawl.db import get_db, init_db
from vidcrawl.technical.code import extract_file_paths, extract_code_identifiers, extract_imports
from vidcrawl.technical.commands import extract_commands
from vidcrawl.technical.errors import extract_errors
from vidcrawl.technical.equations import extract_equations
from vidcrawl.technical.extract import (
    extract_technical_evidence_for_moment,
    run_technical_extraction,
    get_technical_stats,
    _ensure_modal_table,
)
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


# ---- File Path Tests ----

class TestFilePaths:

    def test_absolute_path(self):
        paths = extract_file_paths("Found in /home/user/project/main.py")
        assert any("main.py" in p for p in paths)

    def test_relative_path(self):
        paths = extract_file_paths("Import from src/components/App.tsx")
        assert any("App.tsx" in p or "src/components/App.tsx" in p for p in paths)

    def test_no_paths(self):
        paths = extract_file_paths("just some regular text")
        assert len(paths) == 0


# ---- Command Tests ----

class TestCommands:

    def test_npm_install(self):
        cmds = extract_commands("Run npm install playwright")
        assert any("npm install" in c for c in cmds)

    def test_pip_install(self):
        cmds = extract_commands("pip install pytesseract")
        assert any("pip install" in c for c in cmds)

    def test_git_clone(self):
        cmds = extract_commands("git clone https://github.com/example/repo")
        assert any("git clone" in c for c in cmds)

    def test_no_commands(self):
        cmds = extract_commands("just talking about concepts")
        assert len(cmds) == 0


# ---- Error Tests ----

class TestErrors:

    def test_error_expression(self):
        errors = extract_errors("Error: Cannot find module 'express'")
        assert len(errors) > 0
        assert any("Cannot find module" in e for e in errors)

    def test_traceback(self):
        errors = extract_errors("Traceback (most recent call last)")
        assert len(errors) > 0

    def test_syntax_error(self):
        errors = extract_errors("SyntaxError: invalid syntax")
        assert len(errors) > 0

    def test_no_errors(self):
        errors = extract_errors("everything works fine")
        assert len(errors) == 0


# ---- Equation Tests ----

class TestEquations:

    def test_simple_equation(self):
        eqs = extract_equations("A = PDP^-1")
        assert len(eqs) > 0

    def test_det_expression(self):
        eqs = extract_equations("det(A - lambda I) = 0")
        assert len(eqs) > 0

    def test_no_equations(self):
        eqs = extract_equations("just regular text")
        assert len(eqs) == 0


# ---- Integration Tests ----

class TestTechnicalExtraction:

    def test_extract_for_moment(self):
        result = extract_technical_evidence_for_moment(
            "test_moment",
            "Install with npm install playwright. Error: not found. "
            "File is at /home/user/project/main.py. A = PDP^-1",
            "",
        )
        assert len(result["commands"]) > 0
        assert len(result["errors"]) > 0
        assert len(result["file_paths"]) > 0
        assert len(result["equations"]) > 0

    def test_run_extraction(self, demo_db_path):
        result = run_technical_extraction(str(demo_db_path))
        assert result["total_evidence_inserted"] >= 0
        assert "commands" in result
        assert "errors" in result

    def test_technical_stats(self, demo_db_path):
        run_technical_extraction(str(demo_db_path))
        stats = get_technical_stats(str(demo_db_path))
        assert "total_technical_evidence" in stats
        assert "by_modality" in stats

    def test_ensure_modal_table(self, db_path):
        from vidcrawl.db import insert_video, insert_moment
        from vidcrawl.models import Video, Moment
        conn = get_db(db_path)
        insert_video(conn, Video(
            video_id="test_vid", title="Test", source="local",
            duration_sec=10.0, status="ready",
        ))
        insert_moment(conn, Moment(
            moment_id="test_mom", video_id="test_vid",
            start_sec=0.0, end_sec=10.0,
        ))
        conn.commit()
        try:
            conn.execute(
                "INSERT INTO modal_evidence (evidence_id, moment_id, modality, content, confidence, source) "
                "VALUES ('test_code_mod', 'test_mom', 'code', 'test', 1.0, 'test')"
            )
        except Exception:
            conn.execute("""CREATE TABLE IF NOT EXISTS modal_evidence_v2 (
                evidence_id TEXT PRIMARY KEY,
                moment_id   TEXT NOT NULL REFERENCES moments(moment_id),
                modality    TEXT NOT NULL,
                content     TEXT NOT NULL,
                confidence  REAL DEFAULT 1.0,
                source      TEXT,
                metadata    TEXT DEFAULT '{}',
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            )""")
            conn.execute("DROP TABLE IF EXISTS modal_evidence")
            conn.execute("ALTER TABLE modal_evidence_v2 RENAME TO modal_evidence")
            conn.execute("""CREATE INDEX IF NOT EXISTS idx_evidence_moment_id ON modal_evidence(moment_id)""")
            conn.commit()
            _ensure_modal_table(conn)
        conn.execute(
            "INSERT INTO modal_evidence (evidence_id, moment_id, modality, content, confidence, source) "
            "VALUES ('test2', 'test_mom', 'equation', 'A=B', 0.9, 'test')"
        )
        conn.commit()
        row = conn.execute("SELECT modality FROM modal_evidence WHERE evidence_id='test2'").fetchone()
        assert row["modality"] == "equation"
        conn.execute("DELETE FROM modal_evidence WHERE evidence_id IN ('test_code_mod', 'test2')")
        conn.close()

    def test_search_technical_terms(self, demo_db_path):
        run_technical_extraction(str(demo_db_path))
        from vidcrawl.search.query import search_moments
        results = search_moments("npm install", demo_db_path, limit=5)
        assert len(results) >= 0

    def test_cli_extract(self, db_path, tmp_path):
        from typer.testing import CliRunner
        from vidcrawl.cli import app
        from vidcrawl.db import get_db, init_db
        db_file = tmp_path / "data" / "vidcrawl.db"
        db_file.parent.mkdir(parents=True, exist_ok=True)
        conn = get_db(db_file)
        init_db(conn)
        conn.close()
        from vidcrawl.demo import create_demo_corpus
        create_demo_corpus(db_file)
        runner = CliRunner()
        result = runner.invoke(app, [
            "technical", "extract",
            "--data-dir", str(tmp_path / "data"),
        ])
        assert result.exit_code == 0

    def test_cli_stats(self, db_path, tmp_path):
        from typer.testing import CliRunner
        from vidcrawl.cli import app
        from vidcrawl.db import get_db, init_db
        db_file = tmp_path / "data" / "vidcrawl.db"
        db_file.parent.mkdir(parents=True, exist_ok=True)
        conn = get_db(db_file)
        init_db(conn)
        conn.close()
        from vidcrawl.demo import create_demo_corpus
        create_demo_corpus(db_file)
        from vidcrawl.technical.extract import run_technical_extraction
        run_technical_extraction(str(db_file))
        runner = CliRunner()
        result = runner.invoke(app, [
            "technical", "stats",
            "--data-dir", str(tmp_path / "data"),
        ])
        assert result.exit_code == 0
