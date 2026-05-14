import json
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vidcrawl.cli import app
from vidcrawl.db import (
    get_db,
    get_duplicate_stats,
    get_duplicates_for_moment,
    init_db,
    insert_duplicate,
    insert_moment,
    insert_video,
    make_moment_id,
    rebuild_fts,
)
from vidcrawl.dedupe.cluster import run_dedupe, _canonical_quality, _get_ideas_text
from vidcrawl.dedupe.normalize import content_hash, get_combined_text, normalize_text
from vidcrawl.dedupe.novelty import score_novelty
from vidcrawl.dedupe.similarity import (
    combined_similarity,
    jaccard_similarity,
    ngram_similarity,
    sequence_match_ratio,
)
from vidcrawl.demo import create_demo_corpus
from vidcrawl.models import Duplicate, Idea, Moment, Video
from vidcrawl.search.query import search_moments

runner = CliRunner()


# ============================================================
# Text Normalization
# ============================================================

class TestNormalize:
    def test_normalize_lowercase(self):
        assert normalize_text("Hello WORLD") == "hello world"

    def test_normalize_collapse_whitespace(self):
        assert normalize_text("hello    world") == "hello world"

    def test_normalize_remove_punctuation(self):
        assert normalize_text("hello, world!") == "hello world"

    def test_normalize_empty(self):
        assert normalize_text("") == ""

    def test_content_hash_stable(self):
        h1 = content_hash("hello world")
        h2 = content_hash("hello world")
        assert h1 == h2

    def test_content_hash_different(self):
        h1 = content_hash("hello world")
        h2 = content_hash("goodbye world")
        assert h1 != h2

    def test_get_combined_text(self):
        result = get_combined_text("hello", "world", "test")
        assert "hello" in result
        assert "world" in result
        assert "test" in result

    def test_get_combined_text_empty(self):
        assert get_combined_text("", "", "") == ""


# ============================================================
# Similarity
# ============================================================

class TestSimilarity:
    def test_jaccard_identical(self):
        assert jaccard_similarity("hello world", "hello world") == 1.0

    def test_jaccard_different(self):
        assert jaccard_similarity("hello world", "goodbye moon") < 1.0

    def test_jaccard_empty(self):
        assert jaccard_similarity("", "") == 1.0

    def test_ngram_identical(self):
        assert ngram_similarity("hello world", "hello world") == 1.0

    def test_ngram_different(self):
        assert ngram_similarity("abcdef", "ghijkl") < 1.0

    def test_ngram_empty(self):
        assert ngram_similarity("", "") == 1.0

    def test_sequence_match_identical(self):
        assert sequence_match_ratio("hello world", "hello world") == 1.0

    def test_sequence_match_partial(self):
        assert sequence_match_ratio("hello world", "hello") > 0.5

    def test_combined_similarity(self):
        sim = combined_similarity("hello world test", "hello world test")
        assert sim == 1.0

    def test_combined_similarity_different(self):
        sim = combined_similarity("hello world", "goodbye moon")
        assert sim < 0.5


# ============================================================
# Novelty Scoring
# ============================================================

class TestNovelty:
    def test_identical_text(self):
        result = score_novelty("hello world", "hello world")
        assert result["novelty_score"] == 0

    def test_warning_terms_add_novelty(self):
        result = score_novelty(
            "be careful with this error problem",
            "this is a normal sentence",
        )
        assert result["novelty_score"] > 0
        assert "adds warning terms" in result["reason"]

    def test_example_terms_add_novelty(self):
        result = score_novelty(
            "for example here is a sample",
            "this is normal text",
        )
        assert result["novelty_score"] > 0

    def test_different_ocr_adds_novelty(self):
        result = score_novelty(
            "hello world",
            "hello world",
            candidate_ocr="npm install test",
            canonical_ocr="",
        )
        assert result["novelty_score"] > 0
        assert "different OCR text" in result["reason"]

    def test_new_idea_type_adds_novelty(self):
        result = score_novelty(
            "hello world",
            "hello world",
            candidate_idea_types=["warning"],
            canonical_idea_types=["definition"],
        )
        assert result["novelty_score"] > 0

    def test_longer_explanation_adds_novelty(self):
        result = score_novelty(
            "this is a much longer and more detailed explanation of the concept that was mentioned",
            "short text",
        )
        assert result["novelty_score"] > 0

    def test_novelty_capped_at_1(self):
        result = score_novelty(
            "warning error careful avoid problem mistake caution fail wrong be careful with this error problem",
            "short text",
            candidate_ocr="npm install playwright npx code",
            candidate_idea_types=["warning", "step", "example"],
            canonical_idea_types=["definition"],
        )
        assert result["novelty_score"] <= 1.0

    def test_similarity_with_overlap(self):
        result = score_novelty("hello world", "hello world")
        assert result["similarity_score"] >= 0.9


# ============================================================
# Canonical Selection
# ============================================================

class MockRow:
    def __init__(self, transcript, ocr, ideas, text_len=None):
        self.transcript_text = transcript
        self.ocr_text = ocr
        self.ideas = ideas
        self._text_len = text_len or len(transcript) if transcript else 0

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)


class TestCanonical:
    def test_prefers_has_transcript(self):
        a = MockRow("short", "", "[]")
        b = MockRow("", "", "[]")
        score_a = _canonical_quality(a)
        score_b = _canonical_quality(b)
        assert score_a > score_b

    def test_prefers_has_ocr(self):
        a = MockRow("text", "ocr text", "[]")
        b = MockRow("text", "", "[]")
        assert _canonical_quality(a) > _canonical_quality(b)

    def test_prefers_has_ideas(self):
        a = MockRow("text", "", '[{"type":"step","text":"do it"}]')
        b = MockRow("text", "", "[]")
        assert _canonical_quality(a) > _canonical_quality(b)

    def test_prefers_good_length(self):
        a = MockRow("This is a good length text for scoring", "", "[]")
        b = MockRow("A", "", "[]")
        assert _canonical_quality(a) > _canonical_quality(b)


# ============================================================
# Dedupe Run (requires DB)
# ============================================================

def _setup_dedupable_db(db_path: str) -> None:
    conn = get_db(db_path)
    init_db(conn)

    v = Video(video_id="dedup_test", title="Dedup Test", source="local", duration_sec=60.0)
    insert_video(conn, v)

    m1 = Moment(
        moment_id=make_moment_id("dedup_test", 0.0, 10.0),
        video_id="dedup_test", start_sec=0.0, end_sec=10.0,
        transcript_text="This is a unique moment about Playwright testing",
        ocr_text="Playwright Test",
        ideas=[Idea(idea_id="i1", moment_id=make_moment_id("dedup_test", 0.0, 10.0), type="step", text="Use Playwright")],
    )
    m2 = Moment(
        moment_id=make_moment_id("dedup_test", 10.0, 20.0),
        video_id="dedup_test", start_sec=10.0, end_sec=20.0,
        transcript_text="This is a unique moment about Playwright testing",
        ocr_text="Playwright Test",
        ideas=[Idea(idea_id="i2", moment_id=make_moment_id("dedup_test", 10.0, 20.0), type="step", text="Use Playwright")],
    )
    m3 = Moment(
        moment_id=make_moment_id("dedup_test", 20.0, 30.0),
        video_id="dedup_test", start_sec=20.0, end_sec=30.0,
        transcript_text="This is a unique moment about something completely different",
        ocr_text="",
        ideas=[],
    )
    for m in [m1, m2, m3]:
        insert_moment(conn, m)
    conn.commit()
    rebuild_fts(conn)
    conn.commit()
    conn.close()


@pytest.fixture
def dedup_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    _setup_dedupable_db(db_path)
    yield Path(db_path)
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def demo_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    create_demo_corpus(Path(db_path))
    yield Path(db_path)
    Path(db_path).unlink(missing_ok=True)


class TestDedupeRun:
    def test_dry_run_does_not_mutate(self, dedup_db):
        conn = get_db(str(dedup_db))
        stats = run_dedupe(conn, dry_run=True)
        assert stats["exact_duplicates"] > 0
        count = conn.execute("SELECT COUNT(*) as c FROM duplicates").fetchone()["c"]
        assert count == 0
        conn.close()

    def test_run_creates_duplicate_records(self, dedup_db):
        conn = get_db(str(dedup_db))
        stats = run_dedupe(conn, dry_run=False)
        assert stats["exact_duplicates"] > 0
        count = conn.execute("SELECT COUNT(*) as c FROM duplicates").fetchone()["c"]
        assert count > 0
        conn.close()

    def test_run_on_demo_corpus(self, demo_db):
        conn = get_db(str(demo_db))
        stats = run_dedupe(conn, dry_run=False)
        assert stats["total_before"] > 0
        assert stats["exact_duplicates"] >= 0
        conn.close()


# ============================================================
# Duplicate Stats
# ============================================================

class TestDuplicateStats:
    def test_stats_after_dedupe(self, dedup_db):
        conn = get_db(str(dedup_db))
        run_dedupe(conn, dry_run=False)
        stats = get_duplicate_stats(conn)
        assert stats["total"] > 0
        assert "exact" in stats["by_type"]
        conn.close()

    def test_stats_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        conn = get_db(db_path)
        init_db(conn)
        stats = get_duplicate_stats(conn)
        assert stats["total"] == 0
        conn.close()
        Path(db_path).unlink(missing_ok=True)

    def test_duplicates_for_moment(self, dedup_db):
        conn = get_db(str(dedup_db))
        run_dedupe(conn, dry_run=False)
        mid = make_moment_id("dedup_test", 10.0, 20.0)
        records = get_duplicates_for_moment(conn, mid)
        assert len(records) > 0
        conn.close()


# ============================================================
# Search: Dedupe Integration
# ============================================================

class TestSearchDedupe:
    def test_search_hides_exact_duplicates_by_default(self, dedup_db):
        conn = get_db(str(dedup_db))
        run_dedupe(conn, dry_run=False)
        conn.close()
        results = search_moments("playwright", dedup_db, include_duplicates=False)
        result_ids = [r.moment_id for r in results]
        dup_id = make_moment_id("dedup_test", 10.0, 20.0)
        canonical_id = make_moment_id("dedup_test", 0.0, 10.0)
        if canonical_id in result_ids:
            assert dup_id not in result_ids, f"{dup_id} should be hidden"

    def test_include_duplicates_shows_all(self, dedup_db):
        conn = get_db(str(dedup_db))
        run_dedupe(conn, dry_run=False)
        conn.close()
        results = search_moments("playwright", dedup_db, include_duplicates=True)
        dup_id = make_moment_id("dedup_test", 10.0, 20.0)
        dup_ids = [r.moment_id for r in results]
        assert dup_id in dup_ids

    def test_search_demo_corpus_after_dedupe(self, demo_db):
        conn = get_db(str(demo_db))
        run_dedupe(conn, dry_run=False)
        conn.close()
        results = search_moments("playwright", demo_db, include_duplicates=False)
        assert isinstance(results, list)


# ============================================================
# CLI Dedupe
# ============================================================

class TestCLIDedupe:
    def test_dedupe_run_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["dedupe", "run", "--data-dir", tmpdir]
            )
            assert result.exit_code == 0
            assert "Found" in result.stdout

    def test_dedupe_dry_run_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["dedupe", "run", "--data-dir", tmpdir, "--dry-run"]
            )
            assert result.exit_code == 0
            assert "Dry run" in result.stdout

    def test_dedupe_stats_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            runner.invoke(app, ["dedupe", "run", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["dedupe", "stats", "--data-dir", tmpdir]
            )
            assert result.exit_code == 0
            assert "Total duplicate records" in result.stdout

    def test_dedupe_show_cli(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            runner.invoke(app, ["dedupe", "run", "--data-dir", tmpdir])
            conn = get_db(str(Path(tmpdir) / "vidcrawl.db"))
            rows = conn.execute(
                "SELECT moment_id FROM duplicates LIMIT 1"
            ).fetchall()
            conn.close()
            if rows:
                mid = rows[0]["moment_id"]
                result = runner.invoke(
                    app, ["dedupe", "show", mid, "--data-dir", tmpdir]
                )
                assert result.exit_code == 0
                assert "Dedupe records" in result.stdout

    def test_dedupe_no_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app, ["dedupe", "run", "--data-dir", tmpdir]
            )
            assert result.exit_code == 1
            assert "No database found" in result.stderr

    def test_search_collapsed_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            runner.invoke(app, ["dedupe", "run", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["search", "playwright", "--data-dir", tmpdir]
            )
            assert result.exit_code == 0
            assert "Collapsed" in result.stdout or "Results" in result.stdout

    def test_search_include_duplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            runner.invoke(app, ["dedupe", "run", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["search", "playwright", "--data-dir", tmpdir, "--include-duplicates"]
            )
            assert result.exit_code == 0

    def test_dedupe_run_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["demo", "init", "--data-dir", tmpdir])
            result = runner.invoke(
                app, ["dedupe", "run", "--data-dir", tmpdir, "--json"]
            )
            assert result.exit_code == 0
            data = json.loads(result.stdout)
            assert "exact_duplicates" in data
