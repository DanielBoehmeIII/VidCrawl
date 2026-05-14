import json
import tempfile
from pathlib import Path

import pytest

from vidcrawl.db import get_db, init_db, insert_video, insert_moment, insert_idea, insert_evidence, insert_duplicate
from vidcrawl.db import make_moment_id, make_idea_id, generate_evidence_id, generate_dup_id
from vidcrawl.models import Video, Moment, Idea, Evidence, Duplicate
from vidcrawl.search.features import compute_features
from vidcrawl.search.rerank import compute_graph_score, DEFAULT_WEIGHTS
from vidcrawl.search.diversity import select_diverse_results
from vidcrawl.search.query import search_moments, _graph_has_data
from vidcrawl.graph.build import build_graph


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


def _populate_with_graph(conn):
    from vidcrawl.db import rebuild_fts
    video = Video(
        video_id="test_vid",
        title="Test Video about Playwright MCP",
        source="local",
        url=None,
        duration_sec=100.0,
        status="ready",
    )
    insert_video(conn, video)

    m1_id = make_moment_id("test_vid", 0.0, 10.0)
    i1 = Idea(
        idea_id=make_idea_id(m1_id, 0),
        moment_id=m1_id,
        type="definition",
        text="MCP is a protocol for AI",
    )
    m1 = Moment(
        moment_id=m1_id,
        video_id="test_vid",
        start_sec=0.0,
        end_sec=10.0,
        transcript_text="This is a test about Playwright MCP server. Install it with npm.",
        ocr_text="Playwright MCP Server",
        ideas=[i1],
    )
    insert_moment(conn, m1)
    insert_idea(conn, i1)

    insert_evidence(conn, Evidence(
        evidence_id=generate_evidence_id(),
        moment_id=m1.moment_id,
        modality="transcript",
        content=m1.transcript_text,
    ))
    insert_evidence(conn, Evidence(
        evidence_id=generate_evidence_id(),
        moment_id=m1.moment_id,
        modality="ocr",
        content=m1.ocr_text,
    ))
    insert_evidence(conn, Evidence(
        evidence_id=generate_evidence_id(),
        moment_id=m1.moment_id,
        modality="idea",
        content="[definition] MCP is a protocol for AI",
    ))

    m2_id = make_moment_id("test_vid", 10.0, 20.0)
    i2 = Idea(
        idea_id=make_idea_id(m2_id, 0),
        moment_id=m2_id,
        type="warning",
        text="Be careful with headless browsers",
    )
    m2 = Moment(
        moment_id=m2_id,
        video_id="test_vid",
        start_sec=10.0,
        end_sec=20.0,
        transcript_text="Be careful when using Playwright in headless mode.",
        ocr_text="Warning: headless Playwright",
        ideas=[i2],
    )
    insert_moment(conn, m2)
    insert_idea(conn, i2)

    insert_evidence(conn, Evidence(
        evidence_id=generate_evidence_id(),
        moment_id=m2.moment_id,
        modality="transcript",
        content=m2.transcript_text,
    ))
    insert_evidence(conn, Evidence(
        evidence_id=generate_evidence_id(),
        moment_id=m2.moment_id,
        modality="ocr",
        content=m2.ocr_text,
    ))

    conn.commit()
    rebuild_fts(conn)
    conn.commit()
    return video, [m1, m2], [i1, i2]


# ---- Feature Extraction Tests ----

class TestFeatureExtraction:

    def test_basic_features(self, db_path):
        conn = get_db(db_path)
        v, moments, ideas = _populate_with_graph(conn)
        conn.close()

        build_graph(db_path)

        moment_ids = [m.moment_id for m in moments]
        features = compute_features(str(db_path), moment_ids, query="playwright MCP")
        assert len(features) == 2

        f1 = features.get(moments[0].moment_id, {})
        assert f1.get("has_transcript") is True
        assert f1.get("has_ocr") is True
        assert f1.get("idea_count") == 1
        assert f1.get("evidence_count") == 3
        assert f1.get("modality_count") >= 3
        assert "definition" in f1.get("idea_types", [])

    def test_features_empty(self, db_path):
        features = compute_features(str(db_path), [], query="test")
        assert features == {}

    def test_features_unkown_moment(self, db_path):
        features = compute_features(str(db_path), ["nonexistent"], query="test")
        assert "nonexistent" in features
        assert features["nonexistent"]["has_transcript"] is False

    def test_query_content_features(self, db_path):
        conn = get_db(db_path)
        v, moments, ideas = _populate_with_graph(conn)
        conn.close()

        moment_ids = [m.moment_id for m in moments]
        features = compute_features(str(db_path), moment_ids, query="playwright MCP")
        f0 = features.get(moments[0].moment_id, {})
        assert f0.get("query_in_transcript") is True
        assert f0.get("query_in_ocr") is True


# ---- Graph Reranking Tests ----

class TestGraphReranking:

    def test_graph_score_has_ocr(self):
        from vidcrawl.search.features import _empty_features
        feat = _empty_features()
        feat["has_ocr"] = True
        score, reasons = compute_graph_score(feat, fts_score=1.0)
        assert score > 1.0
        assert "matched OCR" in reasons

    def test_graph_score_ideas(self):
        from vidcrawl.search.features import _empty_features
        feat = _empty_features()
        feat["idea_count"] = 3
        score, reasons = compute_graph_score(feat, fts_score=0.0)
        assert score > 0.15
        assert any("idea" in r for r in reasons)

    def test_graph_score_evidence(self):
        from vidcrawl.search.features import _empty_features
        feat = _empty_features()
        feat["evidence_count"] = 3
        score, reasons = compute_graph_score(feat, fts_score=0.0)
        assert score > 0.05

    def test_exact_duplicate_penalized(self):
        from vidcrawl.search.features import _empty_features
        feat = _empty_features()
        feat["is_exact_duplicate"] = True
        score_without, _ = compute_graph_score(_empty_features(), fts_score=1.0)
        score_with, _ = compute_graph_score(feat, fts_score=1.0, include_duplicates=False)
        assert score_with < score_without

    def test_canonical_boosted(self):
        from vidcrawl.search.features import _empty_features
        feat = _empty_features()
        feat["is_canonical"] = True
        score, reasons = compute_graph_score(feat, fts_score=1.0)
        assert "canonical" in " ".join(reasons)

    def test_variant_boosted_in_diverse(self):
        from vidcrawl.search.features import _empty_features
        feat = _empty_features()
        feat["duplicate_type"] = "variant"
        score, reasons = compute_graph_score(feat, fts_score=1.0, diverse_mode=True)
        assert "variant" in " ".join(reasons)

    def test_entity_boost(self):
        from vidcrawl.search.features import _empty_features
        feat = _empty_features()
        feat["entity_degree"] = 3
        score, reasons = compute_graph_score(feat, fts_score=0.0)
        assert score > 0.05
        assert any("entit" in r for r in reasons)

    def test_code_like_boost(self):
        from vidcrawl.search.features import _empty_features
        feat = _empty_features()
        feat["code_like_match"] = True
        score, reasons = compute_graph_score(feat, fts_score=0.0)
        assert "code" in " ".join(reasons)

    def test_warning_boost(self):
        from vidcrawl.search.features import _empty_features
        feat = _empty_features()
        feat["warning_match"] = True
        score, reasons = compute_graph_score(feat, fts_score=0.0)
        assert "warning" in " ".join(reasons)


# ---- Diversity Selection Tests ----

class TestDiversitySelection:

    def test_diverse_selects_top_first(self, db_path):
        conn = get_db(db_path)
        v, moments, ideas = _populate_with_graph(conn)
        conn.close()
        build_graph(db_path)

        from vidcrawl.search.features import compute_features
        moment_ids = [m.moment_id for m in moments]
        feats = compute_features(str(db_path), moment_ids, query="test")

        from vidcrawl.search.query import SearchResult
        results = [
            SearchResult(moment_id=moments[0].moment_id, video_id="test_vid", video_title="T", score=10.0),
            SearchResult(moment_id=moments[1].moment_id, video_id="test_vid", video_title="T", score=5.0),
        ]

        diverse = select_diverse_results(results, feats, limit=2)
        assert len(diverse) == 2
        assert diverse[0].score == 10.0

    def test_diverse_avoids_same_cluster(self):
        from vidcrawl.search.query import SearchResult
        from vidcrawl.search.features import _empty_features

        results = [
            SearchResult(moment_id="m1", video_id="v1", video_title="A", score=10.0, canonical_moment_id="canon"),
            SearchResult(moment_id="m2", video_id="v1", video_title="B", score=3.0, canonical_moment_id="canon"),
            SearchResult(moment_id="m3", video_id="v2", video_title="C", score=2.9),
        ]
        feats = {
            "m1": _empty_features(),
            "m2": {**_empty_features(), "canonical_moment_id": "canon"},
            "m3": _empty_features(),
        }

        diverse = select_diverse_results(results, feats, limit=2, diversity_strength=0.5)
        assert len(diverse) == 2
        assert diverse[0].moment_id == "m1"
        assert diverse[1].moment_id == "m3"


# ---- Search Integration Tests ----

class TestSearchRerankIntegration:

    def test_search_rerank_works(self, db_path):
        conn = get_db(db_path)
        _populate_with_graph(conn)
        conn.close()
        build_graph(db_path)

        results = search_moments("playwright", db_path, limit=5)
        assert len(results) > 0
        assert results[0].score > 0

    def test_search_raw_ranking(self, db_path):
        conn = get_db(db_path)
        _populate_with_graph(conn)
        conn.close()
        build_graph(db_path)

        results = search_moments("playwright", db_path, limit=5, use_rerank=False)
        assert len(results) > 0

    def test_search_fallback_no_graph(self, db_path):
        results = search_moments("playwright", db_path, limit=5, use_rerank=True)
        assert isinstance(results, list)

    def test_graph_has_data(self, db_path):
        conn = get_db(db_path)
        assert _graph_has_data(conn) is False
        conn.close()

        conn = get_db(db_path)
        _populate_with_graph(conn)
        conn.close()
        build_graph(db_path)

        conn = get_db(db_path)
        assert _graph_has_data(conn) is True
        conn.close()

    def test_ranking_reasons_present(self, db_path):
        conn = get_db(db_path)
        _populate_with_graph(conn)
        conn.close()
        build_graph(db_path)

        results = search_moments("playwright", db_path, limit=5)
        for r in results:
            assert r.ranking_reasons or r.match_reasons

    def test_features_json_present(self, db_path):
        conn = get_db(db_path)
        _populate_with_graph(conn)
        conn.close()
        build_graph(db_path)

        results = search_moments("playwright", db_path, limit=5)
        for r in results:
            assert r.features_json is not None or not results


# ---- Eval Metrics Tests ----

class TestEvalRerankMetrics:

    def test_eval_with_rerank_includes_diversity_metrics(self, db_path):
        from vidcrawl.eval import evaluate_queries

        conn = get_db(db_path)
        _populate_with_graph(conn)
        conn.close()
        build_graph(db_path)

        queries = [{"query": "playwright", "expected_terms": ["playwright"]}]
        metrics = evaluate_queries(db_path, queries, use_rerank=True)
        assert "avg_unique_videos_in_topk" in metrics
        assert "avg_unique_clusters_in_topk" in metrics
