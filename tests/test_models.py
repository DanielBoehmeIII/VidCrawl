from vidcrawl.models import (
    Duplicate,
    Evidence,
    Idea,
    IngestionRun,
    Keyframe,
    Moment,
    SearchResult,
    Video,
)


class TestVideoModel:
    def test_minimal(self):
        v = Video(video_id="abc123", title="Test", source="local", duration_sec=60.0)
        assert v.video_id == "abc123"
        assert v.status == "pending"
        assert v.metadata == {}

    def test_full(self):
        v = Video(
            video_id="abc123",
            title="Test Video",
            source="youtube",
            url="https://youtube.com/watch?v=abc123",
            duration_sec=120.5,
            status="ready",
        )
        assert v.url == "https://youtube.com/watch?v=abc123"
        assert v.source == "youtube"


class TestMomentModel:
    def test_minimal(self):
        m = Moment(moment_id="v1:0:10", video_id="v1", start_sec=0.0, end_sec=10.0)
        assert m.transcript_text == ""
        assert m.ideas == []
        assert m.keyframe_paths == []

    def test_with_ideas(self):
        idea = Idea(
            idea_id="idea:v1:0:10:0",
            moment_id="v1:0:10",
            type="claim",
            text="This is a key insight",
        )
        m = Moment(
            moment_id="v1:0:10",
            video_id="v1",
            start_sec=0.0,
            end_sec=10.0,
            transcript_text="Key insight here",
            ideas=[idea],
        )
        assert len(m.ideas) == 1
        assert m.ideas[0].type == "claim"


class TestEvidenceModel:
    def test_defaults(self):
        e = Evidence(
            evidence_id="ev:abc123",
            moment_id="v1:0:10",
            modality="transcript",
            content="Hello world",
        )
        assert e.confidence == 1.0
        assert e.source is None


class TestSearchResultModel:
    def test_minimal(self):
        m = Moment(moment_id="v1:0:10", video_id="v1", start_sec=0.0, end_sec=10.0)
        r = SearchResult(
            moment=m,
            relevance_score=0.95,
            matched_on=["transcript"],
            video_title="Test",
            source_url="https://youtube.com/watch?v=v1",
            moment_url="https://youtube.com/watch?v=v1&t=0",
        )
        assert r.relevance_score == 0.95
