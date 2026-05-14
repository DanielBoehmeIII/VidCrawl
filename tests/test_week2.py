import json
import tempfile
from pathlib import Path

from typer.testing import CliRunner

from vidcrawl.cli import app
from vidcrawl.db import (
    generate_evidence_id,
    get_db,
    get_evidence_count_by_video,
    get_idea_count_by_video,
    get_keyframe_count_by_video,
    get_moment_count_by_video,
    get_moments_by_video,
    get_video,
    init_db,
    insert_evidence,
    insert_keyframe,
    insert_moment,
    insert_video,
    make_idea_id,
    make_moment_id,
    rebuild_fts,
)
from vidcrawl.ingest.transcript import (
    _parse_srt,
    _parse_txt,
    _parse_vtt,
    load_sidecar_transcript,
    transcribe_audio,
)
from vidcrawl.models import Evidence, Idea, Keyframe, Moment, Video
from vidcrawl.process.chunking import chunk_transcript
from vidcrawl.process.ideas import extract_ideas
from vidcrawl.process.keyframes import extract_keyframes
from vidcrawl.process.ocr import ocr_frames
from vidcrawl.process.pipeline import (
    _create_fallback_chunks,
    _get_keyframes_for_moment,
    _get_ocr_for_moment,
    _insert_evidence_for_moment,
)

runner = CliRunner()


# ============================================================
# Transcript Parsing Tests
# ============================================================

SRT_CONTENT = """1
00:00:01,000 --> 00:00:04,000
Hello world

2
00:00:05,000 --> 00:00:08,500
This is a test.
Second line here.

3
00:00:10,000 --> 00:00:12,000
Goodbye"""

VTT_CONTENT = """WEBVTT

00:00:01.000 --> 00:00:04.000
Hello world

00:00:05.000 --> 00:00:08.500
This is a test.
Second line here.

00:00:10.000 --> 00:00:12.000
Goodbye"""


class TestTranscriptParsing:
    def test_parse_srt_basic(self):
        entries = _parse_srt(SRT_CONTENT)
        assert len(entries) == 3
        assert entries[0]["start_sec"] == 1.0
        assert entries[0]["end_sec"] == 4.0
        assert entries[0]["text"] == "Hello world"
        assert entries[1]["text"] == "This is a test. Second line here."
        assert entries[2]["text"] == "Goodbye"

    def test_parse_srt_empty(self):
        assert _parse_srt("") == []
        assert _parse_srt("   ") == []

    def test_parse_srt_no_timestamps(self):
        assert _parse_srt("just some text") == []

    def test_parse_vtt_basic(self):
        entries = _parse_vtt(VTT_CONTENT)
        assert len(entries) == 3
        assert entries[0]["start_sec"] == 1.0
        assert entries[0]["end_sec"] == 4.0
        assert entries[0]["text"] == "Hello world"
        assert entries[1]["text"] == "This is a test. Second line here."
        assert entries[2]["text"] == "Goodbye"

    def test_parse_vtt_empty(self):
        assert _parse_vtt("") == []
        assert _parse_vtt("WEBVTT") == []

    def test_parse_vtt_with_headers(self):
        vtt = """WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:02.000
Test"""
        entries = _parse_vtt(vtt)
        assert len(entries) == 1
        assert entries[0]["text"] == "Test"

    def test_parse_txt_basic(self):
        text = "First sentence. Second sentence.\n\nThird paragraph."
        entries = _parse_txt(text)
        assert len(entries) >= 3
        assert all(e["start_sec"] <= e["end_sec"] for e in entries)
        assert all(e["text"] for e in entries)

    def test_parse_txt_empty(self):
        assert _parse_txt("") == []
        assert _parse_txt("   ") == []

    def test_load_sidecar_srt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir, "test.mp4")
            video_path.write_text("fake video")
            srt_path = Path(tmpdir, "test.srt")
            srt_path.write_text(SRT_CONTENT)
            entries = load_sidecar_transcript(str(video_path))
            assert entries is not None
            assert len(entries) == 3

    def test_load_sidecar_vtt_preferred(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir, "test.mp4")
            video_path.write_text("fake video")
            srt_path = Path(tmpdir, "test.srt")
            srt_path.write_text(SRT_CONTENT)
            vtt_path = Path(tmpdir, "test.vtt")
            vtt_path.write_text(VTT_CONTENT)
            entries = load_sidecar_transcript(str(video_path))
            assert entries is not None
            assert len(entries) == 3

    def test_load_sidecar_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir, "test.mp4")
            video_path.write_text("fake video")
            json_path = Path(tmpdir, "test.json")
            data = [{"start_sec": 0.0, "end_sec": 5.0, "text": "hello"}]
            json_path.write_text(json.dumps(data))
            entries = load_sidecar_transcript(str(video_path))
            assert entries == data

    def test_load_sidecar_txt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir, "test.mp4")
            video_path.write_text("fake video")
            txt_path = Path(tmpdir, "test.txt")
            txt_path.write_text("Hello world.\n\nSecond paragraph.")
            entries = load_sidecar_transcript(str(video_path))
            assert entries is not None
            assert len(entries) >= 2

    def test_load_sidecar_no_transcript(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir, "test.mp4")
            video_path.write_text("fake video")
            entries = load_sidecar_transcript(str(video_path))
            assert entries is None

    def test_transcribe_audio_whisper_not_installed(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = transcribe_audio("nonexistent.wav")
            assert result == []
            assert any("Whisper" in str(msg.message) for msg in w)


# ============================================================
# Chunking Tests
# ============================================================

class TestChunking:
    def test_empty_entries(self):
        assert chunk_transcript([]) == []

    def test_single_entry(self):
        entries = [{"start_sec": 0.0, "end_sec": 5.0, "text": "Hello world"}]
        chunks = chunk_transcript(entries)
        assert len(chunks) == 1
        assert chunks[0]["start_sec"] == 0.0
        assert chunks[0]["end_sec"] == 5.0
        assert chunks[0]["transcript_text"] == "Hello world"

    def test_multiple_entries_combined(self):
        entries = [
            {"start_sec": 0.0, "end_sec": 5.0, "text": "First part"},
            {"start_sec": 5.0, "end_sec": 10.0, "text": "Second part"},
            {"start_sec": 10.0, "end_sec": 15.0, "text": "Third part"},
        ]
        chunks = chunk_transcript(entries, max_duration=60.0)
        assert len(chunks) == 1
        assert "First part" in chunks[0]["transcript_text"]
        assert "Second part" in chunks[0]["transcript_text"]
        assert "Third part" in chunks[0]["transcript_text"]

    def test_max_duration_splits(self):
        entries = [
            {"start_sec": 0.0, "end_sec": 30.0, "text": "A" * 100},
            {"start_sec": 30.0, "end_sec": 60.0, "text": "B" * 100},
            {"start_sec": 60.0, "end_sec": 90.0, "text": "C" * 100},
        ]
        chunks = chunk_transcript(entries, max_duration=45.0)
        assert len(chunks) >= 2

    def test_overlap_creates_multiple_chunks_with_overlap(self):
        entries = [
            {"start_sec": 0.0, "end_sec": 10.0, "text": "Segment A"},
            {"start_sec": 10.0, "end_sec": 20.0, "text": "Segment B"},
            {"start_sec": 20.0, "end_sec": 30.0, "text": "Segment C"},
            {"start_sec": 30.0, "end_sec": 40.0, "text": "Segment D"},
            {"start_sec": 40.0, "end_sec": 50.0, "text": "Segment E"},
            {"start_sec": 50.0, "end_sec": 60.0, "text": "Segment F"},
            {"start_sec": 60.0, "end_sec": 70.0, "text": "Segment G"},
        ]
        chunks = chunk_transcript(entries, max_duration=25.0, overlap_sec=5.0)
        assert len(chunks) >= 3
        assert chunks[0]["start_sec"] <= chunks[1]["start_sec"]

    def test_normalizes_whitespace(self):
        entries = [
            {"start_sec": 0.0, "end_sec": 5.0, "text": "Hello    world"},
        ]
        chunks = chunk_transcript(entries)
        assert chunks[0]["transcript_text"] == "Hello world"

    def test_skips_empty_entries(self):
        entries = [
            {"start_sec": 0.0, "end_sec": 5.0, "text": ""},
            {"start_sec": 5.0, "end_sec": 10.0, "text": "   "},
            {"start_sec": 10.0, "end_sec": 15.0, "text": "Real text"},
        ]
        chunks = chunk_transcript(entries)
        assert len(chunks) >= 1
        assert "Real text" in chunks[0]["transcript_text"]

    def test_preserves_timestamps(self):
        entries = [
            {"start_sec": 0.0, "end_sec": 5.0, "text": "Hello"},
            {"start_sec": 5.0, "end_sec": 10.0, "text": "World"},
        ]
        chunks = chunk_transcript(entries)
        assert chunks[0]["start_sec"] == 0.0
        assert chunks[0]["end_sec"] == 10.0


# ============================================================
# Idea Extraction Tests
# ============================================================

class TestIdeas:
    def test_definition(self):
        ideas = extract_ideas("A graph database is a type of database.")
        types = [i["idea_type"] for i in ideas]
        assert "definition" in types

    def test_definition_means(self):
        ideas = extract_ideas("This means we need to be careful.")
        types = [i["idea_type"] for i in ideas]
        assert "definition" in types

    def test_definition_defined_as(self):
        ideas = extract_ideas("A widget is defined as a reusable component.")
        types = [i["idea_type"] for i in ideas]
        assert "definition" in types

    def test_step(self):
        ideas = extract_ideas("First, install the package using pip.")
        types = [i["idea_type"] for i in ideas]
        assert "step" in types

    def test_step_next(self):
        ideas = extract_ideas("Next, configure the database connection.")
        types = [i["idea_type"] for i in ideas]
        assert "step" in types

    def test_step_click(self):
        ideas = extract_ideas("Click the submit button to continue.")
        types = [i["idea_type"] for i in ideas]
        assert "step" in types

    def test_step_install(self):
        ideas = extract_ideas("Install the dependencies with npm.")
        types = [i["idea_type"] for i in ideas]
        assert "step" in types

    def test_warning_careful(self):
        ideas = extract_ideas("Be careful when deleting files.")
        types = [i["idea_type"] for i in ideas]
        assert "warning" in types

    def test_warning_avoid(self):
        ideas = extract_ideas("Avoid using deprecated functions.")
        types = [i["idea_type"] for i in ideas]
        assert "warning" in types

    def test_warning_error(self):
        ideas = extract_ideas("This error occurs when memory is full.")
        types = [i["idea_type"] for i in ideas]
        assert "warning" in types

    def test_example(self):
        ideas = extract_ideas("For example, you can use a list comprehension.")
        types = [i["idea_type"] for i in ideas]
        assert "example" in types

    def test_example_eg(self):
        ideas = extract_ideas("Use functional components, e.g., ArrowFunction.")
        types = [i["idea_type"] for i in ideas]
        assert "example" in types

    def test_example_such_as(self):
        ideas = extract_ideas("Use tools such as Docker and Kubernetes.")
        types = [i["idea_type"] for i in ideas]
        assert "example" in types

    def test_comparison(self):
        ideas = extract_ideas("Python is better than Java for this task.")
        types = [i["idea_type"] for i in ideas]
        assert "comparison" in types

    def test_comparison_versus(self):
        ideas = extract_ideas("Option A versus Option B has tradeoffs.")
        types = [i["idea_type"] for i in ideas]
        assert "comparison" in types

    def test_claim_fallback(self):
        ideas = extract_ideas(
            "Quantum computing requires extremely precise control of qubits."
        )
        types = [i["idea_type"] for i in ideas]
        assert "claim" in types

    def test_short_sentence_no_idea(self):
        ideas = extract_ideas("Hi.")
        assert len(ideas) == 0

    def test_empty_text(self):
        assert extract_ideas("") == []
        assert extract_ideas("   ") == []

    def test_multiple_ideas_in_one_sentence(self):
        text = "First, install Docker. Avoid using outdated images."
        ideas = extract_ideas(text)
        types = [i["idea_type"] for i in ideas]
        assert "step" in types
        assert "warning" in types


# ============================================================
# Pipeline and Integration Tests
# ============================================================

class TestPipelineHelpers:
    def test_fallback_chunks_positive_duration(self):
        chunks = _create_fallback_chunks(120.0, "test_vid")
        assert len(chunks) == 1
        assert chunks[0]["start_sec"] == 0.0
        assert chunks[0]["end_sec"] == 60.0
        assert chunks[0]["transcript_text"] == ""

    def test_fallback_chunks_zero_duration(self):
        chunks = _create_fallback_chunks(0.0, "test_vid")
        assert len(chunks) == 1
        assert chunks[0]["end_sec"] == 60.0

    def test_get_keyframes_for_moment(self):
        keyframes = [
            {"timestamp_sec": 5.0, "path": "/f1.jpg"},
            {"timestamp_sec": 15.0, "path": "/f2.jpg"},
            {"timestamp_sec": 25.0, "path": "/f3.jpg"},
        ]
        chunk = {"start_sec": 10.0, "end_sec": 20.0}
        paths = _get_keyframes_for_moment(keyframes, chunk)
        assert paths == ["/f2.jpg"]

    def test_get_keyframes_no_match(self):
        keyframes = [{"timestamp_sec": 5.0, "path": "/f1.jpg"}]
        chunk = {"start_sec": 10.0, "end_sec": 20.0}
        assert _get_keyframes_for_moment(keyframes, chunk) == []

    def test_get_ocr_for_moment(self):
        ocr_results = [
            {"timestamp_sec": 5.0, "text": "Hello", "confidence": 0.9},
            {"timestamp_sec": 15.0, "text": "World", "confidence": 0.8},
        ]
        chunk = {"start_sec": 10.0, "end_sec": 20.0}
        text, ideas = _get_ocr_for_moment(ocr_results, chunk)
        assert "World" in text
        assert len(ideas) >= 0

    def test_evidence_insertion(self, db_conn):
        conn = db_conn
        init_db(conn)
        v = Video(video_id="ev_test", title="Test", source="local", duration_sec=60.0)
        insert_video(conn, v)

        moment_id = make_moment_id("ev_test", 0.0, 10.0)
        moment = Moment(
            moment_id=moment_id,
            video_id="ev_test",
            start_sec=0.0,
            end_sec=10.0,
            transcript_text="Hello world",
            ideas=[
                Idea(
                    idea_id=make_idea_id(moment_id, 0),
                    moment_id=moment_id,
                    type="definition",
                    text="Hello world is a greeting.",
                )
            ],
        )
        insert_moment(conn, moment)
        conn.commit()

        chunk_map = {
            moment_id: {
                "start_sec": 0.0,
                "end_sec": 10.0,
                "transcript_text": "Hello world",
            }
        }

        _insert_evidence_for_moment(
            conn, moment, chunk_map, ocr_results=[], keyframes=[]
        )
        conn.commit()

        count = get_evidence_count_by_video(conn, "ev_test")
        assert count > 0

    def test_keyframe_insertion(self, db_conn):
        conn = db_conn
        init_db(conn)
        v = Video(video_id="kf_test", title="Test", source="local", duration_sec=60.0)
        insert_video(conn, v)

        kf = Keyframe(
            keyframe_id="kf:test123",
            moment_id=None,
            video_id="kf_test",
            timestamp_sec=10.0,
            file_path="/tmp/frame.jpg",
        )
        insert_keyframe(conn, kf)
        conn.commit()

        assert get_keyframe_count_by_video(conn, "kf_test") == 1


class TestIngestProcess:
    def _run_process_and_get_video_id(self, tmpdir: str) -> str:
        result = runner.invoke(
            app,
            [
                "ingest", str(Path(tmpdir, "test_video.mp4")),
                "--data-dir", tmpdir,
                "--process",
            ],
        )
        assert result.exit_code == 0, result.stdout
        for line in result.stdout.splitlines():
            if "Processed video:" in line:
                return line.split("Processed video:")[1].strip()
        return ""

    def test_ingest_local_with_process_creates_moments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir, "test_video.mp4")
            video_path.write_text("fake video content")

            transcript_path = Path(tmpdir, "test_video.vtt")
            transcript_path.write_text(VTT_CONTENT)

            video_id = self._run_process_and_get_video_id(tmpdir)

            db_path = Path(tmpdir) / "vidcrawl.db"
            assert db_path.exists()

            conn = get_db(str(db_path))
            init_db(conn)
            moments = get_moments_by_video(conn, video_id)
            assert len(moments) > 0
            conn.close()

    def test_ingest_local_without_process_registers_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir, "test_video.mp4")
            video_path.write_text("fake video content")

            result = runner.invoke(
                app,
                [
                    "ingest", str(video_path),
                    "--data-dir", tmpdir,
                    "--no-process",
                ],
            )
            assert result.exit_code == 0

    def test_inspect_with_moments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir, "test_video.mp4")
            video_path.write_text("fake video content")

            transcript_path = Path(tmpdir, "test_video.vtt")
            transcript_path.write_text(VTT_CONTENT)

            video_id = self._run_process_and_get_video_id(tmpdir)
            assert video_id, "Could not extract video_id from output"

            result = runner.invoke(
                app,
                ["inspect", video_id, "--data-dir", tmpdir],
            )
            assert result.exit_code == 0, result.stdout
            assert "Moments" in result.stdout
            assert "Evidence" in result.stdout
            assert "Keyframes" in result.stdout
            assert "Ideas" in result.stdout


class TestKeyframeOCR:
    def test_extract_keyframes_no_ffmpeg(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = extract_keyframes(
                "/nonexistent/video.mp4",
                "/tmp/output",
            )
            assert result == []

    def test_ocr_frames_empty(self):
        assert ocr_frames([]) == []

    def test_ocr_frames_nonexistent_file(self):
        result = ocr_frames([{"path": "/nonexistent.jpg", "timestamp_sec": 0.0}])
        assert result == []
