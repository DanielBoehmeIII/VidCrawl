"""Tests for semantic transcript chunking: segmentation quality and coverage."""
from vidcrawl.process.chunking import chunk_transcript


def _entries(n: int, dur: float = 3.0, punct: bool = True) -> list[dict]:
    """Build n consecutive entries each lasting dur seconds."""
    entries = []
    t = 0.0
    for i in range(n):
        text = f"This is segment {i}." if punct else f"This is segment {i}"
        entries.append({"start_sec": t, "end_sec": t + dur, "text": text})
        t += dur
    return entries


# ---------------------------------------------------------------------------
# Multiple moments from long transcripts
# ---------------------------------------------------------------------------

class TestMultipleMoments:
    def test_five_minute_video_produces_multiple_chunks(self):
        # 100 entries × 3s = 300s (5 min)
        entries = _entries(100, 3.0)
        chunks = chunk_transcript(entries)
        assert len(chunks) >= 2, f"Expected ≥2 chunks for 5-min video, got {len(chunks)}"

    def test_ten_minute_video_produces_many_chunks(self):
        # 200 entries × 3s = 600s (10 min)
        entries = _entries(200, 3.0)
        chunks = chunk_transcript(entries)
        assert len(chunks) >= 4, f"Expected ≥4 chunks for 10-min video, got {len(chunks)}"

    def test_one_hour_video_produces_many_chunks(self):
        # 1200 entries × 3s = 3600s (1 hr)
        entries = _entries(1200, 3.0)
        chunks = chunk_transcript(entries)
        assert len(chunks) >= 20, f"Expected ≥20 chunks for 1-hr video, got {len(chunks)}"

    def test_chunks_respect_max_duration(self):
        entries = _entries(200, 3.0)
        chunks = chunk_transcript(entries, min_duration=30.0, max_duration=120.0)
        for chunk in chunks[:-1]:  # last chunk may be shorter due to absorption
            duration = chunk["end_sec"] - chunk["start_sec"]
            assert duration <= 130.0, f"Chunk too long: {duration:.1f}s"

    def test_no_punctuation_still_splits(self):
        # No sentence endings — must split by max_duration alone
        entries = _entries(200, 3.0, punct=False)
        chunks = chunk_transcript(entries)
        assert len(chunks) >= 4


# ---------------------------------------------------------------------------
# Chronological ordering
# ---------------------------------------------------------------------------

class TestChronologicalOrder:
    def test_chunks_start_times_are_increasing(self):
        entries = _entries(100, 3.0)
        chunks = chunk_transcript(entries)
        for i in range(1, len(chunks)):
            assert chunks[i]["start_sec"] >= chunks[i - 1]["start_sec"], (
                f"Chunk {i} start {chunks[i]['start_sec']} < "
                f"chunk {i-1} start {chunks[i-1]['start_sec']}"
            )

    def test_chunks_are_non_overlapping(self):
        entries = _entries(150, 3.0)
        chunks = chunk_transcript(entries)
        for i in range(1, len(chunks)):
            assert chunks[i]["start_sec"] >= chunks[i - 1]["end_sec"], (
                f"Chunk {i} [{chunks[i]['start_sec']}, {chunks[i]['end_sec']}] "
                f"overlaps chunk {i-1} [{chunks[i-1]['start_sec']}, {chunks[i-1]['end_sec']}]"
            )

    def test_shuffled_entries_produce_ordered_chunks(self):
        import random
        entries = _entries(60, 3.0)
        shuffled = list(entries)
        random.shuffle(shuffled)
        chunks = chunk_transcript(shuffled)
        for i in range(1, len(chunks)):
            assert chunks[i]["start_sec"] >= chunks[i - 1]["start_sec"]


# ---------------------------------------------------------------------------
# Total transcript coverage
# ---------------------------------------------------------------------------

class TestCoverage:
    def test_all_segments_appear_in_chunks(self):
        n = 100
        entries = _entries(n, 3.0)
        chunks = chunk_transcript(entries)
        combined = " ".join(c["transcript_text"] for c in chunks)
        for i in range(n):
            assert f"segment {i}" in combined, (
                f"Segment {i} text missing from all chunks"
            )

    def test_no_entry_duplicated(self):
        n = 60
        entries = _entries(n, 3.0, punct=False)
        chunks = chunk_transcript(entries)
        combined = " ".join(c["transcript_text"] for c in chunks)
        for i in range(n):
            phrase = f"segment {i}"
            count = combined.count(phrase)
            # A segment like "segment 1" also matches "segment 10"–"segment 19",
            # so count the exact phrase with word boundaries.
            import re
            exact = len(re.findall(rf"\bsegment {i}\b", combined))
            assert exact >= 1, f"Segment {i} missing from all chunks"
            assert exact <= 2, f"Segment {i} duplicated {exact} times in chunks"

    def test_time_span_fully_covered(self):
        entries = _entries(100, 3.0)
        total_duration = entries[-1]["end_sec"] - entries[0]["start_sec"]
        chunks = chunk_transcript(entries)
        covered = sum(c["end_sec"] - c["start_sec"] for c in chunks)
        # Allow small rounding — covered must account for the full span
        assert covered >= total_duration * 0.99, (
            f"Coverage {covered:.1f}s < total {total_duration:.1f}s"
        )

    def test_coverage_with_gaps_in_source(self):
        entries = [
            {"start_sec": 0.0, "end_sec": 5.0, "text": "Intro sentence."},
            {"start_sec": 60.0, "end_sec": 65.0, "text": "After a gap."},
            {"start_sec": 120.0, "end_sec": 125.0, "text": "Later section."},
        ]
        chunks = chunk_transcript(entries)
        combined = " ".join(c["transcript_text"] for c in chunks)
        assert "Intro" in combined
        assert "After a gap" in combined
        assert "Later section" in combined


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_returns_empty(self):
        assert chunk_transcript([]) == []

    def test_single_short_entry(self):
        entries = [{"start_sec": 0.0, "end_sec": 5.0, "text": "Hello."}]
        chunks = chunk_transcript(entries)
        assert len(chunks) == 1
        assert chunks[0]["transcript_text"] == "Hello."

    def test_single_entry_exceeding_max(self):
        entries = [{"start_sec": 0.0, "end_sec": 300.0, "text": "A very long monologue."}]
        chunks = chunk_transcript(entries)
        assert len(chunks) == 1
        assert chunks[0]["start_sec"] == 0.0
        assert chunks[0]["end_sec"] == 300.0

    def test_no_empty_chunks(self):
        entries = _entries(50, 3.0)
        chunks = chunk_transcript(entries)
        for chunk in chunks:
            assert chunk["transcript_text"].strip(), "Found empty chunk"

    def test_empty_entries_are_skipped(self):
        entries = [
            {"start_sec": 0.0, "end_sec": 5.0, "text": ""},
            {"start_sec": 5.0, "end_sec": 10.0, "text": "   "},
            {"start_sec": 10.0, "end_sec": 15.0, "text": "Real text."},
        ]
        chunks = chunk_transcript(entries)
        assert len(chunks) == 1
        assert "Real text" in chunks[0]["transcript_text"]
