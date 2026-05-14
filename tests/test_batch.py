"""Tests for the batch ingest command."""
import re
import tempfile
from pathlib import Path

from typer.testing import CliRunner

from vidcrawl.cli import app
from vidcrawl.config import get_config
from vidcrawl.db import get_db, init_db, insert_video, make_video_id
from vidcrawl.ingest.downloader import normalize_youtube_url
from vidcrawl.models import Video

runner = CliRunner()


def _long_vtt(n_entries: int = 50, entry_dur: float = 3.0) -> str:
    """VTT with n_entries × entry_dur seconds; sentence end every 10 entries."""
    lines = ["WEBVTT\n"]
    t = 0.0
    for i in range(n_entries):
        h, m_, s = int(t // 3600), int((t % 3600) // 60), t % 60
        start_ts = f"{h:02d}:{m_:02d}:{s:06.3f}"
        t += entry_dur
        h, m_, s = int(t // 3600), int((t % 3600) // 60), t % 60
        end_ts = f"{h:02d}:{m_:02d}:{s:06.3f}"
        punct = "." if (i + 1) % 10 == 0 else ""
        lines.append(
            f"\n{start_ts} --> {end_ts}\n"
            f"This is segment {i} of the long test transcript{punct}\n"
        )
    return "".join(lines)


def _make_source_list(paths_or_urls: list[str]) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    f.write("\n".join(paths_or_urls))
    f.close()
    return Path(f.name)


def _make_fake_video(tmpdir: str, name: str = "video.mp4") -> Path:
    p = Path(tmpdir) / name
    p.write_bytes(b"fake video content")
    return p


class TestBatchLimit:
    def test_limit_respected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [_make_fake_video(tmpdir, f"v{i}.mp4") for i in range(5)]
            source_file = _make_source_list([str(f) for f in files])

            runner.invoke(app, ["init", "--data-dir", tmpdir])
            result = runner.invoke(app, [
                "batch", str(source_file),
                "--data-dir", tmpdir,
                "--limit", "2",
                "--no-process",
                "--rate-limit-sec", "0",
            ])

            source_file.unlink(missing_ok=True)
            assert result.exit_code == 0, result.output
            assert "Attempted:        2" in result.stdout


class TestBatchSkipExisting:
    def test_skips_already_processed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = _make_fake_video(tmpdir)
            video_id = make_video_id("local", str(video_path.resolve()))
            db_path = Path(tmpdir) / "vidcrawl.db"

            runner.invoke(app, ["init", "--data-dir", tmpdir])

            conn = get_db(str(db_path))
            insert_video(conn, Video(
                video_id=video_id,
                title=video_path.stem,
                source="local",
                url=None,
                duration_sec=0.0,
                status="ready",
            ))
            conn.commit()
            conn.close()

            source_file = _make_source_list([str(video_path)])
            result = runner.invoke(app, [
                "batch", str(source_file),
                "--data-dir", tmpdir,
                "--limit", "10",
                "--no-process",
                "--rate-limit-sec", "0",
            ])

            source_file.unlink(missing_ok=True)
            assert result.exit_code == 0, result.output
            assert "Skipped existing: 1" in result.stdout
            assert "Inserted:         0" in result.stdout


class TestBatchContinueOnFailure:
    def test_failure_does_not_stop_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            good_video = _make_fake_video(tmpdir, "good.mp4")
            bad_path = "/nonexistent/does/not/exist/bad.mp4"

            source_file = _make_source_list([bad_path, str(good_video)])
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            result = runner.invoke(app, [
                "batch", str(source_file),
                "--data-dir", tmpdir,
                "--limit", "10",
                "--no-process",
                "--rate-limit-sec", "0",
            ])

            source_file.unlink(missing_ok=True)
            assert result.exit_code == 0, result.output
            assert "Failed:           1" in result.stdout
            assert "Inserted:         1" in result.stdout

    def test_failed_source_is_listed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = "/nonexistent/does/not/exist/bad.mp4"
            source_file = _make_source_list([bad_path])

            runner.invoke(app, ["init", "--data-dir", tmpdir])
            result = runner.invoke(app, [
                "batch", str(source_file),
                "--data-dir", tmpdir,
                "--limit", "10",
                "--no-process",
                "--rate-limit-sec", "0",
            ])

            source_file.unlink(missing_ok=True)
            assert result.exit_code == 0, result.output
            assert bad_path in result.stdout


class TestBatchDryRun:
    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video = _make_fake_video(tmpdir)
            source_file = _make_source_list([str(video)])
            db_path = Path(tmpdir) / "vidcrawl.db"

            runner.invoke(app, ["init", "--data-dir", tmpdir])
            result = runner.invoke(app, [
                "batch", str(source_file),
                "--data-dir", tmpdir,
                "--dry-run",
                "--no-process",
                "--rate-limit-sec", "0",
            ])

            source_file.unlink(missing_ok=True)
            assert result.exit_code == 0, result.output
            assert "dry run" in result.stdout.lower()

            conn = get_db(str(db_path))
            count = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
            conn.close()
            assert count == 0

    def test_dry_run_shows_would_ingest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video = _make_fake_video(tmpdir)
            source_file = _make_source_list([str(video)])

            runner.invoke(app, ["init", "--data-dir", tmpdir])
            result = runner.invoke(app, [
                "batch", str(source_file),
                "--data-dir", tmpdir,
                "--dry-run",
                "--no-process",
                "--rate-limit-sec", "0",
            ])

            source_file.unlink(missing_ok=True)
            assert result.exit_code == 0, result.output
            assert "Would ingest" in result.stdout


class TestYouTubeCaptionBatch:
    """Regression: moments/evidence/ideas must be linked to the YouTube video_id."""

    def test_pipeline_linked_to_youtube_video_id(self):
        """process_local_video with video_id_override attaches all data to
        the YouTube video_id, so inspect and stats reflect it correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            config = get_config(tmpdir)

            yt_url = "https://www.youtube.com/watch?v=dQw4w9WgXcZ"
            video_id = make_video_id("youtube", yt_url)  # "dQw4w9WgXcZ"

            # Simulate a downloaded video file under the YouTube video dir
            video_dir = config.videos_dir / video_id
            video_dir.mkdir(parents=True, exist_ok=True)
            fake_video = video_dir / f"{video_id}.mp4"
            fake_video.write_bytes(b"not a real mp4")

            # Sidecar transcript — triggers warning + step ideas
            sidecar = video_dir / f"{video_id}.vtt"
            sidecar.write_text(
                "WEBVTT\n\n"
                "00:00:00.000 --> 00:00:05.000\n"
                "Warning: always define your variables before using them.\n\n"
                "00:00:05.000 --> 00:00:10.000\n"
                "Then run the install command to finish setup.\n"
            )

            # Register the YouTube video (mimics what _batch_ingest_youtube does)
            db_path = config.db_path
            with get_db(str(db_path)) as conn:
                init_db(conn)
                insert_video(conn, Video(
                    video_id=video_id,
                    title="Test YouTube Video",
                    source="youtube",
                    url=yt_url,
                    duration_sec=10.0,
                    status="pending",
                ))

            # Run the pipeline with the YouTube video_id override
            from vidcrawl.process.pipeline import process_local_video
            returned_id = process_local_video(
                str(fake_video), config,
                video_id_override=video_id,
            )

            assert returned_id == video_id, (
                f"process_local_video should return the override id, got {returned_id!r}"
            )

            # --- stats should show nonzero global counts ---
            stats_result = runner.invoke(app, ["stats", "--data-dir", tmpdir])
            assert stats_result.exit_code == 0, stats_result.output
            stats_lines = {
                line.split(":")[0].strip(): int(line.split(":")[1].strip())
                for line in stats_result.output.splitlines()
                if ":" in line and line.split(":")[1].strip().isdigit()
            }
            assert stats_lines.get("Moments", 0) > 0, "stats: Moments should be > 0"
            assert stats_lines.get("Evidence", 0) > 0, "stats: Evidence should be > 0"
            assert stats_lines.get("Ideas", 0) > 0, "stats: Ideas should be > 0"

            # --- inspect on the YOUTUBE video_id should show the same data ---
            inspect_result = runner.invoke(
                app, ["inspect", video_id, "--data-dir", tmpdir]
            )
            assert inspect_result.exit_code == 0, inspect_result.output
            assert "Status:      ready" in inspect_result.output, (
                "Video status should be 'ready' after pipeline"
            )
            inspect_lines = {
                line.split(":")[0].strip(): line.split(":")[1].strip()
                for line in inspect_result.output.splitlines()
                if ":" in line
            }
            assert int(inspect_lines.get("Moments", "0")) > 0, (
                "inspect: Moments should be > 0 for the YouTube video_id"
            )
            assert int(inspect_lines.get("Evidence", "0")) > 0, (
                "inspect: Evidence should be > 0 for the YouTube video_id"
            )
            assert int(inspect_lines.get("Ideas", "0")) > 0, (
                "inspect: Ideas should be > 0 for the YouTube video_id"
            )

    def test_rerun_clears_old_data_no_unique_constraint(self):
        """Re-processing an existing video must not raise UNIQUE constraint errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            config = get_config(tmpdir)

            yt_url = "https://www.youtube.com/watch?v=abcdefghijk"
            video_id = make_video_id("youtube", yt_url)

            video_dir = config.videos_dir / video_id
            video_dir.mkdir(parents=True, exist_ok=True)
            fake_video = video_dir / f"{video_id}.mp4"
            fake_video.write_bytes(b"not a real mp4")

            sidecar = video_dir / f"{video_id}.vtt"
            sidecar.write_text(
                "WEBVTT\n\n"
                "00:00:00.000 --> 00:00:05.000\n"
                "Warning: do not ignore errors in production.\n"
            )

            with get_db(str(config.db_path)) as conn:
                init_db(conn)
                insert_video(conn, Video(
                    video_id=video_id,
                    title="Rerun Test",
                    source="youtube",
                    url=yt_url,
                    duration_sec=5.0,
                    status="pending",
                ))

            from vidcrawl.process.pipeline import process_local_video

            # First run
            process_local_video(str(fake_video), config, video_id_override=video_id)

            # Second run on same video — must not raise UNIQUE constraint
            process_local_video(str(fake_video), config, video_id_override=video_id)

            inspect_result = runner.invoke(
                app, ["inspect", video_id, "--data-dir", tmpdir]
            )
            assert inspect_result.exit_code == 0, inspect_result.output
            assert "Status:      ready" in inspect_result.output


class TestForceReprocess:
    """Regression: --force must clear old data and rebuild with correct moment counts."""

    def test_force_clears_and_rebuilds_moments(self):
        """process_local_video re-run must produce ≥2 moments and replace (not append) old data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            config = get_config(tmpdir)

            fake_video = Path(tmpdir) / "long_video.mp4"
            fake_video.write_bytes(b"not a real mp4")

            # 50 entries × 3s = 150s total; sentence ends at entries 9,19,29,39,49
            # → chunker must split at ≥2 boundaries before 120s max_duration
            vtt = Path(tmpdir) / "long_video.vtt"
            vtt.write_text(_long_vtt(50, 3.0))

            from vidcrawl.process.pipeline import process_local_video

            # First pass
            video_id = process_local_video(str(fake_video), config)

            with get_db(str(config.db_path)) as conn:
                first_count = conn.execute(
                    "SELECT COUNT(*) FROM moments WHERE video_id = ?", (video_id,)
                ).fetchone()[0]

            assert first_count >= 2, (
                f"Expected ≥2 moments after first pass, got {first_count}"
            )

            # Second pass (simulates --force): must clear then rebuild, not append
            process_local_video(str(fake_video), config)

            with get_db(str(config.db_path)) as conn:
                second_count = conn.execute(
                    "SELECT COUNT(*) FROM moments WHERE video_id = ?", (video_id,)
                ).fetchone()[0]
                v = conn.execute(
                    "SELECT status FROM videos WHERE video_id = ?", (video_id,)
                ).fetchone()

            assert v["status"] == "ready", f"Status should be ready, got {v['status']}"
            assert second_count >= 2, (
                f"Expected ≥2 moments after rebuild, got {second_count}"
            )
            # Data should be replaced, not doubled
            assert second_count <= first_count * 2, (
                f"Moments doubled ({first_count} → {second_count}) — data was appended not rebuilt"
            )

    def test_batch_force_reports_nonzero_rebuilt_counts(self):
        """Batch --force must report nonzero rebuilt counts, not a misleading delta of 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_video = Path(tmpdir) / "long_video.mp4"
            fake_video.write_bytes(b"not a real mp4")

            vtt = Path(tmpdir) / "long_video.vtt"
            vtt.write_text(_long_vtt(50, 3.0))

            source_file = _make_source_list([str(fake_video)])
            runner.invoke(app, ["init", "--data-dir", tmpdir])

            # First batch run (no force)
            runner.invoke(app, [
                "batch", str(source_file),
                "--data-dir", tmpdir,
                "--limit", "1",
                "--rate-limit-sec", "0",
            ])

            # Second batch run with --force
            result = runner.invoke(app, [
                "batch", str(source_file),
                "--data-dir", tmpdir,
                "--limit", "1",
                "--force",
                "--rate-limit-sec", "0",
            ])

            source_file.unlink(missing_ok=True)
            assert result.exit_code == 0, result.output

            # Find the per-video OK line and check rebuilt moment count > 0
            ok_lines = [ln for ln in result.output.splitlines() if "OK:" in ln]
            assert ok_lines, f"No OK line in output:\n{result.output}"
            m = re.search(r"OK:\s*(\d+)\s*moments\s*rebuilt", ok_lines[0])
            assert m is not None, f"Expected 'X moments rebuilt' in: {ok_lines[0]}"
            assert int(m.group(1)) > 0, (
                f"Expected > 0 moments rebuilt, got: {ok_lines[0]}"
            )

    def test_end_time_never_before_start_time(self):
        """Moments produced from YouTube-style VTT (end_sec=0 bug) must have end≥start."""
        from vidcrawl.process.chunking import chunk_transcript

        # Simulate what the broken YouTube VTT parser used to produce:
        # start_sec is correct, end_sec was always 0 due to alignment cue parsing failure.
        broken_entries = [
            {"start_sec": float(i) * 3.0, "end_sec": 0.0,
             "text": f"Segment {i}." if (i + 1) % 10 == 0 else f"Segment {i}"}
            for i in range(50)
        ]

        chunks = chunk_transcript(broken_entries)

        for chunk in chunks:
            assert chunk["end_sec"] >= chunk["start_sec"], (
                f"Chunk has end before start: [{chunk['start_sec']}s - {chunk['end_sec']}s]"
            )

    def test_youtube_vtt_alignment_cues_parse_correctly(self):
        """VTT timestamps with YouTube cue settings must parse to the correct seconds."""
        from vidcrawl.ingest.transcript import _parse_vtt

        vtt_with_cues = (
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:02.819 align:start position:0%\n"
            "Hello world.\n\n"
            "00:00:02.819 --> 00:00:05.200 align:start position:0%\n"
            "How are you doing today.\n\n"
            "00:01:00.000 --> 00:01:03.500 align:start position:0%\n"
            "After one minute.\n"
        )

        entries = _parse_vtt(vtt_with_cues)

        assert len(entries) == 3, f"Expected 3 entries, got {len(entries)}"
        assert entries[0]["start_sec"] == 0.0
        assert abs(entries[0]["end_sec"] - 2.819) < 0.001, (
            f"end_sec should be 2.819, got {entries[0]['end_sec']}"
        )
        assert abs(entries[1]["end_sec"] - 5.2) < 0.001, (
            f"end_sec should be 5.2, got {entries[1]['end_sec']}"
        )
        assert abs(entries[2]["start_sec"] - 60.0) < 0.001
        assert abs(entries[2]["end_sec"] - 63.5) < 0.001

        for e in entries:
            assert e["end_sec"] >= e["start_sec"], (
                f"end_sec {e['end_sec']} < start_sec {e['start_sec']}"
            )


class TestCaptionOnlyConfidence:
    """Regression: Evidence.confidence must be a float, never None, in caption-only batches."""

    def test_caption_only_evidence_confidence_always_float(self):
        """Caption-only pipeline (prefer_yt_captions, no download) must write
        Evidence rows with float confidence even when OCR returns confidence=None."""
        from unittest.mock import patch

        from vidcrawl.db import init_db
        from vidcrawl.models import Video
        from vidcrawl.process.pipeline import process_local_video

        caption_entries = [
            {"start_sec": 0.0, "end_sec": 35.0, "text": "Welcome to the tutorial."},
            {"start_sec": 35.0, "end_sec": 70.0, "text": "First, install the package."},
            {"start_sec": 70.0, "end_sec": 105.0, "text": "Then configure your settings."},
            {"start_sec": 105.0, "end_sec": 120.0, "text": "Finally, run the app."},
        ]
        # Simulate an OCR result whose confidence is None (conf_count==0 in tesseract).
        # This is the exact dict shape ocr.py produces; pipeline must not pass it through as None.
        ocr_with_none = [
            {"timestamp_sec": 15.0, "text": "SLIDE TITLE", "confidence": None},
        ]
        fake_keyframes = [{"timestamp_sec": 15.0, "path": "/fake/frame_000015.jpg"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            config = get_config(tmpdir)

            yt_url = "https://www.youtube.com/watch?v=captionconftest"
            video_id = make_video_id("youtube", yt_url)

            video_dir = config.videos_dir / video_id
            video_dir.mkdir(parents=True, exist_ok=True)
            placeholder = video_dir / f"{video_id}.mp4"
            placeholder.write_bytes(b"")

            with get_db(str(config.db_path)) as conn:
                init_db(conn)
                insert_video(conn, Video(
                    video_id=video_id,
                    title="Caption Confidence Test",
                    source="youtube",
                    url=yt_url,
                    duration_sec=120.0,
                    status="pending",
                ))

            with (
                patch("vidcrawl.process.pipeline.fetch_youtube_captions",
                      return_value=caption_entries),
                patch("vidcrawl.process.pipeline.extract_keyframes",
                      return_value=fake_keyframes),
                patch("vidcrawl.process.pipeline.ocr_frames",
                      return_value=ocr_with_none),
            ):
                returned_id = process_local_video(
                    str(placeholder), config,
                    source_url=yt_url,
                    video_id_override=video_id,
                    prefer_yt_captions=True,
                    allow_whisper=False,
                )

            assert returned_id == video_id

            with get_db(str(config.db_path)) as conn:
                rows = conn.execute(
                    "SELECT confidence FROM modal_evidence "
                    "WHERE moment_id IN "
                    "(SELECT moment_id FROM moments WHERE video_id = ?)",
                    (video_id,),
                ).fetchall()

            assert len(rows) > 0, "Expected at least one Evidence record"
            for row in rows:
                assert row["confidence"] is not None, (
                    "Evidence.confidence must not be None"
                )
                assert isinstance(row["confidence"], float), (
                    f"Evidence.confidence must be float, got {type(row['confidence'])}"
                )


class TestYouTubeUrlNormalization:
    """Regression: messy playlist/radio URLs must collapse to a plain watch URL."""

    CANONICAL = "https://www.youtube.com/watch?v=xsyk-Xmr6wQ"

    def test_playlist_params_stripped(self):
        url = "https://www.youtube.com/watch?v=xsyk-Xmr6wQ&list=RDxsyk-Xmr6wQ&start_radio=1"
        assert normalize_youtube_url(url) == self.CANONICAL

    def test_start_and_index_stripped(self):
        url = "https://www.youtube.com/watch?v=xsyk-Xmr6wQ&start=120&index=3"
        assert normalize_youtube_url(url) == self.CANONICAL

    def test_extra_query_params_stripped(self):
        url = "https://www.youtube.com/watch?v=xsyk-Xmr6wQ&pp=ABCDE&si=xyz"
        assert normalize_youtube_url(url) == self.CANONICAL

    def test_short_youtu_be_url(self):
        url = "https://youtu.be/xsyk-Xmr6wQ"
        assert normalize_youtube_url(url) == self.CANONICAL

    def test_short_youtu_be_with_params(self):
        url = "https://youtu.be/xsyk-Xmr6wQ?si=abc&t=30"
        assert normalize_youtube_url(url) == self.CANONICAL

    def test_mobile_url(self):
        url = "https://m.youtube.com/watch?v=xsyk-Xmr6wQ&list=PLfoo"
        assert normalize_youtube_url(url) == self.CANONICAL

    def test_already_canonical_unchanged(self):
        assert normalize_youtube_url(self.CANONICAL) == self.CANONICAL

    def test_non_youtube_url_unchanged(self):
        url = "https://vimeo.com/123456789"
        assert normalize_youtube_url(url) == url

    def test_normalized_url_produces_same_video_id(self):
        """Normalization ensures make_video_id is deterministic across URL variants."""
        messy = "https://www.youtube.com/watch?v=xsyk-Xmr6wQ&list=RDxsyk-Xmr6wQ&start_radio=1"
        clean = normalize_youtube_url(messy)
        assert make_video_id("youtube", messy) == make_video_id("youtube", clean)

    def test_batch_normalizes_url_before_processing(self):
        """Batch command must print the normalized URL, not the raw messy one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            messy_url = (
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                "&list=RDdQw4w9WgXcQ&start_radio=1&index=2"
            )
            source_file = _make_source_list([messy_url])
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            result = runner.invoke(app, [
                "batch", str(source_file),
                "--data-dir", tmpdir,
                "--limit", "1",
                "--dry-run",
                "--rate-limit-sec", "0",
            ])
            source_file.unlink(missing_ok=True)
            assert result.exit_code == 0, result.output
            # Normalized URL has no list= or start_radio= params
            assert "list=" not in result.stdout
            assert "start_radio=" not in result.stdout


class TestNormalizedUrlStored:
    """Issue 1: The stored video URL must be the normalized clean watch URL."""

    def test_stored_url_is_normalized_on_first_insert(self):
        """When a YouTube video is registered for the first time, its URL in the
        DB must be the normalized form (no list/start_radio params)."""
        from unittest.mock import patch

        messy_url = (
            "https://www.youtube.com/watch?v=STOREDURL0A"
            "&list=RDtest&start_radio=1"
        )
        clean_url = "https://www.youtube.com/watch?v=STOREDURL0A"
        video_id = make_video_id("youtube", messy_url)

        caption_entries = [
            {"start_sec": 0.0, "end_sec": 35.0, "text": "Hello world."},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            config = get_config(tmpdir)

            video_dir = config.videos_dir / video_id
            video_dir.mkdir(parents=True, exist_ok=True)
            placeholder = video_dir / f"{video_id}.mp4"
            placeholder.write_bytes(b"")

            with (
                patch("vidcrawl.ingest.downloader.extract_youtube_metadata",
                      return_value={"duration": 35, "title": "Test"}),
                patch("vidcrawl.ingest.downloader.is_yt_dlp_available", return_value=True),
                patch("vidcrawl.process.pipeline.fetch_youtube_captions",
                      return_value=caption_entries),
                patch("vidcrawl.process.pipeline.extract_keyframes", return_value=[]),
                patch("vidcrawl.process.pipeline.ocr_frames", return_value=[]),
            ):
                source_file = _make_source_list([messy_url])
                result = runner.invoke(app, [
                    "batch", str(source_file),
                    "--data-dir", tmpdir,
                    "--limit", "1",
                    "--no-download",
                    "--prefer-yt-captions",
                    "--rate-limit-sec", "0",
                ])
                source_file.unlink(missing_ok=True)

            assert result.exit_code == 0, result.output

            with get_db(str(config.db_path)) as conn:
                v = conn.execute(
                    "SELECT url FROM videos WHERE video_id = ?", (video_id,)
                ).fetchone()

            assert v is not None, "Video not found in DB"
            assert v[0] == clean_url, (
                f"Expected normalized URL {clean_url!r}, got {v[0]!r}"
            )

    def test_stored_url_updated_to_normalized_for_existing_video(self):
        """Re-running batch over a video that was previously stored with a messy URL
        must update the stored URL to the normalized form."""
        from unittest.mock import patch

        messy_url = (
            "https://www.youtube.com/watch?v=STOREDURL0B"
            "&list=PLtest&index=2"
        )
        clean_url = "https://www.youtube.com/watch?v=STOREDURL0B"
        video_id = make_video_id("youtube", messy_url)

        caption_entries = [
            {"start_sec": 0.0, "end_sec": 35.0, "text": "Hello world."},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            config = get_config(tmpdir)

            # Pre-register with the messy URL (simulates an old ingest run)
            with get_db(str(config.db_path)) as conn:
                init_db(conn)
                insert_video(conn, Video(
                    video_id=video_id,
                    title="Old title",
                    source="youtube",
                    url=messy_url,
                    duration_sec=0.0,
                    status="pending",
                ))

            video_dir = config.videos_dir / video_id
            video_dir.mkdir(parents=True, exist_ok=True)
            placeholder = video_dir / f"{video_id}.mp4"
            placeholder.write_bytes(b"")

            with (
                patch("vidcrawl.ingest.downloader.extract_youtube_metadata",
                      return_value={"duration": 35, "title": "Test"}),
                patch("vidcrawl.ingest.downloader.is_yt_dlp_available", return_value=True),
                patch("vidcrawl.process.pipeline.fetch_youtube_captions",
                      return_value=caption_entries),
                patch("vidcrawl.process.pipeline.extract_keyframes", return_value=[]),
                patch("vidcrawl.process.pipeline.ocr_frames", return_value=[]),
            ):
                source_file = _make_source_list([messy_url])
                result = runner.invoke(app, [
                    "batch", str(source_file),
                    "--data-dir", tmpdir,
                    "--limit", "1",
                    "--no-download",
                    "--prefer-yt-captions",
                    "--force",
                    "--rate-limit-sec", "0",
                ])
                source_file.unlink(missing_ok=True)

            assert result.exit_code == 0, result.output

            with get_db(str(config.db_path)) as conn:
                v = conn.execute(
                    "SELECT url FROM videos WHERE video_id = ?", (video_id,)
                ).fetchone()

            assert v is not None
            assert v[0] == clean_url, (
                f"Stored URL should be normalized; got {v[0]!r}"
            )


class TestDurationInference:
    """Issue 2: When the video file has no readable duration (placeholder / stub),
    duration_sec must be inferred from caption end times, not left as 0."""

    def test_duration_inferred_from_caption_entries(self):
        """process_local_video with a zero-byte placeholder must update duration_sec
        using the max end_sec of the supplied caption entries."""
        from unittest.mock import patch

        from vidcrawl.process.pipeline import process_local_video

        caption_entries = [
            {"start_sec": 0.0, "end_sec": 35.0, "text": "First segment."},
            {"start_sec": 35.0, "end_sec": 70.0, "text": "Second segment."},
            {"start_sec": 70.0, "end_sec": 123.0, "text": "Third segment."},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            config = get_config(tmpdir)

            yt_url = "https://www.youtube.com/watch?v=durtest0001"
            video_id = make_video_id("youtube", yt_url)

            video_dir = config.videos_dir / video_id
            video_dir.mkdir(parents=True, exist_ok=True)
            placeholder = video_dir / f"{video_id}.mp4"
            placeholder.write_bytes(b"")

            with get_db(str(config.db_path)) as conn:
                init_db(conn)
                insert_video(conn, Video(
                    video_id=video_id,
                    title="Duration Test",
                    source="youtube",
                    url=yt_url,
                    duration_sec=0.0,
                    status="pending",
                ))

            with (
                patch("vidcrawl.process.pipeline.fetch_youtube_captions",
                      return_value=caption_entries),
                patch("vidcrawl.process.pipeline.extract_keyframes", return_value=[]),
                patch("vidcrawl.process.pipeline.ocr_frames", return_value=[]),
            ):
                process_local_video(
                    str(placeholder), config,
                    source_url=yt_url,
                    video_id_override=video_id,
                    prefer_yt_captions=True,
                    allow_whisper=False,
                )

            with get_db(str(config.db_path)) as conn:
                row = conn.execute(
                    "SELECT duration_sec FROM videos WHERE video_id = ?", (video_id,)
                ).fetchone()

            assert row is not None
            assert row[0] == 123.0, (
                f"Expected duration_sec=123.0 (max caption end), got {row[0]}"
            )

    def test_duration_not_overridden_when_already_set(self):
        """If the video record already has a nonzero duration, the pipeline must
        leave it untouched even if captions span a different range."""
        from unittest.mock import patch

        from vidcrawl.process.pipeline import process_local_video

        caption_entries = [
            {"start_sec": 0.0, "end_sec": 60.0, "text": "Segment."},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            config = get_config(tmpdir)

            yt_url = "https://www.youtube.com/watch?v=durtest0002"
            video_id = make_video_id("youtube", yt_url)

            video_dir = config.videos_dir / video_id
            video_dir.mkdir(parents=True, exist_ok=True)
            placeholder = video_dir / f"{video_id}.mp4"
            placeholder.write_bytes(b"")

            with get_db(str(config.db_path)) as conn:
                init_db(conn)
                insert_video(conn, Video(
                    video_id=video_id,
                    title="Duration Not Overridden",
                    source="youtube",
                    url=yt_url,
                    duration_sec=300.0,
                    status="pending",
                ))

            with (
                patch("vidcrawl.process.pipeline.fetch_youtube_captions",
                      return_value=caption_entries),
                patch("vidcrawl.process.pipeline.extract_keyframes", return_value=[]),
                patch("vidcrawl.process.pipeline.ocr_frames", return_value=[]),
            ):
                process_local_video(
                    str(placeholder), config,
                    source_url=yt_url,
                    video_id_override=video_id,
                    prefer_yt_captions=True,
                    allow_whisper=False,
                )

            with get_db(str(config.db_path)) as conn:
                row = conn.execute(
                    "SELECT duration_sec FROM videos WHERE video_id = ?", (video_id,)
                ).fetchone()

            assert row[0] == 300.0, (
                f"duration_sec should remain 300.0, got {row[0]}"
            )


class TestMomentCountAccuracy:
    """Issue 3: Per-video OK line must reflect the actual number of written moments,
    even when m_before == m_after (prior moments replaced by an equal count)."""

    def test_moment_count_nonzero_when_prior_moments_replaced(self):
        """If a video with prior moments is re-processed (e.g. status=pending after
        a failed run), the OK line must show the rebuilt count, not a delta of 0."""
        from unittest.mock import patch

        caption_entries = [
            {"start_sec": float(i) * 35.0, "end_sec": float(i + 1) * 35.0,
             "text": f"Segment {i}."}
            for i in range(3)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            config = get_config(tmpdir)

            yt_url = "https://www.youtube.com/watch?v=momentcount01"
            video_id = make_video_id("youtube", yt_url)

            video_dir = config.videos_dir / video_id
            video_dir.mkdir(parents=True, exist_ok=True)
            placeholder = video_dir / f"{video_id}.mp4"
            placeholder.write_bytes(b"")

            # First pass: register + process to write 3 moments
            with get_db(str(config.db_path)) as conn:
                init_db(conn)
                insert_video(conn, Video(
                    video_id=video_id,
                    title="Moment Count Test",
                    source="youtube",
                    url=yt_url,
                    duration_sec=105.0,
                    status="pending",
                ))

            with (
                patch("vidcrawl.process.pipeline.fetch_youtube_captions",
                      return_value=caption_entries),
                patch("vidcrawl.process.pipeline.extract_keyframes", return_value=[]),
                patch("vidcrawl.process.pipeline.ocr_frames", return_value=[]),
            ):
                from vidcrawl.process.pipeline import process_local_video
                process_local_video(
                    str(placeholder), config,
                    source_url=yt_url, video_id_override=video_id,
                    prefer_yt_captions=True, allow_whisper=False,
                )

            # Manually reset status to "pending" to simulate a failed/partial run
            with get_db(str(config.db_path)) as conn:
                conn.execute(
                    "UPDATE videos SET status = 'pending' WHERE video_id = ?",
                    (video_id,),
                )

            # Second batch pass — status is "pending" so not skipped; force=False
            with (
                patch("vidcrawl.ingest.downloader.extract_youtube_metadata",
                      return_value={"duration": 105, "title": "Test"}),
                patch("vidcrawl.ingest.downloader.is_yt_dlp_available", return_value=True),
                patch("vidcrawl.process.pipeline.fetch_youtube_captions",
                      return_value=caption_entries),
                patch("vidcrawl.process.pipeline.extract_keyframes", return_value=[]),
                patch("vidcrawl.process.pipeline.ocr_frames", return_value=[]),
            ):
                source_file = _make_source_list([yt_url])
                result = runner.invoke(app, [
                    "batch", str(source_file),
                    "--data-dir", tmpdir,
                    "--limit", "1",
                    "--no-download",
                    "--prefer-yt-captions",
                    "--rate-limit-sec", "0",
                ])
                source_file.unlink(missing_ok=True)

            assert result.exit_code == 0, result.output

            ok_lines = [ln for ln in result.output.splitlines() if "OK:" in ln]
            assert ok_lines, f"No OK line found:\n{result.output}"
            m = re.search(r"OK:\s*(\d+)\s*moments", ok_lines[0])
            assert m is not None, f"Could not parse moment count from: {ok_lines[0]}"
            assert int(m.group(1)) > 0, (
                f"Expected > 0 moments in OK line, got: {ok_lines[0]}"
            )


class TestSkipOnNoTranscript:
    """Issue 4: In caption-only batch mode, videos with no captions and Whisper
    disabled must be skipped (not written as fallback chunks)."""

    def test_no_captions_no_whisper_skips_video(self):
        """When prefer_yt_captions=True, allow_whisper=False, and no captions are
        available, the video must be skipped rather than writing empty moments."""
        from unittest.mock import patch

        yt_url = "https://www.youtube.com/watch?v=nocaptiontest"
        video_id = make_video_id("youtube", yt_url)

        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            config = get_config(tmpdir)

            video_dir = config.videos_dir / video_id
            video_dir.mkdir(parents=True, exist_ok=True)
            placeholder = video_dir / f"{video_id}.mp4"
            placeholder.write_bytes(b"")

            with (
                patch("vidcrawl.ingest.downloader.extract_youtube_metadata",
                      return_value={"duration": 120, "title": "No Captions"}),
                patch("vidcrawl.ingest.downloader.is_yt_dlp_available", return_value=True),
                patch("vidcrawl.process.pipeline.fetch_youtube_captions",
                      return_value=None),  # no captions available
                patch("vidcrawl.process.pipeline.extract_keyframes", return_value=[]),
                patch("vidcrawl.process.pipeline.ocr_frames", return_value=[]),
            ):
                source_file = _make_source_list([yt_url])
                result = runner.invoke(app, [
                    "batch", str(source_file),
                    "--data-dir", tmpdir,
                    "--limit", "1",
                    "--no-download",
                    "--prefer-yt-captions",
                    "--rate-limit-sec", "0",
                ])
                source_file.unlink(missing_ok=True)

            assert result.exit_code == 0, result.output
            assert "no transcript" in result.output.lower(), (
                f"Expected 'no transcript' in output:\n{result.output}"
            )

            # No moments should have been written
            with get_db(str(config.db_path)) as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM moments WHERE video_id = ?", (video_id,)
                ).fetchone()[0]

            assert count == 0, (
                f"Expected 0 moments for skipped video, got {count}"
            )

    def test_no_captions_with_whisper_allowed_does_not_skip(self):
        """When allow_whisper=True, missing captions must NOT skip the video
        (it falls through to Whisper / fallback chunks)."""
        from unittest.mock import patch

        yt_url = "https://www.youtube.com/watch?v=whisperallowd1"
        video_id = make_video_id("youtube", yt_url)

        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(app, ["init", "--data-dir", tmpdir])
            config = get_config(tmpdir)

            video_dir = config.videos_dir / video_id
            video_dir.mkdir(parents=True, exist_ok=True)
            placeholder = video_dir / f"{video_id}.mp4"
            placeholder.write_bytes(b"")

            with get_db(str(config.db_path)) as conn:
                init_db(conn)
                insert_video(conn, Video(
                    video_id=video_id,
                    title="Whisper Allowed",
                    source="youtube",
                    url=yt_url,
                    duration_sec=60.0,
                    status="pending",
                ))

            with (
                patch("vidcrawl.process.pipeline.fetch_youtube_captions",
                      return_value=None),  # no captions
                # Whisper not installed in test env — transcribe_audio returns []
                patch("vidcrawl.process.pipeline.transcribe_audio", return_value=[]),
                patch("vidcrawl.process.pipeline.extract_keyframes", return_value=[]),
                patch("vidcrawl.process.pipeline.ocr_frames", return_value=[]),
            ):
                from vidcrawl.process.pipeline import process_local_video
                # raise_on_no_transcript defaults to False → should not skip
                process_local_video(
                    str(placeholder), config,
                    source_url=yt_url, video_id_override=video_id,
                    prefer_yt_captions=True,
                    allow_whisper=True,
                )

            with get_db(str(config.db_path)) as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM moments WHERE video_id = ?", (video_id,)
                ).fetchone()[0]

            # fallback chunks should be written (1 chunk for 60s video)
            assert count >= 1, (
                f"Expected at least 1 fallback moment when Whisper allowed, got {count}"
            )
