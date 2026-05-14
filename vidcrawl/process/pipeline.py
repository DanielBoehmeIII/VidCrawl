import json
from pathlib import Path
from typing import Optional

from vidcrawl.config import Config


class NoTranscriptAvailableError(Exception):
    """Raised in caption-only mode when no transcript source is available."""
from vidcrawl.db import (
    complete_ingestion_run,
    generate_evidence_id,
    generate_keyframe_id,
    generate_run_id,
    get_db,
    get_moment_count_by_video,
    get_moments_by_video,
    get_video,
    init_db,
    insert_evidence,
    insert_idea,
    insert_ingestion_run,
    insert_keyframe,
    insert_moment,
    insert_video,
    make_idea_id,
    make_moment_id,
    make_video_id,
    rebuild_fts,
    update_video_status,
)
from vidcrawl.ingest.downloader import accept_local
from vidcrawl.ingest.metadata import extract_duration, extract_file_metadata
from vidcrawl.ingest.transcript import (
    fetch_youtube_captions,
    load_sidecar_transcript,
    transcribe_audio,
)
from vidcrawl.models import Evidence, Idea, IngestionRun, Keyframe, Moment, Video
from vidcrawl.process.chunking import chunk_transcript
from vidcrawl.process.ideas import extract_ideas
from vidcrawl.process.keyframes import extract_keyframes
from vidcrawl.process.ocr import ocr_frames


def process_local_video(
    video_path: str,
    config: Config,
    run_id: Optional[str] = None,
    transcribe_model: str = "tiny",
    transcribe_device: str = "auto",
    transcribe_timeout_sec: Optional[float] = None,
    prefer_yt_captions: bool = False,
    allow_whisper: bool = True,
    caption_timeout_sec: float = 60.0,
    source_url: Optional[str] = None,
    video_id_override: Optional[str] = None,
    raise_on_no_transcript: bool = False,
) -> str:
    path = Path(video_path).resolve()
    video_id = video_id_override or make_video_id("local", str(path))
    title = path.stem

    with get_db(config.db_path) as conn:
        init_db(conn)

        existing = get_video(conn, video_id)
        if existing is None:

            md = extract_file_metadata(str(path))
            duration = extract_duration(str(path))

            video = Video(
                video_id=video_id,
                title=title,
                source="local",
                url=None,
                duration_sec=duration,
                status="ingesting",
                metadata=md,
            )
            insert_video(conn, video)

            if run_id is None:
                run_id = generate_run_id()
            run = IngestionRun(
                run_id=run_id,
                video_id=video_id,
                status="running",
                pipeline_steps=["register_metadata"],
            )
            insert_ingestion_run(conn, run)
        else:
            video = existing
            update_video_status(conn, video_id, "ingesting")
            md = extract_file_metadata(str(path))
            video.metadata = md
            _clear_video_pipeline_data(conn, video_id)

            if run_id is None:
                run_id = generate_run_id()
            run = IngestionRun(
                run_id=run_id,
                video_id=video_id,
                status="running",
                pipeline_steps=[],
            )
            insert_ingestion_run(conn, run)

        conn.commit()

        steps = ["register_metadata"]

        try:
            _run_pipeline(
                conn, config, video, path, run_id, steps,
                transcribe_model=transcribe_model,
                transcribe_device=transcribe_device,
                transcribe_timeout_sec=transcribe_timeout_sec,
                prefer_yt_captions=prefer_yt_captions,
                allow_whisper=allow_whisper,
                caption_timeout_sec=caption_timeout_sec,
                source_url=source_url,
                raise_on_no_transcript=raise_on_no_transcript,
            )
        except NoTranscriptAvailableError:
            update_video_status(conn, video_id, "pending")
            complete_ingestion_run(conn, run_id, "failed", "no transcript available")
            conn.commit()
            raise
        except Exception as exc:
            update_video_status(conn, video_id, "error", str(exc))
            complete_ingestion_run(conn, run_id, "failed", str(exc))
            conn.commit()
            raise

    return video_id


def _run_pipeline(
    conn,
    config: Config,
    video: Video,
    video_path: Path,
    run_id: str,
    steps: list[str],
    transcribe_model: str = "tiny",
    transcribe_device: str = "auto",
    transcribe_timeout_sec: Optional[float] = None,
    prefer_yt_captions: bool = False,
    allow_whisper: bool = True,
    caption_timeout_sec: float = 60.0,
    source_url: Optional[str] = None,
    raise_on_no_transcript: bool = False,
) -> None:
    video_id = video.video_id

    transcript_entries = load_sidecar_transcript(str(video_path))
    if transcript_entries is not None:
        steps.append("load_transcript")
        print(f"  [transcript] Loaded sidecar ({len(transcript_entries)} segments).", flush=True)
    else:
        yt_url = source_url or (video.url if video.source == "youtube" else None)
        if prefer_yt_captions and yt_url:
            print("  [captions] Fetching YouTube captions...", flush=True)
            transcript_entries = fetch_youtube_captions(yt_url, timeout_sec=caption_timeout_sec)
            if transcript_entries:
                steps.append("yt_captions")
                print(
                    f"  [captions] Got {len(transcript_entries)} segments.",
                    flush=True,
                )
            else:
                print("  [captions] No captions found.", flush=True)

        if not transcript_entries:
            if allow_whisper:
                transcript_entries = transcribe_audio(
                    str(video_path),
                    model_name=transcribe_model,
                    device=transcribe_device,
                    timeout_sec=transcribe_timeout_sec,
                )
                if transcript_entries:
                    steps.append("transcribe_asr")
            else:
                print("  [captions] Skipping Whisper (not enabled in this mode).", flush=True)

    if raise_on_no_transcript and not transcript_entries:
        raise NoTranscriptAvailableError("no transcript available")

    print("  [chunking] Chunking transcript...", flush=True)
    chunks = chunk_transcript(transcript_entries) if transcript_entries else []
    if chunks:
        steps.append("chunk_transcript")
        print(f"  [chunking] {len(chunks)} chunk(s).", flush=True)
    else:
        chunks = _create_fallback_chunks(video.duration_sec, video_id)
        print(f"  [chunking] Using {len(chunks)} fallback chunk(s).", flush=True)

    if video.duration_sec <= 0:
        inferred = 0.0
        if transcript_entries:
            inferred = max(
                (e.get("end_sec", 0.0) for e in transcript_entries), default=0.0
            )
        if inferred <= 0 and chunks:
            inferred = max((c["end_sec"] for c in chunks), default=0.0)
        if inferred > 0:
            video.duration_sec = inferred
            conn.execute(
                "UPDATE videos SET duration_sec = ? WHERE video_id = ?",
                (inferred, video_id),
            )

    kf_dir = config.frames_dir / video_id
    kf_dir.mkdir(parents=True, exist_ok=True)

    keyframes = extract_keyframes(
        str(video_path),
        str(kf_dir),
        interval_sec=30.0,
        video_duration=video.duration_sec if video.duration_sec > 0 else None,
    )
    if keyframes:
        steps.append("extract_keyframes")
        for kf in keyframes:
            kf_record = Keyframe(
                keyframe_id=generate_keyframe_id(),
                video_id=video_id,
                timestamp_sec=kf["timestamp_sec"],
                file_path=kf["path"],
            )
            insert_keyframe(conn, kf_record)

    ocr_results = ocr_frames(keyframes)
    if ocr_results:
        steps.append("ocr_frames")

    print(f"  [db] Writing {len(chunks)} moment(s)...", flush=True)
    moments = []
    for chunk in chunks:
        moment_id = make_moment_id(video_id, chunk["start_sec"], chunk["end_sec"])
        transcript_text = chunk.get("transcript_text", "")

        ocr_text, ocr_ideas = _get_ocr_for_moment(ocr_results, chunk)
        chunk_ideas = extract_ideas(transcript_text) if transcript_text else []
        chunk_ideas.extend(ocr_ideas)

        idea_models = []
        for idx, idea_dict in enumerate(chunk_ideas):
            idea = Idea(
                idea_id=make_idea_id(moment_id, idx),
                moment_id=moment_id,
                type=idea_dict["idea_type"],
                text=idea_dict["text"],
                confidence=idea_dict.get("confidence", 0.7),
                source=idea_dict.get("source", "rule"),
            )
            idea_models.append(idea)

        kf_paths = _get_keyframes_for_moment(keyframes, chunk)

        moment = Moment(
            moment_id=moment_id,
            video_id=video_id,
            start_sec=chunk["start_sec"],
            end_sec=chunk["end_sec"],
            transcript_text=transcript_text,
            ocr_text=ocr_text,
            ideas=idea_models,
            keyframe_paths=kf_paths,
        )
        insert_moment(conn, moment)
        moments.append(moment)

        for idea in idea_models:
            insert_idea(conn, idea)

    if moments:
        steps.append("insert_moments")

        for moment in moments:
            _insert_evidence_for_moment(conn, moment, chunk_map={
                m.moment_id: c for m, c in zip(moments, chunks)
            }, ocr_results=ocr_results, keyframes=keyframes)

    rebuild_fts(conn)
    steps.append("rebuild_fts")

    update_video_status(conn, video_id, "ready")
    complete_ingestion_run(conn, run_id, "completed")

    run = conn.execute(
        "SELECT * FROM ingestion_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if run:
        existing_steps = json.loads(run["pipeline_steps"]) if run["pipeline_steps"] else []
        conn.execute(
            "UPDATE ingestion_runs SET pipeline_steps = ? WHERE run_id = ?",
            (json.dumps(existing_steps + steps), run_id),
        )


def _clear_video_pipeline_data(conn, video_id: str) -> None:
    """Delete all pipeline outputs for a video so it can be cleanly re-processed."""
    conn.execute(
        "DELETE FROM duplicates WHERE moment_id IN "
        "(SELECT moment_id FROM moments WHERE video_id = ?)",
        (video_id,),
    )
    conn.execute(
        "DELETE FROM duplicates WHERE canonical_moment_id IN "
        "(SELECT moment_id FROM moments WHERE video_id = ?)",
        (video_id,),
    )
    conn.execute(
        "DELETE FROM ideas WHERE moment_id IN "
        "(SELECT moment_id FROM moments WHERE video_id = ?)",
        (video_id,),
    )
    conn.execute(
        "DELETE FROM modal_evidence WHERE moment_id IN "
        "(SELECT moment_id FROM moments WHERE video_id = ?)",
        (video_id,),
    )
    conn.execute(
        "DELETE FROM keyframes WHERE video_id = ?",
        (video_id,),
    )
    conn.execute(
        "DELETE FROM moments WHERE video_id = ?",
        (video_id,),
    )


def _create_fallback_chunks(
    duration_sec: float, video_id: str
) -> list[dict]:
    if duration_sec <= 0:
        duration_sec = 60.0
    chunk_size = 60.0
    chunks = []
    start = 0.0
    while start < duration_sec:
        end = min(start + chunk_size, duration_sec)
        chunks.append({"start_sec": start, "end_sec": end, "transcript_text": ""})
        start = end
    return chunks


def _get_ocr_for_moment(
    ocr_results: list[dict], chunk: dict
) -> tuple[str, list[dict]]:
    if not ocr_results:
        return "", []

    chunk_start = chunk["start_sec"]
    chunk_end = chunk["end_sec"]
    texts = []
    ideas = []
    seen_texts = set()

    for ocr in ocr_results:
        ts = ocr.get("timestamp_sec", 0)
        if chunk_start <= ts <= chunk_end:
            t = ocr.get("text", "").strip()
            if t and t not in seen_texts:
                seen_texts.add(t)
                texts.append(t)
                for idea in extract_ideas(t):
                    if idea.get("text") not in seen_texts:
                        ideas.append(idea)

    return " ".join(texts), ideas


def _get_keyframes_for_moment(
    keyframes: list[dict], chunk: dict
) -> list[str]:
    chunk_start = chunk["start_sec"]
    chunk_end = chunk["end_sec"]
    paths = []
    for kf in keyframes:
        ts = kf.get("timestamp_sec", 0)
        if chunk_start <= ts <= chunk_end:
            paths.append(kf.get("path", ""))
    return paths


def _insert_evidence_for_moment(
    conn,
    moment: Moment,
    chunk_map: dict[str, dict],
    ocr_results: list[dict],
    keyframes: list[dict],
) -> None:
    chunk = chunk_map.get(moment.moment_id, {})

    if moment.transcript_text:
        insert_evidence(
            conn,
            Evidence(
                evidence_id=generate_evidence_id(),
                moment_id=moment.moment_id,
                modality="transcript",
                content=moment.transcript_text,
                confidence=1.0,
                source="sidecar" if chunk.get("source") != "asr" else "whisper",
            ),
        )

    ocr_for_moment = [
        o for o in ocr_results
        if moment.start_sec <= o.get("timestamp_sec", 0) <= moment.end_sec
    ]
    for ocr in ocr_for_moment:
        _conf = ocr.get("confidence")
        insert_evidence(
            conn,
            Evidence(
                evidence_id=generate_evidence_id(),
                moment_id=moment.moment_id,
                modality="ocr",
                content=ocr.get("text", ""),
                confidence=_conf if _conf is not None else 1.0,
                source="tesseract",
                metadata={"timestamp_sec": ocr.get("timestamp_sec", 0)},
            ),
        )

    kf_for_moment = [
        kf for kf in keyframes
        if moment.start_sec <= kf.get("timestamp_sec", 0) <= moment.end_sec
    ]
    for kf in kf_for_moment:
        insert_evidence(
            conn,
            Evidence(
                evidence_id=generate_evidence_id(),
                moment_id=moment.moment_id,
                modality="keyframe",
                content=kf.get("path", ""),
                confidence=1.0,
                source="ffmpeg",
                metadata={"timestamp_sec": kf.get("timestamp_sec", 0)},
            ),
        )

    for idea in moment.ideas:
        insert_evidence(
            conn,
            Evidence(
                evidence_id=generate_evidence_id(),
                moment_id=moment.moment_id,
                modality="idea",
                content=idea.text,
                confidence=idea.confidence,
                source="rule",
                metadata={"idea_type": idea.type},
            ),
        )
