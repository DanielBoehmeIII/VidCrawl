from __future__ import annotations

import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional, Union

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from vidcrawl import __version__
from vidcrawl.api.jobs import (
    create_job,
    find_job_by_url,
    get_job,
    init_jobs_table,
    list_jobs,
    update_job,
    update_job_heartbeat,
    update_job_progress,
)
from vidcrawl.config import Config
from vidcrawl.db import (
    get_db,
    get_moment,
    get_moment_count_by_video,
    get_moments_by_video,
    get_video,
    init_db,
    list_videos,
    make_video_id,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    source: str
    process: bool = True
    download: bool = True
    prefer_yt_captions: bool = False
    allow_whisper: bool = False
    transcribe_model: str = "tiny"


class BatchRequest(BaseModel):
    sources: list[str]
    process: bool = True
    download: bool = True
    prefer_yt_captions: bool = False
    allow_whisper: bool = False
    transcribe_model: str = "tiny"


class ClipBounceSource(BaseModel):
    url: str
    title: str = ""
    source_type: str = "webpage"
    selected_text: Optional[str] = None
    page_text: Optional[str] = None
    page_html: Optional[str] = None


class CaptureRequest(BaseModel):
    # Single-URL form: {"url": "...", ...}
    url: Optional[str] = None
    title: str = ""
    source_type: str = "webpage"
    tab_id: Optional[int] = None
    selected_text: Optional[str] = None
    page_text: Optional[str] = None
    page_html: Optional[str] = None
    allow_whisper: bool = False
    # Batch form: {"sources": [...], "mode": "captions_first"}
    sources: Optional[list[ClipBounceSource]] = None
    mode: str = "default"


class BatchCaptureResponse(BaseModel):
    mode: str
    jobs: list[JobResponse]


class JobResponse(BaseModel):
    job_id: str
    status: str
    source_url: str
    video_id: Optional[str] = None
    created_at: str
    updated_at: str
    error_message: Optional[str] = None
    result: dict = {}
    stage: str = ""
    progress_message: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_ms: Optional[int] = None


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

_VIDEO_HOSTS: frozenset[str] = frozenset({
    "youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be",
    "vimeo.com", "www.vimeo.com", "player.vimeo.com",
})


def _is_youtube_url(url: str) -> bool:
    return any(
        url.startswith(prefix)
        for prefix in ("https://www.youtube.com", "https://youtu.be",
                       "http://www.youtube.com", "http://youtu.be",
                       "youtube.com", "youtu.be")
    )


def _is_known_video_url(url: str) -> bool:
    try:
        host = urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return False
    return host in _VIDEO_HOSTS


def _ingest_worker(
    config: Config,
    job_id: str,
    source: str,
    process: bool,
    download: bool,
    prefer_yt_captions: bool,
    allow_whisper: bool,
    transcribe_model: str,
    mode: str = "full",
) -> None:
    """Runs in a thread — ingests one source and updates the job record."""
    from vidcrawl.ingest.providers.registry import get_provider
    from vidcrawl.process.pipeline import (
        NeedsTranscriptionError,
        NoTranscriptAvailableError,
        ingest_any_source,
        process_local_video,
    )
    import threading
    import traceback

    db_path = config.db_path

    def _try_set_error(msg: str) -> None:
        try:
            with get_db(db_path) as ec:
                update_job(ec, job_id, "error", error_message=msg)
                update_job_progress(ec, job_id, "error", msg)
        except Exception:
            pass

    _heartbeat_stop = threading.Event()

    def _heartbeat_loop():
        while not _heartbeat_stop.is_set():
            _heartbeat_stop.wait(30)
            if _heartbeat_stop.is_set():
                break
            try:
                with get_db(db_path) as hb_conn:
                    update_job_heartbeat(hb_conn, job_id)
            except Exception:
                pass

    try:
        hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
        hb_thread.start()
    except Exception as exc:
        _try_set_error(f"Failed to start heartbeat thread: {exc}")
        return

    def _cleanup():
        _heartbeat_stop.set()
        hb_thread.join(timeout=3)

    try:
        if source.startswith(("http://", "https://")):
            provider = get_provider(source)
            url = provider.normalize(source)
            video_id = make_video_id(provider.source_type, url)

            with get_db(db_path) as conn:
                update_job(conn, job_id, "running", video_id=video_id)
                update_job_progress(conn, job_id, "metadata", "Starting ingestion")

            if process:
                ingest_any_source(
                    url, config,
                    allow_whisper=allow_whisper,
                    transcribe_model=transcribe_model,
                    job_id=job_id,
                    mode=mode,
                )

        else:
            path = Path(source)
            video_id = make_video_id("local", str(path.resolve()))

            with get_db(db_path) as conn:
                update_job(conn, job_id, "running", video_id=video_id)
                update_job_progress(conn, job_id, "metadata", "Starting ingestion")

            if process:
                process_local_video(
                    str(path), config,
                    transcribe_model=transcribe_model,
                    allow_whisper=allow_whisper,
                    job_id=job_id,
                    mode=mode,
                )

        _cleanup()
        with get_db(db_path) as conn:
            update_job(conn, job_id, "ready", video_id=video_id)

    except NeedsTranscriptionError as exc:
        _cleanup()
        with get_db(db_path) as conn:
            update_job(conn, job_id, "needs_transcription",
                       error_message=str(exc))
            update_job_progress(conn, job_id, "needs_transcription", str(exc))
    except NoTranscriptAvailableError as exc:
        _cleanup()
        with get_db(db_path) as conn:
            update_job(conn, job_id, "skipped_no_transcript",
                       error_message=str(exc))
            update_job_progress(conn, job_id, "skipped_no_transcript", str(exc))
    except Exception as exc:
        _cleanup()
        tb = traceback.format_exc()
        with get_db(db_path) as conn:
            update_job(conn, job_id, "error",
                       error_message=f"{exc}\n{tb}")
            update_job_progress(conn, job_id, "error", str(exc))


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(config: Config) -> FastAPI:
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vidcrawl-worker")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config.ensure_dirs()
        with get_db(config.db_path) as conn:
            init_db(conn)
            init_jobs_table(conn)
        yield
        executor.shutdown(wait=False)

    app = FastAPI(
        title="VidCrawl API",
        version=__version__,
        lifespan=lifespan,
    )

    # ----------------------------------------------------------------
    # GET /health
    # ----------------------------------------------------------------

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    # ----------------------------------------------------------------
    # GET /stats
    # ----------------------------------------------------------------

    @app.get("/stats")
    def stats():
        with get_db(config.db_path) as conn:
            video_count = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
            moment_count = conn.execute("SELECT COUNT(*) FROM moments").fetchone()[0]
            evidence_count = conn.execute("SELECT COUNT(*) FROM modal_evidence").fetchone()[0]
            idea_count = conn.execute("SELECT COUNT(*) FROM ideas").fetchone()[0]
            keyframe_count = conn.execute("SELECT COUNT(*) FROM keyframes").fetchone()[0]
            job_count = conn.execute("SELECT COUNT(*) FROM api_jobs").fetchone()[0]
            fts_count = 0
            try:
                fts_count = conn.execute("SELECT COUNT(*) FROM moments_fts").fetchone()[0]
            except Exception:
                pass
        return {
            "videos": video_count,
            "moments": moment_count,
            "evidence": evidence_count,
            "ideas": idea_count,
            "keyframes": keyframe_count,
            "jobs": job_count,
            "fts_rows": fts_count,
            "db_path": str(config.db_path),
        }

    def _submit(job_id: str, source: str, req_process: bool, req_download: bool,
                req_prefer_captions: bool, req_allow_whisper: bool, req_model: str,
                mode: str = "full") -> None:
        executor.submit(
            _ingest_worker,
            config, job_id, source,
            req_process, req_download, req_prefer_captions,
            req_allow_whisper, req_model, mode,
        )

    # ----------------------------------------------------------------
    # POST /ingest
    # ----------------------------------------------------------------

    @app.post("/ingest", response_model=JobResponse)
    def ingest(req: IngestRequest):
        from vidcrawl.ingest.downloader import normalize_youtube_url

        source = req.source
        if _is_youtube_url(source) or source.startswith(("http://", "https://")):
            source = normalize_youtube_url(source)

        with get_db(config.db_path) as conn:
            existing_job = find_job_by_url(conn, source)
            if existing_job and existing_job["status"] in ("queued", "running", "ready"):
                return JobResponse(**existing_job)
            job_id = create_job(conn, source)

        _submit(job_id, source, req.process, req.download,
                req.prefer_yt_captions, req.allow_whisper, req.transcribe_model)

        with get_db(config.db_path) as conn:
            job = get_job(conn, job_id)
        return JobResponse(**job)

    # ----------------------------------------------------------------
    # POST /batch
    # ----------------------------------------------------------------

    @app.post("/batch", response_model=list[JobResponse])
    def batch(req: BatchRequest):
        from vidcrawl.ingest.downloader import normalize_youtube_url

        responses: list[JobResponse] = []

        for raw_source in req.sources:
            source = raw_source.strip()
            if not source:
                continue
            if _is_youtube_url(source) or source.startswith(("http://", "https://")):
                source = normalize_youtube_url(source)

            with get_db(config.db_path) as conn:
                existing_job = find_job_by_url(conn, source)
                if existing_job and existing_job["status"] in ("queued", "running", "ready"):
                    responses.append(JobResponse(**existing_job))
                    continue
                job_id = create_job(conn, source)

            _submit(job_id, source, req.process, req.download,
                    req.prefer_yt_captions, req.allow_whisper, req.transcribe_model)

            with get_db(config.db_path) as conn:
                job = get_job(conn, job_id)
            responses.append(JobResponse(**job))

        return responses

    # ----------------------------------------------------------------
    # GET /videos
    # ----------------------------------------------------------------

    @app.get("/videos")
    def get_videos():
        with get_db(config.db_path) as conn:
            videos = list_videos(conn)
        return [v.model_dump() for v in videos]

    # ----------------------------------------------------------------
    # GET /videos/{video_id}
    # ----------------------------------------------------------------

    @app.get("/videos/{video_id}")
    def get_video_by_id(video_id: str):
        with get_db(config.db_path) as conn:
            video = get_video(conn, video_id)
        if video is None:
            raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")
        return video.model_dump()

    # ----------------------------------------------------------------
    # GET /search
    # ----------------------------------------------------------------

    @app.get("/search")
    def search(
        q: str = Query(..., description="Search query"),
        limit: int = Query(10, ge=1, le=100),
        video_id: Optional[str] = Query(None),
    ):
        from vidcrawl.search.query import search_moments

        results = search_moments(
            q, config.db_path, limit=limit, video_id=video_id,
            use_rerank=True,
        )
        return [
            {
                "rank": r.rank,
                "score": r.score,
                "moment_id": r.moment_id,
                "video_id": r.video_id,
                "video_title": r.video_title,
                "source_url": r.source_url,
                "start_sec": r.start_sec,
                "end_sec": r.end_sec,
                "timestamp_label": r.timestamp_label,
                "transcript_snippet": r.transcript_snippet,
                "ocr_snippet": r.ocr_snippet,
                "idea_summary": r.idea_summary,
                "idea_types": r.idea_types,
                "match_reasons": r.match_reasons,
            }
            for r in results
        ]

    # ----------------------------------------------------------------
    # GET /moments/{moment_id}
    # ----------------------------------------------------------------

    @app.get("/moments/{moment_id}")
    def get_moment_by_id(moment_id: str):
        with get_db(config.db_path) as conn:
            moment = get_moment(conn, moment_id)
        if moment is None:
            raise HTTPException(status_code=404, detail=f"Moment '{moment_id}' not found")
        return moment.model_dump()

    # ----------------------------------------------------------------
    # POST /clipbounce/capture
    # ----------------------------------------------------------------

    def _capture_one(url_raw: str, allow_whisper: bool = False, mode: str = "default") -> JobResponse:
        """Ingest a single URL and return a JobResponse (deduped)."""
        from vidcrawl.ingest.providers.registry import get_provider

        url = url_raw.strip()

        if not _is_known_video_url(url):
            with get_db(config.db_path) as conn:
                existing_job = find_job_by_url(conn, url)
                if existing_job and existing_job["status"] in (
                    "queued", "running", "ready", "skipped"
                ):
                    return JobResponse(**existing_job)
                job_id = create_job(conn, url)
                update_job(conn, job_id, "skipped",
                           result={"reason": "non-video URL not processed"})
                job = get_job(conn, job_id)
            return JobResponse(**job)

        provider = get_provider(url)
        url = provider.normalize(url)
        video_id = make_video_id(provider.source_type, url)

        with get_db(config.db_path) as conn:
            existing_video = get_video(conn, video_id)
            existing_job = find_job_by_url(conn, url)

        if existing_video and existing_video.status == "ready":
            if existing_job:
                return JobResponse(**existing_job)
            with get_db(config.db_path) as conn:
                job_id = create_job(conn, url, video_id=video_id)
                update_job(conn, job_id, "skipped", video_id=video_id)
                job = get_job(conn, job_id)
            return JobResponse(**job)

        if existing_job and existing_job["status"] in ("queued", "running", "ready"):
            return JobResponse(**existing_job)

        with get_db(config.db_path) as conn:
            job_id = create_job(conn, url, video_id=video_id)

        actual_mode = "fast" if mode == "fast" else "full"
        _submit(job_id, url, True, False, True, allow_whisper, "tiny", mode=actual_mode)

        with get_db(config.db_path) as conn:
            job = get_job(conn, job_id)
        return JobResponse(**job)

    @app.post("/clipbounce/capture")
    def clipbounce_capture(req: CaptureRequest) -> Union[JobResponse, BatchCaptureResponse]:
        if req.sources is not None:
            # Batch form: {"sources": [...], "mode": "fast"}
            jobs = [
                _capture_one(src.url, allow_whisper=req.allow_whisper, mode=req.mode)
                for src in req.sources
                if src.url.strip()
            ]
            return BatchCaptureResponse(mode=req.mode, jobs=jobs)

        # Single-URL form
        if not req.url:
            raise HTTPException(status_code=422, detail="Either 'url' or 'sources' must be provided")
        return _capture_one(req.url, allow_whisper=req.allow_whisper, mode=req.mode)

    # ----------------------------------------------------------------
    # GET /clipbounce/context
    # ----------------------------------------------------------------

    @app.get("/clipbounce/context")
    def clipbounce_context(
        url: str = Query(..., description="Source URL to look up"),
        q: Optional[str] = Query(None, description="Prompt to score moments against"),
    ):
        from vidcrawl.ingest.providers.registry import get_provider
        from vidcrawl.utils.time import timestamp_range
        from vidcrawl.api.jobs import mark_stuck_jobs

        # Normalize URL → video_id using the same path as /clipbounce/capture
        provider = get_provider(url)
        normalized_url = provider.normalize(url)
        video_id = make_video_id(provider.source_type, normalized_url)

        with get_db(config.db_path) as conn:
            mark_stuck_jobs(conn)
            video = get_video(conn, video_id)
            job = find_job_by_url(conn, normalized_url)
            moment_count = get_moment_count_by_video(conn, video_id)

        # 404 only when the URL is completely unknown (no video, no job)
        if video is None and job is None:
            raise HTTPException(status_code=404, detail=f"Video not found for URL: {url}")

        job_status = job["status"] if job else None
        job_stage = job.get("stage", "") if job else ""
        job_progress = job.get("progress_message", "") if job else ""

        # If moments already exist, content is usable even if pipeline hasn't
        # fully set video.status to "ready" (e.g. fast mode).
        if moment_count > 0:
            ctx_status = "ready"
        elif job_status == "needs_transcription":
            ctx_status = "needs_transcription"
        elif job_status == "skipped_no_transcript":
            ctx_status = "skipped_no_transcript"
        elif job_status == "skipped":
            ctx_status = "skipped"
        elif job_status in ("error",) or (video and video.status == "error"):
            video_err = video.error_message if video else ""
            if video_err and "transcri" in video_err.lower():
                ctx_status = "needs_transcription"
            else:
                ctx_status = "skipped"
        else:
            ctx_status = "indexing"

        effective_url = (video.url if video and video.url else url)
        title = video.title if video else normalized_url
        job_id = job["job_id"] if job else None

        base = {
            "video_id": video_id,
            "title": title,
            "url": effective_url,
            "status": ctx_status,
            "job_id": job_id,
            "stage": job_stage,
            "progress_message": job_progress,
        }

        if moment_count == 0:
            return {**base, "moments": [], "ideas": []}

        with get_db(config.db_path) as conn:
            all_moments = get_moments_by_video(conn, video_id)

        def _to_ctx(m, score: float = 0.0) -> dict:
            idea_parts = [f"[{i.type}] {i.text[:80]}" for i in m.ideas[:3]]
            return {
                "timestamp_label": timestamp_range(m.start_sec, m.end_sec),
                "transcript_snippet": (m.transcript_text or "")[:240],
                "idea_summary": "; ".join(idea_parts),
                "score": round(score, 4),
            }

        selected_pairs: list[tuple[dict, str]] = []
        selected_ids: set[str] = set()

        # Top 5 by relevance when q is given
        if q and q.strip():
            from vidcrawl.search.query import search_moments
            sr = search_moments(q, config.db_path, limit=5, video_id=video_id, use_rerank=True)
            for r in sr:
                if r.moment_id not in selected_ids:
                    selected_pairs.append(({
                        "timestamp_label": r.timestamp_label,
                        "transcript_snippet": r.transcript_snippet,
                        "idea_summary": r.idea_summary,
                        "score": round(r.score, 4),
                    }, r.moment_id))
                    selected_ids.add(r.moment_id)

        # 3 timeline anchors (evenly spaced across the moments list)
        n = len(all_moments)
        if n >= 3:
            anchor_indices = [n // 4, n // 2, 3 * n // 4]
        elif n == 2:
            anchor_indices = [0, 1]
        else:
            anchor_indices = [0]

        for idx in anchor_indices:
            m = all_moments[idx]
            if m.moment_id not in selected_ids:
                selected_pairs.append((_to_ctx(m, score=0.5), m.moment_id))
                selected_ids.add(m.moment_id)

        if not selected_pairs:
            for m in all_moments[:5]:
                if m.moment_id not in selected_ids:
                    selected_pairs.append((_to_ctx(m, score=0.5), m.moment_id))
                    selected_ids.add(m.moment_id)

        mid_to_moment = {m.moment_id: m for m in all_moments}
        ideas: list[str] = []
        seen_ideas: set[str] = set()
        for _, mid in selected_pairs:
            m = mid_to_moment.get(mid)
            if m:
                for idea in m.ideas:
                    text = f"[{idea.type}] {idea.text}"
                    if text not in seen_ideas:
                        seen_ideas.add(text)
                        ideas.append(text)

        return {
            **base,
            "status": "ready",
            "moments": [ctx for ctx, _ in selected_pairs],
            "ideas": ideas,
        }

    # ----------------------------------------------------------------
    # GET /clipbounce/jobs
    # ----------------------------------------------------------------

    @app.get("/clipbounce/jobs", response_model=JobResponse)
    def clipbounce_jobs(
        url: str = Query(..., description="Source URL to look up the job for"),
    ):
        from vidcrawl.ingest.providers.registry import get_provider

        provider = get_provider(url)
        normalized_url = provider.normalize(url)

        with get_db(config.db_path) as conn:
            job = find_job_by_url(conn, normalized_url)

        if job is None:
            raise HTTPException(status_code=404, detail=f"No job found for URL: {url}")

        return JobResponse(**job)

    # ----------------------------------------------------------------
    # GET /debug/clipbounce-flow
    # ----------------------------------------------------------------

    @app.get("/debug/clipbounce-flow")
    def debug_clipbounce_flow(
        url: str = Query(..., description="Source URL to diagnose"),
    ):
        from vidcrawl.ingest.providers.registry import get_provider
        from vidcrawl.process.chunking import chunk_transcript

        provider = get_provider(url)
        normalized_url = provider.normalize(url)
        video_id = make_video_id(provider.source_type, normalized_url)

        with get_db(config.db_path) as conn:
            video = get_video(conn, video_id)
            job = find_job_by_url(conn, normalized_url)
            moment_count = get_moment_count_by_video(conn, video_id)

        diagnostic = {
            "video_id": video_id,
            "url": normalized_url,
        }

        if video:
            diagnostic["video"] = {
                "status": video.status,
                "duration_sec": video.duration_sec,
                "error_message": video.error_message,
                "created_at": video.created_at,
            }
        else:
            diagnostic["video"] = None

        if job:
            diagnostic["job"] = {
                "job_id": job["job_id"],
                "status": job["status"],
                "stage": job.get("stage", ""),
                "progress_message": job.get("progress_message", ""),
                "last_heartbeat_at": job.get("last_heartbeat_at"),
                "started_at": job.get("started_at"),
                "finished_at": job.get("finished_at"),
                "duration_ms": job.get("duration_ms"),
                "error_message": job.get("error_message"),
                "created_at": job.get("created_at"),
                "updated_at": job.get("updated_at"),
            }
        else:
            diagnostic["job"] = None

        # Always attempt to check captions availability
        captions_found = 0
        transcript_count = 0
        chunks_count = 0
        try:
            caps = provider.fetch_captions(normalized_url, timeout_sec=15.0)
            if caps:
                captions_found = len(caps)
                chunks = chunk_transcript(caps)
                chunks_count = len(chunks) if chunks else 0
                transcript_count = len(caps)
        except Exception as exc:
            diagnostic["captions_error"] = str(exc)
        diagnostic["captions_found"] = captions_found
        diagnostic["transcript_segments"] = transcript_count
        diagnostic["chunks_possible"] = chunks_count

        diagnostic["moments_count"] = moment_count

        return diagnostic

    return app
