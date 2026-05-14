from __future__ import annotations

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
)
from vidcrawl.config import Config
from vidcrawl.db import (
    get_db,
    get_moment,
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


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _is_youtube_url(url: str) -> bool:
    return any(
        url.startswith(prefix)
        for prefix in ("https://www.youtube.com", "https://youtu.be",
                       "http://www.youtube.com", "http://youtu.be",
                       "youtube.com", "youtu.be")
    )


def _ingest_worker(
    config: Config,
    job_id: str,
    source: str,
    process: bool,
    download: bool,
    prefer_yt_captions: bool,
    allow_whisper: bool,
    transcribe_model: str,
) -> None:
    """Runs in a thread — ingests one source and updates the job record."""
    from vidcrawl.db import (
        generate_run_id,
        insert_ingestion_run,
        insert_video,
    )
    from vidcrawl.ingest.downloader import (
        extract_youtube_metadata,
        is_yt_dlp_available,
        normalize_youtube_url,
    )
    from vidcrawl.models import IngestionRun, Video

    db_path = config.db_path

    try:
        is_url = _is_youtube_url(source) or source.startswith(("http://", "https://"))

        if is_url:
            url = normalize_youtube_url(source)
            video_id = make_video_id("youtube", url)

            with get_db(db_path) as conn:
                update_job(conn, job_id, "running", video_id=video_id)

            meta: dict = {}
            if is_yt_dlp_available():
                meta = extract_youtube_metadata(url)

            title = meta.get("title", f"YouTube video {video_id}")
            duration = float(meta.get("duration", 0))
            video_meta: dict = {}
            if meta.get("description"):
                video_meta["description"] = meta["description"]
            if meta.get("uploader"):
                video_meta["uploader"] = meta["uploader"]

            video = Video(
                video_id=video_id,
                title=title,
                source="youtube",
                url=url,
                duration_sec=duration,
                status="pending",
                metadata=video_meta,
            )
            run = IngestionRun(
                run_id=generate_run_id(),
                video_id=video_id,
                status="running",
                pipeline_steps=["register_metadata"],
            )

            with get_db(db_path) as conn:
                existing = get_video(conn, video_id)
                if existing is None:
                    insert_video(conn, video)
                insert_ingestion_run(conn, run)

            if process:
                from vidcrawl.process.pipeline import process_local_video
                from vidcrawl.ingest.downloader import download_youtube

                if download and is_yt_dlp_available():
                    dl_dir = config.videos_dir / video_id
                    downloaded = download_youtube(url, str(dl_dir), video_id)
                    if downloaded:
                        process_local_video(
                            downloaded, config, source_url=url,
                            video_id_override=video_id,
                            prefer_yt_captions=prefer_yt_captions,
                            allow_whisper=allow_whisper,
                            transcribe_model=transcribe_model,
                        )
                else:
                    # Caption-only: zero-byte placeholder
                    ph_dir = config.videos_dir / video_id
                    ph_dir.mkdir(parents=True, exist_ok=True)
                    ph = ph_dir / f"{video_id}.mp4"
                    if not ph.exists():
                        ph.write_bytes(b"")
                    process_local_video(
                        str(ph), config, source_url=url,
                        video_id_override=video_id,
                        prefer_yt_captions=prefer_yt_captions,
                        allow_whisper=allow_whisper,
                        transcribe_model=transcribe_model,
                        raise_on_no_transcript=prefer_yt_captions and not allow_whisper,
                    )

        else:
            # Local file
            path = Path(source)
            video_id = make_video_id("local", str(path.resolve()))

            with get_db(db_path) as conn:
                update_job(conn, job_id, "running", video_id=video_id)

            if process:
                from vidcrawl.process.pipeline import process_local_video
                process_local_video(
                    str(path), config,
                    transcribe_model=transcribe_model,
                )
            else:
                video = Video(
                    video_id=video_id,
                    title=path.stem,
                    source="local",
                    url=None,
                    duration_sec=0.0,
                    status="pending",
                )
                run = IngestionRun(
                    run_id=generate_run_id(),
                    video_id=video_id,
                    status="running",
                    pipeline_steps=["register_metadata"],
                )
                with get_db(db_path) as conn:
                    existing = get_video(conn, video_id)
                    if existing is None:
                        insert_video(conn, video)
                    insert_ingestion_run(conn, run)

        with get_db(db_path) as conn:
            update_job(conn, job_id, "ready", video_id=video_id)

    except Exception as exc:
        with get_db(db_path) as conn:
            update_job(conn, job_id, "error", error_message=str(exc))


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
                req_prefer_captions: bool, req_allow_whisper: bool, req_model: str) -> None:
        executor.submit(
            _ingest_worker,
            config, job_id, source,
            req_process, req_download, req_prefer_captions,
            req_allow_whisper, req_model,
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

    def _capture_one(url_raw: str) -> JobResponse:
        """Ingest a single URL and return a JobResponse (deduped)."""
        from vidcrawl.ingest.downloader import normalize_youtube_url

        url = url_raw.strip()
        is_yt = _is_youtube_url(url)

        if is_yt:
            url = normalize_youtube_url(url)
            video_id = make_video_id("youtube", url)

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

            _submit(job_id, url, True, False, True, False, "tiny")

            with get_db(config.db_path) as conn:
                job = get_job(conn, job_id)
            return JobResponse(**job)

        else:
            with get_db(config.db_path) as conn:
                existing_job = find_job_by_url(conn, url)
                if existing_job and existing_job["status"] in ("queued", "running", "ready", "skipped"):
                    return JobResponse(**existing_job)
                job_id = create_job(conn, url)
                update_job(conn, job_id, "skipped",
                           result={"reason": "non-youtube URL not processed"})
                job = get_job(conn, job_id)
            return JobResponse(**job)

    @app.post("/clipbounce/capture")
    def clipbounce_capture(req: CaptureRequest) -> Union[JobResponse, BatchCaptureResponse]:
        if req.sources is not None:
            # Batch form: {"sources": [...], "mode": "captions_first"}
            jobs = [_capture_one(src.url) for src in req.sources if src.url.strip()]
            return BatchCaptureResponse(mode=req.mode, jobs=jobs)

        # Single-URL form
        if not req.url:
            raise HTTPException(status_code=422, detail="Either 'url' or 'sources' must be provided")
        return _capture_one(req.url)

    return app
