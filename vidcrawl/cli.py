import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer

from vidcrawl import __version__
from vidcrawl.config import get_config
from vidcrawl.db import (
    complete_ingestion_run,
    generate_run_id,
    get_db,
    get_evidence_by_moment,
    get_evidence_count_by_video,
    get_idea_count_by_video,
    get_keyframe_count_by_video,
    get_moment,
    get_moment_count_by_video,
    get_moments_by_video,
    get_video,
    init_db,
    insert_ingestion_run,
    insert_video,
    list_videos,
    make_video_id,
    rebuild_fts,
    update_video_status,
)
from vidcrawl.ingest.media import validate_video_file
from vidcrawl.models import IngestionRun, Video

app = typer.Typer(
    help="VidCrawl — local-first video intelligence and search",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"VidCrawl v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V",
        help="Show version",
        callback=_version_callback,
    ),
) -> None:
    pass


def _ensure_db(config) -> None:
    if not config.db_path.exists():
        typer.echo(
            "No database found. Run 'vidcrawl init' first.", err=True
        )
        raise typer.Exit(1)


def _human_size(bytes_val: int) -> str:
    if bytes_val < 1024:
        return f"{bytes_val} B"
    if bytes_val < 1024 ** 2:
        return f"{bytes_val / 1024:.1f} KB"
    return f"{bytes_val / 1024 ** 2:.1f} MB"


# ----------------------------------------------------------------
# init
# ----------------------------------------------------------------

@app.command()
def init(
    data_dir: str = typer.Option(
        "data", "--data-dir", "-d",
        help="Data directory path",
    ),
) -> None:
    config = get_config(data_dir)
    config.ensure_dirs()
    with get_db(config.db_path) as conn:
        init_db(conn)
    typer.echo(f"Initialized VidCrawl database at {config.db_path}")
    typer.echo(f"Data directory: {config.data_dir}")


# ----------------------------------------------------------------
# ingest
# ----------------------------------------------------------------

@app.command()
def ingest(
    source: str = typer.Argument(
        ..., help="YouTube URL or local file path"
    ),
    data_dir: str = typer.Option(
        "data", "--data-dir", "-d",
        help="Data directory path",
    ),
    process: bool = typer.Option(
        True, "--process/--no-process",
        help="Run full processing pipeline",
    ),
    download: bool = typer.Option(
        True, "--download/--no-download",
        help="Download YouTube video (requires yt-dlp)",
    ),
) -> None:
    config = get_config(data_dir)
    config.ensure_dirs()

    is_url = str(source).startswith(
        ("http://", "https://", "youtube.com", "youtu.be")
    )

    if is_url:
        _ingest_youtube(source, config, process, download)
        return

    path = Path(source)
    if not path.exists() or not path.is_file():
        typer.echo(f"Error: file not found: {source}", err=True)
        raise typer.Exit(1)
    try:
        validate_video_file(str(path))
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    if process:
        typer.echo(f"Processing local video: {path.resolve()}")
        try:
            from vidcrawl.process.pipeline import process_local_video
            video_id = process_local_video(str(path), config)
            typer.echo(f"Processed video: {video_id}")
            count = _get_moment_count(config, video_id)
            typer.echo(f"Created {count} moments")
        except Exception as e:
            typer.echo(f"Error processing video: {e}", err=True)
            raise typer.Exit(1)
    else:
        _register_local_video(source, config)


def _ingest_youtube(
    source: str, config, process: bool, download: bool
) -> None:
    from vidcrawl.ingest.downloader import (
        download_youtube,
        extract_youtube_metadata,
        is_yt_dlp_available,
        yt_dlp_install_help,
    )

    video_id = make_video_id("youtube", source)

    existing = _get_video_by_id(config, video_id)
    if existing:
        typer.echo(f"Video '{video_id}' already exists. Skipping registration.")
        if process and download:
            _try_youtube_download(source, video_id, config, existing)
        return

    meta = {}
    if is_yt_dlp_available():
        meta = extract_youtube_metadata(source)

    title = meta.get("title", f"YouTube video {video_id}")
    duration = float(meta.get("duration", 0))
    description = meta.get("description", "")
    uploader = meta.get("uploader", "")

    video_metadata = {}
    if description:
        video_metadata["description"] = description
    if uploader:
        video_metadata["uploader"] = uploader

    video = Video(
        video_id=video_id,
        title=title,
        source="youtube",
        url=source,
        duration_sec=duration,
        status="pending",
        metadata=video_metadata,
    )

    run_id = generate_run_id()
    run = IngestionRun(
        run_id=run_id,
        video_id=video_id,
        status="running",
        pipeline_steps=["register_metadata"],
    )

    with get_db(config.db_path) as conn:
        init_db(conn)
        insert_video(conn, video)
        insert_ingestion_run(conn, run)

    typer.echo(f"Registered YouTube video: {title}")
    typer.echo(f"  Video ID:    {video_id}")
    if duration > 0:
        typer.echo(f"  Duration:    {duration:.0f}s")

    if process:
        if download:
            _try_youtube_download(source, video_id, config, video)
        else:
            typer.echo("  Skipping download (--no-download)")
    else:
        typer.echo("  Status: pending (use --process to run pipeline)")


def _try_youtube_download(
    url: str, video_id: str, config, video,
) -> None:
    from vidcrawl.ingest.downloader import (
        download_youtube,
        is_yt_dlp_available,
        yt_dlp_install_help,
    )

    if not is_yt_dlp_available():
        typer.echo(yt_dlp_install_help())
        return

    typer.echo(f"Downloading YouTube video: {video_id}")
    download_dir = config.videos_dir / video_id
    downloaded = download_youtube(url, str(download_dir), video_id)

    if downloaded is None:
        typer.echo(
            "  Download failed. Video registered as metadata only.",
            err=True,
        )
        return

    typer.echo(f"  Downloaded to: {downloaded}")

    from vidcrawl.process.pipeline import process_local_video
    try:
        new_id = process_local_video(downloaded, config)
        count = _get_moment_count(config, new_id)
        typer.echo(f"Processed video: {new_id}")
        typer.echo(f"Created {count} moments")
    except Exception as e:
        typer.echo(f"Error processing video: {e}", err=True)
        raise typer.Exit(1)


def _get_video_by_id(config, video_id: str):
    conn = get_db(config.db_path)
    try:
        return get_video(conn, video_id)
    finally:
        conn.close()


def _register_local_video(source: str, config) -> None:
    path = Path(source).resolve()
    video_id = make_video_id("local", str(path))
    title = path.stem

    typer.echo(f"Registering local video: {path}")

    video = Video(
        video_id=video_id,
        title=title,
        source="local",
        url=None,
        duration_sec=0.0,
        status="pending",
    )

    run_id = generate_run_id()
    run = IngestionRun(
        run_id=run_id,
        video_id=video_id,
        status="running",
        pipeline_steps=["register_metadata"],
    )

    with get_db(config.db_path) as conn:
        init_db(conn)
        insert_video(conn, video)
        insert_ingestion_run(conn, run)

    typer.echo(f"Registered video: {video_id}")
    typer.echo(f"Ingestion run: {run_id}")
    typer.echo("Status: pending (use --process to run pipeline)")


# ----------------------------------------------------------------
# list
# ----------------------------------------------------------------

@app.command(name="list")
def list_videos_cmd(
    data_dir: str = typer.Option(
        "data", "--data-dir", "-d",
        help="Data directory path",
    ),
) -> None:
    config = get_config(data_dir)
    _ensure_db(config)

    conn = get_db(config.db_path)
    try:
        videos = list_videos(conn)
    finally:
        conn.close()

    if not videos:
        typer.echo("No videos in database.")
        return

    header = f"{'video_id':<24} {'title':<40} {'source':<10} {'status':<12} moments"
    typer.echo(header)
    typer.echo("-" * len(header))

    conn = get_db(config.db_path)
    try:
        for v in videos:
            count = get_moment_count_by_video(conn, v.video_id)
            typer.echo(
                f"{v.video_id:<24} {v.title[:38]:<40} "
                f"{v.source:<10} {v.status:<12} {count}"
            )
    finally:
        conn.close()


# ----------------------------------------------------------------
# inspect
# ----------------------------------------------------------------

@app.command()
def inspect(
    video_id: str = typer.Argument(
        ..., help="Video ID to inspect"
    ),
    data_dir: str = typer.Option(
        "data", "--data-dir", "-d",
        help="Data directory path",
    ),
) -> None:
    config = get_config(data_dir)
    _ensure_db(config)

    conn = get_db(config.db_path)
    try:
        video = get_video(conn, video_id)
        if video is None:
            typer.echo(f"Error: Video '{video_id}' not found.", err=True)
            raise typer.Exit(1)

        typer.echo(f"Title:       {video.title}")
        typer.echo(f"Video ID:    {video.video_id}")
        typer.echo(f"Source:      {video.source}")
        if video.url:
            typer.echo(f"URL:         {video.url}")
        typer.echo(f"Duration:    {video.duration_sec:.1f}s")
        typer.echo(f"Status:      {video.status}")
        if video.error_message:
            typer.echo(f"Error:       {video.error_message}")

        moment_count = get_moment_count_by_video(conn, video_id)
        evidence_count = get_evidence_count_by_video(conn, video_id)
        keyframe_count = get_keyframe_count_by_video(conn, video_id)
        idea_count = get_idea_count_by_video(conn, video_id)

        typer.echo(f"Moments:     {moment_count}")
        typer.echo(f"Evidence:    {evidence_count}")
        typer.echo(f"Keyframes:   {keyframe_count}")
        typer.echo(f"Ideas:       {idea_count}")

        if moment_count > 0:
            moments = get_moments_by_video(conn, video_id)
            typer.echo()
            typer.echo("First 3 Moments:")
            for m in moments[:3]:
                ts_range = f"{m.start_sec:.1f}s - {m.end_sec:.1f}s"
                transcript_preview = (
                    m.transcript_text[:60] + "..."
                    if len(m.transcript_text) > 60
                    else m.transcript_text
                )
                ocr_preview = (
                    m.ocr_text[:60] + "..."
                    if len(m.ocr_text) > 60
                    else m.ocr_text
                )
                idea_types = ", ".join(
                    sorted(set(i.type for i in m.ideas))
                ) if m.ideas else "none"
                kf_count = len(m.keyframe_paths)
                typer.echo(f"  [{ts_range}]")
                typer.echo(
                    f"    transcript: {transcript_preview or '(empty)'}"
                )
                typer.echo(
                    f"    ocr:        {ocr_preview or '(empty)'}"
                )
                typer.echo(f"    ideas:      {idea_types}")
                typer.echo(f"    keyframes:  {kf_count}")
            if moment_count > 3:
                typer.echo(
                    f"  ... and {moment_count - 3} more moments"
                )
    finally:
        conn.close()


# ----------------------------------------------------------------
# reindex
# ----------------------------------------------------------------

@app.command()
def reindex(
    data_dir: str = typer.Option(
        "data", "--data-dir", "-d",
        help="Data directory path",
    ),
) -> None:
    config = get_config(data_dir)
    _ensure_db(config)

    conn = get_db(config.db_path)
    try:
        init_db(conn)
        rebuild_fts(conn)
    finally:
        conn.close()

    typer.echo("FTS index rebuilt.")


# ----------------------------------------------------------------
# search
# ----------------------------------------------------------------

@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(
        10, "--limit", "-l", help="Max results"
    ),
    video_id: str = typer.Option(
        None, "--video-id", help="Filter by video ID"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output as JSON"
    ),
    show_snippets: bool = typer.Option(
        True, "--snippets/--no-snippets",
        help="Show transcript/OCR snippets",
    ),
    include_duplicates: bool = typer.Option(
        False, "--include-duplicates",
        help="Include exact duplicate moments in results",
    ),
    diverse: bool = typer.Option(
        False, "--diverse",
        help="Include variant moments for diversity",
    ),
    rerank: bool = typer.Option(
        True, "--rerank/--no-rerank",
        help="Use graph-aware reranking (default: on if graph exists)",
    ),
    explain_ranking: bool = typer.Option(
        False, "--explain-ranking",
        help="Show ranking reasons and score components",
    ),
    raw_ranking: bool = typer.Option(
        False, "--raw-ranking",
        help="Use FTS-only ranking (no rerank)",
    ),
    semantic: bool = typer.Option(
        False, "--semantic",
        help="Include semantic search candidates",
    ),
    semantic_only: bool = typer.Option(
        False, "--semantic-only",
        help="Use semantic search only",
    ),
    hybrid: bool = typer.Option(
        False, "--hybrid",
        help="Merge FTS + semantic candidates then rerank",
    ),
    graph_context: bool = typer.Option(
        False, "--graph-context",
        help="Show graph context (entities, ideas, cluster info)",
    ),
    data_dir: str = typer.Option(
        "data", "--data-dir", "-d",
        help="Data directory path",
    ),
) -> None:
    config = get_config(data_dir)
    _ensure_db(config)

    from vidcrawl.search.query import search_moments
    from vidcrawl.utils.time import youtube_timestamp_url

    use_rerank = rerank and not raw_ranking
    use_semantic = semantic or semantic_only or hybrid
    search_mode = "fts"
    if semantic_only:
        search_mode = "semantic"
    elif hybrid:
        search_mode = "hybrid"
    elif semantic:
        search_mode = "fts_semantic"
    results = search_moments(
        query, config.db_path, limit=limit, video_id=video_id,
        include_duplicates=include_duplicates, diverse=diverse,
        use_rerank=use_rerank, search_mode=search_mode,
    )

    if not results:
        typer.echo(
            "No results found. Try a different query or ingest more videos."
        )
        raise typer.Exit()

    collapsed = results[0].collapsed_count if results else 0

    if json_output:
        from vidcrawl.graph.query import get_graph_context_for_moment
        data = []
        for r in results:
            item = {
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
                "keyframe_paths": r.keyframe_paths,
                "match_reasons": r.match_reasons,
            }
            if r.ranking_reasons:
                item["ranking_reasons"] = r.ranking_reasons
            if r.graph_score is not None:
                item["graph_score"] = r.graph_score
            if r.final_score is not None:
                item["final_score"] = r.final_score
            if r.source_url:
                ctx = get_graph_context_for_moment(config.db_path, r.moment_id)
                item["graph_context"] = {
                    "entities": ctx.entities,
                    "idea_count": ctx.idea_count,
                    "evidence_modalities": ctx.evidence_modalities,
                    "cluster_info": ctx.cluster_info,
                    "duplicate_count": ctx.duplicate_count,
                    "variant_count": ctx.variant_count,
                }
            data.append(item)
        typer.echo(json.dumps(data, indent=2))
        return

    typer.echo(f"Query: {query}")
    typer.echo(f"Results: {len(results)}")
    if collapsed > 0:
        typer.echo(f"Collapsed {collapsed} duplicate moment(s).")
    typer.echo()

    for r in results:
        score_label = f"score {r.score}"
        if r.graph_score is not None and r.final_score is not None:
            score_label = f"score {r.final_score} (raw={r.score}, graph={r.graph_score})"
        elif r.graph_score is not None:
            score_label = f"score {r.graph_score} (raw={r.score})"
        typer.echo(
            f"{r.rank}. {r.video_title} — {r.timestamp_label} — "
            f"{score_label}"
        )
        typer.echo(f"   Moment: {r.moment_id}")
        if show_snippets and r.transcript_snippet:
            typer.echo(f"   Transcript: {r.transcript_snippet}")
        if show_snippets and r.ocr_snippet:
            typer.echo(f"   OCR: {r.ocr_snippet}")
        if r.idea_summary:
            types_str = ", ".join(r.idea_types)
            typer.echo(f"   Ideas: {types_str}")
        if r.match_reasons:
            typer.echo(f"   Match: {' + '.join(r.match_reasons)}")
        if explain_ranking and r.ranking_reasons:
            typer.echo(f"   Ranking: {'; '.join(r.ranking_reasons)}")
        if r.is_duplicate:
            typer.echo(f"   (duplicate of {r.canonical_moment_id})")
        elif r.is_variant:
            typer.echo(f"   (variant of {r.canonical_moment_id})")
        if r.keyframe_paths:
            kf = r.keyframe_paths[0]
            typer.echo(f"   Keyframes: {kf}")
            if len(r.keyframe_paths) > 1:
                typer.echo(f"     ... +{len(r.keyframe_paths) - 1} more")
        yt_url = (
            youtube_timestamp_url(r.source_url, r.start_sec)
            if r.source_url else None
        )
        if yt_url:
            typer.echo(f"   Link: {yt_url}")

        if graph_context:
            from vidcrawl.graph.query import get_graph_context_for_moment
            ctx = get_graph_context_for_moment(config.db_path, r.moment_id)
            parts = []
            if ctx.entities:
                parts.append(f"Entities: {', '.join(ctx.entities[:5])}")
            parts.append(f"Ideas: {ctx.idea_count}")
            if ctx.evidence_modalities:
                parts.append(f"Evidence: {' + '.join(ctx.evidence_modalities)}")
            if ctx.cluster_info:
                parts.append(f"Cluster: {ctx.cluster_info}")
            if ctx.related_moment_count:
                parts.append(f"Related: {ctx.related_moment_count} connected moments")
            if parts:
                typer.echo(f"   Graph: {'; '.join(parts)}")

        typer.echo()


# ----------------------------------------------------------------
# show
# ----------------------------------------------------------------

@app.command()
def show(
    moment_id: str = typer.Argument(
        ..., help="Moment ID to display"
    ),
    data_dir: str = typer.Option(
        "data", "--data-dir", "-d",
        help="Data directory path",
    ),
) -> None:
    config = get_config(data_dir)
    _ensure_db(config)

    conn = get_db(config.db_path)
    try:
        moment = get_moment(conn, moment_id)
        if moment is None:
            typer.echo(
                f"Error: Moment '{moment_id}' not found.", err=True
            )
            raise typer.Exit(1)

        video = get_video(conn, moment.video_id)
        title = video.title if video else "Unknown"

        typer.echo(f"Moment ID:  {moment.moment_id}")
        typer.echo(f"Video ID:   {moment.video_id}")
        typer.echo(f"Title:      {title}")
        if video and video.url:
            from vidcrawl.utils.time import youtube_timestamp_url
            yt_url = youtube_timestamp_url(
                video.url, moment.start_sec
            )
            typer.echo(f"Source URL: {video.url}")
            typer.echo(f"Timestamp:  {yt_url}")

        from vidcrawl.utils.time import timestamp_range
        typer.echo(
            f"Time:       {timestamp_range(moment.start_sec, moment.end_sec)}"
        )

        typer.echo()
        typer.echo("Transcript:")
        typer.echo(f"  {moment.transcript_text or '(empty)'}")

        typer.echo()
        typer.echo("OCR Text:")
        typer.echo(f"  {moment.ocr_text or '(empty)'}")

        if moment.ideas:
            typer.echo()
            typer.echo(f"Ideas ({len(moment.ideas)}):")
            for idea in moment.ideas:
                typer.echo(
                    f"  [{idea.type}] {idea.text} "
                    f"(confidence: {idea.confidence})"
                )

        evidence = get_evidence_by_moment(conn, moment_id)
        if evidence:
            typer.echo()
            typer.echo(f"Evidence Records ({len(evidence)}):")
            for ev in evidence:
                content_preview = (
                    ev["content"][:120] + "..."
                    if len(ev["content"]) > 120
                    else ev["content"]
                )
                typer.echo(f"  [{ev['modality']}] {content_preview}")

        if moment.keyframe_paths:
            typer.echo()
            typer.echo(
                f"Keyframes ({len(moment.keyframe_paths)}):"
            )
            for kf in moment.keyframe_paths:
                typer.echo(f"  {kf}")

        if moment.metadata:
            typer.echo()
            typer.echo("Metadata:")
            typer.echo(
                f"  {json.dumps(moment.metadata, indent=2)}"
            )

        typer.echo()
        typer.echo(
            f"Content Hash: {moment.content_hash or 'N/A'}"
        )

    finally:
        conn.close()


# ----------------------------------------------------------------
# stats
# ----------------------------------------------------------------

@app.command()
def stats(
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show detailed statistics",
    ),
    data_dir: str = typer.Option(
        "data", "--data-dir", "-d",
        help="Data directory path",
    ),
) -> None:
    config = get_config(data_dir)
    _ensure_db(config)

    conn = get_db(config.db_path)
    try:
        video_count = conn.execute(
            "SELECT COUNT(*) as c FROM videos"
        ).fetchone()["c"]
        moment_count = conn.execute(
            "SELECT COUNT(*) as c FROM moments"
        ).fetchone()["c"]
        evidence_count = conn.execute(
            "SELECT COUNT(*) as c FROM modal_evidence"
        ).fetchone()["c"]
        idea_count = conn.execute(
            "SELECT COUNT(*) as c FROM ideas"
        ).fetchone()["c"]
        keyframe_count = conn.execute(
            "SELECT COUNT(*) as c FROM keyframes"
        ).fetchone()["c"]
        duplicate_count = conn.execute(
            "SELECT COUNT(*) as c FROM duplicates"
        ).fetchone()["c"]

        fts_count = 0
        try:
            fts_count = conn.execute(
                "SELECT COUNT(*) as c FROM moments_fts"
            ).fetchone()["c"]
        except Exception:
            fts_count = 0

        typer.echo(f"Videos:     {video_count}")
        typer.echo(f"Moments:    {moment_count}")
        typer.echo(f"Evidence:   {evidence_count}")
        typer.echo(f"Ideas:      {idea_count}")
        typer.echo(f"Keyframes:  {keyframe_count}")
        typer.echo(f"Duplicates: {duplicate_count}")
        typer.echo(f"FTS Rows:   {fts_count}")

        if verbose and moment_count > 0:
            avg_moments = conn.execute(
                """SELECT AVG(cnt) FROM (
                    SELECT COUNT(*) as cnt FROM moments GROUP BY video_id
                )"""
            ).fetchone()[0]
            avg_chars = conn.execute(
                "SELECT AVG(LENGTH(transcript_text)) FROM moments"
            ).fetchone()[0]
            total_chars = conn.execute(
                "SELECT SUM(LENGTH(transcript_text)) FROM moments"
            ).fetchone()[0]

            typer.echo(
                f"Avg moments/video: {round(avg_moments, 1) if avg_moments else 0}"
            )
            typer.echo(
                f"Avg chars/moment:  {round(avg_chars, 1) if avg_chars else 0}"
            )
            typer.echo(
                f"Total chars:       {int(total_chars) if total_chars else 0}"
            )

        if verbose:
            db_size = config.db_path.stat().st_size if config.db_path.exists() else 0
            typer.echo(f"Database size:    {_human_size(db_size)}")

            if config.videos_dir.exists():
                art_size = sum(
                    f.stat().st_size for f in config.videos_dir.rglob("*")
                    if f.is_file()
                )
                typer.echo(
                    f"Artifacts size:   {_human_size(art_size)}"
                )

        typer.echo(f"Database path:  {config.db_path}")
    finally:
        conn.close()


# ----------------------------------------------------------------
# demo
# ----------------------------------------------------------------

@app.command()
def demo(
    subcommand: str = typer.Argument(
        "init", help="Subcommand: init"
    ),
    data_dir: str = typer.Option(
        "data", "--data-dir", "-d",
        help="Data directory path",
    ),
) -> None:
    if subcommand != "init":
        typer.echo(
            f"Unknown demo subcommand: {subcommand}. Use 'init'.",
            err=True,
        )
        raise typer.Exit(1)

    config = get_config(data_dir)
    config.ensure_dirs()

    conn = get_db(config.db_path)
    init_db(conn)
    conn.close()

    from vidcrawl.demo import create_demo_corpus
    moments = create_demo_corpus(config.db_path, config.data_dir)

    typer.echo("Demo corpus created!")
    typer.echo(f"  Database: {config.db_path}")
    typer.echo(f"  Videos:   3 (demo_coding, demo_ml, demo_ux)")
    typer.echo(f"  Moments:  {len(moments)}")
    typer.echo("  (includes duplicate-like moments for dedupe testing)")
    typer.echo()
    typer.echo("Try these commands:")
    typer.echo("  vidcrawl stats")
    typer.echo('  vidcrawl search "playwright browser"')
    typer.echo('  vidcrawl search "warning"')
    typer.echo('  vidcrawl search "definition" --json')
    typer.echo('  vidcrawl search "comparison"')
    typer.echo(f'  vidcrawl show {moments[0]["moment_id"]}')
    typer.echo("  vidcrawl dedupe run")
    typer.echo("  vidcrawl dedupe stats")
    typer.echo("  vidcrawl eval")


# ----------------------------------------------------------------
# eval
# ----------------------------------------------------------------

@app.command(name="eval")
def run_eval(
    query_file: str = typer.Argument(
        None,
        help="Path to JSON query file (optional, uses built-in demo queries)",
    ),
    rerank: bool = typer.Option(
        True, "--rerank/--no-rerank",
        help="Use graph-aware reranking",
    ),
    all_modes: bool = typer.Option(
        False, "--all-modes",
        help="Evaluate all modes (raw, rerank, diverse)",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output as JSON",
    ),
    diverse: bool = typer.Option(
        False, "--diverse",
        help="Use diverse result selection",
    ),
    data_dir: str = typer.Option(
        "data", "--data-dir", "-d",
        help="Data directory path",
    ),
) -> None:
    config = get_config(data_dir)
    _ensure_db(config)

    from vidcrawl.eval import evaluate_queries, format_eval_report, load_queries

    queries = load_queries(query_file)
    if not queries:
        typer.echo(
            "No queries found. Provide a JSON query file or run 'vidcrawl demo init' first.",
            err=True,
        )
        raise typer.Exit(1)

    if all_modes:
        results = {}
        for mode_name, mode_rerank, mode_diverse in [
            ("raw", False, False),
            ("rerank", True, False),
            ("diverse", True, True),
        ]:
            m = evaluate_queries(
                config.db_path, queries,
                use_rerank=mode_rerank, diverse=mode_diverse,
            )
            results[mode_name] = m
        if json_output:
            typer.echo(json.dumps(results, indent=2))
            return
        for mode_name, m in results.items():
            typer.echo(f"\n--- Mode: {mode_name} ---")
            typer.echo(format_eval_report(m))
        return

    metrics = evaluate_queries(
        config.db_path, queries,
        use_rerank=rerank, diverse=diverse,
    )
    if json_output:
        typer.echo(json.dumps(metrics, indent=2))
        return
    report = format_eval_report(metrics)
    typer.echo(report)


# ----------------------------------------------------------------
# dedupe
# ----------------------------------------------------------------

@app.command()
def dedupe(
    action: str = typer.Argument(
        "run", help="Action: run, stats, show"
    ),
    moment_id: str = typer.Argument(
        None, help="Moment ID for 'show' action"
    ),
    threshold: float = typer.Option(
        0.75, "--threshold", "-t",
        help="Similarity threshold for near-duplicate detection",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be done without modifying DB",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output as JSON",
    ),
    data_dir: str = typer.Option(
        "data", "--data-dir", "-d",
        help="Data directory path",
    ),
) -> None:
    config = get_config(data_dir)
    _ensure_db(config)

    if action == "run":
        _dedupe_run(config, threshold, dry_run, json_output)
    elif action == "stats":
        _dedupe_stats(config, json_output)
    elif action == "show":
        if not moment_id:
            typer.echo("Error: moment_id required for 'show' action.", err=True)
            raise typer.Exit(1)
        _dedupe_show(config, moment_id)
    else:
        typer.echo(
            f"Unknown dedupe action: {action}. Use run, stats, or show.",
            err=True,
        )
        raise typer.Exit(1)


def _dedupe_run(config, threshold: float, dry_run: bool, json_output: bool) -> None:
    conn = get_db(config.db_path)
    try:
        from vidcrawl.dedupe.cluster import run_dedupe
        stats = run_dedupe(conn, threshold=threshold, dry_run=dry_run)
    finally:
        conn.close()

    if json_output:
        typer.echo(json.dumps(stats, indent=2))
        return

    if dry_run:
        typer.echo("Dry run — no changes made.")
    typer.echo(
        f"Found {stats['exact_duplicates']} exact, "
        f"{stats['near_duplicates']} near-text, "
        f"{stats['same_idea']} same-idea, "
        f"{stats['variants']} variant"
    )
    typer.echo(
        f"Total: {stats['total_before']} moments → "
        f"{stats['total_after']} unique + "
        f"{stats['exact_duplicates'] + stats['near_duplicates']} duplicates"
    )
    if dry_run:
        typer.echo("Run without --dry-run to persist.")


def _dedupe_stats(config, json_output: bool) -> None:
    conn = get_db(config.db_path)
    try:
        from vidcrawl.db import get_duplicate_stats
        d = get_duplicate_stats(conn)
        total_moments = conn.execute(
            "SELECT COUNT(*) as c FROM moments"
        ).fetchone()["c"]
    finally:
        conn.close()

    if json_output:
        typer.echo(json.dumps(d, indent=2))
        return

    unique_count = total_moments - d["total"]
    ratio = round(d["total"] / max(total_moments, 1) * 100, 1)

    typer.echo(f"Total duplicate records: {d['total']}")
    typer.echo(f"Total moments:          {total_moments}")
    typer.echo(f"Unique moments:         {unique_count}")
    typer.echo(f"Redundancy ratio:       {ratio}%")
    for dup_type, count in sorted(d["by_type"].items()):
        typer.echo(f"  {dup_type}: {count}")
    typer.echo(f"Estimated compression:  ~{ratio}% fewer rows if collapsed")


def _dedupe_show(config, moment_id: str) -> None:
    conn = get_db(config.db_path)
    try:
        from vidcrawl.db import get_duplicates_for_moment
        records = get_duplicates_for_moment(conn, moment_id)
    finally:
        conn.close()

    if not records:
        typer.echo(f"No dedupe records found for moment '{moment_id}'.")
        return

    typer.echo(f"Dedupe records for {moment_id}:")
    for r in records:
        if r["moment_id"] == moment_id:
            typer.echo(
                f"  -> duplicate of {r['canonical_moment_id']} "
                f"({r['duplicate_type']}, sim={r['similarity_score']}, "
                f"novelty={r['novelty_score']})"
            )
            if r["reason"]:
                typer.echo(f"     reason: {r['reason']}")
        else:
            typer.echo(
                f"  <- canonical for {r['moment_id']} "
                f"({r['duplicate_type']}, sim={r['similarity_score']}, "
                f"novelty={r['novelty_score']})"
            )
            if r["reason"]:
                typer.echo(f"     reason: {r['reason']}")


# ----------------------------------------------------------------
# graph
# ----------------------------------------------------------------


@app.command()
def graph(
    action: str = typer.Argument(
        "build", help="Action: build, stats, show, neighbors, export"
    ),
    node_id: str = typer.Argument(
        None, help="Node or ref ID for show/neighbors"
    ),
    rebuild: bool = typer.Option(
        False, "--rebuild", help="Rebuild graph from scratch",
    ),
    fmt: str = typer.Option(
        "json", "--format", help="Export format (json)",
    ),
    out: str = typer.Option(
        None, "--out", help="Export output file path",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output as JSON",
    ),
    data_dir: str = typer.Option(
        "data", "--data-dir", "-d",
        help="Data directory path",
    ),
) -> None:
    config = get_config(data_dir)
    _ensure_db(config)

    if action == "build":
        _graph_build(config, rebuild)
    elif action == "stats":
        _graph_stats(config, json_output)
    elif action == "show":
        if not node_id:
            typer.echo("Error: node_id required for 'show' action.", err=True)
            raise typer.Exit(1)
        _graph_show(config, node_id, json_output)
    elif action == "neighbors":
        if not node_id:
            typer.echo("Error: node_id required for 'neighbors' action.", err=True)
            raise typer.Exit(1)
        _graph_neighbors(config, node_id, json_output)
    elif action == "export":
        _graph_export(config, fmt, out)
    else:
        typer.echo(
            f"Unknown graph action: {action}. "
            f"Use build, stats, show, neighbors, or export.",
            err=True,
        )
        raise typer.Exit(1)


def _graph_build(config, rebuild: bool) -> None:
    from vidcrawl.graph.build import build_graph
    summary = build_graph(config.db_path, rebuild=rebuild)
    typer.echo(f"Graph built: {summary.nodes_created} nodes, {summary.edges_created} edges")
    typer.echo(f"  Video nodes:     {summary.video_nodes}")
    typer.echo(f"  Moment nodes:    {summary.moment_nodes}")
    typer.echo(f"  Idea nodes:      {summary.idea_nodes}")
    typer.echo(f"  Evidence nodes:  {summary.evidence_nodes}")
    typer.echo(f"  Entity nodes:    {summary.entity_nodes}")
    typer.echo(f"  Cluster nodes:   {summary.cluster_nodes}")


def _graph_stats(config, json_output: bool) -> None:
    from vidcrawl.graph.stats import compute_graph_stats
    stats = compute_graph_stats(config.db_path)
    if json_output:
        typer.echo(json.dumps(stats.to_dict(), indent=2))
        return
    typer.echo(f"Total nodes:           {stats.total_nodes}")
    typer.echo(f"Total edges:           {stats.total_edges}")
    typer.echo(f"Average degree:        {stats.average_degree:.2f}")
    typer.echo(f"Duplicate clusters:    {stats.duplicate_clusters}")
    typer.echo(f"Connected components:  {stats.connected_components}")
    typer.echo()
    typer.echo("Nodes by type:")
    for ntype, count in sorted(stats.nodes_by_type.items()):
        typer.echo(f"  {ntype}: {count}")
    typer.echo()
    typer.echo("Edges by type:")
    for etype, count in sorted(stats.edges_by_type.items()):
        typer.echo(f"  {etype}: {count}")


def _graph_show(config, node_id: str, json_output: bool) -> None:
    from vidcrawl.graph.query import get_node
    node = get_node(config.db_path, node_id)
    if node is None:
        typer.echo(f"Node not found: {node_id}", err=True)
        raise typer.Exit(1)
    if json_output:
        typer.echo(json.dumps({
            "node_id": node.node_id,
            "node_type": node.node_type,
            "ref_id": node.ref_id,
            "label": node.label,
            "metadata": node.metadata,
            "created_at": node.created_at,
        }, indent=2))
        return
    typer.echo(f"Node ID:   {node.node_id}")
    typer.echo(f"Type:      {node.node_type}")
    typer.echo(f"Ref ID:    {node.ref_id}")
    typer.echo(f"Label:     {node.label}")
    typer.echo(f"Created:   {node.created_at}")
    if node.metadata:
        typer.echo("Metadata:")
        for k, v in node.metadata.items():
            val = str(v)[:120]
            typer.echo(f"  {k}: {val}")


def _graph_neighbors(config, node_id: str, json_output: bool) -> None:
    from vidcrawl.graph.query import get_neighbors
    result = get_neighbors(config.db_path, node_id)
    if not result:
        typer.echo(f"Node not found: {node_id}", err=True)
        raise typer.Exit(1)
    if json_output:
        typer.echo(json.dumps({
            "node": {
                "node_id": result["node"].node_id,
                "node_type": result["node"].node_type,
                "ref_id": result["node"].ref_id,
                "label": result["node"].label,
            },
            "edges": [
                {"edge_type": e["edge_type"], "source": e["source_node_id"], "target": e["target_node_id"]}
                for e in result["edges"]
            ],
            "neighbors": [
                {"node_id": n.node_id, "node_type": n.node_type, "label": n.label}
                for n in result["neighbors"]
            ],
            "degree": result["degree"],
        }, indent=2))
        return
    typer.echo(f"Node: {result['node'].node_id} ({result['node'].node_type})")
    typer.echo(f"Degree: {result['degree']}")
    typer.echo()
    if result["edges"]:
        typer.echo("Edges:")
        for e in result["edges"]:
            arrow = "→" if e["source_node_id"] == result["node"].node_id else "←"
            other = e["target_node_id"] if e["source_node_id"] == result["node"].node_id else e["source_node_id"]
            typer.echo(f"  {arrow} {other}  [{e['edge_type']}]")
    typer.echo()
    if result["neighbors"]:
        typer.echo("Neighbors:")
        for n in result["neighbors"]:
            typer.echo(f"  {n.node_id}  ({n.node_type})  {n.label[:60]}")


def _ensure_or_init_db(config) -> None:
    if not config.db_path.exists():
        typer.echo(f"Initializing new database at {config.db_path}")
        config.ensure_dirs()
        from vidcrawl.db import get_db, init_db
        with get_db(config.db_path) as conn:
            init_db(conn)


def _graph_export(config, fmt: str, out: Optional[str]) -> None:
    from vidcrawl.graph.export import export_graph
    output_path = Path(out) if out else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    result = export_graph(config.db_path, output_path=output_path, fmt=fmt)
    typer.echo(f"Exported {result['summary']['total_nodes']} nodes, {result['summary']['total_edges']} edges")
    if output_path:
        typer.echo(f"  to {output_path}")


# ----------------------------------------------------------------
# technical
# ----------------------------------------------------------------


@app.command()
def technical(
    action: str = typer.Argument(
        "extract", help="Action: extract, stats"
    ),
    data_dir: str = typer.Option(
        "data", "--data-dir", "-d",
        help="Data directory path",
    ),
) -> None:
    config = get_config(data_dir)
    _ensure_db(config)

    if action == "extract":
        from vidcrawl.technical.extract import run_technical_extraction
        result = run_technical_extraction(str(config.db_path))
        typer.echo(f"Extracted {result['total_evidence_inserted']} technical evidence records")
        typer.echo(f"  Moments with technical evidence: {result['moments_with_technical']}")
        for key in ["file_paths", "commands", "errors", "equations", "code_identifiers"]:
            if result.get(key, 0):
                typer.echo(f"  {key}: {result[key]}")
    elif action == "stats":
        from vidcrawl.technical.extract import get_technical_stats
        stats = get_technical_stats(str(config.db_path))
        typer.echo(f"Total technical evidence: {stats['total_technical_evidence']}")
        typer.echo(f"Moments with technical:  {stats['moments_with_technical']}/{stats['total_moments']}")
        if stats["by_modality"]:
            typer.echo("By modality:")
            for mod, cnt in sorted(stats["by_modality"].items()):
                typer.echo(f"  {mod}: {cnt}")
        else:
            typer.echo("No technical evidence found. Run 'vidcrawl technical extract' first.")
    else:
        typer.echo(f"Unknown technical action: {action}. Use extract or stats.", err=True)
        raise typer.Exit(1)


# ----------------------------------------------------------------
# claims
# ----------------------------------------------------------------


@app.command()
def claims(
    action: str = typer.Argument(
        "extract", help="Action: extract, stats, contradictions"
    ),
    data_dir: str = typer.Option(
        "data", "--data-dir", "-d",
        help="Data directory path",
    ),
) -> None:
    config = get_config(data_dir)
    _ensure_db(config)

    if action == "extract":
        from vidcrawl.claims.extract import run_claim_extraction
        result = run_claim_extraction(str(config.db_path))
        typer.echo(f"Extracted {result['total_claims']} claims")
        for ctype, cnt in sorted(result["by_type"].items()):
            typer.echo(f"  {ctype}: {cnt}")
    elif action == "stats":
        from vidcrawl.claims.extract import get_claim_stats
        stats = get_claim_stats(str(config.db_path))
        typer.echo(f"Total claims: {stats['total_claims']}")
        for ctype, cnt in sorted(stats["by_type"].items()):
            typer.echo(f"  {ctype}: {cnt}")
    elif action == "contradictions":
        from vidcrawl.claims.cluster import detect_contradictions
        contradictions = detect_contradictions(str(config.db_path))
        if contradictions:
            typer.echo(f"Found {len(contradictions)} potential contradictions:")
            for c in contradictions[:10]:
                typer.echo(f"  {c['reason']}: {c['text_a'][:60]} vs {c['text_b'][:60]}")
        else:
            typer.echo("No contradictions detected.")
    else:
        typer.echo(f"Unknown claims action: {action}. Use extract, stats, or contradictions.", err=True)
        raise typer.Exit(1)


# ----------------------------------------------------------------
# report
# ----------------------------------------------------------------


@app.command()
def report(
    action: str = typer.Argument(
        "generate", help="Action: generate"
    ),
    out: str = typer.Option(
        None, "--out", help="Output file path (default: stdout)",
    ),
    data_dir: str = typer.Option(
        "data", "--data-dir", "-d",
        help="Data directory path",
    ),
) -> None:
    config = get_config(data_dir)
    _ensure_db(config)

    if action == "generate":
        from vidcrawl.eval import evaluate_queries, load_queries, format_eval_report
        from vidcrawl.doctor import run_doctor
        from vidcrawl.graph.stats import compute_graph_stats
        from vidcrawl.freshness import get_freshness_stats
        from datetime import datetime

        lines = []
        lines.append("# VidCrawl Research Prototype Report")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("")

        doc = run_doctor(data_dir)
        lines.append("## System Status")
        lines.append(f"- Database: {doc['database']['path']}")
        lines.append(f"- Tables: {len(doc['counts'])}")
        for table, count in sorted(doc['counts'].items()):
            lines.append(f"  - {table}: {count}")
        lines.append(f"- FTS rows: {doc.get('fts_row_count', 0)}")
        lines.append(f"- Graph: {doc['graph']['built']} ({doc['graph']['node_count']} nodes)")
        lines.append(f"- Embeddings: {doc['embeddings']['built']} ({doc['embeddings']['count']} vectors)")
        lines.append("")

        if "graph_nodes" in doc["counts"] and doc["counts"]["graph_nodes"] > 0:
            stats = compute_graph_stats(config.db_path)
            lines.append("## Graph Statistics")
            lines.append(f"- Nodes: {stats.total_nodes}")
            lines.append(f"- Edges: {stats.total_edges}")
            lines.append(f"- Avg degree: {stats.average_degree:.2f}")
            lines.append(f"- Components: {stats.connected_components}")
            lines.append("")

        fresh = get_freshness_stats(str(config.db_path))
        if fresh["total_scored"] > 0:
            lines.append("## Freshness")
            lines.append(f"- Scored: {fresh['total_scored']}")
            lines.append(f"- Avg freshness: {fresh['average_freshness']}")
            lines.append(f"- Stale: {fresh['stale_count']}")
            lines.append(f"- Fresh: {fresh['fresh_count']}")
            lines.append("")

        queries = load_queries(None)
        if queries:
            for mode in ["raw", "rerank"]:
                metrics = evaluate_queries(config.db_path, queries, use_rerank=(mode == "rerank"))
                lines.append(f"## Evaluation ({mode})")
                lines.append(format_eval_report(metrics))
                lines.append("")

        lines.append("## Architecture")
        lines.append("VidCrawl is a local-first video intelligence system.")
        lines.append("- **Moments**: timestamped video segments with transcript, OCR, ideas")
        lines.append("- **Search**: SQLite FTS5 + graph-aware reranking + optional embeddings")
        lines.append("- **Graph**: multimodal idea graph connecting videos, moments, ideas, evidence, entities, claims")
        lines.append("- **Dedupe**: exact hash + near-text + semantic (optional)")
        lines.append("- **Claims**: rule-based extraction from transcript/OCR")
        lines.append("- **Freshness**: keyword-based staleness detection")
        lines.append("")

        text = "\n".join(lines)

        if out:
            Path(out).write_text(text)
            typer.echo(f"Report written to {out}")
        else:
            typer.echo(text)
    else:
        typer.echo(f"Unknown report action: {action}. Use generate.", err=True)
        raise typer.Exit(1)


# ----------------------------------------------------------------
# doctor / benchmark
# ----------------------------------------------------------------


@app.command()
def doctor(
    data_dir: str = typer.Option(
        "data", "--data-dir", "-d",
        help="Data directory path",
    ),
) -> None:
    from vidcrawl.doctor import run_doctor
    report = run_doctor(data_dir)
    typer.echo("VidCrawl System Report")
    typer.echo("=====================")
    typer.echo(f"Database: {report['database']['path']}")
    typer.echo(f"  Exists: {report['database']['exists']}")
    if report["counts"]:
        typer.echo()
        typer.echo("Table counts:")
        for table, count in sorted(report["counts"].items()):
            typer.echo(f"  {table}: {count}")
    if "fts_row_count" in report:
        typer.echo(f"FTS rows: {report['fts_row_count']}")
    typer.echo(f"Indexes:  {report.get('index_count', 0)}")
    typer.echo()
    g = report["graph"]
    typer.echo(f"Graph built: {g['built']} ({g['node_count']} nodes)")
    e = report["embeddings"]
    typer.echo(f"Embeddings:  {e['built']} ({e['count']} vectors)")
    typer.echo()
    typer.echo("Optional tools:")
    for tool, info in report["optional_tools"].items():
        status = "✓" if info["available"] else "✗"
        typer.echo(f"  {status} {tool}")


@app.command()
def benchmark(
    data_dir: str = typer.Option(
        "data", "--data-dir", "-d",
        help="Data directory path",
    ),
) -> None:
    from vidcrawl.doctor import run_benchmark
    results = run_benchmark(data_dir)
    if "error" in results:
        typer.echo(f"Error: {results['error']}", err=True)
        raise typer.Exit(1)
    typer.echo("Benchmark Results")
    typer.echo("================")
    typer.echo(f"Moments:              {results.get('moment_count', 0)}")
    typer.echo(f"Avg FTS query:        {results.get('avg_fts_query_ms', 0)}ms")
    typer.echo(f"Avg insert:           {results.get('avg_insert_us', 0)}µs")


# ----------------------------------------------------------------
# freshness
# ----------------------------------------------------------------


@app.command()
def freshness(
    action: str = typer.Argument(
        "run", help="Action: run, stats"
    ),
    data_dir: str = typer.Option(
        "data", "--data-dir", "-d",
        help="Data directory path",
    ),
) -> None:
    config = get_config(data_dir)
    _ensure_db(config)

    if action == "run":
        from vidcrawl.freshness import run_freshness_scoring
        result = run_freshness_scoring(str(config.db_path))
        typer.echo(f"Scored {result['scored']} moments")
        typer.echo(f"  Fresh:   {result['fresh']}")
        typer.echo(f"  Neutral: {result['neutral']}")
        typer.echo(f"  Stale:   {result['stale']}")
    elif action == "stats":
        from vidcrawl.freshness import get_freshness_stats
        stats = get_freshness_stats(str(config.db_path))
        if stats["total_scored"] > 0:
            typer.echo(f"Total scored:     {stats['total_scored']}")
            typer.echo(f"Avg freshness:    {stats['average_freshness']}")
            typer.echo(f"Stale results:    {stats['stale_count']}")
            typer.echo(f"Fresh results:    {stats['fresh_count']}")
        else:
            typer.echo("No freshness scores. Run 'vidcrawl freshness run' first.")
    else:
        typer.echo(f"Unknown freshness action: {action}. Use run or stats.", err=True)
        raise typer.Exit(1)


# ----------------------------------------------------------------
# embed
# ----------------------------------------------------------------


@app.command()
def embed(
    action: str = typer.Argument(
        "build", help="Action: build, stats"
    ),
    provider: str = typer.Option(
        "hash", "--provider", help="Embedding provider: hash, sentence-transformers",
    ),
    dimension: int = typer.Option(
        64, "--dimension", help="Vector dimension for hash provider",
    ),
    data_dir: str = typer.Option(
        "data", "--data-dir", "-d",
        help="Data directory path",
    ),
) -> None:
    config = get_config(data_dir)
    _ensure_or_init_db(config)

    if action == "build":
        from vidcrawl.embeddings.store import build_embeddings
        try:
            result = build_embeddings(
                str(config.db_path),
                provider_name=provider,
                dimension=dimension,
            )
            typer.echo(f"Built {result.get('vectors_stored', 0)} embeddings ({result.get('provider', '?')}, dim={result.get('dimension', '?')})")
        except ImportError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)
    elif action == "stats":
        from vidcrawl.embeddings.store import get_embedding_stats
        stats = get_embedding_stats(str(config.db_path))
        if stats["has_embeddings"]:
            typer.echo(f"Provider:   {stats['provider']}")
            typer.echo(f"Dimension:  {stats['dimension']}")
            typer.echo(f"Vectors:    {stats['vector_count']}")
            typer.echo(f"Status:     {stats['status']}")
        else:
            typer.echo("No embeddings found. Run 'vidcrawl embed build' first.")
    else:
        typer.echo(f"Unknown embed action: {action}. Use build or stats.", err=True)
        raise typer.Exit(1)


# ----------------------------------------------------------------
# helpers
# ----------------------------------------------------------------

def _get_moment_count(config, video_id: str) -> int:
    from vidcrawl.db import get_db, get_moment_count_by_video
    with get_db(config.db_path) as conn:
        return get_moment_count_by_video(conn, video_id)
