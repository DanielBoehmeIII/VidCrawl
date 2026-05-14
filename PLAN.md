# VidCrawl MVP Plan

## 1. Architecture (Simplest Possible)

```
┌─────────────┐   ┌──────────┐   ┌──────────────┐   ┌───────────┐
│ YouTube URL  │──▶│ yt-dlp   │──▶│ Audio stream  │──▶│ Whisper    │
│ Local file   │   │ ffmpeg   │   │ Video stream  │   │ (ASR)      │
└─────────────┘   └──────────┘   └──────────────┘   └─────┬─────┘
                                    │                      │
                                    ▼                      ▼
                              ┌──────────┐          ┌────────────┐
                              │ Keyframe │          │ Transcript │
                              │ sampler  │          │   text     │
                              └────┬─────┘          └──────┬─────┘
                                   │                       │
                                   ▼                       ▼
                              ┌──────────┐          ┌────────────┐
                              │ Tesseract│          │  Chunker   │
                              │ OCR      │          │  (moment   │
                              └────┬─────┘          │  builder)  │
                                   │                └──────┬─────┘
                                   │                       │
                                   ▼                       ▼
                              ┌─────────────────────────────────┐
                              │         Moment Assembly         │
                              │  (timestamp alignment, merge    │
                              │   transcript + OCR + ideas)     │
                              └───────────────┬─────────────────┘
                                              │
                                              ▼
                              ┌─────────────────────────────────┐
                              │         SQLite DB + FTS5        │
                              │  (videos, moments, ideas,       │
                              │   embeddings optionally)        │
                              └───────────────┬─────────────────┘
                                              │
                ┌─────────────────────────────┼─────────────┐
                │                             │             │
                ▼                             ▼             ▼
          ┌──────────┐                  ┌──────────┐  ┌──────────┐
          │  CLI:    │                  │  Search  │  │ Dedup    │
          │  ingest  │                  │  query   │  │ hash     │
          │  search  │                  │  ranker  │  │ check    │
          │  list    │                  └──────────┘  └──────────┘
          └──────────┘
```

**Key constraint:** Everything runs locally. Zero cloud dependencies for the MVP. Single process, single SQLite file.

---

## 2. Core Data Model (Pydantic / JSON)

### Video
```python
class Video(BaseModel):
    video_id: str          # e.g. "b3J2YVLbFiA" or hash of local path
    title: str
    source: str            # "youtube" | "local"
    url: str | None
    duration_sec: float
    status: str            # "pending" | "ingesting" | "ready" | "error"
    transcript_path: str | None
    created_at: datetime
    metadata: dict         # flexible: upload date, channel, fps, etc.
```

### Moment (the central object)
```python
class Moment(BaseModel):
    moment_id: str         # "video_id:start_sec:end_sec"
    video_id: str
    start_sec: float
    end_sec: float
    transcript_text: str
    ocr_text: str | None
    ideas: list[Idea]
    keyframe_paths: list[str]
    content_hash: str      # SHA256 of transcript_text[:500] + ocr_text[:500]
    parent_moment_id: str | None  # for dedup: if this is a near-duplicate, link to canonical
    metadata: dict
```

### Idea
```python
class Idea(BaseModel):
    idea_id: str           # "idea:<moment_id>:<n>"
    moment_id: str
    type: str              # "claim" | "step" | "definition" | "example" | "warning" | "workflow"
    text: str
    confidence: float      # 0.0 - 1.0
    source: str            # "rule" | "llm"
```

### SearchResult (output)
```python
class SearchResult(BaseModel):
    moment: Moment
    relevance_score: float
    matched_on: list[str]  # ["transcript", "ocr", "ideas"]
    video_title: str
    source_url: str
    moment_url: str        # youtube.com/watch?v=xxx&t=123s
```

---

## 3. Ingestion Pipeline

```
                    ┌─────────────────────┐
                    │  1. Download/Accept  │
                    │  yt-dlp or symlink   │
                    └────────┬────────────┘
                             │
                    ┌────────▼────────┐
                    │  2. Extract Audio│
                    │  ffmpeg -vn     │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼─────┐  ┌────▼─────┐  ┌─────▼────────┐
     │ 3. Transcribe │  │ 4. Sample │  │ 5. OCR       │
     │ Whisper (base)│  │ keyframes │  │ Tesseract    │
     │ .srt or .vtt  │  │ 1/sec or  │  │ on each      │
     │ output        │  │ scene chg │  │ keyframe     │
     └────────┬─────┘  └────┬─────┘  └─────┬────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │  6. Chunk &     │
                    │  Align Moments  │
                    │  - sentence     │
                    │    boundaries   │
                    │  - 30-60s fallbk│
                    │  - overlap ~10% │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  7. Extract     │
                    │  Ideas          │
                    │  - rule-based   │
                    │    patterns     │
                    │  - keyword      │
                    │    triggers     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  8. Store &     │
                    │  Index          │
                    │  - SQLite       │
                    │  - FTS5 index   │
                    │  - optional     │
                    │    embeddings   │
                    └─────────────────┘
```

**Step 6 — Chunking logic detail:**

```python
def chunk_transcript(
    transcript_entries: list[dict],  # [{start, end, text}]
    max_duration: float = 60.0,
    overlap_sec: float = 5.0,
) -> list[dict]:  # [{start, end, text}]
```

1. If transcript has sentence boundaries (SRT with proper punctuation): group sentences until duration approaches `max_duration`, carry 1-2 sentences as overlap.
2. If no sentence boundaries: use fixed 30-60s windows sliding by `max_duration - overlap_sec`.
3. Each chunk becomes a `Moment`.
4. OCR text is assigned to moments by timestamp overlap.

**Step 7 — Idea extraction (rule-based for MVP):**

```python
def extract_ideas(text: str) -> list[Idea]:
    patterns = {
        "definition": r"(is|are|refers to|defined as|means)\s",
        "step": r"(step|first|second|then|next|finally)\s",
        "warning": r"(warning|caution|careful|don't|avoid|never)\s",
        "claim": r"(because|therefore|thus|key insight|important)\s",
        "example": r"(for example|for instance|e\.g\.|like when|such as)\s",
    }
    # Split text into sentences, check each sentence for pattern matches
    # Assign type based on first matching pattern
    # Set confidence: 0.7 for rule-based
```

---

## 4. Search / Query Pipeline

```
  User query: "how do I use MCP with Playwright?"
                    │
                    ▼
          ┌──────────────────────┐
          │ 1. Tokenize &        │
          │    Normalize         │
          └─────────┬────────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
         ▼                     ▼
  ┌──────────────┐    ┌──────────────┐
  │ 2a. FTS5     │    │ 2b. Embed    │
  │ SQLite       │    │ query with   │
  │ MATCH        │    │ sentence-    │
  │ transcript,  │    │ transformers │
  │ ocr, ideas   │    │ → cosine sim │
  └──────┬───────┘    └──────┬───────┘
         │                     │
         └──────────┬──────────┘
                    ▼
          ┌──────────────────────┐
          │ 3. Combine & Rank    │
          │ keyword_score * 0.7 + │
          │ vector_score * 0.3   │
          │ (if vector enabled)  │
          └─────────┬────────────┘
                    ▼
          ┌──────────────────────┐
          │ 4. Return top-K      │
          │ moments with         │
          │ video info, URLs,    │
          │ timestamps, preview  │
          └──────────────────────┘
```

**FTS5 index:**
```sql
CREATE VIRTUAL TABLE moments_fts USING fts5(
    moment_id UNINDEXED,
    transcript_text,
    ocr_text,
    ideas_text,    -- concatenated idea texts
    content='moments'
);
```

The FTS5 content-sync approach keeps the full table in `moments` and only the searchable text in `moments_fts`.

---

## 5. File / Folder Structure

```
vidcrawl/
├── pyproject.toml
├── PLAN.md                         # this file
├── vidcrawl/
│   ├── __init__.py
│   ├── __main__.py                 # python -m vidcrawl entry
│   ├── cli.py                      # Typer CLI (ingest, search, list, info)
│   ├── config.py                   # paths, model choices, thresholds
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py              # Pydantic models (Video, Moment, Idea, SearchResult)
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── connection.py           # get_db() context manager
│   │   ├── schema.py               # CREATE TABLE statements
│   │   └── queries.py              # all SQL queries as functions
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── downloader.py           # yt-dlp wrapper
│   │   ├── audio.py                # ffmpeg audio extraction
│   │   ├── transcript.py           # Whisper ASR
│   │   ├── keyframes.py            # ffmpeg keyframe sampling
│   │   ├── ocr.py                  # Tesseract OCR
│   │   ├── chunking.py             # Moment builder from transcript + OCR
│   │   ├── ideas.py                # Rule-based idea extraction
│   │   └── pipeline.py             # orchestrate the full pipeline
│   │
│   ├── search/
│   │   ├── __init__.py
│   │   ├── indexer.py              # build/rebuild FTS5 index
│   │   ├── query.py                # search execution
│   │   └── ranker.py               # score combination
│   │
│   ├── dedup/
│   │   ├── __init__.py
│   │   └── hash.py                 # SHA256 content hash, exact dedup
│   │
│   └── utils/
│       ├── __init__.py
│       ├── ffmpeg.py               # ffprobe, format helpers
│       └── logging.py              # structured logging
│
├── data/
│   ├── videos/                     # downloaded video files
│   ├── frames/                     # extracted keyframes (.webp)
│   ├── transcripts/                # raw .srt/.vtt output
│   └── db/
│       └── vidcrawl.db             # SQLite database
│
└── tests/
    ├── __init__.py
    ├── conftest.py                 # fixtures (small test video, temp db)
    ├── test_ingestion.py
    ├── test_search.py
    └── test_dedup.py
```

---

## 6. Recommended Tech Stack

| Component | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Best ecosystem for ML/media processing |
| CLI framework | Typer | Simple, auto-docs, built on Click |
| ASR (if needed) | Whisper `base` model | Runs locally, good accuracy-speed tradeoff |
| OCR | Tesseract (via `pytesseract`) | Free, local, good enough for MVP |
| Video download | yt-dlp | Best YouTube downloader, actively maintained |
| Media processing | ffmpeg (via `ffmpeg-python`) | Swiss army knife for video/audio |
| Database | SQLite with FTS5 | Zero setup, file-based, full-text search built in |
| ORM | Raw SQL (with `sqlite3`) | No overhead for MVP; switch to SQLAlchemy later |
| Validation | Pydantic v2 | Schemas, serialization, type safety |
| Embeddings (opt) | sentence-transformers (`all-MiniLM-L6-v2`) | 384-dim, fast, good enough |
| Vector index (opt) | FAISS or numpy dot product | Simple, local, no server needed |

**Installation command:**
```bash
pip install typer pydantic yt-dlp openai-whisper pytesseract ffmpeg-python
# optional:
pip install sentence-transformers faiss-cpu
```

Requires system packages: `ffmpeg`, `tesseract-ocr`

---

## 7. Database Schema (SQLite)

```sql
CREATE TABLE IF NOT EXISTS videos (
    video_id        TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    source          TEXT NOT NULL CHECK(source IN ('youtube', 'local')),
    url             TEXT,
    duration_sec    REAL NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending', 'ingesting', 'ready', 'error')),
    transcript_path TEXT,
    error_message   TEXT,
    metadata        TEXT DEFAULT '{}',  -- JSON blob
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS moments (
    moment_id        TEXT PRIMARY KEY,          -- "video_id:start:end"
    video_id         TEXT NOT NULL REFERENCES videos(video_id),
    start_sec        REAL NOT NULL,
    end_sec          REAL NOT NULL,
    transcript_text  TEXT NOT NULL,
    ocr_text         TEXT,
    ideas            TEXT DEFAULT '[]',         -- JSON array of Idea objects
    keyframe_paths   TEXT DEFAULT '[]',         -- JSON array of paths
    content_hash     TEXT,                      -- SHA256 for exact dedup
    parent_moment_id TEXT REFERENCES moments(moment_id),  -- dedup chain
    embedding        BLOB,                     -- optional float32 numpy array
    metadata         TEXT DEFAULT '{}',         -- JSON blob
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ideas (
    idea_id     TEXT PRIMARY KEY,
    moment_id   TEXT NOT NULL REFERENCES moments(moment_id),
    type        TEXT NOT NULL
                CHECK(type IN ('claim','step','definition','example','warning','workflow')),
    text        TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 0.7,
    source      TEXT NOT NULL DEFAULT 'rule'
                CHECK(source IN ('rule', 'llm')),
    embedding   BLOB
);

-- FTS5 virtual table for full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS moments_fts USING fts5(
    moment_id UNINDEXED,
    transcript_text,
    ocr_text,
    ideas_text,          -- concatenated "type: text" for all ideas in this moment
    content='moments',
    content_rowid='rowid'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS moments_ai AFTER INSERT ON moments BEGIN
    INSERT INTO moments_fts(rowid, moment_id, transcript_text, ocr_text, ideas_text)
    VALUES (new.rowid, new.moment_id, new.transcript_text,
            COALESCE(new.ocr_text, ''),
            (SELECT group_concat(type || ': ' || text, ' ') FROM
                (SELECT value ->> 'type' AS type, value ->> 'text' AS text
                 FROM json_each(new.ideas))));
END;

CREATE TRIGGER IF NOT EXISTS moments_ad AFTER DELETE ON moments BEGIN
    INSERT INTO moments_fts(moments_fts, rowid, moment_id, transcript_text, ocr_text, ideas_text)
    VALUES ('delete', old.rowid, old.moment_id, old.transcript_text, old.ocr_text, '');
END;

CREATE TRIGGER IF NOT EXISTS moments_au AFTER UPDATE ON moments BEGIN
    INSERT INTO moments_fts(moments_fts, rowid, moment_id, transcript_text, ocr_text, ideas_text)
    VALUES ('delete', old.rowid, old.moment_id, old.transcript_text, old.ocr_text, '');
    INSERT INTO moments_fts(rowid, moment_id, transcript_text, ocr_text, ideas_text)
    VALUES (new.rowid, new.moment_id, new.transcript_text,
            COALESCE(new.ocr_text, ''),
            (SELECT group_concat(type || ': ' || text, ' ') FROM
                (SELECT value ->> 'type' AS type, value ->> 'text' AS text
                 FROM json_each(new.ideas))));
END;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_moments_video_id ON moments(video_id);
CREATE INDEX IF NOT EXISTS idx_moments_content_hash ON moments(content_hash);
CREATE INDEX IF NOT EXISTS idx_ideas_moment_id ON ideas(moment_id);
CREATE INDEX IF NOT EXISTS idx_ideas_type ON ideas(type);
```

---

## 8. APIs / Endpoints / Functions

### CLI Commands (Typer)

```python
# cli.py

app = typer.Typer()

@app.command()
def ingest(
    source: str = typer.Argument(..., help="YouTube URL or local file path"),
    whisper_model: str = "base",
    force: bool = False,
    no_ocr: bool = False,
):
    """Download (if URL), transcribe, chunk, OCR, extract ideas, index."""

@app.command()
def search(
    query: str = typer.Argument(...),
    top_k: int = 10,
    include_ocr: bool = True,
    min_score: float = 0.0,
):
    """Search across all ingested moments. Returns ranked results."""

@app.command()
def list_videos():
    """List all ingested videos with status and moment count."""

@app.command()
def info(
    video_id: str = typer.Argument(...),
):
    """Show details for a specific video including moment table of contents."""

@app.command()
def moment(
    moment_id: str = typer.Argument(...),
):
    """Show full detail for a specific moment."""

@app.command()
def status():
    """Show DB stats: video count, moment count, idea count, FTS status."""

@app.command()
def reindex():
    """Rebuild FTS index from scratch."""
```

### Core Python Functions (internal API)

```python
# ingestion/pipeline.py
def ingest_video(source: str, whisper_model: str = "base") -> Video

# ingestion/downloader.py
def download_youtube(url: str, output_dir: str) -> str  # returns video path
def accept_local(path: str) -> str                        # symlink or copy

# ingestion/audio.py
def extract_audio(video_path: str, output_path: str) -> str

# ingestion/transcript.py
def transcribe(audio_path: str, model_name: str = "base") -> list[dict]
# returns [{start, end, text}, ...]

# ingestion/keyframes.py
def extract_keyframes(video_path: str, output_dir: str, fps: float = 0.5) -> list[str]
# returns list of frame file paths

# ingestion/ocr.py
def ocr_frames(frame_paths: list[str]) -> dict[str, str]
# returns {frame_path: recognized_text}

# ingestion/chunking.py
def build_moments(transcript_entries: list[dict],
                  ocr_data: dict[str, str],
                  keyframe_data: dict[float, str],
                  video_id: str,
                  video_title: str) -> list[Moment]

# ingestion/ideas.py
def extract_rules(moment: Moment) -> list[Idea]

# db/queries.py
def insert_video(conn, video: Video) -> None
def insert_moment(conn, moment: Moment) -> None
def get_video(conn, video_id: str) -> Video | None
def get_moment(conn, moment_id: str) -> Moment | None
def get_moments_by_video(conn, video_id: str) -> list[Moment]
def search_moments(conn, query: str, top_k: int) -> list[tuple[Moment, float]]

# search/query.py
def execute_search(conn, query: str, top_k: int = 10) -> list[SearchResult]

# dedup/hash.py
def content_hash(moment: Moment) -> str
def is_duplicate(conn, moment: Moment) -> bool
```

---

## 9. UI / CLI Flow

```
$ vidcrawl ingest "https://youtube.com/watch?v=dQw4w9WgXcQ"

[1/6] Downloading video... ✓
[2/6] Extracting audio... ✓
[3/6] Transcribing (model=base)... ✓ (342 segments, 187s)
[4/6] Extracting keyframes... ✓ (94 frames)
[5/6] Running OCR... ✓ (88/94 frames had text)
[6/6] Chunking & extracting ideas...
  ─ 47 moments created
  ─ 62 ideas extracted (12 claim, 18 step, 8 definition, 10 example, 5 warning, 9 workflow)
  ─ 0 duplicates skipped
Done. video_id: "dQw4w9WgXcQ"


$ vidcrawl search "how do I connect MCP to Playwright"

Top 5 results:

[1] dQw4w9WgXcQ 12:40 — 13:15  (score: 0.89)
    "The useful pattern is to connect Playwright MCP so Claude can
    inspect the browser..."
    └ idea: workflow — "Playwright MCP lets Claude inspect browser UIs"

[2] abc123xyz 45:20 — 46:00  (score: 0.72)
    "MCP servers let you extend Claude with tools like web browsing..."
    └ idea: definition — "MCP is the Model Context Protocol for tool use"

...

$ vidcrawl list

video_id         │ title                          │ moments │ status
─────────────────┼────────────────────────────────┼─────────┼───────
dQw4w9WgXcQ      │ Example Tutorial               │ 47      │ ready
abc123xyz        │ Another Talk                   │ 32      │ ready


$ vidcrawl info dQw4w9WgXcQ

Title: Example Tutorial
Source: youtube (https://youtube.com/watch?v=dQw4w9WgXcQ)
Duration: 187s
Status: ready
Moments:
  00:00 — 00:35  │ Introduction and setup
  00:35 — 01:20  │ Installing dependencies
  01:20 — 01:55  │ Understanding MCP
  ...


$ vidcrawl moment "dQw4w9WgXcQ:760:820"

moment_id:    dQw4w9WgXcQ:760:820
video:        Example Tutorial
timestamp:    12:40 — 13:40
transcript:   The useful pattern is to connect Playwright MCP so
              Claude can inspect the browser after making code edits.
              This gives it a feedback loop...
ocr_text:     src/components/AtlasTree.tsx
              Graph View
              main.tsx
ideas:
  [workflow] Playwright MCP lets Claude inspect browser UIs after
             code edits.
  [claim]   This feedback loop dramatically reduces iteration time.
keyframes:   frames/dQw4w9WgXcQ_772.webp
source:      https://youtube.com/watch?v=dQw4w9WgXcQ&t=760
```

---

## 10. First 3 Test Videos / Corpus Strategy

| # | Video | Duration | Why | What it tests |
|---|-------|----------|-----|---------------|
| 1 | **A short coding tutorial** (e.g., "Build a CLI with Typer" — 8-15 min) | ~10 min | Screen recordings have rich OCR text (code, terminal). Clear spoken explanations with sentence boundaries. Good for chunking and idea extraction. | ASR quality, OCR on code, sentence-boundary chunking, idea extraction (steps, examples) |
| 2 | **A technical conference talk** (e.g., any PyCon/StrangeLoop talk — 20-35 min) | ~25 min | Dense with claims and definitions. Slide text for OCR. Speaker uses structured language. | Longer-form chunking, slide OCR, claim/definition extraction, FTS5 search over technical terms |
| 3 | **A podcast episode** (e.g., Lex Fridman or similar — 10-20 min excerpt) | ~15 min | Conversational, fewer sentence boundaries. No visual OCR. Tests fallback chunking. | Fixed-window chunking, handling disfluent speech, search over informal language |

**Corpus management:**
- Store test video URLs in `tests/test_corpus.json`
- A small `tests/fixtures/` dir with a ~30s clip extracted from each (for CI tests)
- YouTube links are preferred for easy reproducibility
- Always cache downloaded files to avoid re-downloading

---

## 11. 4-Week Build Plan

### Week 1: Foundation

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | `pyproject.toml`, project skeleton, config | `pip install -e .` works |
| 2 | Pydantic models (`Video`, `Moment`, `Idea`, `SearchResult`) | `python -c "from vidcrawl.models import Moment"` |
| 3 | SQLite schema + connection + `insert_video/get_video` | `test_db.py` creates tables and round-trips a row |
| 4 | `downloader.py` + `audio.py` (yt-dlp, ffmpeg) | `ingest()` can download a YouTube video and extract audio |
| 5 | `transcript.py` (Whisper base) | `ingest()` returns transcript entries |
| 6 | CLI skeleton (`ingest` command, basic output) | `vidcrawl ingest URL` downloads + transcribes |
| 7 | Integration test + catch-up / buffer | Full Week 1 flow works end-to-end |

### Week 2: Core Ingestion

| Day | Task | Deliverable |
|-----|------|-------------|
| 8 | `keyframes.py` — extract frames at 0.5fps | Keyframes saved to `data/frames/` |
| 9 | `ocr.py` — Tesseract on each keyframe | `ocr_text` populated on moments |
| 10 | `chunking.py` — sentence-boundary chunking + fallback | Moments created with proper `start_sec`/`end_sec` |
| 11 | `ideas.py` — rule-based idea extraction | Each moment has 1-3 ideas |
| 12 | `pipeline.py` — orchestrate all steps | Full `ingest()` creates Video + Moments + Ideas in DB |
| 13 | `dedup/hash.py` — content hashing + duplicate skip | Duplicate ingestion skips identical chunks |
| 14 | Test ingesting all 3 test videos | Debug issues, tune chunking params |

### Week 3: Search & Query

| Day | Task | Deliverable |
|-----|------|-------------|
| 15 | `db/schema.py` — FTS5 virtual table + triggers | FTS index auto-syncs with moments table |
| 16 | `search/indexer.py` — build/populate FTS index | `reindex` command works |
| 17 | `search/query.py` — execute FTS5 MATCH query | `search()` returns ranked moment_ids |
| 18 | `search/ranker.py` — score combination (optional +embeddings) | Ranked results with scores |
| 19 | CLI: `search` command with formatted output | `vidcrawl search "query"` prints pretty table |
| 20 | CLI: `list`, `info`, `moment`, `status` commands | Full CLI UX |
| 21 | Edge cases: no OCR, empty transcript, long videos | Graceful error handling |

### Week 4: Polish & Test

| Day | Task | Deliverable |
|-----|------|-------------|
| 22 | Test suite: unit tests for each module | `pytest` passes |
| 23 | Test suite: integration tests for full pipeline | CI-ready test coverage |
| 24 | Performance: benchmark on 30min video; profile | < 5 min ingest for 30min video |
| 25 | Error handling: retry logic, partial failures, user feedback | Robust pipeline |
| 26 | Documentation: README with install/usage | Usable by others |
| 27 | Demo prep: 3 test videos, sample queries, output screenshots | Demo-ready |
| 28 | Buffer / fix issues discovered during testing | Ship MVP |

---

## 12. What Should Be Mocked / Stubbed Instead of Fully Built

| Feature | MVP approach | Future full solution |
|---------|--------------|---------------------|
| **Vector embeddings** | Optional, off by default. If enabled: `sentence-transformers` + numpy dot product | Dedicated vector DB (pgvector, Qdrant) + multi-modal embeddings |
| **Idea extraction** | Simple regex/keyword rules (50 LoC) | LLM-based extraction with structured output |
| **Deduplication** | Exact SHA256 hash of text prefix | Semantic dedup with embedding similarity + clustering |
| **Scene detection** | Uniform frame sampling every 2s | PySceneDetect for smart keyframe selection |
| **OCR** | Tesseract (may miss some UI text) | PaddleOCR, TrOCR, or GPT-4V for complex layouts |
| **Transcription** | Whisper base (English only) | Whisper large, speaker diarization, language detection |
| **YouTube captions** | Not fetched; always re-transcribe with Whisper | Prefer existing captions when available (faster, cheaper) |
| **Web UI** | CLI only | FastAPI + React/Svelte UI |
| **Authentication** | None | API keys, user accounts |
| **Knowledge graph** | Not built; ideas are flat JSON | Property graph (Neo4j / SQLite + relations) |
| **Novelty scoring** | None | Embedding novelty scoring against existing corpus |
| **Code/equation detection** | None; OCR captures it raw | Language detection + syntax highlighting in results |
| **Multimodal retrieval** | Text-only search (transcript + OCR + ideas) | CLIP-based image+text joint retrieval |

---

## 13. What Metrics Prove the MVP Works

### Primary (must-have):
1. **Successful ingestion** — `ingest()` runs on all 3 test videos without crashing. All steps complete. 100% of expected moments created.
2. **Search returns relevant results** — For 5 hand-crafted queries per test video, the top-3 results contain the correct moment (human-judged).
3. **Timestamp accuracy** — Every search result's timestamp points within ±5 seconds of the relevant content in the source video.
4. **CLI is usable** — A new user can follow the README and ingest + search without errors.

### Secondary (nice-to-have):
5. **Ingest speed** — A 10-minute video ingests in <3 minutes (Whisper base + Tesseract).
6. **Dedup works** — Re-ingesting the same video adds 0 new moments (exact hash match).
7. **OCR captures key info** — Code snippets, slide titles, UI labels visible in results.
8. **Ideas are meaningful** — ≥60% of extracted ideas pass a "does this capture something real?" sanity check by the developer.

### Measurement:
```bash
# Automated
pytest tests/ --tb=short -v   # test coverage

# Manual
vidcrawl ingest test_video_1.mp4
vidcrawl search "query that should match moment X"
# → visually verify top result is moment X
```

---

## 14. What Mistakes to Avoid

1. **Overbuilding the idea extraction.** Rule-based is fine for MVP. Don't add LLM calls yet — they add latency, cost, and brittleness.

2. **Trying to handle every video format.** Start with `.mp4` and YouTube. Reject anything else with a clear error message.

3. **Not caching intermediate artifacts.** Keep the downloaded video, the extracted audio, and the raw transcript. Makes re-running steps cheap.

4. **Chunking without overlap.** Without overlap, you'll miss moments that span chunk boundaries. Always overlap 5-10s.

5. **Ignoring FFmpeg version issues.** Pin an FFmpeg version in docs. Different builds handle codecs differently.

6. **Making embeddings mandatory.** Offload vector search to "optional, if sentence-transformers is installed." Keep FTS5 as the primary/fallback.

7. **Storing video files in the repo.** Put `data/` in `.gitignore` from day one.

8. **Not cleaning up on failure.** If ingestion fails mid-way, partial DB state and orphan files accumulate. Use transactions + cleanup.

9. **Tuning for perfect results.** The goal is "good enough to prove the concept." 80% precision on search is acceptable for MVP.

10. **Writing too much code before testing.** Get a single end-to-end flow working in Week 1, then iterate.

---

## 15. How to Keep the Architecture Expandable

### Schema expansion points (fields already reserved):
- `Video.metadata` — JSON blob for future fields (channel, tags, language, resolution)
- `Moment.metadata` — JSON blob for novelty scores, scene labels, attention weights
- `Moment.parent_moment_id` — dedup chain ready for near-dedup clusters
- `Moment.embedding` — BLOB column exists; just needs population
- `Idea.embedding` — future multi-modal retrieval
- `Idea.confidence` — ready for LLM-based extraction confidence scores

### Module boundaries that won't need rewriting:

| Module | Extensible To |
|--------|---------------|
| `downloader.py` | Add playlist support, add S3/local network mount |
| `transcript.py` | Return existing captions first, fall back to Whisper; add diarization |
| `keyframes.py` | Switch from uniform sampling to PySceneDetect |
| `ocr.py` | Swap Tesseract for PaddleOCR without changing callers |
| `ideas.py` | Add `"llm"` source alongside `"rule"`; use same `Idea` model |
| `search/` | Add vector search score to the ranker without changing query interface |
| `db/` | Migrate from SQLite to PostgreSQL by swapping `connection.py` |
| `dedup/hash.py` | Add embedding-based near-dedup without changing caller interface |
| `cli.py` | Add UI commands; CLI and API share the same `search.execute_search()` |

### Future architectural additions (bolt-on, not rewrite):
- **Web UI**: New `vidcrawl/ui/` module. FastAPI wraps existing `search/` and `db/` functions.
- **Knowledge graph**: New `vidcrawl/graph/` module. Reads `Moment.ideas` and builds graph edges. No schema changes needed — edges reference `moment_id` and `idea_id`.
- **Novelty scoring**: New `vidcrawl/novelty/` module. Compares new `Moment.embedding` against existing corpus. Uses existing `Moment.metadata` dict for scores.
- **Multimodal search**: Extends `search/ranker.py` to also consider keyframe embeddings via CLIP. Keyframe paths already stored in `Moment.keyframe_paths`.
- **Diarization**: `transcript.py` returns speaker labels. Stored in `Moment.metadata["speaker"]`.

### Golden rule for expandability:
Every new feature should add a file or module, not modify an existing one. The `Moment` schema and `db/queries.py` are the only files expected to accumulate fields over time.
