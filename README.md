# VidCrawl

Local-first video intelligence and search MVP.

VidCrawl indexes video knowledge locally — ingest videos, extract transcripts
+ OCR + keyframes, chunk into timestamped **Moment** objects, extract rule-based
**Ideas**, and search across everything with SQLite FTS5. No cloud dependencies,
no LLM calls.

## Quickstart

```bash
pip install -e .
vidcrawl init
vidcrawl demo init
vidcrawl stats
vidcrawl dedupe run
vidcrawl dedupe stats
vidcrawl graph build
vidcrawl graph stats
vidcrawl search "playwright browser"
vidcrawl search "playwright browser" --graph-context
vidcrawl search "playwright browser" --include-duplicates
vidcrawl search "playwright browser" --diverse
vidcrawl show demo_coding:0.00:12.00
vidcrawl graph show demo_coding:0.00:12.00
vidcrawl graph neighbors demo_coding
vidcrawl graph export --out graph.json
vidcrawl eval
```

## Core Thesis

Videos contain structured knowledge embedded in unstructured streams (audio,
visuals, code). VidCrawl extracts **Moments** — timestamped, annotated segments
of video content — and makes them searchable with local-first tools.

A **Moment** is the core unit: a time range with transcript text, OCR text,
extracted ideas, keyframe references, and evidence records. Moments are
searchable via SQLite FTS5.

A **Graph** layer connects videos, moments, ideas, evidence, entities, and
duplicate/variant clusters into a local multimodal idea graph. This enables
graph-aware search, duplicate cluster navigation, entity discovery, and
prepares the ground for future claim graphs, graph reranking, and
explanation-diverse retrieval.

## Status

**Week 7 — Graph-Aware Reranking & Explanation-Diverse Retrieval** (complete)

- 273+ tests passing
- All existing Week 1–6 functionality preserved
- **Feature extraction**: `vidcrawl/search/features.py` — per-result graph, content, and query features
- **Graph-aware scoring**: `vidcrawl/search/rerank.py` — boosts for OCR, ideas, evidence, entities, modalities, canonical status; penalties for exact duplicates
- **Diverse selection**: `vidcrawl/search/diversity.py` — MMR-like selection that avoids same cluster/video/idea-type redundancy
- **Ranking reasons**: each result explains why it was ranked (e.g. "has 3 extracted idea(s)", "canonical moment")
- **CLI flags**: `--rerank/--no-rerank`, `--explain-ranking`, `--raw-ranking`
- **Eval metrics**: ranking/diversity metrics (unique videos, clusters, entity coverage, idea diversity)
- Search defaults to rerank when graph exists, falls back gracefully without it

## Setup

```bash
pip install -e .
```

Requires Python 3.11+.

### Optional Dependencies

| Tool | Install | Purpose |
|---|---|---|
| ffmpeg | `apt install ffmpeg` (system) | Keyframe extraction |
| Tesseract | `apt install tesseract-ocr` (system) | OCR on keyframes |
| pytesseract | `pip install pytesseract` | Python OCR bindings |
| openai-whisper | `pip install openai-whisper` | Audio transcription |
| yt-dlp | `pip install yt-dlp` | YouTube video download |

All optional tools degrade gracefully. Ingestion works without any of them.

## CLI Reference

### `vidcrawl init`

Initialize the database and directory structure.

```
vidcrawl init [--data-dir PATH]
```

Idempotent — safe to run multiple times.

### `vidcrawl ingest <source>`

Ingest a local video file or register a YouTube URL.

```
vidcrawl ingest ~/videos/tutorial.mp4
vidcrawl ingest ~/videos/tutorial.mp4 --no-process
vidcrawl ingest https://youtube.com/watch?v=abc123
vidcrawl ingest https://youtube.com/watch?v=abc123 --process --no-download
```

Flags:
- `--process/--no-process` — Run full processing pipeline (default: --process)
- `--download/--no-download` — Download YouTube video via yt-dlp (default: --download)
- `--data-dir PATH` — Data directory (default: data/)

YouTube download requires yt-dlp. If missing, prints install help and falls
back to metadata-only registration.

### `vidcrawl list`

List all registered videos with moment counts.

```
vidcrawl list [--data-dir PATH]
```

### `vidcrawl inspect <video_id>`

Show video details, counts, and first 3 moments.

```
vidcrawl inspect <video_id> [--data-dir PATH]
```

### `vidcrawl dedupe run`

Run deduplication over the current corpus. Detects exact duplicates (via content
hash) and near-text duplicates (via Jaccard + n-gram + SequenceMatcher).

```
vidcrawl dedupe run [--data-dir PATH]
vidcrawl dedupe run --threshold 0.8 [--data-dir PATH]
vidcrawl dedupe run --dry-run [--data-dir PATH]
vidcrawl dedupe run --json [--data-dir PATH]
```

Options:
- `--threshold`, `-t` — Similarity threshold for near-duplicate detection (default: 0.75)
- `--dry-run` — Report what would be done without modifying the database
- `--json` — JSON output

Output shows counts of exact, near-text, same-idea, and variant duplicates found.

### `vidcrawl dedupe stats`

Show duplicate statistics for the corpus.

```
vidcrawl dedupe stats [--data-dir PATH]
vidcrawl dedupe stats --json [--data-dir PATH]
```

Shows total duplicate records, unique moments, redundancy ratio, and breakdown
by duplicate type.

### `vidcrawl dedupe show <moment_id>`

Show duplicate/variant relationships for a specific moment.

```
vidcrawl dedupe show <moment_id> [--data-dir PATH]
```

### `vidcrawl search <query>`

Search across all moments using SQLite FTS5 with optional graph-aware reranking.

```
vidcrawl search "playwright browser"
vidcrawl search "error" --limit 5
vidcrawl search "warning" --video-id demo_coding
vidcrawl search "definition" --json
vidcrawl search "install" --no-snippets
vidcrawl search "playwright browser" --explain-ranking
vidcrawl search "playwright browser" --raw-ranking
vidcrawl search "playwright browser" --diverse --graph-context
```

Flags:
- `--limit N`, `-l N` — Max results (default: 10)
- `--video-id ID` — Filter to a specific video
- `--json` — Structured JSON output
- `--snippets/--no-snippets` — Show/hide text snippets (default: show)
- `--include-duplicates` — Include exact duplicate moments in results
- `--diverse` — Include variant moments for diversity in results
- `--graph-context` — Show graph context (entities, ideas, evidence, cluster info)
- `--rerank/--no-rerank` — Enable graph-aware reranking (default: on if graph exists)
- `--explain-ranking` — Show ranking reasons and score components
- `--raw-ranking` — Use FTS-only ranking (no rerank)
- `--data-dir PATH` — Data directory

### Reranking signals

When the graph is built, search ranks results using:

| Signal | Boost | Example reason |
|--------|-------|---------------|
| Has OCR text | +0.10 | "matched OCR" |
| Has keyframe | +0.10 | "has keyframe" |
| Has ideas | +0.15 + per-idea up to 0.30 | "has 3 extracted idea(s)" |
| Evidence records | +0.05 + per-evidence up to 0.20 | "has 3 evidence record(s)" |
| Entity connections | per-entity up to 0.15 | "connected to 2 entit(ies)" |
| Multiple modalities | per-modality up to 0.15 | "3 evidence modalities" |
| Canonical in cluster | +0.20 | "canonical moment" |
| Exact duplicate | -0.50 | "exact duplicate (penalized)" |
| Variant (diverse mode) | +0.20 | "variant preserved for diversity" |
| Query in title | +0.30 | "matched title" |
| Query in idea | +0.10 | "matched idea text" |
| Warning/example terms | +0.05-0.10 | "contains warning terms" |
| Graph support strength | per-connection up to 0.20 | "3 graph support connection(s)" |

Each result shows:
- Rank, video title, timestamp range, relevance score
- Transcript snippet (surrounding matched terms)
- OCR snippet if relevant
- Idea types extracted from the moment
- Match reasons (transcript + OCR + idea + title + keyframes)
- Keyframe paths
- YouTube timestamp link if source URL exists

### `vidcrawl show <moment_id>`

Display full detail for a single Moment.

```
vidcrawl show <moment_id> [--data-dir PATH]
```

Shows:
- Video title, timestamp, source URL with timestamp link
- Full transcript text
- Full OCR text
- All extracted ideas with types and confidence
- All evidence records
- Keyframe paths
- Content hash

### `vidcrawl stats`

Show corpus-level statistics.

```
vidcrawl stats [--data-dir PATH]
vidcrawl stats --verbose [--data-dir PATH]
```

Base: videos, moments, evidence, ideas, keyframes, duplicates, FTS rows,
database path.

With `--verbose`: avg moments/video, avg chars/moment, total chars, database
size, artifacts directory size.

### `vidcrawl demo init`

Create a demo corpus with 3 videos and 13 moments for testing.

```
vidcrawl demo init [--data-dir PATH]
```

Videos: coding tutorial, ML conference talk, UX podcast. Includes transcripts,
OCR text, and extracted ideas covering all 6 idea types. No network or optional
tools required.

### `vidcrawl eval [query_file]`

Run evaluation against a set of test queries.

```
vidcrawl eval [--data-dir PATH]
vidcrawl eval queries.json [--data-dir PATH]
vidcrawl eval --rerank [--data-dir PATH]
vidcrawl eval --no-rerank [--data-dir PATH]
vidcrawl eval --diverse [--data-dir PATH]
```

Without a file, uses built-in demo queries (works after `demo init`). With a
file, loads a JSON file in this format:

```json
{
  "queries": [
    {
      "query": "playwright browser",
      "expected_terms": ["playwright", "browser"],
      "expected_video_id": "demo_coding",
      "expected_moment_contains": "MCP"
    }
  ]
}
```

Reports: total queries, top-1/3/5 hit rates, expected video/term hit rates,
avg results/query, avg query latency.

### `vidcrawl graph build`

Build the multimodal idea graph from existing data. Creates nodes for videos,
moments, ideas, evidence, entities, and duplicate clusters, with typed edges
connecting them.

```
vidcrawl graph build [--data-dir PATH]
vidcrawl graph build --rebuild [--data-dir PATH]
```

Idempotent — safe to run multiple times. Use `--rebuild` to clear and rebuild.

### `vidcrawl graph stats`

Show graph-level statistics.

```
vidcrawl graph stats [--data-dir PATH]
vidcrawl graph stats --json [--data-dir PATH]
```

Shows: total nodes, total edges, average degree, duplicate clusters, connected
components, breakdown by node type and edge type.

### `vidcrawl graph show <node_or_ref_id>`

Display details for a single graph node. Accepts a `node_id` or `ref_id`.

```
vidcrawl graph show demo_coding:0.00:12.00 [--data-dir PATH]
vidcrawl graph show demo_coding [--data-dir PATH]
vidcrawl graph show demo_coding --json [--data-dir PATH]
```

### `vidcrawl graph neighbors <node_or_ref_id>`

Show connected nodes and edges for a graph node.

```
vidcrawl graph neighbors demo_coding [--data-dir PATH]
vidcrawl graph neighbors demo_coding:0.00:12.00 --json [--data-dir PATH]
```

Shows degree, all edges, and neighbor nodes with their types and labels.

### `vidcrawl graph export`

Export the graph to JSON.

```
vidcrawl graph export --out graph.json [--data-dir PATH]
```

### `vidcrawl reindex`

Rebuild the FTS5 search index from scratch.

```
vidcrawl reindex [--data-dir PATH]
```

### `vidcrawl --version`

Show version number.

## Search

VidCrawl uses SQLite FTS5 as the primary search engine. The following fields
are searched:

- **Transcript text** — spoken content from sidecar files or Whisper ASR
- **OCR text** — text detected in video frames
- **Idea text** — extracted ideas with type labels (warning, step, definition, etc.)
- **Video title** — the video's title
- **Video description** — metadata description if available

### Ranking

Ranking combines FTS5 BM25 scores with simple Python-side boosts:

| Signal | Boost |
|---|---|
| Query term in transcript | +0.5 |
| Query term in video title | +0.5 |
| Query term in OCR | +0.3 |
| Query term in idea summary | +0.2 |
| Moment has keyframes | +0.2 |
| Moment has ideas | +0.1 |

Higher score = better match. No ML, no trained ranker.

### Match explanations

Each result includes a `match_reasons` field explaining why it matched:

```
Match: transcript + title
Match: transcript + OCR + idea + has_ideas
```

### Example output

```
Query: playwright browser
Results: 5

1. Building a Playwright MCP Server for Browser Automation — 0:12–0:25 — score 3.01
   Moment: demo_coding:12.00:25.00
   Transcript: First, install the Playwright package using npm...
   OCR: npm install playwright
   Ideas: step
   Match: transcript + title
   Link: https://youtube.com/watch?v=demo_coding_001&t=12s

2. Building a Playwright MCP Server for Browser Automation — 0:00–0:12 — score 2.32
   Moment: demo_coding:0.00:12.00
   Transcript: Welcome to this tutorial on building a Playwright MCP server...
   OCR: Playwright MCP Server Tutorial Introduction
   Ideas: definition
   Match: transcript + OCR + title
```

### JSON mode

```bash
vidcrawl search "definition" --json
```

```json
[
  {
    "rank": 1,
    "score": 2.3,
    "moment_id": "demo_coding:0.00:12.00",
    "video_title": "Building a Playwright MCP Server for Browser Automation",
    "transcript_snippet": "Welcome to this tutorial on building a Playwright MCP server...",
    "match_reasons": ["transcript", "OCR", "title"],
    "youtube_url": "https://youtube.com/watch?v=demo_coding_001&t=0s"
  }
]
```

## Demo Flow

```bash
# 1. Initialize
vidcrawl init

# 2. Create demo corpus
vidcrawl demo init

# 3. View statistics
vidcrawl stats --verbose

# 4. Search
vidcrawl search "playwright browser"
vidcrawl search "warning" --json
vidcrawl search "definition"

# 5. Show a moment
vidcrawl show demo_coding:0.00:12.00

# 6. Run dedupe
vidcrawl dedupe run
vidcrawl dedupe stats

# 7. Build and explore graph
vidcrawl graph build
vidcrawl graph stats
vidcrawl graph show demo_coding
vidcrawl graph neighbors demo_coding:0.00:12.00
vidcrawl graph export --out graph.json

# 8. Search with graph context
vidcrawl search "playwright browser" --graph-context

# 9. Search with reranking and explanations
vidcrawl search "playwright browser" --explain-ranking
vidcrawl search "playwright browser" --raw-ranking
vidcrawl search "playwright browser" --diverse --graph-context --explain-ranking

# 10. Run evaluation with rerank/diversity
vidcrawl eval --rerank
vidcrawl eval --diverse

```

## Evaluation Results (Demo Corpus)

| Metric | Value |
|---|---|
| Top-1 hit rate | 90% |
| Expected term hit rate | 93.3% |
| Expected video hit rate | 6/10 |
| Avg results per query | 2.8 |
| Avg query latency | 0.51 ms |

## Project Structure

```
vidcrawl/
  __init__.py
  __main__.py          # python -m vidcrawl
  cli.py               # Typer CLI (all commands)
  config.py            # paths, directories
  db.py                # SQLite connection, schema, CRUD, FTS
  demo.py              # Demo corpus creation
  eval.py              # Evaluation harness
  models.py            # Pydantic models
  dedupe/
    __init__.py
    normalize.py       # Text normalization, content hashing
    similarity.py      # Jaccard, n-gram, SequenceMatcher similarity
    novelty.py         # Rule-based novelty scoring
    cluster.py         # Dedupe pipeline & canonical selection
  graph/
    __init__.py
    entities.py        # Entity extraction (acronyms, capitalized phrases, code IDs, paths)
    build.py           # Graph builder (node/edge creation, schema)
    query.py           # Graph queries (get_node, get_neighbors, graph context)
    export.py          # Graph export to JSON
    stats.py           # Graph statistics (degree, components, counts by type)
  ingest/
    downloader.py      # YouTube download (optional yt-dlp) + local path resolver
    media.py           # Video file validation
    metadata.py        # File metadata extraction
    transcript.py      # Sidecar loader, SRT/VTT/TXT/JSON parser, Whisper stub
  process/
    chunking.py        # Transcript-to-Moment chunking with overlap
    ideas.py           # Rule-based idea extraction (6 types)
    keyframes.py       # ffmpeg-based frame extraction
    ocr.py             # Tesseract OCR on keyframes
    pipeline.py        # End-to-end ingestion pipeline
  search/
    fts.py             # FTS index rebuild wrapper
    query.py           # Search execution (SearchResult, search_moments, boosts, dedupe filtering)
    features.py        # Per-result feature extraction for reranking
    rerank.py          # Graph-aware scoring with ranking reasons
    diversity.py       # MMR-like diverse result selection
  utils/
    hashing.py         # Content hashing utilities
    time.py            # Timestamp formatting, youtube_timestamp_url
tests/
  test_cli.py          # CLI init/list/inspect/ingest tests
  test_db.py           # Database CRUD and FTS tests
  test_models.py       # Pydantic model validation tests
  test_utils.py        # Hashing and time utility tests
  test_week2.py        # Transcript parsing, chunking, ideas, pipeline tests
  test_week3.py        # Search, query, CLI search/show/stats, FTS rebuild tests
  test_week4.py        # Demo corpus, eval, downloader, robustness, stats verbose
  test_dedupe.py       # Dedupe: normalize, similarity, novelty, cluster, CLI
data/                  # Local storage (gitignored)
```

## Architecture and Extensibility

The search layer (`vidcrawl/search/`) is designed with clear seams for future
enhancement without breaking the current FTS5-based approach:

- **Vector search**: `SearchResult.metadata_json` can hold embedding vectors
- **Graph reranking**: `idea_types` and `idea_summary` support structured reranking
- **Novelty/diversity**: `rank` field enables MMR-based diversification
- **Personalized ranking**: `_apply_boosts()` accepts user history signals
- **Visual/equation/code search**: Additional FTS5 columns can be added

### Current Limitations

- **Hand-tuned weights**: Reranking weights are simple constants, not learned
  from data. May need adjustment per corpus.
- **No embeddings**: Still no semantic understanding. Reranking uses graph
  structure and content features, not semantic similarity.
- **Diversity heuristic is simple**: MMR-like selection uses cluster/video/idea-type
  avoidance. No semantic cluster similarity. No embedding diversity.
- **Graph must be built separately**: `vidcrawl graph build` must run before
  reranking is active. Without it, search falls back to FTS-only.
- **No PageRank**: Graph reranking uses local degree/connection features only.
  No global centrality, personalization, or random-walk features.
- **Rule-based entity extraction**: Capitalized phrases, known terms, and simple
  patterns only.
- **No full claim graph**: `claim_stub` node type exists as a placeholder.
- **Graph is built batch-style**: Rebuilds from scratch on each `--rebuild`.
- **Transitive duplicate closure**: Duplicate edges are pairwise.
- **O(n²) pairwise dedupe**: Near-text dedupe is O(n²) across canonicals.
- **Novelty scoring**: Rule-based with keyword lists.
- BM25 scores not normalized across queries.
- Keyword overlap boosts are binary (present/absent).
- YouTube download requires yt-dlp.

## Recommended Next Milestone

**Add optional local embeddings for semantic search and semantic dedupe.**

Use a clean interface that does not replace FTS/graph ranking:

- sentence-transformers or ONNX-based local embeddings
- Cosine similarity for semantic search reranking
- Embedding-based dedupe (replace Jaccard)
- Embedding-based entity disambiguation
- Embedding diversity for MMR
- All optional — FTS/graph ranking remains as fallback

This builds naturally on the graph reranking layer: embeddings add semantic
understanding to the existing structure.

## License

MIT
