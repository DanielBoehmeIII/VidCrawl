# VidCrawl Architecture

## Overview

VidCrawl is a local-first video intelligence system that extracts timestamped
**Moments** from video content, indexes them with SQLite FTS5, builds a
multimodal idea graph, and searches with hybrid ranking.

## Data Flow

```
Video Input → Transcript Parsing → Chunking → Moment Creation
                                    ↓
                              OCR Extraction → Keyframe Extraction
                                    ↓
                           Rule-based Idea Extraction
                                    ↓
                        Evidence Records (transcript, OCR, idea,
                                          code, command, error, equation)
                                    ↓
                        FTS5 Index → Search (FTS + Graph Rerank)
                                    ↓
                        Graph Build → Graph Tables → Graph Context
                                    ↓
                        Embeddings (optional) → Semantic Search
                                    ↓
                        Claim Extraction → Claim Graph
                                    ↓
                        Freshness Scoring → Staleness Markers
```

## Core Concepts

### Moment
A timestamped video segment (start_sec, end_sec) with:
- transcript_text
- ocr_text
- extracted ideas
- keyframe references
- evidence records

### Evidence
Records attached to moments by modality:
- transcript, OCR, keyframe, idea, metadata
- code, command, error, equation (technical)

### Graph
Nodes: video, moment, idea, evidence, entity, duplicate_cluster, claim
Edges: typed relationships connecting all node types

## Modules

```
vidcrawl/
  cli.py           - Typer CLI
  db.py            - SQLite schema, CRUD
  models.py        - Pydantic models
  config.py        - Paths
  demo.py          - Demo corpus
  eval.py          - Evaluation harness
  freshness.py     - Staleness scoring

  ingest/          - Video ingestion
  process/         - Chunking, ideas, OCR, keyframes
  search/          - FTS5, reranking, features, diversity
  dedupe/          - Normalization, similarity, dedupe
  graph/           - Build, query, export, stats
  embeddings/      - Providers, storage, similarity
  technical/       - Code, commands, errors, equations
  claims/          - Extraction, normalization, clustering
```
