# VidCrawl: Timestamp-Grounded Multimodal Idea Graphs for Redundancy-Aware Video Search

## Abstract

VidCrawl is a local-first research prototype that transforms video search
results into timestamp-grounded, multimodal idea graphs. It extracts structured
knowledge from video streams — transcript text, OCR, code, commands, errors,
equations — and connects them through typed graph edges. The system combines
SQLite FTS5, graph-aware reranking, optional semantic embeddings, and
rule-based claim extraction to provide redundancy-aware, diversity-aware,
explainable video search without any cloud dependencies or LLM calls.

## Problem

Video content contains dense, multimodal knowledge that is hard to search:

1. **Redundancy**: Tutorials repeat the same information across moments and videos
2. **Timestamp ambiguity**: Search results lack precise temporal grounding
3. **Modality separation**: Transcript, visuals, and code are searched independently
4. **Staleness**: Software/API tutorials become outdated without detection
5. **No graph structure**: Search results are flat lists without relational context

## Hypothesis

A timestamp-grounded multimodal idea graph can improve video search quality
across precision, diversity, and explainability — without neural embeddings.

## System Design

### Data Model
- **Moments**: Timestamped (start_sec, end_sec) video segments with transcript, OCR, ideas, keyframes
- **Evidence**: Typed records (transcript, OCR, idea, code, command, error, equation)
- **Ideas**: Rule-extracted knowledge units (definition, step, warning, example, comparison, limitation)
- **Claims**: Structured statements extracted from transcript/OCR
- **Entities**: Acronyms, capitalized phrases, code identifiers, file paths
- **Graph**: Typed nodes and edges connecting all the above

### Search Pipeline
1. FTS5 keyword retrieval
2. Graph-aware feature extraction (evidence, ideas, entities, modalities, cluster info)
3. Deterministic reranking with configurable weights
4. MMR-like diversity selection
5. Ranking reason generation

### Optional Extensions
- Hash-based or sentence-transformer embeddings for semantic search
- Hybrid FTS + semantic + graph search

## Research Contributions

1. **Moment-Centric Video Model**: A data model that preserves temporal grounding
2. **Multimodal Idea Graph**: Typed graph connecting videos, moments, ideas, evidence, entities, claims
3. **Graph-Aware Reranking**: Deterministic scoring using graph features
4. **Deterministic Diversity**: MMR-like selection without embeddings
5. **Technical Evidence Extraction**: Code, commands, errors, equations from video
6. **Rule-Based Claim Graph**: No-LLM claim extraction and contradiction detection
7. **Explanation-Diverse Retrieval**: Ranking reasons and score components per result
8. **Local-First Architecture**: Zero cloud dependencies, fully reproducible

## Evaluation Plan

- **Demo corpus**: 3 videos, 16 moments with duplicates and variants
- **Metrics**: Top-1/3/5 hit rate, term/video hit rate, entity coverage, idea diversity, duplicate collapse rate, latency
- **Modes**: Raw FTS, graph rerank, diverse, semantic, hybrid
- **Benchmarks**: demo_queries.json, technical_queries.json, diversity_queries.json, freshness_queries.json

## Limitations

- Rule-based entity extraction (no semantic disambiguation)
- Hand-tuned reranking weights
- Simple MMR diversity (no semantic similarity)
- No learned ranking model
- No global PageRank or centrality
- Graph is built batch-style (no incremental updates)
- O(n²) pairwise near-text dedupe

## Future Work

- Optional local embeddings for semantic search/dedupe
- Learned ranking with user feedback
- Visual/equation/code entity embeddings
- Hypergraph evidence bundles
- Claim-level ranking and verification
- Temporal freshness decay models
- Incremental graph updates
- Transitive duplicate closure
