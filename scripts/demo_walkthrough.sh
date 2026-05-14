#!/usr/bin/env bash
set -euo pipefail

echo "=== VidCrawl Demo Walkthrough ==="
echo ""

# Clean start
DATA_DIR=$(mktemp -d)
echo "Using data directory: $DATA_DIR"
echo ""

# 1. Init
echo "=== 1. Initialize ==="
vidcrawl init --data-dir "$DATA_DIR"
echo ""

# 2. Demo init
echo "=== 2. Create Demo Corpus ==="
vidcrawl demo init --data-dir "$DATA_DIR"
echo ""

# 3. Stats
echo "=== 3. Statistics ==="
vidcrawl stats --data-dir "$DATA_DIR"
echo ""

# 4. Dedupe
echo "=== 4. Run Deduplication ==="
vidcrawl dedupe run --data-dir "$DATA_DIR"
vidcrawl dedupe stats --data-dir "$DATA_DIR"
echo ""

# 5. Build Graph
echo "=== 5. Build Graph ==="
vidcrawl graph build --data-dir "$DATA_DIR"
vidcrawl graph stats --data-dir "$DATA_DIR"
echo ""

# 6. Embeddings (hash)
echo "=== 6. Build Embeddings ==="
vidcrawl embed build --provider hash --dimension 64 --data-dir "$DATA_DIR"
echo ""

# 7. Technical Evidence
echo "=== 7. Extract Technical Evidence ==="
vidcrawl technical extract --data-dir "$DATA_DIR"
vidcrawl technical stats --data-dir "$DATA_DIR"
echo ""

# 8. Claims
echo "=== 8. Extract Claims ==="
vidcrawl claims extract --data-dir "$DATA_DIR"
vidcrawl claims stats --data-dir "$DATA_DIR"
echo ""

# 9. Freshness
echo "=== 9. Freshness Scoring ==="
vidcrawl freshness run --data-dir "$DATA_DIR"
vidcrawl freshness stats --data-dir "$DATA_DIR"
echo ""

# 10. Search examples
echo "=== 10. Search Examples ==="
echo ""
echo "-- Raw ranking --"
vidcrawl search "playwright browser" --raw-ranking --data-dir "$DATA_DIR"
echo ""
echo "-- Explain ranking --"
vidcrawl search "playwright browser" --explain-ranking --data-dir "$DATA_DIR" | head -20
echo ""
echo "-- Graph context --"
vidcrawl search "playwright browser" --graph-context --data-dir "$DATA_DIR" | head -20
echo ""
echo "-- Diverse --"
vidcrawl search "playwright browser" --diverse --graph-context --data-dir "$DATA_DIR" | head -20
echo ""

# 11. Eval
echo "=== 11. Evaluation ==="
vidcrawl eval --rerank --data-dir "$DATA_DIR"
echo ""

# 12. Report
echo "=== 12. Generate Report ==="
vidcrawl report generate --out "$DATA_DIR/report.md" --data-dir "$DATA_DIR"
echo "Report saved to $DATA_DIR/report.md"
echo ""

# 13. Graph export
echo "=== 13. Export Graph ==="
vidcrawl graph export --out "$DATA_DIR/graph.json" --data-dir "$DATA_DIR"
echo "Graph exported to $DATA_DIR/graph.json"
echo ""

# 14. Doctor
echo "=== 14. System Doctor ==="
vidcrawl doctor --data-dir "$DATA_DIR"
echo ""

rm -rf "$DATA_DIR"
echo ""
echo "=== Demo Walkthrough Complete ==="
