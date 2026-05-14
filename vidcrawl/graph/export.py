import json
from pathlib import Path
from typing import Any, Optional

from vidcrawl.db import get_db


def export_graph(
    db_path: Path,
    output_path: Optional[Path] = None,
    fmt: str = "json",
) -> dict[str, Any]:
    conn = get_db(db_path)
    try:
        node_rows = conn.execute(
            "SELECT node_id, node_type, ref_id, label, metadata_json, created_at "
            "FROM graph_nodes ORDER BY node_type, node_id"
        ).fetchall()

        edge_rows = conn.execute(
            "SELECT edge_id, source_node_id, target_node_id, edge_type, weight, metadata_json, created_at "
            "FROM graph_edges ORDER BY edge_type, edge_id"
        ).fetchall()

        nodes = []
        for r in node_rows:
            nodes.append({
                "node_id": r["node_id"],
                "node_type": r["node_type"],
                "ref_id": r["ref_id"],
                "label": r["label"],
                "metadata": json.loads(r["metadata_json"]) if r["metadata_json"] else {},
                "created_at": r["created_at"],
            })

        edges = []
        for r in edge_rows:
            edges.append({
                "edge_id": r["edge_id"],
                "source_node_id": r["source_node_id"],
                "target_node_id": r["target_node_id"],
                "edge_type": r["edge_type"],
                "weight": r["weight"],
                "metadata": json.loads(r["metadata_json"]) if r["metadata_json"] else {},
                "created_at": r["created_at"],
            })

        result = {
            "graph": {
                "nodes": nodes,
                "edges": edges,
            },
            "summary": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
            },
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

        return result
    finally:
        conn.close()
