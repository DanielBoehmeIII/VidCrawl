from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vidcrawl.db import get_db


@dataclass
class GraphStats:
    total_nodes: int = 0
    total_edges: int = 0
    nodes_by_type: dict[str, int] = field(default_factory=dict)
    edges_by_type: dict[str, int] = field(default_factory=dict)
    duplicate_clusters: int = 0
    connected_components: int = 0
    average_degree: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "nodes_by_type": dict(sorted(self.nodes_by_type.items())),
            "edges_by_type": dict(sorted(self.edges_by_type.items())),
            "duplicate_clusters": self.duplicate_clusters,
            "connected_components": self.connected_components,
            "average_degree": round(self.average_degree, 2),
        }


def compute_graph_stats(db_path: Path) -> GraphStats:
    conn = get_db(db_path)
    try:
        stats = GraphStats()

        stats.total_nodes = conn.execute(
            "SELECT COUNT(*) as c FROM graph_nodes"
        ).fetchone()["c"]

        stats.total_edges = conn.execute(
            "SELECT COUNT(*) as c FROM graph_edges"
        ).fetchone()["c"]

        for row in conn.execute(
            "SELECT node_type, COUNT(*) as c FROM graph_nodes GROUP BY node_type"
        ).fetchall():
            stats.nodes_by_type[row["node_type"]] = row["c"]

        for row in conn.execute(
            "SELECT edge_type, COUNT(*) as c FROM graph_edges GROUP BY edge_type"
        ).fetchall():
            stats.edges_by_type[row["edge_type"]] = row["c"]

        stats.duplicate_clusters = conn.execute(
            "SELECT COUNT(*) as c FROM graph_nodes WHERE node_type = 'duplicate_cluster'"
        ).fetchone()["c"]

        if stats.total_nodes > 0:
            degree_sum = conn.execute(
                """SELECT COUNT(*) as c FROM (
                    SELECT source_node_id FROM graph_edges
                    UNION ALL
                    SELECT target_node_id FROM graph_edges
                )"""
            ).fetchone()["c"]
            stats.average_degree = degree_sum / stats.total_nodes

        stats.connected_components = _estimate_connected_components(conn)

        return stats
    finally:
        conn.close()


def _estimate_connected_components(conn) -> int:
    node_ids = [
        r["node_id"]
        for r in conn.execute("SELECT node_id FROM graph_nodes").fetchall()
    ]
    if not node_ids:
        return 0
    node_set = set(node_ids)
    visited: set[str] = set()
    components = 0

    edge_pairs = conn.execute(
        "SELECT source_node_id, target_node_id FROM graph_edges"
    ).fetchall()
    adj: dict[str, set[str]] = {}
    for src, tgt in edge_pairs:
        adj.setdefault(src, set()).add(tgt)
        adj.setdefault(tgt, set()).add(src)

    for nid in node_ids:
        if nid in visited:
            continue
        components += 1
        stack = [nid]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for neighbor in adj.get(cur, set()):
                if neighbor not in visited:
                    stack.append(neighbor)

    return components
