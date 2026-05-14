import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from vidcrawl.db import get_db


@dataclass
class GraphNodeInfo:
    node_id: str
    node_type: str
    ref_id: Optional[str] = None
    label: str = ""
    metadata: Optional[dict] = None
    created_at: Optional[str] = None


@dataclass
class GraphEdgeInfo:
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str
    weight: float = 1.0
    metadata: Optional[dict] = None
    created_at: Optional[str] = None


@dataclass
class GraphContext:
    entities: list[str] = field(default_factory=list)
    idea_count: int = 0
    evidence_modalities: list[str] = field(default_factory=list)
    cluster_info: Optional[str] = None
    variant_count: int = 0
    duplicate_count: int = 0
    related_moment_count: int = 0


def get_node(
    db_path: Path,
    node_or_ref_id: str,
) -> Optional[GraphNodeInfo]:
    conn = get_db(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM graph_nodes WHERE node_id = ? OR ref_id = ?",
            (node_or_ref_id, node_or_ref_id),
        ).fetchone()
        if row is None:
            return None
        return GraphNodeInfo(
            node_id=row["node_id"],
            node_type=row["node_type"],
            ref_id=row["ref_id"],
            label=row["label"],
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
            created_at=row["created_at"],
        )
    finally:
        conn.close()


def get_neighbors(
    db_path: Path,
    node_or_ref_id: str,
    max_distance: int = 1,
) -> dict[str, Any]:
    conn = get_db(db_path)
    try:
        node_row = conn.execute(
            "SELECT * FROM graph_nodes WHERE node_id = ? OR ref_id = ?",
            (node_or_ref_id, node_or_ref_id),
        ).fetchone()
        if node_row is None:
            return {}

        nid = node_row["node_id"]

        edges = conn.execute(
            """SELECT * FROM graph_edges
               WHERE source_node_id = ? OR target_node_id = ?
               ORDER BY edge_type""",
            (nid, nid),
        ).fetchall()

        neighbor_ids: set[str] = set()
        edge_list: list[dict] = []
        for e in edges:
            other_id = e["target_node_id"] if e["source_node_id"] == nid else e["source_node_id"]
            if other_id != nid:
                neighbor_ids.add(other_id)
            edge_list.append({
                "edge_id": e["edge_id"],
                "source_node_id": e["source_node_id"],
                "target_node_id": e["target_node_id"],
                "edge_type": e["edge_type"],
                "weight": e["weight"],
            })

        neighbor_nodes = []
        for nb_id in neighbor_ids:
            nb = conn.execute(
                "SELECT * FROM graph_nodes WHERE node_id = ?", (nb_id,)
            ).fetchone()
            if nb:
                neighbor_nodes.append(GraphNodeInfo(
                    node_id=nb["node_id"],
                    node_type=nb["node_type"],
                    ref_id=nb["ref_id"],
                    label=nb["label"],
                    metadata=json.loads(nb["metadata_json"]) if nb["metadata_json"] else {},
                    created_at=nb["created_at"],
                ))

        return {
            "node": GraphNodeInfo(
                node_id=node_row["node_id"],
                node_type=node_row["node_type"],
                ref_id=node_row["ref_id"],
                label=node_row["label"],
                metadata=json.loads(node_row["metadata_json"]) if node_row["metadata_json"] else {},
                created_at=node_row["created_at"],
            ),
            "edges": edge_list,
            "neighbors": neighbor_nodes,
            "degree": len(neighbor_ids),
        }
    finally:
        conn.close()


def get_graph_context_for_moment(
    db_path: Path,
    moment_id: str,
) -> GraphContext:
    conn = get_db(db_path)
    try:
        ctx = GraphContext()

        moment_nid = f"moment:{moment_id}"

        edges = conn.execute(
            "SELECT * FROM graph_edges WHERE source_node_id = ? OR target_node_id = ?",
            (moment_nid, moment_nid),
        ).fetchall()

        entity_labels: set[str] = set()
        idea_ids: set[str] = set()
        evidence_modalities: set[str] = set()
        related_moments: set[str] = set()
        has_cluster = False
        variant_count = 0
        duplicate_count = 0

        for e in edges:
            other_nid = e["target_node_id"] if e["source_node_id"] == moment_nid else e["source_node_id"]
            if other_nid == moment_nid:
                continue

            if e["edge_type"] in ("moment_duplicate_of",):
                duplicate_count += 1
            elif e["edge_type"] in ("moment_variant_of", "moment_same_idea_as"):
                variant_count += 1

            if other_nid.startswith("idea:"):
                idea_ids.add(other_nid)
            elif other_nid.startswith("entity:"):
                entity_labels.add(other_nid.split(":", 1)[1] if ":" in other_nid else other_nid)
            elif other_nid.startswith("evidence:"):
                ev = conn.execute(
                    "SELECT metadata_json FROM graph_nodes WHERE node_id = ?",
                    (other_nid,),
                ).fetchone()
                if ev:
                    ev_meta = json.loads(ev["metadata_json"]) if ev["metadata_json"] else {}
                    if "modality" in ev_meta:
                        evidence_modalities.add(ev_meta["modality"])

            if other_nid.startswith("duplicate_cluster:"):
                has_cluster = True

            if other_nid.startswith("moment:") and other_nid != moment_nid:
                related_moments.add(other_nid)

        ctx.entities = sorted(entity_labels)
        ctx.idea_count = len(idea_ids)
        ctx.evidence_modalities = sorted(evidence_modalities)
        ctx.related_moment_count = len(related_moments)
        ctx.duplicate_count = duplicate_count
        ctx.variant_count = variant_count

        if has_cluster:
            total_dup = duplicate_count + variant_count
            cluster_parts = []
            if duplicate_count:
                cluster_parts.append(f"{duplicate_count} duplicate")
            if variant_count:
                cluster_parts.append(f"{variant_count} variant")
            ctx.cluster_info = f"{', '.join(cluster_parts)} moment(s)"

        return ctx
    finally:
        conn.close()
