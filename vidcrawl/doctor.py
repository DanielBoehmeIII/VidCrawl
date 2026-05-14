import time
from pathlib import Path
from typing import Any

from vidcrawl.config import get_config
from vidcrawl.db import get_db


def run_doctor(data_dir: str = "data") -> dict[str, Any]:
    config = get_config(data_dir)
    report: dict[str, Any] = {
        "database": {"path": str(config.db_path), "exists": config.db_path.exists()},
        "counts": {},
        "index_health": {},
        "graph": {"built": False, "node_count": 0},
        "embeddings": {"built": False, "count": 0},
        "optional_tools": {},
    }

    for tool in ["ffmpeg", "tesseract", "whisper", "yt-dlp", "sentence-transformers"]:
        report["optional_tools"][tool] = _check_tool(tool)

    if not config.db_path.exists():
        return report

    conn = get_db(config.db_path)
    try:
        tables = [r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]

        for table in ["videos", "moments", "modal_evidence", "ideas", "duplicates", "graph_nodes", "graph_edges", "claims", "embedding_vectors", "freshness_scores"]:
            if table in tables:
                count = conn.execute(f"SELECT COUNT(*) as c FROM {table}").fetchone()["c"]
                report["counts"][table] = count

        if "graph_nodes" in tables:
            nc = conn.execute("SELECT COUNT(*) as c FROM graph_nodes").fetchone()["c"]
            report["graph"] = {"built": nc > 0, "node_count": nc}

        if "embedding_vectors" in tables:
            ec = conn.execute("SELECT COUNT(*) as c FROM embedding_vectors").fetchone()["c"]
            report["embeddings"] = {"built": ec > 0, "count": ec}

        fts_count = 0
        try:
            fts_count = conn.execute(
                "SELECT COUNT(*) as c FROM moments_fts"
            ).fetchone()["c"]
        except Exception:
            pass
        report["fts_row_count"] = fts_count

        indexes = [r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()]
        report["index_count"] = len(indexes)
    finally:
        conn.close()

    return report


def _check_tool(name: str) -> dict:
    if name == "ffmpeg":
        import shutil
        return {"available": shutil.which("ffmpeg") is not None}
    elif name == "tesseract":
        import shutil
        return {"available": shutil.which("tesseract") is not None}
    elif name == "whisper":
        try:
            import whisper  # noqa: F401
            return {"available": True}
        except ImportError:
            return {"available": False}
    elif name == "yt-dlp":
        import shutil
        return {"available": shutil.which("yt-dlp") is not None}
    elif name == "sentence-transformers":
        try:
            import sentence_transformers  # noqa: F401
            return {"available": True}
        except ImportError:
            return {"available": False}
    return {"available": False}


def run_benchmark(data_dir: str = "data") -> dict[str, Any]:
    config = get_config(data_dir)
    results: dict[str, Any] = {}

    if not config.db_path.exists():
        return {"error": "No database found"}

    conn = get_db(config.db_path)
    try:
        start = time.time()
        row = conn.execute("SELECT COUNT(*) as c FROM moments").fetchone()
        moment_count = row["c"] if row else 0

        if "_bench_fts" in [r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]:
            conn.execute("DROP TABLE IF EXISTS _bench_fts")
        conn.execute("CREATE VIRTUAL TABLE _bench_fts USING fts5(content)")
        test_text = "playwright browser test " * 100
        conn.execute("INSERT INTO _bench_fts VALUES (?)", (test_text,))
        fts_start = time.time()
        for _ in range(100):
            conn.execute("SELECT rowid FROM _bench_fts WHERE _bench_fts MATCH ?", ("playwright",)).fetchall()
        fts_latency = (time.time() - fts_start) / 100
        conn.execute("DROP TABLE IF EXISTS _bench_fts")

        conn.execute("CREATE TABLE IF NOT EXISTS _bench_test (id INTEGER PRIMARY KEY, val TEXT)")
        insert_start = time.time()
        for i in range(100):
            conn.execute(
                "INSERT OR IGNORE INTO _bench_test (id, val) VALUES (?, ?)",
                (i, f"test_{i}"),
            )
        insert_time = time.time() - insert_start
        conn.execute("DROP TABLE IF EXISTS _bench_test")

        results["moment_count"] = moment_count
        results["avg_fts_query_ms"] = round(fts_latency * 1000, 3)
        results["avg_insert_us"] = round(insert_time / 100 * 1_000_000, 1)
    finally:
        conn.close()

    return results
