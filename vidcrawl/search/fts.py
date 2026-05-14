def rebuild_fts_index(db_path: str) -> None:
    from vidcrawl.db import get_db, init_db, rebuild_fts
    with get_db(db_path) as conn:
        init_db(conn)
        rebuild_fts(conn)
