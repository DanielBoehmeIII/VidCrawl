import tempfile
from pathlib import Path

import pytest

from vidcrawl.db import get_db, init_db


@pytest.fixture
def tmp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    conn = get_db(db_path)
    init_db(conn)
    conn.close()
    yield db_path
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def db_conn(tmp_db):
    conn = get_db(tmp_db)
    yield conn
    conn.close()
