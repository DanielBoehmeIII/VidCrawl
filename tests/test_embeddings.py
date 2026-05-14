import json
import tempfile
from pathlib import Path

import pytest

from vidcrawl.db import get_db, init_db
from vidcrawl.embeddings.provider import (
    NullEmbeddingProvider,
    HashEmbeddingProvider,
    get_provider,
)
from vidcrawl.embeddings.similarity import cosine_similarity, cosine_similarity_matrix
from vidcrawl.embeddings.store import (
    pack_vector,
    unpack_vector,
    store_vectors,
    get_vector,
    get_all_vectors,
    build_embeddings,
    get_embedding_stats,
    has_embeddings,
    init_embedding_tables,
)
from vidcrawl.demo import create_demo_corpus


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    conn = get_db(path)
    init_db(conn)
    conn.close()
    yield Path(path)
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def demo_db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    create_demo_corpus(Path(path))
    yield Path(path)
    Path(path).unlink(missing_ok=True)


# ---- Provider Tests ----

class TestEmbeddingProviders:

    def test_null_provider(self):
        p = NullEmbeddingProvider()
        assert p.name == "null"
        assert p.dimension == 0
        assert p.is_available()
        assert p.compute(["hello"]) == [[]]

    def test_hash_provider(self):
        p = HashEmbeddingProvider(dimension=64)
        assert p.name == "hash"
        assert p.dimension == 64
        vecs = p.compute(["hello", "world"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 64
        assert len(vecs[1]) == 64
        assert all(0 <= v <= 1 for v in vecs[0])

    def test_hash_deterministic(self):
        p = HashEmbeddingProvider(dimension=32)
        v1 = p.compute(["same text"])
        v2 = p.compute(["same text"])
        assert v1 == v2

    def test_hash_different(self):
        p = HashEmbeddingProvider(dimension=32)
        v1 = p.compute(["text a"])
        v2 = p.compute(["text b"])
        assert v1 != v2

    def test_get_provider_null(self):
        p = get_provider("null")
        assert isinstance(p, NullEmbeddingProvider)

    def test_get_provider_hash(self):
        p = get_provider("hash", dimension=128)
        assert isinstance(p, HashEmbeddingProvider)
        assert p.dimension == 128

    def test_get_provider_sentence_transformers_not_installed(self):
        with pytest.raises(ImportError):
            get_provider("sentence-transformers", model="all-MiniLM-L6-v2")


# ---- Similarity Tests ----

class TestSimilarity:

    def test_cosine_identical(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_empty(self):
        assert cosine_similarity([], []) == 0.0

    def test_cosine_mismatched_dim(self):
        assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_cosine_matrix(self):
        vectors = {
            "a": [1.0, 0.0],
            "b": [0.0, 1.0],
            "c": [0.5, 0.5],
        }
        query = [1.0, 0.0]
        results = cosine_similarity_matrix(query, vectors, top_k=2)
        assert len(results) == 2
        assert results[0][0] == "a"
        assert results[0][1] == pytest.approx(1.0)


# ---- Store Tests ----

class TestEmbeddingStore:

    def test_pack_unpack(self):
        vec = [0.1, 0.2, 0.3, -0.5, 1.0]
        blob = pack_vector(vec)
        restored = unpack_vector(blob)
        assert restored == pytest.approx(vec)

    def test_store_and_retrieve(self, db_path):
        conn = get_db(db_path)
        init_embedding_tables(conn)
        store_vectors(conn, "test_run", "moment", [("m1", [1.0, 0.0]), ("m2", [0.0, 1.0])])
        conn.commit()

        v = get_vector(conn, "moment", "m1")
        assert v is not None
        assert v[0] == pytest.approx(1.0)

        v = get_vector(conn, "moment", "nonexistent")
        assert v is None
        conn.close()

    def test_get_all_vectors(self, db_path):
        conn = get_db(db_path)
        init_embedding_tables(conn)
        store_vectors(conn, "test_run", "moment", [("m1", [1.0, 0.0]), ("m2", [0.0, 1.0])])
        conn.commit()

        all_v = get_all_vectors(conn, "moment")
        assert len(all_v) == 2
        assert "m1" in all_v
        conn.close()


# ---- Build / Stats Tests ----

class TestBuildEmbeddings:

    def test_build_embeddings_hash(self, demo_db_path):
        result = build_embeddings(str(demo_db_path), provider_name="hash", dimension=32)
        assert result["vectors_stored"] > 0
        assert result["provider"] == "hash"
        assert result["dimension"] == 32

    def test_has_embeddings(self, demo_db_path):
        assert has_embeddings(str(demo_db_path)) is False
        build_embeddings(str(demo_db_path), provider_name="hash", dimension=32)
        assert has_embeddings(str(demo_db_path)) is True

    def test_embedding_stats(self, demo_db_path):
        stats = get_embedding_stats(str(demo_db_path))
        assert stats["has_embeddings"] is False

        build_embeddings(str(demo_db_path), provider_name="hash", dimension=32)
        stats = get_embedding_stats(str(demo_db_path))
        assert stats["has_embeddings"] is True
        assert stats["vector_count"] > 0
        assert stats["provider"] == "hash"


# ---- CLI Tests ----

class TestEmbedCLI:

    def test_embed_build_cli(self, demo_db_path, tmp_path):
        from typer.testing import CliRunner
        from vidcrawl.cli import app
        runner = CliRunner()
        result = runner.invoke(app, [
            "embed", "build", "--provider", "hash",
            "--data-dir", str(tmp_path),
        ])
        assert result.exit_code == 0

    def test_embed_stats_cli(self, demo_db_path, tmp_path):
        from typer.testing import CliRunner
        from vidcrawl.cli import app
        runner = CliRunner()
        result = runner.invoke(app, [
            "embed", "stats",
            "--data-dir", str(tmp_path),
        ])
        assert result.exit_code == 0
