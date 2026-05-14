import hashlib
from abc import ABC, abstractmethod
from typing import Optional


class EmbeddingProvider(ABC):
    @abstractmethod
    def compute(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def is_available(self) -> bool:
        return True


class NullEmbeddingProvider(EmbeddingProvider):
    def compute(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]

    @property
    def dimension(self) -> int:
        return 0

    @property
    def name(self) -> str:
        return "null"

    def is_available(self) -> bool:
        return True


class HashEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimension: int = 64):
        self._dimension = dimension

    def compute(self, texts: list[str]) -> list[list[float]]:
        result = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).hexdigest()
            vec = []
            for i in range(self._dimension):
                idx = i % len(h)
                val = int(h[idx], 16) / 15.0
                vec.append(val)
            result.append(vec)
        return result

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return "hash"


_SENTENCE_TRANSFORMERS_AVAILABLE = False
_ST_PROVIDER = None


def get_sentence_transformer_provider(model_name: str = "all-MiniLM-L6-v2") -> Optional[EmbeddingProvider]:
    global _SENTENCE_TRANSFORMERS_AVAILABLE, _ST_PROVIDER
    if _ST_PROVIDER is not None:
        return _ST_PROVIDER
    try:
        import sentence_transformers  # noqa: F401
        _SENTENCE_TRANSFORMERS_AVAILABLE = True
    except ImportError:
        return None
    return _SentenceTransformerProvider(model_name)


class _SentenceTransformerProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        import sentence_transformers
        self._model = sentence_transformers.SentenceTransformer(model_name)
        self._name = model_name

    def compute(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    @property
    def name(self) -> str:
        return f"sentence-transformers:{self._name}"

    def is_available(self) -> bool:
        return True


def get_provider(name: str = "null", **kwargs) -> EmbeddingProvider:
    if name == "null":
        return NullEmbeddingProvider()
    elif name == "hash":
        return HashEmbeddingProvider(dimension=kwargs.get("dimension", 64))
    elif name.startswith("sentence-transformers"):
        model = kwargs.get("model", "all-MiniLM-L6-v2")
        provider = get_sentence_transformer_provider(model)
        if provider is None:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install with: pip install sentence-transformers"
            )
        return provider
    else:
        raise ValueError(f"Unknown embedding provider: {name}")
