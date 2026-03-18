from .base import VectorStore
from .qdrant import QdrantVectorStore

_REGISTRY: dict[str, type[VectorStore]] = {
    "qdrant": QdrantVectorStore,
}


def create_vector_store(db_type: str, url: str, collection_name: str) -> VectorStore:
    cls = _REGISTRY.get(db_type)
    if cls is None:
        supported = ", ".join(_REGISTRY.keys())
        raise ValueError(f"Unknown vector DB type '{db_type}'. Supported: {supported}")
    return cls(url=url, collection_name=collection_name)
