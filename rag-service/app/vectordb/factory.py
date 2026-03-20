from .base import VectorStore
from .milvus import MilvusVectorStore
from .opensearch import OpenSearchVectorStore
from .qdrant import QdrantVectorStore

_REGISTRY: dict[str, type[VectorStore]] = {
    "qdrant": QdrantVectorStore,
    "milvus": MilvusVectorStore,
    "opensearch": OpenSearchVectorStore,
}


def create_vector_store(
    db_type: str, url: str, collection_name: str, username: str = "", password: str = "",
) -> VectorStore:
    cls = _REGISTRY.get(db_type)
    if cls is None:
        supported = ", ".join(_REGISTRY.keys())
        raise ValueError(f"Unknown vector DB type '{db_type}'. Supported: {supported}")
    return cls(url=url, collection_name=collection_name, username=username, password=password)
