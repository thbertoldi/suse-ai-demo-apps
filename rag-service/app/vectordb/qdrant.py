from opentelemetry import trace
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from .base import Document, VectorStore

tracer = trace.get_tracer("vectordb")


class QdrantVectorStore(VectorStore):
    def __init__(self, url: str, collection_name: str):
        super().__init__(url, collection_name)
        self._client = QdrantClient(url=url)

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[Document]:
        with tracer.start_as_current_span(
            f"search {self._collection_name}",
            kind=trace.SpanKind.CLIENT,
            attributes={
                "db.system": "qdrant",
                "db.operation.name": "search",
                "db.collection.name": self._collection_name,
            },
        ) as span:
            try:
                results = self._client.query_points(
                    collection_name=self._collection_name,
                    query=query_embedding,
                    limit=top_k,
                    with_payload=True,
                )
                docs = []
                for point in results.points:
                    payload = point.payload or {}
                    docs.append(Document(
                        id=str(point.id),
                        content=payload.get("content", ""),
                        metadata=payload.get("metadata", {}),
                        score=point.score,
                    ))
                span.set_attribute("db.response.returned_rows", len(docs))
                return docs
            except Exception as e:
                span.set_status(trace.StatusCode.ERROR, str(e))
                span.set_attribute("error.type", type(e).__name__)
                raise

    def upsert(self, documents: list[Document]) -> None:
        with tracer.start_as_current_span(
            f"upsert {self._collection_name}",
            kind=trace.SpanKind.CLIENT,
            attributes={
                "db.system": "qdrant",
                "db.operation.name": "upsert",
                "db.collection.name": self._collection_name,
            },
        ) as span:
            try:
                points = [
                    PointStruct(
                        id=idx,
                        vector=doc.embedding,
                        payload={"content": doc.content, "metadata": doc.metadata},
                    )
                    for idx, doc in enumerate(documents)
                ]
                self._client.upsert(
                    collection_name=self._collection_name,
                    points=points,
                )
                span.set_attribute("db.operation.batch_size", len(documents))
            except Exception as e:
                span.set_status(trace.StatusCode.ERROR, str(e))
                span.set_attribute("error.type", type(e).__name__)
                raise

    def health(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False

    def collection_exists(self) -> bool:
        try:
            self._client.get_collection(self._collection_name)
            return True
        except Exception:
            return False

    def create_collection(self, vector_size: int) -> None:
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def count(self) -> int:
        info = self._client.get_collection(self._collection_name)
        return info.points_count
