from opentelemetry import trace
from pymilvus import MilvusClient

from .base import Document, VectorStore

tracer = trace.get_tracer("vectordb")


class MilvusVectorStore(VectorStore):
    def __init__(self, url: str, collection_name: str, username: str = "", password: str = ""):
        super().__init__(url, collection_name, username, password)
        connect_kwargs: dict = {"uri": url}
        if username and password:
            connect_kwargs["user"] = username
            connect_kwargs["password"] = password
        self._client = MilvusClient(**connect_kwargs)

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[Document]:
        with tracer.start_as_current_span(
            f"search {self._collection_name}",
            kind=trace.SpanKind.CLIENT,
            attributes={
                "db.system": "milvus",
                "db.operation.name": "search",
                "db.collection.name": self._collection_name,
            },
        ) as span:
            try:
                results = self._client.search(
                    collection_name=self._collection_name,
                    data=[query_embedding],
                    limit=top_k,
                    output_fields=["content", "metadata"],
                    search_params={"metric_type": "COSINE"},
                )
                docs = []
                for hit in results[0]:
                    entity = hit.get("entity", {})
                    docs.append(Document(
                        id=str(hit["id"]),
                        content=entity.get("content", ""),
                        metadata=entity.get("metadata", {}),
                        score=hit["distance"],
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
                "db.system": "milvus",
                "db.operation.name": "upsert",
                "db.collection.name": self._collection_name,
            },
        ) as span:
            try:
                data = [
                    {
                        "id": idx,
                        "vector": doc.embedding,
                        "content": doc.content,
                        "metadata": doc.metadata,
                    }
                    for idx, doc in enumerate(documents)
                ]
                self._client.upsert(
                    collection_name=self._collection_name,
                    data=data,
                )
                span.set_attribute("db.operation.batch_size", len(documents))
            except Exception as e:
                span.set_status(trace.StatusCode.ERROR, str(e))
                span.set_attribute("error.type", type(e).__name__)
                raise

    def health(self) -> bool:
        try:
            self._client.list_collections()
            return True
        except Exception:
            return False

    def collection_exists(self) -> bool:
        try:
            return self._client.has_collection(self._collection_name)
        except Exception:
            return False

    def create_collection(self, vector_size: int) -> None:
        self._client.create_collection(
            collection_name=self._collection_name,
            dimension=vector_size,
            metric_type="COSINE",
        )

    def count(self) -> int:
        stats = self._client.get_collection_stats(self._collection_name)
        return stats["row_count"]
