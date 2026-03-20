from opensearchpy import OpenSearch

from opentelemetry import trace

from .base import Document, VectorStore

tracer = trace.get_tracer("vectordb")


class OpenSearchVectorStore(VectorStore):
    def __init__(self, url: str, collection_name: str, username: str = "", password: str = ""):
        super().__init__(url, collection_name, username, password)
        connect_kwargs: dict = {
            "hosts": [url],
            "use_ssl": url.startswith("https"),
            "verify_certs": url.startswith("https"),
            "ssl_show_warn": False,
        }
        if username and password:
            connect_kwargs["http_auth"] = (username, password)
        self._client = OpenSearch(**connect_kwargs)

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[Document]:
        with tracer.start_as_current_span(
            f"search {self._collection_name}",
            kind=trace.SpanKind.CLIENT,
            attributes={
                "db.system": "opensearch",
                "db.operation.name": "search",
                "db.collection.name": self._collection_name,
            },
        ) as span:
            try:
                body = {
                    "size": top_k,
                    "query": {
                        "knn": {
                            "embedding": {
                                "vector": query_embedding,
                                "k": top_k,
                            }
                        }
                    },
                }
                response = self._client.search(index=self._collection_name, body=body)
                docs = []
                for hit in response["hits"]["hits"]:
                    source = hit["_source"]
                    docs.append(Document(
                        id=hit["_id"],
                        content=source.get("content", ""),
                        metadata=source.get("metadata", {}),
                        score=hit["_score"],
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
                "db.system": "opensearch",
                "db.operation.name": "upsert",
                "db.collection.name": self._collection_name,
            },
        ) as span:
            try:
                bulk_body: list[dict] = []
                for idx, doc in enumerate(documents):
                    bulk_body.append({"index": {"_index": self._collection_name, "_id": str(idx)}})
                    bulk_body.append({
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "embedding": doc.embedding,
                    })
                self._client.bulk(body=bulk_body, refresh=True)
                span.set_attribute("db.operation.batch_size", len(documents))
            except Exception as e:
                span.set_status(trace.StatusCode.ERROR, str(e))
                span.set_attribute("error.type", type(e).__name__)
                raise

    def health(self) -> bool:
        try:
            return self._client.ping()
        except Exception:
            return False

    def collection_exists(self) -> bool:
        try:
            return self._client.indices.exists(index=self._collection_name)
        except Exception:
            return False

    def create_collection(self, vector_size: int) -> None:
        body = {
            "settings": {
                "index": {
                    "knn": True,
                }
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": vector_size,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                        },
                    },
                    "content": {"type": "text"},
                    "metadata": {"type": "object", "enabled": False},
                }
            },
        }
        self._client.indices.create(index=self._collection_name, body=body)

    def count(self) -> int:
        result = self._client.count(index=self._collection_name)
        return result["count"]
