from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass
class Document:
    id: str
    content: str
    embedding: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


class VectorStore(ABC):
    def __init__(self, url: str, collection_name: str, username: str = "", password: str = ""):
        self._url = url
        self._collection_name = collection_name
        self._username = username
        self._password = password

    def span_attributes(self, system: str, operation: str) -> dict:
        """Describe the actual client endpoint, without URL credentials or paths."""
        endpoint = urlsplit(self._url if "://" in self._url else "//" + self._url)
        attributes = {
            "db.system.name": system,
            "db.system": system,  # Compatibility for older consumers.
            "db.operation.name": operation,
            "db.collection.name": self._collection_name,
        }
        if endpoint.hostname:
            attributes["server.address"] = endpoint.hostname
        if endpoint.port is not None:
            attributes["server.port"] = endpoint.port
        return attributes

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 3) -> list[Document]:
        ...

    @abstractmethod
    def upsert(self, documents: list[Document]) -> None:
        ...

    @abstractmethod
    def health(self) -> bool:
        ...

    @abstractmethod
    def collection_exists(self) -> bool:
        ...

    @abstractmethod
    def create_collection(self, vector_size: int) -> None:
        ...

    @abstractmethod
    def count(self) -> int:
        ...
