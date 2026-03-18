from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Document:
    id: str
    content: str
    embedding: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


class VectorStore(ABC):
    def __init__(self, url: str, collection_name: str):
        self._url = url
        self._collection_name = collection_name

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
