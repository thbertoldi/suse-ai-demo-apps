import logging
from app.embedding_client import embed
from app.vectordb.base import Document, VectorStore

logger = logging.getLogger(__name__)

SEED_DOCUMENTS = [
    {
        "content": "A Kubernetes pod is the smallest deployable unit in Kubernetes. A pod represents a single instance of a running process in your cluster. Pods contain one or more containers, such as Docker containers. When a pod runs multiple containers, the containers are managed as a single entity and share the pod's resources.",
        "metadata": {"topic": "kubernetes", "subtopic": "pods"},
    },
    {
        "content": "A Kubernetes Deployment provides declarative updates for Pods and ReplicaSets. You describe a desired state in a Deployment, and the Deployment controller changes the actual state to the desired state at a controlled rate. You can define Deployments to create new ReplicaSets, or to remove existing Deployments and adopt all their resources with new Deployments.",
        "metadata": {"topic": "kubernetes", "subtopic": "deployments"},
    },
    {
        "content": "A Kubernetes Service is an abstraction which defines a logical set of Pods and a policy by which to access them. Services enable loose coupling between dependent Pods. A Service is defined using YAML or JSON, like all Kubernetes objects.",
        "metadata": {"topic": "kubernetes", "subtopic": "services"},
    },
    {
        "content": "Linux containers are a technology that allows you to package and isolate applications with their entire runtime environment. This makes it easy to move the contained application between environments while retaining full functionality. Containers share the host OS kernel and therefore do not require an OS per application.",
        "metadata": {"topic": "containers", "subtopic": "basics"},
    },
    {
        "content": "A container runtime is the software responsible for running containers. It manages the complete lifecycle of containers including image transfer, storage, execution, supervision, and networking. Common container runtimes include containerd, CRI-O, and Docker Engine.",
        "metadata": {"topic": "containers", "subtopic": "runtime"},
    },
    {
        "content": "Container images are lightweight, standalone, executable packages that include everything needed to run a piece of software, including the code, runtime, system tools, libraries, and settings. Images are built from Dockerfiles and stored in container registries.",
        "metadata": {"topic": "containers", "subtopic": "images"},
    },
    {
        "content": "OpenTelemetry is a collection of APIs, SDKs, and tools used to instrument, generate, collect, and export telemetry data (metrics, logs, and traces) to help you analyze your software's performance and behavior. It is a CNCF project and provides a vendor-neutral specification.",
        "metadata": {"topic": "observability", "subtopic": "opentelemetry"},
    },
    {
        "content": "Distributed tracing is a method used to profile and monitor applications built using a microservices architecture. It helps pinpoint where failures occur and what causes poor performance by tracking requests as they flow through distributed systems.",
        "metadata": {"topic": "observability", "subtopic": "tracing"},
    },
]


def seed_if_needed(
    store: VectorStore,
    embedding_base_url: str,
    embedding_model: str,
    provider: str,
) -> None:
    if store.collection_exists() and store.count() > 0:
        logger.info("Collection already populated, skipping seed")
        return

    logger.info("Seeding vector database with sample documents...")

    documents = []
    for i, doc_data in enumerate(SEED_DOCUMENTS):
        embedding = embed(
            base_url=embedding_base_url,
            model=embedding_model,
            provider=provider,
            text=doc_data["content"],
        )
        documents.append(Document(
            id=str(i),
            content=doc_data["content"],
            embedding=embedding,
            metadata=doc_data["metadata"],
        ))

    if not store.collection_exists():
        vector_size = len(documents[0].embedding)
        store.create_collection(vector_size)
        logger.info(f"Created collection with vector_size={vector_size}")

    store.upsert(documents)
    logger.info(f"Seeded {len(documents)} documents")
