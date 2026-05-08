import os
import signal
import logging
import threading
import time
from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from app.generated import demo_pb2_grpc
from app.grpc_server import RAGServiceServicer
from app.otel_setup import setup_otel, shutdown_otel
from app.vectordb import create_vector_store, VectorStore
from app.seed_data import seed_if_needed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SEED_RETRY_INITIAL_DELAY = 5
SEED_RETRY_MAX_DELAY = 60


def _seed_with_retry(
    store: VectorStore,
    embedding_base_url: str,
    embedding_model: str,
    provider: str,
) -> None:
    delay = SEED_RETRY_INITIAL_DELAY
    while True:
        try:
            seed_if_needed(
                store=store,
                embedding_base_url=embedding_base_url,
                embedding_model=embedding_model,
                provider=provider,
            )
            return
        except Exception:
            logger.exception("Seeding failed; retrying in %ds", delay)
            time.sleep(delay)
            delay = min(delay * 2, SEED_RETRY_MAX_DELAY)


def serve():
    service_name = os.environ.get("OTEL_SERVICE_NAME", "rag-service")
    tracer_provider, meter_provider = setup_otel(service_name)

    listen_addr = os.environ.get("GRPC_LISTEN_ADDR", "[::]:50052")

    store = create_vector_store(
        db_type=os.environ.get("VECTOR_DB_TYPE", "qdrant"),
        url=os.environ.get("VECTOR_DB_URL", "http://qdrant:6333"),
        collection_name=os.environ.get("VECTOR_DB_COLLECTION", "demo-docs"),
        username=os.environ.get("VECTOR_DB_USERNAME", ""),
        password=os.environ.get("VECTOR_DB_PASSWORD", ""),
    )

    threading.Thread(
        target=_seed_with_retry,
        kwargs={
            "store": store,
            "embedding_base_url": os.environ.get("EMBEDDING_BASE_URL", "http://ollama:11434"),
            "embedding_model": os.environ.get("EMBEDDING_MODEL", "nomic-embed-text"),
            "provider": os.environ.get("LLM_PROVIDER", "ollama"),
        },
        daemon=True,
        name="seed-retry",
    ).start()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    demo_pb2_grpc.add_RAGServiceServicer_to_server(RAGServiceServicer(store), server)

    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set("demo.RAGService", health_pb2.HealthCheckResponse.SERVING)

    server.add_insecure_port(listen_addr)
    server.start()
    logger.info(f"RAG service listening on {listen_addr}")

    def shutdown(signum, frame):
        logger.info("Shutting down...")
        server.stop(grace=5)
        shutdown_otel(tracer_provider, meter_provider)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
