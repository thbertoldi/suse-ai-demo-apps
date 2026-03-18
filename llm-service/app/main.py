import os
import signal
import logging
from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from app.generated import demo_pb2_grpc
from app.grpc_server import LLMServiceServicer
from app.otel_setup import setup_otel, shutdown_otel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def serve():
    service_name = os.environ.get("OTEL_SERVICE_NAME", "llm-service")
    tracer_provider, meter_provider = setup_otel(service_name)

    listen_addr = os.environ.get("GRPC_LISTEN_ADDR", "[::]:50053")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    demo_pb2_grpc.add_LLMServiceServicer_to_server(LLMServiceServicer(), server)

    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set("demo.LLMService", health_pb2.HealthCheckResponse.SERVING)

    server.add_insecure_port(listen_addr)
    server.start()
    logger.info(f"LLM service listening on {listen_addr}")

    def shutdown(signum, frame):
        logger.info("Shutting down...")
        server.stop(grace=5)
        shutdown_otel(tracer_provider, meter_provider)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
