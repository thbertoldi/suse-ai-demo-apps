import os
import signal
import logging
from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from app.generated import demo_pb2_grpc
from app.grpc_server import AgentServiceServicer
from app.otel_setup import setup_otel, shutdown_otel
from app.agent import create_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def serve():
    service_name = os.environ.get("OTEL_SERVICE_NAME", "agent-service")
    tracer_provider, meter_provider = setup_otel(service_name)

    listen_addr = os.environ.get("GRPC_LISTEN_ADDR", "[::]:50054")
    rag_service_addr = os.environ.get("RAG_SERVICE_ADDR", "rag-service:50052")

    rag_channel = grpc.insecure_channel(rag_service_addr)

    run_agent = create_agent(rag_channel)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    demo_pb2_grpc.add_AgentServiceServicer_to_server(AgentServiceServicer(run_agent), server)

    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set("demo.AgentService", health_pb2.HealthCheckResponse.SERVING)

    server.add_insecure_port(listen_addr)
    server.start()
    logger.info(f"Agent service listening on {listen_addr}")

    def shutdown(signum, frame):
        logger.info("Shutting down...")
        server.stop(grace=5)
        rag_channel.close()
        shutdown_otel(tracer_provider, meter_provider)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
