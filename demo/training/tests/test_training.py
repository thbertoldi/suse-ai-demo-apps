"""Run real CPU training against local MLflow/OTLP protocol fixtures."""
from concurrent.futures import ThreadPoolExecutor
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import unittest

import grpc
from opentelemetry.proto.collector.metrics.v1 import metrics_service_pb2, metrics_service_pb2_grpc
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2, trace_service_pb2_grpc


class MLflow(http.server.BaseHTTPRequestHandler):
    artifact = b""
    updates = []
    fail_upload = False

    def log_message(self, *args):
        pass

    def reply(self, value, status=200, raw=False):
        payload = value if raw else json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.startswith("/api/2.0/mlflow-artifacts/"):
            return self.reply(self.artifact, raw=True)
        self.reply({"experiment": {"experiment_id": "1"}})

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("runs/create"):
            return self.reply({"run": {"info": {"run_id": "fixture-run", "artifact_uri": "mlflow-artifacts:/1/fixture-run/artifacts"}}})
        if self.path.endswith("runs/update"):
            self.updates.append(data["status"])
        self.reply({})

    def do_PUT(self):
        type(self).artifact = self.rfile.read(int(self.headers["Content-Length"]))
        self.reply({}, status=503 if self.fail_upload else 200)


class MetricReceiver(metrics_service_pb2_grpc.MetricsServiceServicer):
    def __init__(self):
        self.requests = []

    def Export(self, request, context):
        self.requests.append(request)
        return metrics_service_pb2.ExportMetricsServiceResponse()


class TraceReceiver(trace_service_pb2_grpc.TraceServiceServicer):
    def __init__(self):
        self.requests = []

    def Export(self, request, context):
        self.requests.append(request)
        return trace_service_pb2.ExportTraceServiceResponse()


class TrainingTests(unittest.TestCase):
    def setUp(self):
        MLflow.artifact, MLflow.updates, MLflow.fail_upload = b"", [], False
        self.http = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MLflow)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.metrics, self.traces = MetricReceiver(), TraceReceiver()
        self.grpc = grpc.server(ThreadPoolExecutor(max_workers=2))
        metrics_service_pb2_grpc.add_MetricsServiceServicer_to_server(self.metrics, self.grpc)
        trace_service_pb2_grpc.add_TraceServiceServicer_to_server(self.traces, self.grpc)
        self.port = self.grpc.add_insecure_port("127.0.0.1:0")
        self.grpc.start()

    def tearDown(self):
        self.grpc.stop(0).wait()
        self.http.shutdown()
        self.http.server_close()
        self.thread.join()

    def run_training(self):
        env = {**os.environ, "MLFLOW_URL": f"http://127.0.0.1:{self.http.server_port}",
               "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{self.port}",
               "OTEL_RESOURCE_ATTRIBUTES": "service.namespace=training-test,service.instance.id=fixture-pod"}
        return subprocess.run([sys.executable, str(Path(__file__).parents[1] / "train.py")],
                              env=env, capture_output=True, text=True, timeout=60)

    def test_real_training_upload_download_and_otlp(self):
        result = self.run_training()
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout.strip())
        self.assertEqual(summary["epochs"], 200)
        self.assertLess(summary["loss"], 0.001)
        self.assertGreater(len(MLflow.artifact), 1000)
        self.assertEqual(summary["artifact_bytes"], len(MLflow.artifact))
        self.assertEqual(MLflow.updates, ["FINISHED"])
        metrics = [metric for req in self.metrics.requests for resource in req.resource_metrics
                   for scope in resource.scope_metrics for metric in scope.metrics]
        epochs = next(metric for metric in metrics if metric.name == "pytorch.training.epochs")
        self.assertEqual(epochs.sum.data_points[0].as_int, 200)
        spans = [span for req in self.traces.requests for resource in req.resource_spans
                 for scope in resource.scope_spans for span in scope.spans]
        self.assertEqual(sum(span.name == "training epoch" for span in spans), 200)

    def test_failed_artifact_upload_marks_run_failed_and_exits_nonzero(self):
        MLflow.fail_upload = True
        result = self.run_training()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(MLflow.updates, ["FAILED"])
        self.assertNotIn("training_completed", result.stdout)
