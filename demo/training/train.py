"""Train a real CPU model, publish its checkpoint to MLflow, and export OTLP."""
import io
import json
import os
import time
import logging

import requests
import torch
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def main():
    resource = Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME", "pytorch-training")})
    traces = TracerProvider(resource=resource)
    traces.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(insecure=True)))
    trace.set_tracer_provider(traces)
    meters = MeterProvider(resource=resource, metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter(insecure=True), export_interval_millis=5000)])
    metrics.set_meter_provider(meters)
    RequestsInstrumentor().instrument()
    tracer = trace.get_tracer("suse.ai.demo.training")
    meter = metrics.get_meter("suse.ai.demo.training")
    loss_metric = meter.create_histogram("pytorch.training.loss")
    epochs = meter.create_counter("pytorch.training.epochs", unit="{epoch}")
    runs = meter.create_counter("pytorch.training.runs", unit="{run}")
    duration = meter.create_histogram("pytorch.training.duration", unit="s")
    base = os.environ["MLFLOW_URL"].rstrip("/")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT", "native-pytorch-training")
    token = os.environ.get("MLFLOW_TRACKING_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def api(path, data):
        response = requests.post(base + path, json=data, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()

    run_id = None
    outcome = "error"
    started = time.monotonic()
    try:
        with tracer.start_as_current_span("train linear regression", attributes={"ml.framework": "pytorch", "ml.training.device": "cpu"}) as training:
            lookup = requests.get(base + "/api/2.0/mlflow/experiments/get-by-name", params={"experiment_name": experiment_name}, headers=headers, timeout=30)
            if lookup.status_code == 404:
                experiment = api("/api/2.0/mlflow/experiments/create", {"name": experiment_name})["experiment_id"]
            else:
                lookup.raise_for_status()
                experiment = lookup.json()["experiment"]["experiment_id"]
            info = api("/api/2.0/mlflow/runs/create", {"experiment_id": experiment, "run_name": "linear-regression-cpu"})["run"]["info"]
            run_id = info["run_id"]
            training.set_attribute("mlflow.run_id", run_id)
            torch.set_num_threads(1)
            torch.manual_seed(2026)
            x = torch.linspace(-1, 1, 128).reshape(-1, 1)
            y = 2 * x + 3
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            samples = []
            for epoch in range(200):
                with tracer.start_as_current_span("training epoch", attributes={"ml.training.epoch": epoch}):
                    optimizer.zero_grad()
                    loss = torch.nn.functional.mse_loss(model(x), y)
                    loss.backward()
                    optimizer.step()
                    value = float(loss.detach())
                    loss_metric.record(value)
                    epochs.add(1)
                    if epoch % 10 == 0 or epoch == 199:
                        samples.append({"key": "loss", "value": value, "timestamp": int(time.time() * 1000), "step": epoch})
            if value >= 0.001:
                raise RuntimeError("Training did not converge")
            api("/api/2.0/mlflow/runs/log-batch", {"run_id": run_id, "params": [{"key": "device", "value": "cpu"}, {"key": "epochs", "value": "200"}], "metrics": samples})
            checkpoint = io.BytesIO()
            torch.save({"state_dict": model.state_dict(), "loss": value, "torch_version": torch.__version__}, checkpoint)
            if not info["artifact_uri"].startswith("mlflow-artifacts:/"):
                raise ValueError("MLflow must serve artifacts (run artifact_uri must use mlflow-artifacts:/)")
            artifact = info["artifact_uri"].removeprefix("mlflow-artifacts:") + "/model.pt"
            artifact_url = base + "/api/2.0/mlflow-artifacts/artifacts" + artifact
            uploaded = requests.put(artifact_url, data=checkpoint.getvalue(), headers=headers, timeout=30)
            uploaded.raise_for_status()
            downloaded = requests.get(artifact_url, headers=headers, timeout=30)
            downloaded.raise_for_status()
            if downloaded.content != checkpoint.getvalue():
                raise RuntimeError("MLflow checkpoint download differs from the upload")
            api("/api/2.0/mlflow/runs/update", {"run_id": run_id, "status": "FINISHED", "end_time": int(time.time() * 1000)})
            runs.add(1, {"outcome": "success"})
            outcome = "success"
            print(json.dumps({"event": "training_completed", "run_id": run_id, "epochs": 200, "loss": value, "artifact_bytes": len(checkpoint.getvalue())}), flush=True)
    except Exception:
        runs.add(1, {"outcome": "error"})
        if run_id:
            try:
                api("/api/2.0/mlflow/runs/update", {"run_id": run_id, "status": "FAILED", "end_time": int(time.time() * 1000)})
            except Exception:
                logging.warning("Could not mark the MLflow training run failed")
        raise
    finally:
        duration.record(time.monotonic() - started, {"outcome": outcome})
        meters.force_flush()
        traces.force_flush()
        meters.shutdown()
        traces.shutdown()


if __name__ == "__main__":
    main()
