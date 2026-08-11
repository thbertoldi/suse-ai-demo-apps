import json
import math
import os
import random
import time
from datetime import datetime, timezone

import grpc
import httpx
from simpleeval import simple_eval
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from app.generated import demo_pb2, demo_pb2_grpc
from app.otel_instrumentation import execute_tool_span, record_tool_result, tracer


KSERVE_PREDICT_URL = os.environ.get(
    "KSERVE_PREDICT_URL",
    "http://sklearn-iris-predictor-default.kserve-test.svc.cluster.local/v1/models/sklearn-iris:predict",
)
MODEL_REGISTRY_URL = os.environ.get(
    "MODEL_REGISTRY_URL",
    "http://model-registry-service.kubeflow.svc.cluster.local:8080",
)
MODEL_REGISTRY_BEARER_TOKEN = os.environ.get("MODEL_REGISTRY_BEARER_TOKEN", "demo")


TOOL_DESCRIPTIONS = {
    "search_docs": "Search the document knowledge base for relevant information on a topic",
    "calculate": "Evaluate a mathematical expression and return the result",
    "web_search": "Search the web for information on a topic",
    "get_current_time": "Get the current date and time in UTC",
    "predict": "Classify an iris flower species from its sepal/petal measurements using the deployed KServe model",
    "list_models": "List the machine learning models registered in the Kubeflow model registry",
}


def _tool_text(value: object, preferred_key: str) -> str:
    """Normalize occasionally nested tool arguments emitted by local models."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and preferred_key in value:
        return _tool_text(value[preferred_key], preferred_key)
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _model_registry_headers() -> dict[str, str]:
    token = MODEL_REGISTRY_BEARER_TOKEN.strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _iris_measurement(value: object, name: str) -> float:
    """Normalize numeric tool arguments emitted as strings or nested values."""
    if isinstance(value, dict):
        if name in value:
            return _iris_measurement(value[name], name)
        if "value" in value:
            return _iris_measurement(value["value"], name)
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _iris_measurement(value[0], name)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive number")
    try:
        measurement = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if not math.isfinite(measurement) or measurement <= 0:
        raise ValueError(f"{name} must be a positive number")
    return measurement


def search_docs(query: str, rag_channel: grpc.Channel, top_k: int = 3, tool_call_id: str = "") -> str:
    with execute_tool_span("search_docs", tool_call_id, TOOL_DESCRIPTIONS["search_docs"]) as span:
        query_text = _tool_text(query, "query")
        args = json.dumps({"query": query_text, "top_k": top_k})
        try:
            stub = demo_pb2_grpc.RAGServiceStub(rag_channel)
            resp = stub.Retrieve(demo_pb2.RetrieveRequest(query=query_text, top_k=top_k), timeout=120)
            sources = list(resp.sources)
            if sources:
                result = "\n\n".join(sources)
            else:
                result = "No relevant documents found."
            span.set_status(trace.StatusCode.OK)
        except grpc.RpcError as e:
            result = f"Error searching documents: {e.details()}"
            span.set_status(trace.StatusCode.ERROR, result)
        record_tool_result(span, args, result)
        return result


def calculate(expression: str, tool_call_id: str = "") -> str:
    with execute_tool_span("calculate", tool_call_id, TOOL_DESCRIPTIONS["calculate"]) as span:
        args = json.dumps({"expression": expression})
        try:
            result = str(simple_eval(expression))
            span.set_status(trace.StatusCode.OK)
        except Exception as e:
            result = f"Error evaluating expression: {e}"
            span.set_status(trace.StatusCode.ERROR, result)
        record_tool_result(span, args, result)
        return result


def web_search(query: str, tool_call_id: str = "") -> str:
    with execute_tool_span("web_search", tool_call_id, TOOL_DESCRIPTIONS["web_search"]) as span:
        args = json.dumps({"query": query})
        time.sleep(random.uniform(0.5, 1.5))
        results = [
            {"title": f"Result 1 for '{query}'", "snippet": f"This is a simulated search result about {query}. It contains relevant information."},
            {"title": f"Result 2 for '{query}'", "snippet": f"Another perspective on {query} from a different source."},
            {"title": f"Result 3 for '{query}'", "snippet": f"Technical documentation related to {query}."},
        ]
        result = json.dumps(results)
        record_tool_result(span, args, result)
        span.set_status(trace.StatusCode.OK)
        return result


def get_current_time(tool_call_id: str = "") -> str:
    with execute_tool_span("get_current_time", tool_call_id, TOOL_DESCRIPTIONS["get_current_time"]) as span:
        result = datetime.now(timezone.utc).isoformat()
        record_tool_result(span, "{}", result)
        span.set_status(trace.StatusCode.OK)
        return result


def predict(sepal_length: float, sepal_width: float, petal_length: float,
            petal_width: float, tool_call_id: str = "") -> str:
    with execute_tool_span("predict", tool_call_id, TOOL_DESCRIPTIONS["predict"]) as span:
        raw_values = [sepal_length, sepal_width, petal_length, petal_width]
        args = json.dumps({"instances": [raw_values]}, default=str)
        try:
            values = [
                _iris_measurement(sepal_length, "sepal_length"),
                _iris_measurement(sepal_width, "sepal_width"),
                _iris_measurement(petal_length, "petal_length"),
                _iris_measurement(petal_width, "petal_width"),
            ]
            payload = {"instances": [values]}
            # Explicit CLIENT span carries the KServe relation attribute so the
            # collector's transform/kubeflow-relations sets peer.service=kserve.
            with tracer.start_as_current_span(
                "predict sklearn-iris",
                kind=SpanKind.CLIENT,
                attributes={"kserve.inference.service": "sklearn-iris"},
            ):
                resp = httpx.post(KSERVE_PREDICT_URL, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            result = json.dumps(data.get("predictions", data))
            span.set_status(trace.StatusCode.OK)
        except Exception as e:
            result = f"Error calling inference service: {e}"
            span.set_status(trace.StatusCode.ERROR, result)
        record_tool_result(span, args, result)
        return result


def list_models(tool_call_id: str = "") -> str:
    with execute_tool_span("list_models", tool_call_id, TOOL_DESCRIPTIONS["list_models"]) as span:
        url = f"{MODEL_REGISTRY_URL}/api/model_registry/v1alpha3/registered_models"
        args = json.dumps({"url": url})
        try:
            resp = httpx.get(url, headers=_model_registry_headers(), timeout=30)
            resp.raise_for_status()
            items = resp.json().get("items", [])
            names = [m.get("name", "") for m in items if m.get("name")]
            result = json.dumps(names) if names else "No models registered."
            span.set_status(trace.StatusCode.OK)
        except Exception as e:
            result = f"Error listing models: {e}"
            span.set_status(trace.StatusCode.ERROR, result)
        record_tool_result(span, args, result)
        return result
