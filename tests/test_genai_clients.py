"""Exercise the actual SDK output from each separately packaged HTTP client."""

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import requests
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


ROOT = Path(__file__).resolve().parents[1]
CLIENTS = (
    ("llm-service/app/llm_client.py", "chat"),
    ("rag-service/app/llm_client.py", "chat"),
    ("rag-service/app/embedding_client.py", "embed"),
)
DURATION = "gen_ai.client.operation.duration"
TOKENS = "gen_ai.client.token.usage"
ABSENT = object()


@contextmanager
def client(relative):
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    spec = importlib.util.spec_from_file_location("isolated_genai_client", ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    with patch.object(metrics, "get_meter", meter_provider.get_meter), \
            patch.object(trace, "get_tracer", tracer_provider.get_tracer):
        spec.loader.exec_module(module)
    module.ENABLE_CONTENT_EVENTS = False
    try:
        yield module, reader, exporter
    finally:
        meter_provider.shutdown()
        tracer_provider.shutdown()


def points(reader, name):
    data = reader.get_metrics_data()
    if data is None:
        return []
    return [point for resource in data.resource_metrics for scope in resource.scope_metrics
            for metric in scope.metrics if metric.name == name for point in metric.data.data_points]


def response(data, status=200):
    result = requests.Response()
    result.status_code = status
    result.url = "http://provider.invalid/fixture"
    result._content = json.dumps(data).encode()
    return result


def payload(operation, usage=ABSENT):
    if operation == "embed":
        result = {"embeddings": [[0.1, 0.2]]}
        if usage is not ABSENT:
            result["prompt_eval_count"] = usage
    else:
        result = {"model": "reported-model", "choices": [
            {"message": {"content": "fixture answer"}, "finish_reason": "stop"},
        ]}
        if usage is not ABSENT:
            result["usage"] = usage
    return result


def invoke(module, operation, result, messages=None):
    post = Mock(side_effect=result) if isinstance(result, Exception) else Mock(return_value=result)
    # Patch this module's clock only; the SDK still uses its actual timestamp clock.
    clock = SimpleNamespace(monotonic=Mock(side_effect=[10.0, 10.4]))
    with patch.object(module.requests, "post", post), patch.object(module, "time", clock):
        if operation == "embed":
            return module.embed("http://provider.invalid", "requested-model", "fixture", "fixture prompt")
        return module.chat_completion("http://provider.invalid", "requested-model", "fixture",
                                      messages or [{"role": "user", "content": "fixture prompt"}])


class GenAIClientTests(unittest.TestCase):
    def test_content_capture_environment_precedence(self):
        standard = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
        legacy = "ENABLE_OTEL_CONTENT_EVENTS"
        cases = (({}, False), ({legacy: "true"}, True),
                 ({standard: "false", legacy: "true"}, False),
                 ({standard: "true", legacy: "false"}, True))
        for path, _ in CLIENTS[:2]:
            for environment, expected in cases:
                with self.subTest(client=path, environment=environment), \
                        patch.dict("os.environ", environment, clear=True), client(path) as (module, _, _):
                    self.assertEqual(module._capture_content(), expected)

    def test_opt_in_content_is_recorded_as_span_attributes(self):
        messages = [{"role": "system", "content": "fixture instructions"},
                    {"role": "user", "content": "fixture prompt"}]
        for path, operation in CLIENTS[:2]:
            for enabled in (False, True):
                with self.subTest(client=path, enabled=enabled), client(path) as (module, _, exporter):
                    module.ENABLE_CONTENT_EVENTS = enabled
                    invoke(module, operation, response(payload(operation)), messages)
                    span = exporter.get_finished_spans()[0]
                    self.assertFalse(span.events)
                    if enabled:
                        self.assertEqual(json.loads(span.attributes["gen_ai.system_instructions"]),
                                         [{"type": "text", "content": "fixture instructions"}])
                        self.assertEqual(json.loads(span.attributes["gen_ai.input.messages"]), messages[1:])
                        self.assertEqual(json.loads(span.attributes["gen_ai.output.messages"]),
                                         [{"role": "assistant", "content": "fixture answer"}])
                    else:
                        for name in ("system_instructions", "input.messages", "output.messages"):
                            self.assertNotIn("gen_ai." + name, span.attributes)

    def assert_duration(self, reader, error_type=None):
        samples = points(reader, DURATION)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample.count, 1)
        self.assertAlmostEqual(sample.sum, 0.4)
        self.assertEqual(sample.attributes.get("error.type"), error_type)
        occupied = next(i for i, count in enumerate(sample.bucket_counts) if count)
        self.assertEqual(sample.explicit_bounds[occupied], 0.5)
        self.assertIn(0.01, sample.explicit_bounds)
        self.assertIn(120, sample.explicit_bounds)

    def test_reported_usage_and_subsecond_success_duration(self):
        for path, operation in CLIENTS:
            with self.subTest(client=path), client(path) as (module, reader, exporter):
                usage = 12 if operation == "embed" else {"prompt_tokens": 12, "completion_tokens": 7}
                data = payload(operation, usage)
                result = invoke(module, operation, response(data))
                self.assertEqual(result, data["embeddings"][0] if operation == "embed" else data)
                self.assert_duration(reader)
                token_points = points(reader, TOKENS)
                actual = {p.attributes["gen_ai.token.type"]: (p.count, p.sum) for p in token_points}
                expected = {"input": (1, 12)}
                if operation == "chat":
                    expected["output"] = (1, 7)
                self.assertEqual(actual, expected)
                spans = exporter.get_finished_spans()
                self.assertEqual(len(spans), 1)
                self.assertEqual(spans[0].kind, trace.SpanKind.CLIENT)
                self.assertEqual(spans[0].status.status_code, trace.StatusCode.UNSET)
                self.assertEqual(spans[0].attributes["gen_ai.usage.input_tokens"], 12)
                self.assertFalse(spans[0].events)

    def test_http_transport_decode_and_response_failures_record_once(self):
        for path, operation in CLIENTS:
            malformed_json = response(None)
            malformed_json._content = b"not json"
            malformed_body = response({"embeddings": []} if operation == "embed" else None)
            cases = (
                (response({"error": "controlled"}, 503), requests.HTTPError, "503"),
                (requests.Timeout("controlled timeout"), requests.Timeout, "Timeout"),
                (malformed_json, requests.JSONDecodeError, "JSONDecodeError"),
                (malformed_body, IndexError if operation == "embed" else AttributeError,
                 "IndexError" if operation == "embed" else "AttributeError"),
            )
            for result, exception, error_type in cases:
                with self.subTest(client=path, error=error_type), client(path) as (module, reader, exporter):
                    with self.assertRaises(exception):
                        invoke(module, operation, result)
                    self.assert_duration(reader, error_type)
                    self.assertEqual(points(reader, TOKENS), [])
                    spans = exporter.get_finished_spans()
                    self.assertEqual(len(spans), 1)
                    self.assertEqual(spans[0].status.status_code, trace.StatusCode.ERROR)
                    self.assertEqual(spans[0].attributes["error.type"], error_type)

    def test_missing_and_malformed_usage_never_becomes_zero(self):
        for path, operation in CLIENTS:
            for usage in (ABSENT, None, {}, -1, True, 1.5, "12"):
                if operation == "chat" and usage not in (ABSENT, None, {}):
                    usage = {"prompt_tokens": usage, "completion_tokens": usage}
                with self.subTest(client=path, usage=usage), client(path) as (module, reader, exporter):
                    invoke(module, operation, response(payload(operation, usage)))
                    self.assert_duration(reader)
                    self.assertEqual(points(reader, TOKENS), [])
                    attrs = exporter.get_finished_spans()[0].attributes
                    self.assertNotIn("gen_ai.usage.input_tokens", attrs)
                    self.assertNotIn("gen_ai.usage.output_tokens", attrs)

    def test_reported_zero_is_preserved(self):
        for path, operation in CLIENTS:
            with self.subTest(client=path), client(path) as (module, reader, exporter):
                usage = 0 if operation == "embed" else {"prompt_tokens": 0, "completion_tokens": 0}
                invoke(module, operation, response(payload(operation, usage)))
                samples = points(reader, TOKENS)
                self.assertEqual(len(samples), 1 if operation == "embed" else 2)
                self.assertTrue(all(p.count == 1 and p.sum == 0 for p in samples))
                self.assertEqual(exporter.get_finished_spans()[0].attributes["gen_ai.usage.input_tokens"], 0)

    def test_partial_usage_keeps_only_the_reported_direction(self):
        for path, operation in CLIENTS[:2]:
            for usage, token_type, count in (({"prompt_tokens": 4}, "input", 4),
                                             ({"prompt_tokens": False, "completion_tokens": 9}, "output", 9)):
                with self.subTest(client=path, usage=usage), client(path) as (module, reader, exporter):
                    invoke(module, operation, response(payload(operation, usage)))
                    samples = points(reader, TOKENS)
                    self.assertEqual(len(samples), 1)
                    self.assertEqual(samples[0].attributes["gen_ai.token.type"], token_type)
                    self.assertEqual((samples[0].count, samples[0].sum), (1, count))
                    self.assertEqual(exporter.get_finished_spans()[0].attributes[f"gen_ai.usage.{token_type}_tokens"], count)


if __name__ == "__main__":
    unittest.main()
