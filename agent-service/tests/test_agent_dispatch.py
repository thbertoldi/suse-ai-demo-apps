from unittest.mock import MagicMock
import pytest

import langchain_openai
from langchain_core.messages import AIMessage

import app.agent as agent_mod
import app.tools as tools


@pytest.mark.parametrize("usage, expected", [
    (None, {}),
    ({}, {}),
    ({"prompt_tokens": 0, "completion_tokens": 0}, {"input": 0, "output": 0}),
    ({"prompt_tokens": 4}, {"input": 4}),
    ({"prompt_tokens": False, "completion_tokens": -1}, {}),
    ({"prompt_tokens": "4", "completion_tokens": 2.5}, {}),
])
def test_agent_records_only_reported_integer_usage(monkeypatch, span_exporter, usage, expected):
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    histogram = provider.get_meter("agent-test").create_histogram("gen_ai.client.token.usage")
    monkeypatch.setattr(agent_mod, "token_usage_histogram", histogram)
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="done", response_metadata={"token_usage": usage})
    monkeypatch.setattr(langchain_openai.ChatOpenAI, "bind_tools", lambda *args: llm)
    try:
        result = agent_mod.create_agent(MagicMock())("Give a brief answer")
        assert result["reply"] == "done"
        data = reader.get_metrics_data()
        points = [] if data is None else [p for r in data.resource_metrics for s in r.scope_metrics
                                         for m in s.metrics for p in m.data.data_points]
        assert {p.attributes["gen_ai.token.type"]: p.sum for p in points} == expected
        assert all(p.count == 1 for p in points)
        chat = next(s for s in span_exporter.get_finished_spans() if s.name.startswith("chat "))
        for token_type in ("input", "output"):
            if token_type in expected:
                assert chat.attributes[f"gen_ai.usage.{token_type}_tokens"] == expected[token_type]
            else:
                assert f"gen_ai.usage.{token_type}_tokens" not in chat.attributes
    finally:
        provider.shutdown()


def test_failed_model_call_records_endpoint_and_error_type(monkeypatch, span_exporter):
    llm = MagicMock()
    llm.invoke.side_effect = TimeoutError("model deadline")
    monkeypatch.setattr(langchain_openai.ChatOpenAI, "bind_tools", lambda *args: llm)
    monkeypatch.setenv("LLM_BASE_URL", "http://models.example:8000/v1")
    histogram = MagicMock()
    monkeypatch.setattr(agent_mod, "operation_duration_histogram", histogram)

    with pytest.raises(TimeoutError):
        agent_mod.create_agent(MagicMock())("Give a brief answer")

    attrs = histogram.record.call_args.kwargs["attributes"]
    assert attrs["error.type"] == "TimeoutError"
    assert attrs["server.address"] == "models.example"
    assert attrs["server.port"] == 8000
    chat = next(s for s in span_exporter.get_finished_spans() if s.name.startswith("chat "))
    assert chat.attributes["server.address"] == attrs["server.address"]
    assert chat.attributes["error.type"] == attrs["error.type"]


class FakeBoundLLM:
    """Emits one tool call, then a final answer."""

    def __init__(self, tool_name):
        self._tool_name = tool_name
        self._calls = 0

    def invoke(self, messages):
        self._calls += 1
        if self._calls == 1:
            return AIMessage(
                content="",
                tool_calls=[{"name": self._tool_name, "args": {}, "id": "c1"}],
            )
        return AIMessage(content="done", response_metadata={"model_name": "test-model"})


def _patch_llm(monkeypatch, tool_name):
    monkeypatch.setattr(
        langchain_openai.ChatOpenAI, "bind_tools",
        lambda self, tools_arg: FakeBoundLLM(tool_name),
    )


def test_agent_dispatches_list_models(monkeypatch):
    called = {}
    monkeypatch.setattr(tools, "list_models",
                        lambda tool_call_id="": called.setdefault("hit", True) or "iris")
    _patch_llm(monkeypatch, "list_models")

    run_agent = agent_mod.create_agent(MagicMock())
    result = run_agent("list registered models")

    assert called.get("hit") is True
    assert result["reply"] == "done"
    assert any(tc["name"] == "list_models" for tc in result["tool_calls_made"])


def test_agent_dispatches_predict(monkeypatch):
    called = {}

    def fake_predict(sepal_length=0.0, sepal_width=0.0, petal_length=0.0,
                     petal_width=0.0, tool_call_id=""):
        called["args"] = (sepal_length, sepal_width, petal_length, petal_width)
        return "[0]"

    monkeypatch.setattr(tools, "predict", fake_predict)
    _patch_llm(monkeypatch, "predict")

    run_agent = agent_mod.create_agent(MagicMock())
    run_agent("classify an iris")

    assert "args" in called


def test_deterministic_lifecycle_calls_registry_and_predict(monkeypatch):
    calls = []
    monkeypatch.setenv("DEMO_DETERMINISTIC_TOOLS", "true")
    monkeypatch.setattr(
        tools,
        "list_models",
        lambda tool_call_id="": calls.append("list_models") or '["iris"]',
    )
    monkeypatch.setattr(
        tools,
        "predict",
        lambda **kwargs: calls.append("predict") or "[0]",
    )
    _patch_llm(monkeypatch, "get_current_time")

    run_agent = agent_mod.create_agent(MagicMock())
    result = run_agent(
        '[demo:lifecycle] {"sepal_length": 5.1, "sepal_width": 3.5, '
        '"petal_length": 1.4, "petal_width": 0.2}'
    )

    assert calls == ["list_models", "predict"]
    assert [item["name"] for item in result["tool_calls_made"]] == [
        "list_models",
        "predict",
    ]
    assert result["model"] == "deterministic-demo"
