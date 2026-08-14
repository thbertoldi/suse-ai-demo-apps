from unittest.mock import MagicMock

import langchain_openai
from langchain_core.messages import AIMessage

import app.agent as agent_mod
import app.tools as tools


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
