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
