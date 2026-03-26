import json
import os
import logging
import time

import grpc
from opentelemetry import trace
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool as langchain_tool
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
import operator

from app.otel_instrumentation import (
    invoke_agent_span, token_usage_histogram, operation_duration_histogram,
    ENABLE_CONTENT_EVENTS,
)
from app import tools as agent_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a helpful assistant with access to tools. Use them when needed to answer the user's question accurately."


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def create_agent(rag_channel: grpc.Channel):
    llm_base_url = os.environ.get("LLM_BASE_URL", "http://vllm:8000/v1")
    llm_model = os.environ.get("LLM_MODEL", "llama3")
    llm_provider = os.environ.get("LLM_PROVIDER", "vllm")
    agent_name = os.environ.get("AGENT_NAME", "demo-agent")
    max_iterations = int(os.environ.get("AGENT_MAX_ITERATIONS", "5"))

    llm = ChatOpenAI(
        base_url=llm_base_url,
        model=llm_model,
        api_key="not-needed",
        temperature=0.7,
        max_tokens=512,
    )

    @langchain_tool
    def search_docs(query: str) -> str:
        """Search the document knowledge base for relevant information on a topic."""
        return "placeholder"

    @langchain_tool
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression and return the result."""
        return "placeholder"

    @langchain_tool
    def web_search(query: str) -> str:
        """Search the web for information on a topic."""
        return "placeholder"

    @langchain_tool
    def get_current_time() -> str:
        """Get the current date and time in UTC."""
        return "placeholder"

    lc_tools = [search_docs, calculate, web_search, get_current_time]
    llm_with_tools = llm.bind_tools(lc_tools)

    tracer = trace.get_tracer("gen_ai")

    def llm_call(state: AgentState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

        start_time = time.monotonic()
        with tracer.start_as_current_span(
            f"chat {llm_model}",
            kind=trace.SpanKind.CLIENT,
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": llm_model,
                "gen_ai.provider.name": llm_provider,
            },
        ) as span:
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                span.set_status(trace.StatusCode.ERROR, str(e))
                span.set_attribute("error.type", type(e).__name__)
                raise
            finally:
                duration = time.monotonic() - start_time
                common_attrs = {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.request.model": llm_model,
                    "gen_ai.provider.name": llm_provider,
                }
                operation_duration_histogram.record(duration, attributes=common_attrs)

            if hasattr(response, "response_metadata"):
                meta = response.response_metadata
                token_usage = meta.get("token_usage", meta.get("usage", {}))
                input_tokens = token_usage.get("prompt_tokens", 0)
                output_tokens = token_usage.get("completion_tokens", 0)
                finish_reason = meta.get("finish_reason", "")
                model_name = meta.get("model_name", meta.get("model", llm_model))

                span.set_attribute("gen_ai.response.model", model_name)
                span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
                span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
                span.set_attribute("gen_ai.response.finish_reasons", [finish_reason])

                token_usage_histogram.record(input_tokens, attributes={
                    **common_attrs, "gen_ai.token.type": "input",
                })
                token_usage_histogram.record(output_tokens, attributes={
                    **common_attrs, "gen_ai.token.type": "output",
                })

            if ENABLE_CONTENT_EVENTS:
                input_msgs = [{"role": m.type, "content": m.content} for m in messages if hasattr(m, "content")]
                span.add_event("gen_ai.input.messages", attributes={
                    "gen_ai.input.messages": json.dumps(input_msgs),
                })
                span.add_event("gen_ai.output.messages", attributes={
                    "gen_ai.output.messages": json.dumps([{"role": "assistant", "content": response.content}]),
                })

        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def tool_node(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        results = []
        for tc in last_message.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_call_id = tc["id"]

            if tool_name == "search_docs":
                result = agent_tools.search_docs(
                    query=tool_args.get("query", ""),
                    rag_channel=rag_channel,
                    top_k=tool_args.get("top_k", 3),
                    tool_call_id=tool_call_id,
                )
            elif tool_name == "calculate":
                result = agent_tools.calculate(
                    expression=tool_args.get("expression", ""),
                    tool_call_id=tool_call_id,
                )
            elif tool_name == "web_search":
                result = agent_tools.web_search(
                    query=tool_args.get("query", ""),
                    tool_call_id=tool_call_id,
                )
            elif tool_name == "get_current_time":
                result = agent_tools.get_current_time(
                    tool_call_id=tool_call_id,
                )
            else:
                result = f"Unknown tool: {tool_name}"

            results.append(ToolMessage(content=result, tool_call_id=tool_call_id))
        return {"messages": results}

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            if state.get("llm_calls", 0) >= max_iterations:
                logger.warning("Max iterations reached, stopping agent loop")
                return END
            return "tool_node"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("llm_call", llm_call)
    graph.add_node("tool_node", tool_node)
    graph.add_edge(START, "llm_call")
    graph.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    graph.add_edge("tool_node", "llm_call")

    compiled = graph.compile()

    def run_agent(message: str) -> dict:
        with invoke_agent_span(agent_name, llm_model) as span:
            result = compiled.invoke({
                "messages": [HumanMessage(content=message)],
                "llm_calls": 0,
            })

            if result.get("llm_calls", 0) >= max_iterations:
                span.set_attribute("gen_ai.agent.truncated", True)

            last = result["messages"][-1]
            reply = last.content if hasattr(last, "content") else str(last)

            # Collect tool call info: correlate AIMessage tool_calls with ToolMessage results
            pending_tool_calls = {}  # tool_call_id -> {name, arguments}
            tool_calls_made = []
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        pending_tool_calls[tc["id"]] = {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"]),
                        }
                elif isinstance(msg, ToolMessage):
                    tc_info = pending_tool_calls.get(msg.tool_call_id, {})
                    tool_calls_made.append({
                        "name": tc_info.get("name", ""),
                        "arguments": tc_info.get("arguments", ""),
                        "result": msg.content,
                    })

            model_used = llm_model
            if hasattr(last, "response_metadata"):
                model_used = last.response_metadata.get("model_name", last.response_metadata.get("model", llm_model))

            return {
                "reply": reply,
                "model": model_used,
                "tool_calls_made": tool_calls_made,
            }

    return run_agent
