# Agent Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a LangGraph-based ReAct agent service to the SUSE AI demo apps, producing rich OpenTelemetry traces following GenAI semantic conventions (`invoke_agent`, `execute_tool`, `chat` spans).

**Architecture:** New Python gRPC service (`agent-service`) behind the existing Go Gateway. Uses LangGraph for the agent loop with `ChatOpenAI` for LLM calls. Four tools: `search_docs` (calls RAG service via gRPC), `calculate`, `web_search` (simulated), `get_current_time`. Manual OTel span wrappers ensure precise semantic convention compliance.

**Tech Stack:** Python 3.12, LangGraph 0.4.8, langchain-openai 0.3.18, gRPC, OpenTelemetry 1.40.0, simpleeval

**Spec:** `docs/superpowers/specs/2026-03-25-agent-service-design.md`

---

## File Map

### New Files

| File | Responsibility |
|------|---------------|
| `agent-service/app/__init__.py` | Package marker |
| `agent-service/app/main.py` | Entrypoint: OTel init, gRPC server, health checks, signal handling |
| `agent-service/app/otel_setup.py` | OTel bootstrap (tracer, meter, propagation, auto-instrumentation) |
| `agent-service/app/otel_instrumentation.py` | Manual span wrappers: `invoke_agent`, `execute_tool`, `chat` |
| `agent-service/app/agent.py` | LangGraph ReAct graph definition |
| `agent-service/app/tools.py` | Tool definitions: search_docs, calculate, web_search, get_current_time |
| `agent-service/app/grpc_server.py` | `AgentServiceServicer` gRPC handler |
| `agent-service/app/generated/__init__.py` | Package marker for generated proto stubs |
| `agent-service/requirements.txt` | Python dependencies |
| `agent-service/Dockerfile` | Container image build |
| `helm/suse-ai-demo/templates/agent-service-deployment.yaml` | K8s Deployment |
| `helm/suse-ai-demo/templates/agent-service-service.yaml` | K8s Service |

### Modified Files

| File | Change |
|------|--------|
| `proto/demo.proto` | Add `AgentChat` RPC, `AgentService`, new messages |
| `proto/Makefile` | Add agent-service output path + sed fixup |
| `gateway/main.go` | Add agent service client conn, `AgentChat` RPC handler |
| `traffic-gen/main.py` | Add agent queries, three-way rotation |
| `traffic-gen/requirements.txt` | No change needed (proto stubs regenerated) |
| `helm/suse-ai-demo/values.yaml` | Add `agentService` section |
| `.github/workflows/build.yaml` | Add agent-service to build matrix |

---

## Task 1: Proto Changes

**Files:**
- Modify: `proto/demo.proto`
- Modify: `proto/Makefile`

- [ ] **Step 1: Add agent messages and services to demo.proto**

Add these after the existing `GenerateResponse` message in `proto/demo.proto`:

```protobuf
service AgentService {
  rpc Run(AgentRequest) returns (AgentResponse);
}

message AgentChatRequest {
  string message = 1;
}

message AgentChatResponse {
  string reply = 1;
  string model = 2;
  repeated AgentToolCall tool_calls_made = 3;
}

message AgentRequest {
  string message = 1;
}

message AgentResponse {
  string reply = 1;
  string model = 2;
  repeated AgentToolCall tool_calls_made = 3;
}

message AgentToolCall {
  string name = 1;
  string arguments = 2;
  string result = 3;
}
```

Also add to the existing `DemoService`:

```protobuf
rpc AgentChat(AgentChatRequest) returns (AgentChatResponse);
```

- [ ] **Step 2: Update Makefile for agent-service stubs**

In `proto/Makefile`, add to the `python` target — after the llm-service block and before the sed lines:

```makefile
	python3 -m grpc_tools.protoc -I. \
		--python_out=../agent-service/app/generated --grpc_python_out=../agent-service/app/generated \
		demo.proto
```

Add to the sed fixup section:

```makefile
	sed -i 's/import demo_pb2/from . import demo_pb2/' ../agent-service/app/generated/demo_pb2_grpc.py
```

Add to the `clean` target:

```makefile
	rm -f ../agent-service/app/generated/demo_pb2*.py
```

- [ ] **Step 3: Create agent-service generated directory**

```bash
mkdir -p agent-service/app/generated
touch agent-service/app/generated/__init__.py
```

- [ ] **Step 4: Regenerate proto stubs**

```bash
cd proto && make all
```

Expected: stubs generated in `gateway/pb/`, all Python `*/generated/` dirs, and `agent-service/app/generated/`.

- [ ] **Step 5: Verify generated stubs**

Check that `agent-service/app/generated/demo_pb2.py` and `demo_pb2_grpc.py` exist and contain `AgentService` references:

```bash
grep -l "AgentService" agent-service/app/generated/demo_pb2_grpc.py
```

Expected: file path printed (match found).

- [ ] **Step 6: Commit**

```bash
git add proto/demo.proto proto/Makefile agent-service/app/generated/ \
  gateway/pb/ rag-service/app/generated/ llm-service/app/generated/ traffic-gen/generated/
git commit -m "feat: add AgentService and AgentChat to proto definition"
```

---

## Task 2: Agent Service — OTel Setup and Instrumentation

**Files:**
- Create: `agent-service/app/__init__.py`
- Create: `agent-service/app/otel_setup.py`
- Create: `agent-service/app/otel_instrumentation.py`

- [ ] **Step 1: Create package init**

Create empty `agent-service/app/__init__.py`.

- [ ] **Step 2: Create otel_setup.py**

Follows the exact same pattern as `rag-service/app/otel_setup.py`, but adds `httpx` auto-instrumentation instead of `requests`:

```python
import os
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.instrumentation.grpc import GrpcInstrumentorServer, GrpcInstrumentorClient
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositeHTTPPropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.baggage.propagation import W3CBaggagePropagator


def setup_otel(service_name: str) -> tuple[TracerProvider, MeterProvider]:
    resource = Resource.create({"service.name": service_name})

    trace_exporter = OTLPSpanExporter(insecure=True)
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
    trace.set_tracer_provider(tracer_provider)

    metric_exporter = OTLPMetricExporter(insecure=True)
    metric_reader = PeriodicExportingMetricReader(metric_exporter)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    set_global_textmap(CompositeHTTPPropagator([
        TraceContextTextMapPropagator(),
        W3CBaggagePropagator(),
    ]))

    GrpcInstrumentorServer().instrument()
    GrpcInstrumentorClient().instrument()
    HTTPXClientInstrumentor().instrument()

    return tracer_provider, meter_provider


def shutdown_otel(tracer_provider: TracerProvider, meter_provider: MeterProvider):
    tracer_provider.shutdown()
    meter_provider.shutdown()
```

- [ ] **Step 3: Create otel_instrumentation.py**

This module provides manual span wrappers for the three GenAI span types:

```python
import json
import os
import time
import uuid
from contextlib import contextmanager

from opentelemetry import trace, metrics

tracer = trace.get_tracer("gen_ai")
meter = metrics.get_meter("gen_ai")

token_usage_histogram = meter.create_histogram(
    name="gen_ai.client.token.usage",
    description="Token usage per GenAI call",
    unit="{token}",
)

operation_duration_histogram = meter.create_histogram(
    name="gen_ai.client.operation.duration",
    description="Duration of GenAI operations",
    unit="s",
)

ENABLE_CONTENT_EVENTS = os.environ.get("ENABLE_OTEL_CONTENT_EVENTS", "false").lower() == "true"


@contextmanager
def invoke_agent_span(agent_name: str, model: str):
    agent_id = str(uuid.uuid4())
    with tracer.start_as_current_span(
        f"invoke_agent {agent_name}",
        kind=trace.SpanKind.INTERNAL,
        attributes={
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": agent_name,
            "gen_ai.agent.id": agent_id,
            "gen_ai.request.model": model,
        },
    ) as span:
        try:
            yield span
        except Exception as e:
            span.set_status(trace.StatusCode.ERROR, str(e))
            span.set_attribute("error.type", type(e).__name__)
            raise


@contextmanager
def execute_tool_span(tool_name: str, tool_call_id: str, tool_description: str = ""):
    with tracer.start_as_current_span(
        f"execute_tool {tool_name}",
        kind=trace.SpanKind.INTERNAL,
        attributes={
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool_name,
            "gen_ai.tool.type": "function",
            "gen_ai.tool.call.id": tool_call_id,
            "gen_ai.tool.description": tool_description,
        },
    ) as span:
        try:
            yield span
        except Exception as e:
            span.set_status(trace.StatusCode.ERROR, str(e))
            span.set_attribute("error.type", type(e).__name__)
            raise


def record_tool_result(span, arguments: str, result: str):
    if ENABLE_CONTENT_EVENTS:
        span.set_attribute("gen_ai.tool.call.arguments", arguments)
    span.set_attribute("gen_ai.tool.call.result", result)
```

- [ ] **Step 4: Commit**

```bash
git add agent-service/app/__init__.py agent-service/app/otel_setup.py agent-service/app/otel_instrumentation.py
git commit -m "feat: add OTel setup and GenAI span instrumentation for agent service"
```

---

## Task 3: Agent Service — Tools

**Files:**
- Create: `agent-service/app/tools.py`

- [ ] **Step 1: Create tools.py**

```python
import json
import random
import time
from datetime import datetime, timezone

import grpc
from simpleeval import simple_eval

from app.generated import demo_pb2, demo_pb2_grpc
from app.otel_instrumentation import execute_tool_span, record_tool_result


TOOL_DESCRIPTIONS = {
    "search_docs": "Search the document knowledge base for relevant information on a topic",
    "calculate": "Evaluate a mathematical expression and return the result",
    "web_search": "Search the web for information on a topic",
    "get_current_time": "Get the current date and time in UTC",
}


def search_docs(query: str, rag_channel: grpc.Channel, top_k: int = 3, tool_call_id: str = "") -> str:
    with execute_tool_span("search_docs", tool_call_id, TOOL_DESCRIPTIONS["search_docs"]) as span:
        args = json.dumps({"query": query, "top_k": top_k})
        try:
            stub = demo_pb2_grpc.RAGServiceStub(rag_channel)
            resp = stub.Retrieve(demo_pb2.RetrieveRequest(query=query, top_k=top_k), timeout=120)
            sources = list(resp.sources)
            if sources:
                result = "\n\n".join(sources)
            else:
                result = "No relevant documents found."
        except grpc.RpcError as e:
            result = f"Error searching documents: {e.details()}"
            from opentelemetry import trace
            span.set_status(trace.StatusCode.ERROR, result)
        record_tool_result(span, args, result)
        return result


def calculate(expression: str, tool_call_id: str = "") -> str:
    with execute_tool_span("calculate", tool_call_id, TOOL_DESCRIPTIONS["calculate"]) as span:
        args = json.dumps({"expression": expression})
        try:
            result = str(simple_eval(expression))
        except Exception as e:
            result = f"Error evaluating expression: {e}"
            from opentelemetry import trace
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
        return result


def get_current_time(tool_call_id: str = "") -> str:
    with execute_tool_span("get_current_time", tool_call_id, TOOL_DESCRIPTIONS["get_current_time"]) as span:
        result = datetime.now(timezone.utc).isoformat()
        record_tool_result(span, "{}", result)
        return result
```

- [ ] **Step 2: Commit**

```bash
git add agent-service/app/tools.py
git commit -m "feat: add agent tools (search_docs, calculate, web_search, get_current_time)"
```

---

## Task 4: Agent Service — LangGraph Agent

**Files:**
- Create: `agent-service/app/agent.py`

- [ ] **Step 1: Create agent.py**

This defines the LangGraph ReAct graph with manual OTel instrumentation on each node:

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add agent-service/app/agent.py
git commit -m "feat: add LangGraph ReAct agent with OTel-instrumented nodes"
```

---

## Task 5: Agent Service — gRPC Server and Main

**Files:**
- Create: `agent-service/app/grpc_server.py`
- Create: `agent-service/app/main.py`

- [ ] **Step 1: Create grpc_server.py**

```python
import logging

import grpc

from app.generated import demo_pb2, demo_pb2_grpc

logger = logging.getLogger(__name__)


class AgentServiceServicer(demo_pb2_grpc.AgentServiceServicer):
    def __init__(self, run_agent):
        self._run_agent = run_agent

    def Run(self, request, context):
        try:
            result = self._run_agent(request.message)

            tool_calls = []
            for tc in result.get("tool_calls_made", []):
                tool_calls.append(demo_pb2.AgentToolCall(
                    name=tc.get("name", ""),
                    arguments=tc.get("arguments", ""),
                    result=tc.get("result", ""),
                ))

            return demo_pb2.AgentResponse(
                reply=result["reply"],
                model=result["model"],
                tool_calls_made=tool_calls,
            )
        except Exception as e:
            logger.exception("Error in agent Run")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return demo_pb2.AgentResponse()
```

- [ ] **Step 2: Create main.py**

```python
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
```

- [ ] **Step 3: Commit**

```bash
git add agent-service/app/grpc_server.py agent-service/app/main.py
git commit -m "feat: add agent service gRPC server with health checks"
```

---

## Task 6: Agent Service — Requirements and Dockerfile

**Files:**
- Create: `agent-service/requirements.txt`
- Create: `agent-service/Dockerfile`

- [ ] **Step 1: Create requirements.txt**

```
langgraph==0.4.8
langchain-openai==0.3.18
langchain-core==0.3.51
simpleeval==1.0.3
grpcio==1.78.0
grpcio-tools==1.78.0
grpcio-health-checking==1.78.0
protobuf==6.31.1
opentelemetry-api==1.40.0
opentelemetry-sdk==1.40.0
opentelemetry-exporter-otlp-proto-grpc==1.40.0
opentelemetry-instrumentation-grpc==0.61b0
opentelemetry-instrumentation-httpx==0.61b0
```

- [ ] **Step 2: Create Dockerfile**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-m", "app.main"]
```

- [ ] **Step 3: Verify dependencies resolve**

```bash
cd agent-service && pip install --dry-run -r requirements.txt
```

Expected: no conflicts reported.

- [ ] **Step 4: Commit**

```bash
git add agent-service/requirements.txt agent-service/Dockerfile
git commit -m "feat: add agent service requirements and Dockerfile"
```

---

## Task 7: Gateway Changes

**Files:**
- Modify: `gateway/main.go`

- [ ] **Step 1: Add agent service connection and AgentChat handler**

In `gateway/main.go`:

1. Add `agentConn` field to `gatewayServer` struct:

```go
type gatewayServer struct {
	pb.UnimplementedDemoServiceServer
	ragConn   *grpc.ClientConn
	llmConn   *grpc.ClientConn
	agentConn *grpc.ClientConn
}
```

2. Add `AgentChat` method:

```go
func (s *gatewayServer) AgentChat(ctx context.Context, req *pb.AgentChatRequest) (*pb.AgentChatResponse, error) {
	client := pb.NewAgentServiceClient(s.agentConn)
	resp, err := client.Run(ctx, &pb.AgentRequest{
		Message: req.Message,
	})
	if err != nil {
		return nil, err
	}
	return &pb.AgentChatResponse{
		Reply:          resp.Reply,
		Model:          resp.Model,
		ToolCallsMade:  resp.ToolCallsMade,
	}, nil
}
```

3. In `main()`, after the `llmConn` block, add:

```go
agentAddr := envOrDefault("AGENT_SERVICE_ADDR", "agent-service:50054")

agentConn, err := grpc.NewClient(agentAddr,
    grpc.WithTransportCredentials(insecure.NewCredentials()),
    grpc.WithStatsHandler(otelgrpc.NewClientHandler()),
)
if err != nil {
    log.Fatalf("failed to connect to Agent service: %v", err)
}
defer agentConn.Close()
```

4. Update the `RegisterDemoServiceServer` call to pass `agentConn`:

```go
pb.RegisterDemoServiceServer(srv, &gatewayServer{ragConn: ragConn, llmConn: llmConn, agentConn: agentConn})
```

5. Add health status for the agent service:

```go
healthSrv.SetServingStatus("demo.AgentService", healthpb.HealthCheckResponse_SERVING)
```

- [ ] **Step 2: Build the gateway to verify compilation**

```bash
cd gateway && go build -o /dev/null .
```

Expected: successful build, no errors.

- [ ] **Step 3: Commit**

```bash
git add gateway/main.go
git commit -m "feat: add agent service routing to gateway"
```

---

## Task 8: Traffic Generator Changes

**Files:**
- Modify: `traffic-gen/main.py`

- [ ] **Step 1: Add agent queries and three-way rotation**

In `traffic-gen/main.py`:

1. Add `AGENT_MESSAGES` list after `CHAT_MESSAGES`:

```python
AGENT_MESSAGES = [
    "Search our docs for what a Kubernetes pod is, then tell me the current time",
    "Calculate 256 * 384 and search for information about container runtimes",
    "What time is it right now?",
    "Search the docs about distributed tracing and also look up what OpenTelemetry is on the web",
    "Calculate 1024 / 16 and tell me the current time",
    "Search our knowledge base for information about container images",
]
```

2. Replace the query interleaving logic (lines 86-91) with three-way rotation:

```python
    query_types = ["rag", "chat", "agent"]
    query_lists = {
        "rag": RAG_QUERIES,
        "chat": CHAT_MESSAGES,
        "agent": AGENT_MESSAGES,
    }
    query_indices = {"rag": 0, "chat": 0, "agent": 0}

    type_idx = 0
```

3. Replace the main loop body (lines 96-112) with:

```python
    while running:
        query_type = query_types[type_idx % len(query_types)]
        type_idx += 1
        query_list = query_lists[query_type]
        query_text = query_list[query_indices[query_type] % len(query_list)]
        query_indices[query_type] += 1

        try:
            if query_type == "rag":
                logger.info(f"Sending RAG query: {query_text}")
                resp = stub.Query(demo_pb2.QueryRequest(query=query_text, top_k=3), timeout=120)
                logger.info(f"RAG response model={resp.model}, sources={len(resp.sources)}")
            elif query_type == "chat":
                logger.info(f"Sending Chat message: {query_text}")
                resp = stub.Chat(demo_pb2.ChatRequest(message=query_text), timeout=120)
                logger.info(f"Chat response model={resp.model}")
            elif query_type == "agent":
                logger.info(f"Sending Agent message: {query_text}")
                resp = stub.AgentChat(demo_pb2.AgentChatRequest(message=query_text), timeout=180)
                logger.info(f"Agent response model={resp.model}, tools_used={len(resp.tool_calls_made)}")
        except grpc.RpcError as e:
            logger.warning(f"gRPC error: {e.code()} {e.details()}")
        except Exception as e:
            logger.warning(f"Error: {e}")

        jitter = random.uniform(0, 2)
        sleep_time = interval + jitter
        start = time.monotonic()
        while running and (time.monotonic() - start) < sleep_time:
            time.sleep(0.5)
```

- [ ] **Step 2: Commit**

```bash
git add traffic-gen/main.py
git commit -m "feat: add agent queries to traffic generator with three-way rotation"
```

---

## Task 9: Helm Chart

**Files:**
- Create: `helm/suse-ai-demo/templates/agent-service-deployment.yaml`
- Create: `helm/suse-ai-demo/templates/agent-service-service.yaml`
- Modify: `helm/suse-ai-demo/values.yaml`

- [ ] **Step 1: Create agent-service-deployment.yaml**

Follows the exact pattern of `llm-service-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-service
  labels:
    app: agent-service
spec:
  replicas: {{ .Values.agentService.replicas }}
  selector:
    matchLabels:
      app: agent-service
  template:
    metadata:
      labels:
        app: agent-service
    spec:
      containers:
        - name: agent-service
          image: "{{ .Values.agentService.image.repository }}:{{ .Values.agentService.image.tag }}"
          imagePullPolicy: {{ .Values.agentService.image.pullPolicy }}
          ports:
            - containerPort: 50054
              protocol: TCP
          env:
            - name: GRPC_LISTEN_ADDR
              value: "[::]:50054"
            - name: LLM_BASE_URL
              value: {{ .Values.agentService.llm.baseUrl | quote }}
            - name: LLM_MODEL
              value: {{ .Values.agentService.llm.model | quote }}
            - name: LLM_PROVIDER
              value: {{ .Values.agentService.llm.provider | quote }}
            - name: RAG_SERVICE_ADDR
              value: {{ .Values.agentService.ragServiceAddr | quote }}
            - name: AGENT_NAME
              value: {{ .Values.agentService.agentName | quote }}
            - name: AGENT_MAX_ITERATIONS
              value: {{ .Values.agentService.maxIterations | quote }}
            - name: ENABLE_OTEL_CONTENT_EVENTS
              value: {{ .Values.agentService.enableContentEvents | quote }}
            - name: OTEL_EXPORTER_OTLP_ENDPOINT
              value: {{ .Values.otel.exporterEndpoint | quote }}
            - name: OTEL_SERVICE_NAME
              value: "agent-service"
          readinessProbe:
            grpc:
              port: 50054
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            tcpSocket:
              port: 50054
            initialDelaySeconds: 5
            periodSeconds: 15
          resources:
            {{- toYaml .Values.agentService.resources | nindent 12 }}
```

- [ ] **Step 2: Create agent-service-service.yaml**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: agent-service
spec:
  selector:
    app: agent-service
  ports:
    - port: 50054
      targetPort: 50054
      protocol: TCP
```

- [ ] **Step 3: Add agentService section to values.yaml**

Add after the `llmService` section in `helm/suse-ai-demo/values.yaml`:

```yaml
agentService:
  image:
    repository: ghcr.io/thbertoldi/suse-ai-demo-agent-service
    tag: latest
    pullPolicy: Always
  replicas: 1
  llm:
    baseUrl: http://vllm-router-service.suse-private-ai.svc.cluster.local:80/v1
    model: llama3
    provider: vllm
  ragServiceAddr: rag-service:50052
  agentName: demo-agent
  maxIterations: "5"
  enableContentEvents: "false"
  resources:
    requests:
      cpu: 100m
      memory: 256Mi
    limits:
      cpu: 500m
      memory: 512Mi
```

Note: memory is higher than other services (256Mi/512Mi vs 128Mi/256Mi) due to LangGraph/LangChain overhead.

- [ ] **Step 4: Add agentServiceAddr to gateway values and deployment**

In `helm/suse-ai-demo/values.yaml`, add to the `gateway` section:

```yaml
  agentServiceAddr: agent-service:50054
```

In `helm/suse-ai-demo/templates/gateway-deployment.yaml`, add to the `env` section:

```yaml
            - name: AGENT_SERVICE_ADDR
              value: {{ .Values.gateway.agentServiceAddr | quote }}
```

- [ ] **Step 5: Commit**

```bash
git add helm/suse-ai-demo/templates/agent-service-deployment.yaml \
  helm/suse-ai-demo/templates/agent-service-service.yaml \
  helm/suse-ai-demo/values.yaml \
  helm/suse-ai-demo/templates/gateway-deployment.yaml
git commit -m "feat: add agent service to Helm chart"
```

---

## Task 10: GitHub Actions

**Files:**
- Modify: `.github/workflows/build.yaml`

- [ ] **Step 1: Add agent-service to build matrix**

In `.github/workflows/build.yaml`, add to the `matrix.include` list:

```yaml
          - service: agent-service
            context: ./agent-service
            image: suse-ai-demo-agent-service
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/build.yaml
git commit -m "ci: add agent-service to container image build matrix"
```

---

## Task 11: README Update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

1. Update the architecture diagram to include the Agent Service (add after the LLM Service line):

```
                              | gRPC
                              '-------> [Agent Service (Python)]
                                              |
                                              | HTTP (OpenAI-compat)
                                              +-------> [vLLM / Ollama]
                                              |
                                              | gRPC (search_docs tool)
                                              '-------> [RAG Service]
```

2. Add Agent Service row to the service table:

```
| **Agent Service** | Python | LangGraph ReAct agent with tools (search_docs, calculate, web_search, get_current_time). Produces `invoke_agent` and `execute_tool` OTel spans. | 50054 |
```

3. Add agent service traces section under OpenTelemetry Instrumentation > Traces:

```
- **Agent spans**: `invoke_agent {agent_name}` with `gen_ai.agent.name`, `gen_ai.agent.id`
- **Tool execution spans**: `execute_tool {tool_name}` with `gen_ai.tool.name`, `gen_ai.tool.type`, `gen_ai.tool.call.id`
```

4. Add Agent Service configuration table.

5. Add `agent-service/` to the project structure tree.

6. Add agent service build command to the Building section:

```bash
docker build -t suse-ai-demo-agent-service ./agent-service
```

7. Add agent service local run instructions.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add agent service to README"
```
