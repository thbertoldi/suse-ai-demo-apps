# Agent Service — Design Spec

## Overview

A new Python microservice (`agent-service`) that runs a LangGraph-based ReAct agent with tools, integrated into the existing SUSE AI demo architecture behind the Go Gateway. The agent consumes the RAG service, provides additional tools (calculate, web_search, get_current_time), and generates rich OpenTelemetry traces following the GenAI semantic conventions — including the newer `invoke_agent` and `execute_tool` span types.

The primary goal is producing visually interesting, standards-compliant agentic traces in the observability backend, not a specific use case.

## Architecture

```
                    gRPC                  gRPC                HTTP
  [Traffic Gen] -------> [Gateway (Go)] -------> [RAG (Python)] -------> [Qdrant]
                              |                      |
                              |                      | HTTP (OpenAI-compat)
                              |                      '-------> [Ollama]
                              |
                              | gRPC
                              +-------> [LLM Service (Python)]
                              |               '-------> [vLLM]
                              |
                              | gRPC
                              '-------> [Agent Service (Python)]
                                              |
                                              | HTTP (OpenAI-compat)
                                              +-------> [vLLM / Ollama]
                                              |
                                              | gRPC (search_docs tool)
                                              '-------> [RAG Service]
```

### Integration Points

- **Gateway** gains a new `DemoService.AgentChat` RPC that routes to `AgentService.Run`
- **Agent service** calls the RAG service directly via gRPC (for the `search_docs` tool)
- **Agent service** calls the LLM backend via OpenAI-compat HTTP API (through `ChatOpenAI` from langchain-openai)
- **Traffic generator** adds agent queries to its round-robin rotation (RAG → Chat → Agent)

### Proto Changes

Add to `proto/demo.proto`:

```protobuf
// In DemoService:
rpc AgentChat(AgentChatRequest) returns (AgentChatResponse);

// New service:
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

Port: **50054** (gateway=50051, rag=50052, llm=50053).

### Proto Code Generation

Proto stubs are regenerated via the existing `proto/Makefile` (`make all`), which produces Go stubs in `gateway/pb/` and Python stubs in `{rag,llm,agent}-service/app/generated/` and `traffic-gen/generated/`. The Makefile must be updated to include the agent-service output path and the corresponding `sed` fixup for relative imports in `demo_pb2_grpc.py` (matching the existing fixup lines for rag-service and llm-service).

## Agent Service Internals

### LangGraph Structure

A ReAct graph with two nodes:

```
START → llm_call → should_continue? → tool_node → llm_call (loop)
                         |
                         '→ END (no tool calls)
```

- **`llm_call` node**: Invokes `ChatOpenAI` with tools bound. Produces a `chat {model}` OTel span.
- **`tool_node` node**: Iterates over tool calls from the LLM response, executes each, returns `ToolMessage`s. Each tool execution produces an `execute_tool {tool_name}` OTel span.
- **`should_continue`**: Conditional edge — if the last message has tool calls, route to `tool_node`; otherwise, route to `END`.
- **Max iterations**: Capped at 5 LLM calls per request.

### LLM Integration

Uses `ChatOpenAI` from `langchain-openai`, pointed at a configurable OpenAI-compat endpoint. This handles tool binding and tool call parsing natively. The provider can be vLLM, Ollama, or any OpenAI-compat server.

### Tools

1. **`search_docs`** — Calls the RAG service via gRPC (`RAGService.Retrieve`). Takes a `query` string and optional `top_k`. Note: `Retrieve` runs the full RAG pipeline (embed → vector search → LLM completion) and returns both `answer` and `sources`. The `search_docs` tool returns `resp.sources` (the retrieved document snippets) to the agent, discarding `resp.answer`. This means the agent's LLM re-synthesizes from the raw chunks rather than passing through the RAG service's answer — producing a richer trace with two distinct LLM calls visible (one in RAG service, one in agent).

2. **`calculate`** — Evaluates a simple math expression using the `simpleeval` library (safe sandboxed evaluation supporting arithmetic operators, comparisons, and basic math functions). Takes an `expression` string, returns the numeric result.

3. **`web_search`** — Simulated web search. Takes a `query` string, returns 2-3 fake search results with titles/snippets after a small random delay (0.5-1.5s) to simulate external latency.

4. **`get_current_time`** — Returns the current UTC timestamp. Trivial but shows the agent deciding when a tool is needed vs not.

### System Prompt

Generic: "You are a helpful assistant with access to tools. Use them when needed to answer the user's question accurately."

## OpenTelemetry Instrumentation

### Approach

Hybrid: LangGraph handles the agent loop and tool binding. Manual OTel spans wrap agent invocation, LLM calls, and tool executions to follow the GenAI semantic conventions precisely.

### Context Propagation Strategy

The `invoke_agent` span is started before calling `graph.invoke()`. Since LangGraph's synchronous `invoke()` runs on the calling thread, the OTel context is automatically propagated to node functions. Node functions create child spans using `tracer.start_as_current_span()` which inherits the active context. This ensures `chat` and `execute_tool` spans are children of `invoke_agent` without explicit context passing.

Note: `ChatOpenAI` internally uses `httpx` (not `requests`). The `opentelemetry-instrumentation-httpx` dependency is included to auto-instrument these outbound HTTP calls. LangChain's own tracing callbacks are NOT used — all OTel spans are created manually to ensure precise semantic convention compliance.

### Span Hierarchy

Typical agent request trace:

```
invoke_agent demo-agent (INTERNAL, gen_ai.agent.name=demo-agent)
  ├── chat {model} (CLIENT)                    ← LLM decides to use tools
  │     ├── gen_ai.input.messages event         (opt-in)
  │     └── gen_ai.output.messages event        (opt-in)
  ├── execute_tool search_docs (INTERNAL)
  │     ├── gen_ai.tool.name = search_docs
  │     ├── gen_ai.tool.call.id = call_xxx
  │     ├── gen_ai.tool.call.arguments attr     (opt-in)
  │     ├── gen_ai.tool.call.result attr        (opt-in)
  │     └── [child gRPC span → RAG service → embed + vector search + chat]
  ├── execute_tool calculate (INTERNAL)
  │     ├── gen_ai.tool.name = calculate
  │     ├── gen_ai.tool.type = function
  │     └── gen_ai.tool.call.result attr
  ├── chat {model} (CLIENT)                    ← LLM produces final answer
  └── gen_ai.usage metrics recorded
```

### Span Attributes

| Span | `gen_ai.operation.name` | Kind | Key Attributes |
|------|------------------------|------|----------------|
| Agent invocation | `invoke_agent` | `INTERNAL` | `gen_ai.agent.name`, `gen_ai.agent.id` (UUID generated per request), `gen_ai.request.model` |
| LLM call | `chat` | `CLIENT` | `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.provider.name`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.response.finish_reasons`, `gen_ai.response.id` |
| Tool execution | `execute_tool` | `INTERNAL` | `gen_ai.tool.name`, `gen_ai.tool.type` (`function`), `gen_ai.tool.call.id`, `gen_ai.tool.description` |

### Content Capture (opt-in via `ENABLE_OTEL_CONTENT_EVENTS`)

- **LLM call spans**: `gen_ai.input.messages` / `gen_ai.output.messages` as span **events** (same pattern as existing services)
- **Tool execution spans**: `gen_ai.tool.call.arguments` / `gen_ai.tool.call.result` as span **attributes** (per GenAI semantic conventions — these are attributes on `execute_tool` spans, not events)

### Metrics

Same instruments as existing services:
- `gen_ai.client.token.usage` — histogram, attributes: `gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.token.type`
- `gen_ai.client.operation.duration` — histogram, attributes: `gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.provider.name`

### Error Handling

- LLM HTTP errors → `chat` span status set to `ERROR`, `error.type` attribute set
- Tool execution errors → `execute_tool` span status set to `ERROR`, `error.type` attribute set
- Agent-level errors → `invoke_agent` span status set to `ERROR`
- Max iterations exceeded → agent returns best available answer, span gets attribute indicating truncation
- gRPC errors → auto-instrumentation records status code

### OTel Setup

Reuses the same `otel_setup.py` pattern: OTLP/gRPC exporter, auto-instrument gRPC server/client and HTTP/httpx requests, W3C TraceContext propagation. Logging uses `logging.basicConfig(level=logging.INFO)` consistent with existing services.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GRPC_LISTEN_ADDR` | `[::]:50054` | gRPC listen address |
| `LLM_BASE_URL` | `http://vllm:8000/v1` | LLM endpoint (OpenAI-compat) |
| `LLM_MODEL` | `llama3` | LLM model name |
| `LLM_PROVIDER` | `vllm` | Provider name for OTel attributes |
| `RAG_SERVICE_ADDR` | `rag-service:50052` | RAG service gRPC address (for search_docs tool) |
| `AGENT_NAME` | `demo-agent` | Agent name for OTel `gen_ai.agent.name` |
| `AGENT_MAX_ITERATIONS` | `5` | Max LLM calls per request |
| `ENABLE_OTEL_CONTENT_EVENTS` | `false` | Enable content/tool argument events |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://otel-collector:4317` | OTel collector |
| `OTEL_SERVICE_NAME` | `agent-service` | Service name for telemetry |

## Project Structure

```
agent-service/
├── app/
│   ├── __init__.py
│   ├── main.py                  # Entrypoint, OTel init, gRPC server
│   ├── otel_setup.py            # Same pattern as existing services
│   ├── grpc_server.py           # AgentServiceServicer
│   ├── agent.py                 # LangGraph graph definition
│   ├── tools.py                 # Tool definitions
│   ├── otel_instrumentation.py  # Manual span wrappers (invoke_agent, execute_tool, chat)
│   │                            # Separated from llm_client.py because the agent has three
│   │                            # distinct span types vs one in existing services
│   └── generated/               # Proto stubs
├── requirements.txt
└── Dockerfile
```

## Service Lifecycle

### Startup
- Initialize OTel (tracer, meter, propagation, auto-instrumentation)
- Open a shared gRPC channel to the RAG service (reused across requests, not per-request)
- Compile the LangGraph graph
- Start the gRPC server with health checking (`grpc.health.v1.Health`)

### Shutdown
- Handle SIGTERM/SIGINT signals
- Stop the gRPC server gracefully
- Close the RAG service gRPC channel
- Flush and shutdown OTel (`tracer_provider.shutdown()`, `meter_provider.shutdown()`)

### Dockerfile
Follows the existing pattern: `python:3.12-slim`, `pip install -r requirements.txt`, `CMD ["python", "-m", "app.main"]`. LangGraph/LangChain dependencies increase image size compared to other services but no special build considerations are needed.

## Dependencies

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

Note: `ChatOpenAI` uses `httpx` internally (not `requests`), so `opentelemetry-instrumentation-httpx` is required to capture outbound LLM HTTP calls. Unlike the existing services which use `requests` directly, this service does not need `opentelemetry-instrumentation-requests`. Pinned versions should be verified with `pip-compile` or equivalent to ensure compatibility between langchain packages.

## Changes to Existing Services

### Gateway (Go)
- New `DemoService.AgentChat` RPC routing to `AgentService.Run`
- New env var `AGENT_SERVICE_ADDR` (default `agent-service:50054`)
- New gRPC client connection to agent service with OTel interceptors

### Traffic Generator (Python)
- Third query type added: agent queries
- The existing interleaving logic (alternating RAG/Chat by index) is replaced with a simple three-way rotation: maintain three separate query lists (RAG, Chat, Agent) and cycle through types in order (RAG → Chat → Agent → RAG → ...), picking the next query from each list round-robin
- Agent queries designed to trigger tool use:
  - "Search our docs for what a Kubernetes pod is, then tell me the current time"
  - "Calculate 256 * 384 and search for information about container runtimes"
  - "What time is it right now?"

### Proto (`demo.proto`)
- New `AgentChat` RPC in `DemoService`
- New `AgentService` with `Run` RPC
- New messages: `AgentChatRequest`, `AgentChatResponse`, `AgentRequest`, `AgentResponse`, `AgentToolCall`

### Helm Chart
- New `agent-service-deployment.yaml` and `agent-service-service.yaml` templates
- New `agentService` section in `values.yaml`

### GitHub Actions
- Add `agent-service` to the build matrix in `.github/workflows/build.yaml`
