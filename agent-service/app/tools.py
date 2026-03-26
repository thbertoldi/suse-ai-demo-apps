import json
import random
import time
from datetime import datetime, timezone

import grpc
from simpleeval import simple_eval
from opentelemetry import trace

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
