import os
import logging
import time

import grpc
from opentelemetry import metrics

from app.generated import demo_pb2, demo_pb2_grpc
from app.llm_client import chat_completion
from app.embedding_client import embed
from app.vectordb import VectorStore

logger = logging.getLogger(__name__)
meter = metrics.get_meter("suse-ai.rag")
retrieved_documents = meter.create_histogram(
    "suse.ai.rag.retrieval.documents",
    unit="1",
    description="Number of documents returned by vector retrieval.",
)
retrieval_top_score = meter.create_histogram(
    "suse.ai.rag.retrieval.top_score",
    unit="1",
    description="Score of the highest-ranked retrieved document.",
)
retrieval_no_hits = meter.create_counter(
    "suse.ai.rag.retrieval.no_hits",
    unit="{request}",
    description="RAG requests for which retrieval returned no documents.",
)
context_size = meter.create_histogram(
    "suse.ai.rag.context.size",
    unit="By",
    description="UTF-8 size of context supplied to the LLM.",
)
request_duration = meter.create_histogram(
    "suse.ai.rag.request.duration",
    unit="s",
    description="End-to-end RAG request duration.",
)


class RAGServiceServicer(demo_pb2_grpc.RAGServiceServicer):
    def __init__(self, store: VectorStore):
        self._store = store
        self._llm_base_url = os.environ.get("LLM_BASE_URL", "http://ollama:11434/v1")
        self._llm_model = os.environ.get("LLM_MODEL", "llama3")
        self._llm_provider = os.environ.get("LLM_PROVIDER", "ollama")
        self._embedding_base_url = os.environ.get("EMBEDDING_BASE_URL", "http://ollama:11434")
        self._embedding_model = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")

    def Retrieve(self, request, context):
        started = time.monotonic()
        outcome = "success"
        top_k = request.top_k if request.top_k > 0 else 3
        query = request.query
        docs = []

        try:
            query_embedding = embed(
                base_url=self._embedding_base_url,
                model=self._embedding_model,
                provider=self._llm_provider,
                text=query,
            )

            docs = self._store.search(query_embedding, top_k=top_k)
            sources = [doc.content for doc in docs]
            metric_attributes = {"provider": self._llm_provider}
            retrieved_documents.record(len(docs), metric_attributes)
            if docs:
                retrieval_top_score.record(float(docs[0].score), metric_attributes)
            else:
                retrieval_no_hits.add(1, metric_attributes)

            if sources:
                context_text = "\n\n".join(sources)
                prompt = f"Based on the following context, answer the question.\n\nContext:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
            else:
                context_text = ""
                prompt = query
            context_size.record(
                len(context_text.encode("utf-8")),
                metric_attributes,
            )

            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer questions based on the provided context. If no context is provided, answer based on your general knowledge."},
                {"role": "user", "content": prompt},
            ]
            result = chat_completion(
                base_url=self._llm_base_url,
                model=self._llm_model,
                provider=self._llm_provider,
                messages=messages,
            )

            answer = result["choices"][0]["message"]["content"]
            model = result.get("model", self._llm_model)

            return demo_pb2.RetrieveResponse(
                answer=answer,
                sources=sources,
                model=model,
            )
        except Exception as e:
            outcome = "error"
            logger.exception("Error in Retrieve")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return demo_pb2.RetrieveResponse()
        finally:
            request_duration.record(
                time.monotonic() - started,
                {"provider": self._llm_provider, "outcome": outcome},
            )
