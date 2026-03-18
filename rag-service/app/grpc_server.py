import os
import logging

import grpc

from app.generated import demo_pb2, demo_pb2_grpc
from app.llm_client import chat_completion
from app.embedding_client import embed
from app.vectordb import VectorStore

logger = logging.getLogger(__name__)


class RAGServiceServicer(demo_pb2_grpc.RAGServiceServicer):
    def __init__(self, store: VectorStore):
        self._store = store
        self._llm_base_url = os.environ.get("LLM_BASE_URL", "http://ollama:11434/v1")
        self._llm_model = os.environ.get("LLM_MODEL", "llama3")
        self._llm_provider = os.environ.get("LLM_PROVIDER", "ollama")
        self._embedding_base_url = os.environ.get("EMBEDDING_BASE_URL", "http://ollama:11434/v1")
        self._embedding_model = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")

    def Retrieve(self, request, context):
        top_k = request.top_k if request.top_k > 0 else 3
        query = request.query

        try:
            query_embedding = embed(
                base_url=self._embedding_base_url,
                model=self._embedding_model,
                provider=self._llm_provider,
                text=query,
            )

            docs = self._store.search(query_embedding, top_k=top_k)
            sources = [doc.content for doc in docs]

            if sources:
                context_text = "\n\n".join(sources)
                prompt = f"Based on the following context, answer the question.\n\nContext:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
            else:
                prompt = query

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
            logger.exception("Error in Retrieve")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return demo_pb2.RetrieveResponse()
