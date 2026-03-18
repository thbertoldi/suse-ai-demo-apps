import os
import logging

import grpc

from app.generated import demo_pb2, demo_pb2_grpc
from app.llm_client import chat_completion

logger = logging.getLogger(__name__)


class LLMServiceServicer(demo_pb2_grpc.LLMServiceServicer):
    def __init__(self):
        self._base_url = os.environ.get("LLM_BASE_URL", "http://vllm:8000/v1")
        self._model = os.environ.get("LLM_MODEL", "llama3")
        self._provider = os.environ.get("LLM_PROVIDER", "vllm")

    def Generate(self, request, context):
        try:
            messages = [
                {"role": "user", "content": request.prompt},
            ]
            result = chat_completion(
                base_url=self._base_url,
                model=self._model,
                provider=self._provider,
                messages=messages,
            )
            text = result["choices"][0]["message"]["content"]
            model = result.get("model", self._model)

            return demo_pb2.GenerateResponse(text=text, model=model)
        except Exception as e:
            logger.exception("Error in Generate")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return demo_pb2.GenerateResponse()
