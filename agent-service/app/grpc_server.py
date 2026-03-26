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
