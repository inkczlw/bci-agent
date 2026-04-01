"""Agent 响应流式输出。

支持 token-by-token 的流式响应，降低用户感知延迟（Time to First Token）。
与 FastAPI 的 SSE（Server-Sent Events）衔接使用。

"""

import time
import sys
from typing import AsyncGenerator, Generator
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class StreamingHandler(BaseCallbackHandler):
    """实时打印 LLM 生成的 token。

    用于终端交互模式——Agent 一边生成一边输出，而不是等全部完成再打印。
    """

    def __init__(self, print_fn=None):
        """
        Args:
            print_fn: 自定义输出函数。默认 sys.stdout.write（实现逐字打印）。
                      FastAPI 场景下替换为 SSE 推送函数。
        """
        self.print_fn = print_fn or self._default_print
        self.token_count = 0
        self.first_token_time: float | None = None
        self.start_time: float | None = None
        self._is_final_response = False  # 标记是否在最终回答阶段

    def on_llm_start(self, *args, **kwargs):
        """LLM 开始推理。"""
        self.start_time = time.time()
        self.token_count = 0
        self.first_token_time = None

    def on_llm_new_token(self, token: str, **kwargs):
        """收到一个新 token。这是流式输出的核心回调。"""
        if self.first_token_time is None:
            self.first_token_time = time.time()

        self.token_count += 1

        # 只在最终回答阶段打印（跳过中间的 tool call 决策 token）
        # 注意：在 ReAct 循环中，LLM 会多次被调用：
        #   - 前几次是 tool selection（输出 function call JSON）
        #   - 最后一次是最终回答（输出自然语言）
        # 我们只 stream 最终回答，tool selection 的 token 不输出
        self.print_fn(token)

    def on_llm_end(self, response: LLMResult, **kwargs):
        """LLM 完成推理。"""
        if self.first_token_time and self.start_time:
            ttft = round((self.first_token_time - self.start_time) * 1000, 2)
            total = round((time.time() - self.start_time) * 1000, 2)

    def get_streaming_stats(self) -> dict:
        """返回本次 streaming 的统计信息。"""
        stats = {"token_count": self.token_count}
        if self.first_token_time and self.start_time:
            stats["ttft_ms"] = round((self.first_token_time - self.start_time) * 1000, 2)
            stats["total_ms"] = round((time.time() - self.start_time) * 1000, 2)
            if self.token_count > 0:
                stats["tokens_per_second"] = round(
                    self.token_count / ((time.time() - self.first_token_time) or 0.001), 2
                )
        return stats

    @staticmethod
    def _default_print(token: str):
        """默认输出：逐字打印到终端。"""
        sys.stdout.write(token)
        sys.stdout.flush()


def stream_agent_response(agent, messages: list[dict], callbacks: list = None) -> Generator[str, None, None]:
    """同步 generator：逐步 yield Agent 的 streaming 输出。

    用法：
        for event in stream_agent_response(agent, messages):
            print(event)  # 每个 event 是一个中间状态或最终 token

    这个函数用 LangGraph 的 stream_mode="updates"，
    逐步返回每个 node 的执行结果。
    """
    config = {}
    if callbacks:
        config["callbacks"] = callbacks

    for event in agent.stream({"messages": messages}, config=config, stream_mode="updates"):
        # event 格式: {"node_name": {"messages": [...]}}
        for node_name, node_output in event.items():
            if "messages" in node_output:
                for msg in node_output["messages"]:
                    yield {
                        "node": node_name,
                        "type": type(msg).__name__,
                        "content": getattr(msg, "content", ""),
                    }


async def astream_agent_response(agent, messages: list[dict], callbacks: list = None) -> AsyncGenerator[dict, None]:
    """异步 generator：用于 FastAPI 的 SSE 推流。

    用法（FastAPI）：
        @app.post("/chat/stream")
        async def chat_stream(request: ChatRequest):
            async def event_generator():
                async for event in astream_agent_response(agent, messages):
                    yield f"data: {json.dumps(event)}\\n\\n"
            return StreamingResponse(event_generator(), media_type="text/event-stream")
    """
    config = {}
    if callbacks:
        config["callbacks"] = callbacks

    async for event in agent.astream({"messages": messages}, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            if "messages" in node_output:
                for msg in node_output["messages"]:
                    yield {
                        "node": node_name,
                        "type": type(msg).__name__,
                        "content": getattr(msg, "content", ""),
                    }