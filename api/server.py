"""BCI Agent HTTP API 服务。

用 FastAPI 把 Agent 包装成 HTTP 服务，支持：
- POST /chat       — 同步调用，返回完整响应
- POST /chat/stream — SSE 流式推送
- GET  /health     — 健康检查
- GET  /metrics    — 运行指标


启动方式：
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import time
import signal
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.bci_agent import create_bci_agent
from utils.tracer import AgentTracer
from utils.metrics import MetricsCollector
from utils.streaming import astream_agent_response
from config import get_llm

logger = logging.getLogger("bci_agent.api")


# ── Request / Response 模型 ──────────────────────────────────

class ChatRequest(BaseModel):
    """聊天请求。"""
    query: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    session_id: str = Field(default="default", description="会话 ID（用于记忆隔离）")
    stream: bool = Field(default=False, description="是否使用流式响应")


class ChatResponse(BaseModel):
    """聊天响应。"""
    answer: str
    session_id: str
    trace_id: str | None = None
    duration_ms: float
    tool_calls: list[str] = []
    token_usage: dict = {}


class HealthResponse(BaseModel):
    """健康检查响应。"""
    status: str  # "healthy" | "degraded" | "unhealthy"
    uptime_seconds: float
    agent_ready: bool
    llm_available: bool


# ── 全局状态 ─────────────────────────────────────────────────

class AppState:
    """应用级状态。lifespan 里初始化，请求里读取。"""
    agent = None
    tracer: AgentTracer | None = None
    metrics: MetricsCollector | None = None
    start_time: float = 0.0
    is_shutting_down: bool = False
    # 简易 rate limiter: {client_ip: [timestamp, ...]}
    request_timestamps: dict[str, list[float]] = {}

state = AppState()

# Rate limit 配置
RATE_LIMIT_REQUESTS = 20      # 每个 IP 最多
RATE_LIMIT_WINDOW = 60        # 在多少秒内


# ── Lifespan（启动/关闭钩子）────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 的生命周期管理。

    - startup: 初始化 Agent、Tracer、Metrics
    - shutdown: 优雅关闭（flush logs、保存 metrics）

    """
    # ── Startup ──
    logger.info("Initializing BCI Agent API...")
    state.start_time = time.time()

    try:
        state.agent = create_bci_agent()
        state.tracer = AgentTracer(log_dir="traces")
        state.metrics = MetricsCollector()
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise

    yield  # <-- 应用运行期间在这里暂停

    # ── Shutdown ──
    logger.info("Shutting down BCI Agent API...")
    state.is_shutting_down = True

    # 保存最终 metrics
    if state.metrics:
        final_stats = state.metrics.to_dict()
        logger.info(f"Final metrics: {json.dumps(final_stats)}")

    logger.info("Shutdown complete")


# ── App 实例 ─────────────────────────────────────────────────

app = FastAPI(
    title="BCI Agent API",
    description="Brain-Computer Interface 行业分析 Agent 服务",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS（开发阶段允许所有来源，生产环境需要限制）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 中间件：Rate Limiting ────────────────────────────────────

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """简易 IP 限流。

    生产环境用 Redis + sliding window 或 token bucket。
    这里用内存 dict 做演示。
    """
    if state.is_shutting_down:
        return JSONResponse(
            status_code=503,
            content={"detail": "Service is shutting down"},
        )

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    # 只对 /chat 端点限流
    if request.url.path.startswith("/chat"):
        timestamps = state.request_timestamps.get(client_ip, [])
        # 清理窗口外的请求
        timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]

        if len(timestamps) >= RATE_LIMIT_REQUESTS:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Rate limit exceeded: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s",
                    "retry_after": RATE_LIMIT_WINDOW,
                },
            )

        timestamps.append(now)
        state.request_timestamps[client_ip] = timestamps

    response = await call_next(request)
    return response


# ── Endpoints ────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查。

    Kubernetes 的 readiness/liveness probe 会调这个端点。
    返回 Agent 和 LLM 的可用状态。
    """
    agent_ready = state.agent is not None
    llm_available = True  # TODO: 实际检测 LLM API 可达性

    status = "healthy" if (agent_ready and llm_available) else "degraded"
    if not agent_ready:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        uptime_seconds=round(time.time() - state.start_time, 2),
        agent_ready=agent_ready,
        llm_available=llm_available,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """同步聊天接口。等 Agent 完成后一次性返回结果。"""
    if state.agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    start = time.time()
    tracer = state.tracer

    try:
        if tracer:
            tracer.start_trace(request.query)

        result = state.agent.invoke(
            {"messages": [{"role": "user", "content": request.query}]},
            config={"callbacks": [tracer] if tracer else []},
        )

        trace = tracer.end_trace() if tracer else None

        ai_response = result["messages"][-1].content
        duration = round((time.time() - start) * 1000, 2)

        # 收集 trace 信息
        tool_calls_list = []
        token_usage = {}
        if trace:
            tool_calls_list = [
                s.name for s in trace.spans if s.span_type == "tool"
            ]
            token_usage = {
                "total": sum(s.token_usage.get("total_tokens", 0) for s in trace.spans),
            }
            # 喂给 MetricsCollector
            if state.metrics:
                state.metrics.record_trace(trace)

        return ChatResponse(
            answer=ai_response,
            session_id=request.session_id,
            trace_id=trace.trace_id if trace else None,
            duration_ms=duration,
            tool_calls=tool_calls_list,
            token_usage=token_usage,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE 流式聊天接口。

    返回 Server-Sent Events 流，前端用 EventSource 接收。
    每个 event 包含一个 Agent 执行步骤或 token。
    """
    if state.agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for event in astream_agent_response(
                state.agent,
                [{"role": "user", "content": request.query}],
            ):
                # SSE 格式: "data: {json}\n\n"
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            # 发送结束信号
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁止 nginx 缓冲
        },
    )


@app.get("/metrics")
async def get_metrics():
    """运行指标。

    返回 AgentTracer + MetricsCollector 的聚合数据。
    生产环境会接 Prometheus exporter。
    """
    result = {}

    if state.metrics:
        result["agent_metrics"] = state.metrics.to_dict()

    if state.tracer:
        result["tracer_stats"] = state.tracer.get_stats()

    result["rate_limiter"] = {
        "active_clients": len(state.request_timestamps),
        "config": {
            "max_requests": RATE_LIMIT_REQUESTS,
            "window_seconds": RATE_LIMIT_WINDOW,
        },
    }

    return result


# ── Graceful Shutdown ────────────────────────────────────────

def setup_signal_handlers():
    """注册信号处理器，支持优雅关闭。

    收到 SIGTERM 后要：
    1. 停止接收新请求（rate limiter 返回 503）
    2. 等待正在处理的请求完成
    3. flush 日志和 metrics
    4. 退出
    """
    def handle_shutdown(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        state.is_shutting_down = True

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)


# ── 启动入口 ─────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    setup_signal_handlers()
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,           # 开发模式热重载
        log_level="info",
    )