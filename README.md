# BCI Industry Analysis Agent

A production-grade AI Agent for brain-computer interface (BCI) industry analysis, built with LangGraph + LangChain + DeepSeek.

This project demonstrates end-to-end Agent engineering: from ReAct architecture and RAG retrieval to observability, security hardening, and automated quality evaluation — with a systems engineering perspective rooted in C++ concurrency and performance optimization.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  SecurityGuard                      │
│  InputValidator → InjectionDetector │
│  → ToolSandbox                      │
└─────────────┬───────────────────────┘
              ▼
┌─────────────────────────────────────┐
│  FastAPI Service                    │
│  /chat  /chat/stream  /health      │
│  /metrics  (rate limiting)         │
└──────┬──────────────┬───────────────┘
       ▼              ▼
   Sync Agent    AsyncTaskQueue
   (ReAct)       (BatchAnalyzer)
       │              │
       ▼              ▼
┌─────────────────────────────────────┐
│  LLM Optimization Layer            │
│  Concurrent tools │ SSE streaming  │
│  L1/L2 LLM cache                  │
└─────────────┬───────────────────────┘
              ▼
┌──────┬──────┬──────┬──────┬────────┐
│search│ news │ RAG  │analyz│compare │  ← 5 Tools
└──────┴──────┴──┬───┴──────┴────────┘
                 ▼
         ChromaDB (163 chunks)
         Memory (buffer/summary/entity/vector)
         
Observability: AgentTracer + Structured Logging + MetricsCollector
Evaluation: 11 test cases × 7 dimensions + LLM-as-judge
```

## Key Features

**Agent Core** — ReAct architecture via LangGraph `create_react_agent`. 5 tools across 3 categories: static data lookup, RAG retrieval, structured analysis. Tool registration via `@register` decorator pattern with per-tool timeout and fallback.

**RAG Pipeline** — PDF → RecursiveCharacterTextSplitter (500 chars / 50 overlap) → all-MiniLM-L6-v2 embeddings (384-dim) → ChromaDB vector store. 163 chunks indexed from BCI domain documents.

**Structured Output** — `with_structured_output(method="function_calling")` with 3-layer fallback JSON parsing (direct parse → markdown extraction → brace matching). Designed to work around DeepSeek's lack of native `response_format` support.

**Memory System** — 4-layer architecture: buffer memory (window-based), summary memory (LLM-triggered compression), entity memory (LLM-based JSON extraction), long-term vector memory (ChromaDB `PersistentClient`).

**LLM Optimization** — ThreadPoolExecutor-based concurrent tool execution, SSE response streaming, two-layer LLM cache (L1 SHA256 exact match, L2 embedding cosine similarity semantic match).

**Observability** — Custom `AgentTracer` (LangChain `BaseCallbackHandler`) capturing span-level traces. Dual-format structured logging (.jsonl). `MetricsCollector` with p50/p95/p99 latency, token usage, per-tool success rates.

**Security** — 3-layer defense: `InputValidator` (length, control chars, zero-width chars), `InjectionDetector` (18 pattern rules + heuristic analysis), `ToolSandbox` (per-tool rate limiting, session call caps, parameter validation, audit logging).

**Async & Batch** — `AsyncTaskQueue` with `asyncio` + `ThreadPoolExecutor` for concurrent Agent execution. `BatchAnalyzer` for bulk company analysis with progress tracking and speedup reporting.

**Evaluation Pipeline** — 11 test cases across 5 categories (factual, analysis, comparison, RAG, edge cases). 7 scoring dimensions: tool selection accuracy, keyword coverage, field completeness, latency, tool efficiency, error handling, LLM-as-judge (relevance × coherence × density). JSON result persistence for historical comparison.

**API Service** — FastAPI with 4 endpoints, rate limiting middleware, lifespan RAII pattern, graceful shutdown.

## Project Structure

```
bci-agent/
├── agents/bci_agent.py           # ReAct Agent definition
├── tools/                         # 5 tool implementations
│   ├── bci_search.py             # Static company data lookup
│   ├── bci_news.py               # News retrieval
│   ├── rag_search.py             # RAG vector search
│   ├── bci_analyzer.py           # Structured company analysis
│   └── bci_comparator.py         # Multi-step comparison
├── rag/                           # RAG pipeline
│   ├── loader.py                 # PDF → chunks
│   └── vectorstore.py            # ChromaDB operations
├── schemas/bci_models.py          # Pydantic models
├── utils/
│   ├── tool_registry.py          # @register decorator + discovery
│   ├── result_store.py           # Tool result caching (anti-paraphrase)
│   ├── memory.py                 # 4-layer memory system
│   ├── llm_parser.py             # Defensive JSON parsing
│   ├── tracer.py                 # AgentTracer (span-level tracing)
│   ├── metrics.py                # MetricsCollector
│   ├── concurrent.py             # Parallel tool executor
│   ├── streaming.py              # SSE streaming handler
│   └── llm_cache.py              # L1/L2 LLM cache
├── api/server.py                  # FastAPI service
├── security/guard.py              # 3-layer security
├── async_tasks/
│   ├── task_queue.py             # AsyncTaskQueue
│   └── batch_analyzer.py         # Batch analysis orchestrator
├── evaluation/
│   ├── test_cases.py             # 11 test case definitions
│   ├── evaluator.py              # Execution engine + scorers
│   └── report.py                 # Console + JSON reporting
├── tests/                         # Test scripts (per-module)
├── data/                          # BCI PDF documents
├── chroma_db/                     # ChromaDB persistence
├── config.py                      # LLM config + .env loading
└── main.py                        # Interactive REPL
```

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph + LangChain |
| LLM | DeepSeek (OpenAI-compatible API) |
| Vector Store | ChromaDB (PersistentClient) |
| Embeddings | all-MiniLM-L6-v2 (384-dim, local) |
| API | FastAPI + Uvicorn |
| Async | asyncio + ThreadPoolExecutor |
| Security | Custom rule-based + heuristic |

## Quick Start

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env: set DEEPSEEK_API_KEY and DEEPSEEK_BASE_URL

# Index documents
python -m rag.loader

# Run interactive REPL
python main.py

# Run API server
python -m api.server

# Run evaluation
python -m tests.test_evaluation quick
```

## Design Decisions

**Why `result_store` for tool output caching?** — In the ReAct loop, LLM paraphrases tool outputs before passing them to downstream tools. When `bci_comparator` needs raw structured data from two `bci_analyzer` calls, paraphrased text corrupts the comparison. `result_store` caches raw structured outputs keyed by tool call ID, bypassing LLM re-narration.

**Why custom `AgentTracer` instead of LangSmith?** — Self-implementing the tracer demonstrates understanding of the span/trace model (each Agent execution = trace, each LLM call or tool call = span with start/end time, I/O, status). Production migration path is OpenTelemetry integration with the same span semantics.

**Why `method="function_calling"` for structured output?** — DeepSeek doesn't support OpenAI's `response_format` parameter. `with_structured_output(method="function_calling")` is the workaround, paired with 3-layer defensive JSON parsing as fallback.

**Why explicit tool module lists over auto-discovery?** — `_discover_tools()` takes an explicit module list rather than scanning the filesystem. Auto-discovery is fragile (picks up test files, __pycache__), harder to debug, and doesn't communicate intent. Explicit lists are controllable and self-documenting.

## C++ Systems Background

This project deliberately applies systems engineering patterns from C++ to LLM application development:

| C++ Concept | Agent Implementation |
|---|---|
| `std::async` / `std::future` | `ThreadPoolExecutor` concurrent tool calls |
| RAII | FastAPI `lifespan` context manager |
| Function pointer dispatcher | `tool_registry.py` `@register` decorator |
| Cache hierarchy (L1/L2/L3) | L1 hash exact match + L2 embedding semantic match |
| `std::future::wait_for(timeout)` | Per-tool timeout with `Future.result(timeout=N)` |
| Producer-consumer queue | `AsyncTaskQueue` with `Semaphore` |
| Profiling / structured logging | `AgentTracer` span model |


### Evaluation
  tool_selection       ██████████████░░░░░░ 0.73
  field_completeness   ██████████████░░░░░░ 0.73
  latency              ███████████████░░░░░ 0.80
  llm_judge            █████████████████░░░ 0.89
  keyword_coverage     ███████████████████░ 0.97
  tool_efficiency      ████████████████████ 1.00
  error_handling       ████████████████████ 1.00
