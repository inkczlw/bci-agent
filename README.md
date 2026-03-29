# BCI Agent

A production-oriented LLM Agent system for BCI (Brain-Computer Interface) industry analysis, built with LangGraph and LangChain.

This project is a hands-on engineering exercise targeting **AI Agent Engineer** competencies: Agent orchestration, RAG pipeline, tool registry design, structured output, and reliability patterns.

---

## Architecture

```
main.py
└── agents/bci_agent.py          # LangGraph ReAct loop, tool registration
    ├── tools/bci_search.py      # Company profile lookup
    ├── tools/bci_news.py        # News retrieval
    ├── tools/rag_search.py      # Vector retrieval (Chroma)
    ├── tools/bci_analyzer.py    # Structured analysis output
    └── tools/bci_comparator.py  # Multi-company comparison

rag/
├── loader.py                    # PDF ingestion → RecursiveCharacterTextSplitter
└── vectorstore.py               # Chroma + all-MiniLM-L6-v2 embeddings (~163 chunks)

schemas/bci_models.py            # Pydantic models: BCICompanyAnalysis, BCICompanyComparison

utils/
├── tool_registry.py             # Auto-discovery via pkgutil (open/closed principle)
├── tool_wrapper.py              # ThreadPoolExecutor-based timeout + fallback decorator
├── llm_parser.py                # Defensive JSON parsing for structured output
├── result_store.py              # Tool result caching
└── memory.py                    # Conversation memory management
```

---

## Key Engineering Decisions

**Tool auto-discovery** — `utils/tool_registry.py` scans the `tools/` package using `pkgutil.iter_modules` and collects every module that exposes a `TOOLS` list. Adding a new tool requires no changes to the Agent core, only declaring `TOOLS = [my_tool]` in the new module.

**Timeout + fallback decorator** — `utils/tool_wrapper.py` wraps tool functions with `ThreadPoolExecutor.submit(...).result(timeout=N)`. On timeout or exception, it returns a structured error string (e.g. `[TIMEOUT] rag_search exceeded 10s`) that the LLM can reason about and retry or reroute. Decorator order is `@tool` outer, `@with_timeout` inner to preserve docstring for schema generation.

**Structured output without `response_format`** — DeepSeek does not support OpenAI's `response_format`. Structured output is handled via `with_structured_output(method="function_calling")` + Pydantic models, with a defensive JSON parser in `utils/llm_parser.py` as fallback.

**RAG pipeline** — PDFs chunked with `RecursiveCharacterTextSplitter` (500 chars, 50 overlap), embedded with `all-MiniLM-L6-v2`, stored in Chroma with persistence. Retrieval is exposed as a standard LangChain tool so the Agent treats it identically to any other tool call.

---

## Stack

| Layer | Choice |
|---|---|
| Agent framework | LangGraph + LangChain |
| LLM | DeepSeek (OpenAI-compatible API) |
| Vector DB | Chroma |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Document parsing | PyPDF + RecursiveCharacterTextSplitter |
| Schema validation | Pydantic v2 |

---

## Setup

```bash
git clone https://github.com/inkczlw/bci-agent.git
cd bci-agent
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

Index the BCI documents (place PDFs in `data/`):

```bash
python -c "from rag.vectorstore import build_vectorstore; build_vectorstore()"
```

Run the agent:

```bash
python main.py
```
