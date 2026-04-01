"""

用法：
    python -m tests.test_optimization concurrent   # 测试并发 tool call
    python -m tests.test_optimization streaming     # 测试流式输出
    python -m tests.test_optimization cache         # 测试 LLM 缓存
    python -m tests.test_optimization all           # 全部运行
"""

import sys
import time
import json
from pathlib import Path

# .env 加载
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")


def test_concurrent():
    """测试并发 tool call：对比串行 vs 并发执行的耗时差异。"""
    print("\n" + "=" * 60)
    print("  测试 1：并发 Tool Call")
    print("=" * 60)

    from utils.concurrent import execute_tools_concurrent, analyze_parallelism, ToolResult
    from tools.bci_search import search_bci_company
    from tools.bci_news import get_bci_news

    # 构造一个"对比两家公司"的 tool call 场景
    # 这两次搜索没有依赖关系，可以并行
    tool_calls = [
        {"name": "search_bci_company", "args": {"company_name": "Neuralink"}},
        {"name": "search_bci_company", "args": {"company_name": "BrainCo"}},
        {"name": "get_bci_news", "args": {"topic": "Neuralink"}},
    ]

    tool_map = {
        "search_bci_company": search_bci_company,
        "get_bci_news": get_bci_news,
    }

    # 1. 先分析并行度
    analysis = analyze_parallelism(tool_calls)
    print(f"\n并行度分析:")
    for k, v in analysis.items():
        print(f"  {k}: {v}")

    # 2. 串行执行（baseline）
    print(f"\n--- 串行执行 ---")
    serial_start = time.time()
    serial_results = []
    for call in tool_calls:
        name = call["name"]
        args = call["args"]
        t0 = time.time()
        result = tool_map[name].invoke(args)
        elapsed = round((time.time() - t0) * 1000, 2)
        serial_results.append(ToolResult(
            tool_name=name, args=args, result=result, duration_ms=elapsed
        ))
        print(f"  {name}({args}) → {elapsed}ms")
    serial_total = round((time.time() - serial_start) * 1000, 2)
    print(f"  串行总耗时: {serial_total}ms")

    # 3. 并发执行
    print(f"\n--- 并发执行 ---")
    concurrent_start = time.time()
    concurrent_results = execute_tools_concurrent(tool_calls, tool_map)
    concurrent_total = round((time.time() - concurrent_start) * 1000, 2)
    for r in concurrent_results:
        status_icon = "✓" if r.status == "ok" else "✗"
        print(f"  {status_icon} {r.tool_name}({r.args}) → {r.duration_ms}ms [{r.status}]")
    print(f"  并发总耗时: {concurrent_total}ms")

    # 4. 对比
    speedup = round(serial_total / concurrent_total, 2) if concurrent_total > 0 else 0
    print(f"\n加速比: {speedup}x ({serial_total}ms → {concurrent_total}ms)")
    print("  注意: 本地 dict lookup 的 tool 差异不大，")
    print("  真正的加速在 API 调用型 tool（rag_search, analyzer）上体现。")


def test_streaming():
    """测试流式输出：对比 invoke（等完再输出）vs stream（逐步输出）。"""
    print("\n" + "=" * 60)
    print("  测试 2：Response Streaming")
    print("=" * 60)

    from agents.bci_agent import create_bci_agent
    from utils.streaming import StreamingHandler, stream_agent_response

    agent = create_bci_agent()
    query = "简单介绍一下 Neuralink"

    # 1. 普通 invoke（baseline）
    print(f"\n--- invoke 模式（等全部完成再输出）---")
    t0 = time.time()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    invoke_time = round((time.time() - t0) * 1000, 2)
    response = result["messages"][-1].content
    print(f"  响应: {response[:100]}...")
    print(f"  总耗时: {invoke_time}ms")

    # 2. stream 模式（逐步输出）
    print(f"\n--- stream 模式（逐步输出）---")
    streaming_handler = StreamingHandler()
    t0 = time.time()
    first_output_time = None

    print("  响应: ", end="")
    for event in stream_agent_response(
        agent,
        [{"role": "user", "content": query}],
        callbacks=[streaming_handler],
    ):
        if event["node"] == "agent" and event["content"]:
            if first_output_time is None:
                first_output_time = time.time()
            # stream_agent_response 返回的是完整 message，不是逐 token
            # 逐 token 输出通过 StreamingHandler 的 on_llm_new_token 实现
            pass

    stream_time = round((time.time() - t0) * 1000, 2)
    ttft = round((first_output_time - t0) * 1000, 2) if first_output_time else None

    print(f"\n  总耗时: {stream_time}ms")
    if ttft:
        print(f"  TTFT (Time to First Token): {ttft}ms")
    print(f"\n  streaming 统计: {streaming_handler.get_streaming_stats()}")

    print(f"\n对比:")
    print(f"  invoke 模式: 用户等 {invoke_time}ms 后一次性看到全部内容")
    print(f"  stream 模式: 用户等 {ttft}ms 后开始看到内容流入")


def test_cache():
    """测试 LLM 缓存：L1 精确匹配 + L2 语义匹配。"""
    print("\n" + "=" * 60)
    print("  测试 3：LLM 调用缓存")
    print("=" * 60)

    from utils.llm_cache import LLMCache

    # 不启用语义缓存（先测 L1）
    cache = LLMCache(cache_dir="cache/test", default_ttl=60)

    # 模拟写入缓存
    print(f"\n--- L1: 精确匹配 ---")
    cache.put("Neuralink 的核心技术", "Neuralink 的核心技术是柔性电极...")
    cache.put("BrainCo 的产品", "BrainCo 主打 Focus 脑电头环...")

    # 精确命中
    result = cache.get("Neuralink 的核心技术")
    print(f"  query='Neuralink 的核心技术' → {'HIT' if result else 'MISS'} (level={result['cache_level'] if result else 'N/A'})")

    # 字面不同，miss
    result = cache.get("告诉我 Neuralink 的技术")
    print(f"  query='告诉我 Neuralink 的技术' → {'HIT' if result else 'MISS'}")

    # 完全无关
    result = cache.get("今天天气怎么样")
    print(f"  query='今天天气怎么样' → {'HIT' if result else 'MISS'}")

    print(f"\n  缓存统计: {cache.get_stats()}")

    # L2: 语义匹配（需要 embedding 模型）
    print(f"\n--- L2: 语义匹配 ---")
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        cache_l2 = LLMCache(
            cache_dir="cache/test_l2",
            default_ttl=60,
            similarity_threshold=0.80,
            embedding_fn=lambda text: embeddings.embed_query(text),
        )

        # 写入
        cache_l2.put("Neuralink 的核心技术是什么", "柔性电极 + 手术机器人...")

        # 语义相似的 query
        result = cache_l2.get("介绍一下 Neuralink 的技术")
        print(f"  query='介绍一下 Neuralink 的技术' → {'HIT' if result else 'MISS'}", end="")
        if result:
            print(f" (level={result['cache_level']}, similarity={result.get('similarity', 'N/A')})")
        else:
            print()

        # 不太相似的
        result = cache_l2.get("BrainCo 和 Neuralink 的区别")
        print(f"  query='BrainCo 和 Neuralink 的区别' → {'HIT' if result else 'MISS'}")

        print(f"\n  缓存统计: {cache_l2.get_stats()}")

    except ImportError:
        print("  ⚠ 未安装 sentence-transformers，跳过 L2 测试")
        print("  安装: pip install sentence-transformers")


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "all"

    tests = {
        "concurrent": test_concurrent,
        "streaming": test_streaming,
        "cache": test_cache,
    }

    if sub == "all":
        for name, fn in tests.items():
            fn()
    elif sub in tests:
        tests[sub]()
    else:
        print(f"未知测试: {sub}")
        print(f"可用: {', '.join(tests.keys())}, all")
        sys.exit(1)

    print("\n✅ 测试完成")


if __name__ == "__main__":
    main()