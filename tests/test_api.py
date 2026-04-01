"""
先启动服务：
    python -m api.server

然后另开一个终端运行测试：
    python -m tests.test_api health     # 健康检查
    python -m tests.test_api chat       # 同步聊天
    python -m tests.test_api stream     # SSE 流式聊天
    python -m tests.test_api metrics    # 查看指标
    python -m tests.test_api ratelimit  # 测试限流
    python -m tests.test_api all        # 全部运行
"""

import sys
import json
import time
import requests


BASE_URL = "http://localhost:8000"


def test_health():
    """测试健康检查端点。"""
    print("\n" + "=" * 60)
    print("  测试: GET /health")
    print("=" * 60)

    resp = requests.get(f"{BASE_URL}/health")
    print(f"  Status: {resp.status_code}")
    print(f"  Response: {json.dumps(resp.json(), indent=2, ensure_ascii=False)}")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("healthy", "degraded", "unhealthy")
    print("  ✓ 健康检查通过")


def test_chat():
    """测试同步聊天端点。"""
    print("\n" + "=" * 60)
    print("  测试: POST /chat")
    print("=" * 60)

    payload = {
        "query": "介绍一下 Neuralink 的核心技术",
        "session_id": "test-001",
    }

    t0 = time.time()
    resp = requests.post(f"{BASE_URL}/chat", json=payload)
    elapsed = round((time.time() - t0) * 1000, 2)

    print(f"  Status: {resp.status_code}")
    print(f"  耗时: {elapsed}ms")

    if resp.status_code == 200:
        data = resp.json()
        print(f"  Answer: {data['answer'][:200]}...")
        print(f"  Trace ID: {data.get('trace_id', 'N/A')}")
        print(f"  Tool calls: {data.get('tool_calls', [])}")
        print(f"  Token usage: {data.get('token_usage', {})}")
        print(f"  Server duration: {data.get('duration_ms', 'N/A')}ms")
        print("  ✓ 同步聊天通过")
    else:
        print(f"  ✗ 失败: {resp.text}")


def test_stream():
    """测试 SSE 流式聊天端点。"""
    print("\n" + "=" * 60)
    print("  测试: POST /chat/stream (SSE)")
    print("=" * 60)

    payload = {
        "query": "BrainCo 做什么产品？",
        "session_id": "test-002",
        "stream": True,
    }

    t0 = time.time()
    first_event_time = None
    event_count = 0

    # SSE 需要 stream=True 的 requests
    with requests.post(f"{BASE_URL}/chat/stream", json=payload, stream=True) as resp:
        print(f"  Status: {resp.status_code}")
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                if first_event_time is None:
                    first_event_time = time.time()

                event_data = json.loads(line[6:])  # 去掉 "data: " 前缀
                event_count += 1

                if event_data.get("type") == "done":
                    print(f"\n  [stream 结束]")
                    break

                node = event_data.get("node", "")
                content = event_data.get("content", "")
                if content:
                    preview = content[:80].replace("\n", " ")
                    print(f"  [{event_count}] {node}: {preview}...")

    total = round((time.time() - t0) * 1000, 2)
    ttfe = round((first_event_time - t0) * 1000, 2) if first_event_time else None

    print(f"\n  总耗时: {total}ms")
    print(f"  TTFE (Time to First Event): {ttfe}ms")
    print(f"  事件数: {event_count}")
    print("  ✓ SSE 流式聊天通过")


def test_metrics():
    """测试指标端点。"""
    print("\n" + "=" * 60)
    print("  测试: GET /metrics")
    print("=" * 60)

    resp = requests.get(f"{BASE_URL}/metrics")
    print(f"  Status: {resp.status_code}")
    print(f"  Response: {json.dumps(resp.json(), indent=2, ensure_ascii=False)}")
    print("  ✓ 指标查询通过")


def test_ratelimit():
    """测试限流：快速发大量请求，触发 429。"""
    print("\n" + "=" * 60)
    print("  测试: Rate Limiting")
    print("=" * 60)

    payload = {"query": "test", "session_id": "rate-test"}
    success_count = 0
    limited_count = 0

    # 快速发 25 个请求（限制是 20/60s）
    for i in range(25):
        resp = requests.post(f"{BASE_URL}/chat", json=payload)
        if resp.status_code == 200:
            success_count += 1
        elif resp.status_code == 429:
            limited_count += 1
            if limited_count == 1:
                print(f"  第 {i+1} 个请求被限流: {resp.json()['detail']}")
        else:
            print(f"  第 {i+1} 个请求异常: {resp.status_code}")

    print(f"  成功: {success_count}, 被限流: {limited_count}")
    print(f"  {'✓' if limited_count > 0 else '✗'} 限流{'生效' if limited_count > 0 else '未生效'}")


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "all"

    tests = {
        "health": test_health,
        "chat": test_chat,
        "stream": test_stream,
        "metrics": test_metrics,
        "ratelimit": test_ratelimit,
    }

    if sub == "all":
        for name, fn in tests.items():
            try:
                fn()
            except requests.ConnectionError:
                print(f"\n  ✗ 连接失败 — 确保服务已启动: python -m api.server")
                sys.exit(1)
            except Exception as e:
                print(f"\n  ✗ {name} 测试失败: {e}")
    elif sub in tests:
        try:
            tests[sub]()
        except requests.ConnectionError:
            print(f"\n  ✗ 连接失败 — 确保服务已启动: python -m api.server")
            sys.exit(1)
    else:
        print(f"未知测试: {sub}")
        print(f"可用: {', '.join(tests.keys())}, all")
        sys.exit(1)

    print("\n✅ 测试完成")


if __name__ == "__main__":
    main()