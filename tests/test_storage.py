"""存储层集成测试。

验证：
1. 数据库初始化（建表）
2. 分析结果写入和查询（db_writer tool）
3. 交互日志采集和分析（interaction_logger）
4. 缓存过期判断

用法：
    python -m tests.test_storage           # 全部测试
    python -m tests.test_storage db        # 只测数据库
    python -m tests.test_storage logger    # 只测日志采集
    python -m tests.test_storage analyzer  # 只测日志分析
"""

import json
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_db_init():
    """测试数据库初始化。"""
    print("\n=== 测试 1: 数据库初始化 ===")
    from storage.database import DB_URL, init_db

    init_db()
    print(f"  ✅ 数据库创建成功: {DB_URL}")

    # 验证表存在
    from sqlalchemy import inspect

    from storage.database import get_engine

    inspector = inspect(get_engine())
    tables = inspector.get_table_names()
    print(f"  ✅ 表: {tables}")
    assert "analysis_results" in tables, "analysis_results 表不存在"
    assert "interaction_logs" in tables, "interaction_logs 表不存在"
    print("  ✅ 所有表已创建")


def test_db_writer():
    """测试分析结果写入和查询。"""
    print("\n=== 测试 2: 分析结果 CRUD ===")
    from tools.db_writer import (
        list_analyzed_companies,
        query_analysis_result,
        save_analysis_result,
    )

    # 写入
    result = save_analysis_result.invoke(
        {
            "company_name": "Neuralink",
            "technology_route": "侵入式脑机接口",
            "funding_stage": "D轮",
            "core_technology": "高密度柔性电极阵列 + 手术机器人",
            "competitive_advantage": "Elon Musk 品牌效应 + 先发优势",
            "application_areas": "医疗康复、神经疾病治疗",
            "source_query": "分析一下 Neuralink",
        }
    )
    parsed = json.loads(result)
    print(f"  ✅ 写入: {parsed['message']}")
    assert parsed["status"] == "success"

    # 查询 —— 应该命中
    result = query_analysis_result.invoke(
        {
            "company_name": "Neuralink",
            "max_age_hours": 24,
        }
    )
    parsed = json.loads(result)
    print(f"  ✅ 查询: {parsed['message']}")
    assert parsed["status"] == "found"
    assert parsed["data"]["company_name"] == "Neuralink"

    # 查询不存在的公司
    result = query_analysis_result.invoke(
        {
            "company_name": "FakeCompany_XYZ",
        }
    )
    parsed = json.loads(result)
    print(f"  ✅ 未命中: {parsed['message']}")
    assert parsed["status"] == "not_found"

    # 列出所有公司
    result = list_analyzed_companies.invoke({})
    parsed = json.loads(result)
    print(f"  ✅ 已分析公司: {parsed['total']} 家")
    assert parsed["total"] >= 1

    # 写入第二家公司
    save_analysis_result.invoke(
        {
            "company_name": "BrainCo",
            "technology_route": "非侵入式脑机接口",
            "funding_stage": "B轮",
            "core_technology": "EEG 信号处理 + AI 算法",
            "competitive_advantage": "消费级产品化能力",
            "application_areas": "教育、康复、消费电子",
        }
    )
    result = list_analyzed_companies.invoke({})
    parsed = json.loads(result)
    print(f"  ✅ 更新后: {parsed['total']} 家公司")


def test_interaction_logger():
    """测试交互日志采集。"""
    print("\n=== 测试 3: 交互日志采集 ===")
    from storage.interaction_logger import InteractionLogger

    il = InteractionLogger(session_id="test_session_001", agent_version="1.0")

    # 模拟一次正常交互的 trace 数据
    mock_trace = {
        "trace_id": "test-trace-001",
        "total_duration_ms": 3500.0,
        "spans": [
            {
                "span_type": "llm",
                "name": "llm_call",
                "duration_ms": 2000.0,
                "status": "ok",
                "token_usage": {
                    "prompt_tokens": 500,
                    "completion_tokens": 200,
                    "total_tokens": 700,
                },
                "metadata": {"model": "deepseek-chat"},
            },
            {
                "span_type": "tool",
                "name": "search_bci_company",
                "duration_ms": 800.0,
                "status": "ok",
            },
            {
                "span_type": "tool",
                "name": "analyze_bci_company",
                "duration_ms": 700.0,
                "status": "ok",
            },
        ],
    }

    il.log_interaction(
        user_query="分析一下 Neuralink 这家公司",
        response="Neuralink 是一家由 Elon Musk 创立的脑机接口公司...",
        trace=mock_trace,
        quality_score=0.85,
        query_category="analysis",
    )
    print("  ✅ 交互日志已写入")

    # 模拟一次有错误的交互
    mock_trace_error = {
        "trace_id": "test-trace-002",
        "total_duration_ms": 15500.0,
        "spans": [
            {
                "span_type": "llm",
                "name": "llm_call",
                "duration_ms": 1500.0,
                "status": "ok",
                "token_usage": {
                    "prompt_tokens": 400,
                    "completion_tokens": 100,
                    "total_tokens": 500,
                },
                "metadata": {"model": "deepseek-chat"},
            },
            {
                "span_type": "tool",
                "name": "search_bci_docs",
                "duration_ms": 14000.0,
                "status": "timeout",
            },
        ],
    }

    il.log_interaction(
        user_query="BCI 领域最新的论文有哪些",
        response="抱歉，检索超时，无法获取最新论文信息...",
        trace=mock_trace_error,
        quality_score=0.3,
        query_category="rag",
    )
    print("  ✅ 含错误的交互日志已写入")

    # 查询 session 历史
    history = il.get_session_history()
    print(f"  ✅ Session 历史: {len(history)} 条记录")
    for h in history:
        print(f"     {h['query'][:40]} | tools={h['tools']} | {h['latency_ms']}ms")


def test_interaction_analyzer():
    """测试日志分析功能。"""
    print("\n=== 测试 4: 日志分析 ===")
    from storage.interaction_logger import InteractionAnalyzer

    # Tool 错误统计
    tool_stats = InteractionAnalyzer.get_tool_error_summary(hours=24)
    print(f"  Tool 错误统计:")
    for name, stats in tool_stats.items():
        print(f"    {name}: {stats['total']} 次调用, 错误率 {stats['error_rate']}")

    # 按类别延迟统计
    category_stats = InteractionAnalyzer.get_latency_by_category()
    print(f"  按类别延迟:")
    for cat, stats in category_stats.items():
        print(
            f"    {cat}: 平均 {stats['avg_latency_ms']}ms, 平均 {stats['avg_tokens']} tokens"
        )

    # 版本对比
    version_stats = InteractionAnalyzer.compare_agent_versions()
    print(f"  版本对比:")
    for ver, stats in version_stats.items():
        print(
            f"    v{ver}: {stats['count']} 次交互, 平均 {stats['avg_latency_ms']}ms, "
            f"错误率 {stats['error_rate']}, 质量 {stats['avg_quality']}"
        )

    print("  ✅ 日志分析完成")


def test_cache_staleness():
    """测试缓存过期判断。"""
    print("\n=== 测试 5: 缓存过期 ===")
    from datetime import datetime, timedelta, timezone

    from storage.models import AnalysisResult

    # 新建一个刚创建的记录
    fresh = AnalysisResult(company_name="TestCo")
    fresh.created_at = datetime.now(timezone.utc)
    assert not fresh.is_stale(max_age_hours=24), "新记录不应该过期"
    print("  ✅ 新记录未过期")

    # 模拟一个 25 小时前的记录
    old = AnalysisResult(company_name="OldCo")
    old.created_at = datetime.now(timezone.utc) - timedelta(hours=25)
    assert old.is_stale(max_age_hours=24), "25h 前的记录应该过期"
    print("  ✅ 旧记录已过期")

    # 自定义过期时间
    recent = AnalysisResult(company_name="RecentCo")
    recent.created_at = datetime.now(timezone.utc) - timedelta(hours=2)
    assert not recent.is_stale(max_age_hours=4), "2h 前的记录在 4h 窗口内不应过期"
    assert recent.is_stale(max_age_hours=1), "2h 前的记录在 1h 窗口内应过期"
    print("  ✅ 自定义过期窗口正常")


def main():
    test_name = sys.argv[1] if len(sys.argv) > 1 else "all"

    tests = {
        "db": [test_db_init, test_db_writer],
        "logger": [test_db_init, test_interaction_logger],
        "analyzer": [test_db_init, test_interaction_logger, test_interaction_analyzer],
        "all": [
            test_db_init,
            test_db_writer,
            test_interaction_logger,
            test_interaction_analyzer,
            test_cache_staleness,
        ],
    }

    if test_name not in tests:
        print(f"未知测试: {test_name}")
        print(f"可用: {', '.join(tests.keys())}")
        sys.exit(1)

    print(f"🚀 运行存储层测试: {test_name}")
    start = time.time()

    for test_fn in tests[test_name]:
        test_fn()

    elapsed = round(time.time() - start, 1)
    print(f"\n✅ 全部通过 ({elapsed}s)")


if __name__ == "__main__":
    main()
