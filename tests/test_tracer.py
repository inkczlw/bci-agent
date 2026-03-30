"""BCI Agent 入口 — 带 tracing 的版本。"""

from agents.bci_agent import create_bci_agent
from utils.memory import ConversationMemory
from utils.tracer import AgentTracer
from config import get_llm


def run_conversation(agent, memory: ConversationMemory, query: str, tracer: AgentTracer):
    """带记忆 + tracing 的对话。"""
    print(f"\n{'='*60}")
    print(f"用户: {query}")
    print(f"{'='*60}")

    memory.add_user_message(query)

    # 开始 trace
    tracer.start_trace(query)

    messages = memory.get_messages()
    result = agent.invoke(
        {"messages": [
            {"role": _get_role(m), "content": m.content}
            for m in messages
        ]},
        config={"callbacks": [tracer]},  # <-- 关键：tracer 以 callback 注入
    )

    # 结束 trace
    trace = tracer.end_trace()

    ai_response = result["messages"][-1].content
    memory.add_ai_message(ai_response)

    # 输出 Agent 回答
    print(f"\nAgent: {ai_response[:500]}")
    print(f"[记忆状态: {memory.turn_count} 轮 | 有摘要: {memory.has_summary}]")

    # 输出 trace 报告
    if trace:
        print(trace.detail_report())


def _get_role(msg) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage
    if isinstance(msg, HumanMessage):
        return "user"
    elif isinstance(msg, SystemMessage):
        return "system"
    return "assistant"


def main():
    agent = create_bci_agent()
    memory = ConversationMemory(max_turns=10, summary_threshold=5, llm=get_llm())
    tracer = AgentTracer(log_dir="traces")  # trace 数据保存到 traces/ 目录

    queries = [
        "介绍一下 Neuralink 的核心技术",
        "它的主要竞争对手有哪些？",
        "对比一下 Neuralink 和 BrainCo 的技术路线",
    ]

    for q in queries:
        run_conversation(agent, memory, q, tracer)

    # 全部跑完后输出聚合统计
    print("\n" + "="*70)
    print("  聚合统计")
    print("="*70)
    stats = tracer.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()