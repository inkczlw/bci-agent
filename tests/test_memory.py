"""测试记忆系统 — 四种模式独立验证。

验证：
1. Buffer Memory — 窗口截断
2. Summary Memory — LLM 压缩
3. Entity Memory — 实体提取与注入
4. Long-Term Vector Memory — Chroma 持久化检索

用法：python -m tests.test_memory
      python -m tests.test_memory buffer
      python -m tests.test_memory summary
      python -m tests.test_memory entity
      python -m tests.test_memory longterm
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.bci_agent import create_bci_agent
from utils.memory import ConversationMemory, EntityMemory
from utils.long_term_memory import LongTermMemory
from config import get_llm
from langchain_core.messages import HumanMessage, SystemMessage


def _get_role(msg) -> str:
    if isinstance(msg, HumanMessage):
        return "user"
    elif isinstance(msg, SystemMessage):
        return "system"
    return "assistant"


def run_conversation(agent, memory: ConversationMemory, query: str):
    """标准对话，无实体/长期记忆注入。"""
    print(f"\n{'='*60}")
    print(f"用户: {query}")
    print(f"{'='*60}")

    memory.add_user_message(query)
    result = agent.invoke({
        "messages": [
            {"role": _get_role(m), "content": m.content}
            for m in memory.get_messages()
        ]
    })
    ai_response = result["messages"][-1].content
    memory.add_ai_message(ai_response)

    print(f"\nAgent: {ai_response[:500]}")
    print(f"[记忆状态: {memory.turn_count} 轮 | 有摘要: {memory.has_summary}]")


# ── 1. Buffer Memory ────────────────────────────────────────

def test_buffer():
    print("\n" + "="*60)
    print("  Test: Buffer Memory")
    print("="*60)

    agent = create_bci_agent()
    memory = ConversationMemory(max_turns=10)

    for q in ["介绍一下 Neuralink", "它的主要竞争对手是谁？"]:
        run_conversation(agent, memory, q)

    print(f"\n  ✓ Buffer memory 完成，共 {memory.turn_count} 轮")


# ── 2. Summary Memory ───────────────────────────────────────

def test_summary():
    print("\n" + "="*60)
    print("  Test: Summary Memory (阈值=3轮触发压缩)")
    print("="*60)

    agent = create_bci_agent()
    llm = get_llm()
    memory = ConversationMemory(max_turns=10, summary_threshold=3, llm=llm)

    for q in [
        "介绍一下 BrainCo",
        "它的主要产品是什么？",
        "它和 Neuralink 有什么区别？",
        "总结一下你刚才说的内容",
    ]:
        run_conversation(agent, memory, q)

    print(f"\n  ✓ Summary memory 完成，摘要存在: {memory.has_summary}")


# ── 3. Entity Memory ────────────────────────────────────────

def test_entity():
    print("\n" + "="*60)
    print("  Test: Entity Memory")
    print("="*60)

    agent = create_bci_agent()
    llm = get_llm()
    entity_mem = EntityMemory(llm=llm)
    memory = ConversationMemory(max_turns=10)

    for q in [
        "介绍一下 Neuralink",
        "BrainCo 和它比怎么样？",
        "Synchron 呢？",
        "这三家公司里谁最有可能上市？",
    ]:
        print(f"\n{'='*60}")
        print(f"用户: {q}")
        print(f"{'='*60}")

        entity_context = entity_mem.get_relevant_context(q)
        augmented_query = f"{q}\n\n[背景参考]\n{entity_context}" if entity_context else q

        memory.add_user_message(augmented_query)
        result = agent.invoke({
            "messages": [
                {"role": _get_role(m), "content": m.content}
                for m in memory.get_messages()
            ]
        })
        ai_response = result["messages"][-1].content
        memory.add_ai_message(ai_response)
        entity_mem.extract_and_update(q, ai_response)

        print(f"\nAgent: {ai_response[:400]}")
        print(f"[实体记忆: {entity_mem.entity_count} 个实体]")

    print(f"\n  ✓ Entity memory 完成，共 {entity_mem.entity_count} 个实体")


# ── 4. Long-Term Vector Memory ──────────────────────────────

def test_longterm():
    print("\n" + "="*60)
    print("  Test: Long-Term Vector Memory")
    print("="*60)

    agent = create_bci_agent()
    ltm = LongTermMemory(persist_dir="chroma_db")
    memory = ConversationMemory(max_turns=10)

    print(f"[启动时长期记忆中已有 {ltm.total_turns} 轮历史]")

    for q in [
        "介绍一下 Synchron",
        "它获得了哪些 FDA 认证？",
        "Neuralink 和 Synchron 谁的临床进展更快？",
    ]:
        print(f"\n{'='*60}")
        print(f"用户: {q}")
        print(f"{'='*60}")

        history_context = ltm.search(q)
        if history_context:
            augmented_query = f"{q}\n\n[历史参考]\n{history_context}"
            print(f"[检索到相关历史，已注入 context]")
        else:
            augmented_query = q

        memory.add_user_message(augmented_query)
        result = agent.invoke({
            "messages": [
                {"role": _get_role(m), "content": m.content}
                for m in memory.get_messages()
            ]
        })
        ai_response = result["messages"][-1].content
        memory.add_ai_message(ai_response)
        ltm.save_turn(q, ai_response)

        print(f"\nAgent: {ai_response[:400]}")
        print(f"[长期记忆总轮数: {ltm.total_turns}]")

    print(f"\n  ✓ Long-term memory 完成，共 {ltm.total_turns} 轮")


# ── 入口 ────────────────────────────────────────────────────

TEST_MAP = {
    "buffer": test_buffer,
    "summary": test_summary,
    "entity": test_entity,
    "longterm": test_longterm,
}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        name = sys.argv[1]
        if name in TEST_MAP:
            TEST_MAP[name]()
        else:
            print(f"未知测试: {name}，可选: {', '.join(TEST_MAP.keys())}")
    else:
        # 无参数时全跑
        for name, fn in TEST_MAP.items():
            fn()