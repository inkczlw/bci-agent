"""BCI Agent 入口 — 最简版本。"""

from agents.bci_agent import create_bci_agent
from utils.memory import ConversationMemory
from config import get_llm
from langchain_core.messages import HumanMessage, SystemMessage


def _get_role(msg) -> str:
    if isinstance(msg, HumanMessage):
        return "user"
    elif isinstance(msg, SystemMessage):
        return "system"
    return "assistant"


def main():
    agent = create_bci_agent()
    memory = ConversationMemory(max_turns=10)

    print("BCI Agent 已启动，输入 quit 退出。\n")

    while True:
        query = input("用户: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        memory.add_user_message(query)
        result = agent.invoke({
            "messages": [
                {"role": _get_role(m), "content": m.content}
                for m in memory.get_messages()
            ]
        })

        ai_response = result["messages"][-1].content
        memory.add_ai_message(ai_response)
        print(f"\nAgent: {ai_response}\n")


if __name__ == "__main__":
    main()