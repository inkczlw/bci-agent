from agents.bci_agent import create_bci_agent


def main():
    agent = create_bci_agent()

    query = "Neuralink 目前的临床进展如何？"
    print(f"问题: {query}\n")

    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="updates",
    ):
        for node_name, node_output in event.items():
            print(f"\n{'='*60}")
            print(f"节点: {node_name}")
            print(f"{'='*60}")

            for msg in node_output.get("messages", []):

                # 1. Agent 决策：完整的 LLM 输出
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    # LLM 的思考文本（如果有）
                    if msg.content:
                        print(f"\n  [LLM 思考]")
                        print(f"  {msg.content[:500]}")

                    # tool 调用决策
                    for tc in msg.tool_calls:
                        print(f"\n  [Tool 调用]")
                        print(f"  函数: {tc['name']}")
                        print(f"  参数: {tc['args']}")
                        print(f"  调用ID: {tc['id']}")

                    # token 用量
                    if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                        u = msg.usage_metadata
                        print(f"\n  [Token 消耗]")
                        print(f"  输入: {u.get('input_tokens', '?')}")
                        print(f"  输出: {u.get('output_tokens', '?')}")
                        print(f"  总计: {u.get('total_tokens', '?')}")

                    # 模型信息
                    if hasattr(msg, "response_metadata") and msg.response_metadata:
                        rm = msg.response_metadata
                        model = rm.get("model_name") or rm.get("model", "?")
                        finish = rm.get("finish_reason", "?")
                        print(f"\n  [模型信息]")
                        print(f"  模型: {model}")
                        print(f"  结束原因: {finish}")

                # 2. Tool 执行结果
                elif hasattr(msg, "name"):
                    content = str(msg.content)
                    print(f"\n  [Tool 结果] {msg.name}")
                    print(f"  状态: {'成功' if msg.status == 'success' else msg.status if hasattr(msg, 'status') else '未知'}")
                    print(f"  内容: {content[:300]}{'...' if len(content) > 300 else ''}")

                # 3. 最终回答
                else:
                    content = str(msg.content)
                    print(f"\n  [最终回答]")
                    print(f"  {content[:500]}{'...' if len(content) > 500 else ''}")

                    if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                        u = msg.usage_metadata
                        print(f"\n  [Token 消耗]")
                        print(f"  输入: {u.get('input_tokens', '?')}")
                        print(f"  输出: {u.get('output_tokens', '?')}")
                        print(f"  总计: {u.get('total_tokens', '?')}")


if __name__ == "__main__":
    main()