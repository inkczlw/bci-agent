"""对话记忆管理。
提供对话历史的存储和检索，支持窗口限制避免 token 爆炸。
支持可选的摘要压缩模式（summary memory）。
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class ConversationMemory:
    """基于窗口的对话记忆。保留最近 max_turns 轮对话。

    可选开启 summary 模式：当对话超过 summary_threshold 轮时，
    把早期对话压缩为摘要，节省 token。
    """

    def __init__(self, max_turns: int = 10, summary_threshold: int = 6, llm=None):
        self.max_turns = max_turns
        self.summary_threshold = summary_threshold
        self.llm = llm  # 传入则启用 summary 模式
        self.messages: list = []
        self.summary: str = ""  # 压缩后的历史摘要

    def add_user_message(self, content: str):
        self.messages.append(HumanMessage(content=content))
        self._maybe_summarize()

    def add_ai_message(self, content: str):
        self.messages.append(AIMessage(content=content))

    def get_messages(self) -> list:
        """返回给 Agent 的完整消息列表。
        如果有摘要，把摘要作为第一条 system message 注入。
        """
        if self.summary:
            summary_msg = SystemMessage(
                content=f"以下是早期对话的摘要，供你参考：\n{self.summary}"
            )
            return [summary_msg] + list(self.messages)
        return list(self.messages)

    def _maybe_summarize(self):
        """当消息数超过阈值时，压缩最早的一半对话。"""
        if self.llm is None:
            # 没有 LLM，降级为普通 buffer trim
            self._trim()
            return

        total_turns = len(self.messages) // 2
        if total_turns < self.summary_threshold:
            return

        # 取最早的 (threshold // 2) 轮压缩，保留最近的留在窗口
        compress_count = (self.summary_threshold // 2) * 2  # 保持 user/ai 配对
        to_compress = self.messages[:compress_count]
        self.messages = self.messages[compress_count:]

        # 构造压缩 prompt
        history_text = "\n".join(
            f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content[:300]}"
            for m in to_compress
        )
        prompt = f"""请将以下对话历史压缩为一段简洁的摘要（100字以内），保留关键信息点：{history_text} 摘要："""

        try:
            response = self.llm.invoke(prompt)
            new_summary = response.content.strip()
            # 如果已有摘要，合并
            if self.summary:
                self.summary = f"{self.summary}\n{new_summary}"
            else:
                self.summary = new_summary
            print(f"\n[记忆压缩触发] 摘要: {self.summary[:100]}...")
        except Exception as e:
            print(f"\n[记忆压缩失败，降级为 buffer] {e}")
            self._trim()

    def _trim(self):
        """普通 buffer 截断，无 LLM 时使用。"""
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def clear(self):
        self.messages = []
        self.summary = ""

    @property
    def turn_count(self) -> int:
        return len(self.messages) // 2

    @property
    def has_summary(self) -> bool:
        return bool(self.summary)


class EntityMemory:
    """实体记忆。从对话中提取并跟踪关键实体的信息摘要。"""

    def __init__(self, llm=None):
        self.llm = llm
        self.entities: dict[str, str] = {}  # entity_name -> summary

    def extract_and_update(self, user_msg: str, ai_msg: str):
        """从一轮对话中提取实体并更新记忆。需要 LLM。"""
        if self.llm is None:
            return

        prompt = f"""从以下对话中提取重要实体（公司名、技术名称、人名等），
并为每个实体生成一句话摘要。

用户: {user_msg}
AI: {ai_msg[:500]}

已知实体信息:
{self._format_entities()}

请以 JSON 格式返回需要新增或更新的实体，格式：
{{"实体名": "一句话描述", ...}}
只返回 JSON，不要其他内容。"""

        try:
            response = self.llm.invoke(prompt)
            import json, re
            text = response.content.strip()
            # 提取 JSON 部分
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                updates = json.loads(match.group())
                self.entities.update(updates)
                print(f"[实体记忆更新] {list(updates.keys())}")
        except Exception as e:
            print(f"[实体提取失败] {e}")

    def get_relevant_context(self, query: str) -> str:
        """返回与当前 query 相关的实体信息，注入到 context。"""
        if not self.entities:
            return ""

        # 简单关键词匹配，找相关实体
        relevant = {
            name: summary
            for name, summary in self.entities.items()
            if name.lower() in query.lower()
        }

        # 没有精确匹配就返回所有实体（对话短时实体少，不会太长）
        target = relevant if relevant else self.entities

        lines = [f"- {name}: {summary}" for name, summary in target.items()]
        return "已知实体信息：\n" + "\n".join(lines)

    def _format_entities(self) -> str:
        if not self.entities:
            return "（暂无）"
        return "\n".join(f"- {k}: {v}" for k, v in self.entities.items())

    @property
    def entity_count(self) -> int:
        return len(self.entities)