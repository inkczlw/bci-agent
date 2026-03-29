"""长期向量记忆。
把对话历史持久化到 Chroma，支持跨对话的语义检索。
"""

import uuid
from datetime import datetime
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

COLLECTION_NAME = "conversation_history"


class LongTermMemory:
    """基于向量存储的长期记忆。

    存储单位：一轮对话（user + ai 各一条）
    检索方式：用当前 query 语义检索相关历史轮次
    持久化：写入 chroma_db，进程重启后依然存在
    """

    def __init__(self, persist_dir: str = "chroma_db", n_results: int = 3):
        self.client = PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=DefaultEmbeddingFunction(),
        )
        self.n_results = n_results

    def save_turn(self, user_msg: str, ai_msg: str, metadata: dict = None):
        """保存一轮对话到向量库。"""
        turn_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # 把 user + ai 拼在一起作为检索文本
        document = f"用户: {user_msg}\nAI: {ai_msg[:600]}"

        self.collection.add(
            ids=[turn_id],
            documents=[document],
            metadatas=[{
                "timestamp": timestamp,
                "user_msg": user_msg[:200],
                **(metadata or {}),
            }],
        )
        print(f"[长期记忆已保存] turn_id={turn_id[:8]}...")

    def search(self, query: str) -> str:
        """检索与当前 query 语义相关的历史对话。"""
        count = self.collection.count()
        if count == 0:
            return ""

        n = min(self.n_results, count)
        results = self.collection.query(
            query_texts=[query],
            n_results=n,
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        if not docs:
            return ""

        lines = []
        for doc, meta in zip(docs, metas):
            ts = meta.get("timestamp", "")[:16]  # 只取到分钟
            lines.append(f"[{ts}]\n{doc}")

        return "相关历史对话：\n" + "\n---\n".join(lines)

    @property
    def total_turns(self) -> int:
        return self.collection.count()