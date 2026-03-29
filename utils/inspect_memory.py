"""查看长期记忆的存储内容。"""
from utils.long_term_memory import LongTermMemory


def main():
    ltm = LongTermMemory(persist_dir="../chroma_db")
    print(f"总轮数: {ltm.total_turns}\n")

    if ltm.total_turns == 0:
        print("暂无记忆")
        return

    # 拉取所有记录
    results = ltm.collection.get(
        include=["documents", "metadatas"]
    )

    docs = results["documents"]
    metas = results["metadatas"]

    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        print(f"── 第 {i} 轮 ──────────────────────")
        print(f"时间: {meta.get('timestamp', '')[:16]}")
        print(f"用户: {meta.get('user_msg', '')}")
        print(f"内容预览:\n{doc[:300]}")
        print()

    # 测试检索
    print("══ 检索测试 ══════════════════════")
    query = "Synchron FDA"
    print(f"Query: {query}")
    print(ltm.search(query))


if __name__ == "__main__":
    main()
