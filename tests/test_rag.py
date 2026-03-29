from rag.loader import load_documents
from rag.vectorstore import build_vectorstore, search


def main():
    # 1. 加载并分块
    chunks = load_documents("../data")

    # 2. 存入向量数据库
    collection = build_vectorstore(chunks)

    # 3. 测试检索
    test_queries = [
        "brain computer interface applications",
        "EEG signal processing",
        "neural decoding",
    ]

    for q in test_queries:
        print(f"\n{'='*50}")
        print(f"查询: {q}")
        print(f"{'='*50}")
        results = search(q)
        for i, doc in enumerate(results):
            print(f"\n--- 结果 {i+1} ---")
            print(doc[:200] + "..." if len(doc) > 200 else doc)


if __name__ == "__main__":
    main()
