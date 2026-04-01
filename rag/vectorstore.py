from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


_db_path = str(Path(__file__).parent.parent / "chroma_db")
_client = chromadb.PersistentClient(path=_db_path)
_embedding_fn = DefaultEmbeddingFunction()

def build_vectorstore(chunks: list, collection_name: str = "bci_docs"):
    """把 chunk 存入 Chroma 向量数据库"""
    collection = _client.get_or_create_collection(
        name=collection_name,
        embedding_function=_embedding_fn,
    )

    # 避免重复插入
    if collection.count() > 0:
        print(f"集合 '{collection_name}' 已有 {collection.count()} 条记录，跳过构建")
        return collection

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    # Chroma 一次最多插入 5000 条，分批处理
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )

    print(f"已存入 {collection.count()} 个 chunk 到向量数据库")
    return collection


def search(query: str, collection_name: str = "bci_docs", n_results: int = 3) -> list[str]:
    """检索最相关的 chunk"""
    collection = _client.get_collection(
        name=collection_name,
        embedding_function=_embedding_fn,
    )
    results = collection.query(query_texts=[query], n_results=n_results)
    return results["documents"][0]