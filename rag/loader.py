from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(data_dir: str = "data") -> list:
    """加载 data/ 目录下所有 PDF，切分成 chunk"""
    docs = []
    for pdf_path in Path(data_dir).glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"加载了 {len(docs)} 页，切分成 {len(chunks)} 个 chunk")
    return chunks