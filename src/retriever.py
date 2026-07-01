from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader

def load_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="keepitreal/vietnamese-sbert"
    )
    vectorstore = Chroma(
        persist_directory=r"E:\PROJECTS\luatbot\src\chroma_db",
        embedding_function=embeddings
    )

    semantic = vectorstore.as_retriever(search_kwargs={"k":5})

    #Load lại documents để tạo BM25
    loader = TextLoader(r"E:\PROJECTS\luatbot\data\luat_lao_dong.txt", encoding="utf-8")
    documents = loader.load()
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\nĐiều ", "\nKhoản ", "\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(documents)

    bm25 = BM25Retriever.from_documents(chunks, k=5)

    hybrid = EnsembleRetriever(
        retrievers=[bm25, semantic],
        weights=[0.5, 0.5]
    )
    return hybrid


if __name__ == "__main__":
    retriever = load_retriever()

    query = "Người lao động được nghỉ thai sản mấy tháng?"
    results = retriever.invoke(query)

    print(f"Tìm được {len(results)} đoạn")  # thêm dòng này
    print(f"\nCâu hỏi: {query}\n")
    
    for i, doc in enumerate(results):
        print(f"--- Đoạn {i+1} ---")
        print(doc.page_content[:300])
        print()  # dòng trống giữa các đoạn, không phải print(results)