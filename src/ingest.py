from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

def ingest_pdf(pdf_path: str):
    print("Đang đọc PDF ...")
    loader = TextLoader(pdf_path, encoding="utf-8")
    documents = loader.load()
    print(repr(documents[0].page_content[:200]))
    print(f"Đọc xong {len(documents)} trang")

    print("Đang cắt nhỏ văn bản ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # mỗi đoạn tối đa 1000 ký tự
        chunk_overlap=200, # 200 ký tự trùng nhau giữa 2 đoạn liền kề
        separators = ["\nĐiều", "\nKhoản", "\n\n", "\n", " "]
        # ưu tiên cắt tại "Điều", "Khoản" để giữ nguyên cấu trúc luật
    )
    chunks = splitter.split_documents(documents)
    print(f"Cắt được {len(chunks)} đoạn")

    print("Đang embedding và lưu vào ChromaDB...")
    embeddings = HuggingFaceEmbeddings(
        model_name="keepitreal/vietnamese-sbert"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Xong! Đã lưu vào Chroma_db/")
    print(documents[0].page_content[:500])
    return vectorstore

if __name__ == "__main__":
    ingest_pdf(r"E:\PROJECTS\luatbot\data\luat_lao_dong.txt")