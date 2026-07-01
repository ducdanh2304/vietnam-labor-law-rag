from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

def ingest_pdf(pdf_path: str):
    print("Đang đọc PDF ...")
    loader = TextLoader(pdf_path, encoding="utf-8")
    documents = loader.load()
    print(repr(documents[0].page_content[:200]))
    print(f"Đọc xong {len(documents)} trang")

    print("Đang cắt nhỏ văn bản ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\nĐiều", "\nKhoản", "\n\n", "\n", " "]
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
        persist_directory=CHROMA_DIR
    )
    print(f"Xong! Đã lưu vào {CHROMA_DIR}")
    return vectorstore