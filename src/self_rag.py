from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

def check_relevance(query: str, docs: list) -> str:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    context = "\n\n".join(doc.page_content for doc in docs)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Nhiệm vụ: đánh giá xem các điều khoản có trả lời TRỰC TIẾP câu hỏi không.

Quy tắc:
- SUFFICIENT: các điều khoản có thông tin trực tiếp trả lời câu hỏi
- INSUFFICIENT: các điều khoản không liên quan hoặc chỉ đề cập gián tiếp

Chỉ trả về đúng 1 từ: SUFFICIENT hoặc INSUFFICIENT"""),
        ("human", """Câu hỏi: {query}

Các điều khoản tìm được:
{context}

Đánh giá (chỉ 1 từ):"""),
        ("human", """Câu hỏi: {query}
        
Các điều khoản tìm được:
{context}""")
    ])
    
    chain = prompt | llm
    result = chain.invoke({"query": query, "context": context})
    return result.content.strip()

if __name__ == "__main__":
    from retriever import load_retriever
    
    retriever = load_retriever()
    
    tests = [
        "nghỉ thai sản mấy tháng",      # có trong DB
        "luật bảo hiểm thất nghiệp",    # không có trong DB
    ]
    
    for q in tests:
        docs = retriever.invoke(q)
        print(f"\nCâu hỏi: {q}")
        print("--- Chunks retrieve được ---")
        for i, doc in enumerate(docs):
            print(f"Chunk {i+1}: {doc.page_content[:150]}")
        verdict = check_relevance(q, docs)
        print(f"Đánh giá: {verdict}")