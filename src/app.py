from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from query_rewriter import rewrite_query
from retriever import load_retriever

load_dotenv()

def format_docs(docs):
    seen = set()
    unique_docs = []
    for doc in docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)
    return "\n\n".join(doc.page_content for doc in unique_docs)

def build_chain():
    retriever = load_retriever()
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

    system_prompt = """Bạn là trợ lý pháp lý chuyên về Bộ luật Lao động Việt Nam 2019.

Dựa vào các điều khoản dưới đây để trả lời câu hỏi:
- Trả lời thẳng vào câu hỏi
- Luôn trích dẫn số Điều cụ thể
- Nếu context không đủ thông tin thì nói "Thông tin này không có trong dữ liệu hiện tại"

{context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    def retrieve_and_dedupe(input_data):
        query = input_data["input"] if isinstance(input_data, dict) else input_data
        docs = retriever.invoke(query)
        seen = set()
        unique = []
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique.append(doc)
        return "\n\n".join(doc.page_content for doc in unique)

    chain = (
        {"context": RunnableLambda(retrieve_and_dedupe), "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever, prompt

if __name__ == "__main__":
    chain, retriever, prompt = build_chain()
    
    query = "bị đuổi việc không báo trước thì sao"
    
    # Bước 1: rewrite
    rewritten = rewrite_query(query)
    print(f"Gốc:     {query}")
    print(f"Rewrite: {rewritten}\n")
    
    # Bước 2: dùng câu rewrite để tìm và trả lời
    result = chain.invoke(rewritten)
    print(result)  