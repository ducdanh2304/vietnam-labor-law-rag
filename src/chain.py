from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.query_rewriter import rewrite_query
from src.retriever import load_retriever

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
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    def retrieve_and_dedupe(input_data):
        query = input_data["input"] if isinstance(input_data, dict) else input_data
        docs = retriever.invoke(query)
        return format_docs(docs)

    chain = (
        {"context": RunnableLambda(retrieve_and_dedupe),
         "input": RunnablePassthrough(),
         "chat_history": lambda x: x["chat_history"]}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever, prompt

if __name__ == "__main__":
    chain, retriever, prompt = build_chain()

    query = "bị đuổi việc không báo trước thì sao"

    rewritten = rewrite_query(query)
    print(f"Gốc:     {query}")
    print(f"Rewrite: {rewritten}\n")

    result = chain.invoke({"input": rewritten, "chat_history": []})
    print(result)