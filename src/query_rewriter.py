from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

def rewrite_query(query: str, chat_history: list = None) -> str:
    if chat_history is None:
        chat_history = []
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
   ("system", """Bạn là chuyên gia pháp lý Việt Nam.
Nhiệm vụ: viết lại câu hỏi của người dùng thành câu truy vấn pháp lý ngắn gọn, dùng đúng thuật ngữ luật.
Quy tắc:
- Chỉ viết lại câu hỏi, KHÔNG trả lời
- Dùng thuật ngữ pháp lý chính xác
- Giữ dạng câu hỏi
- Chỉ trả về câu hỏi đã viết lại, không có gì khác"""),
    MessagesPlaceholder("chat_history"),
        ("human", "Câu hỏi gốc: {query}\nCâu truy vấn pháp lý:")
    ])

    chain = prompt | llm
    result = chain.invoke({"query": query, "chat_history": chat_history} )
    return result.content

if __name__ == "__main__":
    test_cases = [
        "bị đuổi việc không báo trước thì sao",
        "nghỉ đẻ mấy tháng",
        "lương tháng 13 có bắt bược không"
    ]
    for q in test_cases:
        rewritten = rewrite_query(q)
        print(f"Gốc: {q}")
        print(f"Rewrite: {rewritten}")
        print()