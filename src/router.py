from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

def classify_query(query: str, chat_history: list = None) -> str:
    if chat_history is None:
        chat_history = []
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Phân loại câu hỏi của người dùng vào 1 trong 3 loại:
- LEGAL: câu hỏi liên quan đến luật lao động, quyền lợi, hợp đồng, lương, nghỉ phép...
- OUT_OF_SCOPE: câu hỏi không liên quan đến luật lao động
- UNCLEAR: câu hỏi quá mơ hồ, cần hỏi thêm thông tin

Chỉ trả về đúng 1 từ: LEGAL, OUT_OF_SCOPE, hoặc UNCLEAR"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{query}")
    ])

    chain = prompt | llm
    result = chain.invoke({"query": query, "chat_history": chat_history})
    return result.content.strip()

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage, AIMessage
    history = [
        HumanMessage(content="phụ nữ mang thai nghỉ mấy tháng"),
        AIMessage(content="Theo Điều 139, lao động nữ được nghỉ thai sản 06 tháng."),
    ]

    tests = [
        ("bị đuổi việc không báo trước thì sao", []),
        ("hôm nay thời tiết thế nào", []),
        ("tôi bị vấn đề", []),
        ("đàn ông thì sao?", history),   # kỳ vọng: LEGAL
    ]
    for q, h in tests:
        print(f"{q} → {classify_query(q, h)}")