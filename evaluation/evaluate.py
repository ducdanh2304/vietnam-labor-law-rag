from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.retriever import load_retriever
from src.query_rewriter import rewrite_query
from src.chain import build_chain

load_dotenv()

test_cases = [
    # Có trong data, retrieval dễ
    {"question": "Lao động nữ được nghỉ thai sản bao nhiêu tháng?", 
     "ground_truth": "6 tháng, Điều 139"},
    {"question": "Người lao động được nghỉ bao nhiêu ngày khi kết hôn?", 
     "ground_truth": "3 ngày, Điều 115"},
    {"question": "Công ty có thể sa thải người lao động đang mang thai không?", 
     "ground_truth": "Không được, Điều 137"},
    {"question": "Thời gian làm thêm tối đa trong năm là bao nhiêu giờ?", 
     "ground_truth": "200 giờ, trường hợp đặc biệt 300 giờ, Điều 107"},
    {"question": "Làm thêm giờ vào ngày nghỉ lễ được trả lương bao nhiêu phần trăm?", 
     "ground_truth": "Ít nhất 300%, Điều 98"},
    
    # Có trong data, retrieval khó hơn
    {"question": "Người lao động có thể đơn phương chấm dứt hợp đồng không xác định thời hạn không?", 
     "ground_truth": "Có, báo trước 45 ngày, Điều 35"},
    {"question": "Tuổi lao động tối thiểu là bao nhiêu?", 
     "ground_truth": "15 tuổi, Điều 143"},
    {"question": "Hình thức kỷ luật lao động có những loại nào?", 
     "ground_truth": "Khiển trách, kéo dài nâng lương, cách chức, sa thải, Điều 124"},
    {"question": "Người sử dụng lao động có thể trả lương chậm không?", 
     "ground_truth": "Không quá 30 ngày, phải trả lãi nếu chậm từ 15 ngày, Điều 97"},
    {"question": "Thời hiệu xử lý kỷ luật lao động là bao nhiêu tháng?", 
     "ground_truth": "6 tháng, trường hợp đặc biệt 12 tháng, Điều 123"},
]
def judge(question, answer, ground_truth):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Chấm điểm câu trả lời từ 1-5:
5: Chính xác hoàn toàn, có trích dẫn Điều đúng
4: Đúng nội dung nhưng thiếu số Điều
3: Đúng một phần
2: Sai nhưng có liên quan
1: Sai hoàn toàn hoặc không trả lời được

Chỉ trả về đúng 1 số từ 1-5."""),
        ("human", f"Câu hỏi: {question}\nCâu trả lời chuẩn: {ground_truth}\nCâu trả lời cần chấm: {answer}")
    ])
    result = (prompt | llm).invoke({})
    return int(result.content.strip())

if __name__ == "__main__":
    retriever = load_retriever()
    chain, _, _ = build_chain()
    
    scores = []
    for case in test_cases:
        rewritten = rewrite_query(case["question"])
        docs = retriever.invoke(rewritten)
        print(f"Chunks: {[doc.page_content[:80] for doc in docs]}")
        result = chain.invoke({"input": rewritten, "chat_history": []})
        answer = result
        score = judge(case["question"], answer, case["ground_truth"])
        scores.append(score)
        print(f"Q: {case['question']}")
        print(f"A: {answer[:100]}...")
        print(f"Score: {score}/5\n")
    
    print(f"=== TỔNG KẾT ===")
    print(f"Điểm trung bình: {sum(scores)/len(scores):.1f}/5")