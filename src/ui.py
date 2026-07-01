import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.chain import build_chain
from src.query_rewriter import rewrite_query
from src.router import classify_query 
from src.self_rag import check_relevance

st.set_page_config(page_title="LuatBot", page_icon="⚖️")
st.title("⚖️ LuatBot")
st.caption("Hỏi đáp Bộ luật Lao động Việt Nam 2019")

def get_chat_history_messages(max_turns=3):
    history = st.session_state.messages[-(max_turns*2):]
    return [
        HumanMessage(content=m["content"]) if m["role"] == "user"
        else AIMessage(content=m["content"])
        for m in history
    ]
# Khởi tạo chain 1 lần duy nhất, lưu vào session_state
if "chain" not in st.session_state:
    with st.spinner("Đang khởi động..."):
        st.session_state.chain, st.session_state.retriever, _ = build_chain()

# Lưu lịch sử hội thoại
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Ô nhập câu hỏi
if query := st.chat_input("Nhập câu hỏi về luật lao động..."):
    
    chat_history = get_chat_history_messages()
    # Hiển thị câu hỏi user
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Xử lý và trả lời
    with st.chat_message("assistant"):
        category = classify_query(query, chat_history)
        
        if category == "OUT_OF_SCOPE":
            answer = "Xin lỗi, mình chỉ hỗ trợ các câu hỏi về Bộ luật Lao động Việt Nam 2019. Bạn có thể hỏi về quyền lợi, hợp đồng, lương, nghỉ phép..."
            st.write(answer)
        
        elif category == "UNCLEAR":
            answer = "Câu hỏi của bạn hơi mơ hồ. Bạn có thể mô tả cụ thể hơn về tình huống của mình không? Ví dụ: bạn đang gặp vấn đề gì với công ty, liên quan đến lương hay hợp đồng?"
            st.write(answer)
        
        else:  # LEGAL
            with st.spinner("Đang tìm kiếm điều khoản..."):
                # Bước 1: rewrite query
                rewritten = rewrite_query(query, chat_history)
                
                # Bước 2: retrieve
                docs = st.session_state.retriever.invoke(rewritten)
                
                # Bước 3: self-RAG check
                verdict = check_relevance(rewritten, docs)
                
                if verdict == "INSUFFICIENT":
                    # Thử rewrite theo hướng khác rồi retrieve lần 2
                    rewritten2 = rewrite_query(query + " theo quy định pháp luật lao động Việt Nam", chat_history)
                    docs2 = st.session_state.retriever.invoke(rewritten2)
                    verdict2 = check_relevance(rewritten2, docs2)
                    
                    if verdict2 == "SUFFICIENT":
                        docs = docs2
                        rewritten = rewritten2
                    else:
                        answer = "Thông tin này nằm ngoài phạm vi Bộ luật Lao động 2019. Bạn có thể tham khảo các luật chuyên ngành liên quan."
                        st.write(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.stop()
                
                result = st.session_state.chain.invoke({"input": rewritten, "chat_history": chat_history})
                # st.write(type(result), result)
                answer = str(result)
            
            st.write(answer)
            with st.expander("🔍 Câu truy vấn đã được tối ưu"):
                st.write(rewritten)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})