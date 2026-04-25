import streamlit as st
from rag_pipeline import ask_question

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Invertis AI Chatbot",
    page_icon="🎓",
    layout="centered"
)

# -------------------------
# CUSTOM CSS
# -------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.chat-title {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    color: #2c3e50;
}
.chat-subtitle {
    text-align: center;
    font-size: 16px;
    color: gray;
    margin-bottom: 20px;
}
.user-msg {
    background-color: #d1e7dd;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 5px;
}
.bot-msg {
    background-color: #ffffff;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 5px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# TITLE
# -------------------------
st.markdown('<div class="chat-title">🎓 Invertis University AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="chat-subtitle">Ask about courses, fees, admissions & more</div>', unsafe_allow_html=True)

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.header("📌 About")
    st.write("AI chatbot for Invertis University queries.")
    
    st.markdown("---")
    st.subheader("💡 Example Questions")
    st.write("• What courses are available?")
    st.write("• What is BCA fee?")
    st.write("• Hostel facility details?")
    
    st.markdown("---")
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = []

# -------------------------
# SESSION STATE
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = "invertis_user"

# -------------------------
# DISPLAY CHAT
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)

# -------------------------
# INPUT
# -------------------------
user_input = st.chat_input("💬 Ask something about Invertis University...")

# -------------------------
# CHAT LOGIC
# -------------------------
if user_input:
    # Save & show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(f'<div class="user-msg">{user_input}</div>', unsafe_allow_html=True)

    # Generate response safely
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            try:
                answer = ask_question(
                    user_input,
                    session_id=st.session_state.session_id
                )
            except Exception as e:
                answer = "⚠️ Something went wrong. Please try again later."

        st.markdown(f'<div class="bot-msg">{answer}</div>', unsafe_allow_html=True)

    # Save response
    st.session_state.messages.append({"role": "assistant", "content": answer})
