import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from langchain_community.document_loaders import TextLoader
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# -------------------------
# API KEYS
# -------------------------
groq_api_key = st.secrets["GROQ_API_KEY"]
os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]


# -------------------------
# LLM
# -------------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)


# -------------------------
# EMBEDDINGS
# -------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------------
# LOAD FILE
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "Invertis_FAQ_Database.txt")

loader = TextLoader(file_path)
docs = loader.load()


# -------------------------
# SPLIT TEXT
# -------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splits = text_splitter.split_documents(docs)


# -------------------------
# CLEAN TEXT
# -------------------------
texts = []
metadatas = []

for doc in splits:
    if doc.page_content and doc.page_content.strip():
        texts.append(doc.page_content.strip())
        metadatas.append(doc.metadata)


# -------------------------
# VECTOR STORE
# -------------------------
vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,
    persist_directory="chroma_db"
)

retriever = vectorstore.as_retriever()


# -------------------------
# CONTEXTUALIZE QUESTION (FIRST SYSTEM PROMPT)
# -------------------------
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt
)


# -------------------------
#  MAIN SYSTEM PROMPT 
# -------------------------
system_prompt = (
    "You are an AI assistant for Invertis University, Bareilly, Uttar Pradesh, India. "
    "Answer only using the provided context.\n"

    "If answer not found, say: "
    "'I'm not sure. Please visit official website.'\n"

    "Keep answer short (max 4 sentences).\n\n"
    "{context}"
)


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)


# -------------------------
# CHAINS
# -------------------------
question_answer_chain = create_stuff_documents_chain(
    llm,
    qa_prompt
)

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain
)


# -------------------------
# MEMORY
# -------------------------
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


# -------------------------
# FUNCTION
# -------------------------
def ask_question(question, session_id="user1"):
    response = conversational_rag_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    )
    return response["answer"]
