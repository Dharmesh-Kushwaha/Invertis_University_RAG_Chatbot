import os
import streamlit as st

from groq import Groq

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_history_aware_retriever

from langchain_community.document_loaders import TextLoader
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# -------------------------
# API KEYS
# -------------------------

os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
client = Groq(api_key=st.secrets["GROQ_API_KEY"])


# -------------------------
# EMBEDDINGS
# -------------------------

embeddings = HuggingFaceEmbeddings(
    model="all-MiniLM-L6-v2"
)


# -------------------------
# LOAD DOCUMENTS
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "Invertis_FAQ_Database.txt")

loader = TextLoader(file_path)
docs = loader.load()


# -------------------------
# TEXT SPLITTING
# -------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splits = text_splitter.split_documents(docs)


# -------------------------
# VECTOR DATABASE
# -------------------------

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="chroma_db"
)

retriever = vectorstore.as_retriever()


# -------------------------
# HISTORY AWARE RETRIEVER
# -------------------------

contextualize_q_system_prompt = (
    "Rephrase the user question based on chat history. Do not answer."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

history_aware_retriever = create_history_aware_retriever(
    None,  # not using LLM here
    retriever,
    contextualize_q_prompt
)


# -------------------------
# SIMPLE QA (NO LLM INSIDE LANGCHAIN)
# -------------------------

def get_context(question, session_id="user1"):
    response = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in response])
    return context


# -------------------------
# CHAT MEMORY
# -------------------------

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# -------------------------
# MAIN FUNCTION
# -------------------------

def ask_question(question, session_id="user1"):

    # Step 1: Get context
    context = get_context(question)

    # Step 2: Send to Groq LLM
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI assistant for Invertis University. "
                    "Answer ONLY using the provided context. "
                    "If answer is not present, say you don't know. "
                    "Keep answer short (max 4 sentences).\n\n"
                    f"Context:\n{context}"
                )
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return completion.choices[0].message.content
