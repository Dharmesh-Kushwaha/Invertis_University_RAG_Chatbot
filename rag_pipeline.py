import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
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
# LLM SETUP
# -------------------------

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)


# -------------------------
# EMBEDDINGS
# -------------------------

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(
    model="all-MiniLM-L6-v2"
)


# -------------------------
# LOAD DOCUMENTS
# -------------------------

loader = TextLoader("Invertis_FAQ_Database.txt")
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
    embedding=embeddings
)

retriever = vectorstore.as_retriever()


# -------------------------
# HISTORY AWARE RETRIEVER
# -------------------------

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
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
# SYSTEM PROMPT
# -------------------------

system_prompt = (
    "You are an AI assistant for Invertis University, Bareilly, Uttar Pradesh, India. "
    "Your job is to help students, parents, and visitors by answering questions about "
    "the university such as courses, admissions, campus facilities, departments, "
    "placements, hostels, infrastructure, and other university-related information.\n"

    "Use ONLY the following retrieved context to answer the user's question.\n"

    "If the answer is not present in the context, say: "
    "'I'm not sure about that. Please visit the official Invertis University website.'\n"

    "Keep your answers clear and concise (maximum 4 sentences).\n\n"
    "{context}"
)


# -------------------------
# QA PROMPT
# -------------------------

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)


# -------------------------
# QA CHAIN
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
# CHAT MEMORY
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
# FUNCTION FOR STREAMLIT
# -------------------------

def ask_question(question, session_id="user1"):

    response = conversational_rag_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    )

    return response["answer"]