# Invertis_University_RAG_Chatbot

Live demo: [https://invertis-chatbot.streamlit.app](https://invertisuniversityragchatbot-gw2yrd3j4ct2sutdpm9cse.streamlit.app/)

A chatbot built to answer real questions about Invertis University — courses, fees, admissions, hostel facilities, placements, and more. The goal is simple: instead of navigating multiple pages or PDFs, users can just ask a question and get a direct answer.

---

## The Problem

University websites are not always user-friendly when it comes to finding specific information. A simple query like “What is the BCA fee?” often requires navigating through multiple sections, downloading documents, and manually searching.

This project solves that by providing a conversational interface where users can ask questions naturally and get immediate responses.

---

## How It Works

This project uses a Retrieval-Augmented Generation (RAG) approach:

1. A curated FAQ dataset is used as the knowledge base  
2. The dataset is split into smaller chunks and converted into embeddings  
3. These embeddings are stored in a vector database (ChromaDB)  
4. When a user asks a question:
   - Relevant chunks are retrieved from the database  
   - The LLM generates an answer using only that context  
5. Conversation history is maintained so follow-up questions are understood correctly  

The system is designed to avoid hallucinations. If the answer is not found in the dataset, it returns a fallback response.

---

## Tech Stack

| Component | Tool | Reason |
|----------|------|--------|
| Frontend | Streamlit | Simple and fast UI with minimal setup |
| LLM | Groq (LLaMA 3.1) | Fast inference and reliable responses |
| Embeddings | HuggingFace MiniLM | Lightweight and efficient |
| Vector Database | ChromaDB | Easy to use, no separate server required |
| Framework | LangChain | Handles chaining, retrieval, and memory |
| Memory | RunnableWithMessageHistory | Maintains conversation context |

---

## Features

**Conversational memory**  
The chatbot understands follow-up questions. For example, asking “Tell me about BCA” and then “What is the fee?” works as expected.

**Context-aware retrieval**  
Follow-up questions are reformulated into standalone queries before retrieval, improving accuracy.

**Grounded responses**  
Answers are strictly generated using retrieved context. No guessing or unrelated information.

**Fallback handling**  
If the system cannot find relevant information, it returns a standard response instead of generating incorrect answers.

**Session-based interaction**  
Each user session maintains its own conversation history independently.

---

## Performance

- Average response time: 2–3 seconds  
- Works well for most queries covered in the dataset  
- Handles multi-turn conversations with reasonable accuracy  

Performance mainly depends on the external LLM API call.

---

## Project Structure
invertis-chatbot/
├── app.py # Streamlit frontend
├── rag_pipeline.py # RAG pipeline (retrieval + generation)
├── Invertis_FAQ_Database.txt # Knowledge base
├── requirements.txt # Dependencies
└── README.md
