# 🌐 AI Web Scraper + 🤖 RAG Chatbot

An intelligent web app that combines **real-time web scraping**, **PDF document parsing**, and **Retrieval-Augmented Generation (RAG)** to enable users to ask contextual questions and receive AI-generated answers powered by **LLaMA 3** from Groq and **Hugging Face Embeddings**.

Built with 🧠 **LangChain**, 🖥️ **Streamlit**, 📄 **ChromaDB**, and ⚙️ **Groq LLM APIs**.

---

## 📌 Features

### 🔍 Web Scraping
- Scrape a given website URL and extract the readable content from the `<body>` tag.
- Cleaned content is shown in an expandable box and stored for chat-based queries.

### 📄 PDF Support
- Loads PDF documents using `PyPDFLoader`.
- Automatically chunks documents into semantic pieces with `RecursiveCharacterTextSplitter`.

### 🧠 RAG-Based Q&A
- Uses Hugging Face `all-MiniLM-L6-v2` embeddings for semantic vector storage via Chroma.
- Retrieves relevant document chunks based on the question context.
- Uses Groq’s LLaMA 3 (llama3-8b-8192) via LangChain to answer user queries.

### 💬 Conversational Memory
- Maintains per-session chat history using LangChain’s `ChatMessageHistory`.
- Capable of understanding follow-up questions using history-aware retrieval.

---

## 📁 Project Structure


---

## 🔧 Tech Stack

| Component        | Description                              |
|------------------|------------------------------------------|
| Streamlit        | UI for input/output interaction          |
| LangChain        | RAG pipeline + memory + chaining         |
| Chroma           | Vector DB for storing document chunks    |
| HuggingFace      | MiniLM Embeddings                        |
| Groq             | LLaMA 3 LLM (`llama3-8b-8192`)           |
| PyPDFLoader      | PDF document reading                     |

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

