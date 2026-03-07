import sys
import os
import pysqlite3

sys.modules["sqlite3"] = pysqlite3

import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain_core.documents import Document

from scrape import scrape_website, extract_body_content, clean_body_content

from pypdf import PdfReader


load_dotenv()

st.set_page_config(page_title="AI Web Scraper + RAG", layout="wide")

st.title("🌐 AI Web Scraper + 🤖 RAG Chatbot")

col1, col2 = st.columns([2,3])


# -------------------------------
# SCRAPER
# -------------------------------

with col1:

    st.subheader("Data Source")

    source = st.radio(
        "Choose Source",
        ["Website URL","Upload PDF"]
    )

    # WEBSITE SCRAPER
    if source == "Website URL":

        url = st.text_input("Enter Website URL")

        if st.button("Scrape Website"):

            with st.spinner("Scraping website..."):

                html = scrape_website(url)

                body = extract_body_content(html)

                cleaned = clean_body_content(body)

                st.session_state.dom = cleaned

                st.success("Website scraped successfully!")

    # PDF UPLOAD
    if source == "Upload PDF":

        pdf_file = st.file_uploader(
            "Upload PDF",
            type="pdf"
        )

        if pdf_file is not None:

            reader = PdfReader(pdf_file)

            text = ""

            for page in reader.pages:
                text += page.extract_text()

            st.session_state.dom = text

            st.success("PDF uploaded successfully!")

    if "dom" in st.session_state:

        st.text_area(
            "Content",
            st.session_state.dom,
            height=300
        )


# -------------------------------
# RAG CHATBOT
# -------------------------------

with col2:

    st.subheader("Chat With Data")

    session_id = st.text_input("Session ID", value="default_session")

    if "dom" in st.session_state:

        text = st.session_state.dom

        documents = [Document(page_content=text)]

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        docs = splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=None
        )

        retriever = vectorstore.as_retriever(
            search_kwargs={"k":3}
        )

        api_key = os.getenv("GROQ_API_KEY")

        if api_key:

            llm = ChatGroq(
                groq_api_key=api_key,
                model_name="llama-3.3-70b-versatile"
            )

            contextualize_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system","Rephrase question standalone"),
                    MessagesPlaceholder("chat_history"),
                    ("human","{input}")
                ]
            )

            history_retriever = create_history_aware_retriever(
                llm,
                retriever,
                contextualize_prompt
            )

            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system",
                     """You are an assistant.

Answer ONLY using the provided content.

Rules:
- Do NOT guess
- Do NOT add new information
- Only answer from the context
- If answer not found say:
'I cannot find this information in the document.'

Content:
{context}
"""
                    ),

                    MessagesPlaceholder("chat_history"),
                    ("human","{input}")
                ]
            )

            qa_chain = create_stuff_documents_chain(
                llm,
                qa_prompt
            )

            rag_chain = create_retrieval_chain(
                history_retriever,
                qa_chain
            )

            if "store" not in st.session_state:
                st.session_state.store = {}

            def get_session_history(session_id: str) -> BaseChatMessageHistory:

                if session_id not in st.session_state.store:
                    st.session_state.store[session_id] = ChatMessageHistory()

                return st.session_state.store[session_id]

            conversational_rag = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            question = st.text_input("Ask Question")

            if question:

                response = conversational_rag.invoke(
                    {"input":question},
                    config={"configurable":{"session_id":session_id}}
                )

                st.success(response["answer"])

        else:

            st.warning("Set GROQ_API_KEY in .env")

    else:

        st.info("Please scrape a website or upload a PDF first.")