import streamlit as st
import pandas as pd
import base64
import os
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from datetime import datetime
import pdfplumber

def pdf_get_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_file:
            for page in pdf_file.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    return text


def get_text_chunks(text, model_name):
    if model_name == "Google AI":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=1000,
        )
    chunks= text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks, model_name, api_key=None):
    if not chunks:
        raise ValueError("❌ Teks kosong — tidak bisa membuat vector store dari dokumen kosong.")

    if model_name == "Google AI":
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",  # ✅ Correct
            google_api_key=api_key
        )
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store



def get_conversational_chain(model_name, vector_store=None, api_key=None):
    if model_name == "Google AI":
        prompt_template = """
        Kamu adalah sebuah chatbot yang akan membantu menjawab pertanyaan dari pengguna
        Context: {context}
        Question: {question}

        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )
        chain = load_qa_chain(
            model,
            chain_type="stuff",
            prompt=prompt,
        )
        return chain

def user_input(user_question, model_name, api_key, text_chunks, conversation_history):
    vector_store = get_vector_store(text_chunks, model_name, api_key)
    user_question_output = ""
    response_output = ""

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain("Google AI", vector_store=new_db, api_key=api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    user_question_output = user_question
    response_output = response["output_text"]

    conversation_history.append((user_question_output, response_output, model_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Buku Panduan Tim"))

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_question_output)

    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response_output)

    if len(st.session_state.conversation_history) > 0:
        df = pd.DataFrame(st.session_state.conversation_history, columns=["Question", "Response", "Model", "Timestamp", "PDF Name"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download Conversation History as a CSV file</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
        st.markdown("To download the conversation history, click the button above.")
    st.success("✅ Dokumen berhasil diproses.")	

def main():
    st.set_page_config(page_title="Chat with Buku Panduan Tim", page_icon="📘")
    st.header("KNOWLEDGE CONTINUITY ASSISTANT 🤖")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    model_name = "Google AI"

    api_key = st.sidebar.text_input("Enter your Google API Key:")
    st.sidebar.markdown("Click [here](https://ai.google.dev/) to get an API key.")
    
    if not api_key:
        st.sidebar.warning("Please enter your Google API Key to proceed.")
        return

    st.sidebar.title("Menu:")
    col1, col2 = st.columns(2)
    reset_button = col2.button("Reset")
    clear_button = col1.button("Rerun")

    if reset_button:
        st.session_state.conversation_history = []
        st.session_state.user_question = None

    elif clear_button:
        if 'user_question' in st.session_state:
            st.warning("The previous query will be discarded.")
            st.session_state.user_question = ""
            if len(st.session_state.conversation_history) > 0:
                st.session_state.conversation_history.pop()
        else:
            st.warning("The question in the input will be queried again.")

    # Load PDF file once
    file_path = "Buku Panduan Tim Support.pdf"
    if not os.path.exists(file_path):
        st.error("File 'Buku Panduan Tim Support.pdf' tidak ditemukan di direktori aplikasi.")
        return

    with st.spinner("Processing 'Buku Panduan Tim Support.pdf'..."):
        with open(file_path, "rb") as f:
            pdf_docs = [f]
            text = pdf_get_text(pdf_docs)
            text_chunks = get_text_chunks(text, model_name)
        st.success("Dokumen untuk menjawab pertanyaan mengenai Panduan Tim Support berhasil diproses. Silakan ajukan pertanyaan.")

    user_question = st.text_input("Tanyakan sesuatu berdasarkan 'Buku Panduan Tim Support':")

    if user_question:
        user_input(user_question, model_name, api_key, text_chunks, st.session_state.conversation_history)
        st.session_state.user_question = ""

if __name__ == "__main__":
    main()
