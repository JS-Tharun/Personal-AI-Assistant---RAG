import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import pipeline

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# -------------------------------------------------
# 1. Streamlit UI
# -------------------------------------------------

st.set_page_config(
    page_title="Document Q&A Chatbot",
    layout="wide"
)

st.title("Document Q&A Chatbot")
st.write("Upload a PDF document and ask questions about its content!")

# -------------------------------------------------
# 2. File Upload + Preprocessing
# -------------------------------------------------

uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])

if uploaded_file:
    with open("uploaded_document.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    # Load and preprocess document
    loader = PyPDFLoader("uploaded_document.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    # Embeddings + FAISS
    @st.cache_resource
    def huggingface_embedding():
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    @st.cache_resource
    def faiss_vectorstore(chunks, _embeddings):
        return FAISS.from_documents(chunks, _embeddings)

    embeddings = huggingface_embedding()
    vectorstore = faiss_vectorstore(chunks, embeddings)

    # Build RetrievalQA
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=GEMINI_API_KEY
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k":3}),
        chain_type="stuff"
    )

    # -------------------------------------------------
    # 4. Chat Interface
    # -------------------------------------------------

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Ask a question about the contract:")

    if query:
        with st.spinner("Thinking..."):
            response = qa_chain.invoke(query)

        # Save chat history
        chat = {
            "You" : query,
            "Bot" : response['result']
        }
        st.session_state.chat_history.append(chat)

        st.write(f"{response['result']}")

    # Chat History

    for chat in reversed(st.session_state.chat_history):
        for entity, context in chat.items():
            if entity == 'You':
                st.write(f"{entity}: {context}")
            else:
                st.write(f"{entity}: {context}")
