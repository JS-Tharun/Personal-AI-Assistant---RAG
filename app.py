import streamlit as st
import os
import io
import tempfile
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_classic.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



st.set_page_config(
    page_title="QA Chatbot",
    layout='wide'
)

@st.cache_resource
def ollama_llm():
    llm = OllamaLLM(model='mistral')
    return llm

@st.cache_resource
def embedding():
    embed = OllamaEmbeddings(model='nomic-embed-text')
    return embed

@st.cache_resource
def semantic_chunker():
    embed = embedding()
    splitter = SemanticChunker(
        embed,
        breakpoint_threshold_type='percentile'
    )
    return splitter

@st.cache_resource
def vector_store(_chunks, collection_name):
    store = Chroma.from_documents(
        documents=_chunks,
        collection_name=collection_name,
        embedding=embedding()
    )
    return store



# ---------------------------------------------------------------------------
# File Upload and Chunking (in Sidebar)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Processing documents...")
def process_documents(file_bytes_map: dict[str, bytes]):
    all_documents = []
    for file, file_bytes in file_bytes_map.items():
        file_extension = file.split(".")[-1]

        # -------------------------
        # PDF and Docx temp file
        # -------------------------
        if file_extension in ['pdf','docx']:
            # Creating temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
                temp_file.write(file_bytes)
                temp_path = temp_file.name

            try:
                if file_extension == "pdf":
                    loader = PyPDFLoader(temp_path)
                    docs = loader.load()

                elif file_extension == 'docx':
                    loader = Docx2txtLoader(temp_path)
                    docs = loader.load()
                
                all_documents.extend(docs)

            finally:
                
                os.remove(temp_path)


        # ------------------------
        # CSV
        # ------------------------
        elif file_extension == "csv":
            df = pd.read_csv(io.BytesIO(file_bytes))
            documents = []

            for _, row in df.iterrows():
                text = " ".join([str(value) for value in row.values])
                documents.append(Document(
                    page_content=text
                ))

            all_documents.extend(documents)

        # -------------------------
        # Excel
        # -------------------------
        elif file_extension == 'xlsx':
            excel_data = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
            documents = []
            for sheet_name, df in excel_data.items():
                for _, row in df.iterrows():
                    text = " ".join([str(value) for value in row.values])
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"sheet": sheet_name}
                        )
                    )
            all_documents.extend(documents)

    #text_splitter = semantic_chunker()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(all_documents)

    return documents

    

if "langchain_docs" not in st.session_state:
    st.session_state.langchain_docs = []

if "scanned_files" not in st.session_state:
    st.session_state.scanned_files = []


with st.sidebar:
    file_uploader = st.file_uploader(
        label='Upload file here',
        type=['pdf', 'docx', 'csv','xlsx'],
        accept_multiple_files=True,
        key='uploaded_files'
    )
    scan_button = st.button("Upload and Scan")


    if st.session_state.uploaded_files and scan_button:
        file_bytes_map = {f.name: f.read() for f in file_uploader}
        docs = process_documents(file_bytes_map)
        st.session_state.langchain_docs = docs
        st.session_state.scanned_files = list(file_bytes_map.keys())
        st.success(f"Uploaded scaned {len(file_uploader)} documents")





# --------------------------------------------------------------------------------
# RAG
# --------------------------------------------------------------------------------

st.title("QA Chatbot")

def rag(chunks, collection_name, question):
    

    local_llm = ollama_llm()
    vec_store = vector_store(chunks, collection_name)
    retriever = vec_store.as_retriever()

    prompt_template = """Answer the question like reading from a text book, based only on the following context, without saying "Based on the context provided":
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        {
            'context': retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | local_llm
        | StrOutputParser()
    )

    query = str(question)
    
    if query.lower() in ['exit', 'quit', 'q']:
        return None

    try:
        
        result = chain.invoke(query)
        return result
    
    except Exception as e:
        return e


if not st.session_state.uploaded_files and not st.session_state.langchain_docs:
    st.write("Scan and Upload Documents to the Chatbot")

elif st.session_state.langchain_docs:
    st.info("Scanned Documents")
    for scanned_files in st.session_state.scanned_files:
        st.write(scanned_files)
    st.write(st.session_state.langchain_docs)


    user_input = st.text_input("Ask the bot")
    send_button = st.button("Send")

    if user_input and send_button:
        answer = rag(chunks=st.session_state.langchain_docs, collection_name='recursive', question=str(input))
        st.write(answer)
    


    