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
    llm = OllamaLLM(
        model='mistral',
        #base_url="http://host.docker.internal:11434" # For local docker container runtime
        #base_url="http://172.17.0.1:11434" # Docker bridge IP (AWS)
    )
    return llm

@st.cache_resource
def embedding():
    embed = OllamaEmbeddings(
        model='nomic-embed-text',
        #base_url="http://host.docker.internal:11434" # For local docker container runtime
        #base_url="http://172.17.0.1:11434" # Docker bridge IP (AWS)
    )
    return embed

@st.cache_resource
def semantic_chunker():
    embed = embedding()
    splitter = SemanticChunker(
        embed,
        breakpoint_threshold_type='percentile'
    )
    return splitter


def vector_store(chunks, collection_name):
    store = Chroma.from_documents(
        documents=chunks,
        collection_name=collection_name,
        embedding=embedding()
    )
    return store




# ---------------------------------------------------------------------------
# File Upload and Chunking (in Sidebar)
# ---------------------------------------------------------------------------



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
    st.header("Upload documents")
    st.caption("Supported formats: PDF, DOCX, CSV, XLSX. Upload one or more files and click Upload and Scan to build the knowledge base.")

    file_uploader = st.file_uploader(
        label='Upload file here',
        type=['pdf', 'docx', 'csv','xlsx'],
        accept_multiple_files=True,
        key='uploaded_files'
    )
    scan_button = st.button("Upload and Scan")

    if st.session_state.uploaded_files and scan_button:
        file_bytes_map = {f.name: f.read() for f in file_uploader}

        if file_bytes_map != st.session_state.get("last_file_bytes_map"):
            
            last_file_bytes_map = st.session_state.get("last_file_bytes_map", {})

            # Find files that are retained, new, and removed
            retained_files = {f for f in file_bytes_map if f in last_file_bytes_map}
            new_files = {f for f in file_bytes_map if f not in last_file_bytes_map}
            
            if not retained_files:
                # No common files — wipe the entire vector store
                if "vector_store" in st.session_state:
                    st.session_state.vector_store.delete_collection()
                    del st.session_state.vector_store
                
                # Clear chat history since documents have changed
                st.session_state.chat_history = []

                docs = process_documents(file_bytes_map)
                st.session_state.vector_store = vector_store(
                    docs,
                    collection_name=f"session_{st.session_state.get('session_id', 'default')}"
                )
                st.session_state.langchain_docs = docs

            elif new_files:
                # Some files are retained — only process and add new files
                new_file_bytes_map = {f: file_bytes_map[f] for f in new_files}
                new_docs = process_documents(new_file_bytes_map)

                st.session_state.vector_store.add_documents(new_docs)
                st.session_state.langchain_docs = st.session_state.langchain_docs + new_docs

            else:
                # All uploaded files already exist in vector store — do nothing
                st.info("All uploaded files are already scanned. Nothing to update.")

            st.session_state.scanned_files = list(file_bytes_map.keys())
            st.session_state.last_file_bytes_map = file_bytes_map

        st.success(f"Uploaded and scanned {len(file_uploader)} documents")





# --------------------------------------------------------------------------------
# RAG
# --------------------------------------------------------------------------------

def rag(question):
    if "vector_store" not in st.session_state:
        return None

    local_llm = ollama_llm()
    retriever = st.session_state.vector_store.as_retriever(
        search_kwargs={'k': 6}
    )

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
        return str(e)



# --------------------------------------------------------------------------------
# Chat Interface
# --------------------------------------------------------------------------------

st.title("🤖 QA Chatbot")
st.caption("Ask questions about the content of your uploaded documents. The chatbot answers using only scanned document context.")

scanned_files = st.session_state.get("scanned_files", [])

if scanned_files:
    # Show scanned files
    st.subheader("📂 Scanned Files")
    for file in scanned_files:
        st.markdown(f"- `{file}`")

    st.divider()

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Render chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag(user_input)

            if response is None:
                response = "Sorry, I couldn't find an answer based on the uploaded documents."

            st.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})

else:
    st.warning("Please upload your files in the sidebar and click **Upload and Scan** to get started.") 