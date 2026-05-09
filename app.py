import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ----------------------------------------------------------------
# 1. Load and preprocess document
#-----------------------------------------------------------------

print("Loading Document....")
loader = PyPDFLoader("data/THARUN_J_S_Resume.pdf")
docs = loader.load()
print("Document Loaded!")

print("Splitting Document into chunks....")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunks = splitter.split_documents(docs)
print(f"Document split into {len(chunks)} chunks!")


# ----------------------------------------------------------------
# 2. Embeddding + FAISS Vector Store
# ----------------------------------------------------------------

print("Creating embeddings and FAISS index....")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
print("Embeddings and FAISS index created!")


# ----------------------------------------------------------------
# 3. Build RetrievalQA
# ----------------------------------------------------------------

print("Building RetrievalQA chain....")
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

print("RetrievalQA chain built!")


# ----------------------------------------------------------------
# 4. Interactive Terminal Chatbot
# ----------------------------------------------------------------

print("\n🤖 Document Q&A Chatbot is ready! (type 'exit' to quit)\n")

while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit", "q"]:
        print("👋 Goodbye!")
        break

    try:
        response = qa_chain.invoke(query)
        print("Bot:", response, "\n")
    except Exception as e:
        print("❌ Error:", e)