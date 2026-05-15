from rich import print
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


local_llm = OllamaLLM(model='mistral')

# RAG

def rag(chunks, collection_name):
    vector_store = Chroma.from_documents(
        documents=documents,
        collection_name=collection_name,
        embedding=OllamaEmbeddings(model='nomic-embed-text')
    )

    retriever = vector_store.as_retriever()

    prompt_template = """Answer the question based only on the following context:
    {context}
    Question: {questions}
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

    result = chain.invoke("What is the use of Text Splitting?")
    print(result)

# 1. Character Text Splitting
print("### Character Text Splitting")

text = "Text splitting in Langchain is a critical feature that facilitates the division of large texts into smaller, manageable segments. "

# Manual Splitting
chunks = []
chunk_size = 35
for i in range(0, len(text), chunk_size):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)

documents = [Document(page_content=chunk, metadata={'source': "local"}) for chunk in chunks]
print(documents)


# Automatic Splitting
from langchain_classic.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    chunk_size=35,
    chunk_overlap=0,
    separator='',
    strip_whitespace=False
)
documents = text_splitter.create_documents([text])
print(documents)



# 2. Recursive Character Text Splitting
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500, 
    chunk_overlap=0
) # ["\n\n", "\n", " ", ""] 65,450
print(text_splitter.create_documents([text]))




# 3. Document Specific Splitting

# Markdown file splitting
from langchain_classic.text_splitter import MarkdownTextSplitter
with open('sample.md', 'r', encoding='utf-8') as file:
    markdown_text = file.read()

splitter = MarkdownTextSplitter(
    chunk_size=500,
    chunk_overlap=0
)

print(splitter.create_documents([markdown_text]))



# Python file splitting
from langchain_classic.text_splitter import PythonCodeTextSplitter

with open('main.py', 'r', encoding='utf-8') as file:
    python_text = file.read()

python_splitter = PythonCodeTextSplitter(
    chunk_size=1000,
    chunk_overlap=30
)
print(python_splitter.create_documents([python_text]))




# Javascript file
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter, Language

javascript_text = """

// Function is called, the return value will end up in x
let x = myFunction(4,3)

function myFunction(a,b) {
    return a * b;
}

"""

js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS,
    chunk_size=65,
    chunk_overlap=0
)

print(js_splitter.create_documents([javascript_text]))




# 4. Semantic chunking
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama.embeddings import OllamaEmbeddings


loader = PyPDFLoader("data/BIG Data Analytics.pdf")
docs = loader.load()
text_splitter = SemanticChunker(
    OllamaEmbeddings(model='nomic-embed-text'), 
    breakpoint_threshold_type="percentile"
)

documents = text_splitter.split_documents(docs)
print(documents)
