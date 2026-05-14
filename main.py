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