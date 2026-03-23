from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

client = QdrantClient("http://localhost:6333")
embedding = OllamaEmbeddings(model = "embeddinggemma")

vector_store = QdrantVectorStore(
    client=client,
    collection_name="my_documents",
    embedding= embedding,
)

retriever = vector_store.as_retriever(k=3)

def format_docs(docs):
    for doc in docs:
        print(doc)
    return "\n\n".join(doc.page_content for doc in docs)

template = [("human", """You are an assistant for question-answering tasks. 
                    Use the following pieces of retrieved context to answer the question. 
                    If you don't know the answer, just say that you don't know. 
                    Use three sentences maximum and keep the answer concise.
                    Question: {question} 
                    Context: {context} 
                    Answer:""")]

prompt = ChatPromptTemplate(template)

llm = ChatOllama(model="gemma3:4b", temperature=0.5)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
