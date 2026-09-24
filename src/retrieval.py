from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from qdrant_client import QdrantClient
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore

from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


client = QdrantClient("http://localhost:6333")
embedding = OllamaEmbeddings(model = "embeddinggemma")

vector_store = QdrantVectorStore(
    client=client,
    collection_name="my_documents",
    embedding= embedding,
)

base_retriever = vector_store.as_retriever(k=30)

qdrant_docs = [
    doc for doc, _ in vector_store.similarity_search_with_score("", k=10000)
]

bm25_retriever = BM25Retriever.from_documents(qdrant_docs, k=30)
ensemble_retriever = EnsembleRetriever(
    retrievers=[base_retriever, bm25_retriever], 
    weights=[0.7, 0.3])

def format_docs(docs):
    for doc in docs:
        print(doc)
    return "\n\n".join(doc.page_content for doc in docs)

template = [
    ("system", "You are an assistant for question-answering tasks. "
    "Use the retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise."),
    ("human", "Question: {question}\n\nContext: {context}\n\nAnswer:")
]

prompt = ChatPromptTemplate(template)

llm = ChatOllama(model="gemma3:4b", temperature=0.5)

cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)


compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=ensemble_retriever,
)

rag_chain = (
    {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

