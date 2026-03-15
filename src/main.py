from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from uuid import uuid4

file_path = "./pdf/hyperpolyglot-handbook.pdf"
loader = PyPDFLoader(file_path)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,  
    add_start_index=True, 
)

docs = loader.load()
print(len(docs))
split_docs = text_splitter.split_documents(docs)
print(len(split_docs))

embeddings = OllamaEmbeddings(model="embeddinggemma")

uuids = [str(uuid4()) for _ in range(len(split_docs))]

qdrant = QdrantVectorStore.from_documents(
    split_docs,
    embeddings,
    ids = uuids,
    url="http://localhost:6333",
    collection_name="my_documents",
)

found_docs = qdrant.similarity_search("What are the top reasons to learn a language")
for found in found_docs:
    print(found)

