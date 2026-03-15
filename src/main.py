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
print(split_docs[67])

embed = OllamaEmbeddings(model="embeddinggemma")

vector_client = QdrantClient(url="http://localhost:6333")

vector_client.create_collection(
    collection_name="demo_collection",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=vector_client,
    collection_name="demo_collection",
    embedding=embed,
)

uuids = [str(uuid4()) for _ in range(len(split_docs))]
vector_store.add_documents(documents=split_docs, ids=uuids)

