from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from uuid import uuid4
import os

FOLDER_PATH = ".\\pdf"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,  
    add_start_index=True, 
)

split_docs = []
for file_name in os.listdir(FOLDER_PATH):
    file_path = os.path.join(FOLDER_PATH, file_name)
    loader = PyPDFLoader(file_path)

    docs = loader.load()
    print(len(docs))
    split_docs.extend(text_splitter.split_documents(docs))
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
