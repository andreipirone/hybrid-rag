from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = "./pdf/the-stranger-albert-camus.pdf"
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

for chunk in split_docs:
    print(chunk)