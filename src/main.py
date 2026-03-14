from langchain_community.document_loaders import PyPDFLoader

file_path = "./pdf/spring-start-here.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()
print(docs[44])