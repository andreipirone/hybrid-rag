from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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

retriever = qdrant.as_retriever(k=3)
# found_docs = retriever._get_relevant_documents()

# found_docs = qdrant.similarity_search("What are the top reasons to learn a language")

# for docs in found_docs:
#     print(docs.page_content)
#     print(docs.metadata)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

template = [("human", """You are an assistant for question-answering tasks. 
                    Use the following pieces of retrieved context to answer the question. 
                    If you don't know the answer, just say that you don't know. 
                    Use three sentences maximum and keep the answer concise.
                    Question: {question} 
                    Context: {context} 
                    Answer:""")]

prompt = ChatPromptTemplate(template)

llm = ChatOllama(model="gemma3:1b", temperature=0.5)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = input("> ")

response = rag_chain.invoke(query)
print(response)
