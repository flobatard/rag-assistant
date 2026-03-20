from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os

DB_PATH = "db/"

def load_documents(folder="data"):
    docs = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder, file))
            docs.extend(loader.load())
    return docs


def create_vector_db():
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    db = FAISS.from_documents(chunks, embeddings)

    db.save_local(DB_PATH)
    print("✅ Base vectorielle créée")


def load_vector_db():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(DB_PATH, embeddings)


def create_qa_chain():
    db = load_vector_db()

    retriever = db.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa