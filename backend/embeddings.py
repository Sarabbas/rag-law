from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter



def get_embeddings():
    return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base", model_kwargs={"device": "cuda"})

def initialize_vectorstores():
    embeddings = get_embeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    qdrant_english = Qdrant.from_documents(
        text_splitter.split_documents(PyPDFLoader("/docs/Executive Regulation Law No 6-2016 - English.pdf").load()),
        embeddings,
        location=":memory:",
        collection_name="law_documents_en",
    )

    qdrant_arabic = Qdrant.from_documents(
        text_splitter.split_documents(PyPDFLoader("/docs/Executive Regulation Law No 6-2016.pdf").load()),
        embeddings,
        location=":memory:",
        collection_name="law_documents_ar",
    )

    return qdrant_english, qdrant_arabic
