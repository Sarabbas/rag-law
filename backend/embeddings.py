from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

def initialize_vectorstores():
    embeddings = get_embeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    docs_en = text_splitter.split_documents(PyPDFLoader("/app/docs/Executive Regulation Law No 6-2016 - English.pdf").load())
    docs_ar = text_splitter.split_documents(PyPDFLoader("/app/docs/Executive Regulation Law No 6-2016.pdf").load())

    # print("English Documents:", docs_en)
    # print("Arabic Documents:", docs_ar)

    qdrant_english = Qdrant.from_documents(
        docs_en,
        embeddings,
        location="http://qdrant:6333", 
        collection_name="law_documents_en",
    )

    qdrant_arabic = Qdrant.from_documents(
        docs_ar,
        embeddings,
        location="http://qdrant:6333", 
        collection_name="law_documents_ar",
    )
    
    print("English Collection:", qdrant_english)
    print("Arabic Collection:", qdrant_arabic)

    return qdrant_english, qdrant_arabic
