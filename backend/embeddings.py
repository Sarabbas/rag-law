from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio
from concurrent.futures import ThreadPoolExecutor

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")


def load_and_split(pdf_path, text_splitter):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return text_splitter.split_documents(documents)

async def initialize_vectorstores():
    embeddings = get_embeddings()  # Ensure this function is not async if it doesn’t need to be
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # Load documents asynchronously if the loading is I/O-bound
    docs_en = await asyncio.to_thread(lambda: text_splitter.split_documents(PyPDFLoader("/app/docs/Executive Regulation Law No 6-2016 - English.pdf").load()))
    docs_ar = await asyncio.to_thread(lambda: text_splitter.split_documents(PyPDFLoader("/app/docs/Executive Regulation Law No 6-2016.pdf").load()))

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
