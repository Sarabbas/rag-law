from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio
from concurrent.futures import ThreadPoolExecutor

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

def load_and_split(pdf_path, text_splitter):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return text_splitter.split_documents(documents)

async def initialize_vectorstores():
    embeddings = get_embeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

    # Initialize Qdrant client
    qdrant_client = QdrantClient(url="http://qdrant:6333")

    # Define collection names
    en_collection_name = "law_documents_en"
    ar_collection_name = "law_documents_ar"

    # Check if collections exist, and delete them if necessary
    await check_and_delete_collection(qdrant_client, en_collection_name)
    await check_and_delete_collection(qdrant_client, ar_collection_name)

    # Load documents asynchronously
    docs_en = await asyncio.to_thread(lambda: load_and_split("/app/docs/Executive Regulation Law No 6-2016 - English.pdf", text_splitter))
    docs_ar = await asyncio.to_thread(lambda: load_and_split("/app/docs/Executive Regulation Law No 6-2016.pdf", text_splitter))

    # Recreate collections and insert documents
    qdrant_english = Qdrant.from_documents(
        docs_en[:100],
        embeddings,
        url="http://qdrant:6333",
        collection_name=en_collection_name,
        prefer_grpc=False,
    )

    qdrant_arabic = Qdrant.from_documents(
        docs_ar[:100],
        embeddings,
        url="http://qdrant:6333",
        collection_name=ar_collection_name,
        prefer_grpc=False,
    )

    print("English Collection:", qdrant_english)
    print("Arabic Collection:", qdrant_arabic)

    return qdrant_english, qdrant_arabic

async def check_and_delete_collection(client: QdrantClient, collection_name: str):
    """
    Check if the collection exists in Qdrant. If it exists, delete it to overwrite.
    """
    try:
        # Attempt to get the collection
        collection_info = client.get_collection(collection_name)
        if collection_info is not None:
            print(f"Collection '{collection_name}' exists. Deleting it to overwrite.")
            client.delete_collection(collection_name)
            print(f"Collection '{collection_name}' deleted.")
    except Exception as e:
        print(f"Collection '{collection_name}' does not exist. Proceeding to create a new one. Error: {e}")