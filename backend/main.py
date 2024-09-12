from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio
import uvicorn
from dotenv import load_dotenv

#loading env variable
load_dotenv()

#creating fastapi app
app = FastAPI()

#adding middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#embedding model
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base", model_kwargs={"device": "cuda"})

#initializing 2 index db
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
