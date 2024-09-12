from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
import asyncio
import uvicorn
from dotenv import load_dotenv

from .embeddings import get_embeddings,initialize_vectorstores
from .utils import route_query

#loading env variable
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

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

#llm
# llm = HuggingFaceHub(
#     repo_id="Qwen/Qwen2-7B"
#     model_kwargs={"temperature": 0.3, "max_length": 1024},
#     huggingfacehub_api_token=huggingface_token
# )

llm = HuggingFaceHub(
    repo_id="Qwen/Qwen2-7B",
    model_kwargs={"temperature": 0.3, "max_length": 512, "device": "cuda"},  # Reduce max_length
    huggingfacehub_api_token=huggingface_token
)

#conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


