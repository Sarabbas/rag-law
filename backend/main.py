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



prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer: """

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

#conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

embeddings = get_embeddings()
qdrant_english, qdrant_arabic = initialize_vectorstores()



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            question = await websocket.receive_text()
            
            # Route the query
            routed_retriever = route_query(question)
            
            # Retrieve relevant documents
            docs = routed_retriever.get_relevant_documents(question)

            # Prepare context
            context = "\n".join([doc.page_content for doc in docs])
            
            # Prepare prompt
            prompt = PROMPT.format(context=context, question=question)
            
            # Generate response
            response = await asyncio.to_thread(llm, prompt)
            
            # Stream the response word by word
            for word in response.split():
                await websocket.send_text(word + " ")
                await asyncio.sleep(0.05)
            
            await websocket.send_text("\n")  # Send a newline to indicate end of response
        except Exception as e:
            await websocket.send_text(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)


