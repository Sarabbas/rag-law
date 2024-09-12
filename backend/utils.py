from .embeddings import get_embeddings,qdrant_arabic,qdrant_english
from langchain.prompts import PromptTemplate

def route_query(query: str):
    if any("\u0600" <= c <= "\u06FF" for c in query):
        return qdrant_arabic.as_retriever(search_kwargs={"k": 2})
    else:
        return qdrant_english.as_retriever(search_kwargs={"k": 2})



prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer: """

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)