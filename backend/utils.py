from .embeddings import get_embeddings,qdrant_arabic,qdrant_english


def route_query(query: str):
    if any("\u0600" <= c <= "\u06FF" for c in query):
        return qdrant_arabic.as_retriever(search_kwargs={"k": 2})
    else:
        return qdrant_english.as_retriever(search_kwargs={"k": 2})


