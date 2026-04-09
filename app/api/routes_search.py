from fastapi import APIRouter
from app.api.routes_embeddings import store, embedder

router = APIRouter()

@router.post("/semantic")

def search(payload: dict
    """
    Perform semantic search over stored prompt embeddings.

    Args:
        payload (SemanticSearchRequest):
            - query: natural language search query
            - limit: number of results to return (default 10 if not provided)

    Returns:
        List[SemanticSearchResult]:
            Ranked list of prompts sorted by cosine similarity score.
    """

    query = payload["query"]
    k = payload.get("limit", 10)

    q_vec = embedder.embed([query])[0]
    results = store.search(q_vec, k)

    return [{"prompt_id": pid, "score": score} for pid, score in results]
