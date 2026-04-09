from fastapi import APIRouter
from app.api.routes_embeddings import store, embedder

router = APIRouter()

@router.post("/semantic")
def search(payload: dict):
    query = payload["query"]
    k = payload.get("limit", 10)

    q_vec = embedder.embed([query])[0]
    results = store.search(q_vec, k)

    return [{"prompt_id": pid, "score": score} for pid, score in results]
