from fastapi import APIRouter
from app.services.embedding_service import EmbeddingService
from app.db.faiss_store import FAISSStore

router = APIRouter()

embedder = EmbeddingService()
store = FAISSStore()

@router.post("/generate")
def generate(payload: dict):
    prompts = payload["prompts"]

    texts = [p["content"] for p in prompts]
    embeddings = embedder.embed(texts)

    for p, e in zip(prompts, embeddings):
        store.add(p["prompt_id"], e)

    return {"status": "ok", "count": len(prompts)}
