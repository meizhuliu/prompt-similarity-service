from fastapi import APIRouter
from app.api.routes_embeddings import store
from app.services.similarity_service import SimilarityService

# Router for prompt-level similarity operations
# This groups endpoints related to "finding similar prompts"
router = APIRouter()

# Service layer responsible for computing cosine similarity
# Uses embeddings stored in the vector store
service = SimilarityService(store)

@router.get("/{prompt_id}/similar")
def similar(prompt_id: str, limit: int = 5, threshold: float = 0.8):
    results = service.find_similar(prompt_id, threshold, limit)

    return [
        {
            "prompt_id": pid,
            "similarity_score": score
        }
        for pid, score in results
    ]
