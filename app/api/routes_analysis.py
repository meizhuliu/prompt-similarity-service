from fastapi import APIRouter, Query
from app.api.routes_embeddings import store
from app.services.clustering_service import cluster_duplicates

# Create a FastAPI router for grouping analysis-related endpoints
# This helps modularize API structure (separating analysis from embeddings/search/etc.)
router = APIRouter()


@router.get("/duplicates")
def get_duplicates(
    threshold: float = Query(
        0.9, ge=0.0, le=1.0
    )
):
    """
    Detect duplicate or near-duplicate prompts using embedding similarity.

    This endpoint:
    1. Computes similarity between prompt embeddings (via clustering logic)
    2. Groups prompts into connected components (clusters)
    3. Returns clusters of semantically similar prompts

    Args:
        threshold (float):
            Minimum cosine similarity required to consider two prompts as "connected".
            - Higher value (e.g., 0.95) → stricter duplicates
            - Lower value (e.g., 0.80) → broader semantic grouping

    Returns:
        List of clusters, where each cluster contains prompt IDs that are
        semantically similar or transitively connected.

    Example response:
    [
        {
            "cluster_id": 0,
            "prompts": ["survey.question.base", "survey.question.with_options"]
        }
    ]
    """

    # Step 1: Run clustering logic over all stored embeddings
    # This internally:
    # - computes pairwise or FAISS-filtered similarity
    # - builds a graph of connected prompts
    # - extracts connected components (clusters)
    clusters = cluster_duplicates(store, threshold)

    # Step 2: Format output into API-friendly structure
    # Each cluster gets a numeric ID for UI/debugging purposes
    return [
        {
            "cluster_id": i,
            "prompts": cluster  # list of prompt_ids in this cluster
        }
        for i, cluster in enumerate(clusters)
    ]
