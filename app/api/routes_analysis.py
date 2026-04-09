from fastapi import APIRouter, Query
from app.api.routes_embeddings import store
from app.services.clustering_service import cluster_duplicates

router = APIRouter()

@router.get("/duplicates")
def get_duplicates(threshold: float = Query(0.9, ge=0.0, le=1.0)):
    clusters = cluster_duplicates(store, threshold)

    return [
        {
            "cluster_id": i,
            "prompts": cluster
        }
        for i, cluster in enumerate(clusters)
    ]