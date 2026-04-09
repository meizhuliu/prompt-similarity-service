# FastAPI router for embedding-related endpoints
# This module is responsible for generating embeddings and storing them in a vector DB
from fastapi import APIRouter

# Embedding service: converts raw text into vector embeddings using a transformer model
from app.services.embedding_service import EmbeddingService

# FAISS-based vector store: stores embeddings and supports fast similarity search
from app.db.faiss_store import FAISSStore

# Create a router instance for grouping related endpoints
router = APIRouter()

# Initialize embedding model once (important for performance)
# Loading models is expensive → should NOT happen per request
embedder = EmbeddingService()

# Initialize vector store (in-memory FAISS index in this case)
# In production, this could be replaced with Redis / Pinecone / Weaviate
store = FAISSStore()


@router.post("/generate")
def generate(payload: dict):
    """
    Generate embeddings for a batch of prompts and store them in the vector database.

    Expected input format:
    {
        "prompts": [
            {
                "prompt_id": "example.id",
                "content": "text of prompt"
            }
        ]
    }

    Returns:
        - status: success indicator
        - count: number of prompts processed
    """

    # Extract prompt list from request payload
    prompts = payload["prompts"]

    # Extract raw text content from each prompt
    texts = [p["content"] for p in prompts]

    # Convert texts into dense vector embeddings
    # Shape: [num_prompts, embedding_dim]
    embeddings = embedder.embed(texts)

    # Store each prompt embedding in FAISS index
    # This enables future similarity search and deduplication
    for p, e in zip(prompts, embeddings):
        store.add(p["prompt_id"], e)

    # Return success response with processed count
    return {
        "status": "ok",
        "count": len(prompts)
    }
