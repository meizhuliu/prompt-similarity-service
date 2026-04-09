# FAISS is a high-performance library for similarity search over dense vectors
# We use it here to efficiently find similar prompt embeddings using cosine similarity (via inner product)
import faiss
import numpy as np


class FAISSStore:
    """
    A simple in-memory vector store built on FAISS.

    Responsibilities:
    - Store prompt embeddings
    - Maintain mapping between vector index and prompt IDs
    - Perform fast similarity search

    NOTE:
    This is an in-memory implementation.
    In production, this can be replaced with:
    - FAISS persisted index (disk)
    - Pinecone / Weaviate / Milvus
    """

    def __init__(self, dim=384):
        """
        Initialize FAISS index.

        Args:
            dim (int): dimensionality of embedding vectors
                      (e.g., MiniLM = 384 dimensions)
        """

        # IndexFlatIP = Inner Product similarity
        # When embeddings are normalized → this is equivalent to cosine similarity
        self.index = faiss.IndexFlatIP(dim)

        # Maps FAISS index positions → prompt_id
        # Example: index 0 → "survey.question.base"
        self.id_map = []

        # Stores actual embeddings keyed by prompt_id
        # Useful for retrieval, debugging, or recomputation
        self.vectors = {}

    def add(self, prompt_id, vector):
        """
        Add a single prompt embedding into FAISS index.

        Args:
            prompt_id (str): unique identifier of the prompt
            vector (List[float] | np.array): embedding vector
        """

        # FAISS expects float32 numpy arrays of shape (1, dim)
        vector = np.array([vector]).astype("float32")

        # Add vector to FAISS index for ANN search
        self.index.add(vector)

        # Maintain ID mapping (FAISS does not store metadata)
        self.id_map.append(prompt_id)

        # Store raw vector for direct access / similarity computation
        self.vectors[prompt_id] = vector[0]

    def search(self, vector, k=5):
        """
        Search for top-k most similar prompts.

        Args:
            vector: query embedding
            k (int): number of nearest neighbors

        Returns:
            List of tuples: (prompt_id, similarity_score)
        """

        # Convert query vector into FAISS-compatible format
        vector = np.array([vector]).astype("float32")

        # FAISS returns:
        # - scores: similarity scores
        # - idxs: indices in the index
        scores, idxs = self.index.search(vector, k)

        results = []

        # Iterate over top-k results
        for score, idx in zip(scores[0], idxs[0]):

            # -1 means no result found (can happen if index is small)
            if idx == -1:
                continue

            # Map FAISS internal index → prompt_id
            results.append((self.id_map[idx], float(score)))

        return results
