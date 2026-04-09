import numpy as np

class SimilarityService:
    def __init__(self, store):
        self.store = store

    def find_similar(self, prompt_id, threshold=0.8, limit=5):
        base = self.store.vectors[prompt_id]
        results = []import numpy as np
from typing import Dict, List, Tuple, Any


class SimilarityService:
    """
    Service responsible for computing similarity between prompt embeddings
    and retrieving nearest neighbors based on cosine similarity (dot product
    on normalized vectors).
    """

    def __init__(self, store: Any) -> None:
        """
        Initialize with a vector store.

        Args:
            store:
                A vector store object that must expose:
                - store.vectors: Dict[str, np.ndarray]
                  mapping prompt_id → embedding vector
        """
        self.store = store

    def find_similar(
        self,
        prompt_id: str,
        threshold: float = 0.8,
        limit: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find prompts similar to the given prompt_id using cosine similarity.

        Args:
            prompt_id (str):
                ID of the reference prompt.

            threshold (float):
                Minimum similarity score required to include a match.
                Range: [0.0, 1.0]

            limit (int):
                Maximum number of results to return.

        Returns:
            List of tuples:
                [
                    (prompt_id, similarity_score),
                    ...
                ]
                Sorted in descending order of similarity.
        """

        # ----------------------------------------
        # 1. Retrieve embedding for base prompt
        # ----------------------------------------
        base: np.ndarray = self.store.vectors[prompt_id]

        results: List[Tuple[str, float]] = []

        # ----------------------------------------
        # 2. Compare against all stored embeddings
        #    (O(n) scan — can be optimized with FAISS)
        # ----------------------------------------
        for pid, vec in self.store.vectors.items():

            # Skip self-comparison
            if pid == prompt_id:
                continue

            # ----------------------------------------
            # 3. Compute cosine similarity
            #    (dot product assumes normalized vectors)
            # ----------------------------------------
            score: float = float(np.dot(base, vec))

            # ----------------------------------------
            # 4. Apply threshold filter
            # ----------------------------------------
            if score >= threshold:
                results.append((pid, score))

        # ----------------------------------------
        # 5. Sort by similarity (descending)
        #    and return top-K results
        # ----------------------------------------
        return sorted(results, key=lambda x: x[1], reverse=True)[:limit]
        for pid, vec in self.store.vectors.items():
            if pid == prompt_id:
                continue
            score = float(np.dot(base, vec))
            if score >= threshold:
                results.append((pid, score))

        return sorted(results, key=lambda x: x[1], reverse=True)[:limit]
