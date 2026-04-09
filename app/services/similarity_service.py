import numpy as np

class SimilarityService:
    def __init__(self, store):
        self.store = store

    def find_similar(self, prompt_id, threshold=0.8, limit=5):
        base = self.store.vectors[prompt_id]
        results = []

        for pid, vec in self.store.vectors.items():
            if pid == prompt_id:
                continue
            score = float(np.dot(base, vec))
            if score >= threshold:
                results.append((pid, score))

        return sorted(results, key=lambda x: x[1], reverse=True)[:limit]
