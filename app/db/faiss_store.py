import faiss
import numpy as np

class FAISSStore:
    def __init__(self, dim=384):
        self.index = faiss.IndexFlatIP(dim)
        self.id_map = []
        self.vectors = {}

    def add(self, prompt_id, vector):
        vector = np.array([vector]).astype("float32")
        self.index.add(vector)
        self.id_map.append(prompt_id)
        self.vectors[prompt_id] = vector[0]

    def search(self, vector, k=5):
        vector = np.array([vector]).astype("float32")
        scores, idxs = self.index.search(vector, k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.id_map[idx], float(score)))
        return results
