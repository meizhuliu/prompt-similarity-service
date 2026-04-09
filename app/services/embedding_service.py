from sentence_transformers import SentenceTransformer
import re

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def normalize(self, text: str) -> str:
        return re.sub(r"\{\{.*?\}\}", "<VAR>", text)

    def embed(self, texts):
        texts = [self.normalize(t) for t in texts]
        return self.model.encode(texts, normalize_embeddings=True)
