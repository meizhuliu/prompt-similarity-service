from sentence_transformers import SentenceTransformer
import re


class EmbeddingService:
    """
    Service responsible for converting prompt text into vector embeddings.
    """

    def __init__(self):
        # Load pretrained SentenceTransformer model
        # MiniLM is chosen for speed + good semantic performance
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def normalize(self, text: str) -> str:
        """
        Normalize prompt text by replacing template variables.

        Example:
        "Ask {{user_name}}" → "Ask <VAR>"
        """

        # Replace all {{variable}} placeholders with <VAR>
        return re.sub(r"\{\{.*?\}\}", "<VAR>", text)

    def embed(self, texts):
        """
        Convert list of texts into normalized embedding vectors.
        """

        # Normalize template variables to reduce noise in embeddings
        texts = [self.normalize(t) for t in texts]

        # Generate embeddings
        # normalize_embeddings=True ensures cosine similarity can be used via dot product
        return self.model.encode(texts, normalize_embeddings=True)
