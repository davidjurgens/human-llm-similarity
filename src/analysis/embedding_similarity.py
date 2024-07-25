from sentence_transformers import SentenceTransformer, util
import torch
from typing import List
import numpy as np


class EmbeddingSimilarity(object):
    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2"
    ):
        device = 'cpu'
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        self.device = device
        self.embedder = SentenceTransformer(model_name, device=self.device)

    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return self.embedder.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)

    def average_cosine_similarity(self, embeddings_1: np.ndarray, embeddings_2: np.ndarray) -> float:
        """
        Measures the similarity by calculating the average cosine similarity between embeddings_1 and embeddings_2.
        """
        similarities = util.dot_score(embeddings_1, embeddings_2)  # A matrix of similarities
        similarities = similarities.cpu().numpy()  # Convert to numpy array
        
        if embeddings_1 is embeddings_2:
            # If same set, zero out the diagonal to avoid self-similarity
            np.fill_diagonal(similarities, 0)
        
        average_similarity = similarities.mean()  # mean of all elements
        return float(average_similarity)

if __name__ == "__main__":

    human_text_samples = ["I'm a human", "I'm a human", "I'm a human", "I'm a human", "I'm a human"]
    llm_text_samples = ["I'm an LLM", "I'm an LLM but cooler", "I'm an LLM but nicer", "I'm an LLM but more helpful", "I'm an LLM but more helpful"]

    embeddings = EmbeddingSimilarity()
    embeddings_1 = embeddings.get_embeddings(human_text_samples)
    embeddings_2 = embeddings.get_embeddings(llm_text_samples)

    print(f"Average cosine similarity: {embeddings.average_cosine_similarity(embeddings_1, embeddings_2)}")