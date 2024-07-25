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

    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        return self.embedder.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True, convert_to_tensor=True)

    def average_cosine_similarity(self, embeddings_1: torch.Tensor, embeddings_2: torch.Tensor) -> float:
        """
        Measures the similarity by calculating the average cosine similarity between embeddings_1 and embeddings_2.
        """
        # Move tensors to the same device as the model
        embeddings_1 = embeddings_1.to(self.device)
        embeddings_2 = embeddings_2.to(self.device)
        
        similarities = util.dot_score(embeddings_1, embeddings_2)  # A matrix of similarities
        
        average_similarity = similarities.mean().item()  # Mean of all elements as a Python float
        return average_similarity
    
    
    def average_pairwise_cosine_similarity(self, embeddings_1: torch.Tensor, embeddings_2: torch.Tensor) -> float:
        """
        Measures the similarity by calculating the average cosine similarity between pairs of embeddings at corresponding indices (assume normalized).
        """
        if embeddings_1.shape[0] != embeddings_2.shape[0]:
            raise ValueError("Both embedding arrays must have the same number of samples")
        
        # Compute dot product for corresponding pairs
        similarities = torch.sum(embeddings_1 * embeddings_2, dim=1)
        
        # Compute mean and convert to Python float
        average_similarity = similarities.mean().item()
        
        return average_similarity


if __name__ == "__main__":

    human_text_samples = ["I'm a human", "I'm a human", "I'm a human", "I'm a human", "I'm a human"]
    llm_text_samples = ["I'm an LLM", "I'm an LLM but cooler", "I'm an LLM but nicer", "I'm an LLM but more helpful", "I'm an LLM but more helpful"]

    embeddings = EmbeddingSimilarity()
    embeddings_1 = embeddings.get_embeddings(human_text_samples)
    embeddings_2 = embeddings.get_embeddings(llm_text_samples)

    print(f"Average cosine similarity: {embeddings.average_cosine_similarity(embeddings_1, embeddings_2)}")
    print(f"Average pairwise cosine similarity: {embeddings.average_pairwise_cosine_similarity(embeddings_1, embeddings_2)}")