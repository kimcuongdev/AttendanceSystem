import numpy as np
from face_recognition.similarity import Similarity


class Recognizer:
    def __init__(self, similarity: Similarity):
        self.similarity = similarity
    def recognize(self, input_embedding: np.ndarray, retriever_results):
        """
        Output: (best_scores, best_item)
        """
        best_scores = -float("inf")
        best_item = None
        for item in retriever_results:
            embedding = item.get('embedding', None)
            if embedding is None:
                continue
            score = self.similarity.compute_similarity(input_embedding, embedding)
            if score > best_scores:
                best_scores = score
                best_item = item
        return best_scores, best_item