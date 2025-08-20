import numpy as np
from abc import ABC, abstractmethod
class Similarity(ABC):
    @abstractmethod
    def compute_similarity(self, u: np.ndarray, v: np.ndarray):
        pass


class Cosine_Similarity(Similarity):
    def compute_similarity(self, u: np.ndarray, v: np.ndarray):
        numerator = np.dot(u, v)
        denominator = np.linalg.norm(u) * np.linalg.norm(v)
        return numerator / denominator
        