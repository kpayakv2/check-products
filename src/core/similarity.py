"""
Similarity calculation implementations.

This module provides implementations of the SimilarityCalculator interface
for different similarity metrics and algorithms.
"""

from typing import List, Tuple, Optional, Dict, Any
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..config import SimilarityConfig
from .interfaces import SimilarityCalculator


class CosineSimilarityCalculator(SimilarityCalculator):
    """Cosine similarity implementation."""
    
    def __init__(self, config: SimilarityConfig):
        """
        Initialize cosine similarity calculator.
        
        Args:
            config: Similarity configuration
        """
        self.config = config
    
    def compute_similarity(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between embeddings.
        
        Args:
            embeddings1: First set of embeddings [N, D]
            embeddings2: Second set of embeddings [M, D]
            
        Returns:
            Similarity matrix [N, M]
        """
        # Ensure tensors are on CPU for sklearn
        if isinstance(embeddings1, torch.Tensor):
            embeddings1 = embeddings1.cpu().numpy()
        if isinstance(embeddings2, torch.Tensor):
            embeddings2 = embeddings2.cpu().numpy()
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(embeddings1, embeddings2)
        
        return torch.from_numpy(similarity_matrix)
    
    def find_matches(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> List[Tuple[int, int, float]]:
        """
        Find matches above threshold.
        
        Args:
            embeddings1: First set of embeddings [N, D]
            embeddings2: Second set of embeddings [M, D]
            
        Returns:
            List of (idx1, idx2, similarity_score) tuples
        """
        similarity_matrix = self.compute_similarity(embeddings1, embeddings2)
        
        matches = []
        for i in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                score = float(similarity_matrix[i, j])
                if score >= self.config.threshold:
                    matches.append((i, j, score))
        
        # Sort by similarity score (descending)
        matches.sort(key=lambda x: x[2], reverse=True)
        
        return matches
    
    def get_top_k_matches(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor, k: int = 5) -> List[Tuple[int, int, float]]:
        """
        Get top-k matches regardless of threshold.
        
        Args:
            embeddings1: First set of embeddings [N, D]
            embeddings2: Second set of embeddings [M, D]
            k: Number of top matches to return
            
        Returns:
            List of top-k (idx1, idx2, similarity_score) tuples
        """
        similarity_matrix = self.compute_similarity(embeddings1, embeddings2)
        
        # Flatten matrix and get top-k indices
        flat_matrix = similarity_matrix.flatten()
        top_k_indices = torch.topk(flat_matrix, min(k, len(flat_matrix))).indices
        
        matches = []
        for flat_idx in top_k_indices:
            i = flat_idx // similarity_matrix.shape[1]
            j = flat_idx % similarity_matrix.shape[1]
            score = float(similarity_matrix[i, j])
            matches.append((int(i), int(j), score))
        
        return matches


class DotProductSimilarityCalculator(SimilarityCalculator):
    """Dot product similarity implementation."""
    
    def __init__(self, config: SimilarityConfig):
        """
        Initialize dot product similarity calculator.
        
        Args:
            config: Similarity configuration
        """
        self.config = config
    
    def compute_similarity(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        """
        Compute dot product similarity between embeddings.
        
        Args:
            embeddings1: First set of embeddings [N, D]
            embeddings2: Second set of embeddings [M, D]
            
        Returns:
            Similarity matrix [N, M]
        """
        # Normalize embeddings if specified in config
        if getattr(self.config, 'normalize_embeddings', False):
            embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=-1)
            embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=-1)
        
        # Compute dot product
        return torch.mm(embeddings1, embeddings2.T)
    
    def find_matches(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> List[Tuple[int, int, float]]:
        """Find matches above threshold."""
        similarity_matrix = self.compute_similarity(embeddings1, embeddings2)
        
        matches = []
        for i in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                score = float(similarity_matrix[i, j])
                if score >= self.config.threshold:
                    matches.append((i, j, score))
        
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches
    
    def get_top_k_matches(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor, k: int = 5) -> List[Tuple[int, int, float]]:
        """Get top-k matches regardless of threshold."""
        similarity_matrix = self.compute_similarity(embeddings1, embeddings2)
        
        flat_matrix = similarity_matrix.flatten()
        top_k_indices = torch.topk(flat_matrix, min(k, len(flat_matrix))).indices
        
        matches = []
        for flat_idx in top_k_indices:
            i = flat_idx // similarity_matrix.shape[1]
            j = flat_idx % similarity_matrix.shape[1]
            score = float(similarity_matrix[i, j])
            matches.append((int(i), int(j), score))
        
        return matches


class EuclideanSimilarityCalculator(SimilarityCalculator):
    """Euclidean distance-based similarity implementation."""
    
    def __init__(self, config: SimilarityConfig):
        """
        Initialize Euclidean similarity calculator.
        
        Args:
            config: Similarity configuration
        """
        self.config = config
    
    def compute_similarity(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity based on Euclidean distance.
        
        Similarity = 1 / (1 + distance)
        
        Args:
            embeddings1: First set of embeddings [N, D]
            embeddings2: Second set of embeddings [M, D]
            
        Returns:
            Similarity matrix [N, M]
        """
        # Compute pairwise Euclidean distances
        distances = torch.cdist(embeddings1, embeddings2, p=2)
        
        # Convert to similarity: similarity = 1 / (1 + distance)
        similarities = 1 / (1 + distances)
        
        return similarities
    
    def find_matches(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> List[Tuple[int, int, float]]:
        """Find matches above threshold."""
        similarity_matrix = self.compute_similarity(embeddings1, embeddings2)
        
        matches = []
        for i in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                score = float(similarity_matrix[i, j])
                if score >= self.config.threshold:
                    matches.append((i, j, score))
        
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches
    
    def get_top_k_matches(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor, k: int = 5) -> List[Tuple[int, int, float]]:
        """Get top-k matches regardless of threshold."""
        similarity_matrix = self.compute_similarity(embeddings1, embeddings2)
        
        flat_matrix = similarity_matrix.flatten()
        top_k_indices = torch.topk(flat_matrix, min(k, len(flat_matrix))).indices
        
        matches = []
        for flat_idx in top_k_indices:
            i = flat_idx // similarity_matrix.shape[1]
            j = flat_idx % similarity_matrix.shape[1]
            score = float(similarity_matrix[i, j])
            matches.append((int(i), int(j), score))
        
        return matches


class HybridSimilarityCalculator(SimilarityCalculator):
    """Hybrid similarity calculator combining multiple metrics."""
    
    def __init__(self, config: SimilarityConfig, weights: Optional[Dict[str, float]] = None):
        """
        Initialize hybrid similarity calculator.
        
        Args:
            config: Similarity configuration
            weights: Weights for different similarity metrics
        """
        self.config = config
        self.weights = weights or {'cosine': 0.7, 'euclidean': 0.3}
        
        self.calculators = {
            'cosine': CosineSimilarityCalculator(config),
            'euclidean': EuclideanSimilarityCalculator(config)
        }
    
    def compute_similarity(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted combination of similarity metrics.
        
        Args:
            embeddings1: First set of embeddings [N, D]
            embeddings2: Second set of embeddings [M, D]
            
        Returns:
            Combined similarity matrix [N, M]
        """
        combined_similarity = None
        total_weight = 0
        
        for metric, weight in self.weights.items():
            if metric in self.calculators:
                similarity = self.calculators[metric].compute_similarity(embeddings1, embeddings2)
                
                if combined_similarity is None:
                    combined_similarity = weight * similarity
                else:
                    combined_similarity += weight * similarity
                
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            combined_similarity = combined_similarity / total_weight
        
        return combined_similarity
    
    def find_matches(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> List[Tuple[int, int, float]]:
        """Find matches above threshold using combined similarity."""
        similarity_matrix = self.compute_similarity(embeddings1, embeddings2)
        
        matches = []
        for i in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                score = float(similarity_matrix[i, j])
                if score >= self.config.threshold:
                    matches.append((i, j, score))
        
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches
    
    def get_top_k_matches(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor, k: int = 5) -> List[Tuple[int, int, float]]:
        """Get top-k matches using combined similarity."""
        similarity_matrix = self.compute_similarity(embeddings1, embeddings2)
        
        flat_matrix = similarity_matrix.flatten()
        top_k_indices = torch.topk(flat_matrix, min(k, len(flat_matrix))).indices
        
        matches = []
        for flat_idx in top_k_indices:
            i = flat_idx // similarity_matrix.shape[1]
            j = flat_idx % similarity_matrix.shape[1]
            score = float(similarity_matrix[i, j])
            matches.append((int(i), int(j), score))
        
        return matches


def create_similarity_calculator(config: SimilarityConfig, calculator_type: str = "cosine") -> SimilarityCalculator:
    """
    Factory function to create similarity calculators.
    
    Args:
        config: Similarity configuration
        calculator_type: Type of calculator ('cosine', 'dot_product', 'euclidean', 'hybrid')
        
    Returns:
        Configured similarity calculator instance
    """
    if calculator_type.lower() == "cosine":
        return CosineSimilarityCalculator(config)
    elif calculator_type.lower() == "dot_product":
        return DotProductSimilarityCalculator(config)
    elif calculator_type.lower() == "euclidean":
        return EuclideanSimilarityCalculator(config)
    elif calculator_type.lower() == "hybrid":
        return HybridSimilarityCalculator(config)
    else:
        raise ValueError(f"Unknown similarity calculator type: {calculator_type}")
