"""
Abstract base classes for extensible components.

This module defines the interfaces that can be extended to add new functionality
to the product similarity checker.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict, Optional
import torch
from pathlib import Path


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def encode(self, texts: List[str], **kwargs) -> torch.Tensor:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            **kwargs: Additional model-specific parameters
            
        Returns:
            Tensor of shape [len(texts), embedding_dim]
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension of this model."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about this model."""
        pass


class SimilarityCalculator(ABC):
    """Abstract base class for similarity calculation algorithms."""
    
    @abstractmethod
    def compute_similarity(
        self,
        query_embeddings: torch.Tensor,
        reference_embeddings: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute similarity between query and reference embeddings.
        
        Args:
            query_embeddings: Query embeddings of shape [n_queries, embedding_dim]
            reference_embeddings: Reference embeddings of shape [n_refs, embedding_dim]
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            Similarity matrix of shape [n_queries, n_refs]
        """
        pass
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Get the name of this similarity algorithm."""
        pass


class TextPreprocessor(ABC):
    """Abstract base class for text preprocessing."""
    
    @abstractmethod
    def preprocess(self, text: str) -> str:
        """
        Preprocess a single text string.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        pass
    
    @abstractmethod
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of text strings.
        
        Args:
            texts: List of input texts to preprocess
            
        Returns:
            List of preprocessed texts
        """
        pass
    
    @abstractmethod
    def get_preprocessor_name(self) -> str:
        """Get the name of this preprocessor."""
        pass


class DataLoader(ABC):
    """Abstract base class for data loading."""
    
    @abstractmethod
    def load_data(self, file_path: Path) -> List[str]:
        """
        Load product names from a file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            List of product names
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        pass


class DataWriter(ABC):
    """Abstract base class for data writing."""
    
    @abstractmethod
    def write_results(
        self,
        results: List[Dict[str, Any]],
        output_path: Path,
        **kwargs
    ) -> None:
        """
        Write similarity results to a file.
        
        Args:
            results: List of result dictionaries
            output_path: Path where to write the results
            **kwargs: Additional writer-specific parameters
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get list of supported output formats."""
        pass


class CacheManager(ABC):
    """Abstract base class for caching systems."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached item or None if not found
        """
        pass
    
    @abstractmethod
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Put item in cache.
        
        Args:
            key: Cache key
            value: Item to cache
            ttl: Time to live in seconds (None for no expiration)
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cache status."""
        pass


class ProgressReporter(ABC):
    """Abstract base class for progress reporting."""
    
    @abstractmethod
    def start(self, total: int, description: str = "") -> None:
        """
        Start progress tracking.
        
        Args:
            total: Total number of items to process
            description: Description of the operation
        """
        pass
    
    @abstractmethod
    def update(self, increment: int = 1) -> None:
        """
        Update progress.
        
        Args:
            increment: Number of items completed
        """
        pass
    
    @abstractmethod
    def finish(self) -> None:
        """Finish progress tracking."""
        pass


class ResultFilter(ABC):
    """Abstract base class for filtering similarity results."""
    
    @abstractmethod
    def filter_results(
        self,
        results: List[Tuple[str, float]],
        **kwargs
    ) -> List[Tuple[str, float]]:
        """
        Filter similarity results based on criteria.
        
        Args:
            results: List of (product_name, similarity_score) tuples
            **kwargs: Filter-specific parameters
            
        Returns:
            Filtered list of results
        """
        pass
    
    @abstractmethod
    def get_filter_name(self) -> str:
        """Get the name of this filter."""
        pass
