"""
Concrete implementations of embedding models.

This module provides implementations of the EmbeddingModel interface
for different types of models.
"""

from typing import List, Dict, Any, Optional
import torch
from sentence_transformers import SentenceTransformer

from ..config import ModelConfig
from .interfaces import EmbeddingModel


class SentenceTransformerModel(EmbeddingModel):
    """Implementation using SentenceTransformers library."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the SentenceTransformer model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model = SentenceTransformer(
            config.name,
            cache_folder=config.cache_dir,
            device=config.device
        )
        self._embedding_dim = None
    
    def encode(self, texts: List[str], **kwargs) -> torch.Tensor:
        """Encode texts using SentenceTransformer."""
        batch_size = kwargs.get('batch_size', self.config.batch_size)
        convert_to_tensor = kwargs.get('convert_to_tensor', True)
        
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=convert_to_tensor,
            show_progress_bar=kwargs.get('show_progress_bar', False),
            **{k: v for k, v in kwargs.items() 
               if k not in ['batch_size', 'convert_to_tensor', 'show_progress_bar']}
        )
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension by encoding a dummy text."""
        if self._embedding_dim is None:
            dummy_embedding = self.encode(["dummy text"])
            self._embedding_dim = dummy_embedding.shape[-1]
        return self._embedding_dim
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "name": self.config.name,
            "type": "SentenceTransformer",
            "embedding_dim": self.get_embedding_dim(),
            "device": str(self.model.device),
            "max_seq_length": getattr(self.model, 'max_seq_length', None),
            "config": self.config.__dict__
        }


class MockEmbeddingModel(EmbeddingModel):
    """Mock implementation for testing purposes."""
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize mock model.
        
        Args:
            embedding_dim: Dimension of embeddings to generate
        """
        self.embedding_dim = embedding_dim
    
    def encode(self, texts: List[str], **kwargs) -> torch.Tensor:
        """Generate random embeddings for testing."""
        batch_size = len(texts)
        return torch.randn(batch_size, self.embedding_dim)
    
    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embedding_dim
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model information."""
        return {
            "name": "MockModel",
            "type": "Mock",
            "embedding_dim": self.embedding_dim,
            "device": "cpu",
            "config": {"embedding_dim": self.embedding_dim}
        }


class CachedEmbeddingModel(EmbeddingModel):
    """Wrapper that adds caching to any embedding model."""
    
    def __init__(self, base_model: EmbeddingModel, cache_manager: Optional[Any] = None):
        """
        Initialize cached model.
        
        Args:
            base_model: The underlying embedding model
            cache_manager: Cache manager instance (optional)
        """
        self.base_model = base_model
        self.cache_manager = cache_manager
        self._cache = {}  # Simple in-memory cache if no cache manager provided
    
    def _get_cache_key(self, texts: List[str], **kwargs) -> str:
        """Generate cache key for texts and parameters."""
        import hashlib
        
        # Create a deterministic key from texts and parameters
        text_str = "|".join(sorted(texts))
        kwargs_str = "|".join(f"{k}:{v}" for k, v in sorted(kwargs.items()))
        combined = f"{text_str}||{kwargs_str}"
        
        return hashlib.md5(combined.encode()).hexdigest()
    
    def encode(self, texts: List[str], **kwargs) -> torch.Tensor:
        """Encode with caching."""
        cache_key = self._get_cache_key(texts, **kwargs)
        
        # Try to get from cache
        if self.cache_manager:
            cached_result = self.cache_manager.get(cache_key)
        else:
            cached_result = self._cache.get(cache_key)
        
        if cached_result is not None:
            return cached_result
        
        # Compute embeddings
        embeddings = self.base_model.encode(texts, **kwargs)
        
        # Store in cache
        if self.cache_manager:
            self.cache_manager.put(cache_key, embeddings)
        else:
            self._cache[cache_key] = embeddings
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Delegate to base model."""
        return self.base_model.get_embedding_dim()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model info with caching information."""
        base_info = self.base_model.get_model_info()
        base_info.update({
            "cached": True,
            "cache_type": type(self.cache_manager).__name__ if self.cache_manager else "dict",
            "cache_size": len(self._cache) if not self.cache_manager else "unknown"
        })
        return base_info


def create_embedding_model(config: ModelConfig, enable_caching: bool = False) -> EmbeddingModel:
    """
    Factory function to create embedding models.
    
    Args:
        config: Model configuration
        enable_caching: Whether to enable caching
        
    Returns:
        Configured embedding model instance
    """
    # Create base model
    if config.name.lower() == "mock":
        base_model = MockEmbeddingModel(config.embedding_dim)
    else:
        base_model = SentenceTransformerModel(config)
    
    # Add caching if requested
    if enable_caching:
        base_model = CachedEmbeddingModel(base_model)
    
    return base_model
