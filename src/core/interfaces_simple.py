"""
Abstract base classes for extensible components (simplified for testing).
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict, Optional
from pathlib import Path
import numpy as np


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            **kwargs: Additional model-specific parameters
            
        Returns:
            Array of shape [len(texts), embedding_dim]
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        pass


class SimilarityCalculator(ABC):
    """Abstract base class for similarity calculators."""
    
    @abstractmethod
    def calculate_similarity(self, 
                           embedding1: np.ndarray, 
                           embedding2: np.ndarray) -> float:
        """
        Calculate similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (typically between 0 and 1)
        """
        pass
    
    @abstractmethod
    def calculate_batch_similarity(self, 
                                 embeddings1: np.ndarray, 
                                 embeddings2: np.ndarray) -> np.ndarray:
        """
        Calculate similarities between batches of embeddings.
        
        Args:
            embeddings1: First batch of embeddings [n, dim]
            embeddings2: Second batch of embeddings [m, dim]
            
        Returns:
            Similarity matrix of shape [n, m]
        """
        pass


class TextPreprocessor(ABC):
    """Abstract base class for text preprocessors."""
    
    @abstractmethod
    def preprocess(self, text: str) -> str:
        """
        Preprocess a single text.
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text string
        """
        pass
    
    @abstractmethod
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of input text strings
            
        Returns:
            List of preprocessed text strings
        """
        pass
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning (default implementation).
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        return text.strip()


class DataLoader(ABC):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def load(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load data from file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            List of data records
        """
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        pass


class DataWriter(ABC):
    """Abstract base class for data writers."""
    
    @abstractmethod
    def write(self, data: List[Dict[str, Any]], file_path: Path) -> None:
        """
        Write data to file.
        
        Args:
            data: List of data records to write
            file_path: Path where to write the data
        """
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        pass
