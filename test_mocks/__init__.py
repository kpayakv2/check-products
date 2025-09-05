"""
Mock implementations for testing sentence-transformers functionality.

This module provides lightweight mock classes that mimic the behavior of 
sentence-transformers without requiring large model downloads or heavy computation.
Used exclusively for testing purposes.
"""

from . import util


class SentenceTransformer:
    """
    Mock implementation of SentenceTransformer for testing.
    
    This class mimics the interface of sentence_transformers.SentenceTransformer
    but generates random embeddings instead of using actual pretrained models.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize mock transformer.
        
        Args:
            model_name: Name of the model (not actually used, just stored)
        """
        self.model_name = model_name

    def encode(self, sentences, convert_to_tensor=True):
        """
        Generate mock embeddings for input sentences.
        
        Args:
            sentences: String or list of strings to encode
            convert_to_tensor: Whether to return PyTorch tensor or numpy array
            
        Returns:
            Mock embeddings with shape [num_sentences, 384]
        """
        import torch
        
        # สร้าง mock embeddings ที่มี shape ถูกต้อง
        # ใช้ embedding dimension = 384 (เหมือน MiniLM model)
        embedding_dim = 384
        
        if isinstance(sentences, str):
            sentences = [sentences]
        
        # สร้าง random embeddings สำหรับแต่ละ sentence
        batch_size = len(sentences)
        embeddings = torch.randn(batch_size, embedding_dim)
        
        if convert_to_tensor:
            return embeddings
        else:
            return embeddings.numpy()
