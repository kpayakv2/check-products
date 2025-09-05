"""
Mock utility functions for testing sentence-transformers functionality.

This module provides mock implementations of utility functions from 
sentence-transformers, particularly cosine similarity computation.
"""

import torch
import torch.nn.functional as F


def cos_sim(a, b):
    """
    Compute cosine similarity between two tensors.
    
    This is a real implementation of cosine similarity that works with
    both real embeddings and mock embeddings from the test framework.
    
    Args:
        a: First tensor of shape [batch_size_a, embedding_dim]
        b: Second tensor of shape [batch_size_b, embedding_dim]
    
    Returns:
        Tensor of shape [batch_size_a, batch_size_b] containing cosine similarities
    """
    # Ensure both tensors are on the same device
    if a.device != b.device:
        b = b.to(a.device)
    
    # Ensure both tensors are 2D
    if a.dim() == 1:
        a = a.unsqueeze(0)  # [embedding_dim] -> [1, embedding_dim]
    if b.dim() == 1:
        b = b.unsqueeze(0)  # [embedding_dim] -> [1, embedding_dim]
    
    # Ensure both tensors have the same embedding dimension
    if a.shape[-1] != b.shape[-1]:
        raise ValueError(f"Embedding dimensions must match: a.shape[-1]={a.shape[-1]}, b.shape[-1]={b.shape[-1]}")
    
    # Normalize both tensors
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    
    # Compute cosine similarity via matrix multiplication
    return torch.matmul(a, b.T)
