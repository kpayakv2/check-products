#!/usr/bin/env python3
"""
Advanced Embedding Models
========================

Extended embedding models including Sentence Transformers support.
"""

import numpy as np
from typing import List, Optional
from pathlib import Path

from fresh_architecture import EmbeddingModel


class SentenceTransformerModel(EmbeddingModel):
    """Sentence Transformer based embedding model."""
    
    def __init__(self, 
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 cache_dir: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize Sentence Transformer model.
        
        Args:
            model_name: HuggingFace model name
            cache_dir: Cache directory for model storage
            device: Device to run model on ('cpu', 'cuda', etc.)
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerModel. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        
        # Initialize model
        print(f"🔄 Loading Sentence Transformer model: {model_name}")
        if cache_dir:
            self.model = SentenceTransformer(model_name, cache_folder=cache_dir, device=device)
        else:
            self.model = SentenceTransformer(model_name, device=device)
        
        print(f"✅ Model loaded successfully! Dimension: {self.get_dimension()}")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Sentence Transformer."""
        if not texts:
            return np.array([])
        
        # Convert to embeddings
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
            show_progress_bar=len(texts) > 100  # Show progress for large batches
        )
        
        return embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "dimension": self.get_dimension(),
            "device": str(self.model.device),
            "max_seq_length": getattr(self.model, 'max_seq_length', 'unknown')
        }


class OptimizedTFIDFModel(EmbeddingModel):
    """Optimized TF-IDF model using sklearn."""
    
    def __init__(self, 
                 max_features: int = 5000,
                 dimension: int = None,
                 ngram_range: tuple = (1, 2),
                 cache_dir: Optional[str] = None):
        """
        Initialize optimized TF-IDF model.
        
        Args:
            max_features: Maximum number of features
            dimension: Target dimension (if None, uses max_features)
            ngram_range: N-gram range for feature extraction
            cache_dir: Cache directory for model storage
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
        except ImportError:
            raise ImportError(
                "scikit-learn is required for OptimizedTFIDFModel. "
                "Install with: pip install scikit-learn"
            )
        
        self.max_features = max_features
        self.dimension = dimension or max_features
        self.ngram_range = ngram_range
        self.cache_dir = cache_dir
        
        # Initialize components
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            stop_words=None,  # Keep all words for product names
            token_pattern=r'\b\w+\b'
        )
        
        # Dimensionality reduction if needed
        self.reducer = None
        if self.dimension < max_features:
            self.reducer = TruncatedSVD(n_components=self.dimension, random_state=42)
        
        self.is_fitted = False
        print(f"🔧 Initialized Optimized TF-IDF (features={max_features}, dim={self.dimension})")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using optimized TF-IDF."""
        if not texts:
            return np.array([])
        
        if not self.is_fitted:
            print(f"🔄 Fitting TF-IDF vectorizer on {len(texts)} texts...")
            # Fit vectorizer
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Fit reducer if needed
            if self.reducer:
                print(f"🔄 Fitting dimensionality reduction...")
                self.reducer.fit(tfidf_matrix)
            
            self.is_fitted = True
            print(f"✅ TF-IDF model fitted successfully!")
        else:
            # Transform only
            tfidf_matrix = self.vectorizer.transform(texts)
        
        # Apply dimensionality reduction
        if self.reducer:
            embeddings = self.reducer.transform(tfidf_matrix)
        else:
            embeddings = tfidf_matrix.toarray()
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings = embeddings / norms
        
        return embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_type": "optimized_tfidf",
            "max_features": self.max_features,
            "dimension": self.dimension,
            "ngram_range": self.ngram_range,
            "is_fitted": self.is_fitted,
            "vocabulary_size": len(self.vectorizer.vocabulary_) if self.is_fitted else 0
        }


# Popular multilingual models
RECOMMENDED_MODELS = {
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
    "multilingual_mpnet": "paraphrase-multilingual-mpnet-base-v2", 
    "distiluse": "distiluse-base-multilingual-cased",
    "labse": "sentence-transformers/LaBSE",
    "thai": "airesearch/wangchanberta-base-att-spm-uncased"  # Thai-specific
}


def create_sentence_transformer(model_name: str = None, **kwargs) -> SentenceTransformerModel:
    """Create Sentence Transformer model with recommended defaults."""
    if model_name is None:
        model_name = RECOMMENDED_MODELS["multilingual"]
    elif model_name in RECOMMENDED_MODELS:
        model_name = RECOMMENDED_MODELS[model_name]
    
    return SentenceTransformerModel(model_name=model_name, **kwargs)


def create_optimized_tfidf(**kwargs) -> OptimizedTFIDFModel:
    """Create optimized TF-IDF model with recommended defaults."""
    return OptimizedTFIDFModel(**kwargs)
