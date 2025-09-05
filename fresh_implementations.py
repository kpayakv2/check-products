"""
Fresh Extensible Architecture - Concrete Implementations
=======================================================

Lightweight implementations for immediate testing
"""

import re
import unicodedata
import json
import csv
from typing import List, Dict, Any, Union
from pathlib import Path
import numpy as np
import pandas as pd

from fresh_architecture import (
    EmbeddingModel, SimilarityCalculator, TextProcessor, 
    DataSource, DataSink, Config
)


# =============================================================================
# TEXT PROCESSING IMPLEMENTATIONS
# =============================================================================

class BasicTextProcessor(TextProcessor):
    """Basic text cleaning and normalization."""
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 normalize_whitespace: bool = True):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.normalize_whitespace = normalize_whitespace
        
        # Compile regex patterns
        self.punct_pattern = re.compile(r'[^\w\s]', re.UNICODE)
        self.whitespace_pattern = re.compile(r'\s+')
    
    def process(self, text: str) -> str:
        """Process single text."""
        if not isinstance(text, str):
            return str(text)
        
        result = text
        
        # Normalize unicode
        result = unicodedata.normalize('NFKD', result)
        
        # Lowercase
        if self.lowercase:
            result = result.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            result = self.punct_pattern.sub(' ', result)
        
        # Normalize whitespace
        if self.normalize_whitespace:
            result = self.whitespace_pattern.sub(' ', result)
        
        return result.strip()
    
    def process_batch(self, texts: List[str]) -> List[str]:
        """Process multiple texts."""
        return [self.process(text) for text in texts]


class ThaiTextProcessor(BasicTextProcessor):
    """Thai-specific text processing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.thai_pattern = re.compile(r'[\u0e00-\u0e7f]+')
    
    def process(self, text: str) -> str:
        """Process with Thai-specific handling."""
        result = super().process(text)
        
        # Additional Thai processing could go here
        # e.g., Thai word segmentation, tone mark handling
        
        return result


# =============================================================================
# EMBEDDING MODEL IMPLEMENTATIONS
# =============================================================================

class MockEmbeddingModel(EmbeddingModel):
    """Mock embedding model for testing (no dependencies)."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        np.random.seed(42)  # For reproducible results
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate mock embeddings based on text length and content."""
        embeddings = []
        
        for text in texts:
            # Create pseudo-embedding based on text characteristics
            text_hash = hash(text.lower().strip())
            np.random.seed(abs(text_hash) % 2**31)
            
            # Base embedding from random
            embedding = np.random.normal(0, 1, self.dimension)
            
            # Add text-specific features
            embedding[0] = len(text) / 100.0  # Length feature
            embedding[1] = len(text.split()) / 50.0  # Word count feature
            embedding[2] = 1.0 if any(c.isdigit() for c in text) else 0.0  # Has numbers
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension


class TFIDFEmbeddingModel(EmbeddingModel):
    """TF-IDF based embedding model (lightweight alternative)."""
    
    def __init__(self, max_features: int = 1000, dimension: int = 384):
        self.max_features = max_features
        self.dimension = dimension
        self.vocabulary = {}
        self.idf_weights = {}
        self.is_fitted = False
    
    def _build_vocabulary(self, texts: List[str]) -> None:
        """Build vocabulary from texts."""
        word_counts = {}
        doc_counts = {}
        
        for text in texts:
            words = text.lower().split()
            unique_words = set(words)
            
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            for word in unique_words:
                doc_counts[word] = doc_counts.get(word, 0) + 1
        
        # Select top words by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        self.vocabulary = {word: i for i, (word, _) in enumerate(sorted_words[:self.max_features])}
        
        # Calculate IDF weights
        total_docs = len(texts)
        for word in self.vocabulary:
            self.idf_weights[word] = np.log(total_docs / (doc_counts.get(word, 1) + 1))
    
    def _text_to_tfidf(self, text: str) -> np.ndarray:
        """Convert text to TF-IDF vector."""
        words = text.lower().split()
        word_counts = {}
        
        for word in words:
            if word in self.vocabulary:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Create TF-IDF vector
        vector = np.zeros(len(self.vocabulary))
        total_words = len(words)
        
        for word, count in word_counts.items():
            if word in self.vocabulary:
                tf = count / total_words
                idf = self.idf_weights[word]
                vector[self.vocabulary[word]] = tf * idf
        
        return vector
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to TF-IDF embeddings."""
        if not self.is_fitted:
            self._build_vocabulary(texts)
            self.is_fitted = True
        
        embeddings = []
        for text in texts:
            tfidf_vector = self._text_to_tfidf(text)
            
            # Pad or truncate to desired dimension
            if len(tfidf_vector) < self.dimension:
                padded = np.zeros(self.dimension)
                padded[:len(tfidf_vector)] = tfidf_vector
                tfidf_vector = padded
            else:
                tfidf_vector = tfidf_vector[:self.dimension]
            
            # Normalize
            norm = np.linalg.norm(tfidf_vector)
            if norm > 0:
                tfidf_vector = tfidf_vector / norm
            
            embeddings.append(tfidf_vector)
        
        return np.array(embeddings)
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension


# =============================================================================
# SIMILARITY CALCULATOR IMPLEMENTATIONS
# =============================================================================

class CosineSimilarityCalculator(SimilarityCalculator):
    """Cosine similarity calculator."""
    
    def calculate(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def calculate_batch(self, 
                       embeddings1: np.ndarray, 
                       embeddings2: np.ndarray) -> np.ndarray:
        """Calculate cosine similarities between batches."""
        # Normalize embeddings
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        # Avoid division by zero
        norm1 = np.where(norm1 == 0, 1, norm1)
        norm2 = np.where(norm2 == 0, 1, norm2)
        
        normalized1 = embeddings1 / norm1
        normalized2 = embeddings2 / norm2
        
        # Calculate cosine similarity matrix
        return np.dot(normalized1, normalized2.T)


class DotProductSimilarityCalculator(SimilarityCalculator):
    """Dot product similarity calculator."""
    
    def calculate(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate dot product similarity."""
        return float(np.dot(embedding1, embedding2))
    
    def calculate_batch(self, 
                       embeddings1: np.ndarray, 
                       embeddings2: np.ndarray) -> np.ndarray:
        """Calculate dot product similarities between batches."""
        return np.dot(embeddings1, embeddings2.T)


# =============================================================================
# DATA SOURCE/SINK IMPLEMENTATIONS
# =============================================================================

class CSVDataSource(DataSource):
    """CSV file data source."""
    
    def __init__(self, encoding: str = 'utf-8'):
        self.encoding = encoding
    
    def load(self, source: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(source, encoding=self.encoding)
            return df.to_dict('records')
        except Exception as e:
            raise ValueError(f"Failed to load CSV from {source}: {e}")


class JSONDataSource(DataSource):
    """JSON file data source."""
    
    def __init__(self, encoding: str = 'utf-8'):
        self.encoding = encoding
    
    def load(self, source: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        try:
            with open(source, 'r', encoding=self.encoding) as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                raise ValueError("JSON must contain list or dict")
                
        except Exception as e:
            raise ValueError(f"Failed to load JSON from {source}: {e}")


class CSVDataSink(DataSink):
    """CSV file data sink."""
    
    def __init__(self, encoding: str = 'utf-8-sig'):
        self.encoding = encoding
    
    def save(self, data: List[Dict[str, Any]], destination: Union[str, Path]) -> None:
        """Save data to CSV file."""
        try:
            df = pd.DataFrame(data)
            df.to_csv(destination, index=False, encoding=self.encoding)
        except Exception as e:
            raise ValueError(f"Failed to save CSV to {destination}: {e}")


class JSONDataSink(DataSink):
    """JSON file data sink."""
    
    def __init__(self, encoding: str = 'utf-8', indent: int = 2):
        self.encoding = encoding
        self.indent = indent
    
    def save(self, data: List[Dict[str, Any]], destination: Union[str, Path]) -> None:
        """Save data to JSON file."""
        try:
            with open(destination, 'w', encoding=self.encoding) as f:
                json.dump(data, f, indent=self.indent, ensure_ascii=False)
        except Exception as e:
            raise ValueError(f"Failed to save JSON to {destination}: {e}")


# =============================================================================
# COMPONENT FACTORY
# =============================================================================

class ComponentFactory:
    """Factory for creating components with different configurations."""
    
    @staticmethod
    def create_embedding_model(model_type: str = "mock", **kwargs) -> EmbeddingModel:
        """Create embedding model with advanced options."""
        # Handle cache-related arguments
        cache_enabled = kwargs.pop('cache_enabled', False)
        cache_dir = kwargs.pop('cache_dir', None)
        
        if model_type == "mock":
            return MockEmbeddingModel(**kwargs)
        elif model_type == "tfidf":
            return TFIDFEmbeddingModel(**kwargs)
        elif model_type == "optimized-tfidf":
            # Import here to avoid import errors if sklearn not available
            from advanced_models import OptimizedTFIDFModel
            if cache_enabled and cache_dir:
                kwargs['cache_dir'] = cache_dir
            return OptimizedTFIDFModel(**kwargs)
        elif model_type == "sentence-bert" or model_type == "sentence-transformer":
            # Import here to avoid import errors if sentence-transformers not available
            from advanced_models import SentenceTransformerModel
            if cache_enabled and cache_dir:
                kwargs['cache_dir'] = cache_dir
            return SentenceTransformerModel(**kwargs)
        else:
            raise ValueError(f"Unknown embedding model type: {model_type}. Available: mock, tfidf, optimized-tfidf, sentence-bert")
    
    @staticmethod
    def create_similarity_calculator(calc_type: str = "cosine") -> SimilarityCalculator:
        """Create similarity calculator."""
        if calc_type == "cosine":
            return CosineSimilarityCalculator()
        elif calc_type == "dot_product":
            return DotProductSimilarityCalculator()
        else:
            raise ValueError(f"Unknown similarity calculator type: {calc_type}")
    
    @staticmethod
    def create_text_processor(processor_type: str = "basic", **kwargs) -> TextProcessor:
        """Create text processor."""
        if processor_type == "basic":
            return BasicTextProcessor(**kwargs)
        elif processor_type == "thai":
            return ThaiTextProcessor(**kwargs)
        else:
            raise ValueError(f"Unknown text processor type: {processor_type}")
    
    @staticmethod
    def create_data_source(source_type: str = "csv", **kwargs) -> DataSource:
        """Create data source."""
        if source_type == "csv":
            return CSVDataSource(**kwargs)
        elif source_type == "json":
            return JSONDataSource(**kwargs)
        else:
            raise ValueError(f"Unknown data source type: {source_type}")
    
    @staticmethod
    def create_data_sink(sink_type: str = "csv", **kwargs) -> DataSink:
        """Create data sink."""
        if sink_type == "csv":
            return CSVDataSink(**kwargs)
        elif sink_type == "json":
            return JSONDataSink(**kwargs)
        else:
            raise ValueError(f"Unknown data sink type: {sink_type}")


if __name__ == "__main__":
    print("🔧 Fresh Architecture - Concrete Implementations Ready")
    print("🧪 Next: Test complete pipeline")
