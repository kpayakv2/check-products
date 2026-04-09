"""
Fresh Extensible Architecture - Concrete Implementations
=======================================================

Lightweight implementations for immediate testing
"""

import re
import unicodedata
import json
import csv
import hashlib
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
        
        # 1. Protect decimal numbers before punctuation removal
        # (Replace dots in numbers with a temporary placeholder)
        result = re.sub(r'(\d)\.(\d)', r'\1_DOT_\2', result)
        
        # 2. Remove punctuation
        if self.remove_punctuation:
            result = self.punct_pattern.sub(' ', result)
        
        # 3. Restore decimal dots
        result = result.replace('_DOT_', '.')
        
        # 4. Normalize whitespace
        if self.normalize_whitespace:
            result = self.whitespace_pattern.sub(' ', result)
        
        return result.strip()
    
    def process_batch(self, texts: List[str]) -> List[str]:
        """Process multiple texts."""
        return [self.process(text) for text in texts]


class ThaiTextProcessor(BasicTextProcessor):
    """Thai-specific text processing with advanced cleaning."""
    
    def __init__(self, 
                 normalize_thai_chars: bool = True,
                 remove_tone_marks: bool = False,
                 standardize_spaces: bool = True,
                 normalize_numbers: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.normalize_thai_chars = normalize_thai_chars
        self.remove_tone_marks = remove_tone_marks
        self.standardize_spaces = standardize_spaces
        self.normalize_numbers = normalize_numbers
        
        # Thai character range
        self.thai_pattern = re.compile(r'[\u0e00-\u0e7f]+')
        
        # Thai number mapping
        self.thai_to_arabic = str.maketrans('๐๑๒๓๔๕๖๗๘๙', '0123456789')
        
        # Tone marks and floating vowels
        self.tone_marks = ['\u0E48', '\u0E49', '\u0E4A', '\u0E4B', '\u0E4C', '\u0E4D', '\u0E4E']
    
    def process(self, text: str) -> str:
        """Process with comprehensive Thai handling."""
        if not isinstance(text, str):
            return str(text)
            
        # 1. Unicode Normalization (NFKC is better for compatibility)
        result = unicodedata.normalize('NFKC', text)
        
        # 2. Lowercase
        if self.lowercase:
            result = result.lower()
            
        # 3. Thai Character Normalization (Fix ordering)
        if self.normalize_thai_chars:
            result = result.replace('เ็', 'เ')
            result = result.replace('แ็', 'แ')
            # Fix floating vowels without base (simplified)
            result = re.sub(r'[\u0E31\u0E34-\u0E37\u0E47-\u0E4E]{2,}', lambda m: m.group(0)[0], result)
            
        # 4. Thai Number Conversion (DO THIS FIRST)
        if self.normalize_numbers:
            result = result.translate(self.thai_to_arabic)
            
        # 5. Tone Mark Removal (Optional)
        if self.remove_tone_marks:
            for mark in self.tone_marks:
                result = result.replace(mark, '')
                
        # 6. Protect decimal numbers before punctuation removal
        result = re.sub(r'(\d)\.(\d)', r'\1_DOT_\2', result)
                
        # 7. Remove Punctuation
        if self.remove_punctuation:
            result = self.punct_pattern.sub(' ', result)
            
        # 8. Restore decimal dots
        result = result.replace('_DOT_', '.')
            
        # 9. Aggressive Whitespace Normalization (New logic added)
        if self.normalize_whitespace or self.standardize_spaces:
            # Replace various Unicode spaces with standard space
            result = re.sub(r'[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000]', ' ', result)
            # Remove multiple spaces into one
            result = self.whitespace_pattern.sub(' ', result)
            
        return result.strip()


# =============================================================================
# EMBEDDING MODEL IMPLEMENTATIONS
# =============================================================================

class MockEmbeddingModel(EmbeddingModel):
    """Mock embedding model for testing (no dependencies)."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate mock embeddings based on text length and content."""
        embeddings = []
        
        for text in texts:
            processed_text = text.lower().strip()
            digest = hashlib.sha1(processed_text.encode('utf-8')).digest()
            seed = int.from_bytes(digest[:8], 'big', signed=False)
            rng = np.random.default_rng(seed)

            # Base embedding from random
            embedding = rng.normal(0, 1, self.dimension)

            # Add text-specific features
            embedding[0] = len(processed_text) / 100.0  # Length feature
            embedding[1] = len(processed_text.split()) / 50.0  # Word count feature
            embedding[2] = 1.0 if any(c.isdigit() for c in processed_text) else 0.0  # Has numbers

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
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

        self.is_fitted = True
    
    def prepare_corpus(self, texts: List[str]) -> None:
        """Prepare TF-IDF vocabulary for a corpus of texts."""
        if not texts:
            self.vocabulary = {}
            self.idf_weights = {}
            self.is_fitted = False
            return

        self._build_vocabulary(texts)

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
            self.prepare_corpus(texts)
        
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
        # Input validation
        if embeddings1.size == 0 or embeddings2.size == 0:
            return np.array([])
        
        # Ensure 2D arrays
        if len(embeddings1.shape) == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if len(embeddings2.shape) == 1:
            embeddings2 = embeddings2.reshape(1, -1)
        
        # Dimension check
        if embeddings1.shape[1] != embeddings2.shape[1]:
            raise ValueError(f"Dimension mismatch: {embeddings1.shape[1]} vs {embeddings2.shape[1]}")
        
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
        # Input validation
        if embeddings1.size == 0 or embeddings2.size == 0:
            return np.array([])
        
        # Ensure 2D arrays
        if len(embeddings1.shape) == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if len(embeddings2.shape) == 1:
            embeddings2 = embeddings2.reshape(1, -1)
        
        # Dimension check
        if embeddings1.shape[1] != embeddings2.shape[1]:
            raise ValueError(f"Dimension mismatch: {embeddings1.shape[1]} vs {embeddings2.shape[1]}")
        
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
