"""
Fresh Extensible Architecture - Phase 2
========================================

Core Interfaces for Product Similarity System
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np


# =============================================================================
# CORE ABSTRACTIONS
# =============================================================================

class EmbeddingModel(ABC):
    """Abstract interface for text embedding models."""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass


class SimilarityCalculator(ABC):
    """Abstract interface for similarity calculation."""
    
    @abstractmethod
    def calculate(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate similarity between two embeddings."""
        pass
    
    @abstractmethod
    def calculate_batch(self, 
                       embeddings1: np.ndarray, 
                       embeddings2: np.ndarray) -> np.ndarray:
        """Calculate similarities between batches."""
        pass


class TextProcessor(ABC):
    """Abstract interface for text preprocessing."""
    
    @abstractmethod
    def process(self, text: str) -> str:
        """Process a single text."""
        pass
    
    @abstractmethod
    def process_batch(self, texts: List[str]) -> List[str]:
        """Process multiple texts."""
        pass


class DataSource(ABC):
    """Abstract interface for data loading."""
    
    @abstractmethod
    def load(self, source: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load data from source."""
        pass


class DataSink(ABC):
    """Abstract interface for data writing."""
    
    @abstractmethod  
    def save(self, data: List[Dict[str, Any]], destination: Union[str, Path]) -> None:
        """Save data to destination."""
        pass


# =============================================================================
# CONFIGURATION SYSTEM
# =============================================================================

class Config:
    """Central configuration manager."""
    
    def __init__(self):
        self.model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        self.similarity_threshold = 0.6
        self.batch_size = 32
        self.top_k = 3
        self.cache_enabled = True
        self.output_format = "csv"
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


# =============================================================================
# CORE BUSINESS LOGIC
# =============================================================================

class ProductMatcher:
    """Core product matching engine with pluggable components."""
    
    def __init__(self,
                 embedding_model: EmbeddingModel,
                 similarity_calculator: SimilarityCalculator,
                 text_processor: Optional[TextProcessor] = None,
                 config: Optional[Config] = None):
        """
        Initialize product matcher.
        
        Args:
            embedding_model: Model for generating embeddings
            similarity_calculator: Calculator for similarity scores
            text_processor: Optional text preprocessor
            config: Configuration object
        """
        self.embedding_model = embedding_model
        self.similarity_calculator = similarity_calculator
        self.text_processor = text_processor
        self.config = config or Config()
        
        # Cache for embeddings
        self._embedding_cache = {}
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text if processor is available."""
        if self.text_processor:
            return self.text_processor.process(text)
        return text.strip()
    
    def get_embeddings(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """Get embeddings for texts with optional caching."""
        if not use_cache or not self.config.cache_enabled:
            processed_texts = [self.preprocess_text(text) for text in texts]
            return self.embedding_model.encode(processed_texts)
        
        # Cache-enabled path
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            processed_text = self.preprocess_text(text)
            if processed_text in self._embedding_cache:
                embeddings.append(self._embedding_cache[processed_text])
            else:
                embeddings.append(None)
                uncached_texts.append(processed_text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = self.embedding_model.encode(uncached_texts)
            for idx, embedding in zip(uncached_indices, new_embeddings):
                processed_text = self.preprocess_text(texts[idx])
                self._embedding_cache[processed_text] = embedding
                embeddings[idx] = embedding
        
        return np.array(embeddings)
    
    def find_matches(self, 
                    query_products: List[str],
                    reference_products: List[str]) -> List[Dict[str, Any]]:
        """
        Find matches between query and reference products.
        
        Args:
            query_products: Products to find matches for
            reference_products: Reference product catalog
            
        Returns:
            List of match results with scores
        """
        # Get embeddings
        print(f"🔄 Processing {len(query_products)} query products...")
        query_embeddings = self.get_embeddings(query_products)
        
        print(f"🔄 Processing {len(reference_products)} reference products...")
        reference_embeddings = self.get_embeddings(reference_products)
        
        # Calculate similarities
        print(f"🔄 Calculating similarities...")
        similarity_matrix = self.similarity_calculator.calculate_batch(
            query_embeddings, reference_embeddings
        )
        
        # Extract top matches
        results = []
        for i, query_product in enumerate(query_products):
            scores = similarity_matrix[i]
            
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:self.config.top_k]
            
            for rank, ref_idx in enumerate(top_indices):
                score = scores[ref_idx]
                
                if score >= self.config.similarity_threshold:
                    results.append({
                        'query_product': query_product,
                        'matched_product': reference_products[ref_idx],
                        'similarity_score': float(score),
                        'rank': rank + 1
                    })
        
        return results


# =============================================================================
# PIPELINE ORCHESTRATOR  
# =============================================================================

class ProductSimilarityPipeline:
    """High-level pipeline orchestrator."""
    
    def __init__(self,
                 data_source: DataSource,
                 data_sink: DataSink,
                 product_matcher: ProductMatcher):
        """Initialize pipeline with components."""
        self.data_source = data_source
        self.data_sink = data_sink
        self.product_matcher = product_matcher
    
    def run(self,
            query_data_source: Union[str, Path],
            reference_data_source: Union[str, Path], 
            output_destination: Union[str, Path],
            query_column: str = "รายการ",
            reference_column: str = "name") -> None:
        """
        Run the complete similarity pipeline.
        
        Args:
            query_data_source: Source of query products
            reference_data_source: Source of reference products  
            output_destination: Where to save results
            query_column: Column name for query products
            reference_column: Column name for reference products
        """
        # Load data
        print(f"📁 Loading query data from {query_data_source}")
        query_data = self.data_source.load(query_data_source)
        query_products = [item[query_column] for item in query_data]
        
        print(f"📁 Loading reference data from {reference_data_source}")
        reference_data = self.data_source.load(reference_data_source)
        reference_products = [item[reference_column] for item in reference_data]
        
        # Find matches
        print(f"🔍 Finding matches...")
        matches = self.product_matcher.find_matches(query_products, reference_products)
        
        # Save results
        print(f"💾 Saving {len(matches)} matches to {output_destination}")
        self.data_sink.save(matches, output_destination)
        
        print(f"✅ Pipeline complete! Found {len(matches)} matches above threshold.")
        
        # Return matches for further processing
        return matches


if __name__ == "__main__":
    print("🎯 Fresh Extensible Architecture - Core Interfaces Defined")
    print("📋 Next: Implement concrete classes for each interface")
