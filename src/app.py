"""
Main application entry point with extensible architecture.

This module provides the main ProductSimilarityChecker class that orchestrates
all components using the extensible plugin architecture.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import time

from .config import Settings, load_settings
from .core import (
    EmbeddingModel,
    SimilarityCalculator,
    TextPreprocessor,
    create_embedding_model,
    create_similarity_calculator,
    create_default_thai_product_preprocessor
)
from .data import (
    create_default_product_loader,
    create_default_result_writer
)
from .utils import (
    create_default_cache_manager,
    create_default_progress_reporter,
    create_default_product_filter
)


class ProductSimilarityChecker:
    """
    Main application class for product similarity checking.
    
    This class orchestrates all components and provides a clean API
    for product similarity analysis.
    """
    
    def __init__(self, 
                 settings: Optional[Settings] = None,
                 embedding_model: Optional[EmbeddingModel] = None,
                 similarity_calculator: Optional[SimilarityCalculator] = None,
                 text_preprocessor: Optional[TextPreprocessor] = None):
        """
        Initialize ProductSimilarityChecker.
        
        Args:
            settings: Application settings (loaded from config if None)
            embedding_model: Custom embedding model (created from settings if None)
            similarity_calculator: Custom similarity calculator (created from settings if None)
            text_preprocessor: Custom text preprocessor (default Thai product preprocessor if None)
        """
        # Load settings
        self.settings = settings or load_settings()
        
        # Initialize components
        self.embedding_model = embedding_model or self._create_embedding_model()
        self.similarity_calculator = similarity_calculator or self._create_similarity_calculator()
        self.text_preprocessor = text_preprocessor or create_default_thai_product_preprocessor()
        
        # Initialize utilities
        self.data_loader = create_default_product_loader()
        self.data_writer = create_default_result_writer()
        self.result_filter = create_default_product_filter(
            min_similarity=self.settings.similarity.threshold
        )
        
        # Initialize cache and progress reporting
        if self.settings.performance.enable_caching:
            self.cache_manager = create_default_cache_manager(
                self.settings.performance.cache_dir
            )
        else:
            self.cache_manager = None
        
        self.progress_reporter = create_default_progress_reporter(
            verbose=self.settings.logging.level != "CRITICAL"
        )
        
        # Setup logging
        self._setup_logging()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("ProductSimilarityChecker initialized")
        self.logger.info(f"Model: {self.embedding_model.get_model_info()['name']}")
        self.logger.info(f"Similarity: {type(self.similarity_calculator).__name__}")
        self.logger.info(f"Preprocessor: {type(self.text_preprocessor).__name__}")
    
    def _create_embedding_model(self) -> EmbeddingModel:
        """Create embedding model from settings."""
        return create_embedding_model(
            config=self.settings.model,
            enable_caching=self.settings.performance.enable_caching
        )
    
    def _create_similarity_calculator(self) -> SimilarityCalculator:
        """Create similarity calculator from settings."""
        return create_similarity_calculator(
            config=self.settings.similarity,
            calculator_type="cosine"  # Default to cosine similarity
        )
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.settings.logging.level),
            format=self.settings.logging.format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.settings.logging.file) if self.settings.logging.file else logging.NullHandler()
            ]
        )
    
    def load_products(self, source: Union[str, Path]) -> List[str]:
        """
        Load product texts from a data source.
        
        Args:
            source: Path to data file
            
        Returns:
            List of product text strings
        """
        self.logger.info(f"Loading products from {source}")
        
        # Load raw data
        products = self.data_loader.load(source)
        
        # Preprocess texts
        self.progress_reporter.start(len(products), "Preprocessing products")
        
        preprocessed_products = []
        for i, product in enumerate(products):
            preprocessed_product = self.text_preprocessor.preprocess(product)
            if preprocessed_product.strip():  # Only keep non-empty products
                preprocessed_products.append(preprocessed_product)
            
            if i % 100 == 0:  # Update every 100 items
                self.progress_reporter.update(i)
        
        self.progress_reporter.finish(f"Loaded {len(preprocessed_products)} valid products")
        
        self.logger.info(f"Loaded and preprocessed {len(preprocessed_products)} products")
        return preprocessed_products
    
    def compute_embeddings(self, texts: List[str]) -> Any:
        """
        Compute embeddings for texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Tensor containing embeddings
        """
        self.logger.info(f"Computing embeddings for {len(texts)} texts")
        
        # Check cache if enabled
        cache_key = None
        if self.cache_manager:
            cache_key = f"embeddings_{hash(tuple(texts))}"
            cached_embeddings = self.cache_manager.get(cache_key)
            if cached_embeddings is not None:
                self.logger.info("Using cached embeddings")
                return cached_embeddings
        
        # Compute embeddings
        self.progress_reporter.start(len(texts), "Computing embeddings")
        
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.settings.model.batch_size,
            show_progress_bar=False  # We handle progress ourselves
        )
        
        self.progress_reporter.finish("Embeddings computed")
        
        # Cache embeddings if enabled
        if self.cache_manager and cache_key:
            self.cache_manager.put(cache_key, embeddings)
            self.logger.debug("Cached embeddings")
        
        self.logger.info(f"Computed embeddings with shape {embeddings.shape}")
        return embeddings
    
    def find_similar_products(self, 
                            products1: List[str], 
                            products2: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Find similar products between two sets.
        
        Args:
            products1: First set of products
            products2: Second set of products (if None, compare within products1)
            
        Returns:
            List of similarity results
        """
        self.logger.info(f"Finding similar products: {len(products1)} vs {len(products2 or products1)}")
        
        start_time = time.time()
        
        # Compute embeddings
        embeddings1 = self.compute_embeddings(products1)
        
        if products2 is not None:
            embeddings2 = self.compute_embeddings(products2)
        else:
            embeddings2 = embeddings1
            products2 = products1
        
        # Find matches
        self.logger.info("Computing similarity matches")
        matches = self.similarity_calculator.find_matches(embeddings1, embeddings2)
        
        # Convert to result format
        results = []
        for idx1, idx2, score in matches:
            if products2 is products1 and idx1 >= idx2:
                # Skip duplicate pairs when comparing within same set
                continue
            
            result = {
                'text1': products1[idx1],
                'text2': products2[idx2],
                'similarity_score': score,
                'index1': idx1,
                'index2': idx2
            }
            results.append(result)
        
        # Apply filtering
        self.logger.info(f"Applying filters to {len(results)} raw matches")
        filtered_results = self.result_filter.filter(results)
        
        elapsed_time = time.time() - start_time
        
        self.logger.info(f"Found {len(filtered_results)} similar products in {elapsed_time:.2f}s")
        
        return filtered_results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
        """
        Save similarity results to file.
        
        Args:
            results: List of similarity results
            output_path: Path to output file
        """
        self.logger.info(f"Saving {len(results)} results to {output_path}")
        
        # Add metadata to results
        enhanced_results = []
        for result in results:
            enhanced_result = result.copy()
            enhanced_result.update({
                'model_name': self.embedding_model.get_model_info()['name'],
                'similarity_metric': type(self.similarity_calculator).__name__,
                'threshold': self.settings.similarity.threshold,
                'timestamp': time.time()
            })
            enhanced_results.append(enhanced_result)
        
        # Save results
        self.data_writer.write(enhanced_results, output_path)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def run_similarity_check(self, 
                           input_file1: Union[str, Path],
                           input_file2: Optional[Union[str, Path]] = None,
                           output_file: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        """
        Run complete similarity check workflow.
        
        Args:
            input_file1: First input file
            input_file2: Second input file (optional)
            output_file: Output file path (optional)
            
        Returns:
            List of similarity results
        """
        self.logger.info("Starting similarity check workflow")
        
        # Load products
        products1 = self.load_products(input_file1)
        
        if input_file2:
            products2 = self.load_products(input_file2)
        else:
            products2 = None
        
        # Find similar products
        results = self.find_similar_products(products1, products2)
        
        # Save results if output file specified
        if output_file:
            self.save_results(results, output_file)
        
        self.logger.info("Similarity check workflow completed")
        
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the current system configuration.
        
        Returns:
            Dictionary containing system information
        """
        info = {
            'embedding_model': self.embedding_model.get_model_info(),
            'similarity_calculator': self.similarity_calculator.__class__.__name__,
            'text_preprocessor': self.text_preprocessor.get_config(),
            'settings': self.settings.__dict__,
            'cache_enabled': self.cache_manager is not None,
            'cache_stats': self.cache_manager.get_stats() if self.cache_manager else None
        }
        
        return info


def create_default_checker(config_file: Optional[Union[str, Path]] = None) -> ProductSimilarityChecker:
    """
    Create a ProductSimilarityChecker with default settings.
    
    Args:
        config_file: Optional path to configuration file
        
    Returns:
        Configured ProductSimilarityChecker instance
    """
    if config_file:
        settings = load_settings(config_file)
    else:
        settings = load_settings()
    
    return ProductSimilarityChecker(settings=settings)


# Convenience function for backward compatibility
def check_product_similarity(input_file1: Union[str, Path],
                           input_file2: Optional[Union[str, Path]] = None,
                           output_file: Optional[Union[str, Path]] = None,
                           threshold: float = 0.6) -> List[Dict[str, Any]]:
    """
    Convenience function for product similarity checking.
    
    Args:
        input_file1: First input file
        input_file2: Second input file (optional)
        output_file: Output file path (optional)
        threshold: Similarity threshold
        
    Returns:
        List of similarity results
    """
    # Create checker with custom threshold
    settings = load_settings()
    settings.similarity.threshold = threshold
    
    checker = ProductSimilarityChecker(settings=settings)
    
    return checker.run_similarity_check(input_file1, input_file2, output_file)
