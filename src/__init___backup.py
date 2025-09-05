"""
Product Similarity Checker - Extensible Architecture

This package provides a modular, extensible system for checking product similarity
using embedding models and configurable similarity metrics.

Key Features:
- Plugin-based architecture with abstract interfaces
- Multiple embedding model support (SentenceTransformers, custom models)
- Various similarity calculators (cosine, dot product, euclidean, hybrid)
- Flexible text preprocessing for Thai and multilingual content
- Comprehensive caching and progress reporting
- Multiple data format support (CSV, JSON, Excel)
- Configurable result filtering and post-processing

Quick Start:
    >>> from src import ProductSimilarityChecker
    >>> checker = ProductSimilarityChecker()
    >>> results = checker.run_similarity_check('products1.csv', 'products2.csv', 'results.csv')

For more advanced usage, see the documentation and examples.
"""

from .app import (
    ProductSimilarityChecker,
    create_default_checker,
    check_product_similarity
)

from .config import (
    Settings,
    load_settings,
    ModelConfig,
    SimilarityConfig,
    DataConfig,
    OutputConfig,
    PerformanceConfig,
    LoggingConfig
)

# Core interfaces for extending functionality
from .core import (
    EmbeddingModel,
    SimilarityCalculator,
    TextPreprocessor,
    create_embedding_model,
    create_similarity_calculator,
    create_text_preprocessor,
    create_default_thai_product_preprocessor
)

# Data processing
from .data import (
    create_data_loader,
    create_data_writer,
    create_default_product_loader,
    create_default_result_writer
)

# Utilities
from .utils import (
    create_cache_manager,
    create_progress_reporter,
    create_result_filter,
    create_default_cache_manager,
    create_default_progress_reporter,
    create_default_product_filter
)

__version__ = "2.0.0"
__author__ = "Product Checker Team"
__description__ = "Extensible product similarity checker with multilingual support"

__all__ = [
    # Main application
    'ProductSimilarityChecker',
    'create_default_checker',
    'check_product_similarity',
    
    # Configuration
    'Settings',
    'load_settings',
    'ModelConfig',
    'SimilarityConfig',
    'DataConfig',
    'OutputConfig',
    'PerformanceConfig',
    'LoggingConfig',
    
    # Core interfaces
    'EmbeddingModel',
    'SimilarityCalculator',
    'TextPreprocessor',
    'create_embedding_model',
    'create_similarity_calculator',
    'create_text_preprocessor',
    'create_default_thai_product_preprocessor',
    
    # Data processing
    'create_data_loader',
    'create_data_writer',
    'create_default_product_loader',
    'create_default_result_writer',
    
    # Utilities
    'create_cache_manager',
    'create_progress_reporter',
    'create_result_filter',
    'create_default_cache_manager',
    'create_default_progress_reporter',
    'create_default_product_filter',
    
    # Package info
    '__version__',
    '__author__',
    '__description__'
]


def get_version() -> str:
    """Get package version."""
    return __version__


def get_system_info() -> dict:
    """Get system and package information."""
    import sys
    import platform
    
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = "Not installed"
    
    try:
        import sentence_transformers
        st_version = sentence_transformers.__version__
    except ImportError:
        st_version = "Not installed"
    
    return {
        'package_version': __version__,
        'python_version': sys.version,
        'platform': platform.platform(),
        'torch_version': torch_version,
        'sentence_transformers_version': st_version
    }
