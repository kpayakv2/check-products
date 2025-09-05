"""
Simplified configuration management for testing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
import os
import json


@dataclass
class ModelConfig:
    """Configuration for embedding models."""
    name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384
    cache_dir: Optional[str] = None
    device: str = "auto"
    batch_size: int = 32


@dataclass
class SimilarityConfig:
    """Configuration for similarity calculation."""
    algorithm: str = "cosine"
    top_k: int = 100
    threshold: float = 0.6


@dataclass
class DataConfig:
    """Configuration for data processing."""
    old_products_column: str = "product_name"
    new_products_column: str = "product_name"
    encoding: str = "utf-8-sig"


@dataclass
class OutputConfig:
    """Configuration for output generation."""
    format: str = "csv"
    include_vectors: bool = True
    include_metadata: bool = True
    encoding: str = "utf-8-sig"


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    enable_caching: bool = True
    cache_dir: Optional[str] = None
    parallel_processing: bool = True
    max_workers: Optional[int] = None


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    console: bool = True


@dataclass
class Settings:
    """Main configuration class that combines all config sections."""
    model: ModelConfig = field(default_factory=ModelConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_settings(config_file: Optional[Path] = None) -> Settings:
    """
    Load configuration with simple defaults.
    
    Args:
        config_file: Path to configuration file (ignored for now)
        
    Returns:
        Configured Settings instance
    """
    # For now, just return default settings
    return Settings()
