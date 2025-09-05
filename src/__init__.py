"""
Product Similarity Checker - Minimal exports for testing

This version only exports configuration to avoid circular imports.
"""

from .config import (
    Settings,
    load_settings
)
__version__ = "2.0.0"
__author__ = "Product Checker Team"

__all__ = [
    # Configuration only
    'Settings',
    'load_settings',
    
    # Package info
    '__version__',
    '__author__'
]
