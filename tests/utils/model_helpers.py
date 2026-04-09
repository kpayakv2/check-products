"""
Model Testing Utilities - Common model loading and testing functions
===================================================================

Centralized model utilities to reduce duplication
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Any
from ..config import TestConfig


class ModelTestHelper:
    """Helper class for model testing"""
    
    @staticmethod
    def load_sentence_transformer(model_name: Optional[str] = None, cache_dir: Optional[str] = None) -> Tuple[bool, Any, str]:
        """
        Load SentenceTransformer model with error handling
        
        Args:
            model_name: Model name to load (default: multilingual)
            cache_dir: Cache directory (default: from config)
            
        Returns:
            Tuple of (success, model_object, error_message)
        """
        if model_name is None:
            model_name = TestConfig.get_model_name('multilingual')
            
        if cache_dir is None:
            cache_dir = TestConfig.MODELS['cache_dir']
        
        try:
            from sentence_transformers import SentenceTransformer
            
            if cache_dir and Path(cache_dir).exists():
                model = SentenceTransformer(model_name, cache_folder=cache_dir)
            else:
                model = SentenceTransformer(model_name)
            
            return True, model, ""
            
        except ImportError:
            return False, None, "sentence-transformers library not installed"
        except Exception as e:
            return False, None, f"Error loading model: {str(e)}"
    
    @staticmethod
    def check_model_cache(model_name: Optional[str] = None, cache_dir: Optional[str] = None) -> bool:
        """
        Check if model exists in cache
        
        Args:
            model_name: Model name to check
            cache_dir: Cache directory to check
            
        Returns:
            True if model exists in cache
        """
        if model_name is None:
            model_name = TestConfig.get_model_name('multilingual')
            
        if cache_dir is None:
            cache_dir = TestConfig.MODELS['cache_dir']
        
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            return False
        
        # Check for model directory pattern
        model_dirs = list(cache_path.glob(f"*{model_name}*"))
        return len(model_dirs) > 0
    
    @staticmethod
    def test_model_encoding(model, test_texts: Optional[list] = None) -> Tuple[bool, Any, str]:
        """
        Test model encoding with sample texts
        
        Args:
            model: Loaded model object
            test_texts: List of texts to encode (default: sample Thai texts)
            
        Returns:
            Tuple of (success, embeddings, error_message)
        """
        if test_texts is None:
            test_texts = [
                "ทดสอบโมเดล",
                "Test model", 
                "เสื้อยืด Nike"
            ]
        
        try:
            embeddings = model.encode(test_texts, convert_to_numpy=True)
            return True, embeddings, ""
        except Exception as e:
            return False, None, f"Error encoding texts: {str(e)}"
    
    @staticmethod
    def get_model_info(model) -> dict:
        """
        Get model information
        
        Args:
            model: Loaded model object
            
        Returns:
            Dictionary with model information
        """
        try:
            return {
                "dimension": model.get_sentence_embedding_dimension(),
                "max_seq_length": getattr(model, 'max_seq_length', 'unknown'),
                "device": str(model.device) if hasattr(model, 'device') else 'unknown'
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def force_offline_mode():
        """Force offline mode by setting environment variables"""
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    @staticmethod
    def clear_offline_mode():
        """Clear offline mode environment variables"""
        if 'HF_HUB_OFFLINE' in os.environ:
            del os.environ['HF_HUB_OFFLINE']
        if 'TRANSFORMERS_OFFLINE' in os.environ:
            del os.environ['TRANSFORMERS_OFFLINE']