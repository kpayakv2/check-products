"""
Text preprocessing implementations (simplified for testing).
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional, Set

from .interfaces_simple import TextPreprocessor


class BasicTextPreprocessor(TextPreprocessor):
    """Basic text preprocessing with common operations."""
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 normalize_unicode: bool = True,
                 remove_extra_whitespace: bool = True,
                 remove_numbers: bool = False,
                 min_length: int = 1):
        """
        Initialize the basic text preprocessor.
        
        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation marks
            normalize_unicode: Normalize Unicode characters
            remove_extra_whitespace: Remove extra whitespace
            remove_numbers: Remove numeric characters
            min_length: Minimum length for processed text
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.normalize_unicode = normalize_unicode
        self.remove_extra_whitespace = remove_extra_whitespace
        self.remove_numbers = remove_numbers
        self.min_length = min_length
        
        # Compile regex patterns for better performance
        self.punctuation_pattern = re.compile(r'[^\w\s]', re.UNICODE)
        self.number_pattern = re.compile(r'\d+')
        self.whitespace_pattern = re.compile(r'\s+')
    
    def clean_text(self, text: str) -> str:
        """Clean text using configured operations."""
        if not text or not isinstance(text, str):
            return ""
        
        result = text
        
        # Normalize Unicode
        if self.normalize_unicode:
            result = unicodedata.normalize('NFKD', result)
        
        # Convert to lowercase
        if self.lowercase:
            result = result.lower()
        
        # Remove numbers
        if self.remove_numbers:
            result = self.number_pattern.sub(' ', result)
        
        # Remove punctuation
        if self.remove_punctuation:
            result = self.punctuation_pattern.sub(' ', result)
        
        # Remove extra whitespace
        if self.remove_extra_whitespace:
            result = self.whitespace_pattern.sub(' ', result)
        
        # Final cleanup
        result = result.strip()
        
        # Check minimum length
        if len(result) < self.min_length:
            return ""
        
        return result
    
    def preprocess(self, text: str) -> str:
        """Preprocess a single text."""
        return self.clean_text(text)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts."""
        return [self.preprocess(text) for text in texts]


class ThaiTextPreprocessor(BasicTextPreprocessor):
    """Thai-specific text preprocessing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Thai-specific patterns
        self.thai_pattern = re.compile(r'[\u0e00-\u0e7f]+')
    
    def is_thai_text(self, text: str) -> bool:
        """Check if text contains Thai characters."""
        return bool(self.thai_pattern.search(text))
    
    def preprocess(self, text: str) -> str:
        """Preprocess with Thai-specific handling."""
        result = super().preprocess(text)
        
        if self.is_thai_text(text):
            # Additional Thai-specific processing could go here
            pass
        
        return result


class ConfigurablePreprocessor(BasicTextPreprocessor):
    """Preprocessor with runtime configuration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration dictionary."""
        super().__init__(
            lowercase=config.get('lowercase', True),
            remove_punctuation=config.get('remove_punctuation', True),
            normalize_unicode=config.get('normalize_unicode', True),
            remove_extra_whitespace=config.get('remove_extra_whitespace', True),
            remove_numbers=config.get('remove_numbers', False),
            min_length=config.get('min_length', 1)
        )
        
        # Additional configuration
        self.custom_replacements = config.get('custom_replacements', {})
        self.stop_words = set(config.get('stop_words', []))
    
    def preprocess(self, text: str) -> str:
        """Preprocess with custom configuration."""
        result = super().preprocess(text)
        
        # Apply custom replacements
        for old, new in self.custom_replacements.items():
            result = result.replace(old, new)
        
        # Remove stop words
        if self.stop_words:
            words = result.split()
            words = [word for word in words if word not in self.stop_words]
            result = ' '.join(words)
        
        return result
