"""
Text preprocessing implementations.

This module provides implementations of the TextPreprocessor interface
for different text preprocessing strategies.
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional, Set
from abc import ABC, abstractmethod

from .interfaces import TextPreprocessor


class BasicTextPreprocessor(TextPreprocessor):
    """Basic text preprocessing with common operations."""
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_extra_spaces: bool = True,
                 remove_special_chars: bool = False,
                 normalize_unicode: bool = True):
        """
        Initialize basic text preprocessor.
        
        Args:
            lowercase: Convert text to lowercase
            remove_extra_spaces: Remove extra whitespace
            remove_special_chars: Remove special characters
            normalize_unicode: Normalize unicode characters
        """
        self.lowercase = lowercase
        self.remove_extra_spaces = remove_extra_spaces
        self.remove_special_chars = remove_special_chars
        self.normalize_unicode = normalize_unicode
    
    def preprocess(self, text: str) -> str:
        """Apply basic preprocessing steps."""
        if not isinstance(text, str):
            text = str(text)
        
        # Normalize unicode
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove special characters (keep Thai, English, numbers, and basic punctuation)
        if self.remove_special_chars:
            text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s\.,!?()-]', '', text)
        
        # Remove extra spaces
        if self.remove_extra_spaces:
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Apply preprocessing to a batch of texts."""
        return [self.preprocess(text) for text in texts]
    
    def get_config(self) -> Dict[str, Any]:
        """Get preprocessor configuration."""
        return {
            "type": "BasicTextPreprocessor",
            "lowercase": self.lowercase,
            "remove_extra_spaces": self.remove_extra_spaces,
            "remove_special_chars": self.remove_special_chars,
            "normalize_unicode": self.normalize_unicode
        }


class ThaiTextPreprocessor(TextPreprocessor):
    """Specialized preprocessor for Thai text."""
    
    def __init__(self,
                 normalize_thai_chars: bool = True,
                 remove_tone_marks: bool = False,
                 standardize_spaces: bool = True,
                 normalize_numbers: bool = True):
        """
        Initialize Thai text preprocessor.
        
        Args:
            normalize_thai_chars: Normalize Thai character variations
            remove_tone_marks: Remove Thai tone marks
            standardize_spaces: Standardize whitespace
            normalize_numbers: Normalize number representations
        """
        self.normalize_thai_chars = normalize_thai_chars
        self.remove_tone_marks = remove_tone_marks
        self.standardize_spaces = standardize_spaces
        self.normalize_numbers = normalize_numbers
        
        # Thai tone marks (optional removal)
        self.tone_marks = set(['\u0E48', '\u0E49', '\u0E4A', '\u0E4B'])
        
        # Thai number mappings
        self.thai_to_arabic = str.maketrans(
            '๐๑๒๓๔๕๖๗๘๙',
            '0123456789'
        )
    
    def preprocess(self, text: str) -> str:
        """Apply Thai-specific preprocessing."""
        if not isinstance(text, str):
            text = str(text)
        
        # Normalize unicode (important for Thai)
        text = unicodedata.normalize('NFKC', text)
        
        # Normalize Thai characters
        if self.normalize_thai_chars:
            # Replace common variations
            text = text.replace('เ็', 'เ')  # Fix common ordering issues
            text = text.replace('แ็', 'แ')
        
        # Remove tone marks if specified
        if self.remove_tone_marks:
            text = ''.join(char for char in text if char not in self.tone_marks)
        
        # Normalize numbers
        if self.normalize_numbers:
            text = text.translate(self.thai_to_arabic)
        
        # Standardize spaces
        if self.standardize_spaces:
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Apply Thai preprocessing to a batch of texts."""
        return [self.preprocess(text) for text in texts]
    
    def get_config(self) -> Dict[str, Any]:
        """Get preprocessor configuration."""
        return {
            "type": "ThaiTextPreprocessor",
            "normalize_thai_chars": self.normalize_thai_chars,
            "remove_tone_marks": self.remove_tone_marks,
            "standardize_spaces": self.standardize_spaces,
            "normalize_numbers": self.normalize_numbers
        }


class ProductTextPreprocessor(TextPreprocessor):
    """Specialized preprocessor for product descriptions."""
    
    def __init__(self,
                 remove_brand_prefixes: bool = True,
                 normalize_units: bool = True,
                 remove_promotional_text: bool = True,
                 standardize_colors: bool = True,
                 custom_stopwords: Optional[Set[str]] = None):
        """
        Initialize product text preprocessor.
        
        Args:
            remove_brand_prefixes: Remove common brand prefixes
            normalize_units: Normalize measurement units
            remove_promotional_text: Remove promotional phrases
            standardize_colors: Standardize color names
            custom_stopwords: Additional words to remove
        """
        self.remove_brand_prefixes = remove_brand_prefixes
        self.normalize_units = normalize_units
        self.remove_promotional_text = remove_promotional_text
        self.standardize_colors = standardize_colors
        self.custom_stopwords = custom_stopwords or set()
        
        # Common brand prefixes to remove
        self.brand_prefixes = {
            'แบรนด์', 'ยี่ห้อ', 'brand', 'original', 'authentic',
            'genuine', 'official', 'imported'
        }
        
        # Unit normalizations
        self.unit_mappings = {
            'กก.': 'กิโลกรัม',
            'ก.': 'กรัม',
            'มล.': 'มิลลิลิตร',
            'ลิตร': 'ลิตร',
            'ซม.': 'เซนติเมตร',
            'ม.': 'เมตร',
            'kg': 'กิโลกรัม',
            'g': 'กรัม',
            'ml': 'มิลลิลิตร',
            'l': 'ลิตร',
            'cm': 'เซนติเมตร',
            'm': 'เมตร'
        }
        
        # Promotional phrases to remove
        self.promotional_phrases = {
            'ราคาพิเศษ', 'โปรโมชั่น', 'ลดราคา', 'พิเศษ', 'ส่งฟรี',
            'ของแถม', 'แถมฟรี', 'ซื้อ1แถม1', 'special', 'promotion',
            'sale', 'discount', 'free shipping', 'limited time'
        }
        
        # Color standardizations
        self.color_mappings = {
            'แดง': 'สีแดง',
            'เขียว': 'สีเขียว',
            'ฟ้า': 'สีฟ้า',
            'เหลือง': 'สีเหลือง',
            'ม่วง': 'สีม่วง',
            'ส้ม': 'สีส้ม',
            'ชมพู': 'สีชมพู',
            'ดำ': 'สีดำ',
            'ขาว': 'สีขาว',
            'เทา': 'สีเทา',
            'red': 'สีแดง',
            'green': 'สีเขียว',
            'blue': 'สีฟ้า',
            'yellow': 'สีเหลือง',
            'purple': 'สีม่วง',
            'orange': 'สีส้ม',
            'pink': 'สีชมพู',
            'black': 'สีดำ',
            'white': 'สีขาว',
            'gray': 'สีเทา',
            'grey': 'สีเทา'
        }
    
    def preprocess(self, text: str) -> str:
        """Apply product-specific preprocessing."""
        if not isinstance(text, str):
            text = str(text)
        
        # Basic normalization
        text = unicodedata.normalize('NFKC', text.lower())
        
        # Remove brand prefixes
        if self.remove_brand_prefixes:
            for prefix in self.brand_prefixes:
                pattern = rf'\b{re.escape(prefix)}\s*'
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Normalize units
        if self.normalize_units:
            for old_unit, new_unit in self.unit_mappings.items():
                pattern = rf'\b{re.escape(old_unit)}\b'
                text = re.sub(pattern, new_unit, text, flags=re.IGNORECASE)
        
        # Remove promotional text
        if self.remove_promotional_text:
            for phrase in self.promotional_phrases:
                pattern = rf'\b{re.escape(phrase)}\b'
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Standardize colors
        if self.standardize_colors:
            for old_color, new_color in self.color_mappings.items():
                pattern = rf'\b{re.escape(old_color)}\b'
                text = re.sub(pattern, new_color, text, flags=re.IGNORECASE)
        
        # Remove custom stopwords
        if self.custom_stopwords:
            words = text.split()
            words = [word for word in words if word not in self.custom_stopwords]
            text = ' '.join(words)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Apply product preprocessing to a batch of texts."""
        return [self.preprocess(text) for text in texts]
    
    def get_config(self) -> Dict[str, Any]:
        """Get preprocessor configuration."""
        return {
            "type": "ProductTextPreprocessor",
            "remove_brand_prefixes": self.remove_brand_prefixes,
            "normalize_units": self.normalize_units,
            "remove_promotional_text": self.remove_promotional_text,
            "standardize_colors": self.standardize_colors,
            "custom_stopwords_count": len(self.custom_stopwords)
        }


class ChainedTextPreprocessor(TextPreprocessor):
    """Preprocessor that chains multiple preprocessing steps."""
    
    def __init__(self, preprocessors: List[TextPreprocessor]):
        """
        Initialize chained preprocessor.
        
        Args:
            preprocessors: List of preprocessors to apply in order
        """
        self.preprocessors = preprocessors
    
    def preprocess(self, text: str) -> str:
        """Apply all preprocessors in sequence."""
        for preprocessor in self.preprocessors:
            text = preprocessor.preprocess(text)
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Apply chained preprocessing to a batch of texts."""
        return [self.preprocess(text) for text in texts]
    
    def get_config(self) -> Dict[str, Any]:
        """Get chained preprocessor configuration."""
        return {
            "type": "ChainedTextPreprocessor",
            "preprocessors": [pp.get_config() for pp in self.preprocessors]
        }


def create_text_preprocessor(preprocessor_type: str, **kwargs) -> TextPreprocessor:
    """
    Factory function to create text preprocessors.
    
    Args:
        preprocessor_type: Type of preprocessor ('basic', 'thai', 'product', 'chained')
        **kwargs: Additional arguments for the preprocessor
        
    Returns:
        Configured text preprocessor instance
    """
    if preprocessor_type.lower() == "basic":
        return BasicTextPreprocessor(**kwargs)
    elif preprocessor_type.lower() == "thai":
        return ThaiTextPreprocessor(**kwargs)
    elif preprocessor_type.lower() == "product":
        return ProductTextPreprocessor(**kwargs)
    elif preprocessor_type.lower() == "chained":
        # For chained preprocessor, expect 'preprocessors' in kwargs
        return ChainedTextPreprocessor(**kwargs)
    else:
        raise ValueError(f"Unknown preprocessor type: {preprocessor_type}")


def create_default_thai_product_preprocessor() -> TextPreprocessor:
    """
    Create a default preprocessor optimized for Thai product descriptions.
    
    Returns:
        Chained preprocessor with Thai and product-specific preprocessing
    """
    thai_preprocessor = ThaiTextPreprocessor(
        normalize_thai_chars=True,
        remove_tone_marks=False,  # Keep tone marks for better semantic understanding
        standardize_spaces=True,
        normalize_numbers=True
    )
    
    product_preprocessor = ProductTextPreprocessor(
        remove_brand_prefixes=True,
        normalize_units=True,
        remove_promotional_text=True,
        standardize_colors=True
    )
    
    basic_preprocessor = BasicTextPreprocessor(
        lowercase=True,
        remove_extra_spaces=True,
        remove_special_chars=False,  # Keep for Thai compatibility
        normalize_unicode=True
    )
    
    return ChainedTextPreprocessor([
        basic_preprocessor,
        thai_preprocessor,
        product_preprocessor
    ])
