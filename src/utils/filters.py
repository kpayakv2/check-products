"""
Result filtering implementations.

This module provides implementations of the ResultFilter interface
for different result filtering and post-processing strategies.
"""

from typing import List, Dict, Any, Callable, Optional, Set, Tuple
import re

from ..core.interfaces import ResultFilter


class ThresholdResultFilter(ResultFilter):
    """Filter results based on similarity threshold."""
    
    def __init__(self, threshold: float):
        """
        Initialize threshold filter.
        
        Args:
            threshold: Minimum similarity score to keep
        """
        self.threshold = threshold
    
    def filter(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results above threshold."""
        return [
            result for result in results
            if result.get('similarity_score', 0) >= self.threshold
        ]
    
    def get_config(self) -> Dict[str, Any]:
        """Get filter configuration."""
        return {
            "type": "ThresholdResultFilter",
            "threshold": self.threshold
        }


class TopKResultFilter(ResultFilter):
    """Filter to keep only top-k results."""
    
    def __init__(self, k: int):
        """
        Initialize top-k filter.
        
        Args:
            k: Number of top results to keep
        """
        self.k = k
    
    def filter(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep only top-k results."""
        # Sort by similarity score (descending)
        sorted_results = sorted(
            results,
            key=lambda x: x.get('similarity_score', 0),
            reverse=True
        )
        
        return sorted_results[:self.k]
    
    def get_config(self) -> Dict[str, Any]:
        """Get filter configuration."""
        return {
            "type": "TopKResultFilter",
            "k": self.k
        }


class DuplicateResultFilter(ResultFilter):
    """Filter to remove duplicate results."""
    
    def __init__(self, 
                 key_fields: List[str] = ['text1', 'text2'],
                 keep_first: bool = True):
        """
        Initialize duplicate filter.
        
        Args:
            key_fields: Fields to use for duplicate detection
            keep_first: Whether to keep first occurrence (else keep last)
        """
        self.key_fields = key_fields
        self.keep_first = keep_first
    
    def filter(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results."""
        seen_keys: Set[Tuple] = set()
        filtered_results = []
        
        if not self.keep_first:
            results = reversed(results)
        
        for result in results:
            # Create key tuple from specified fields
            key = tuple(
                result.get(field, '') for field in self.key_fields
            )
            
            if key not in seen_keys:
                seen_keys.add(key)
                filtered_results.append(result)
        
        if not self.keep_first:
            filtered_results = list(reversed(filtered_results))
        
        return filtered_results
    
    def get_config(self) -> Dict[str, Any]:
        """Get filter configuration."""
        return {
            "type": "DuplicateResultFilter",
            "key_fields": self.key_fields,
            "keep_first": self.keep_first
        }


class TextLengthResultFilter(ResultFilter):
    """Filter results based on text length."""
    
    def __init__(self, 
                 min_length: Optional[int] = None,
                 max_length: Optional[int] = None,
                 text_fields: List[str] = ['text1', 'text2']):
        """
        Initialize text length filter.
        
        Args:
            min_length: Minimum text length to keep
            max_length: Maximum text length to keep
            text_fields: Text fields to check
        """
        self.min_length = min_length
        self.max_length = max_length
        self.text_fields = text_fields
    
    def filter(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results by text length."""
        filtered_results = []
        
        for result in results:
            keep_result = True
            
            for field in self.text_fields:
                text = result.get(field, '')
                text_len = len(str(text))
                
                if self.min_length is not None and text_len < self.min_length:
                    keep_result = False
                    break
                
                if self.max_length is not None and text_len > self.max_length:
                    keep_result = False
                    break
            
            if keep_result:
                filtered_results.append(result)
        
        return filtered_results
    
    def get_config(self) -> Dict[str, Any]:
        """Get filter configuration."""
        return {
            "type": "TextLengthResultFilter",
            "min_length": self.min_length,
            "max_length": self.max_length,
            "text_fields": self.text_fields
        }


class RegexResultFilter(ResultFilter):
    """Filter results based on regular expression patterns."""
    
    def __init__(self, 
                 patterns: Dict[str, str],
                 match_mode: str = 'include'):
        """
        Initialize regex filter.
        
        Args:
            patterns: Dictionary of field -> regex pattern
            match_mode: 'include' to keep matches, 'exclude' to remove matches
        """
        self.patterns = patterns
        self.match_mode = match_mode
        
        # Compile patterns
        self.compiled_patterns = {
            field: re.compile(pattern, re.IGNORECASE)
            for field, pattern in patterns.items()
        }
    
    def filter(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results by regex patterns."""
        filtered_results = []
        
        for result in results:
            matches = []
            
            for field, pattern in self.compiled_patterns.items():
                text = str(result.get(field, ''))
                match = pattern.search(text) is not None
                matches.append(match)
            
            # Determine if result should be kept
            if self.match_mode == 'include':
                # Keep if any pattern matches
                keep_result = any(matches)
            else:  # exclude
                # Keep if no patterns match
                keep_result = not any(matches)
            
            if keep_result:
                filtered_results.append(result)
        
        return filtered_results
    
    def get_config(self) -> Dict[str, Any]:
        """Get filter configuration."""
        return {
            "type": "RegexResultFilter",
            "patterns": self.patterns,
            "match_mode": self.match_mode
        }


class CustomFunctionResultFilter(ResultFilter):
    """Filter results using a custom function."""
    
    def __init__(self, filter_function: Callable[[Dict[str, Any]], bool]):
        """
        Initialize custom function filter.
        
        Args:
            filter_function: Function that takes a result dict and returns bool
        """
        self.filter_function = filter_function
    
    def filter(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results using custom function."""
        return [
            result for result in results
            if self.filter_function(result)
        ]
    
    def get_config(self) -> Dict[str, Any]:
        """Get filter configuration."""
        return {
            "type": "CustomFunctionResultFilter",
            "has_filter_function": self.filter_function is not None
        }


class ChainedResultFilter(ResultFilter):
    """Filter that chains multiple filters."""
    
    def __init__(self, filters: List[ResultFilter]):
        """
        Initialize chained filter.
        
        Args:
            filters: List of filters to apply in order
        """
        self.filters = filters
    
    def filter(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply all filters in sequence."""
        filtered_results = results
        
        for filter_instance in self.filters:
            filtered_results = filter_instance.filter(filtered_results)
        
        return filtered_results
    
    def get_config(self) -> Dict[str, Any]:
        """Get filter configuration."""
        return {
            "type": "ChainedResultFilter",
            "filters": [f.get_config() for f in self.filters]
        }


class ProductSpecificResultFilter(ResultFilter):
    """Filter specifically designed for product similarity results."""
    
    def __init__(self, 
                 min_similarity: float = 0.5,
                 remove_identical: bool = True,
                 remove_short_products: bool = True,
                 min_product_name_length: int = 3,
                 remove_promotional: bool = True):
        """
        Initialize product-specific filter.
        
        Args:
            min_similarity: Minimum similarity threshold
            remove_identical: Remove results with identical text
            remove_short_products: Remove very short product names
            min_product_name_length: Minimum length for product names
            remove_promotional: Remove promotional/spam-like results
        """
        self.min_similarity = min_similarity
        self.remove_identical = remove_identical
        self.remove_short_products = remove_short_products
        self.min_product_name_length = min_product_name_length
        self.remove_promotional = remove_promotional
        
        # Promotional/spam patterns
        self.promotional_patterns = [
            r'โปรโมชั่น|ลดราคา|ราคาพิเศษ|ส่งฟรี',
            r'promotion|sale|discount|free shipping',
            r'ซื้อ\s*\d+\s*แถม\s*\d+',
            r'ของแถม|แถมฟรี',
            r'limited time|special offer'
        ]
        
        self.compiled_promotional = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.promotional_patterns
        ]
    
    def filter(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply product-specific filtering."""
        filtered_results = []
        
        for result in results:
            # Check similarity threshold
            if result.get('similarity_score', 0) < self.min_similarity:
                continue
            
            # Get text fields
            text1 = str(result.get('text1', ''))
            text2 = str(result.get('text2', ''))
            
            # Remove identical text
            if self.remove_identical and text1.strip().lower() == text2.strip().lower():
                continue
            
            # Remove short product names
            if self.remove_short_products:
                if (len(text1.strip()) < self.min_product_name_length or 
                    len(text2.strip()) < self.min_product_name_length):
                    continue
            
            # Remove promotional content
            if self.remove_promotional:
                is_promotional = False
                
                for pattern in self.compiled_promotional:
                    if pattern.search(text1) or pattern.search(text2):
                        is_promotional = True
                        break
                
                if is_promotional:
                    continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def get_config(self) -> Dict[str, Any]:
        """Get filter configuration."""
        return {
            "type": "ProductSpecificResultFilter",
            "min_similarity": self.min_similarity,
            "remove_identical": self.remove_identical,
            "remove_short_products": self.remove_short_products,
            "min_product_name_length": self.min_product_name_length,
            "remove_promotional": self.remove_promotional
        }


def create_result_filter(filter_type: str, **kwargs) -> ResultFilter:
    """
    Factory function to create result filters.
    
    Args:
        filter_type: Type of filter ('threshold', 'topk', 'duplicate', 'length', 
                                   'regex', 'custom', 'chained', 'product')
        **kwargs: Additional arguments for the filter
        
    Returns:
        Configured result filter instance
    """
    if filter_type.lower() == "threshold":
        return ThresholdResultFilter(**kwargs)
    elif filter_type.lower() == "topk":
        return TopKResultFilter(**kwargs)
    elif filter_type.lower() == "duplicate":
        return DuplicateResultFilter(**kwargs)
    elif filter_type.lower() == "length":
        return TextLengthResultFilter(**kwargs)
    elif filter_type.lower() == "regex":
        return RegexResultFilter(**kwargs)
    elif filter_type.lower() == "custom":
        return CustomFunctionResultFilter(**kwargs)
    elif filter_type.lower() == "chained":
        return ChainedResultFilter(**kwargs)
    elif filter_type.lower() == "product":
        return ProductSpecificResultFilter(**kwargs)
    else:
        raise ValueError(f"Unknown result filter type: {filter_type}")


def create_default_product_filter(min_similarity: float = 0.6) -> ResultFilter:
    """
    Create a default filter optimized for product similarity results.
    
    Args:
        min_similarity: Minimum similarity threshold
        
    Returns:
        Chained filter with product-optimized settings
    """
    filters = [
        # First remove very low similarity results
        ThresholdResultFilter(threshold=min_similarity),
        
        # Remove duplicates
        DuplicateResultFilter(
            key_fields=['text1', 'text2'],
            keep_first=True
        ),
        
        # Apply product-specific filtering
        ProductSpecificResultFilter(
            min_similarity=min_similarity,
            remove_identical=True,
            remove_short_products=True,
            min_product_name_length=3,
            remove_promotional=True
        ),
        
        # Keep only top 1000 results to avoid huge output files
        TopKResultFilter(k=1000)
    ]
    
    return ChainedResultFilter(filters)
