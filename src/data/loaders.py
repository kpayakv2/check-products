"""
Data loading implementations.

This module provides implementations of the DataLoader interface
for different data sources and formats.
"""

import pandas as pd
import json
import csv
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from ..core.interfaces import DataLoader


class CSVDataLoader(DataLoader):
    """CSV file data loader."""
    
    def __init__(self, 
                 text_column: str = 'product_name',
                 encoding: str = 'utf-8',
                 delimiter: str = ','):
        """
        Initialize CSV data loader.
        
        Args:
            text_column: Name of the column containing text data
            encoding: File encoding
            delimiter: CSV delimiter
        """
        self.text_column = text_column
        self.encoding = encoding
        self.delimiter = delimiter
    
    def load(self, source: Union[str, Path]) -> List[str]:
        """
        Load text data from CSV file.
        
        Args:
            source: Path to CSV file
            
        Returns:
            List of text strings
        """
        try:
            df = pd.read_csv(
                source, 
                encoding=self.encoding,
                delimiter=self.delimiter
            )
            
            if self.text_column not in df.columns:
                raise ValueError(f"Column '{self.text_column}' not found in CSV file")
            
            # Convert to string and remove nulls
            texts = df[self.text_column].astype(str).fillna('').tolist()
            
            # Filter out empty strings
            texts = [text.strip() for text in texts if text.strip()]
            
            return texts
            
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV data from {source}: {str(e)}")
    
    def load_with_metadata(self, source: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load text data with additional metadata.
        
        Args:
            source: Path to CSV file
            
        Returns:
            List of dictionaries containing text and metadata
        """
        try:
            df = pd.read_csv(
                source,
                encoding=self.encoding,
                delimiter=self.delimiter
            )
            
            if self.text_column not in df.columns:
                raise ValueError(f"Column '{self.text_column}' not found in CSV file")
            
            results = []
            for idx, row in df.iterrows():
                text = str(row[self.text_column]).strip()
                if text:  # Only include non-empty texts
                    item = {
                        'text': text,
                        'index': idx,
                        'metadata': row.to_dict()
                    }
                    results.append(item)
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV data with metadata from {source}: {str(e)}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get loader configuration."""
        return {
            "type": "CSVDataLoader",
            "text_column": self.text_column,
            "encoding": self.encoding,
            "delimiter": self.delimiter
        }


class JSONDataLoader(DataLoader):
    """JSON file data loader."""
    
    def __init__(self, 
                 text_field: str = 'text',
                 encoding: str = 'utf-8'):
        """
        Initialize JSON data loader.
        
        Args:
            text_field: Name of the field containing text data
            encoding: File encoding
        """
        self.text_field = text_field
        self.encoding = encoding
    
    def load(self, source: Union[str, Path]) -> List[str]:
        """
        Load text data from JSON file.
        
        Args:
            source: Path to JSON file
            
        Returns:
            List of text strings
        """
        try:
            with open(source, 'r', encoding=self.encoding) as f:
                data = json.load(f)
            
            texts = []
            
            if isinstance(data, list):
                # List of objects
                for item in data:
                    if isinstance(item, dict) and self.text_field in item:
                        text = str(item[self.text_field]).strip()
                        if text:
                            texts.append(text)
                    elif isinstance(item, str):
                        text = item.strip()
                        if text:
                            texts.append(text)
            elif isinstance(data, dict):
                # Single object or nested structure
                if self.text_field in data:
                    if isinstance(data[self.text_field], list):
                        texts = [str(t).strip() for t in data[self.text_field] if str(t).strip()]
                    else:
                        text = str(data[self.text_field]).strip()
                        if text:
                            texts = [text]
            
            return texts
            
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON data from {source}: {str(e)}")
    
    def load_with_metadata(self, source: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load text data with additional metadata.
        
        Args:
            source: Path to JSON file
            
        Returns:
            List of dictionaries containing text and metadata
        """
        try:
            with open(source, 'r', encoding=self.encoding) as f:
                data = json.load(f)
            
            results = []
            
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    if isinstance(item, dict) and self.text_field in item:
                        text = str(item[self.text_field]).strip()
                        if text:
                            results.append({
                                'text': text,
                                'index': idx,
                                'metadata': item
                            })
                    elif isinstance(item, str):
                        text = item.strip()
                        if text:
                            results.append({
                                'text': text,
                                'index': idx,
                                'metadata': {'original': item}
                            })
            elif isinstance(data, dict) and self.text_field in data:
                if isinstance(data[self.text_field], list):
                    for idx, text in enumerate(data[self.text_field]):
                        text = str(text).strip()
                        if text:
                            results.append({
                                'text': text,
                                'index': idx,
                                'metadata': data
                            })
                else:
                    text = str(data[self.text_field]).strip()
                    if text:
                        results.append({
                            'text': text,
                            'index': 0,
                            'metadata': data
                        })
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON data with metadata from {source}: {str(e)}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get loader configuration."""
        return {
            "type": "JSONDataLoader",
            "text_field": self.text_field,
            "encoding": self.encoding
        }


class TextDataLoader(DataLoader):
    """Plain text file data loader."""
    
    def __init__(self, 
                 encoding: str = 'utf-8',
                 split_by_lines: bool = True,
                 strip_empty: bool = True):
        """
        Initialize text data loader.
        
        Args:
            encoding: File encoding
            split_by_lines: Whether to split text by lines
            strip_empty: Whether to remove empty lines
        """
        self.encoding = encoding
        self.split_by_lines = split_by_lines
        self.strip_empty = strip_empty
    
    def load(self, source: Union[str, Path]) -> List[str]:
        """
        Load text data from plain text file.
        
        Args:
            source: Path to text file
            
        Returns:
            List of text strings
        """
        try:
            with open(source, 'r', encoding=self.encoding) as f:
                content = f.read()
            
            if self.split_by_lines:
                texts = content.split('\n')
                if self.strip_empty:
                    texts = [text.strip() for text in texts if text.strip()]
                else:
                    texts = [text.strip() for text in texts]
            else:
                texts = [content.strip()] if content.strip() else []
            
            return texts
            
        except Exception as e:
            raise RuntimeError(f"Failed to load text data from {source}: {str(e)}")
    
    def load_with_metadata(self, source: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load text data with additional metadata.
        
        Args:
            source: Path to text file
            
        Returns:
            List of dictionaries containing text and metadata
        """
        texts = self.load(source)
        
        results = []
        for idx, text in enumerate(texts):
            results.append({
                'text': text,
                'index': idx,
                'metadata': {
                    'source_file': str(source),
                    'line_number': idx + 1 if self.split_by_lines else 0
                }
            })
        
        return results
    
    def get_config(self) -> Dict[str, Any]:
        """Get loader configuration."""
        return {
            "type": "TextDataLoader",
            "encoding": self.encoding,
            "split_by_lines": self.split_by_lines,
            "strip_empty": self.strip_empty
        }


class MultiSourceDataLoader(DataLoader):
    """Data loader that can handle multiple file formats."""
    
    def __init__(self, 
                 csv_config: Optional[Dict[str, Any]] = None,
                 json_config: Optional[Dict[str, Any]] = None,
                 text_config: Optional[Dict[str, Any]] = None):
        """
        Initialize multi-source data loader.
        
        Args:
            csv_config: Configuration for CSV loader
            json_config: Configuration for JSON loader
            text_config: Configuration for text loader
        """
        self.csv_config = csv_config or {}
        self.json_config = json_config or {}
        self.text_config = text_config or {}
        
        self._loaders = {
            'csv': CSVDataLoader(**self.csv_config),
            'json': JSONDataLoader(**self.json_config),
            'txt': TextDataLoader(**self.text_config),
            'text': TextDataLoader(**self.text_config)
        }
    
    def _get_file_type(self, source: Union[str, Path]) -> str:
        """Determine file type from extension."""
        path = Path(source)
        suffix = path.suffix.lower().lstrip('.')
        
        if suffix in self._loaders:
            return suffix
        elif suffix == 'jsonl':
            return 'json'
        else:
            # Default to text for unknown extensions
            return 'txt'
    
    def load(self, source: Union[str, Path]) -> List[str]:
        """
        Load text data from file (auto-detect format).
        
        Args:
            source: Path to data file
            
        Returns:
            List of text strings
        """
        file_type = self._get_file_type(source)
        loader = self._loaders[file_type]
        return loader.load(source)
    
    def load_with_metadata(self, source: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load text data with additional metadata (auto-detect format).
        
        Args:
            source: Path to data file
            
        Returns:
            List of dictionaries containing text and metadata
        """
        file_type = self._get_file_type(source)
        loader = self._loaders[file_type]
        return loader.load_with_metadata(source)
    
    def get_config(self) -> Dict[str, Any]:
        """Get loader configuration."""
        return {
            "type": "MultiSourceDataLoader",
            "csv_config": self.csv_config,
            "json_config": self.json_config,
            "text_config": self.text_config,
            "supported_formats": list(self._loaders.keys())
        }


def create_data_loader(loader_type: str, **kwargs) -> DataLoader:
    """
    Factory function to create data loaders.
    
    Args:
        loader_type: Type of loader ('csv', 'json', 'text', 'multi')
        **kwargs: Additional arguments for the loader
        
    Returns:
        Configured data loader instance
    """
    if loader_type.lower() == "csv":
        return CSVDataLoader(**kwargs)
    elif loader_type.lower() == "json":
        return JSONDataLoader(**kwargs)
    elif loader_type.lower() == "text":
        return TextDataLoader(**kwargs)
    elif loader_type.lower() == "multi":
        return MultiSourceDataLoader(**kwargs)
    else:
        raise ValueError(f"Unknown data loader type: {loader_type}")


def create_default_product_loader() -> DataLoader:
    """
    Create a default data loader optimized for product data.
    
    Returns:
        Multi-source data loader with product-optimized settings
    """
    return MultiSourceDataLoader(
        csv_config={
            'text_column': 'product_name',
            'encoding': 'utf-8',
            'delimiter': ','
        },
        json_config={
            'text_field': 'name',
            'encoding': 'utf-8'
        },
        text_config={
            'encoding': 'utf-8',
            'split_by_lines': True,
            'strip_empty': True
        }
    )
