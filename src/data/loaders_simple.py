"""
Simplified data loading implementations.
"""

import pandas as pd
import json
import csv
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from ..core.interfaces_simple import DataLoader


class CSVDataLoader(DataLoader):
    """CSV file data loader."""
    
    def __init__(self, 
                 encoding: str = 'utf-8',
                 delimiter: str = ',',
                 header: int = 0):
        """
        Initialize CSV loader.
        
        Args:
            encoding: File encoding
            delimiter: CSV delimiter
            header: Header row number
        """
        self.encoding = encoding
        self.delimiter = delimiter
        self.header = header
    
    def load(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(
                file_path,
                encoding=self.encoding,
                delimiter=self.delimiter,
                header=self.header
            )
            return df.to_dict('records')
        except Exception as e:
            raise ValueError(f"Failed to load CSV file {file_path}: {e}")
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return ['.csv']


class JSONDataLoader(DataLoader):
    """JSON file data loader."""
    
    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize JSON loader.
        
        Args:
            encoding: File encoding
        """
        self.encoding = encoding
    
    def load(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                data = json.load(f)
            
            # Ensure data is a list of dictionaries
            if isinstance(data, dict):
                return [data]
            elif isinstance(data, list):
                return data
            else:
                raise ValueError("JSON data must be a dictionary or list of dictionaries")
                
        except Exception as e:
            raise ValueError(f"Failed to load JSON file {file_path}: {e}")
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return ['.json', '.jsonl']


class TextDataLoader(DataLoader):
    """Plain text file data loader."""
    
    def __init__(self, 
                 encoding: str = 'utf-8',
                 text_column: str = 'text',
                 id_column: str = 'id'):
        """
        Initialize text loader.
        
        Args:
            encoding: File encoding
            text_column: Column name for text content
            id_column: Column name for ID
        """
        self.encoding = encoding
        self.text_column = text_column
        self.id_column = id_column
    
    def load(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load data from text file."""
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                lines = f.readlines()
            
            records = []
            for i, line in enumerate(lines):
                line = line.strip()
                if line:  # Skip empty lines
                    records.append({
                        self.id_column: i + 1,
                        self.text_column: line
                    })
            
            return records
                
        except Exception as e:
            raise ValueError(f"Failed to load text file {file_path}: {e}")
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return ['.txt']
