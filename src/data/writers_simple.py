"""
Simplified data writing implementations.
"""

import pandas as pd
import json
import csv
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..core.interfaces_simple import DataWriter


class CSVDataWriter(DataWriter):
    """CSV file data writer."""
    
    def __init__(self, 
                 encoding: str = 'utf-8',
                 delimiter: str = ',',
                 index: bool = False):
        """
        Initialize CSV writer.
        
        Args:
            encoding: File encoding
            delimiter: CSV delimiter
            index: Whether to write row indices
        """
        self.encoding = encoding
        self.delimiter = delimiter
        self.index = index
    
    def write(self, data: List[Dict[str, Any]], file_path: Path) -> None:
        """Write data to CSV file."""
        if not data:
            raise ValueError("No data to write")
        
        try:
            df = pd.DataFrame(data)
            df.to_csv(
                file_path,
                encoding=self.encoding,
                sep=self.delimiter,
                index=self.index
            )
        except Exception as e:
            raise ValueError(f"Failed to write CSV file {file_path}: {e}")
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return ['.csv']


class JSONDataWriter(DataWriter):
    """JSON file data writer."""
    
    def __init__(self, 
                 encoding: str = 'utf-8',
                 indent: int = 2,
                 ensure_ascii: bool = False):
        """
        Initialize JSON writer.
        
        Args:
            encoding: File encoding
            indent: JSON indentation
            ensure_ascii: Whether to escape non-ASCII characters
        """
        self.encoding = encoding
        self.indent = indent
        self.ensure_ascii = ensure_ascii
    
    def write(self, data: List[Dict[str, Any]], file_path: Path) -> None:
        """Write data to JSON file."""
        if not data:
            raise ValueError("No data to write")
        
        try:
            with open(file_path, 'w', encoding=self.encoding) as f:
                json.dump(data, f, 
                         indent=self.indent, 
                         ensure_ascii=self.ensure_ascii)
        except Exception as e:
            raise ValueError(f"Failed to write JSON file {file_path}: {e}")
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return ['.json']


class TextDataWriter(DataWriter):
    """Plain text file data writer."""
    
    def __init__(self, 
                 encoding: str = 'utf-8',
                 text_column: str = 'text',
                 separator: str = '\n'):
        """
        Initialize text writer.
        
        Args:
            encoding: File encoding
            text_column: Column name containing text to write
            separator: Line separator
        """
        self.encoding = encoding
        self.text_column = text_column
        self.separator = separator
    
    def write(self, data: List[Dict[str, Any]], file_path: Path) -> None:
        """Write data to text file."""
        if not data:
            raise ValueError("No data to write")
        
        try:
            lines = []
            for record in data:
                if self.text_column in record:
                    lines.append(str(record[self.text_column]))
                else:
                    # If text_column not found, use string representation
                    lines.append(str(record))
            
            content = self.separator.join(lines)
            
            with open(file_path, 'w', encoding=self.encoding) as f:
                f.write(content)
                
        except Exception as e:
            raise ValueError(f"Failed to write text file {file_path}: {e}")
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return ['.txt']
