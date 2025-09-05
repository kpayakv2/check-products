"""
Data writing implementations.

This module provides implementations of the DataWriter interface
for different output formats.
"""

import pandas as pd
import json
import csv
from typing import List, Dict, Any, Union, Optional
from pathlib import Path

from ..core.interfaces import DataWriter


class CSVDataWriter(DataWriter):
    """CSV file data writer."""
    
    def __init__(self, 
                 encoding: str = 'utf-8',
                 delimiter: str = ',',
                 include_index: bool = False):
        """
        Initialize CSV data writer.
        
        Args:
            encoding: File encoding
            delimiter: CSV delimiter
            include_index: Whether to include row index
        """
        self.encoding = encoding
        self.delimiter = delimiter
        self.include_index = include_index
    
    def write(self, data: List[Dict[str, Any]], destination: Union[str, Path]) -> None:
        """
        Write data to CSV file.
        
        Args:
            data: List of dictionaries to write
            destination: Output file path
        """
        try:
            if not data:
                # Create empty file
                Path(destination).touch()
                return
            
            df = pd.DataFrame(data)
            df.to_csv(
                destination,
                encoding=self.encoding,
                sep=self.delimiter,
                index=self.include_index
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to write CSV data to {destination}: {str(e)}")
    
    def append(self, data: List[Dict[str, Any]], destination: Union[str, Path]) -> None:
        """
        Append data to CSV file.
        
        Args:
            data: List of dictionaries to append
            destination: Output file path
        """
        try:
            if not data:
                return
            
            df = pd.DataFrame(data)
            
            # Check if file exists
            if Path(destination).exists():
                df.to_csv(
                    destination,
                    mode='a',
                    header=False,
                    encoding=self.encoding,
                    sep=self.delimiter,
                    index=self.include_index
                )
            else:
                # Write with header if file doesn't exist
                df.to_csv(
                    destination,
                    encoding=self.encoding,
                    sep=self.delimiter,
                    index=self.include_index
                )
                
        except Exception as e:
            raise RuntimeError(f"Failed to append CSV data to {destination}: {str(e)}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get writer configuration."""
        return {
            "type": "CSVDataWriter",
            "encoding": self.encoding,
            "delimiter": self.delimiter,
            "include_index": self.include_index
        }


class JSONDataWriter(DataWriter):
    """JSON file data writer."""
    
    def __init__(self, 
                 encoding: str = 'utf-8',
                 indent: Optional[int] = 2,
                 ensure_ascii: bool = False):
        """
        Initialize JSON data writer.
        
        Args:
            encoding: File encoding
            indent: JSON indentation (None for compact)
            ensure_ascii: Whether to escape non-ASCII characters
        """
        self.encoding = encoding
        self.indent = indent
        self.ensure_ascii = ensure_ascii
    
    def write(self, data: List[Dict[str, Any]], destination: Union[str, Path]) -> None:
        """
        Write data to JSON file.
        
        Args:
            data: List of dictionaries to write
            destination: Output file path
        """
        try:
            with open(destination, 'w', encoding=self.encoding) as f:
                json.dump(
                    data,
                    f,
                    indent=self.indent,
                    ensure_ascii=self.ensure_ascii,
                    default=str  # Handle non-serializable objects
                )
                
        except Exception as e:
            raise RuntimeError(f"Failed to write JSON data to {destination}: {str(e)}")
    
    def append(self, data: List[Dict[str, Any]], destination: Union[str, Path]) -> None:
        """
        Append data to JSON file.
        
        Note: This loads the entire file, appends data, and rewrites.
        Not efficient for very large files.
        
        Args:
            data: List of dictionaries to append
            destination: Output file path
        """
        try:
            existing_data = []
            
            # Load existing data if file exists
            if Path(destination).exists():
                with open(destination, 'r', encoding=self.encoding) as f:
                    try:
                        existing_data = json.load(f)
                        if not isinstance(existing_data, list):
                            existing_data = [existing_data]
                    except json.JSONDecodeError:
                        existing_data = []
            
            # Combine and write
            combined_data = existing_data + data
            self.write(combined_data, destination)
            
        except Exception as e:
            raise RuntimeError(f"Failed to append JSON data to {destination}: {str(e)}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get writer configuration."""
        return {
            "type": "JSONDataWriter",
            "encoding": self.encoding,
            "indent": self.indent,
            "ensure_ascii": self.ensure_ascii
        }


class JSONLDataWriter(DataWriter):
    """JSON Lines (.jsonl) data writer."""
    
    def __init__(self, 
                 encoding: str = 'utf-8',
                 ensure_ascii: bool = False):
        """
        Initialize JSONL data writer.
        
        Args:
            encoding: File encoding
            ensure_ascii: Whether to escape non-ASCII characters
        """
        self.encoding = encoding
        self.ensure_ascii = ensure_ascii
    
    def write(self, data: List[Dict[str, Any]], destination: Union[str, Path]) -> None:
        """
        Write data to JSONL file.
        
        Args:
            data: List of dictionaries to write
            destination: Output file path
        """
        try:
            with open(destination, 'w', encoding=self.encoding) as f:
                for item in data:
                    json.dump(
                        item,
                        f,
                        ensure_ascii=self.ensure_ascii,
                        default=str
                    )
                    f.write('\n')
                    
        except Exception as e:
            raise RuntimeError(f"Failed to write JSONL data to {destination}: {str(e)}")
    
    def append(self, data: List[Dict[str, Any]], destination: Union[str, Path]) -> None:
        """
        Append data to JSONL file.
        
        Args:
            data: List of dictionaries to append
            destination: Output file path
        """
        try:
            with open(destination, 'a', encoding=self.encoding) as f:
                for item in data:
                    json.dump(
                        item,
                        f,
                        ensure_ascii=self.ensure_ascii,
                        default=str
                    )
                    f.write('\n')
                    
        except Exception as e:
            raise RuntimeError(f"Failed to append JSONL data to {destination}: {str(e)}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get writer configuration."""
        return {
            "type": "JSONLDataWriter",
            "encoding": self.encoding,
            "ensure_ascii": self.ensure_ascii
        }


class ExcelDataWriter(DataWriter):
    """Excel file data writer."""
    
    def __init__(self, 
                 sheet_name: str = 'Sheet1',
                 include_index: bool = False):
        """
        Initialize Excel data writer.
        
        Args:
            sheet_name: Name of the Excel sheet
            include_index: Whether to include row index
        """
        self.sheet_name = sheet_name
        self.include_index = include_index
    
    def write(self, data: List[Dict[str, Any]], destination: Union[str, Path]) -> None:
        """
        Write data to Excel file.
        
        Args:
            data: List of dictionaries to write
            destination: Output file path
        """
        try:
            if not data:
                # Create empty workbook
                pd.DataFrame().to_excel(destination, sheet_name=self.sheet_name, index=False)
                return
            
            df = pd.DataFrame(data)
            df.to_excel(
                destination,
                sheet_name=self.sheet_name,
                index=self.include_index
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to write Excel data to {destination}: {str(e)}")
    
    def append(self, data: List[Dict[str, Any]], destination: Union[str, Path]) -> None:
        """
        Append data to Excel file.
        
        Note: This loads the entire file, appends data, and rewrites.
        
        Args:
            data: List of dictionaries to append
            destination: Output file path
        """
        try:
            if not data:
                return
            
            new_df = pd.DataFrame(data)
            
            # Check if file exists
            if Path(destination).exists():
                try:
                    existing_df = pd.read_excel(destination, sheet_name=self.sheet_name)
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                except:
                    # If reading fails, just write new data
                    combined_df = new_df
            else:
                combined_df = new_df
            
            combined_df.to_excel(
                destination,
                sheet_name=self.sheet_name,
                index=self.include_index
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to append Excel data to {destination}: {str(e)}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get writer configuration."""
        return {
            "type": "ExcelDataWriter",
            "sheet_name": self.sheet_name,
            "include_index": self.include_index
        }


class MultiFormatDataWriter(DataWriter):
    """Data writer that can handle multiple output formats."""
    
    def __init__(self, 
                 csv_config: Optional[Dict[str, Any]] = None,
                 json_config: Optional[Dict[str, Any]] = None,
                 jsonl_config: Optional[Dict[str, Any]] = None,
                 excel_config: Optional[Dict[str, Any]] = None):
        """
        Initialize multi-format data writer.
        
        Args:
            csv_config: Configuration for CSV writer
            json_config: Configuration for JSON writer
            jsonl_config: Configuration for JSONL writer
            excel_config: Configuration for Excel writer
        """
        self.csv_config = csv_config or {}
        self.json_config = json_config or {}
        self.jsonl_config = jsonl_config or {}
        self.excel_config = excel_config or {}
        
        self._writers = {
            'csv': CSVDataWriter(**self.csv_config),
            'json': JSONDataWriter(**self.json_config),
            'jsonl': JSONLDataWriter(**self.jsonl_config),
            'xlsx': ExcelDataWriter(**self.excel_config),
            'xls': ExcelDataWriter(**self.excel_config)
        }
    
    def _get_file_type(self, destination: Union[str, Path]) -> str:
        """Determine file type from extension."""
        path = Path(destination)
        suffix = path.suffix.lower().lstrip('.')
        
        if suffix in self._writers:
            return suffix
        else:
            # Default to CSV for unknown extensions
            return 'csv'
    
    def write(self, data: List[Dict[str, Any]], destination: Union[str, Path]) -> None:
        """
        Write data to file (auto-detect format).
        
        Args:
            data: List of dictionaries to write
            destination: Output file path
        """
        file_type = self._get_file_type(destination)
        writer = self._writers[file_type]
        writer.write(data, destination)
    
    def append(self, data: List[Dict[str, Any]], destination: Union[str, Path]) -> None:
        """
        Append data to file (auto-detect format).
        
        Args:
            data: List of dictionaries to append
            destination: Output file path
        """
        file_type = self._get_file_type(destination)
        writer = self._writers[file_type]
        writer.append(data, destination)
    
    def get_config(self) -> Dict[str, Any]:
        """Get writer configuration."""
        return {
            "type": "MultiFormatDataWriter",
            "csv_config": self.csv_config,
            "json_config": self.json_config,
            "jsonl_config": self.jsonl_config,
            "excel_config": self.excel_config,
            "supported_formats": list(self._writers.keys())
        }


def create_data_writer(writer_type: str, **kwargs) -> DataWriter:
    """
    Factory function to create data writers.
    
    Args:
        writer_type: Type of writer ('csv', 'json', 'jsonl', 'excel', 'multi')
        **kwargs: Additional arguments for the writer
        
    Returns:
        Configured data writer instance
    """
    if writer_type.lower() == "csv":
        return CSVDataWriter(**kwargs)
    elif writer_type.lower() == "json":
        return JSONDataWriter(**kwargs)
    elif writer_type.lower() == "jsonl":
        return JSONLDataWriter(**kwargs)
    elif writer_type.lower() == "excel":
        return ExcelDataWriter(**kwargs)
    elif writer_type.lower() == "multi":
        return MultiFormatDataWriter(**kwargs)
    else:
        raise ValueError(f"Unknown data writer type: {writer_type}")


def create_default_result_writer() -> DataWriter:
    """
    Create a default data writer optimized for similarity results.
    
    Returns:
        Multi-format data writer with result-optimized settings
    """
    return MultiFormatDataWriter(
        csv_config={
            'encoding': 'utf-8',
            'delimiter': ',',
            'include_index': False
        },
        json_config={
            'encoding': 'utf-8',
            'indent': 2,
            'ensure_ascii': False
        },
        jsonl_config={
            'encoding': 'utf-8',
            'ensure_ascii': False
        },
        excel_config={
            'sheet_name': 'Results',
            'include_index': False
        }
    )
