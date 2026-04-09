"""
Data Testing Utilities - Common data processing functions
========================================================

Centralized data utilities to reduce duplication
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from io import BytesIO
from ..config import TestConfig


class DataTestHelper:
    """Helper class for data testing"""
    
    @staticmethod
    def create_sample_dataframe(product_type='old', custom_products=None) -> pd.DataFrame:
        """
        Create sample DataFrame for testing
        
        Args:
            product_type: 'old' or 'new' for predefined samples
            custom_products: List of custom product names
            
        Returns:
            pandas DataFrame with sample data
        """
        if custom_products:
            products = custom_products
        else:
            products = TestConfig.SAMPLE_PRODUCTS.get(product_type, TestConfig.SAMPLE_PRODUCTS['old'])
        
        return pd.DataFrame({'รายการ': products})
    
    @staticmethod
    def create_test_csv_bytes(product_type='old', custom_products=None) -> bytes:
        """
        Create CSV data as bytes for API testing
        
        Args:
            product_type: 'old' or 'new' for predefined samples
            custom_products: List of custom product names
            
        Returns:
            CSV data as bytes
        """
        df = DataTestHelper.create_sample_dataframe(product_type, custom_products)
        return df.to_csv(index=False).encode('utf-8')
    
    @staticmethod
    def load_input_data() -> Dict[str, List[str]]:
        """
        Load real input data from files
        
        Returns:
            Dictionary with 'new' and 'old' product lists
        """
        products = {'new': [], 'old': []}
        
        # Load new products
        new_file = Path(TestConfig.DATA_PATHS['new_products'])
        if new_file.exists():
            try:
                df_new = pd.read_csv(new_file)
                if 'รายการ' in df_new.columns:
                    products['new'] = df_new['รายการ'].dropna().tolist()
            except Exception as e:
                print(f"Warning: Could not load new products: {e}")
        
        # Load old products  
        old_file = Path(TestConfig.DATA_PATHS['old_products'])
        if old_file.exists():
            try:
                df_old = pd.read_csv(old_file)
                if 'รายการ' in df_old.columns:
                    products['old'] = df_old['รายการ'].dropna().tolist()
                elif 'name' in df_old.columns:
                    products['old'] = df_old['name'].dropna().tolist()
            except Exception as e:
                print(f"Warning: Could not load old products: {e}")
        
        return products
    
    @staticmethod
    def save_test_results(data: pd.DataFrame, filename: str, output_dir: str = None) -> Path:
        """
        Save test results to CSV file
        
        Args:
            data: DataFrame to save
            filename: Output filename
            output_dir: Output directory (default: from config)
            
        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = TestConfig.DATA_PATHS['output_dir']
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / filename
        data.to_csv(file_path, index=False, encoding='utf-8-sig')
        
        return file_path
    
    @staticmethod
    def create_similarity_matrix(new_products: List[str], old_products: List[str], similarity_scores: List[List[float]]) -> pd.DataFrame:
        """
        Create similarity matrix DataFrame
        
        Args:
            new_products: List of new product names
            old_products: List of old product names  
            similarity_scores: 2D list of similarity scores
            
        Returns:
            DataFrame with similarity matrix
        """
        return pd.DataFrame(
            similarity_scores,
            index=new_products,
            columns=old_products
        )
    
    @staticmethod
    def create_matches_dataframe(matches: List[Dict]) -> pd.DataFrame:
        """
        Create matches DataFrame from match results
        
        Args:
            matches: List of match dictionaries
            
        Returns:
            DataFrame with match results
        """
        if not matches:
            return pd.DataFrame()
        
        return pd.DataFrame(matches)
    
    @staticmethod
    def validate_csv_structure(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame has required columns
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            Tuple of (is_valid, missing_columns)
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        return len(missing_columns) == 0, missing_columns
    
    @staticmethod
    def get_sample_thai_products(count: int = 5) -> Dict[str, List[str]]:
        """
        Get sample Thai products for testing
        
        Args:
            count: Number of products to return
            
        Returns:
            Dictionary with 'old' and 'new' product lists
        """
        old_products = TestConfig.SAMPLE_PRODUCTS['old'][:count]
        new_products = TestConfig.SAMPLE_PRODUCTS['new'][:count]
        
        return {
            'old': old_products,
            'new': new_products
        }