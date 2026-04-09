#!/usr/bin/env python3
"""
Product Data Utilities
======================

Shared utilities for product data extraction and processing.
Consolidates common functions used across multiple modules to eliminate code duplication.
"""

import logging
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class ColumnNames:
    """Standard column name configurations."""
    THAI_COLUMNS = ['รายการ', 'ชื่อสินค้า', 'สินค้า', 'รายการสินค้า']
    ENGLISH_COLUMNS = ['name', 'product_name', 'product', 'item_name', 'item', 'title']


class ThresholdConfig:
    """Similarity threshold configurations."""
    PERFECT_MATCH = 0.95    # Identical products (auto-exclude)
    HIGH_SIMILARITY = 0.8   # Likely duplicate (needs review)
    LOW_SIMILARITY = 0.3    # Unique product
    MAX_CONFIDENCE = 0.95
    HIGH_CONFIDENCE = 0.8


def extract_product_names(df: pd.DataFrame, column_hint: Optional[str] = None) -> List[str]:
    """
    Extract product names from DataFrame with smart column detection.
    
    Args:
        df: DataFrame containing product data
        column_hint: Optional specific column name to use
        
    Returns:
        List of product names as strings
        
    Raises:
        ValueError: If no suitable column is found
    """
    if df is None or df.empty:
        logger.warning("Empty or None DataFrame provided")
        return []
    
    cols = list(df.columns)
    
    # If column hint is provided, use it directly
    if column_hint and column_hint in cols:
        result = df[column_hint].dropna().astype(str).tolist()
        logger.info(f"Using specified column '{column_hint}': {len(result)} items")
        return result
    
    # Try Thai column names first, then English
    all_column_names = ColumnNames.THAI_COLUMNS + ColumnNames.ENGLISH_COLUMNS
    for col_name in all_column_names:
        if col_name in cols:
            result = df[col_name].dropna().astype(str).tolist()
            logger.info(f"Found product names in column '{col_name}': {len(result)} items")
            return result
    
    # Find text column by sampling
    for col in cols:
        sample = df[col].dropna()
        if len(sample) > 0:
            sample_val = sample.iloc[0]
            if isinstance(sample_val, str) and len(sample_val) > 2:
                result = df[col].dropna().astype(str).tolist()
                logger.info(f"Using column '{col}' as product names: {len(result)} items")
                return result
    
    # Fallback to first column
    if len(cols) > 0:
        result = df.iloc[:, 0].dropna().astype(str).tolist()
        logger.warning(f"Using first column '{cols[0]}' as fallback: {len(result)} items")
        return result
    
    logger.error("No suitable columns found for product names")
    raise ValueError("No suitable columns found for product names")


def classify_products(matches: List[Dict[str, Any]], 
                     new_products: List[str], 
                     threshold: float) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Classify products into unique, duplicate, and auto-excluded categories based on similarity scores.
    
    Args:
        matches: List of similarity match results
        new_products: List of all new product names
        threshold: Similarity threshold for classification
        
    Returns:
        Tuple of (unique_products, duplicate_check_needed, excluded_duplicates) lists
    """
    if not matches:
        logger.warning("No matches provided for classification")
        return [], [], []
    
    if not new_products:
        logger.warning("No new products provided for classification")
        return [], [], []
    
    unique_products = []
    duplicate_check_needed = []
    
    # Create mapping of products to their best matches
    product_matches = {}
    for match in matches:
        query_product = match.get('query_product', '')
        similarity = match.get('similarity_score', 0.0)
        
        if query_product not in product_matches or similarity > product_matches[query_product]['similarity_score']:
            product_matches[query_product] = match
    
    # Process all new products
    excluded_duplicates = []  # New category for perfect matches
    
    for new_product in new_products:
        if new_product in product_matches:
            match = product_matches[new_product]
            similarity = match.get('similarity_score', 0.0)
            matched_product = match.get('matched_product', 'ไม่พบ')
        else:
            similarity = 0.0
            matched_product = 'ไม่มีสินค้าคล้าย'
        
        if similarity >= ThresholdConfig.PERFECT_MATCH:  # Perfect match = auto-exclude
            excluded_duplicates.append({
                'สินค้าใหม่': new_product,
                'สินค้าเก่าที่คล้ายที่สุด': matched_product,
                'ความคล้าย_%': f"{(similarity*100):.1f}%",
                'สถานะ': 'ซ้ำ - ไม่นำเข้า',
                'คำแนะนำ': 'สินค้าซ้ำ 95%+ กับสินค้าเก่า ระบบตัดออกอัตโนมัติ',
                'การจัดการ': 'อัตโนมัติ'
            })
            logger.info(f"🚫 Auto-excluded perfect duplicate: {new_product[:50]}... (similarity: {similarity:.3f})")
        elif similarity >= ThresholdConfig.HIGH_SIMILARITY:  # High similarity = likely duplicate
            duplicate_check_needed.append({
                'สินค้าใหม่': new_product,
                'สินค้าเก่าที่คล้ายที่สุด': matched_product,
                'ความคล้าย_%': f"{(similarity*100):.1f}%",
                'สถานะ': 'สงสัยว่าซ้ำ',
                'คำแนะนำ': 'อาจซ้ำกับสินค้าเก่า ตรวจสอบกับผู้เชี่ยวชาญ',
                'การจัดการ': 'ต้องตรวจสอบ'
            })
        elif similarity >= threshold:  # Medium similarity = needs review
            duplicate_check_needed.append({
                'สินค้าใหม่': new_product,
                'สินค้าเก่าที่คล้ายที่สุด': matched_product,
                'ความคล้าย_%': f"{(similarity*100):.1f}%",
                'สถานะ': 'ต้องตรวจสอบเพิ่ม',
                'คำแนะนำ': 'ตรวจสอบกับผู้เชี่ยวชาญก่อนตัดสินใจ',
                'การจัดการ': 'ต้องตรวจสอบ'
            })
        else:  # Low similarity = unique product
            recommendation = ('สามารถนำเข้าได้เลย' if similarity < ThresholdConfig.LOW_SIMILARITY 
                            else 'แตกต่างจากสินค้าเก่า แนะนำให้นำเข้า')
            unique_products.append({
                'สินค้าใหม่': new_product,
                'สินค้าเก่าที่คล้ายที่สุด': matched_product if similarity > 0.1 else 'ไม่มีสินค้าคล้าย',
                'ความแตกต่าง_%': f"{((1-similarity)*100):.1f}%",
                'สถานะ': 'สินค้าใหม่ไม่ซ้ำ',
                'คำแนะนำ': recommendation,
                'การจัดการ': 'อนุมัติ'
            })
    
    logger.info(f"Classification complete: {len(unique_products)} unique, {len(duplicate_check_needed)} need review, {len(excluded_duplicates)} auto-excluded")
    return unique_products, duplicate_check_needed, excluded_duplicates


def calculate_simple_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity using word intersection/union.
    
    Args:
        text1: First text to compare
        text2: Second text to compare
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    words1 = set(str(text1).lower().split())
    words2 = set(str(text2).lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def load_products_from_file(filepath: str, column_name: Optional[str] = None) -> List[str]:
    """
    Load products from CSV or Excel file with smart column detection.
    
    Args:
        filepath: Path to the data file
        column_name: Optional specific column name
        
    Returns:
        List of product names
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported or no products found
    """
    from pathlib import Path
    
    file_path = Path(filepath)
    
    if not file_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์: {filepath}")
    
    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(filepath, encoding='utf-8-sig')
    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"รองรับเฉพาะไฟล์ .csv และ .xlsx")
    
    products = extract_product_names(df, column_name)
    logger.info(f"📦 โหลดสินค้า {len(products)} รายการจากไฟล์ {filepath}")
    return products