#!/usr/bin/env python3
"""
Demo: Text Preprocessing Functions
==================================

ทดสอบการทำงานของฟังก์ชันทำความสะอาดข้อความก่อนนำเข้า ML
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.preprocessing import *

def main():
    print("🧪 Testing Text Preprocessing Functions")
    print("=" * 60)
    
    # 1. BasicTextPreprocessor
    print("1️⃣  BasicTextPreprocessor")
    print("-" * 30)
    basic = BasicTextPreprocessor()
    
    test_cases_basic = [
        "iPhone   14   PRO  MAX!!!",
        "Samsung  Galaxy   S23+",
        "MACBOOK    pro   M2",
    ]
    
    for text in test_cases_basic:
        result = basic.preprocess(text)
        print(f"Input:  '{text}'")
        print(f"Output: '{result}'")
        print()
    
    # 2. ThaiTextPreprocessor
    print("2️⃣  ThaiTextPreprocessor")
    print("-" * 30)
    thai = ThaiTextPreprocessor()
    
    test_cases_thai = [
        "ไอโฟน ๑๔ โปร แม็กซ์",
        "แซมซุง แกแลกซี่ เอส๒๓",
        "แม็กบุ๊ก โปร เอ็ม๒",
    ]
    
    for text in test_cases_thai:
        result = thai.preprocess(text)
        print(f"Input:  '{text}'")
        print(f"Output: '{result}'")
        print()
    
    # 3. ProductTextPreprocessor
    print("3️⃣  ProductTextPreprocessor")
    print("-" * 30)
    product = ProductTextPreprocessor()
    
    test_cases_product = [
        "แบรนด์ iPhone 14 Pro Max สีแดง 256GB ราคาพิเศษ",
        "ยี่ห้อ Samsung Galaxy S23 สีดำ 128GB โปรโมชั่น",
        "brand MacBook Pro M2 16 นิ้ว สีเงิน ลดราคา",
    ]
    
    for text in test_cases_product:
        result = product.preprocess(text)
        print(f"Input:  '{text}'")
        print(f"Output: '{result}'")
        print()
    
    # 4. ChainedTextPreprocessor (รวมทั้งหมด)
    print("4️⃣  ChainedTextPreprocessor (รวมทั้งหมด)")
    print("-" * 40)
    chained = create_default_thai_product_preprocessor()
    
    test_cases_chained = [
        "แบรนด์ ไอโฟน ๑๔ โปร แม็กซ์ สีแดง ๒๕๖GB ราคาพิเศษ!!!",
        "ยี่ห้อ แซมซุง แกแลกซี่ เอส๒๓ สีดำ ๑๒๘GB โปรโมชั่น",
        "Brand MACBOOK   PRO   M๒ ๑๖ นิ้ว สีเงิน SALE!!!",
    ]
    
    for text in test_cases_chained:
        result = chained.preprocess(text)
        print(f"Input:  '{text}'")
        print(f"Output: '{result}'")
        print()
    
    # 5. การเปรียบเทียบผลลัพธ์
    print("5️⃣  เปรียบเทียบผลลัพธ์")
    print("-" * 30)
    sample_text = "แบรนด์ ไอโฟน ๑๔ โปร แม็กซ์ สีแดง ราคาพิเศษ!!!"
    
    print(f"Original: '{sample_text}'")
    print(f"Basic:    '{basic.preprocess(sample_text)}'")
    print(f"Thai:     '{thai.preprocess(sample_text)}'")
    print(f"Product:  '{product.preprocess(sample_text)}'")
    print(f"Chained:  '{chained.preprocess(sample_text)}'")
    print()
    
    # 6. Configuration
    print("6️⃣  การตั้งค่า (Configuration)")
    print("-" * 30)
    print("Basic Config:", basic.get_config())
    print("Thai Config:", thai.get_config())
    print("Product Config:", product.get_config())
    print("Chained Config:", chained.get_config())

if __name__ == "__main__":
    main()
