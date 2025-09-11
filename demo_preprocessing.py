#!/usr/bin/env python3
"""
Demo: Text Preprocessing Functions
==================================

ทดสอบการทำงานของฟังก์ชันทำความสะอาดข้อความก่อนนำเข้า ML
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Use Fresh Architecture instead of src
from fresh_implementations import ComponentFactory

def main():
    print("🧪 Testing Text Preprocessing Functions")
    print("=" * 60)
    
    # 1. BasicTextProcessor
    print("1️⃣  BasicTextProcessor")
    print("-" * 30)
    basic = ComponentFactory.create_text_processor("basic")
    
    test_cases_basic = [
        "iPhone   14   PRO  MAX!!!",
        "Samsung  Galaxy   S23+",
        "MACBOOK    pro   M2",
    ]
    
    for text in test_cases_basic:
        result = basic.process(text)
        print(f"Input:  '{text}'")
        print(f"Output: '{result}'")
        print()
    
    # 2. ThaiTextProcessor
    print("2️⃣  ThaiTextProcessor")
    print("-" * 30)
    thai = ComponentFactory.create_text_processor("thai")
    
    test_cases_thai = [
        "ไอโฟน ๑๔ โปร แม็กซ์",
        "แซมซุง แกแลกซี่ เอส๒๓",
        "แม็กบุ๊ก โปร เอ็ม๒",
    ]
    
    for text in test_cases_thai:
        result = thai.process(text)
        print(f"Input:  '{text}'")
        print(f"Output: '{result}'")
        print()
    
    # 3. Thai Text Processing (Advanced)
    print("3️⃣  Thai Text Processing (Advanced)")
    print("-" * 30)
    thai_advanced = ComponentFactory.create_text_processor("thai")

    test_cases_product = [
        "แบรนด์ iPhone 14 Pro Max สีแดง 256GB ราคาพิเศษ",
        "ยี่ห้อ Samsung Galaxy S23 สีดำ 128GB โปรโมชั่น",
        "brand MacBook Pro M2 16 นิ้ว สีเงิน ลดราคา",
    ]

    for text in test_cases_product:
        result = thai_advanced.process(text)
        print(f"Input:  '{text}'")
        print(f"Output: '{result}'")
        print()

    # 4. Basic vs Thai Comparison
    print("4️⃣  Basic vs Thai Comparison")
    print("-" * 40)
    
    test_cases_comparison = [
        "แบรนด์ ไอโฟน ๑๔ โปร แม็กซ์ สีแดง ๒๕๖GB ราคาพิเศษ!!!",
        "ยี่ห้อ แซมซุง แกแลกซี่ เอส๒๓ สีดำ ๑๒๘GB โปรโมชั่น",
        "Brand MACBOOK   PRO   M๒ ๑๖ นิ้ว สีเงิน SALE!!!",
    ]
    
    for text in test_cases_comparison:
        basic_result = basic.process(text)
        thai_result = thai.process(text)
        print(f"Input:  '{text}'")
        print(f"Basic:  '{basic_result}'")
        print(f"Thai:   '{thai_result}'")
        print()
    
    print("✅ Demo completed successfully!")

if __name__ == "__main__":
    main()
