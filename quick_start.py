#!/usr/bin/env python3
"""
🚀 Quick Start - Human-in-the-Loop Product Deduplication
========================================================

เริ่มใช้งานระบบหาสินค้าซ้ำ + Human Review แบบง่าย ๆ

Usage:
    python quick_start.py
"""

import pandas as pd
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

def quick_demo():
    """Demo ระบบแบบง่าย ๆ"""
    print("🤖 ยินดีต้อนรับสู่ระบบ Human-in-the-Loop Product Deduplication!")
    print("=" * 70)
    
    # ตัวอย่างข้อมูลสำหรับทดสอบ
    sample_products = [
        "iPhone 14 Pro Max 256GB สีดำ",
        "iPhone 14 Pro Max 256GB Black",
        "ไอโฟน 14 โปร แม็กซ์ 256GB ดำ",
        "Samsung Galaxy S23 Ultra",
        "Samsung Galaxy S23 Ultra 256GB",
        "แซมซุง กาแล็กซี่ S23 อัลตร้า",
        "MacBook Pro 14 inch M2",
        "MacBook Pro 14\" M2 Chip",
        "แมคบุ๊ค โปร 14 นิ้ว M2"
    ]
    
    print("📦 ตัวอย่างข้อมูลสินค้า:")
    for i, product in enumerate(sample_products, 1):
        print(f"  {i:2d}. {product}")
    
    print(f"\n🔍 เริ่มวิเคราะห์ความคล้าย...")
    
    # Simulate analysis
    potential_duplicates = [
        {
            'product1': 'iPhone 14 Pro Max 256GB สีดำ',
            'product2': 'iPhone 14 Pro Max 256GB Black',
            'similarity': 0.92,
            'confidence': 0.85,
            'ml_prediction': 'duplicate'
        },
        {
            'product1': 'iPhone 14 Pro Max 256GB สีดำ',
            'product2': 'ไอโฟน 14 โปร แม็กซ์ 256GB ดำ',
            'similarity': 0.88,
            'confidence': 0.72,
            'ml_prediction': 'similar'
        },
        {
            'product1': 'Samsung Galaxy S23 Ultra',
            'product2': 'Samsung Galaxy S23 Ultra 256GB',
            'similarity': 0.95,
            'confidence': 0.90,
            'ml_prediction': 'duplicate'
        },
        {
            'product1': 'MacBook Pro 14 inch M2',
            'product2': 'MacBook Pro 14\" M2 Chip',
            'similarity': 0.89,
            'confidence': 0.78,
            'ml_prediction': 'similar'
        }
    ]
    
    print(f"✅ พบสินค้าที่อาจซ้ำ: {len(potential_duplicates)} คู่")
    print("\n📋 รายละเอียด:")
    
    for i, item in enumerate(potential_duplicates, 1):
        print(f"\n{i}. ความคล้าย: {item['similarity']:.3f} | ความมั่นใจ: {item['confidence']:.3f}")
        print(f"   🤖 ML ทำนาย: {item['ml_prediction']}")
        print(f"   A: {item['product1']}")
        print(f"   B: {item['product2']}")
    
    print(f"\n👤 ขั้นตอนถัดไป: Human Review")
    print("=" * 50)
    
    # Simulate human review process
    print("ให้ผู้เชี่ยวชาญตรวจสอบรายการที่ ML ไม่แน่ใจ:")
    print("• ใช้ Web Interface: http://localhost:8000/web/human_review.html")
    print("• หรือ Command Line: python complete_deduplication_pipeline.py --mode review")
    
    # Show next steps
    print(f"\n🎯 ขั้นตอนทั้งหมด:")
    steps = [
        "1. 📥 Input: โหลดข้อมูลสินค้า",
        "2. 🔍 Analyze: AI วิเคราะห์ความคล้าย", 
        "3. 👤 Review: มนุษย์ตรวจสอบและให้ feedback",
        "4. 🧠 Learn: ML เรียนรู้จาก feedback",
        "5. 📦 Extract: สกัดสินค้าที่ไม่ซ้ำ",
        "6. 🔄 Repeat: ปรับปรุงต่อเนื่อง"
    ]
    
    for step in steps:
        print(f"   {step}")
    
    print(f"\n📚 การใช้งานแบบละเอียด:")
    print("   📖 อ่านเอกสาร: HUMAN_FEEDBACK_README.md")
    print("   🌐 Web Interface: เริ่มด้วย python api_server.py")
    print("   💻 Command Line: python complete_deduplication_pipeline.py --help")

def show_file_examples():
    """แสดงตัวอย่างไฟล์ input"""
    print(f"\n📁 ตัวอย่างรูปแบบไฟล์ Input:")
    print("=" * 40)
    
    # ตัวอย่าง CSV
    print("📄 products.csv:")
    print("```csv")
    print("product_name")
    print("iPhone 14 Pro Max 256GB สีดำ")
    print("Samsung Galaxy S23 Ultra")
    print("MacBook Pro 14 inch M2")
    print("iPad Air 5th Generation")
    print("```")
    
    print(f"\n🎯 คำสั่งวิเคราะห์:")
    print("```bash")
    print("# วิเคราะห์สินค้าซ้ำ")
    print("python complete_deduplication_pipeline.py --input products.csv --mode analyze")
    print("")
    print("# เริ่ม human review")
    print("python complete_deduplication_pipeline.py --input products.csv --mode review --reviewer \"ชื่อของคุณ\"")
    print("")
    print("# สกัดสินค้าที่ไม่ซ้ำ")
    print("python complete_deduplication_pipeline.py --input products.csv --mode extract")
    print("```")

def test_with_real_data():
    """ทดสอบกับข้อมูลจริง"""
    print(f"\n🧪 ทดสอบกับข้อมูลจริง:")
    print("=" * 40)
    
    # ตรวจสอบไฟล์ข้อมูลจริง
    old_file = "input/old_product/cleaned_products.csv"
    new_file = "input/new_product/new_products.csv"
    
    if Path(old_file).exists() and Path(new_file).exists():
        print("✅ พบไฟล์ข้อมูลจริง:")
        print(f"   📂 สินค้าเดิม: {old_file}")
        print(f"   📂 สินค้าใหม่: {new_file}")
        
        # Read sample data
        try:
            old_df = pd.read_csv(old_file, encoding='utf-8-sig')
            new_df = pd.read_csv(new_file, encoding='utf-8-sig')
            
            print(f"   📊 จำนวนสินค้าเดิม: {len(old_df):,} รายการ")
            print(f"   📊 จำนวนสินค้าใหม่: {len(new_df):,} รายการ")
            
            print(f"\n🎯 เริ่มทดสอบ:")
            print("```bash")
            print("# วิเคราะห์ข้อมูลจริง") 
            print("python simple_real_test.py")
            print("")
            print("# หรือใช้ระบบเต็ม")
            print(f"python complete_deduplication_pipeline.py --input {new_file} --reference {old_file} --mode analyze")
            print("```")
            
        except Exception as e:
            print(f"⚠️ ไม่สามารถอ่านไฟล์ได้: {e}")
    else:
        print("📂 ไม่พบไฟล์ข้อมูลจริง")
        print("   💡 คุณสามารถใส่ไฟล์ CSV ของคุณเองใน:")
        print(f"      • input/old_product/your_file.csv")
        print(f"      • input/new_product/your_file.csv")

def main():
    """Main function"""
    print("🚀 ระบบ Human-in-the-Loop Product Deduplication")
    print("💡 เริ่มต้นใช้งานแบบง่าย ๆ")
    print("=" * 70)
    
    # Run demo
    quick_demo()
    
    # Show file examples
    show_file_examples()
    
    # Test with real data
    test_with_real_data()
    
    print(f"\n🎉 พร้อมใช้งานแล้ว!")
    print("💡 เลือกขั้นตอนถัดไปที่ต้องการ:")
    print("   1. 🌐 เริ่ม Web Interface: python api_server.py")
    print("   2. 💻 ใช้ Command Line: python complete_deduplication_pipeline.py --help")
    print("   3. 🧪 ทดสอบข้อมูลจริง: python simple_real_test.py")
    print("   4. 📖 อ่านเอกสาร: HUMAN_FEEDBACK_README.md")

if __name__ == "__main__":
    main()
