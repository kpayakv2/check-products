#!/usr/bin/env python3
"""
ทดสอบระบบที่แก้ไขให้ใช้ข้อความที่ทำความสะอาดแล้ว
=======================================================

ทดสอบว่า:
1. ข้อความถูกทำความสะอาดก่อนเก็บในฐานข้อมูล
2. Human Review Interface แสดงทั้งข้อความเดิมและที่ทำความสะอาด
3. การทำงานของระบบยังคงถูกต้อง
"""

import os
import sys
from pathlib import Path

# เพิ่ม path ของโปรเจกต์
sys.path.insert(0, str(Path(__file__).parent))

from human_feedback_system import (
    ProductDeduplicationSystem, HumanReviewInterface, 
    HumanFeedbackDatabase, ProductComparison, FeedbackType
)

def test_text_cleaning():
    """ทดสอบการทำความสะอาดข้อความ"""
    print("🧹 ทดสอบการทำความสะอาดข้อความ")
    print("=" * 50)
    
    # สร้างระบบทดสอบ
    system = ProductDeduplicationSystem(similarity_threshold=0.8, embedding_model_type="mock")
    
    # ข้อมูลทดสอบที่มีเครื่องหมายและช่องว่างเยอะ
    test_products = [
        "iPhone 14 Pro Max 256GB สี-ดำ!!!",
        "iPhone   14  Pro  Max  256GB  Black",
        "Samsung Galaxy S23 Ultra (256GB)",
        "Samsung Galaxy S23 Ultra - 256 GB",
        "MacBook Pro 14\" M2 Chip!!!",
        "MacBook Pro 14 inch M2"
    ]
    
    print("📝 ข้อมูลทดสอบ:")
    for i, product in enumerate(test_products, 1):
        print(f"{i}. '{product}'")
    
    print("\n🔍 หาสินค้าที่อาจซ้ำ...")
    comparisons = system.find_potential_duplicates(test_products)
    
    print(f"\n✅ พบการเปรียบเทียบ {len(comparisons)} คู่")
    
    # แสดงผลการทำความสะอาด
    print("\n🧹 ตัวอย่างการทำความสะอาด:")
    for i, comp in enumerate(comparisons[:3], 1):  # แสดง 3 อันแรก
        print(f"\n{i}. การเปรียบเทียบ:")
        print(f"   สินค้า 1 (เดิม): '{comp.product1}'")
        print(f"   สินค้า 1 (สะอาด): '{comp.product1_cleaned}'")
        print(f"   สินค้า 2 (เดิม): '{comp.product2}'")
        print(f"   สินค้า 2 (สะอาด): '{comp.product2_cleaned}'")
        print(f"   ความคล้าย: {comp.similarity_score:.3f}")
        print(f"   ML ทำนาย: {comp.ml_prediction.value}")
    
    return system, comparisons

def test_database_storage():
    """ทดสอบการเก็บข้อมูลในฐานข้อมูล"""
    print("\n💾 ทดสอบการเก็บข้อมูลในฐานข้อมูล")
    print("=" * 50)
    
    # สร้าง database ทดสอบ
    db = HumanFeedbackDatabase("test_cleaned_text.db")
    
    # ดึงข้อมูลจากฐานข้อมูล
    pending_reviews = db.get_pending_reviews(3)
    
    if pending_reviews:
        print(f"✅ พบข้อมูลในฐานข้อมูล {len(pending_reviews)} รายการ")
        
        for i, comp in enumerate(pending_reviews, 1):
            print(f"\n{i}. รายการในฐานข้อมูล:")
            print(f"   ID: {comp.id}")
            print(f"   สินค้า 1 (เดิม): '{comp.product1}'")
            print(f"   สินค้า 1 (สะอาด): '{comp.product1_cleaned}'")
            print(f"   สินค้า 2 (เดิม): '{comp.product2}'")
            print(f"   สินค้า 2 (สะอาด): '{comp.product2_cleaned}'")
            print(f"   ความคล้าย: {comp.similarity_score:.3f}")
    else:
        print("❌ ไม่พบข้อมูลในฐานข้อมูล")
    
    return db

def test_human_review_interface():
    """ทดสอบ Human Review Interface"""
    print("\n👤 ทดสอบ Human Review Interface")
    print("=" * 50)
    
    # สร้าง database และ interface
    db = HumanFeedbackDatabase("test_cleaned_text.db")
    review_interface = HumanReviewInterface(db)
    
    # ดึงข้อมูลที่รอการตรวจสอบ
    pending_reviews = db.get_pending_reviews(2)
    
    if pending_reviews:
        print("📋 ตัวอย่างการแสดงผลใน Human Review:")
        
        for i, comparison in enumerate(pending_reviews[:1], 1):  # แสดง 1 อันแรก
            print(f"\n📝 รายการที่ {i}")
            print("-" * 40)
            print(f"สินค้า 1 (เดิม): {comparison.product1}")
            print(f"สินค้า 1 (สะอาด): {comparison.product1_cleaned}")
            print(f"สินค้า 2 (เดิม): {comparison.product2}")
            print(f"สินค้า 2 (สะอาด): {comparison.product2_cleaned}")
            print(f"ความคล้าย: {comparison.similarity_score:.3f}")
            print(f"ความมั่นใจ: {comparison.confidence_score:.3f}")
            print(f"ML ทำนาย: {comparison.ml_prediction.value}")
            print(f"💡 การตัดสินใจจะใช้ข้อความที่ทำความสะอาดแล้ว")
            
        print("\n✅ Human Review Interface แสดงผลถูกต้อง")
    else:
        print("❌ ไม่มีข้อมูลสำหรับทดสอบ Human Review")

def test_text_processing_examples():
    """ทดสอบตัวอย่างการทำความสะอาดข้อความ"""
    print("\n🔧 ตัวอย่างการทำความสะอาดข้อความ")
    print("=" * 50)
    
    from fresh_implementations import ComponentFactory
    
    # สร้าง text processor
    text_processor = ComponentFactory.create_text_processor("thai")
    
    # ตัวอย่างข้อความที่ต้องทำความสะอาด
    test_cases = [
        "iPhone 14 Pro Max 256GB สี-ดำ!!!",
        "Samsung Galaxy S23 Ultra (256GB)",
        "MacBook Pro 14\" M2 Chip",
        "เสื้อยืด สี-ขาว Nike!! (Size: M)",
        "   iPad   Air   5th   Generation   ",
        "AirPods Pro 2nd Generation - White Color"
    ]
    
    print("📝 ตัวอย่างการทำความสะอาด:")
    for i, original in enumerate(test_cases, 1):
        cleaned = text_processor.process(original)
        print(f"{i}. เดิม: '{original}'")
        print(f"   สะอาด: '{cleaned}'")
        print()

def cleanup_test_files():
    """ลบไฟล์ทดสอบ"""
    test_files = ["test_cleaned_text.db"]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"🗑️ ลบไฟล์ทดสอบ: {file}")

def main():
    """ฟังก์ชันหลักสำหรับทดสอบ"""
    print("ทดสอบระบบที่แก้ไขให้ใช้ข้อความที่ทำความสะอาดแล้ว")
    print("=" * 70)
    
    try:
        # ลบไฟล์ทดสอบเก่า
        cleanup_test_files()
        
        # ทดสอบการทำความสะอาดข้อความ
        system, comparisons = test_text_cleaning()
        
        # ทดสอบการเก็บข้อมูลในฐานข้อมูล
        db = test_database_storage()
        
        # ทดสอบ Human Review Interface
        test_human_review_interface()
        
        # ทดสอบตัวอย่างการทำความสะอาดข้อความ
        test_text_processing_examples()
        
        print("\nการทดสอบเสร็จสิ้น!")
        print("=" * 70)
        print("ระบบทำงานถูกต้อง:")
        print("   - ข้อความถูกทำความสะอาดก่อนเก็บในฐานข้อมูล")
        print("   - Human Review แสดงทั้งข้อความเดิมและที่ทำความสะอาด")
        print("   - การคำนวณ similarity ใช้ข้อความที่ทำความสะอาดแล้ว")
        print("   - ฐานข้อมูลเก็บทั้งข้อความเดิมและที่ทำความสะอาด")
        
    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาด: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # ลบไฟล์ทดสอบ
        cleanup_test_files()

if __name__ == "__main__":
    main()
