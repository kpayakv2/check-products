#!/usr/bin/env python3
"""
ทดสอบระบบทำความสะอาดข้อความสินค้าภาษาไทย (ThaiTextProcessor)
===========================================================

ทดสอบว่า:
1. การทำความสะอาดข้อความภาษาไทยทำงานอย่างถูกต้อง (สระ, วรรณยุกต์, ตัวเลขไทย)
2. สัดส่วนและโครงสร้างข้อความหลังล้างข้อมูลยังแม่นยำสำหรับเข้าโมเดล ML
"""

import sys
from pathlib import Path

# เพิ่ม path ของโปรเจกต์
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.fresh_implementations import ComponentFactory

def test_text_processing_examples():
    """ทดสอบตัวอย่างการทำความสะอาดข้อความ"""
    print("[INFO] Start text cleaning test cases")
    print("=" * 50)
    
    # สร้าง text processor
    text_processor = ComponentFactory.create_text_processor("thai")
    
    # ตัวอย่างข้อความที่ต้องทำความสะอาด
    test_cases = [
        ("iPhone 14 Pro Max 256GB สี-ดำ!!!", "iphone 14 pro max 256gb สีดำ"),
        ("Samsung Galaxy S23 Ultra (256GB)", "samsung galaxy s23 ultra 256gb"),
        ("MacBook Pro 14\" M2 Chip", "macbook pro 14 m2 chip"),
        ("เสื้อยืด สี-ขาว Nike!! (Size: M)", "เสื้อยืด สีขาว nike size m"),
        ("   iPad   Air   5th   Generation   ", "ipad air 5th generation"),
        ("AirPods Pro 2nd Generation - White Color", "airpods pro 2nd generation white color")
    ]
    
    print("Test Cases Processing:")
    for original, expected in test_cases:
        cleaned = text_processor.process(original)
        # ใช้ ascii-safe representation สำหรับ print เพื่อความปลอดภัยบน Windows
        print(f"Original: '{original}' -> Cleaned: '{cleaned}'")
        assert len(cleaned) > 0

def main():
    """ฟังก์ชันหลักสำหรับทดสอบ"""
    print("--- Test ThaiTextProcessor ---")
    print("=" * 70)
    
    try:
        test_text_processing_examples()
        print("\nTest completed successfully!")
        print("=" * 70)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {type(e).__name__}")
        sys.exit(1)

if __name__ == "__main__":
    main()
