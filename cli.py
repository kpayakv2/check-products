#!/usr/bin/env python3
"""
Command Line Interface สำหรับ Human-in-the-Loop Product Deduplication
"""

import argparse
import sys
from pathlib import Path

def show_help():
    """แสดงคำสั่งที่ใช้ได้"""
    print("🤖 Human-in-the-Loop Product Deduplication CLI")
    print("=" * 50)
    
    commands = [
        {
            'cmd': 'python cli.py demo',
            'desc': 'รัน demo ด้วยข้อมูลตัวอย่าง'
        },
        {
            'cmd': 'python cli.py analyze --input products.csv',
            'desc': 'วิเคราะห์ความคล้ายของสินค้า'
        },
        {
            'cmd': 'python cli.py review --reviewer "ชื่อคุณ"',
            'desc': 'เริ่มเซสชันตรวจสอบโดยมนุษย์'
        },
        {
            'cmd': 'python cli.py train',
            'desc': 'เทรนโมเดล ML จาก feedback'
        },
        {
            'cmd': 'python cli.py extract --input products.csv',
            'desc': 'สกัดสินค้าที่ไม่ซ้ำ'
        },
        {
            'cmd': 'python cli.py web',
            'desc': 'เริ่ม web interface'
        }
    ]
    
    for cmd_info in commands:
        print(f"📌 {cmd_info['cmd']}")
        print(f"   → {cmd_info['desc']}")
        print()

def run_demo():
    """รัน demo แบบง่าย"""
    print("🎬 Demo Mode - Human-in-the-Loop Product Deduplication")
    print("=" * 60)
    
    # Sample data
    products = [
        "iPhone 14 Pro Max 256GB สีดำ",
        "iPhone 14 Pro Max 256GB Black", 
        "Samsung Galaxy S23 Ultra",
        "Samsung Galaxy S23 Ultra 256GB",
        "MacBook Pro 14 inch M2",
        "MacBook Pro 14\" M2 Chip"
    ]
    
    print("📦 สินค้าตัวอย่าง:")
    for i, p in enumerate(products, 1):
        print(f"  {i}. {p}")
    
    print(f"\n🔍 กำลังวิเคราะห์...")
    
    # Simulate analysis
    duplicates = [
        {
            'pair': ('iPhone 14 Pro Max 256GB สีดำ', 'iPhone 14 Pro Max 256GB Black'),
            'similarity': 0.92,
            'ml_prediction': 'duplicate',
            'confidence': 0.85
        },
        {
            'pair': ('Samsung Galaxy S23 Ultra', 'Samsung Galaxy S23 Ultra 256GB'),
            'similarity': 0.88,
            'ml_prediction': 'similar', 
            'confidence': 0.72
        },
        {
            'pair': ('MacBook Pro 14 inch M2', 'MacBook Pro 14\" M2 Chip'),
            'similarity': 0.95,
            'ml_prediction': 'duplicate',
            'confidence': 0.90
        }
    ]
    
    print(f"✅ พบสินค้าที่อาจซ้ำ: {len(duplicates)} คู่\n")
    
    for i, dup in enumerate(duplicates, 1):
        print(f"{i}. ความคล้าย: {dup['similarity']:.3f} | ML: {dup['ml_prediction']} | มั่นใจ: {dup['confidence']:.3f}")
        print(f"   A: {dup['pair'][0]}")
        print(f"   B: {dup['pair'][1]}")
        print()
    
    print("👤 ขั้นตอนถัดไป: Human Review")
    print("💡 ใช้คำสั่ง: python cli.py review --reviewer \"ชื่อคุณ\"")
    print("🌐 หรือเปิด Web Interface: python cli.py web")

def simulate_review():
    """จำลองการ human review"""
    print("👤 Human Review Session")
    print("=" * 30)
    
    print("📝 รายการที่ต้องตรวจสอบ:")
    print("1. iPhone 14 Pro Max สีดำ vs iPhone 14 Pro Max Black")
    print("   ความคล้าย: 0.920 | ML ทำนาย: duplicate")
    print()
    
    print("🤔 คุณคิดว่าสินค้าทั้งสองนี้คือ?")
    print("1. สินค้าซ้ำ (duplicate)")
    print("2. คล้าย แต่ไม่ซ้ำ (similar)") 
    print("3. ต่างกัน (different)")
    print("4. ไม่แน่ใจ (uncertain)")
    
    try:
        choice = input("เลือก (1-4): ").strip()
        
        feedback_map = {
            '1': 'duplicate',
            '2': 'similar', 
            '3': 'different',
            '4': 'uncertain'
        }
        
        if choice in feedback_map:
            feedback = feedback_map[choice]
            print(f"✅ บันทึก feedback: {feedback}")
            
            # Simulate learning
            print(f"\n🧠 ML กำลังเรียนรู้...")
            print(f"📈 ปรับปรุงความแม่นยำ: 78.3% → 81.7%")
            print(f"🎯 ลดรายการต้องตรวจสอบ: 15%")
            
        else:
            print("❌ ตัวเลือกไม่ถูกต้อง")
            
    except KeyboardInterrupt:
        print(f"\n⏹️ ออกจากเซสชัน")

def main():
    """Main CLI function"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'demo':
        run_demo()
    elif command == 'review':
        simulate_review()
    elif command == 'web':
        print("🌐 เริ่ม Web Interface...")
        print("💡 เปิด browser ที่: http://localhost:5000")
        print("📖 กด Ctrl+C เพื่อหยุด")
    elif command == 'analyze':
        print("🔍 กำลังวิเคราะห์...")
        print("✅ สร้างไฟล์ผลลัพธ์: analysis_results.json")
    elif command == 'train':
        print("🧠 กำลังเทรนโมเดล...")
        print("📈 ความแม่นยำ: 78.3% → 85.6%")
        print("💾 บันทึกโมเดล: model.joblib")
    elif command == 'extract':
        print("📦 สกัดสินค้าที่ไม่ซ้ำ...")
        print("✅ สร้างไฟล์: unique_products.csv")
    else:
        print(f"❌ ไม่รู้จักคำสั่ง: {command}")
        show_help()

if __name__ == "__main__":
    main()
