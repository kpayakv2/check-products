#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import sys
import os

def main():
    try:
        # ตรวจสอบว่าไฟล์ CSV มีอยู่หรือไม่
        csv_path = 'output/matched_products.csv'
        if not os.path.exists(csv_path):
            print(f"ไม่พบไฟล์ {csv_path}")
            return
        
        # อ่านไฟล์ CSV
        print("กำลังอ่านไฟล์ CSV...")
        df = pd.read_csv(csv_path)
        
        print("=== ข้อมูลตัวอย่างจากไฟล์ matched_products.csv ===")
        print(f"จำนวนแถวทั้งหมด: {len(df):,}")
        print(f"คอลัมน์: {list(df.columns)}")
        
        print("\n=== 5 แถวแรก ===")
        print(df.head().to_string())
        
        print("\n=== สถิติของคะแนนความคล้ายคลึง ===")
        print(f"คะแนนสูงสุด: {df['score'].max():.4f}")
        print(f"คะแนนต่ำสุด: {df['score'].min():.4f}")
        print(f"คะแนนเฉลี่ย: {df['score'].mean():.4f}")
        print(f"คะแนนมัธยฐาน: {df['score'].median():.4f}")
        print(f"ส่วนเบี่ยงเบนมาตรฐาน: {df['score'].std():.4f}")
        
        print("\n=== ตัวอย่างสินค้าที่มีความคล้ายคลึงสูงสุด (Top 5) ===")
        top_scores = df.nlargest(5, 'score')
        for idx, row in top_scores.iterrows():
            print(f"คะแนน: {row['score']:.4f}")
            print(f"  สินค้าใหม่: {row['new_product']}")
            print(f"  สินค้าเก่าที่ตรงกัน: {row['matched_old_product']}")
            print()
        
        print("\n=== ตัวอย่างสินค้าที่มีความคล้ายคลึงต่ำสุด (Bottom 5) ===")
        low_scores = df.nsmallest(5, 'score')
        for idx, row in low_scores.iterrows():
            print(f"คะแนน: {row['score']:.4f}")
            print(f"  สินค้าใหม่: {row['new_product']}")
            print(f"  สินค้าเก่าที่ตรงกัน: {row['matched_old_product']}")
            print()
        
        # วิเคราะห์การกระจายของคะแนน
        print("\n=== การกระจายของคะแนน ===")
        score_ranges = [
            (0.9, 1.0, "สูงมาก (0.9-1.0)"),
            (0.8, 0.9, "สูง (0.8-0.9)"),
            (0.7, 0.8, "ปานกลาง-สูง (0.7-0.8)"),
            (0.6, 0.7, "ปานกลาง (0.6-0.7)"),
            (0.5, 0.6, "ต่ำ-ปานกลาง (0.5-0.6)"),
            (0.0, 0.5, "ต่ำ (0.0-0.5)")
        ]
        
        for min_score, max_score, label in score_ranges:
            if min_score == 0.0:
                count = len(df[df['score'] < max_score])
            else:
                count = len(df[(df['score'] >= min_score) & (df['score'] < max_score)])
            percentage = (count / len(df)) * 100
            print(f"{label}: {count:,} รายการ ({percentage:.1f}%)")
        
        # ตรวจสอบสินค้าที่มีคะแนนสูงมาก (อาจเป็นสินค้าซ้ำ)
        print("\n=== สินค้าที่มีคะแนนสูงมาก (อาจเป็นสินค้าซ้ำ) ===")
        very_high_scores = df[df['score'] > 0.95]
        if len(very_high_scores) > 0:
            print(f"พบ {len(very_high_scores)} รายการที่มีคะแนน > 0.95")
            for idx, row in very_high_scores.head(3).iterrows():
                print(f"คะแนน: {row['score']:.4f}")
                print(f"  สินค้าใหม่: {row['new_product']}")
                print(f"  สินค้าเก่า: {row['matched_old_product']}")
                print()
        else:
            print("ไม่พบสินค้าที่มีคะแนน > 0.95")
            
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

