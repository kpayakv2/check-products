#!/usr/bin/env python3
"""ทดสอบการเรียกใช้โมเดลใน web_server.py"""

import pandas as pd
import os
import sys

# Test data
test_old_products = [
    "น้ำดื่มเซเว่น 600ml", 
    "นมสดเมจิ 180ml",
    "กาแฟลาเต้ 200ml",
    "โค้กโซดา 325ml",
    "ขนมปังโฮลวีท"
]

test_new_products = [
    "น้ำดื่มแบรนด์เซเว่น 600มล",  # Similar to first
    "นมสดเมจิคิง 180มล",          # Similar to second  
    "กาแฟเอสเปรสโซ่ 250ml",       # Different
    "เป๊ปซี่โค้ล 330ml",           # Similar to fourth
    "ขนมปังข้าวโอ๊ต"              # Similar to fifth
]

def test_model_execution():
    print("🔍 ทดสอบการเรียกใช้โมเดล...")
    
    try:
        # Import web_server modules
        from web_server import find_unique_new_products, app_state, WEB_CONFIG
        from main import create_enhanced_pipeline
        
        print("✅ Imports successful")
        
        # Create mock dataframes
        old_df = pd.DataFrame({'รายการ': test_old_products})
        new_df = pd.DataFrame({'รายการ': test_new_products})
        
        # Set app state (app_state is now an AppState object, not dict)
        app_state.old_products_data = old_df
        app_state.new_products_data = new_df
        
        print(f"📊 Old products: {len(old_df)}")
        print(f"📊 New products: {len(new_df)}")
        print(f"⚙️ Config: {WEB_CONFIG}")
        
        # Test model creation directly
        print("\n🔧 ทดสอบการสร้าง pipeline...")
        
        class MockArgs:
            def __init__(self, web_config):
                self.model = web_config.get('model_type', 'tfidf')
                self.similarity = web_config.get('similarity_method', 'cosine')
                self.threshold = web_config.get('threshold', 0.6)
                self.top_k = web_config.get('top_k', 10)
                self.enhanced = True
                self.track_performance = True
                self.include_metadata = True
                self.confidence_scores = True
                self.export_report = False
        
        args = MockArgs(WEB_CONFIG)
        pipeline, system_config = create_enhanced_pipeline(args)
        matcher = pipeline.product_matcher
        
        print(f"✅ Pipeline created: {type(pipeline)}")
        print(f"✅ Matcher created: {type(matcher)}")
        print(f"📐 Threshold: {system_config.similarity_threshold}")
        
        # Test model execution
        print("\n🚀 ทดสอบการประมวลผล...")
        
        matches = matcher.find_matches(
            query_products=test_new_products,
            reference_products=test_old_products
        )
        
        print(f"📊 Matches found: {len(matches)}")
        
        for i, match in enumerate(matches):
            print(f"  {i+1}. {match.get('query_product', 'N/A')} -> {match.get('matched_product', 'N/A')} ({match.get('similarity_score', 0):.3f})")
        
        # Test full function
        print("\n🧪 ทดสอบฟังก์ชันเต็ม...")
        unique_products, duplicate_check_needed = find_unique_new_products(app_state)
        
        print(f"✅ Unique products: {len(unique_products)}")
        print(f"⚠️ Need review: {len(duplicate_check_needed)}")
        
        if unique_products:
            print("\n🎯 ตัวอย่างสินค้าไม่ซ้ำ:")
            for item in unique_products[:3]:
                print(f"  - {item.get('สินค้าใหม่', '')}")
        
        if duplicate_check_needed:
            print("\n🔍 ตัวอย่างสินค้าต้องตรวจสอบ:")
            for item in duplicate_check_needed[:3]:
                print(f"  - {item.get('สินค้าใหม่', '')} -> {item.get('สินค้าเก่าที่คล้ายที่สุด', '')} ({item.get('ความคล้าย_%', '')})")
        
        print("\n🎉 การทดสอบเสร็จสิ้น - โมเดลทำงานปกติ!")
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_execution()