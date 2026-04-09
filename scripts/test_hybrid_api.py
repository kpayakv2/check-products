
import sys
import os
import json
import requests
from pprint import pprint

# Fix encoding for Windows terminal
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_classification():
    # Use 127.0.0.1 for win32 as per state snapshot
    url = "http://127.0.0.1:8000/api/classify/category"
    
    test_products = [
        "กรรไกรตัดกิ่งไม้ 8 นิ้ว",
        "ถังน้ำพลาสติก 20 ลิตร สีน้ำเงิน",
        "สว่านไฟฟ้าไร้สาย Makita",
        "จานเซรามิก ลายไทย 10 นิ้ว"
    ]
    
    print(f"🚀 Testing Hybrid Classification (60% Keyword + 40% Embedding)")
    print("-" * 50)
    
    for product in test_products:
        print(f"\n📦 Product: {product}")
        try:
            response = requests.post(
                url, 
                json={"product_name": product, "top_k": 3},
                timeout=10
            )
            
            if response.status_code == 200:
                results = response.json().get("suggestions", [])
                for i, res in enumerate(results):
                    print(f"  {i+1}. {res['category_name']} ({res['category_id'][:8]}...)")
                    print(f"     Confidence: {res['confidence']:.4f} | Method: {res.get('method', 'hybrid')}")
            else:
                print(f"  ❌ Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"  ❌ Failed to connect to API: {e}")

if __name__ == "__main__":
    test_classification()
