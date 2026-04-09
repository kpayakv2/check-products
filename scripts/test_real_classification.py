import asyncio
import httpx
import json
import os
from dotenv import load_dotenv

# โหลด Environment Variables
load_dotenv('taxonomy-app/.env.local')
supabase_url = os.environ.get("SUPABASE_URL")
anon_key = os.environ.get("SUPABASE_ANON_KEY")
edge_function_url = f"{supabase_url}/functions/v1/hybrid-classification-local"

# รายการสินค้าตัวอย่างจากไฟล์ "ของจริง"
test_products = [
    "คราดมือเสือ 023 คราด",
    "กระติกเหลี่ยม ขนาดกลาง (3 ลิตร) กระติก",
    "ขวดซอส 7015 BG",
    "กรรไกร 5.5\" NO.304 กุหลาบ TSL",
    "เหยือกตวง #9859 พลาสติก PP ขีดแดง"
]

async def test_classification():
    headers = {
        "Authorization": f"Bearer {anon_key}",
        "Content-Type": "application/json"
    }
    
    print("🚀 Testing Real Product Classification...")
    print("="*50)

    async with httpx.AsyncClient() as client:
        for product_name in test_products:
            print(f"📦 Product: {product_name}")
            
            try:
                # เรียก Edge Function
                payload = {
                    "product_name": product_name,
                    "method": "hybrid",
                    "top_k": 3
                }
                
                response = await client.post(edge_function_url, json=payload, headers=headers, timeout=30.0)
                
                if response.status_code == 200:
                    data = response.json()
                    top = data.get('top_suggestion')
                    
                    if top:
                        print(f"✅ Predicted: {top['category_name']} (Confidence: {top['confidence']:.2f})")
                        print(f"🔍 Method: {top['method']} | Matched: {top.get('matched_keyword', 'N/A')}")
                    else:
                        print("❌ No suggestion found.")
                else:
                    print(f"❌ Error {response.status_code}: {response.text}")
                    
            except Exception as e:
                print(f"⚠️ Exception: {e}")
            
            print("-" * 50)

if __name__ == "__main__":
    asyncio.run(test_classification())
