import os
from supabase import create_client
import json

def count_products():
    url = "http://127.0.0.1:54331"
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    supabase = create_client(url, key)
    
    print("📊 รายงานสรุปจำนวนสินค้าในฐานข้อมูล:")
    print("-" * 40)
    
    try:
        # นับยอดรวม
        res = supabase.table("products").select("id", count="exact", head=True).execute()
        print(f"✅ จำนวนสินค้าทั้งหมด: {res.count if res.count else 0} รายการ")
        
        # นับแยกตามสถานะ
        statuses = ['approved', 'pending', 'pending_review_dedup', 'pending_review_category']
        for status in statuses:
            res = supabase.table("products").select("id", count="exact", head=True).eq("status", status).execute()
            count = res.count if res.count else 0
            print(f"🔹 สถานะ '{status}': {count} รายการ")
            
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {str(e)}")

if __name__ == "__main__":
    count_products()
