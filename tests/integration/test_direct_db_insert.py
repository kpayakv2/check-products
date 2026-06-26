import os
from supabase import create_client
import uuid

def test_direct_insert():
    url = "http://127.0.0.1:54331"
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    supabase = create_client(url, key)
    
    print("🚀 กำลังทดสอบการบันทึกข้อมูลเข้าตาราง 'products' โดยตรง...")
    
    test_product = {
        "id": str(uuid.uuid4()),
        "name_th": "สินค้าทดสอบระบบบันทึก",
        "sku": f"TEST-{uuid.uuid4().hex[:8]}",
        "status": "pending_review_category",
        "confidence_score": 0.5,
        "metadata": {"test": True, "source": "direct_python_test"}
    }
    
    try:
        res = supabase.table("products").insert(test_product).execute()
        print(f"✅ บันทึกสำเร็จ! ID: {res.data[0]['id']}")
        
        # ลองลบทิ้งเพื่อความสะอาด
        supabase.table("products").delete().eq("id", test_product["id"]).execute()
        print("🗑️ ลบข้อมูลทดสอบเรียบร้อย")
        
    except Exception as e:
        print(f"❌ บันทึกไม่สำเร็จ! สาเหตุ: {str(e)}")

if __name__ == "__main__":
    test_direct_insert()
