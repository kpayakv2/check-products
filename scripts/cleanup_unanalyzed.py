import os
from supabase import create_client, Client
from dotenv import load_dotenv

# โหลด config
load_dotenv('taxonomy-app/.env.local')

url = os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") # ใช้ Service Role สำหรับลบ

if not url or not key:
    print("❌ ไม่พบ Supabase credentials ใน .env.local")
    exit(1)

supabase: Client = create_client(url, key)

def cleanup():
    print("🚀 เริ่มการลบสินค้าที่ยังไม่ผ่านการวิเคราะห์ (1,620 รายการ)...")
    
    # ดำเนินการลบ
    # เงื่อนไข: confidence_score == 0 หรือ embedding IS NULL
    try:
        response = supabase.table("products") \
            .delete() \
            .or_("confidence_score.eq.0,embedding.is.null") \
            .execute()
        
        count = len(response.data)
        print(f"✅ ลบสำเร็จทั้งหมด: {count} รายการ")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {str(e)}")

if __name__ == "__main__":
    cleanup()
