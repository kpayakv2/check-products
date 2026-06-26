from supabase import create_client
import os
import tempfile

def test_storage():
    url = os.getenv("SUPABASE_URL", "http://127.0.0.1:54331")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not key:
        print("⚠️ Warning: SUPABASE_SERVICE_ROLE_KEY is not set.")
    supabase = create_client(url, key)
    
    print(f"📡 กำลังทดสอบเชื่อมต่อ Supabase: {url}")
    
    try:
        # 1. เช็คว่าถัง uploads มีอยู่จริงไหม
        buckets = supabase.storage.list_buckets()
        bucket_names = [b.name for b in buckets]
        print(f"📦 รายชื่อถังเก็บข้อมูล: {bucket_names}")
        
        if 'uploads' not in bucket_names:
            print("❌ ไม่พบถังเก็บข้อมูลชื่อ 'uploads'!")
            return

        # 2. ทดลองอัปโหลดไฟล์จำลอง
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"Test upload from Phayak")
            tmp_path = tmp.name
            
        print(f"📤 กำลังทดลองอัปโหลดไฟล์ทดสอบ...")
        res = supabase.storage.from_('uploads').upload(
            path='test_from_phayak.txt',
            file=tmp_path,
            file_options={"upsert": "true"}
        )
        print(f"✅ อัปโหลดสำเร็จ! ผลลัพธ์: {res}")
        
        # 3. ทดลองลบไฟล์ทดสอบ
        supabase.storage.from_('uploads').remove(['test_from_phayak.txt'])
        print("🗑️ ลบไฟล์ทดสอบเรียบร้อย")
        
        os.unlink(tmp_path)
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {str(e)}")

if __name__ == "__main__":
    test_storage()
