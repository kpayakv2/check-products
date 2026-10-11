# 06: Supabase Storage Bucket Setup

**What to build:**
เพิ่ม migration / seed script สำหรับสร้าง Storage Bucket ชื่อ `uploads` ในตาราง `storage.buckets` ของ Supabase พร้อมตั้งค่า RLS policy ให้ service role และ anon client สามารถอ่าน/เขียนตามที่ระบบต้องการ เพื่อแก้ไขปัญหาคำสั่งดาวน์โหลดไฟล์ในหน้า StorageImport และ ImportHistory ล้มเหลว

**Blocked by:** None (can start immediately)

**Status:** closed

**ข้อค้นพบที่ปิด (1 ข้อ):**
- #F-024 (P2)

**ไฟล์ที่เกี่ยวข้อง:**
- `taxonomy-app/supabase/migrations/20261011000000_create_uploads_storage_bucket.sql` (สร้าง migration file ใหม่สำหรับ storage bucket)

**เทสต์ที่ต้องรันก่อน/หลัง:**
- Query ผ่าน MCP `postgres`: `SELECT id, name FROM storage.buckets;` (ต้องมี bucket `uploads`)
- `tests/integration/test_storage_connection.py` ผ่าน

- [x] สร้าง SQL migration เพิ่ม bucket `uploads` ลงใน `storage.buckets`
- [x] รัน migration หรือ apply เข้า local database
- [x] ตรวจสอบว่า `storage.from('uploads')` ใช้งานได้โดยไม่ error (ทดสอบอัปโหลด/ดาวน์โหลด/ลบไฟล์สำเร็จ)
