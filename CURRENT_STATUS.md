# 📊 Current Project Status (Compact Context)
*Last Updated: 2026-04-09 (By Phayak)*

## 🎯 Current Focus
- ระบบ Import CSV ทำงานได้เสถียรบน Local Development
- ชื่อไฟล์ภาษาไทยถูกรักษาไว้ครบถ้วนตั้งแต่ Upload จนถึง UI

## ✅ Recently Completed (Session 9 เม.ย. 2569)

### 🐛 Bug Fixes — Import Module
- [x] **Bug #1 — Storage Invalid Key (400):** ไฟล์ชื่อไทยถูก reject โดย Supabase Storage
  - แก้: เพิ่ม `sanitizeFileName()` ใน `ProcessingStep.tsx` แปลงชื่อเป็น ASCII ก่อน upload
  - ชื่อไทยต้นฉบับยังถูกบันทึกแยกใน `imports.file_name`
- [x] **Bug #2 — RLS 401 Unauthorized:** ตาราง `imports` มี RLS เปิดอยู่แต่ไม่มี policy ใดเลย
  - แก้: รัน SQL เพิ่ม 4 policies (INSERT/SELECT/UPDATE/DELETE) ผ่าน Docker psql
  - ไฟล์ fix: `taxonomy-app/fix_imports_rls.sql`

### ✨ New Features — Import Module
- [x] **บันทึกข้อมูลไฟล์ครบถ้วน:** `ProcessingStep.tsx` บันทึก `file_name` (ชื่อไทย), `file_size`, `file_type`, `metadata.storage_path`
- [x] **ImportHistory component:** `components/Import/ImportHistory.tsx` (ใหม่)
  - แสดงประวัติ 20 รายการล่าสุดบนหน้า `/import`
  - แสดงชื่อไทยต้นฉบับ, สถานะ, ขนาดไฟล์, จำนวนแถว
  - คลิกขยายดู storage path, ปุ่มดาวน์โหลดชื่อไทย
- [x] **StorageImport เขียนใหม่:** `components/Import/StorageImport.tsx`
  - เปลี่ยนแหล่งข้อมูลจาก `products/` folder → `imports` table ใน DB
  - แสดงชื่อไทยต้นฉบับแทนชื่อ sanitized
  - เพิ่มช่องค้นหาภาษาไทย
  - ส่ง File object พร้อมชื่อไทยกลับเข้า Wizard

## 🚧 In Progress
- [ ] **Feedback Loop Automation:** เชื่อมต่อปุ่มยืนยันใน UI เข้ากับระบบการสร้าง Keyword Rules อัตโนมัติ
- [ ] **Data Sync Optimization:** ปรับปรุงความเร็วในการยิงข้อมูลเข้า Supabase (Batch Insert)

## 📋 Next Steps
1. ทดสอบ Upload ไฟล์ใหม่ (ชื่อไทย) ครบ flow → ตรวจสอบใน "ประวัติการนำเข้า"
2. ทดสอบ "ใช้ไฟล์จาก Storage" ใน Wizard หลัง Upload ใหม่เสร็จ
3. ต่อยอด Feedback Loop — เชื่อมปุ่ม Verify กับ Keyword Rule auto-create

## ⚠️ ข้อควรระวัง
- Record เก่าใน `imports` ที่มี `metadata = {}` จะไม่แสดงในหน้า "ใช้ไฟล์จาก Storage" (ไม่มี storage_path)
- `fix_imports_rls.sql` เป็น **local dev only** — Production ต้องใช้ policy ที่จำกัด role

## 💡 System State Summary
- **Frontend (Next.js):** http://127.0.0.1:3000 (Running — port 3000)
- **Backend (FastAPI):** http://127.0.0.1:8000 (Running — Python .venv)
- **Supabase Local:** http://127.0.0.1:54331 | DB: postgresql://postgres:postgres@127.0.0.1:54325/postgres
- **Supabase Studio:** http://127.0.0.1:54323
- **Import Pipeline:** Upload → Storage `imports/{ts}-{ascii}.csv` + DB record (ชื่อไทย) → AI Process → Review
- **AI Thresholds:** Dedup >= 0.95 (Auto), Classify >= 0.80 (Auto)
- **Hybrid Algorithm:** Keyword 60% + Embedding 40% → Accuracy ~72%

---
