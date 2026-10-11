# 08: Documentation Rewrite & Alignment

**What to build:**
ปรับปรุงเอกสารทั่วทั้งรีโปให้ตรงกับความเป็นจริงของระบบ ป้องกันไม่ให้นักพัฒนาหรือเอเจนต์เข้าใจสถาปัตยกรรมผิด:
1. **`AGENTS.md`:** ปรับถ้อยคำในกฎเหล็กข้อ 7 ให้ระบุชัดเจนว่าอนุญาตให้ FastAPI bind `0.0.0.0` บน LAN ได้ และการเชื่อมต่อในเครื่องให้ใช้ `127.0.0.1` ไม่ใช้ `localhost` ตามคำตัดสินของผู้ใช้ (#F-002)
2. **`docs/api/api-reference.md`:** Rewrite ใหม่ให้เป็น Single Source of Truth ของ API docs (แก้ตัวอย่าง payload `POST /api/v1/clean` เป็น string เดี่ยว, เพิ่ม docs สำหรับ `classify` และ `internal_match`, เพิ่มสรุป Next.js API routes) (#F-028)
3. **`docs/api/API_ARCHITECTURE.md`:** Rewrite ให้สะท้อนสถาปัตยกรรมจริง (Next.js rewrites -> FastAPI + Supabase DB) ลบหลักการปลอมเรื่อง Edge Function First (#F-028)
4. **`docs/api/quick-reference.md`:** ลบไฟล์ทิ้ง (เนื่องจากยุบรวมเข้า `api-reference.md` แล้ว) (#F-028)
5. **`docs/api/API_CORRECTION.md` และ `analyze-capabilities.md`:** ย้ายเข้าโฟลเดอร์ `docs/archive_docs/` (#F-028)
6. **`taxonomy-app/app/README.md` และ `taxonomy-app/README.md`:** แก้ไขตาราง API routes ให้ตรงกับ 19 routes ที่ใช้งานจริง และลบ 6 เส้นทางผีออก (#F-029)

**Blocked by:** 02 (FastAPI Cleanup), 03 (Edge Cleanup), 05 (Next.js Routes Fixes)

**Status:** closed

**ข้อค้นพบที่ปิด (3 ข้อ):**
- #F-002 (P2)
- #F-028 (P2)
- #F-029 (P2)

**ไฟล์ที่เกี่ยวข้อง:**
- `AGENTS.md`
- `docs/api/api-reference.md`
- `docs/api/API_ARCHITECTURE.md`
- `docs/api/quick-reference.md` (ลบไฟล์)
- `docs/api/API_CORRECTION.md` (ย้ายไฟล์)
- `docs/api/analyze-capabilities.md` (ย้ายไฟล์)
- `taxonomy-app/app/README.md`
- `taxonomy-app/README.md`

**เทสต์ที่ต้องรันก่อน/หลัง:**
- ตรวจสอบความถูกต้องของ Markdown links และความสอดคล้องกับโค้ดจริง

- [x] แก้ไขกฎเหล็กข้อ 7 ใน `AGENTS.md`
- [x] เขียนใหม่ `docs/api/api-reference.md` ให้เป็น Single Source of Truth
- [x] เขียนใหม่ `docs/api/API_ARCHITECTURE.md` สะท้อน Next.js rewrites -> FastAPI + DB
- [x] ลบ `docs/api/quick-reference.md`
- [x] ย้าย `API_CORRECTION.md` และ `analyze-capabilities.md` เข้า `docs/archive_docs/`
- [x] แก้ไขตาราง API routes ใน `taxonomy-app/app/README.md` และ `taxonomy-app/README.md`
