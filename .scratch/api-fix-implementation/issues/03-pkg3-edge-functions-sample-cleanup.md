# 03: Edge Functions & Dead Samples Cleanup

**What to build:**
กำจัดโค้ดตกค้างและลดความสับสนทางสถาปัตยกรรมในชั้น Supabase Edge Functions (Layer 3) และโค้ดตัวอย่าง: ลบโฟลเดอร์ Supabase Edge Functions ทั้งหมด (`taxonomy-app/supabase/functions/`) เนื่องจากเป็น dead proxy ที่ไม่มี container รันในเครื่องและไม่มีโค้ดใดในโปรเจกต์เรียกใช้งานจริง, ลบเอกสาร `functions/README.md` ที่โฆษณาฟังก์ชันผีและ OpenAI, ลบไฟล์สคริปต์ตัวอย่างตาย (`utils/api.ts`, `test_functions.js`, `scripts/test-api.js`), และปรับปรุงโค้ดตัวอย่างใน `README_DATASET.md`

**Blocked by:** None (can start immediately)

**Status:** closed

**ข้อค้นพบที่ปิด (5 ข้อ):**
- #F-018 (P1)
- #F-019 (P2)
- #F-022 (P3)
- #F-030 (P2)
- #F-031 (P3)

**ไฟล์ที่เกี่ยวข้อง:**
- `taxonomy-app/supabase/functions/` (ลบทั้งโฟลเดอร์: `generate-embeddings-local`, `hybrid-classification-local`, `product-deduplication`, `README.md`)
- `taxonomy-app/supabase/config.toml` (ลบ section `[functions.*]` ที่ไม่ได้ใช้)
- `taxonomy-app/utils/api.ts` (ลบไฟล์)
- `taxonomy-app/test_functions.js` (ลบไฟล์)
- `taxonomy-app/scripts/test-api.js` (ลบไฟล์)
- `taxonomy-app/data/README_DATASET.md` (แก้ตัวอย่างเมธอดที่ไม่มีอยู่จริง)

**เทสต์ที่ต้องรันก่อน/หลัง:**
- `npx jest --ci` ใน `taxonomy-app/`
- `npx tsc --noEmit` ใน `taxonomy-app/`

- [x] ลบโฟลเดอร์ `taxonomy-app/supabase/functions/` ทั้งหมด
- [x] ถอดฟังก์ชันคอนฟิกที่ไม่ใช้ออกจาก `taxonomy-app/supabase/config.toml`
- [x] ลบไฟล์ `taxonomy-app/utils/api.ts`
- [x] ลบไฟล์ `taxonomy-app/test_functions.js` และ `taxonomy-app/scripts/test-api.js`
- [x] ปรับแก้ตัวอย่างโค้ดใน `taxonomy-app/data/README_DATASET.md`
- [x] รัน Jest และ TypeScript check ผ่านฉลุย (286 tests passed, 0 errors ใหม่)
