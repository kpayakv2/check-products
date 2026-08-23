# รายงานตรวจสอบรีโป — 19 สิงหาคม 2026

ตรวจโดยรันเทสต์จริงทั้ง Python และ Frontend, สแกน secret/สิทธิ์การเข้าถึง, ตรวจ CI workflow และโครงสร้างไฟล์ที่ track ใน git

**สถานะรวม:** โครงสร้างโค้ดจัดวางดี ใช้งาน local ได้ แต่ยังไม่พร้อม deploy — มีช่องโหว่ระดับวิกฤต 2 ข้อ และ CI/เทสต์พังอยู่

---

## 🔴 Critical — ความปลอดภัย

### 1. Edge Function `exec-sql` รัน SQL อะไรก็ได้โดยไม่ต้อง auth

- `taxonomy-app/supabase/config.toml:324-325` ตั้ง `verify_jwt = false`
- `taxonomy-app/supabase/functions/exec-sql/index.ts:29-46` สร้าง client ด้วย `SUPABASE_SERVICE_ROLE_KEY` แล้วส่ง `query` จาก request body เข้า `rpc('exec_sql')` ตรงๆ พร้อม CORS `Access-Control-Allow-Origin: '*'`
- `taxonomy-app/supabase/migrations/20250101000006_create_exec_sql_function.sql:18` เป็น `SECURITY DEFINER` ที่ `EXECUTE query_text` ดิบๆ (รับ `query_params` มาแต่ไม่ได้ใช้)

**ผลกระทบ:** ถ้า deploy จริง ใครที่รู้ URL สามารถ `DROP TABLE` หรือดึงข้อมูลทั้งฐานได้ด้วย curl คำสั่งเดียว
**แนวทางแก้:** ลบฟังก์ชันนี้ทิ้ง หรือเปิด `verify_jwt` + จำกัด origin + whitelist คำสั่ง

### 2. API route ทั้ง 13 เส้นไม่มีการตรวจสอบสิทธิ์

ค้น `auth|session|getUser|Authorization` ใน `taxonomy-app/app/api/` ไม่พบสักไฟล์ และไม่มี `middleware.ts`
รวมถึงเส้นที่ทำลายข้อมูลได้ (DELETE taxonomy/synonyms, import/approve) และสองเส้นที่ใช้ service-role key ซึ่ง bypass RLS ทั้งหมด:
- `taxonomy-app/app/api/import/process-local/route.ts:6`
- `taxonomy-app/app/api/import/process-storage/route.ts:6`

**ประเด็นรอง:** `src/api/api_server.py:58` ตั้ง `allow_origins=["*"]` (มีคอมเมนต์ "Configure properly in production" คาไว้)

---

## 🟠 High — CI และเทสต์

### 3. CI แดงตั้งแต่ขั้นติดตั้ง
`requirements.txt:29` ใส่ `sqlite3` ซึ่งเป็น stdlib ไม่มีบน PyPI
ยืนยันแล้ว: `ERROR: No matching distribution found for sqlite3` → `.github/workflows/ci.yml:24` ล้มทุกครั้ง

### 4. Python: 6 failed / 91 passed / 14 skipped
ที่ fail ทั้งหมดคือ integration test ที่ต้องมีเซิร์ฟเวอร์ที่ `localhost:8000` แต่ไม่ได้ mark `live`
ทำให้ `addopts = -m "not live and not db"` ใน `pytest.ini:19` กรองไม่ตก
- `tests/integration/test_api_client.py` (4 tests)
- `tests/integration/test_internal_scan.py`
- `tests/integration/test_taxonomy_import.py:210`

### 5. Frontend: 8 จาก 19 suite fail (20/127 tests)
- `taxonomy-app/jest.config.js:12` สะกดผิดเป็น `moduleNameMapping` (ต้องเป็น `moduleNameMapper`) — jest เตือน "Unknown option" ตอนรัน; บรรทัด 47 สะกดถูกอยู่แล้ว
- integration suite ที่ต้องต่อ Supabase จริงถูกดูดเข้ามารันด้วย config หลัก แล้ว throw ที่ `taxonomy-app/__tests__/setup/database-setup.ts:8` ทั้งที่มี `jest.integration.config.js` แยกอยู่แล้ว → ควรใส่ `testPathIgnorePatterns`
- assertion เพี้ยนจริง เช่น ProcessingStep คาดข้อความ "กำลังประมวลผล" แต่ component render "เกิดข้อผิดพลาด"

### 6. Next.js 14.0.3 เก่า
มี CVE สะสมหลายตัวตั้งแต่ปลายปี 2024 ควรอัปเป็น 14.2.x ขึ้นไปเป็นอย่างต่ำ

---

## 🟡 Medium — สุขอนามัยของรีโป

- **ไฟล์ build/artifact ถูก commit 73 ไฟล์** ใน `coverage/`, `playwright-report/`, `test-results/` รวมวิดีโอ .webm 4.2MB สองไฟล์ และ `tsconfig.tsbuildinfo` บวก `recovered_products.json` 26MB → ควรใส่ .gitignore
- **RLS เปิด `USING (true)`** สำหรับ SELECT บน products/taxonomy/synonyms (`20250924120000_init_hybrid_schema.sql:433-451`) — อาจตั้งใจ แต่คู่กับข้อ 2 แปลว่าข้อมูลสินค้าอ่านได้สาธารณะ
- **README ชี้ไฟล์ที่ไม่มีแล้ว** — `api_server.py` → `src/api/`, `fresh_implementations.py` → `src/core/`, `/supabase` อยู่ใต้ `taxonomy-app/`
- **dependency ตาย** — requirements.txt ประกาศ Flask/Werkzeug แต่ทั้งโปรเจกต์ไม่มี `import flask`
- **เอกสารบวม** — 176 ไฟล์ .md (53 ใน docs/, 25 เป็น docs/reports/ ที่เป็นรายงานครั้งเดียวจบ) เทียบกับโค้ด Python 105 ไฟล์
- **TypeScript 11 errors** อยู่ใน test/e2e ทั้งหมด โค้ด app/components/utils สะอาด 0 error; `taxonomy-app/e2e/real-user-workflows.spec.ts:296` `Cannot find name 'fileInput'` เป็นบั๊กจริง
- **ไฟล์ค้าง untracked** 3,455 บรรทัด: `docs/CONSOLIDATED_SCHEMA.sql`, `taxonomy-app/schema_export.sql` — ตัดสินใจว่าจะ commit หรือ ignore

---

## ✅ ส่วนที่ดี

- ไม่มี secret จริงหลุดใน git (มีแต่ placeholder ในเอกสาร, `.env.local` ถูก ignore ถูกต้อง)
- เทสต์ Python 91 ตัวผ่าน ครอบคลุม text processing / security / algorithm
- โค้ดแอปจริงผ่าน typecheck 0 error
- pytest marker แยก live/db/offline ออกแบบไว้ดี
- โครงสร้าง `src/` แยก api/core/services/utils ชัดเจน

---

## ลำดับที่แนะนำให้แก้

1. ปิด/ลบ `exec-sql` Edge Function
2. แก้ `sqlite3` ใน requirements.txt + typo `moduleNameMapper` → CI/เทสต์เขียว (2 บรรทัด)
3. วางชั้น auth ให้ API routes
4. อัป Next.js
5. ล้างไฟล์ build/data ออกจาก git
