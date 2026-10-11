# 06: รีวิวรอยต่อระหว่างผู้เรียกกับผู้รับ

**Labels:** `wayfinder:research` (AFK — สกิล research ไม่ได้ติดตั้ง ทำ inline)
**Parent:** [แผนที่](../map.md)
**Blocked by:** [01: สำรวจรายการ](01-inventory.md), [02: เกณฑ์ผ่าน](02-pass-criteria.md)
**Assignee:** Antigravity
**Status:** closed

## Question

ทุกคู่ผู้เรียก→ผู้รับในคอลัมน์ "ผู้เรียก" ของ [`matrix.md`](../matrix.md) ส่ง payload ตรงกับที่ผู้รับรอ และจัดการ error ที่ผู้รับคืนมาอย่างถูกต้องหรือไม่ — มิติ 2 ทั้งหมด + มิติ 5 ฝั่งผู้เรียก ครอบ 3 รอยต่อ:

1. **frontend → Next.js `/api/*`** — body ที่ component ส่ง vs ที่ route อ่าน, response ที่ route คืน vs ที่ component ใช้
2. **Next.js → FastAPI** — route ที่ proxy/เรียก `:8000` ส่ง field ชื่อตรงกับ Pydantic model ไหม (`camelCase` vs `snake_case`), field optional/required ตรงกันไหม
3. **frontend → PostgREST ตรง** (ชั้น 4) — column ที่ select/insert มีอยู่จริงใน schema ไหม (ตรวจกับ DB จริงผ่าน MCP `postgres`), การเขียนด้วย anon key ที่ RLS กั้นแบบเงียบ (`200 []` / `204`) — ดูตาราง RLS ใน AGENTS.md หัวข้อ Authentication

มิติ 5 ฝั่งผู้เรียก: ผู้เรียกเช็ก `res.ok` / `error` ก่อนขึ้น toast สำเร็จไหม

## วิธี

- อ่านโค้ดทั้งสองฝั่งของทุกคู่ — ใช้ `socraticode` หาผู้เรียก (path `D:\` ตัวใหญ่)
- ยืนยัน schema จริงด้วย MCP `postgres`
- คู่ที่อ่านอย่างเดียว: ยิงจริงเทียบ response shape กับ type ที่ผู้เรียกประกาศ

## เกณฑ์ว่าเสร็จ

- [x] ทุกแถวที่มีผู้เรียกมีค่าในช่องมิติ 2 (แถวที่ "ไม่พบผู้เรียก" ใส่ `ไม่เกี่ยว` และอ้างข้อค้นพบมิติ 6)
- [x] ทุกข้อค้นพบอยู่ใน [`findings.md`](../findings.md) ตามรูปแบบเดียวกับตั๋ว 03
- [x] มีรายการ type ที่ประกาศซ้ำหลายที่ (ชื่อ + ไฟล์ทุกที่) แยกไว้ท้าย findings ให้ตั๋ว 08 ใช้
- [x] นับได้ว่าชั้น 4 มีกี่จุด และพังแบบเงียบกี่จุด (ป้อนเข้า fog "ย้าย PostgREST ไปหลัง route")

## Resolution

ได้ตรวจสอบรอยต่อทุกจุดในระบบและประเมินครบทุกแถวใน `matrix.md`:

### 1. มิติ 2 ครบทั้ง 103 แถว
- **ชั้น 1 (FastAPI, 24 แถว):** 12 แถวมีผู้เรียกผ่านการตรวจ payload/type ครบถ้วน (`ผ่าน`), 12 แถวไม่มีผู้เรียก (`ไม่เกี่ยว` อ้างอิง #F-008)
- **ชั้น 2 (Next.js `/api/*`, 25 แถว):** 19 แถวที่มีผู้เรียกส่ง payload สอดคล้องกับ schema (`ผ่าน`), 6 แถวไม่มีผู้เรียก (`ไม่เกี่ยว` อ้างอิง #F-015)
- **ชั้น 3 (Edge Functions, 3 แถว):** พบ mismatch รุนแรงใน L3-01 (#F-018) และ L3-03 (#F-019), L3-02 ส่งต่อ payload ถูกต้อง (`ผ่าน`)
- **ชั้น 4 (PostgREST ตรง, 44 แถว):** ตรวจ schema กับ Postgres จริงทุกจุด
- **ชั้น 5 (WebSocket, 7 แถว):** ไม่มี client receiver ใน frontend (`ไม่เกี่ยว` ทั้งหมด อ้างอิง #F-021)

### 2. ข้อค้นพบใหม่ในตั๋ว 06 (บันทึกลง findings.md แล้ว)
- **#F-023 (P1):** `StorageImport.tsx` (L4-06) อ่านตาราง `imports` ด้วย anon key โดน RLS `imports_read` คืน `[]` แบบเงียบ ทำให้หน้าจอไม่แสดงรายการนำเข้าเสมอ
- **#F-024 (P2):** Storage bucket `uploads` ไม่มีอยู่ในตาราง `storage.buckets` ใน DB จริง ทำให้ L4-07 และ L4-08 ดาวน์โหลดไฟล์ล้มเหลว
- **#F-025 (P2):** เมธอดเขียนข้อมูล 20 ตัวใน `DatabaseService` (4b และ 4c) ใช้ anon key ขัดกับ RLS (11 เมธอดล้มเหลวแบบเงียบ คืน 200 `[]` หรือ 204 และ 9 เมธอดโดน 403) ทั้งหมดเป็น dead code ในหน้าเว็บจริง
- **#F-026 (P3):** เมธอดอ่านข้อมูลที่เป็น dead code 7 เมธอดใน `DatabaseService` (L4-10, L4-14, L4-15, L4-23, L4-24, L4-27, L4-33) ไม่มีผู้เรียกจาก UI
- **#F-027 (P3):** Type definitions ประกาศซ้ำกระจัดกระจาย 11 รายการ ใน 25 ไฟล์ทั่ว frontend ขาด Single Source of Truth

### 3. การวิเคราะห์ตัวเลขสำหรับประเด็น Fog ("ย้าย PostgREST ไปหลัง route ทั้งหมดหรือไม่")
- **จำนวนจุดทั้งหมดในชั้น 4:** 44 จุด (4a: 8 จุด, 4b: 14 จุด, 4c: 22 จุด)
- **จุดที่ทำงานจริงในหน้าเว็บ (Active UI):** 16 จุด
  - 13 จุดทำงานได้ปกติและปลอดภัย เพราะเป็นการ SELECT ตารางที่มี RLS แบบเปิด (`qual: true` สำหรับ `anon`) เช่น `products`, `taxonomy_nodes`, `keyword_rules`, `regex_rules`, `synonym_lemmas`
  - 1 จุดพังแบบเงียบ: L4-06 (`StorageImport.tsx` SELECT `imports`)
  - 2 จุดพังเพราะขาด bucket: L4-07, L4-08 (Storage download `uploads`)
- **จำนวนจุดที่พังแบบเงียบทั้งหมดในชั้น 4:** **12 จุด**
  - L4-06 (SELECT `imports`)
  - L4-37 (SELECT `system_settings`)
  - UPDATE/DELETE 10 จุด: L4-26, L4-28, L4-30, L4-31, L4-35, L4-38, L4-40, L4-41, L4-43, L4-44
- **คำแนะนำต่อ Fog:**
  - **ไม่ต้อง** ย้าย SELECT ที่มี RLS `true` ไปหลัง Next.js route ทั้งหมด เพราะการอ่านตรงผ่าน PostgREST ทำงานได้ถูกต้องอยู่แล้ว และการครอบ route จะเพิ่ม server hop โดยไม่จำเป็น
  - **จำเป็นต้องย้ายเฉพาะ L4-06** ใน `StorageImport.tsx` ให้เปลี่ยนไปเรียก `GET /api/import/history` (ซึ่งใช้ service role) เหมือน `ImportHistory.tsx`
  - เมธอดเขียน 20 ตัวและเมธอดอ่านตาย 7 ตัวใน `DatabaseService` **ให้ลบทิ้ง (Delete dead code)** ในตั๋ว 10/11 ไม่ต้องสร้าง API route ใหม่มารองรับเพราะ UI ปัจจุบันไม่ได้ใช้งานอยู่แล้ว

### 4. รายการ Type ซ้ำ
ได้รวบรวมตาราง Type ซ้ำ 11 รายการ บันทึกไว้ท้าย [`findings.md`](../findings.md) เรียบร้อยเพื่อส่งต่อให้ตั๋ว 08 ดำเนินการต่อ
