# 04: รีวิวชั้น Next.js `/api/*`

**Labels:** `wayfinder:research` (AFK — สกิล research ไม่ได้ติดตั้ง ทำ inline)
**Parent:** [แผนที่](../map.md)
**Blocked by:** [01: สำรวจรายการ](01-inventory.md), [02: เกณฑ์ผ่าน](02-pass-criteria.md)
**Assignee:** Antigravity
**Status:** closed

## Question

ทุกแถวชั้น 2 (Next.js `taxonomy-app/app/api/**/route.ts`) ใน [`matrix.md`](../matrix.md) ผ่านหรือไม่ผ่านมิติ 1, 3, 4, 5, 6, 7, 8 ตามเกณฑ์ของตั๋ว 02

จุดที่ต้องดูเป็นพิเศษ:
- **มิติ 4:** `middleware.ts` ครอบทุก non-GET จริงไหม และ GET เส้นไหนบ้างที่ใช้ `supabaseAdmin` (bypass RLS) โดยไม่ผ่าน middleware
- **มิติ 5:** route ที่เรียก Supabase แล้วไม่เช็ก `error` หรือไม่เช็กว่าแถวที่กระทบ > 0 — นี่คือที่มาของ toast สำเร็จปลอมเมื่อ 28–30 ส.ค.
- **มิติ 6:** route ที่ไม่มีหน้าไหนเรียก (ดูคอลัมน์ผู้เรียกจากตั๋ว 01)

## วิธี

- GET: ยิงจริงกับ dev server ในเครื่อง (`localhost:3000`)
- non-GET: อ่านโค้ด + jest/Playwright ที่มีอยู่เท่านั้น **ห้ามยิง**

## เกณฑ์ว่าเสร็จ

- [x] ทุกช่องมิติของแถวชั้น 2 มีค่า (กรอกครบทั้ง 25 แถวใน `matrix.md` โดยมิติ 2 กำกับ "รอตั๋ว 06")
- [x] ทุกข้อค้นพบอยู่ใน [`findings.md`](../findings.md) ตามรูปแบบเดียวกับตั๋ว 03 (บันทึก F-011 ถึง F-017 ครบถ้วนพร้อมหลักฐานและคำตัดสินที่เสนอ)
- [x] P0 ถูกรายงานทันที (ไม่พบ P0 ตัวใหม่ในชั้น Next.js — ร้ายแรงสุดคือ P1 F-011 และ F-012)

## Resolution

- **ยิงจริง endpoint อ่านอย่างเดียว (`localhost:3000`):**
  - ทดสอบยิงจริง 8 เส้น (`/api/recheck`, `/api/pos-export`, `/api/settings`, `/api/import/history`, `/api/similarity/dummy-id`, `/api/taxonomy`, `/api/synonyms`, `/api/products`)
  - พบข้อบกพร่องชัดเจนจากการยิงจริง:
    - `GET /api/recheck` ตอบ 200 พร้อมข้อมูล unreviewed แม้ไม่ได้ส่ง session cookie (พิสูจน์ F-011)
    - `GET /api/similarity/dummy-id` แครชเป็น 500 เนื่องจาก UUID ผิด ไม่มีการ validate parameter (พิสูจน์ F-013)
- **ข้อค้นพบใหม่ (F-011 ถึง F-017):**
  - **F-011 (P1):** `GET /api/recheck` ใช้ `supabaseAdmin` อ่านข้อมูล แต่ตกหล่นจาก `GATED_READ_PATHS` ใน `middleware.ts` ทำให้บุคคลภายนอกดึงข้อมูล unreviewed ทั้งหมดได้โดยไม่ต้อง unlock
  - **F-012 (P1):** `POST /api/similarity/resolve` สั่ง update สถานะสินค้าโดยไม่ตรวจ affected rows คืน 200 `{success: true}` เสมอแม้ไม่มีสินค้า id นั้นอยู่จริง (สาเหตุของ toast สำเร็จปลอม)
  - **F-013 (P2):** ขาด Zod schema validation ใน 6 route (`synonyms/[id]` PUT, `taxonomy/[id]` PUT, `settings` PUT, `synonyms` POST, `similarity/[id]` GET/POST, `products` POST) ส่งผลให้ invalid params หลุดไป crash 500 ที่ระดับฐานข้อมูล
  - **F-014 (P2):** 6 route ตกค้างยังเรียกใช้ `DatabaseService` (anon key) ขัดกับ RLS model ทำให้ล้มเหลวเงียบๆ หรือพัง
  - **F-015 (P3):** พบ 6 endpoint เป็นเส้นตายที่ไม่มีหน้า UI หรือ component ใดเรียกใช้งาน (L2-12, L2-13, L2-14, L2-16, L2-19, L2-20)
  - **F-016 (P3):** Envelope ไม่สม่ำเสมอใน 5 route (คืน metadata, action, หรือ summary ที่รูทแทนที่จะอยู่ในฟิลด์ `data`)
  - **F-017 (P3):** ขาด automated tests ชี้ตรงถึง 4 endpoint (`similarity/resolve`, `similarity/[id]` GET/POST, `keyword-rules/[id]` DELETE)
- **อัปเดตไฟล์:** กรอกผลประเมินครบทุกมิติใน [`matrix.md`](../matrix.md) (L2-01 ถึง L2-25) และบันทึกข้อค้นพบทั้งหมดลงใน [`findings.md`](../findings.md) เรียบร้อย

