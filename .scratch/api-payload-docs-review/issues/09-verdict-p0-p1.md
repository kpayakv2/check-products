# 09: ตัดสินข้อค้นพบ P0/P1 ทีละข้อ

**Labels:** `wayfinder:grilling`
**Parent:** [แผนที่](../map.md)
**Blocked by:** [03](03-review-fastapi.md), [04](04-review-nextjs-api.md), [05](05-review-edge-ws.md), [06](06-review-boundaries.md), [07](07-review-docs-claims.md)
**Assignee:** Antigravity
**Status:** closed

## Question

ข้อค้นพบระดับ P0 และ P1 ทุกข้อใน [`findings.md`](../findings.md) — แต่ละข้อ **แก้ / ลบ / ปล่อยไว้** (ปล่อยไว้ต้องมีเหตุผลที่ผู้ใช้ยอมรับ เช่น "รอโปรเจกต์ auth")

## วิธี

- เอเจนต์เสนอคำตัดสิน + เหตุผล + ขนาดงานแก้โดยประมาณ ทีละข้อ
- ผู้ใช้ยืนยัน/แก้ **ทีละข้อ** (ไม่ใช่เป็นชุด — ตาม map Notes)
- ก่อนเสนอ ให้ตรวจซ้ำว่าข้อค้นพบยังจริงอยู่ (โค้ดอาจเปลี่ยนระหว่างรีวิว) — โดยเฉพาะ P0 ที่ผู้ใช้อาจสั่งแก้นอกแผนที่ไปแล้ว

## เกณฑ์ว่าตอบครบ

- [x] ทุก P0/P1 มีคำตัดสินที่ผู้ใช้ยืนยัน บันทึกใน `findings.md`
- [x] ข้อที่ตัดสินว่า "ลดระดับ" (จริงๆ เป็น P2/P3) ถูกย้ายไปให้ตั๋ว 10

## Resolution

ได้นำเสนอคำตัดสินทีละข้อ พร้อมการตรวจซ้ำของจริงและขนาดงานแก้ ผู้ใช้ให้คำยืนยันครบทั้ง 9 ข้อแล้ว บันทึกลงใน [`findings.md`](../findings.md) เรียบร้อย:

| รหัส | ระดับ | หัวข้อ | สถานะปัจจุบัน | คำตัดสินที่ผู้ใช้ยืนยัน | ส่งต่อไปยัง |
|---|---|---|---|---|---|
| **#F-001** | P0 | อ่านฐานข้อมูลทั้งหมดจากอินเทอร์เน็ตผ่าน `/api/supabase/*` | จริง | **ปล่อยไว้ (เก็บไว้แก้ทีหลัง)** — ผู้ใช้ตัดสินใจคงเดิมไว้ก่อน ไม่จัดเข้าชุดงานแก้เร่งด่วน แต่ต้องระวังการเปิด Cloudflare Tunnel | แผนที่อื่นในอนาคต |
| **#F-002** | P0 → ลดระดับเป็น P2 (เอกสาร) | FastAPI `0.0.0.0` เขียนข้อมูลได้ข้าม `middleware.ts` | จริง | **ปล่อยโค้ดไว้ / แก้กฎข้อ 7 ใน AGENTS.md** — ผู้ใช้ตั้งใจเปิดให้ LAN เข้าถึง FastAPI ตรง จึงไม่แก้โค้ด `api_server.py` แต่ปรับแก้กฎข้อ 7 ใน `AGENTS.md` ให้ระบุว่าอนุญาต bind `0.0.0.0` บน LAN ได้ | ตั๋ว 10 / 11 |
| **#F-003** | P1 | `POST /api/v1/match/batch` re-wrap 400 เป็น 500 | จริง (dead code) | **ลบ (Delete)** — ลบ endpoint ออกจาก `src/api/routers/match.py` | ตั๋ว 11 |
| **#F-004** | P1 | `POST /api/v1/upload/csv` ไม่จำกัดขนาด/เสี่ยง path traversal | จริง (dead code) | **ลบ (Delete)** — ลบ endpoint ออกจาก `src/api/routers/system.py` | ตั๋ว 11 |
| **#F-005** | P1 | `POST /api/v1/match/single` แก้ singleton config เกิด race | จริง (dead code) | **ลบ (Delete)** — ลบ endpoint ออกจาก `src/api/routers/match.py` | ตั๋ว 11 |
| **#F-011** | P1 | `GET /api/recheck` หลุดจาก `GATED_READ_PATHS` ใน `middleware.ts` | จริง | **แก้ (Fix)** — เพิ่ม `'/api/recheck'` ลงใน `GATED_READ_PATHS` ของ `taxonomy-app/middleware.ts:18` | ตั๋ว 11 |
| **#F-012** | P1 | `POST /api/similarity/resolve` ไม่ตรวจ affected rows เกิด false success | จริง | **แก้ (Fix)** — แก้ `resolve/route.ts` ให้ตรวจแถวที่อัปเดตจริงผ่าน `.select('id')` หรือ `updateRow` | ตั๋ว 11 |
| **#F-018** | P1 | Edge Function `generate-embeddings-local` schema ไม่ตรง FastAPI | จริง (dead code) | **ลบ (Delete)** — ลบโฟลเดอร์ `taxonomy-app/supabase/functions/generate-embeddings-local` | ตั๋ว 11 |
| **#F-023** | P1 | `StorageImport.tsx` อ่าน `imports` ด้วย anon key โดน RLS คืน `[]` | จริง | **แก้ (Fix)** — เปลี่ยนจากการ query PostgREST ตรงมาใช้ `fetch('/api/import/history')` ใน `StorageImport.tsx` | ตั๋ว 11 |
