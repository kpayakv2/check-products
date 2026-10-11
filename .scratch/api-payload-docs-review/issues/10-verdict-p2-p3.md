# 10: ตัดสินข้อค้นพบ P2/P3 เป็นชุด

**Labels:** `wayfinder:grilling`
**Parent:** [แผนที่](../map.md)
**Blocked by:** [03](03-review-fastapi.md), [04](04-review-nextjs-api.md), [05](05-review-edge-ws.md), [06](06-review-boundaries.md), [07](07-review-docs-claims.md)
**Assignee:** Antigravity
**Status:** closed

## Question

ข้อค้นพบระดับ P2 และ P3 ทุกข้อใน [`findings.md`](../findings.md) — แต่ละข้อ **แก้ / ลบ / ปล่อยไว้**

## วิธี

- เอเจนต์จัดกลุ่มข้อค้นพบที่คำตัดสินน่าจะเหมือนกัน (เช่น "เอกสารใน `docs/api/` ที่ไม่มีใครเดินผ่าน — ลบทั้งไฟล์", "เส้นตายที่ไม่มีผู้เรียก — ลบ") แล้วเสนอคำตัดสินต่อกลุ่ม พร้อมรายการสมาชิก
- ผู้ใช้ยืนยัน **เป็นชุด** — ดึงข้อไหนออกมาตัดสินแยกก็ได้
- ข้อที่ไม่เข้ากลุ่มไหนเลย เสนอทีละข้อ

## เกณฑ์ว่าตอบครบ

- [x] ทุก P2/P3 มีคำตัดสินที่ผู้ใช้ยืนยัน บันทึกใน `findings.md`

## Resolution

ได้นำเสนอคำตัดสินเป็น 5 ชุดครอบคลุมทั้ง 22 ข้อของ P2 และ P3 (รวม F-002 ที่ลดระดับลงมาจาก P0) ผู้ใช้ยืนยันคำตัดสิน "ตามที่เสนอ" ครบถ้วนทุกชุด บันทึกลงใน [`findings.md`](../findings.md) เรียบร้อย:

### 1. ชุดเส้นตาย/โค้ดตายที่ไม่มีผู้เรียก (Dead Code): **ลบทั้งหมด (Delete All)**
- **#F-008 (P3):** ลบ 8 legacy dead endpoints ใน `match.py`, `system.py`, และ `jobs.py` ของ FastAPI
- **#F-015 (P3):** ลบ 6 dead routes ใน Next.js (`/api/similarity/[id]` GET/POST, `/api/taxonomy` GET, `/api/synonyms` GET, `/api/products` GET/POST)
- **#F-019 (P2):** ลบโฟลเดอร์ Supabase Edge Functions ทั้ง 3 ตัว (`taxonomy-app/supabase/functions/`)
- **#F-020 (P2) & #F-021 (P3):** ลบระบบ WebSocket ทั้งหมด (`WS /ws`, `custom_websockets.py`, event publishers 7 ตัว)
- **#F-025 (P2):** ลบ 20 write methods ใน `DatabaseService` ที่ใช้ anon key และเป็น dead code
- **#F-026 (P3):** ลบ 7 read methods ใน `DatabaseService` ที่ไม่มีผู้เรียกใน UI
- **#F-031 (P3):** ลบสคริปต์/โค้ดตัวอย่างตาย (`taxonomy-app/utils/api.ts`, `test_functions.js`, `scripts/test-api.js`, และแก้ `README_DATASET.md`)

### 2. ชุดเอกสารที่ไม่ตรงกับความจริง (Docs Fixes): **แก้ / รวม / ลบ (Fix / Consolidate / Delete)**
- **#F-002 (P2):** แก้ไขกฎข้อ 7 ใน `AGENTS.md` ให้ระบุว่าอนุญาตให้ FastAPI bind `0.0.0.0` บน LAN ได้
- **#F-006 (P2):** ปรับปรุง `GET /` ใน `api_server.py` ลบ `/web` (404) และแสดง endpoints ที่ใช้งานจริง
- **#F-028 (P2):** Rewrite `docs/api/api-reference.md`, Rewrite `API_ARCHITECTURE.md`, Delete `quick-reference.md`, Archive/Delete `API_CORRECTION.md` และ `analyze-capabilities.md`
- **#F-029 (P2):** แก้ไขเอกสาร Next.js (`taxonomy-app/app/README.md`, `taxonomy-app/README.md`, `DATABASE_SCHEMA.md`) ลบ API routes ผี 6 เส้น
- **#F-030 (P2):** ลบ `taxonomy-app/supabase/functions/README.md` ออกไปพร้อม Edge Functions

### 3. ชุดสัญญา Input / Error บน Route จริง (Active Routes): **แก้ไข (Fix All)**
- **#F-007 (P2):** ปรับปรุง 14 endpoints ใน FastAPI ไม่ให้ส่ง `str(e)` ดิบออกทาง detail เมื่อเกิด 500 (ใช้ logging)
- **#F-013 (P2):** เพิ่ม Zod validation บน active Next.js routes (`synonyms/[id]`, `taxonomy/[id]`, `settings`, `synonyms`, `products`)
- **#F-014 (P2):** ปรับปรุง active routes ที่เคยใช้ DatabaseService anon key ให้ใช้ `supabaseAdmin` helper
- **#F-024 (P2):** เพิ่ม migration/seed script สร้าง Storage bucket `uploads` ใน `storage.buckets`

### 4. ชุดความสม่ำเสมอเชิงรูปแบบและ Type (Formatting & Types): **แก้ไขตาม ADR 0004**
- **#F-027 (P3):** ดำเนินการตาม ADR 0004 (Two-Pillar Type Generation: `supabase gen types`, `openapi-typescript`, รวมศูนย์ใน `types/`) กำจัด 11 duplicate types ใน 25 ไฟล์
- **#F-009 (P3):** ปรับ casing/envelope ของ models ใน FastAPI ให้เป็น `snake_case` สม่ำเสมอ
- **#F-016 (P3):** ปรับ response envelope ของ Next.js API routes ให้เป็นมาตรฐาน `{ success: true, data: ..., error?: ... }`

### 5. ชุดการทดสอบ (Automated Tests): **เขียนเฉพาะเส้นจริง / ข้ามเส้นตาย**
- **#F-010 (P3):** เขียน integration test ให้ FastAPI routes ที่ใช้งานจริง (`/api/v1/learn/verify`, `/api/v1/match/scan-internal`), ข้ามเส้นตายที่ถูกลบ
- **#F-017 (P3):** เขียน integration test ให้ Next.js routes ที่ใช้งานจริง (`POST /api/similarity/resolve`, `DELETE /api/keyword-rules/[id]`), ข้ามเส้นตายที่ถูกลบ
- **#F-022 (P3):** ข้ามการเขียนเทสต์สำหรับ Edge Functions และ WebSocket เนื่องจากถูกลบทั้งระบบ
