# 08: ควรมีแหล่งความจริงของ payload type ที่เดียวไหม

**Labels:** `wayfinder:grilling`
**Parent:** [แผนที่](../map.md)
**Blocked by:** [06: รีวิวรอยต่อ](06-review-boundaries.md)
**Assignee:** Antigravity
**Status:** closed

## Question

จากรายการ type ที่ประกาศซ้ำหลายที่และข้อค้นพบมิติ 2 ของตั๋ว 06 — ควรมีแหล่งความจริงของ payload type ที่เดียวหรือไม่ และถ้ามี อยู่ฝั่งไหน:

- (a) ปล่อยแบบเดิม — ประกาศซ้ำแต่ละฝั่ง แก้เฉพาะจุดที่ไม่ตรง
- (b) Pydantic เป็นต้นฉบับ → gen TS จาก `/openapi.json` ของ FastAPI
- (c) TS/zod เป็นต้นฉบับ (ใช้ได้กับ Next↔frontend แต่ไม่ครอบ FastAPI)
- (d) Supabase generated types สำหรับฝั่ง PostgREST + (b) สำหรับฝั่ง FastAPI

ต้องตอบด้วยว่าขอบเขตของ "ที่เดียว" ครอบรอยต่อไหนบ้าง — ทั้ง 3 รอยต่อ หรือเฉพาะ Next↔FastAPI

## ทำไมต้องรอตั๋ว 06

ถ้าตั๋ว 06 พบว่าไม่ตรงกันแค่ 1–2 จุด (a) ก็พอ ถ้าพบเป็นสิบ คำตอบจะต่างไป — ห้ามตอบก่อนเห็นตัวเลข

## เกณฑ์ว่าตอบครบ

- [x] เลือกทางเดียว พร้อมเหตุผลที่อ้างตัวเลขจากตั๋ว 06
- [x] ถ้าเลือกทางที่แก้คืนยาก + มีทางเลือกจริง → เขียน ADR ใน `docs/adr/`
- [x] ระบุผลต่อ fog "กลไกกันสัญญา API เน่าซ้ำ" ใน map
- [x] ผู้ใช้ยืนยันแล้ว

## Resolution

ผ่านการ Grilling รอบ 1 และผู้ใช้ให้คำยืนยันแล้ว ("ตามนั้น"):

1. **การตัดสินใจเลือกทาง (d): สองเสาหลักแบ่งตามหน้าที่จริง (Two-Pillar Single Source of Truth)**
   - **เสา Database Schema (PostgreSQL):** ใช้ PostgreSQL (`127.0.0.1:54325`) เป็นต้นฉบับ สร้าง `taxonomy-app/types/database.types.ts` ด้วย `npx supabase gen types typescript`
   - **เสา AI Inference Services (FastAPI / Pydantic):** ใช้ Pydantic models ใน Python (`src/api/models.py`) เป็นต้นฉบับ สร้าง `taxonomy-app/types/fastapi.types.ts` ด้วย `npx openapi-typescript`
   - **การรวมศูนย์ใน Frontend:** ใช้ `taxonomy-app/types/` เป็นจุดรวมศูนย์ของ Domain/View types (เช่น `types/index.ts`, `types/import.ts`) โดย import จาก 2 เสาข้างต้น กำจัดปัญหา Duplicate Types ทั้ง 11 รายการใน 25 ไฟล์ (#F-027) ได้ทั้งหมด
2. **วงรอบการ Generate:**
   - ใช้คำสั่ง on-demand script `npm run gen:types` ใน `package.json` แล้ว commit ไฟล์ generated types เข้า git โดยไม่ผูกกับ `prebuild` เพื่อป้องกันไม่ให้คำสั่ง build ล่มเมื่อ DB หรือ FastAPI ไม่ได้เปิดอยู่
3. **ขอบเขตการ Refactor (ส่งต่อตั๋ว 11):**
   - แทนที่ 11 Duplicate Types ใน 25 ไฟล์ และผูก interface ของ API calls ให้ตรงกับ generated types ทันที ควบคุม blast radius ไม่แตะต้อง generic components ที่ทำงานปกติอยู่แล้ว
4. **บันทึก ADR:**
   - บันทึกการตัดสินใจลงใน [`docs/adr/0004-two-pillar-single-source-of-truth-types.md`](file:///d:/product_checker/check-products/docs/adr/0004-two-pillar-single-source-of-truth-types.md) เรียบร้อยแล้ว
5. **ข้อสรุปประเด็น Fog ใน `map.md`:**
   - "กลไกกันสัญญา API เน่าซ้ำ": ได้รับคำตอบชัดเจนโดยใช้ Two-Pillar Type Generation + การตรวจสอบ type ผ่าน `tsc --noEmit` ใน CI/test suite เป็นกลไกป้องกันสัญญาหลุดในอนาคต
