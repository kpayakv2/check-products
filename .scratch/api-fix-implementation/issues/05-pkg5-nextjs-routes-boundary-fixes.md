# 05: Next.js API Routes & Boundary Fixes

**What to build:**
ยกระดับความปลอดภัยและความเสถียรของชั้น Next.js API Routes (Layer 2 & Boundary Layer 4):
1. เพิ่ม `'/api/recheck'` ลงใน `GATED_READ_PATHS` ของ `middleware.ts` เพื่อป้องกันไม่ให้อ่าน unreviewed products ได้โดยไม่ต้อง unlock (#F-011)
2. ลบ 6 dead routes ใน `app/api/` (`/api/similarity/[id]` GET/POST, `/api/taxonomy` GET, `/api/synonyms` GET, `/api/products` GET/POST) (#F-015, #F-014)
3. ใส่ Zod input validation ให้กับ active routes (`synonyms/[id]`, `taxonomy/[id]`, `settings`, `synonyms`, `products`) ป้องกัน server crash 500 เมื่อ UUID หรือ schema ไม่ถูกต้อง (#F-013)
4. ปรับแก้ `POST /api/similarity/resolve` ให้ตรวจจำนวนแถวที่ถูกอัปเดตจริงผ่าน `.select('id')` หรือ helper `updateRow` ป้องกัน false success เมื่อ `remove_id` ไม่มีอยู่จริง (#F-012)
5. ปรับ response envelope ของ Next.js API routes ให้เป็นรูปแบบมาตรฐาน `{ success: true, data: ..., error?: ... }` (#F-016)
6. ย้าย `StorageImport.tsx` จากการ query PostgREST ตรงด้วย anon key (ที่โดน RLS คืน `[]`) ให้มาเรียกผ่าน `GET /api/import/history` (#F-023)

**Blocked by:** 01 (Type SSOT), 04 (DatabaseService Cleanup)

**Status:** closed

**ข้อค้นพบที่ปิด (7 ข้อ):**
- #F-011 (P1)
- #F-012 (P1)
- #F-013 (P2)
- #F-014 (P2)
- #F-015 (P3)
- #F-016 (P3)
- #F-023 (P1)

**ไฟล์ที่เกี่ยวข้อง:**
- `taxonomy-app/middleware.ts`
- `taxonomy-app/app/api/recheck/route.ts`
- `taxonomy-app/app/api/similarity/resolve/route.ts`
- `taxonomy-app/app/api/similarity/[id]/route.ts` (ลบ route/โฟลเดอร์)
- `taxonomy-app/app/api/taxonomy/route.ts`, `taxonomy/[id]/route.ts`
- `taxonomy-app/app/api/synonyms/route.ts`, `synonyms/[id]/route.ts`
- `taxonomy-app/app/api/settings/route.ts`
- `taxonomy-app/app/api/products/route.ts` (ลบ route)
- `taxonomy-app/components/Import/StorageImport.tsx`

**เทสต์ที่ต้องรันก่อน/หลัง:**
- `npx tsc --noEmit` ใน `taxonomy-app/`
- `npx jest --ci` ใน `taxonomy-app/`

- [x] เพิ่ม `'/api/recheck'` ใน `GATED_READ_PATHS` ของ `taxonomy-app/middleware.ts:18`
- [x] ลบ route `/api/similarity/[id]` และ route ที่ไม่ได้ใช้งาน (`/api/products`, `GET /api/taxonomy`, `GET /api/synonyms`)
- [x] ใส่ Zod validation ให้กับ parameters และ body ใน active routes (`synonyms/[id]`, `taxonomy/[id]`, `settings`, `synonyms`)
- [x] ปรับ `resolve/route.ts` ให้ตรวจแถวอัปเดตจริง คืน 404 หากไม่พบสินค้า
- [x] ปรับ response envelope ใน API routes ให้เป็นมาตรฐาน `{ success, data, error? }`
- [x] ปรับ `StorageImport.tsx` ให้ใช้ `fetch('/api/import/history')` แทน anon query
- [x] รัน `tsc` และ `jest` ผ่านฉลุย (286 tests passed, 0 type errors ใหม่)
