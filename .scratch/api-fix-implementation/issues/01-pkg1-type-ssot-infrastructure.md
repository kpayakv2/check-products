# 01: Type SSOT Infrastructure (ADR 0004)

**What to build:**
วางโครงสร้างแหล่งความจริงของ Type สองเสาหลักตาม [ADR 0004](../../docs/adr/0004-two-pillar-single-source-of-truth-types.md) โดยเพิ่ม script `npm run gen:types` ใน `package.json`, สร้างคำสั่ง generate `database.types.ts` จาก PostgreSQL (พอร์ต 54325) ผ่าน Supabase CLI และ `fastapi.types.ts` จาก FastAPI `/openapi.json` (พอร์ต 8000) ผ่าน `openapi-typescript` พร้อมสร้างไฟล์รวมศูนย์ Domain/View Types กลางใน `taxonomy-app/types/` เพื่อกำจัด 11 Duplicate Types ใน 25 ไฟล์

**Blocked by:** None (can start immediately)

**Status:** closed

**ข้อค้นพบที่ปิด:** #F-027 (P3)

**ไฟล์ที่เกี่ยวข้อง:**
- `taxonomy-app/package.json`
- `taxonomy-app/types/database.types.ts` (สร้างใหม่)
- `taxonomy-app/types/fastapi.types.ts` (สร้างใหม่)
- `taxonomy-app/types/index.ts` (สร้างใหม่/รวมศูนย์)
- 25 ไฟล์ใน frontend ที่เคยประกาศ type ซ้ำ (`RecheckTab.tsx`, `VerifyTab.tsx`, `AutoLearnTab.tsx`, `StorageImport.tsx`, `pos-export.ts` ฯลฯ)

**เทสต์ที่ต้องรันก่อน/หลัง:**
- `npx tsc --noEmit` ใน `taxonomy-app/` (ต้องไม่มี type errors เพิ่มขึ้น)
- `npx jest --ci` ใน `taxonomy-app/`

- [x] เพิ่มคำสั่ง `gen:types` ใน `taxonomy-app/package.json`
- [x] สร้าง `taxonomy-app/types/database.types.ts` จาก local Postgres 54325
- [x] สร้าง `taxonomy-app/types/fastapi.types.ts` จาก FastAPI 8000 `/openapi.json`
- [x] สร้าง `taxonomy-app/types/index.ts` รวมศูนย์ types: `TaxonomyNode`, `Product`, `KeywordRule`, `ImportRecord`, `PosExportRow`, `TaxonomyNodeLite`, `ExportableProduct`, `ItemIdCandidate`, `ApiResponse`
- [x] ปรับแก้ 25 ไฟล์ใน frontend ให้ import type จาก `@/types` แทนการประกาศ interface ซ้ำเอง
- [x] ตรวจสอบด้วย `npx tsc --noEmit` ผ่านฉลุย (คงเหลือเพียง 2 integration test errors เดิมตาม baseline)
