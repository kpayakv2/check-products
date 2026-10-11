# 11: จัดชุดงานแก้พร้อมส่งต่อ

**Labels:** `wayfinder:task`
**Parent:** [แผนที่](../map.md)
**Blocked by:** [08](08-single-type-source.md), [09](09-verdict-p0-p1.md), [10](10-verdict-p2-p3.md)
**Assignee:** Antigravity
**Status:** closed

## Question

ข้อค้นพบทุกข้อที่ได้คำตัดสิน "แก้" หรือ "ลบ" — จัดกลุ่มเป็นชุดงานที่ลงมือได้อย่างไร และเรียงลำดับอย่างไร

ทะเบียนเรียงตามมิติ แต่งานแก้ควรเรียงตามไฟล์/ชั้น (แก้ route เดียวทีเดียวครบทุกข้อ) และงานที่พึ่งคำตอบตั๋ว 08 (เช่น gen type) ต้องมาก่อนงานแก้ payload ทีละจุด

## วิธี

อ่านและทำตาม `.agents/skills/to-tickets/SKILL.md` (ไม่มีใน `.claude/skills/` — อ่านไฟล์ตรง)

## เกณฑ์ว่าเสร็จ

- [x] ทุกข้อค้นพบ "แก้"/"ลบ" อยู่ในงานใดงานหนึ่งพอดีหนึ่งงาน (ตรวจด้วยการนับเทียบ: 30 ข้อพอดี ไม่ตกหล่นและไม่ซ้ำซ้อน)
- [x] แต่ละงานระบุไฟล์ที่แตะ, ข้อค้นพบที่ปิด, เทสต์ที่ต้องรันก่อน/หลัง (ตามกฎเหล็กข้อ 5 ใน AGENTS.md)
- [x] มีลำดับ/เส้นบล็อกระหว่างงาน
- [x] ผู้ใช้ยืนยันชุดงานแล้ว — ถึง Destination ("เหมาะสม พร้อม")

## Resolution

ได้จัดกลุ่มข้อค้นพบ 30 ข้อ (แก้/ลบ) ออกเป็น 8 ชุดงาน (Work Packages) โดยอ้างอิงลำดับความสัมพันธ์ของไฟล์และรอยต่อข้ามระบบ ผู้ใช้ยืนยันความเหมาะสมและอนุมัติแล้ว พร้อมสร้างตั๋วงานสำหรับลงมือทำจริงไว้ที่ [`.scratch/api-fix-implementation/`](../api-fix-implementation/map.md):

| ตั๋วงานแก้ | ชื่อชุดงาน | ข้อค้นพบที่ปิด (รวม 30 ข้อ) | ไฟล์หลักที่แตะ | Blocker |
|---|---|---|---|---|
| [**01**](../api-fix-implementation/issues/01-pkg1-type-ssot-infrastructure.md) | **Type SSOT Infrastructure (ADR 0004)** | #F-027 (1) | `taxonomy-app/package.json`, `taxonomy-app/types/` | None |
| [**02**](../api-fix-implementation/issues/02-pkg2-fastapi-dead-code-error-handling.md) | **FastAPI Dead Code & Errors** | #F-003, #F-004, #F-005, #F-006, #F-007, #F-008, #F-009, #F-020, #F-021 (9) | `src/api/api_server.py`, `src/api/routers/*` | None |
| [**03**](../api-fix-implementation/issues/03-pkg3-edge-functions-sample-cleanup.md) | **Edge Functions & Samples Cleanup** | #F-018, #F-019, #F-022, #F-030, #F-031 (5) | `taxonomy-app/supabase/functions/` (ลบทั้งโฟลเดอร์), `utils/api.ts` | None |
| [**04**](../api-fix-implementation/issues/04-pkg4-databaseservice-dead-code-removal.md) | **DatabaseService Dead Code Removal** | #F-025, #F-026 (2) | `taxonomy-app/utils/supabase.ts` | 01 |
| [**05**](../api-fix-implementation/issues/05-pkg5-nextjs-routes-boundary-fixes.md) | **Next.js API Routes & Boundary Fixes** | #F-011, #F-012, #F-013, #F-014, #F-015, #F-016, #F-023 (7) | `taxonomy-app/middleware.ts`, `app/api/*`, `StorageImport.tsx` | 01, 04 |
| [**06**](../api-fix-implementation/issues/06-pkg6-storage-bucket-setup.md) | **Supabase Storage Bucket Setup** | #F-024 (1) | `taxonomy-app/supabase/migrations/` | None |
| [**07**](../api-fix-implementation/issues/07-pkg7-automated-tests-active-routes.md) | **Automated Tests for Active Routes** | #F-010, #F-017 (2) | `tests/integration/`, `taxonomy-app/__tests__/api/` | 02, 05 |
| [**08**](../api-fix-implementation/issues/08-pkg8-documentation-rewrite.md) | **Documentation Rewrite & Alignment** | #F-002, #F-028, #F-029 (3) | `AGENTS.md`, `docs/api/*`, `taxonomy-app/app/README.md` | 02, 03, 05 |

*หมายเหตุ: ข้อค้นพบ #F-001 ได้รับคำตัดสินจากผู้ใช้ให้ "ปล่อยไว้ (เก็บไว้แก้ทีหลัง)" จึงไม่อยู่ใน 8 ชุดงานแก้ข้างต้น*
