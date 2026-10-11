# แผนที่: ดำเนินการแก้ไขและชำระล้าง API, Payload Types และเอกสาร (Implementation Map)

**Labels:** `wayfinder:map`
**Parent:** [แผนที่รีวิวต้นทาง](../api-payload-docs-review/map.md)
**ตั๋วทั้งหมด:** `./issues/` (8 ตั๋ว)
- [01: Type SSOT Infrastructure (ADR 0004)](issues/01-pkg1-type-ssot-infrastructure.md) — Blocked by: None
- [02: FastAPI Dead Code, Error Sanitization & Models Cleanup](issues/02-pkg2-fastapi-dead-code-error-handling.md) — Blocked by: None
- [03: Edge Functions & Dead Samples Cleanup](issues/03-pkg3-edge-functions-sample-cleanup.md) — Blocked by: None
- [04: DatabaseService Dead Code Removal](issues/04-pkg4-databaseservice-dead-code-removal.md) — Blocked by: 01
- [05: Next.js API Routes & Boundary Fixes](issues/05-pkg5-nextjs-routes-boundary-fixes.md) — Blocked by: 01, 04
- [06: Supabase Storage Bucket Setup](issues/06-pkg6-storage-bucket-setup.md) — Blocked by: None
- [07: Automated Integration Tests for Active Routes](issues/07-pkg7-automated-tests-active-routes.md) — Blocked by: 02, 05
- [08: Documentation Rewrite & Alignment](issues/08-pkg8-documentation-rewrite.md) — Blocked by: 02, 03, 05

---

## Destination

ระบบ API ทั้ง 5 ชั้นของโปรเจกต์ได้รับการแก้ไข ชำระล้างโค้ดตาย ป้องกันข้อผิดพลาดเงียบ มีระบบ Type กลางตาม [ADR 0004](../../docs/adr/0004-two-pillar-single-source-of-truth-types.md) และเอกสารตรงกับความจริง 100% โดยปิดข้อค้นพบทั้งหมด 30 ข้อใน [`findings.md`](../api-payload-docs-review/findings.md) (เว้น F-001 ที่ผู้ใช้ตัดสินใจปล่อยไว้)
