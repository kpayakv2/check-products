# 07: Automated Integration Tests for Active Routes

**What to build:**
เพิ่มความครอบคลุมของการทดสอบอัตโนมัติ (Automated Tests) ให้กับ Endpoint และ Routes ที่ใช้งานจริงในระบบแต่ยังขาด Integration Test ชี้ตรง:
1. **FastAPI (pytest):** เขียน integration tests ครอบคลุม `POST /api/v1/learn/verify` และ `POST /api/v1/match/scan-internal`
2. **Next.js (jest):** เขียน integration tests ครอบคลุม `POST /api/similarity/resolve` และ `DELETE /api/keyword-rules/[id]`

**Blocked by:** 02 (FastAPI Cleanup), 05 (Next.js Routes Fixes)

**Status:** closed

**ข้อค้นพบที่ปิด (2 ข้อ):**
- #F-010 (P3)
- #F-017 (P3)

**ไฟล์ที่เกี่ยวข้อง:**
- `tests/integration/test_api_client.py` (หรือสร้าง `tests/integration/test_active_routes.py`)
- `taxonomy-app/__tests__/api/similarity-resolve.test.ts`
- `taxonomy-app/__tests__/api/keyword-rules-delete.test.ts`

**เทสต์ที่ต้องรันก่อน/หลัง:**
- `.venv/Scripts/python.exe -m pytest` ใน repo root
- `npx jest --ci` ใน `taxonomy-app/`

- [x] เพิ่ม pytest test cases สำหรับ `/api/v1/learn/verify` และ `/api/v1/match/scan-internal`
- [x] เพิ่ม jest test cases สำหรับ `/api/similarity/resolve` และ `/api/keyword-rules/[id]` (DELETE)
- [x] รันทั้งสอง test suites ผ่าน 100% ไม่มี regression
