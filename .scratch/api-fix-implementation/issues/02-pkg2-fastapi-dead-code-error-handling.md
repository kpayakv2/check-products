# 02: FastAPI Dead Code, Error Sanitization & Models Cleanup

**What to build:**
ทำความสะอาดและยกระดับความปลอดภัยให้กับชั้น FastAPI (Layer 1 & Layer 5): ลบ 8 legacy dead endpoints (`POST /api/v1/match/batch`, `POST /api/v1/upload/csv`, `POST /api/v1/match/single`, `POST /api/v1/config`, `GET /api/v1/jobs/{id}`, `GET /api/v1/results/{id}`, `POST /api/v1/match/file-dedup`, `POST /api/v1/match/import-dedup`), ลบระบบ WebSocket ทั้งหมด (`WS /ws`, `custom_websockets.py`, และ event publishers 7 ตัว), ลบ `/web` (404) ออกจาก response ของ root `GET /`, ปรับปรุง error handling ใน 14 endpoints ไม่ให้ส่ง `str(e)` ดิบออกทาง `{detail: ...}` เมื่อเกิด error 500 (ใช้ logging แทน), และปรับ response models ให้เป็น `snake_case`

**Blocked by:** None (can start immediately)

**Status:** closed

**ข้อค้นพบที่ปิด (9 ข้อ):**
- #F-003 (P1)
- #F-004 (P1)
- #F-005 (P1)
- #F-006 (P2)
- #F-007 (P2)
- #F-008 (P3)
- #F-009 (P3)
- #F-020 (P2)
- #F-021 (P3)

**ไฟล์ที่เกี่ยวข้อง:**
- `src/api/api_server.py`
- `src/api/custom_websockets.py` (ลบไฟล์ทิ้ง)
- `src/api/routers/match.py`
- `src/api/routers/system.py`
- `src/api/routers/jobs.py` (ลบไฟล์ทิ้ง)
- `src/api/routers/embed.py`
- `src/api/routers/learn.py`
- `src/api/routers/internal_match.py`
- `src/api/routers/classify.py`
- `src/api/models.py`
- `src/api/services/background_jobs.py`

**เทสต์ที่ต้องรันก่อน/หลัง:**
- `.venv/Scripts/python.exe -m pytest` (ต้องรันผ่านครบทุก unit/integration tests)

- [x] ลบ endpoint `POST /api/v1/match/batch` และ `POST /api/v1/match/single` จาก `match.py`
- [x] ลบ endpoint `POST /api/v1/upload/csv` และ `POST/GET /api/v1/config` จาก `system.py`
- [x] ลบไฟล์ `src/api/routers/jobs.py` และถอด router ออกจาก `api_server.py`
- [x] ลบ WebSocket `/ws` ใน `api_server.py`, ลบ `custom_websockets.py` และการเรียก notify ทั้งหมดใน routers
- [x] แก้ไข response ของ `GET /` ใน `api_server.py` ลบ `/web` ออก
- [x] แก้ไข 14 endpoints ใน routers ไม่ให้ส่ง `str(e)` ใน 500 detail (ใช้ `logger.exception` และส่ง generic detail)
- [x] ปรับ field names ใน `models.py` ให้เป็น `snake_case` พร้อม backward-compatible getter/setters
- [x] รัน pytest ครบ 100% ผ่าน
