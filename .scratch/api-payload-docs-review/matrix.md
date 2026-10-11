# เมทริกซ์: endpoint × 8 มิติ

**สร้างโดย:** [ตั๋ว 01](issues/01-inventory.md) (2026-10-11), สำรวจจากการอ่านโค้ดอย่างเดียว ยังไม่ได้ยิง request ใดๆ
**เกณฑ์ "ผ่าน" ของแต่ละมิติ:** [ตั๋ว 02](issues/02-pass-criteria.md) (ผู้ใช้ยืนยัน 2026-10-11) — ดูหัวข้อถัดไป
**ค่าในช่องมิติ:** `ผ่าน` / `มีข้อค้นพบ #n` (n = เลขใน [findings.md](findings.md)) / `ไม่เกี่ยว` ตั๋ว 01 เว้นว่างไว้ทุกช่อง

มิติ: 1 เอกสาร vs โค้ด · 2 type ข้ามรอยต่อ · 3 ตรวจ input · 4 สิทธิ์เข้าถึง · 5 สัญญา error · 6 เส้นตาย/ซ้ำซ้อน · 7 ความสม่ำเสมอเชิงรูปแบบ · 8 เทสต์

## เกณฑ์ "ผ่าน" ต่อมิติ (ตั๋ว 02)

ช่องจะได้ `ผ่าน` ก็ต่อเมื่อครบทุกข้อของมิตินั้น ไม่ครบข้อใด = `มีข้อค้นพบ #n` (ตั๋วรีวิวเป็นผู้กำหนดระดับ P0–P3 ตาม map Notes)

1. **เอกสาร vs โค้ด** — ทุกข้อกล่าวอ้างเรื่อง path / method / field / พอร์ตในไฟล์ใดๆ ของรีโปตรงกับโค้ดปัจจุบัน · ข้อกล่าวอ้างที่ผิดแต่ไม่พาใครทำผิดทาง = P3 · ที่พาคน/เอเจนต์ทำผิดทาง = P2
2. **type ข้ามรอยต่อ** — type ที่ประกาศของสองฝั่งมี field, ชนิด และ optional ตรงกัน และ shape ที่ผู้เรียกอ่านมีอยู่จริงใน response (ยิงจริงเฉพาะเส้นอ่านอย่างเดียว)
3. **ตรวจ input** — ทุก field ของ body/query ที่ handler ใช้ถูกตรวจชนิดและขอบเขตก่อนใช้ (Pydantic, zod หรือเช็กมือก็ได้) และผิดแล้วคืน 4xx ไม่ตกไป 500 · field ที่ไหลถึง DB/ไฟล์ (path, id, ขนาดไฟล์, จำนวนแถว) ต้องครอบทุกกรณี · `request.json()` ที่ไม่ตรวจเลย หรือ dict อิสระ = ไม่ผ่าน
4. **สิทธิ์เข้าถึง** (ถือว่าแอปเปิดสู่อินเทอร์เน็ต ไม่มีล็อกอินจริง) — เส้นเขียน/เทรน/แตะไฟล์/คำนวณหนักต้องผ่าน gate ที่ใช้ได้จริงของชั้นนั้น (`middleware.ts` สำหรับ Next.js · FastAPI ต้องมีกลไกกั้น ถ้าไม่มี = ข้อค้นพบ ตามแบบ F-002) · เส้นอ่านอย่างเดียวผ่านได้เมื่อข้อมูลไม่ใช่ที่ตั้งใจจำกัด · เส้นที่อ่านด้วย `supabaseAdmin` (ข้าม RLS) ต้องอยู่ใน `GATED_READ_PATHS` · ข้อค้นพบที่ต้องมีล็อกอินจริงถึงแก้ได้ → คำตัดสิน "ปล่อยไว้ รอโปรเจกต์ auth"
5. **สัญญา error** — (ก) รูป error เหมือนกันภายในชั้นเดียวกัน (ไม่บังคับข้ามชั้น: FastAPI `{detail}` · Next.js `{success:false, error}`) (ข) 4xx = ความผิดผู้เรียก (400/422 input ผิด, 404 ไม่พบ, 409 ซ้ำ) · 5xx = ความผิดระบบ · `str(e)` รั่วเข้า `detail` ของ 500 = ไม่ผ่าน (ค) การเขียนที่ไม่กระทบแถวใดต้องคืน 404 · 200 ที่มี `status:"skipped"`/`"no_change"` ต้องระบุอยู่ใน type ที่ประกาศ · การเรียก PostgREST ที่ได้ `200 []`/`204` โดยไม่ได้ทำอะไรต้องถูกตรวจจับ ไม่งั้นไม่ผ่าน
6. **เส้นตาย/ซ้ำซ้อน** — มีผู้เรียกอย่างน้อยหนึ่งจุดในแอป (หรือเอกสารระบุว่าเป็นเส้นสำหรับสคริปต์) และไม่มี endpoint อื่นทำงานเดียวกัน
7. **ความสม่ำเสมอเชิงรูปแบบ** — key ใน payload ที่ข้ามรอยต่อเป็น `snake_case` ไม่มีการแปล (กลุ่ม camelCase เช่น `ProductMatchResult`, `FileDeduplication*` = P3) · Next.js ใช้ `{success, data?, error?}` โดย key อื่นต้องอยู่ใน `data` · FastAPI คืน model ตรงไม่ต้องห่อ
8. **เทสต์** — มีเทสต์อัตโนมัติที่ชี้ถึง endpoint นั้นตรงๆ ≥1 ชนิด (unit/integration/e2e) ครอบทั้งกรณีสำเร็จและล้มเหลวอย่างน้อยหนึ่งกรณี · เทสต์ที่ skip/fail อยู่ไม่นับ · เส้นเขียนที่ยิงจริงไม่ได้ นับว่าครอบถ้ามีเทสต์ที่ mock DB · ค่าช่อง `ผ่าน` / `ขาด` โดยข้อค้นพบเป็น P3 เสมอ

---

ตัวย่อคอลัมน์:
- **อ/ข** — `อ` อ่านอย่างเดียว · `ข` เขียนข้อมูล (DB/ไฟล์/โมเดล/state) · `ค` คำนวณล้วน ไม่มีผลข้างเคียงถาวร · `ส` แตะแค่คุกกี้ session
- **payload** — `J` JSON · `M` multipart · `F` ไฟล์ดาวน์โหลด · `WS` ข้อความ websocket · `PG` PostgREST ผ่าน supabase-js · `ST` Supabase Storage
- **เอกสาร** — ชื่อไฟล์ย่อ (บรรทัดเต็มอยู่ใน[ภาคผนวก ก](#ภาคผนวก-ก-ตำแหน่งเอกสารที่อ้างถึงแต่ละ-path)) · `—` = ไม่มีเอกสารพูดถึง
- ผู้เรียกจากเบราว์เซอร์ไปชั้น 1 ผ่านพร็อกซี `/api/fastapi/*` ของ Next.js (`NEXT_PUBLIC_API_URL=/api/fastapi` ใน `.env.local:23`, rewrite ใน `next.config.js:18`) ถ้าไม่ได้ตั้ง env นี้ โค้ดจะ fallback ไปยิง `http://127.0.0.1:8000` ตรงๆ

---

## 1. ตารางเมทริกซ์

### ชั้น 1: FastAPI `:8000` (24 แถว)

| ID | endpoint | ไฟล์:บรรทัด | payload | อ/ข | type ที่ประกาศ (req → res) | ผู้เรียก | เอกสาร | มิติ1 | มิติ2 | มิติ3 | มิติ4 | มิติ5 | มิติ6 | มิติ7 | มิติ8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L1-01 | `GET /` | `src/api/api_server.py:86` | J | อ | ไม่มี → dict ไม่มี model | ไม่พบผู้เรียก | — | มีข้อค้นพบ #F-006 | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-008 | มีข้อค้นพบ #F-009 | มีข้อค้นพบ #F-010 |
| L1-02 | `GET /api/v1/health` | `src/api/routers/system.py:18` | J | อ | ไม่มี → `SystemHealth` | สคริปต์: `taxonomy-app/scripts/generate-category-embeddings.ts:78`, `scripts/check_endpoints.py:85` · (เทสต์หลายไฟล์ใน `tests/integration/`) | quick-reference, api-reference, API_CORRECTION, SYSTEM_OVERVIEW_TH, DEPLOY_LOCAL_MODEL + 6 | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L1-03 | `GET /api/v1/config` | `src/api/routers/system.py:35` | J | อ | ไม่มี → `APIConfig` | ไม่พบผู้เรียก | — | ไม่เกี่ยว | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-008 | ผ่าน | มีข้อค้นพบ #F-010 |
| L1-04 | `POST /api/v1/config` | `src/api/routers/system.py:40` | J | ข (state ในหน่วยความจำ + broadcast ws) | `APIConfig` → dict ไม่มี model | ไม่พบผู้เรียก | — | ไม่เกี่ยว | ไม่เกี่ยว | ผ่าน | มีข้อค้นพบ #F-002 | ผ่าน | มีข้อค้นพบ #F-008 | มีข้อค้นพบ #F-009 | มีข้อค้นพบ #F-010 |
| L1-05 | `POST /api/v1/upload/csv` | `src/api/routers/system.py:52` | M | ข (เขียนไฟล์ลง `uploads/` + เริ่มงานเบื้องหลัง) | `UploadFile` → dict ไม่มี model | ไม่พบผู้เรียก | — | มีข้อค้นพบ #F-004 | ไม่เกี่ยว | มีข้อค้นพบ #F-004 | มีข้อค้นพบ #F-002 | มีข้อค้นพบ #F-007 | มีข้อค้นพบ #F-008 | มีข้อค้นพบ #F-009 | มีข้อค้นพบ #F-010 |
| L1-06 | `POST /api/v1/clean` | `src/api/routers/system.py:97` | J | ค | `CleanTextRequest` → `CleanTextResponse` | `components/Import/DataCleaningStep.tsx:41` | quick-reference, api-reference, SYSTEM_OVERVIEW_TH, IMPORT_ROUTES_FLOW, import-wizard-progress, Import/README | ผ่าน | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-007 | ผ่าน | ผ่าน | ผ่าน |
| L1-07 | `POST /api/v1/match/single` | `src/api/routers/match.py:16` | J | ค (+ broadcast ws) | `ProductMatchRequest` → `List[MatchResult]` | ไม่พบผู้เรียก (มีแต่เทสต์: `test_api_client.py:46`, `test_taxonomy_import.py:181`) | quick-reference, api-reference, API_CORRECTION, SYSTEM_OVERVIEW_TH, human-feedback, capabilities-summary | ผ่าน | ไม่เกี่ยว | มีข้อค้นพบ #F-005 | มีข้อค้นพบ #F-002 | มีข้อค้นพบ #F-007 | มีข้อค้นพบ #F-008 | ผ่าน | ผ่าน |
| L1-08 | `POST /api/v1/match/batch` | `src/api/routers/match.py:73` | J | ข (สร้างงาน + เขียน `results/batch_*.json`) | `BatchMatchRequest` → `BatchMatchResponse` | ไม่พบผู้เรียก (มีแต่เทสต์: `test_api_client.py:86`) | quick-reference, api-reference, SYSTEM_OVERVIEW_TH, IMPORT_ROUTES_FLOW, import-wizard-progress, Import/README | ผ่าน | ไม่เกี่ยว | มีข้อค้นพบ #F-003 | มีข้อค้นพบ #F-002 | มีข้อค้นพบ #F-003, #F-007 | มีข้อค้นพบ #F-008 | ผ่าน | ผ่าน |
| L1-09 | `POST /api/embed` | `src/api/routers/embed.py:11` | J | ค | `EmbeddingRequest {text}` → `EmbeddingResponse` | Edge `generate-embeddings-local/index.ts:25` · สคริปต์: `scripts/fix_missing_embeddings.py:16`, `taxonomy-app/scripts/generate-category-embeddings.ts:34`, `taxonomy-app/scripts/verify_mock_removal.js:12`, `scripts/check_endpoints.py:91` | 5 ไฟล์ใน docs/api + อีก ~20 ไฟล์ | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-002 | มีข้อค้นพบ #F-007 | ผ่าน | ผ่าน | ผ่าน |
| L1-10 | `POST /api/embed/batch` | `src/api/routers/embed.py:37` | J | ค | `BatchEmbeddingRequest {texts}` → `BatchEmbeddingResponse` | `app/api/import/commit/route.ts:88` (ฝั่ง server) · สคริปต์: `scripts/import_legacy_products.py:50`, `scripts/check_endpoints.py:97` | quick-reference, api-reference, API_CORRECTION, SYSTEM_OVERVIEW_TH + 9 | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-002 | มีข้อค้นพบ #F-007 | ผ่าน | ผ่าน | ผ่าน |
| L1-11 | `GET /api/v1/jobs/{job_id}` | `src/api/routers/jobs.py:15` | J | อ | path → `JobStatus` | ไม่พบผู้เรียก (มีแต่เทสต์: `test_api_client.py:102`) | quick-reference, api-reference, API_CORRECTION, SYSTEM_OVERVIEW_TH, human-feedback, PROJECT_STRUCTURE | ผ่าน | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-008 | ผ่าน | ผ่าน |
| L1-12 | `GET /api/v1/jobs` | `src/api/routers/jobs.py:23` | J | อ | ไม่มี → `List[JobStatus]` | ไม่พบผู้เรียก | (ตามแถว L1-11) | ผ่าน | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-008 | ผ่าน | มีข้อค้นพบ #F-010 |
| L1-13 | `GET /api/v1/results/{job_id}` | `src/api/routers/jobs.py:28` | F (`application/json`) | อ | path → `FileResponse` ไม่มี model | ไม่พบผู้เรียก (URL ถูกฝังใน `JobStatus.results_url` ที่ `services/background_jobs.py:93`) | api-reference | ผ่าน | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-008 | ผ่าน | มีข้อค้นพบ #F-010 |
| L1-14 | `POST /api/v1/learn/verify` | `src/api/routers/learn.py:11` | J | ข (`keyword_rules` ผ่าน `taxonomy_service.py:99`) | `LearnVerifyRequest` → dict ไม่มี model | `app/api/verify/route.ts:58`, `app/api/recheck/route.ts:160` (ฝั่ง server ทั้งคู่) | AGENTS.md:43, quick-reference, api-reference, API_CORRECTION, SYSTEM_OVERVIEW_TH, PROJECT_STRUCTURE, CURRENT_STATUS | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-002 | มีข้อค้นพบ #F-007 | ผ่าน | มีข้อค้นพบ #F-009 | ผ่าน |
| L1-15 | `POST /api/v1/learn/retrain` | `src/api/routers/learn.py:37` | J | ข (โมเดล ML + `ml_training_history`) | ไม่มี → dict `{status,message,data}` | `components/data-quality/DeduplicationTab.tsx:210`, `app/ai-brain/page.tsx:124` | API_ARCHITECTURE, CURRENT_STATUS | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-002 | มีข้อค้นพบ #F-007 | ผ่าน | มีข้อค้นพบ #F-009 | ผ่าน |
| L1-16 | `GET /api/v1/learn/status` | `src/api/routers/learn.py:64` | J | อ | ไม่มี → dict ไม่มี model | `app/ai-brain/page.tsx:98` | API_ARCHITECTURE, CURRENT_STATUS | ผ่าน | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-007 | ผ่าน | มีข้อค้นพบ #F-009 | ผ่าน |
| L1-17 | `GET /api/v1/learn/history?limit` | `src/api/routers/learn.py:133` | J | อ | query `limit:int` → dict ไม่มี model | `app/ai-brain/page.tsx:99` | — | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-007 | ผ่าน | มีข้อค้นพบ #F-009 | ผ่าน |
| L1-18 | `POST /api/v1/match/scan-internal` | `src/api/routers/internal_match.py:437` | J | ค (เริ่มงานเบื้องหลัง เก็บผลในหน่วยความจำ ไม่พบการเขียน DB/ไฟล์ใน `src/api` แต่ยังไม่ได้ตามเข้า `src/services`) | `ScanInternalRequest` → `ScanStartResponse` (ประกาศในไฟล์ router เอง `:106`) | `components/data-quality/DeduplicationTab.tsx:121` · สคริปต์ `test_scan.py:4` (ราก repo) | API_ARCHITECTURE (เขียนเป็น `internal/scan` ผิด path) | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-002 | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L1-19 | `GET /api/v1/match/scan-internal/status/{task_id}` | `src/api/routers/internal_match.py:458` | J | อ | path → `ScanTaskStatusResponse` (`:111`) | `components/data-quality/DeduplicationTab.tsx:143` | — | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L1-20 | `POST /api/v1/match/file-dedup` | `src/api/routers/internal_match.py:476` | J | ค (ดาวน์โหลดไฟล์จาก Storage) | `FileDeduplicationRequest` → `FileDeduplicationResponse` | Edge `product-deduplication/index.ts:37` · สคริปต์ `taxonomy-app/scripts/test-api.js:164` (ผ่าน Edge) | — | ไม่เกี่ยว | ผ่าน | ผ่าน | มีข้อค้นพบ #F-002 | มีข้อค้นพบ #F-007 | ผ่าน | มีข้อค้นพบ #F-009 | มีข้อค้นพบ #F-010 |
| L1-21 | `POST /api/v1/match/import-dedup` | `src/api/routers/internal_match.py:683` | J | ค | `ImportDeduplicationRequest` → `List[ProductMatchResult]` | `components/Import/DeduplicationStep.tsx:60` | CURRENT_STATUS, import-wizard-progress | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-002 | มีข้อค้นพบ #F-007 | ผ่าน | มีข้อค้นพบ #F-009 | มีข้อค้นพบ #F-010 |
| L1-22 | `POST /api/classify/category` | `src/api/routers/classify.py:247` | J | ค | `CategoryRequest` → `CategoryResponse` (ประกาศใน `classify.py:20-53` ไม่ได้อยู่ใน `models.py`) | `components/Import/CategorizationStep.tsx:66` · Edge `hybrid-classification-local/index.ts:25` · สคริปต์ `scripts/test_hybrid_api.py:15` | analyze-capabilities, SYSTEM_OVERVIEW_TH, README_CLASSIFIER, thai-taxonomy-expert SKILL + 9 | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-002 | มีข้อค้นพบ #F-007 | ผ่าน | ผ่าน | ผ่าน |
| L1-23 | `POST /api/classify/batch` | `src/api/routers/classify.py:293` | J | ค | `BatchCategoryRequest` → `BatchCategoryResponse` | สคริปต์ `scripts/recheck_legacy_categories.py:51` · (เทสต์ `test_api_classifier.py:125`) | README_CLASSIFIER, SUMMARY_ARCHITECTURE, BACKEND_UPDATED + 3 | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-002 | มีข้อค้นพบ #F-007 | ผ่าน | ผ่าน | ผ่าน |
| L1-24 | `GET /docs`, `/redoc`, `/openapi.json` (FastAPI สร้างอัตโนมัติ) | `src/api/api_server.py:51-52` | J/HTML | อ | — | (เทสต์ `test_api_client.py:123`) | START_PHAYAK.bat:40 | ผ่าน | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |

### ชั้น 2: Next.js `/api/*` (25 แถว ใน 16 ไฟล์)

| ID | endpoint | ไฟล์:บรรทัด | payload | อ/ข | type ที่ประกาศ (req → res) | ผู้เรียก | เอกสาร | มิติ1 | มิติ2 | มิติ3 | มิติ4 | มิติ5 | มิติ6 | มิติ7 | มิติ8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L2-01 | `POST /api/similarity/resolve` | `app/api/similarity/resolve/route.ts:26` | J | ข | zod `ResolveDuplicateSchema` → ไม่มี type | `components/data-quality/DeduplicationTab.tsx:232` | — | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-012 | ผ่าน | ผ่าน | มีข้อค้นพบ #F-017 |
| L2-02 | `GET /api/pos-export` | `app/api/pos-export/route.ts:138` | J | อ (service role, gated read) | ไม่มี → ไม่มี type | `components/data-quality/PosExportTab.tsx:184` | plans/2026-09-06-pos-export, specs/…-ui-brief | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-016 | ผ่าน |
| L2-03 | `POST /api/pos-export` | `app/api/pos-export/route.ts:171` | J → F | ข (กำหนด SKU ให้ `products`) | zod `ExportRequestSchema` → ไฟล์ข้อความ | `components/data-quality/PosExportTab.tsx:204` (ฝั่งรับประกาศ `ExportResult` ที่ `:25`) | plans/2026-09-06-pos-export | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L2-04 | `PUT /api/synonyms/[id]` | `app/api/synonyms/[id]/route.ts:10` | J | ข | ไม่มี zod → ไม่มี type | `components/Taxonomy/SynonymsPanel.tsx:162` | taxonomy-app/README, app/README | ผ่าน | ผ่าน | มีข้อค้นพบ #F-013 | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L2-05 | `DELETE /api/synonyms/[id]` | `app/api/synonyms/[id]/route.ts:32` | J | ข | path → ไม่มี type | `components/Taxonomy/SynonymsPanel.tsx:150` | taxonomy-app/README, app/README | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L2-06 | `PUT /api/taxonomy/[id]` | `app/api/taxonomy/[id]/route.ts:12` | J | ข | ไม่มี zod → ไม่มี type | `app/taxonomy/page.tsx:124` | taxonomy-app/README, app/README | ผ่าน | ผ่าน | มีข้อค้นพบ #F-013 | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L2-07 | `DELETE /api/taxonomy/[id]` | `app/api/taxonomy/[id]/route.ts:34` | J | ข | path → ไม่มี type | `app/taxonomy/page.tsx:99` | taxonomy-app/README, app/README | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L2-08 | `POST /api/verify` | `app/api/verify/route.ts:70` | J | ข (+ เรียก L1-14) | zod `DecisionSchema` → ไม่มี type | `components/data-quality/VerifyTab.tsx:114` | SYSTEM_OVERVIEW_TH, CURRENT_STATUS, plans/…pos-export | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-016 | ผ่าน |
| L2-09 | `GET /api/settings` | `app/api/settings/route.ts:12` | J | อ (service role, gated read) | ไม่มี → ไม่มี type | `app/settings/page.tsx:163` | CURRENT_STATUS, plans/…pos-export | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L2-10 | `PUT /api/settings` | `app/api/settings/route.ts:19` | J | ข | ไม่มี zod → ไม่มี type | `app/settings/page.tsx:181` | CURRENT_STATUS | ผ่าน | ผ่าน | มีข้อค้นพบ #F-013 | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L2-11 | `GET /api/import/history` | `app/api/import/history/route.ts:13` | J | อ (service role, gated read) | query → ไม่มี type | `components/Import/ImportHistory.tsx:72` | CURRENT_STATUS, plans/…pos-export | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L2-12 | `GET /api/similarity/[id]` | `app/api/similarity/[id]/route.ts:4` | J | อ (anon ผ่าน `DatabaseService`) | path → ไม่มี type | ไม่พบผู้เรียก | taxonomy-app/README, app/README | ผ่าน | ไม่เกี่ยว | มีข้อค้นพบ #F-013 | มีข้อค้นพบ #F-014 | มีข้อค้นพบ #F-013 | มีข้อค้นพบ #F-015 | ผ่าน | มีข้อค้นพบ #F-017 |
| L2-13 | `POST /api/similarity/[id]` | `app/api/similarity/[id]/route.ts:21` | J | ข (anon INSERT ผ่าน `DatabaseService`) | ไม่มี zod → ไม่มี type | ไม่พบผู้เรียก | taxonomy-app/README | ผ่าน | ไม่เกี่ยว | มีข้อค้นพบ #F-013 | มีข้อค้นพบ #F-014 | มีข้อค้นพบ #F-014 | มีข้อค้นพบ #F-015 | ผ่าน | มีข้อค้นพบ #F-017 |
| L2-14 | `GET /api/taxonomy` | `app/api/taxonomy/route.ts:77` | J | อ (anon ผ่าน `DatabaseService`) | zod `GetTaxonomySchema` → `ApiResponse<T>` (ประกาศใน route `:27`) | ไม่พบผู้เรียก (หน้า taxonomy ใช้ `DatabaseService.getTaxonomyTree` ตรง) | taxonomy-app/README, app/README, TESTING.md, archive DEPLOYMENT_GUIDE, CURRENT_STATUS | ผ่าน | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-015 | มีข้อค้นพบ #F-016 | ผ่าน |
| L2-15 | `POST /api/taxonomy` | `app/api/taxonomy/route.ts:153` | J | ข | zod `CreateTaxonomySchema` → `ApiResponse<T>` | `app/taxonomy/page.tsx:130` | taxonomy-app/README, app/README, archive DEPLOYMENT_GUIDE | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L2-16 | `GET /api/synonyms` | `app/api/synonyms/route.ts:6` | J | อ (anon ผ่าน `DatabaseService`) | ไม่มี → ไม่มี type | ไม่พบผู้เรียก (SynonymsPanel ใช้ `DatabaseService.getSynonyms` ตรง) | taxonomy-app/README, app/README, CURRENT_STATUS | ผ่าน | ไม่เกี่ยว | มีข้อค้นพบ #F-013 | ผ่าน | ผ่าน | มีข้อค้นพบ #F-015 | ผ่าน | ผ่าน |
| L2-17 | `POST /api/synonyms` | `app/api/synonyms/route.ts:31` | J | ข | ไม่มี zod → ไม่มี type | `components/Taxonomy/SynonymsPanel.tsx:174` | taxonomy-app/README, app/README | ผ่าน | ผ่าน | มีข้อค้นพบ #F-013 | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L2-18 | `POST /api/unlock` | `app/api/unlock/route.ts:24` | J | ส | zod `UnlockSchema` → ไม่มี type | `app/unlock/page.tsx:32` | REAL_DATABASE_RELATIONSHIPS, DATABASE_SCHEMA, AGENTS.md:72 | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L2-19 | `GET /api/products` | `app/api/products/route.ts:4` | J | อ (anon ผ่าน `DatabaseService`) | query → ไม่มี type | ไม่พบผู้เรียก | taxonomy-app/README, app/README, CURRENT_STATUS | ผ่าน | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-015 | ผ่าน | ผ่าน |
| L2-20 | `POST /api/products` | `app/api/products/route.ts:21` | J | ข (anon INSERT `products` + `product_attributes`) | ไม่มี zod → ไม่มี type | ไม่พบผู้เรียก | taxonomy-app/README, app/README, CURRENT_STATUS | ผ่าน | ไม่เกี่ยว | มีข้อค้นพบ #F-013 | มีข้อค้นพบ #F-014 | มีข้อค้นพบ #F-014 | มีข้อค้นพบ #F-015 | ผ่าน | ผ่าน |
| L2-21 | `POST /api/import/commit` | `app/api/import/commit/route.ts:336` | J | ข (2 โหมด: dedup / categorize + เรียก L1-10) | zod `DedupCommitSchema` \| `CategorizeCommitSchema` (`ItemSchema`) → ไม่มี type (ฝั่งรับใช้ `SaveResult` ใน `types/import.ts:57`) | `components/Import/WizardTab.tsx:119`, `:181` | SYSTEM_OVERVIEW_TH, CURRENT_STATUS, import-wizard-progress | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-016 | ผ่าน |
| L2-22 | `GET /api/recheck` | `app/api/recheck/route.ts:26` | J | อ (**service role** แต่ไม่อยู่ใน `GATED_READ_PATHS`) | query → ไม่มี type | `components/data-quality/RecheckTab.tsx:48` | CURRENT_STATUS | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-011 | ผ่าน | ผ่าน | มีข้อค้นพบ #F-016 | ผ่าน |
| L2-23 | `POST /api/recheck` | `app/api/recheck/route.ts:76` | J | ข (+ เรียก L1-14) | zod `DecisionSchema` → ไม่มี type | `components/data-quality/RecheckTab.tsx:82` | CURRENT_STATUS | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-016 | ผ่าน |
| L2-24 | `POST /api/lock` | `app/api/lock/route.ts:10` | J | ส | ไม่มี → ไม่มี type | `components/Layout/Header.tsx:47` | — | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L2-25 | `DELETE /api/keyword-rules/[id]` | `app/api/keyword-rules/[id]/route.ts:12` | J | ข | path → ไม่มี type | `components/data-quality/AutoLearnTab.tsx:67` | CURRENT_STATUS | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-017 |

### ชั้น 3: Supabase Edge Functions (3 แถว)

| ID | endpoint | ไฟล์:บรรทัด | payload | อ/ข | type ที่ประกาศ (req → res) | ผู้เรียก | เอกสาร | มิติ1 | มิติ2 | มิติ3 | มิติ4 | มิติ5 | มิติ6 | มิติ7 | มิติ8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L3-01 | `POST /functions/v1/generate-embeddings-local` | `supabase/functions/generate-embeddings-local/index.ts:8` | J | ค (proxy → L1-09) | ไม่มี (destructure `{text}`) → `{embedding}` \| `{error,help}` | ไม่พบผู้เรียกในแอป | DEPLOY_LOCAL_MODEL, STATUS_CHECK_FINAL | มีข้อค้นพบ #F-018 | มีข้อค้นพบ #F-018 | มีข้อค้นพบ #F-018 | มีข้อค้นพบ #F-019 | มีข้อค้นพบ #F-018 | มีข้อค้นพบ #F-019 | ผ่าน | มีข้อค้นพบ #F-022 |
| L3-02 | `POST /functions/v1/hybrid-classification-local` | `supabase/functions/hybrid-classification-local/index.ts:8` | J | ค (proxy → L1-22) | ไม่มี (destructure `{product_name,method,top_k}`) → ส่ง response ของ FastAPI ต่อ | สคริปต์ `scripts/test_real_classification.py:11` | DEPLOY_LOCAL_MODEL, STATUS_CHECK_FINAL | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-019 | ผ่าน | มีข้อค้นพบ #F-019 | ผ่าน | มีข้อค้นพบ #F-022 |
| L3-03 | `POST /functions/v1/product-deduplication` | `supabase/functions/product-deduplication/index.ts:16` | J | ค (proxy → L1-20) | `DeduplicationRequest` (TS interface `:9`, camelCase) → ส่ง response ของ FastAPI ต่อ | สคริปต์ `taxonomy-app/scripts/test-api.js:164` | — | มีข้อค้นพบ #F-019 | มีข้อค้นพบ #F-019 | ผ่าน | มีข้อค้นพบ #F-019 | ผ่าน | มีข้อค้นพบ #F-019 | ผ่าน | มีข้อค้นพบ #F-022 |

`supabase/config.toml:321-325` มีบล็อก `verify_jwt = false` ที่ถูกคอมเมนต์ไว้ (ของ function ที่ไม่มีแล้ว) ทั้ง 3 ตัวจึงใช้ค่าเริ่มต้นของ Supabase

### ชั้น 4: frontend → PostgREST/Storage ตรงด้วย anon key (44 แถว)

client คือ `supabase` จาก `utils/supabase.ts:22` (anon key, URL ผ่านพร็อกซี `/api/supabase` เมื่อรันในเบราว์เซอร์)

**4a. เรียกตรงจาก component/หน้า (8 แถว)**

| ID | ตาราง / operation | ไฟล์:บรรทัด | payload | อ/ข | type ที่ประกาศ | ผู้เรียก | เอกสาร | มิติ1 | มิติ2 | มิติ3 | มิติ4 | มิติ5 | มิติ6 | มิติ7 | มิติ8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L4-01 | `products` SELECT count ×3 (แยกตาม status) | `components/data-quality/VerifyTab.tsx:46-58` | PG | อ | ไม่มี | ตัวเอง | — | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L4-02 | `products` SELECT รายการ | `components/data-quality/VerifyTab.tsx:74` | PG | อ | ไม่มี | ตัวเอง | — | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L4-03 | `taxonomy_nodes` SELECT | `components/data-quality/VerifyTab.tsx:96` | PG | อ | ไม่มี | ตัวเอง | — | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L4-04 | `keyword_rules` SELECT | `components/data-quality/AutoLearnTab.tsx:43` | PG | อ | ไม่มี | ตัวเอง | — | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L4-05 | `taxonomy_nodes` SELECT | `components/data-quality/RecheckTab.tsx:66` | PG | อ | ไม่มี | ตัวเอง | — | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L4-06 | `imports` SELECT | `components/Import/StorageImport.tsx:89` | PG | อ | `ImportRecord` ในไฟล์ (`:32`) | ตัวเอง | — | ไม่เกี่ยว | ผ่าน | ผ่าน | มีข้อค้นพบ #F-023 | มีข้อค้นพบ #F-023 | ผ่าน | ผ่าน | ผ่าน |
| L4-07 | Storage `uploads` download | `components/Import/StorageImport.tsx:116` | ST | อ | ไม่มี | ตัวเอง | — | มีข้อค้นพบ #F-024 | ผ่าน | ผ่าน | มีข้อค้นพบ #F-024 | มีข้อค้นพบ #F-024 | ผ่าน | ผ่าน | ผ่าน |
| L4-08 | Storage `uploads` download | `components/Import/ImportHistory.tsx:94` | ST | อ | ไม่มี | ตัวเอง | — | มีข้อค้นพบ #F-024 | ผ่าน | ผ่าน | มีข้อค้นพบ #F-024 | มีข้อค้นพบ #F-024 | ผ่าน | ผ่าน | ผ่าน |

**4b. ผ่าน `DatabaseService` (`utils/supabase.ts:252`) ที่มีผู้เรียก (14 แถว)**

| ID | เมธอด → ตาราง / operation | ไฟล์:บรรทัด | payload | อ/ข | type ที่ประกาศ (res) | ผู้เรียก | เอกสาร | มิติ1 | มิติ2 | มิติ3 | มิติ4 | มิติ5 | มิติ6 | มิติ7 | มิติ8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L4-09 | `getTaxonomyTree` → `taxonomy_nodes` SELECT | `utils/supabase.ts:254` | PG | อ | `TaxonomyNode[]` | `app/taxonomy/page.tsx:70`, `components/Taxonomy/SynonymsPanel.tsx:97`, `app/products/page.tsx:101`, `app/api/taxonomy/route.ts:127` (server) | app/README, archive setup | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L4-10 | `getTaxonomyNodes` → `taxonomy_nodes` SELECT | `utils/supabase.ts:282` | PG | อ | `TaxonomyNode[]` | `app/api/taxonomy/route.ts:118` (server) | — | ไม่เกี่ยว | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-026 | ผ่าน | ผ่าน |
| L4-11 | `getSynonyms` → `synonym_lemmas` SELECT | `utils/supabase.ts:403` | PG | อ | `Synonym[]` | `components/Taxonomy/SynonymsPanel.tsx:96`, `app/api/synonyms/route.ts:11` (server) | app/README, archive setup | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L4-12 | `searchProducts` → `products` SELECT | `utils/supabase.ts:509` | PG | อ | ประกาศใน signature | `app/products/page.tsx:71` | — | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L4-13 | `getProductStatusCounts` → `products` SELECT count | `utils/supabase.ts:548` | PG | อ | `Record<ProductStatus, number>` | `app/products/page.tsx:102` | — | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L4-14 | `getProducts` → `products` SELECT | `utils/supabase.ts:570` | PG | อ | `Product[]` | `app/api/products/route.ts:10` (server) | app/README, archive setup | ผ่าน | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-026 | ผ่าน | ผ่าน |
| L4-15 | `getSimilarityMatches` → `similarity_matches` SELECT | `utils/supabase.ts:590` | PG | อ | `SimilarityMatch[]` | `app/api/similarity/[id]/route.ts:10` (server) | — | ไม่เกี่ยว | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-026 | ผ่าน | ผ่าน |
| L4-16 | `createSimilarityMatch` → `similarity_matches` INSERT | `utils/supabase.ts:609` | PG | ข | `SimilarityMatch` | `app/api/similarity/[id]/route.ts:37` (server) | — | ไม่เกี่ยว | ไม่เกี่ยว | ผ่าน | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | ผ่าน | ผ่าน |
| L4-17 | `createProduct` → `products` INSERT | `utils/supabase.ts:647` | PG | ข | `Product` | `app/api/products/route.ts:44` (server) | — | ไม่เกี่ยว | ไม่เกี่ยว | ผ่าน | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | ผ่าน | ผ่าน |
| L4-18 | `createProductAttribute` → `product_attributes` INSERT | `utils/supabase.ts:669` | PG | ข | `ProductAttribute` | `app/api/products/route.ts:60` (server) | — | ไม่เกี่ยว | ไม่เกี่ยว | ผ่าน | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | ผ่าน | ผ่าน |
| L4-19 | `getRegexRules` → `regex_rules` SELECT | `utils/supabase.ts:681` | PG | อ | `any[]` | `app/settings/page.tsx:147` | — | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L4-20 | `getKeywordRules` → `keyword_rules` SELECT | `utils/supabase.ts:691` | PG | อ | `any[]` | `app/settings/page.tsx:148` | — | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L4-21 | `getDashboardStats` → count 9 ครั้ง (`taxonomy_nodes`, `synonym_lemmas`, `products`×4, `similarity_matches`×2, `review_history`) | `utils/supabase.ts:790` | PG | อ | `DashboardStats` | `app/page.tsx:133` | — | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |
| L4-22 | `getRecheckAgreement` → RPC `recheck_agreement_stats` | `utils/supabase.ts:842` | PG | อ | `{total,agreed} \| null` | `getDashboardStats` (`:816`) | — | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน | ผ่าน |

**4c. เมธอดของ `DatabaseService` ที่ไม่พบผู้เรียกนอกเทสต์ (22 แถว)**

| ID | เมธอด → ตาราง / operation | ไฟล์:บรรทัด | อ/ข | มิติ1 | มิติ2 | มิติ3 | มิติ4 | มิติ5 | มิติ6 | มิติ7 | มิติ8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L4-23 | `getTaxonomyNode` → `taxonomy_nodes` SELECT | `utils/supabase.ts:319` | อ | ผ่าน | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-026 | ผ่าน | ผ่าน |
| L4-24 | `getNextSortOrder` → `taxonomy_nodes` SELECT | `:333` | อ | ผ่าน | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-026 | ผ่าน | ผ่าน |
| L4-25 | `createTaxonomyNode` → `taxonomy_nodes` INSERT | `:352` | ข | ผ่าน | ไม่เกี่ยว | ผ่าน | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | ผ่าน | ผ่าน |
| L4-26 | `updateTaxonomyNode` → `taxonomy_nodes` UPDATE | `:369` | ข | ผ่าน | ไม่เกี่ยว | ผ่าน | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | ผ่าน | ผ่าน |
| L4-27 | `searchTaxonomyNodes` → `taxonomy_nodes` SELECT | `:381` | อ | ผ่าน | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-026 | ผ่าน | ผ่าน |
| L4-28 | `deleteTaxonomyNode` → `taxonomy_nodes` DELETE | `:393` | ข | ผ่าน | ไม่เกี่ยว | ผ่าน | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | ผ่าน | ผ่าน |
| L4-29 | `createSynonym` → `synonym_lemmas` INSERT | `:425` | ข | ผ่าน | ไม่เกี่ยว | ผ่าน | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | ผ่าน | ผ่าน |
| L4-30 | `updateSynonym` → `synonym_lemmas` UPDATE | `:443` | ข | ผ่าน | ไม่เกี่ยว | ผ่าน | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | ผ่าน | ผ่าน |
| L4-31 | `deleteSynonym` → `synonym_lemmas` DELETE | `:462` | ข | ผ่าน | ไม่เกี่ยว | ผ่าน | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | ผ่าน | ผ่าน |
| L4-32 | `createSynonymTerm` → `synonym_terms` INSERT | `:475` | ข | ผ่าน | ไม่เกี่ยว | ผ่าน | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | ผ่าน | ผ่าน |
| L4-33 | `getSynonymTerms` → `synonym_terms` SELECT | `:492` | อ | ผ่าน | ไม่เกี่ยว | ผ่าน | ผ่าน | ผ่าน | มีข้อค้นพบ #F-026 | ผ่าน | ผ่าน |
| L4-34 | `createImport` → `imports` INSERT | `:624` | ข | ผ่าน | ไม่เกี่ยว | ผ่าน | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | ผ่าน | ผ่าน |
| L4-35 | `updateImport` → `imports` UPDATE | `:635` | ข | ผ่าน | ไม่เกี่ยว | ผ่าน | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | ผ่าน | ผ่าน |
| L4-36 | `createProductCategorySuggestion` → `product_category_suggestions` INSERT | `:658` | ข | ผ่าน | ไม่เกี่ยว | ผ่าน | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | ผ่าน | ผ่าน |
| L4-37 | `getSystemSettings` → `system_settings` SELECT | `:704` | อ | ผ่าน | ไม่เกี่ยว | ผ่าน | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025, #F-026 | ผ่าน | ผ่าน |
| L4-38 | `updateSystemSettings` → `system_settings` UPDATE | `:714` | ข | ผ่าน | ไม่เกี่ยว | ผ่าน | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | ผ่าน | ผ่าน |
| L4-39 | `createRegexRule` → `regex_rules` INSERT | `:725` | ข | ผ่าน | ไม่เกี่ยว | ผ่าน | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | ผ่าน | ผ่าน |
| L4-40 | `updateRegexRule` → `regex_rules` UPDATE | `:736` | ข | ผ่าน | ไม่เกี่ยว | ผ่าน | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | ผ่าน | ผ่าน |
| L4-41 | `deleteRegexRule` → `regex_rules` DELETE | `:748` | ข | ผ่าน | ไม่เกี่ยว | ผ่าน | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | ผ่าน | ผ่าน |
| L4-42 | `createKeywordRule` → `keyword_rules` INSERT | `:757` | ข | ผ่าน | ไม่เกี่ยว | ผ่าน | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | ผ่าน | ผ่าน |
| L4-43 | `updateKeywordRule` → `keyword_rules` UPDATE | `:768` | ข | ผ่าน | ไม่เกี่ยว | ผ่าน | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | ผ่าน | ผ่าน |
| L4-44 | `deleteKeywordRule` → `keyword_rules` DELETE | `:780` | ข | ผ่าน | ไม่เกี่ยว | ผ่าน | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | มีข้อค้นพบ #F-025 | ผ่าน | ผ่าน |

เอกสารที่อ้างถึง 4c: `taxonomy-app/docs/archive_docs/setup.md:106-135`, `CHANGELOG.md:20-21`, `docs/archive_docs/DEPLOYMENT_GUIDE.md:14-15`

### ชั้น 5: WebSocket (7 แถว)

| ID | endpoint / ข้อความ | ไฟล์:บรรทัด | payload | อ/ข | type ที่ประกาศ | ผู้ส่ง / ผู้รับ | เอกสาร | มิติ1 | มิติ2 | มิติ3 | มิติ4 | มิติ5 | มิติ6 | มิติ7 | มิติ8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L5-01 | `WS /ws` (เชื่อมต่อ, ข้อความจาก client ถูกอ่านแล้วทิ้ง) | `src/api/api_server.py:103`, `custom_websockets.py:5` | WS | อ | ไม่มี | ผู้รับฝั่ง client: ไม่พบ (ไม่มี `new WebSocket` ใน frontend) | module-relationships-analysis:155 | ผ่าน | ไม่เกี่ยว | มีข้อค้นพบ #F-020 | มีข้อค้นพบ #F-020 | มีข้อค้นพบ #F-020 | มีข้อค้นพบ #F-021 | ผ่าน | มีข้อค้นพบ #F-022 |
| L5-02 | `batch_job_progress` | `src/api/services/background_jobs.py:57` | WS | — | dict ไม่มี model | ผู้ส่ง: งาน batch (L1-05, L1-08) · ผู้รับ: ไม่พบ | — | ไม่เกี่ยว | ไม่เกี่ยว | ไม่เกี่ยว | มีข้อค้นพบ #F-020 | ผ่าน | มีข้อค้นพบ #F-021 | ผ่าน | มีข้อค้นพบ #F-022 |
| L5-03 | `batch_job_completed` | `src/api/services/background_jobs.py:96` | WS | — | dict ไม่มี model | ผู้ส่ง: งาน batch · ผู้รับ: ไม่พบ | — | ไม่เกี่ยว | ไม่เกี่ยว | ไม่เกี่ยว | มีข้อค้นพบ #F-020 | ผ่าน | มีข้อค้นพบ #F-021 | ผ่าน | มีข้อค้นพบ #F-022 |
| L5-04 | `batch_job_failed` | `src/api/services/background_jobs.py:110` | WS | — | dict ไม่มี model | ผู้ส่ง: งาน batch · ผู้รับ: ไม่พบ | — | ไม่เกี่ยว | ไม่เกี่ยว | ไม่เกี่ยว | มีข้อค้นพบ #F-020 | มีข้อค้นพบ #F-007 | มีข้อค้นพบ #F-021 | ผ่าน | มีข้อค้นพบ #F-022 |
| L5-05 | `config_updated` | `src/api/routers/system.py:46` | WS | — | dict ไม่มี model | ผู้ส่ง: L1-04 · ผู้รับ: ไม่พบ | — | ไม่เกี่ยว | ไม่เกี่ยว | ไม่เกี่ยว | มีข้อค้นพบ #F-020 | ผ่าน | มีข้อค้นพบ #F-021 | ผ่าน | มีข้อค้นพบ #F-022 |
| L5-06 | `single_match_completed` | `src/api/routers/match.py:60` | WS | — | dict ไม่มี model | ผู้ส่ง: L1-07 · ผู้รับ: ไม่พบ | — | ไม่เกี่ยว | ไม่เกี่ยว | ไม่เกี่ยว | มีข้อค้นพบ #F-020 | ผ่าน | มีข้อค้นพบ #F-021 | ผ่าน | มีข้อค้นพบ #F-022 |
| L5-07 | `batch_job_started` | `src/api/routers/match.py:107` | WS | — | dict ไม่มี model | ผู้ส่ง: L1-08 · ผู้รับ: ไม่พบ | — | ไม่เกี่ยว | ไม่เกี่ยว | ไม่เกี่ยว | มีข้อค้นพบ #F-020 | ผ่าน | มีข้อค้นพบ #F-021 | ผ่าน | มีข้อค้นพบ #F-022 |

**รวม 103 แถว:** ชั้น 1 = 24 · ชั้น 2 = 25 · ชั้น 3 = 3 · ชั้น 4 = 44 (8 + 14 + 22) · ชั้น 5 = 7

ตรวจนับเทียบ Glob/Grep แล้ว:
- route file 16 ไฟล์ → handler ที่ export 25 ตัว → 25 แถว
- `@router.*`/`@app.*` ใน `src/api` มี 24 ตัว (router 22 + `@app.get("/")` = 23 HTTP → L1-01–L1-23, และ `@app.websocket("/ws")` → L5-01) ส่วน L1-24 เป็นเส้นที่ FastAPI สร้างเอง
- Edge function 3 โฟลเดอร์ → 3 แถว
- `.from(`/`.rpc(` ใน `components/`, `app/` (นอก `app/api/`), `utils/supabase.ts` ครบทุกจุด ไม่รวม `utils/admin-db.ts` ซึ่งเป็น service role ฝั่ง server (ภายในชั้น 2)

---

## 2. ข้อกล่าวอ้างในเอกสาร/โค้ดที่ไม่ตรงกับแถวใด

ใช้ในตั๋ว 07 คอลัมน์ "อาจเป็นบันทึกประวัติ" หมายถึงไฟล์ที่ระบุวันที่ หรืออยู่ใน `docs/reports/`/`archive_docs/` ตั๋ว 07 ต้องตัดสินตามข้อแยกแยะในตั๋ว

| path ที่อ้าง | อ้างที่ | ความจริง | อาจเป็นบันทึกประวัติ |
|---|---|---|---|
| `/api/import/process` | `docs/SYSTEM_OVERVIEW_TH.md:168`, `taxonomy-app/app/README.md:216,319`, `docs/architecture/DATABASE_SCHEMA.md:82`, `docs/CURRENT_STATUS.md:195,393` | ไม่มี route นี้ | CURRENT_STATUS ใช่ (session log) |
| `/api/import/process-local` | `taxonomy-app/app/README.md:217`, `docs/reports/STATUS_CHECK_FINAL.md:131`, `taxonomy-app/DEPLOY_LOCAL_MODEL.md:286`, `docs/architecture/REAL_DATABASE_RELATIONSHIPS.md:129` | ไม่มี | STATUS_CHECK_FINAL ใช่ |
| `/api/import/process-storage` | `taxonomy-app/app/README.md:218` | ไม่มี | — |
| `/api/import/pending` | `taxonomy-app/app/README.md:219,220,324,326`, `taxonomy-app/docs/import-wizard-progress.md:84` | ไม่มี | — |
| `/api/import/approve` | `taxonomy-app/app/README.md:221`, `taxonomy-app/docs/import-wizard-progress.md:84` | ไม่มี | — |
| `/api/products/[id]/review` | `taxonomy-app/app/README.md:229,335`, `taxonomy-app/README.md:187`, `docs/CURRENT_STATUS.md:191` | ไม่มี (หน้า `/products` เป็นแบบอ่านอย่างเดียวแล้วตาม AGENTS.md) | CURRENT_STATUS ใช่ |
| `/api/status` | `tests/README.md:120`, `docs/development/module-relationships-analysis.md:154`, `docs/reports/test-*.md` (4 ไฟล์) | ไม่มี (ของจริงคือ `/api/v1/health`) | reports ใช่ |
| `/api/batch`, `/api/match` | `docs/development/module-relationships-analysis.md:137,152,153` | ไม่มี (ของจริงอยู่ใต้ `/api/v1/match/*`) | — |
| `/api/v1/match/internal/scan` | `docs/api/API_ARCHITECTURE.md:96` | ของจริงคือ `/api/v1/match/scan-internal` (L1-18) | — |
| `/web` (หน้าเว็บ static ของ FastAPI) | คำตอบของ L1-01 (`api_server.py:98`), `tests/integration/test_api_client.py:171` | `src/api/web/` ไม่มีอยู่ จึงไม่ถูก mount (`api_server.py:66`) | — |
| `WS /ws` ฝั่ง client | `docs/development/module-relationships-analysis.md:155` | มีแค่ฝั่ง server ไม่มีผู้รับ | — |
| Edge `category-suggestions`, `generate-embeddings`, `hybrid-search`, `exec-sql` | `taxonomy-app/supabase/functions/README.md:9,29,41,58` | ไม่มีทั้ง 4 ตัว (`exec-sql` ถูกลบตาม AGENTS.md) | — |
| Edge `hybrid-classification` | `docs/development/MIGRATION_PLAN.md:604` | ของจริงชื่อ `hybrid-classification-local` | — |
| **โค้ด** เรียก Edge `hybrid-search`, `category-suggestions`, `generate-embeddings`, `similarity-matching`, `batch-processing`, `thai-text-processing` | `taxonomy-app/utils/api.ts:80-155` | ไม่มีทั้ง 6 ตัว และ `utils/api.ts` ไม่มีไฟล์ไหน import | — |
| **โค้ด** เรียก Edge `category-suggestions`, `generate-embeddings`, `hybrid-search`, `exec-sql` | `taxonomy-app/test_functions.js:8,37,62,88`, `taxonomy-app/scripts/test-api.js:68,98,128` | ไม่มีทั้ง 4 ตัว | — |
| `DatabaseService.getSynonymsByTerm`, `getLemmasByCategory`, `updateProductStatus` | `taxonomy-app/data/README_DATASET.md:167,170`, `taxonomy-app/docs/archive_docs/setup.md:135` | ไม่มีเมธอดเหล่านี้ | archive ใช่ |

prefix ที่ถูกอ้างแบบไม่ระบุเส้น (ไม่นับเป็นข้อกล่าวอ้างระดับ endpoint): `/api/classify` (MIGRATION_PLAN:721), `/api/similarity` (app/README:249), `/api/import` (app/README:212), `/api/v1/match` (quick-reference:31, PROJECT_STRUCTURE:32) ส่วนพร็อกซี `/api/fastapi`, `/api/supabase` (AGENTS.md:68, rules-windows.md) มีจริงใน `next.config.js:18,22`

---

## 3. ขนาดต่อชั้น และความเห็นเรื่องแตกตั๋ว

| ชั้น | แถว | โค้ดหลักที่ต้องอ่าน | ตั๋ว | ใหญ่เกินหนึ่งเซสชันไหม |
|---|---|---|---|---|
| 1 FastAPI | 24 | router + models + dependencies + background_jobs ≈ 2,190 บรรทัด (`internal_match.py` 853 บรรทัดคนเดียว) และต้องตามเข้า `src/services/` บางจุดสำหรับมิติ 5 | 03 | **ไม่เกิน** แต่ใกล้ขอบ ถ้าตั๋ว 03 ต้องไล่ `src/services/*` ลึก ให้แตกเป็น (a) `internal_match` + `classify` (b) ที่เหลือ |
| 2 Next.js | 25 | route 16 ไฟล์ ≈ 1,760 บรรทัด + `utils/{error-handler,validation,rate-limit,internal-auth,admin-db}.ts` + `middleware.ts` | 04 | ไม่เกิน |
| 3 Edge + 5 WS | 3 + 7 | ≈ 190 + 40 บรรทัด | 05 | ไม่เกิน (เล็กมาก) |
| รอยต่อ | 3 รอยต่อ: ผู้เรียกฝั่ง frontend ~20 ไฟล์, Next→FastAPI 3 จุด, PostgREST 22 แถวที่มีผู้เรียก (4a + 4b) + schema จริงจาก MCP `postgres` | ≈ 880 บรรทัด `utils/supabase.ts` + component ที่เรียก | 06 | ไม่เกิน ถ้ามองเฉพาะบรรทัด fetch/`.from()` กับ type รอบๆ ไม่อ่านทั้ง component |
| เอกสาร | ~65 ไฟล์ .md ที่อ้าง path/พอร์ต (ไม่นับ `.scratch/` และ rule ทั่วไปของสกิล vercel-react) | — | 07 | **ไม่เกิน** ถ้าใช้ภาคผนวก ก เป็นรายการตั้งต้นแทนการอ่านทั้งไฟล์ |

**ข้อสรุป: ไม่ต้องแตกตั๋วตอนนี้** ความเสี่ยงเดียวคือตั๋ว 03 ถ้าเกินขอบให้แตกตามข้อเสนอในตาราง

---

## 4. รูปแบบที่พบบ่อย (ป้อนเข้าตั๋ว 02)

### ชั้น 1: FastAPI
- **envelope ของคำตอบที่สำเร็จ:** คืน object/list ตรงๆ ตาม `response_model` ไม่มี envelope 15 จาก 23 เส้น · dict อิสระไม่มี model 7 เส้น (L1-01, 04, 05, 14, 15, 16, 17) + `FileResponse` 1 เส้น (L1-13) · แบบมี `success` ปน: `FileDeduplicationResponse.success` (L1-20), `{status,message,data}` (L1-15)
- **รูปแบบ error:** `HTTPException(detail=<string>)` ทุกจุด → body `{"detail": "..."}` · validation ล้มเหลว → 422 อัตโนมัติจาก Pydantic (`{"detail":[...]}`)
- **status code ที่ประกาศเอง (33 จุด):** 500 ×15 · 503 ×8 · 404 ×6 · 400 ×4 ในจำนวนนี้ 500 ทั้ง 14 จุดที่อยู่ใน `except Exception` ต่อ `str(e)` เข้า `detail` (และ 404 อีก 2 จุดใน `internal_match.py:497,503`)
- **ชื่อ field:** `snake_case` เกือบทั้งหมด ยกเว้นกลุ่ม file/import-dedup (`ProductMatchResult`, `FileDeduplicationStats`, `FileDeduplicationResponse`) ที่ใช้ `camelCase`
- **"สำเร็จแต่ไม่ได้ทำ":** L1-14 คืน 200 พร้อม `status: "skipped"` หรือ `"no_change"`

### ชั้น 2: Next.js
- **envelope:** `{ success: boolean, data?, error? }` ทั้ง 25 handler (ยกเว้น L2-03 ที่คืนไฟล์เมื่อสำเร็จ) แต่ key ข้าง `success` ไม่เหมือนกัน: `data` (ส่วนใหญ่), `action`/`category_id`/`learned` (L2-08), `pagination` (L2-22), `updated`/`suggestions` (L2-21) ฯลฯ
- **error:** `{ success:false, error:<string> }` ตอบเอง · ส่วน `withErrorHandling` (`utils/error-handler.ts:112`) ใส่ `code`, `timestamp`, `requestId` เพิ่ม และซ่อนข้อความจริงเมื่อไม่ใช่ development
- **ใช้ `withErrorHandling`:** 19/25 handler · ไม่ใช้ 6 (L2-12, 13, 16, 19, 20, 24) ซึ่งใช้ try/catch เองแล้วคืน 500 ข้อความคงที่
- **ตรวจ input ด้วย zod:** 8/25 handler (L2-01, 03, 08, 14, 15, 18, 21, 23) ที่เหลือตรวจเองหรือไม่ตรวจ
- **rate limit:** 7 handler (L2-01, 08, 14, 15, 18, 21, 23)
- **status code ที่ประกาศเอง:** 400 ×14 · 404 ×11 · 500 ×9 · 429 ×7 · 201 ×2 · 200 ×1 · 401 ×1 · 409 ×1 · บวก 401 จาก `middleware.ts:42` · ที่เหลือมาจาก `error.status` ใน `withErrorHandling`
- **ที่ใช้ 404 เมื่อแก้/ลบแล้วไม่โดนแถวไหน:** L2-04, 05, 06, 07, 08, 25 (แนวที่รีโปเลือกใช้กันบั๊ก "200 แต่ไม่ได้ทำ")
- **ชื่อ field ใน body:** `snake_case` (`category_id`, `suggestion_id`, `keep_id`, `import_batch_id`) ตรงกับคอลัมน์ DB

### ชั้น 3: Edge Functions
- สำเร็จ: L3-01 ห่อเป็น `{embedding}` · L3-02/03 ส่งคำตอบของ FastAPI ต่อตรงๆ
- error: `{error, help?}` 400 (input ขาด) / 500 (อย่างอื่นทั้งหมด รวม 4xx จาก FastAPI)
- CORS `*` ทั้ง 3 ตัว

### ชั้น 4: PostgREST ตรง
- `{data, error}` ของ supabase-js · ฝั่ง `DatabaseService` ส่วนใหญ่ `throw error` · component ที่เรียกตรงบางจุดไม่เช็ก `error` เลย (ตั๋ว 06 ต้องนับให้ชัด)

### ชั้น 5: WebSocket
- ข้อความ `{type: <snake_case string>, ...}` 6 ชนิด ไม่มี schema

### ฝั่งผู้เรียก (ข้อมูลตั้งต้นให้ตั๋ว 06)
- ทุกจุดที่ `fetch` แล้ว `await response.json()` ได้ค่า `any` ไม่มีการประกาศ type ของคำตอบ ยกเว้น L2-03 (`ExportResult`) และ L2-21 (`SaveResult`) · `DeduplicationStep.tsx:74` ประกาศตรงๆ เป็น `any[]`

---

## หมายเหตุ: สิ่งที่เห็นระหว่างสำรวจ (ยังไม่ได้ประเมิน)

ข้อเหล่านี้จดไว้ตามข้อห้ามในตั๋ว 01 ยังไม่ใช่ข้อค้นพบ ตั๋วที่ระบุในวงเล็บต้องยืนยันหรือปัดตก

1. **[สงสัย P0 → ตั๋ว 03 มิติ 4]** FastAPI bind `0.0.0.0:8000` (`api_server.py:133`) และตั้ง CORS `allow_origins=["*"]` + `allow_credentials=True` (`:56-62`) โดยไม่มีการยืนยันตัวตนเลย เส้นที่เขียนข้อมูล (L1-14 เขียน `keyword_rules`, L1-15 เทรนโมเดลใหม่, L1-04 แก้ config, L1-05 เขียนไฟล์) จึงยิงตรงที่ `:8000` ได้โดยไม่ผ่าน `middleware.ts` (ซึ่งกั้นเฉพาะเส้นทางพร็อกซี `/api/fastapi/*`) ใน LAN จะยิงได้จริงหรือไม่ขึ้นกับ Windows Firewall ซึ่ง**ยังไม่ได้ตรวจ**
2. **[ตั๋ว 04 มิติ 4]** L2-22 `GET /api/recheck` อ่านด้วย `supabaseAdmin` (bypass RLS) แต่ไม่อยู่ใน `GATED_READ_PATHS` (`middleware.ts:18`) จึงอ่านได้โดยไม่ปลดล็อก
3. **[ตั๋ว 05/06 มิติ 2]** L3-01 ส่ง `{texts:[text]}` ไปที่ L1-09 ซึ่งรอ `{text}` และอ่านคำตอบเป็น `data.embeddings[0]` ซึ่ง `EmbeddingResponse` ไม่มี field นี้ ทั้งสองทิศทางไม่ตรงกัน · Edge function ทั้ง 3 ตัวยิง `127.0.0.1:8000` จากใน container ของ edge runtime ซึ่งน่าจะชี้ไปที่ container เอง ไม่ใช่เครื่อง host
4. **[ตั๋ว 04/06 มิติ 6]** `taxonomy-app/utils/api.ts` ไม่มีไฟล์ไหน import แต่เรียก Edge function ที่ไม่มีอยู่ 6 ตัว
5. **[ตั๋ว 03 มิติ 6]** `internal_match.py` ประกาศ `router = APIRouter(...)` ซ้ำ 2 ครั้ง (`:163`, `:435`) ตัวแรกไม่มี route ผูก
6. **[ตั๋ว 06 มิติ 5]** L4-06 อ่าน `imports` ด้วย anon key แต่ AGENTS.md ระบุว่า SELECT ของ `imports` ไม่มีวันผ่านสำหรับ anon จึงน่าจะได้รายการว่างแบบเงียบ (ขณะที่ L2-11 อ่านตารางเดียวกันผ่าน service role)
7. **[ตั๋ว 04 มิติ 4/5/6]** L2-13 และ L2-20 เขียนผ่าน anon key (`DatabaseService`) ซึ่ง RLS น่าจะกั้นแบบเงียบ และทั้งสองเส้นไม่มีผู้เรียก
8. **[ตั๋ว 06 มิติ 5]** `CategorizationStep.tsx:91,128` มีทาง fallback เป็น `_source: 'mock'` เมื่อเรียก backend ไม่สำเร็จ
9. **[ตั๋ว 07]** คำตอบของ L1-01 โฆษณา `/web` ที่ไม่มีอยู่
10. **[ตั๋ว 05 มิติ 6]** websocket ไม่มีผู้รับฝั่ง client เลย ทั้ง 7 แถว

---

## ภาคผนวก ก: ตำแหน่งเอกสารที่อ้างถึงแต่ละ path

กวาดจากทุกไฟล์ `*.md` ในรีโป ยกเว้น `.scratch/`, `node_modules/` และ rule ทั่วไปของสกิล `vercel-react-best-practices` (ตัวอย่างโค้ดทั่วไป ไม่ได้พูดถึง API ของรีโปนี้) ไม่รวมการอ้าง path ของไฟล์ซอร์ส เช่น `src/api/api_server.py` หรือ `.../route.ts`

| path | ไฟล์:บรรทัด |
|---|---|
| `/api/classify/batch` | docs/README_CLASSIFIER.md:83,127,257,325; docs/architecture/SUMMARY_ARCHITECTURE.md:256; docs/development/MIGRATION_PLAN.md:523; docs/development/INTEGRATION_STRATEGY.md:515; docs/development/INTEGRATION_STEPS.md:123; docs/reports/BACKEND_UPDATED.md:114,118,210,263,355 |
| `/api/classify/category` | docs/api/analyze-capabilities.md:5; docs/SYSTEM_OVERVIEW_TH.md:343,422; docs/README_CLASSIFIER.md:38,50,126,196,252,324; .claude/skills/thai-taxonomy-expert/SKILL.md:24,102 (+ สำเนาใน .agents/skills/); docs/architecture/SUMMARY_ARCHITECTURE.md:237,255; docs/architecture/REAL_DATABASE_RELATIONSHIPS.md:136; docs/development/MIGRATION_PLAN.md:522; docs/development/INTEGRATION_STRATEGY.md:246,514; docs/architecture/DATABASE_SCHEMA.md:100; docs/development/INTEGRATION_STEPS.md:68,205,313,446,467; docs/reports/BACKEND_UPDATED.md:78,82,202,221,229,262,281,354 |
| `/api/embed` | docs/api/analyze-capabilities.md:5; docs/api/quick-reference.md:30,43; docs/api/API_CORRECTION.md:29,49; docs/api/API_ARCHITECTURE.md:36; docs/api/api-reference.md:28; docs/SYSTEM_OVERVIEW_TH.md:320,357,366,407; docs/README_CLASSIFIER.md:96,122; .claude/skills/thai-taxonomy-expert/SKILL.md:95; .claude/skills/pgvector-semantic-search/SKILL.md:38,54 (+ สำเนาใน .agents/skills/); docs/architecture/SUMMARY_ARCHITECTURE.md:250; docs/architecture/REAL_DATABASE_RELATIONSHIPS.md:133; docs/development/MIGRATION_PLAN.md:528,542,548; docs/architecture/PROJECT_STRUCTURE.md:31; docs/development/INTEGRATION_STRATEGY.md:425,510,525; docs/reports/STATUS_CHECK_FINAL.md:54,177; docs/architecture/DATABASE_SCHEMA.md:96; docs/development/INTEGRATION_STEPS.md:264,273,311,462; docs/development/EMBEDDING_INTEGRATION.md:39,44,86,124; docs/development/architecture.md:30; docs/reports/BACKEND_UPDATED.md:258; taxonomy-app/DEPLOY_LOCAL_MODEL.md:320,385 |
| `/api/embed/batch` | docs/api/quick-reference.md:44; docs/api/API_CORRECTION.md:29,64; docs/api/api-reference.md:46; docs/SYSTEM_OVERVIEW_TH.md:321; docs/README_CLASSIFIER.md:123; docs/architecture/SUMMARY_ARCHITECTURE.md:251; docs/development/MIGRATION_PLAN.md:529,543; docs/development/INTEGRATION_STRATEGY.md:511; docs/development/INTEGRATION_STEPS.md:312; docs/development/EMBEDDING_INTEGRATION.md:40,101; docs/reports/BACKEND_UPDATED.md:259; taxonomy-app/DEPLOY_LOCAL_MODEL.md:321 |
| `/api/v1/clean` | docs/api/quick-reference.md:47; docs/api/api-reference.md:84; docs/SYSTEM_OVERVIEW_TH.md:173,324; taxonomy-app/docs/IMPORT_ROUTES_FLOW.md:17; taxonomy-app/docs/import-wizard-progress.md:24; taxonomy-app/components/Import/README.md:17 |
| `/api/v1/health` | docs/api/quick-reference.md:32,48; docs/api/API_CORRECTION.md:31; docs/api/api-reference.md:18,93; docs/SYSTEM_OVERVIEW_TH.md:325,501; docs/guides/human-feedback.md:40; docs/development/MIGRATION_PLAN.md:541; docs/reports/test-duplication-analysis.md:25; docs/architecture/PROJECT_STRUCTURE.md:33; docs/reports/STATUS_CHECK_FINAL.md:176; docs/reports/BACKEND_UPDATED.md:257; taxonomy-app/DEPLOY_LOCAL_MODEL.md:88,356 |
| `/api/v1/jobs`, `/api/v1/jobs/{id}` | docs/api/quick-reference.md:33,49; docs/api/API_CORRECTION.md:32; docs/api/api-reference.md:76,102; docs/SYSTEM_OVERVIEW_TH.md:326; docs/guides/human-feedback.md:38; docs/architecture/PROJECT_STRUCTURE.md:34 |
| `/api/v1/results/{job_id}` | docs/api/api-reference.md:107 |
| `/api/v1/learn/verify` | AGENTS.md:43; docs/api/quick-reference.md:34,50; docs/api/API_CORRECTION.md:33; docs/api/api-reference.md:116; docs/SYSTEM_OVERVIEW_TH.md:327; docs/architecture/PROJECT_STRUCTURE.md:35; docs/CURRENT_STATUS.md:174 |
| `/api/v1/learn/retrain`, `/status` | docs/api/API_ARCHITECTURE.md:76,87; docs/CURRENT_STATUS.md:541 |
| `/api/v1/match/single` | docs/api/quick-reference.md:45; docs/api/API_CORRECTION.md:30; docs/api/api-reference.md:59; docs/SYSTEM_OVERVIEW_TH.md:322; docs/guides/human-feedback.md:39; docs/reports/capabilities-summary.md:159 |
| `/api/v1/match/batch` | docs/api/quick-reference.md:46; docs/api/api-reference.md:73; docs/SYSTEM_OVERVIEW_TH.md:323; taxonomy-app/docs/IMPORT_ROUTES_FLOW.md:25; taxonomy-app/docs/import-wizard-progress.md:34,63; taxonomy-app/components/Import/README.md:23 |
| `/api/v1/match/import-dedup` | docs/CURRENT_STATUS.md:500,503; taxonomy-app/docs/import-wizard-progress.md:71,78 |
| `/api/import/commit` | docs/SYSTEM_OVERVIEW_TH.md:169; docs/CURRENT_STATUS.md:194,491; taxonomy-app/docs/import-wizard-progress.md:47 |
| `/api/import/history` | docs/superpowers/plans/2026-09-06-pos-export.md:1270; docs/CURRENT_STATUS.md:187,289 |
| `/api/keyword-rules/[id]` | docs/CURRENT_STATUS.md:170 |
| `/api/pos-export` | docs/superpowers/plans/2026-09-06-pos-export.md (38 จุด, บรรทัด 7–2007); docs/specs/2026-09-06-pos-export-ui-brief.md:52 |
| `/api/products` | taxonomy-app/app/README.md:223,227,228; taxonomy-app/README.md:185,186; docs/CURRENT_STATUS.md:189,258 |
| `/api/recheck` | docs/CURRENT_STATUS.md:174 |
| `/api/settings` | docs/superpowers/plans/2026-09-06-pos-export.md:1270; docs/CURRENT_STATUS.md:187,221,292 |
| `/api/similarity/[id]` | taxonomy-app/app/README.md:253; taxonomy-app/README.md:190,191 |
| `/api/synonyms`, `/api/synonyms/[id]` | taxonomy-app/app/README.md:240,244–247; taxonomy-app/README.md:179–182; docs/CURRENT_STATUS.md:108,220,224 |
| `/api/taxonomy`, `/api/taxonomy/[id]` | taxonomy-app/app/README.md:231,235–238; taxonomy-app/README.md:173–176; taxonomy-app/TESTING.md:153; taxonomy-app/docs/archive_docs/DEPLOYMENT_GUIDE.md:183,184; docs/CURRENT_STATUS.md:220 |
| `/api/unlock` | docs/architecture/REAL_DATABASE_RELATIONSHIPS.md:127,217,218; docs/architecture/DATABASE_SCHEMA.md:76,432 |
| `/api/verify` | docs/SYSTEM_OVERVIEW_TH.md:274; docs/superpowers/plans/2026-09-06-pos-export.md:1177; docs/CURRENT_STATUS.md:170 |
| `/functions/v1/generate-embeddings-local` | docs/reports/STATUS_CHECK_FINAL.md:184; taxonomy-app/DEPLOY_LOCAL_MODEL.md:123 |
| `/functions/v1/hybrid-classification-local` | docs/reports/STATUS_CHECK_FINAL.md:191; taxonomy-app/DEPLOY_LOCAL_MODEL.md:197 |
| `/ws` | docs/development/module-relationships-analysis.md:155 |
| พร็อกซี `/api/fastapi`, `/api/supabase` | AGENTS.md:68; .agents/rules/rules-windows.md:68,69,80,81,95,124,125 |

ไฟล์ที่อ้างพอร์ต `:8000`/`54331` โดยไม่ระบุ path ไม่ได้แจกแจงในภาคผนวกนี้ ตั๋ว 07 ให้ grep `:8000|54331|:3000` เพิ่มเอง (พบใน `.agents/rules/rules-windows.md`, `README.md`, `docs/CURRENT_STATUS.md` ฯลฯ รวม 84 ไฟล์เมื่อนับรวม path)
