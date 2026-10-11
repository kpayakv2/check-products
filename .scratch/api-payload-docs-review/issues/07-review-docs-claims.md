# 07: ตรวจข้อกล่าวอ้างในเอกสารทุกข้อเทียบของจริง

**Labels:** `wayfinder:research` (AFK — สกิล research ไม่ได้ติดตั้ง ทำ inline)
**Parent:** [แผนที่](../map.md)
**Blocked by:** [01: สำรวจรายการ](01-inventory.md)
**Assignee:** Antigravity
**Status:** closed

## Question

ทุกข้อกล่าวอ้างเรื่อง endpoint/payload/พอร์ต/วิธีเรียก ในเอกสารที่ตั๋ว 01 ไล่ไว้ (คอลัมน์ "เอกสารที่อ้างถึง" + ตารางย่อย "ไม่ตรงกับแถวใด" ใน [`matrix.md`](../matrix.md)) ยังจริงอยู่ไหม — มิติ 1 ฝั่งเอกสาร

ขอบเขต: `docs/api/*` (5 ไฟล์), `AGENTS.md`, `README.md`, `docs/CURRENT_STATUS.md`, `.agents/rules/*` และเอกสารอื่นที่ตั๋ว 01 เจอ (docstring/คอมเมนต์ในโค้ดอยู่ตั๋ว 03–05)

ต่อไฟล์ใน `docs/api/` ให้บันทึกด้วยว่ามี inbound link จากทางเดินที่มีชีวิต (`README.md` → `docs/INDEX.md` → ...) หรือไม่ — ไฟล์ที่ไม่มีใครเดินผ่านแต่โกหกเต็มไฟล์ ควรได้คำตัดสินคนละแบบกับไฟล์ที่คนอ่านจริง

## ข้อแยกแยะ (สืบจากแผนที่ก่อน)

รายงานที่ลงวันที่ใน `docs/reports/` บันทึกสิ่งที่จริง ณ ตอนนั้น — ไม่นับเป็นข้อค้นพบ เว้นแต่มีทางเดินที่มีชีวิตชี้เข้าไปเหมือนเป็นเอกสารปัจจุบัน (ตัดสินแบบเดียวกับตั๋ว 03 ของ `../rules-workflows-memory-conflicts/`)

## เกณฑ์ว่าเสร็จ

- [x] ทุกข้อกล่าวอ้างมีสถานะ: จริง / ผิด (+ ความจริงคืออะไร) / ตรวจไม่ได้ (+ เหตุผล)
- [x] ข้อที่ผิดอยู่ใน [`findings.md`](../findings.md) ระดับ P2 เป็นหลัก (ยกระดับได้ถ้าทำให้ทำผิดเชิงความปลอดภัย)
- [x] ทุกไฟล์ใน `docs/api/` มีคำตัดสินที่เสนอระดับไฟล์ (เก็บ+แก้ / ยุบรวม / ลบ) พร้อมสถานะ inbound link
- [x] ช่องมิติ 1 ในเมทริกซ์ถูกเติมสำหรับทุกแถวที่มีเอกสารอ้าง (แถวที่ไม่มีเอกสารอ้างเลย ใส่ `ไม่เกี่ยว` ไม่ใช่ `ผ่าน`)

## Resolution

ได้ตรวจสอบข้อกล่าวอ้างทุกข้อในเอกสารทั่วทั้งรีโปเทียบกับโค้ดและระบบจริง:

### 1. ช่องมิติ 1 ใน `matrix.md`
- ปรับทุกแถวที่ไม่มีเอกสารอ้างอิง (`—`) จากเดิมที่เคยใส่ `ผ่าน` ให้เป็น `ไม่เกี่ยว` ตามเกณฑ์อย่างครบถ้วน (รวม 29 แถวใน Layer 1, 2, 4, 5)

### 2. ผลการตรวจสอบไฟล์ใน `docs/api/` (5 ไฟล์)

| ไฟล์ | Inbound Link จากทางเดินมีชีวิต | สถานะความจริง | คำตัดสินที่เสนอระดับไฟล์ |
|---|---|---|---|
| `docs/api/api-reference.md` | **มี** (`docs/INDEX.md:17`, `PROJECT_STRUCTURE.md`) | **จริงบางส่วน** — เส้นทางหลักรันได้จริง แต่ตัวอย่าง payload `POST /api/v1/clean` ผิดเป็น array (`{ texts: [...] }` ซึ่งโค้ดจริงรับ `{ text: str }` ทำให้ติด 422), ละเลย router สำคัญ `classify.py` และ `internal_match.py`, อ้างอิงสคริปต์เก่า `complete_deduplication_pipeline.py` (SQLite), ขาด Next.js APIs ทั้งหมด | **เก็บ + แก้ไขครั้งใหญ่ (Keep + Rewrite)** เป็น Single Source of Truth ของ API docs: แก้ payload clean, เพิ่มเส้นทางที่ใช้จริงทั้งหมด และลบ CLI เก่า |
| `docs/api/API_ARCHITECTURE.md` | **มี** (`docs/INDEX.md:18`, `CURRENT_STATUS.md:595`) | **เท็จกว่า 70%** — อ้างหลักการ "Edge Function First / Business Logic ทั้งหมดเก็บใน Edge Functions", บัญญัติกฎเหล็กปลอมว่าห้ามเขียน logic ใน FastAPI, อ้างฟังก์ชัน `hybrid-search` ที่ไม่มีอยู่, อ้าง path ผิด `/api/v1/match/internal/scan` (ของจริงคือ `scan-internal`) | **เขียนใหม่ทั้งหมด (Rewrite)** ให้สะท้อนสถาปัตยกรรมจริง (Next.js rewrites -> FastAPI + Supabase DB) หรือย้ายรวมเข้า `docs/architecture/` |
| `docs/api/quick-reference.md` | **ไม่มี** (มีเพียงใน tree ของ `PROJECT_STRUCTURE.md`) | **ซ้ำซ้อน 90%** กับ `api-reference.md` และขาด router ปัจจุบัน | **ยุบรวมและลบออก (Consolidate & Delete)** ยุบส่วนสรุปคำสั่งเข้า `api-reference.md` แล้วลบไฟล์นี้ทิ้ง |
| `docs/api/API_CORRECTION.md` | **ไม่มี** (0 inbound links จากทุกไฟล์) | **จริงในอดีต (บันทึกประวัติ)** บันทึกการ refactor FastAPI เป็น modular เมื่อ พ.ค. 2569 ขาด router ปัจจุบัน | **ย้ายเข้าคลังประวัติหรือลบ (Archive / Delete)** ไปที่ `docs/archive_docs/` |
| `docs/api/analyze-capabilities.md` | **ไม่มี** (มีเพียงใน tree ของ `PROJECT_STRUCTURE.md`) | **เท็จ** เรื่องการบันทึก feedback ลง `product_deduplication.db` (SQLite) ขัดกับกฎเหล็กข้อ 2 ของโปรเจกต์ | **ลบหรือย้ายเข้าคลังประวัติ (Delete / Archive)** เนื่องจากเนื้อหาเรื่อง Hybrid Algorithm มีอยู่ใน `AGENTS.md` แล้ว |

### 3. ผลการตรวจสอบข้อกล่าวอ้างที่ไม่ตรงกับแถวใด (Section 2 ใน `matrix.md`)

| path / ข้อกล่าวอ้าง | ไฟล์ที่อ้าง | สถานะ | ความจริง / หมายเหตุ |
|---|---|---|---|
| `/api/import/process` | `SYSTEM_OVERVIEW_TH.md`, `app/README.md`, `DATABASE_SCHEMA.md`, `CURRENT_STATUS.md` | **ผิด** (ใน `SYSTEM_OVERVIEW_TH` และ `CURRENT_STATUS` เป็นบันทึกประวัติ) | ไม่มี route นี้ในระบบจริง ถูกแทนที่ด้วย `POST /api/import/commit` ใน `WizardTab.tsx` |
| `/api/import/process-local` | `app/README.md`, `STATUS_CHECK_FINAL.md`, `DEPLOY_LOCAL_MODEL.md`, `REAL_DATABASE_RELATIONSHIPS.md` | **ผิด** (`STATUS_CHECK_FINAL` เป็นบันทึกประวัติ) | ไม่มี route นี้ |
| `/api/import/process-storage` | `app/README.md` | **ผิด** | ไม่มี route นี้ |
| `/api/import/pending` | `app/README.md`, `import-wizard-progress.md` | **ผิด** | ไม่มี route นี้ |
| `/api/import/approve` | `app/README.md`, `import-wizard-progress.md` | **ผิด** | ไม่มี route นี้ |
| `/api/products/[id]/review` | `app/README.md`, `taxonomy-app/README.md`, `CURRENT_STATUS.md` | **ผิด** (`CURRENT_STATUS` เป็นบันทึกประวัติ) | ไม่มี route นี้ หน้า `/products` ปรับเป็นแบบอ่านอย่างเดียวแล้ว |
| `/api/status` | `tests/README.md`, `module-relationships-analysis.md`, `docs/reports/test-*.md` | **ผิด** (ใน `tests/README.md` เป็นตัวอย่าง Old Style, ใน reports เป็นบันทึกประวัติ) | ไม่มี route นี้ ของจริงคือ `/api/v1/health` |
| `/api/batch`, `/api/match` | `module-relationships-analysis.md` | **ผิด** | ไม่มี route นี้ ของจริงอยู่ใต้ `/api/v1/match/*` |
| `/api/v1/match/internal/scan` | `docs/api/API_ARCHITECTURE.md` | **ผิด** (path คลาดเคลื่อน) | ของจริงคือ `/api/v1/match/scan-internal` (L1-18) |
| `/web` (FastAPI static web) | คำตอบของ L1-01 (`api_server.py:98`), `tests/integration/test_api_client.py:171` | **ผิด** | ไม่มีโฟลเดอร์ `src/api/web/` จึงไม่ถูก mount (บันทึกใน #F-006) |
| `WS /ws` ฝั่ง client | `module-relationships-analysis.md` | **ผิด** | ไม่มี client ฝั่ง frontend ต่อเลย (บันทึกใน #F-020, #F-021) |
| Edge `category-suggestions`, `generate-embeddings`, `hybrid-search`, `exec-sql` | `taxonomy-app/supabase/functions/README.md` | **ผิด** | ไม่มีทั้ง 4 ตัว (`exec-sql` ถูกลบตาม AGENTS.md, `generate-embeddings` ในเอกสารอ้าง OpenAI ขัดกฎเหล็ก) |
| Edge `hybrid-classification` | `MIGRATION_PLAN.md:604` | **ผิด** (ชื่อคลาดเคลื่อน) | ของจริงในโฟลเดอร์ชื่อ `hybrid-classification-local` |
| โค้ดเรียก Edge 6 ตัวผี | `taxonomy-app/utils/api.ts` | **ผิด** (โค้ดตาย) | ฟังก์ชันใน Edge ไม่มีอยู่จริง และไฟล์ `utils/api.ts` ไม่มีใคร import |
| โค้ดทดสอบเรียก Edge ผี | `test_functions.js`, `scripts/test-api.js` | **ผิด** | ยิงเข้าฟังก์ชันที่ไม่มีอยู่จริง |
| `DatabaseService.getSynonymsByTerm`, `getLemmasByCategory` | `taxonomy-app/data/README_DATASET.md`, `archive_docs/setup.md` | **ผิด** | ไม่มีเมธอดเหล่านี้ใน `DatabaseService` จริง |

### 4. การตรวจสอบเอกสารหลัก (Constitution & Core Docs)
- **`AGENTS.md`**: ตรวจสอบครบทุกข้อกล่าวอ้าง (พอร์ต 54331, 54325, 8000, rewrites `/api/fastapi`, `/api/supabase`, `/unlock`, `/api/v1/learn/verify`, RLS policies, test counts) **ถูกต้องตรงกับระบบจริง 100%**
- **`README.md` (root)**: ถูกต้องเกือบทั้งหมด มีจุดผิดจุดเดียวคือข้อความบรรทัด 15 ที่ระบุว่า Edge Functions ประมวลผล AI Logic (ซึ่งของจริงทำใน FastAPI)
- **`.agents/rules/*` (`rules-windows.md`, `rules-supabase.md`)**: พอร์ตและการตั้งค่า network rewrites ถูกต้องตรงกับความจริง 100%
- **`docs/CURRENT_STATUS.md`**: เป็น session log บันทึกประวัติและตัวเลขจริงตามกาลเวลา ถูกต้องตามบทบาท

### 5. ข้อค้นพบใหม่ในตั๋ว 07 (บันทึกลง findings.md แล้ว)
- **#F-028 (P2):** เอกสารใน `docs/api/*` มีความขัดแย้งเชิงสถาปัตยกรรม (อ้าง Edge Function First, อ้าง SQLite, ตัวอย่าง payload `/api/v1/clean` ผิดประเภท)
- **#F-029 (P2):** เอกสาร Next.js (`app/README.md`, `taxonomy-app/README.md`, `DATABASE_SCHEMA.md`) โฆษณา API routes นำเข้าและรีวิวสินค้าที่ไม่มีอยู่จริง 6 เส้น
- **#F-030 (P2):** `taxonomy-app/supabase/functions/README.md` โฆษณา Edge Functions ผี 4 ตัว และแนะนำ OpenAI โมเดลที่ขัดต่อกฎเหล็กโปรเจกต์
- **#F-031 (P3):** สคริปต์และโค้ดตัวอย่างใน `taxonomy-app/` อ้างอิง Edge Functions และ DatabaseService methods ที่ไม่มีอยู่จริง
