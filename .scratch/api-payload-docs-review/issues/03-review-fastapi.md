# 03: รีวิวชั้น FastAPI

**Labels:** `wayfinder:research` (AFK — สกิล research ไม่ได้ติดตั้ง ทำ inline)
**Parent:** [แผนที่](../map.md)
**Blocked by:** [01: สำรวจรายการ](01-inventory.md), [02: เกณฑ์ผ่าน](02-pass-criteria.md)
**Assignee:** Antigravity
**Status:** closed

## Question

ทุกแถวชั้น 1 (FastAPI `:8000`) ใน [`matrix.md`](../matrix.md) ผ่านหรือไม่ผ่านมิติ 1, 3, 4, 5, 6, 7, 8 ตามเกณฑ์ของตั๋ว 02

(มิติ 2 ข้ามรอยต่ออยู่ตั๋ว 06 · มิติ 1 ฝั่งเอกสารภายนอกโค้ดอยู่ตั๋ว 07 — ตั๋วนี้ดูมิติ 1 เฉพาะ docstring/คอมเมนต์ใน `src/api/` และ OpenAPI ที่ FastAPI สร้างเอง)

## วิธี

- อ่านโค้ด router + `models.py` + `dependencies.py`
- endpoint อ่านอย่างเดียว: ยิงจริงกับ FastAPI ในเครื่อง (`127.0.0.1:8000` ไม่ใช่ `localhost`) — ถ้า server ไม่รัน ให้เปิดเองตามวิธีใน AGENTS.md
- endpoint เขียนข้อมูล: อ่านโค้ด + เทสต์ใน `tests/` เท่านั้น **ห้ามยิง**
- เทียบ `/openapi.json` ที่ server คืนจริงกับ Pydantic model ในโค้ด

## เกณฑ์ว่าเสร็จ

- [x] ทุกช่องมิติของแถวชั้น 1 มีค่า (กรอกครบทั้ง 24 แถวใน `matrix.md` โดยมิติ 2 กำกับ "รอตั๋ว 06")
- [x] ทุกข้อค้นพบอยู่ใน [`findings.md`](../findings.md) พร้อม: ระดับ P0–P3, ป้าย `ยิงจริงแล้ว`/`อ่านโค้ดอย่างเดียว`, ไฟล์:บรรทัด, หลักฐาน (คำสั่ง + ผลลัพธ์จริง), คำตัดสินที่เสนอ + เหตุผล
- [x] P0 ที่เจอถูกรายงานผู้ใช้ทันทีตามข้อยกเว้นใน map Notes (ยืนยัน F-002 เรื่อง Windows Firewall Allow บน Public profile สำเร็จ ไม่พบ P0 ตัวใหม่เพิ่ม)

## Resolution

- **ยืนยัน Firewall สำหรับ F-002 (P0):** ตรวจสอบผ่าน PowerShell พบว่า PID 22936 (`python.exe` 3.14) กำลัง Listen `0.0.0.0:8000` และ Windows Firewall มี Inbound Rule ชื่อ `TCP Query User...` ในสถานะ `Allow` บน Profile `Public` ทุกพอร์ต ทำให้เครื่องอื่นในวง Wi-Fi เดียวกันยิงเข้าเส้นเขียนข้อมูลและคำนวณหนักได้โดยตรง ข้าม `middleware.ts` 100% (ข้อมูลส่งต่อรอตั๋ว 09)
- **ยิงจริง endpoint อ่านอย่างเดียว:** ยิงทดสอบครบทุกเส้นผ่าน `127.0.0.1:8000` (L1-01, L1-02, L1-03, L1-11, L1-12, L1-13, L1-16, L1-17, L1-19, L1-24) ทุกเส้นตอบสนองและคืนรูปแบบ error 404 สม่ำเสมอ `{detail: ...}`
- **ข้อค้นพบใหม่ (F-003 ถึง F-010):**
  - **F-003 (P1):** `POST /api/v1/match/batch` ดัก `HTTPException(400)` ด้วย `except Exception as e:` ทำให้ input เกินกลายเป็น 500
  - **F-004 (P1):** `POST /api/v1/upload/csv` ไม่ตรวจขนาดไฟล์ (เสี่ยง DoS) และไม่ sanitize ชื่อไฟล์ (เสี่ยง path traversal)
  - **F-005 (P1):** `POST /api/v1/match/single` mutate pipeline config บน singleton ทำให้เกิด race condition
  - **F-006 (P2):** `GET /` โฆษณา `/web` ซึ่งคืน 404 เนื่องจากไม่มีโฟลเดอร์ `src/api/web`
  - **F-007 (P2):** สัญญา error 500 รั่วไหล `str(e)` เข้า `{detail}` ใน 14 จาก 24 endpoint
  - **F-008 (P3):** ชุด Legacy Matching & Config 9 เส้นเป็นเส้นตาย ไม่มีผู้เรียกใน UI
  - **F-009 (P3):** Response model บางตัวใช้ camelCase (`ProductMatchResult`, `FileDeduplicationStats`) และซ้อน envelope ไม่ตรงกัน
  - **F-010 (P3):** ขาด automated tests ชี้ตรงถึง 8 endpoint
- **อัปเดตไฟล์:** กรอกครบทุกมิติใน [`matrix.md`](../matrix.md) (L1-01 ถึง L1-24) และบันทึกข้อค้นพบทั้งหมดลงใน [`findings.md`](../findings.md) เรียบร้อย
