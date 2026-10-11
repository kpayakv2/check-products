# ✅ API Architecture — บันทึกการแก้ไข (v4.0)

**อัปเดต:** 24 พฤษภาคม 2569

---

## 📜 ประวัติการแก้ไข

### v3.0 (2025-10-04)
พบว่าเอกสารเดิมอ้างถึง **embed_service.py** (Flask Port 5000) ที่ไม่มีอยู่จริง
แก้ไขให้ถูกต้องว่า Embed API อยู่ใน `api_server.py` (FastAPI Port 8000)

### v4.0 (2026-05-24) ← ปัจจุบัน
`api_server.py` ถูก Refactor จาก Monolith (840+ บรรทัด) เป็นโครงสร้าง Modular

---

## ✅ โครงสร้างปัจจุบัน (v4.0 — Modular)

```
src/api/
├── api_server.py           ← Entry Point หลัก
├── models.py               ← Pydantic Schemas
├── dependencies.py         ← Global State, init pipeline/model
├── websockets.py           ← ConnectionManager (WebSocket)
├── services/
│   └── background_jobs.py  ← Async Batch Processing
└── routers/
    ├── embed.py            ← POST /api/embed, /api/embed/batch
    ├── match.py            ← POST /api/v1/match/single, /batch
    ├── system.py           ← GET/POST /api/v1/health, /config, /clean, /upload
    ├── jobs.py             ← GET /api/v1/jobs, /results
    └── learn.py            ← POST /api/v1/learn/verify
```

**วิธีรัน:**
```bash
START_PHAYAK.bat              # แนะนำ (เปิด Backend + Frontend พร้อมกัน)
# หรือ
.venv\Scripts\python src\api\api_server.py
```

---

## 🧪 ทดสอบ Embed API

```bash
# Test Single Embedding
curl -X POST http://127.0.0.1:8000/api/embed \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"กล่องล็อค 560 มล\"}"

# Expected Response:
{
  "embedding": [0.234, -0.567, 0.123, ...],
  "dimension": 384,
  "model": "paraphrase-multilingual-MiniLM-L12-v2",
  "processing_time": 0.045
}
```

```bash
# Test Batch Embedding
curl -X POST http://127.0.0.1:8000/api/embed/batch \
  -H "Content-Type: application/json" \
  -d "{\"texts\": [\"กล่อง 560\", \"ถัง 1000\", \"ขวด 500\"]}"
```

---

## 📊 สรุปความแตกต่างจากเดิม

| รายการ | v3.0 (เก่า) | v4.0 (ปัจจุบัน) |
|--------|-------------|-----------------|
| **โครงสร้าง** | Monolith (1 ไฟล์ 840+ บรรทัด) | Modular (10 ไฟล์ แยก Router) |
| **Entry Point** | `api_server.py` (Root) | `src/api/api_server.py` |
| **Embed Endpoint** | ใน `api_server.py` โดยตรง | `src/api/routers/embed.py` |
| **Match Endpoint** | ใน `api_server.py` โดยตรง | `src/api/routers/match.py` |
| **embed_service.py** | ~~ไม่มี (อ้างอิงผิด)~~ | ยืนยัน: ไม่มีไฟล์นี้ |
| **Docker** | มี Dockerfile.backend | ลบออกแล้ว (ไม่ได้ใช้) |

---

**ทุกอย่างถูกต้องและพร้อมใช้งาน** ✅
