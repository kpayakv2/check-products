# 🚀 **Quick Reference - Thai Product AI (v4.0)**

## 💡 **คำสั่งลัดสำหรับการรันระบบ**

```bash
# วิธีที่ 1: ดับเบิลคลิก (แนะนำ)
START_PHAYAK.bat   # รัน Backend + Frontend พร้อมกันอัตโนมัติ

# วิธีที่ 2: รันแยก
# Backend — FastAPI Port 8000
.venv\Scripts\python src\api\api_server.py

# Frontend — Next.js Port 3000
cd taxonomy-app && npm run dev
```

---

## 🗂️ **โครงสร้าง API (Modular v4.0)**

```
src/api/
├── api_server.py       ← Entry Point (รันตัวนี้)
├── models.py           ← Pydantic Schemas
├── dependencies.py     ← Pipeline & Model Init
├── websockets.py       ← Real-time Updates
├── services/
│   └── background_jobs.py
└── routers/
    ├── embed.py        ← /api/embed/*
    ├── match.py        ← /api/v1/match/*
    ├── system.py       ← /api/v1/health, /config, /clean
    ├── jobs.py         ← /api/v1/jobs, /results
    └── learn.py        ← /api/v1/learn/verify
```

---

## 🎯 **Endpoints ที่ใช้บ่อย**

| งานที่ต้องการ | Endpoint | Method | Router |
|---------------|----------|--------|--------|
| **สร้าง Embedding (หลัก)** | `/api/embed` | POST | `embed.py` |
| **สร้าง Embedding (Batch)** | `/api/embed/batch` | POST | `embed.py` |
| **จับคู่สินค้า (Single)** | `/api/v1/match/single` | POST | `match.py` |
| **จับคู่สินค้า (Batch)** | `/api/v1/match/batch` | POST | `match.py` |
| **ล้างชื่อสินค้าภาษาไทย** | `/api/v1/clean` | POST | `system.py` |
| **เช็คสถานะระบบ** | `/api/v1/health` | GET | `system.py` |
| **เช็คสถานะ Job** | `/api/v1/jobs/{id}` | GET | `jobs.py` |
| **สอน AI จากการตรวจ** | `/api/v1/learn/verify` | POST | `learn.py` |
| **Swagger UI** | `/docs` | GET | — |

---

## 🏗️ **การตรวจซ้ำและสอน AI (Deduplication)**

```bash
# ตรวจสอบไฟล์สินค้าใหม่
python scripts/complete_deduplication_pipeline.py --input new.csv --mode analyze

# ตรวจสอบงานที่ค้าง (Human Review)
python scripts/complete_deduplication_pipeline.py --input new.csv --mode review

# สั่งให้ AI เรียนรู้จากสิ่งที่คนตรวจ
python scripts/complete_deduplication_pipeline.py --mode train
```

---

## 🔍 **การตั้งค่า (Configuration)**

- **Port**: 8000 (Default)
- **Model**: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim)
- **Weights**: Keyword 60% / Embedding 40%
- **Env**: `taxonomy-app/.env.local`

---

## 🛠️ **Troubleshooting (เช็คเมื่อมีปัญหา)**

| ปัญหา | วิธีแก้ |
|-------|--------|
| **Error Port 8000** | เช็คว่ารัน `src/api/api_server.py` ซ้อนกันหรือไม่ |
| **AI ไม่แม่น** | ตรวจสอบว่าได้รันสคริปต์ `download_models.py` หรือยัง |
| **DB เชื่อมต่อไม่ได้** | ตรวจสอบไฟล์ `.env.local` ของ Supabase ใน `taxonomy-app/` |

---

**📅 Last Updated**: 24 พฤษภาคม 2569 (v4.0 — Modular Architecture)
**⚡ พร้อมใช้งาน!** รันผ่าน `START_PHAYAK.bat` ได้เลย
