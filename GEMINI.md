# GEMINI.md — Thai Product Taxonomy Manager & Similarity Checker
*(ระบบจัดการหมวดหมู่สินค้าไทยและตรวจสอบความคล้ายคลึง)*

## 📋 Project Overview
AI-powered system สำหรับจัดการ Thai Product Taxonomy และ Similarity Matching (Deduplication)
- **Backend:** Python + FastAPI (Embedding Provider, 384-dim)
- **Frontend:** Next.js + Supabase (Edge Functions + pgvector)
- **Algorithm:** Hybrid (Keyword 60% + Embedding 40%) → Accuracy ≥ 72%

---

## 🏛️ Project Constitution (กฎเหล็ก)

| # | กฎ | รายละเอียด |
|---|-----|------------|
| 1 | **โครงสร้าง src/ เท่านั้น** | ใช้โครงสร้างโฟลเดอร์แบบใหม่ ย้ายโค้ดหลักเข้า `src/` เพื่อความเป็นระเบียบ |
| 2 | **Supabase เท่านั้น** | ใช้ Supabase เป็น Single Source of Truth (ลบการพึ่งพา SQLite/human_feedback.db) |
| 3 | **384-dim เท่านั้น** | `paraphrase-multilingual-MiniLM-L12-v2`, column type `vector(384)` |
| 4 | **No Tailwind นอก taxonomy-app/** | ห้ามใช้ Tailwind CSS นอกโฟลเดอร์ `taxonomy-app/` |
| 5 | **Test ก่อนเสมอ** | ห้ามแก้โค้ด/DB โดยไม่รัน Pytest/Jest |
| 6 | **Benchmark ≥ 72%** | การแก้ Similarity Algorithm ต้องรักษา F1-score ไว้ |
| 7 | **Smart Testing** | ต้องรันการทดสอบตาม Smart Testing Matrix ครบทั้ง 5 ขั้นตอนก่อน Commit หรือ Deploy เสมอ |
| 8 | **Standard MCP Tools** | ต้องใช้ชุดเครื่องมือมาตรฐาน 6 ตัว และ Socraticode ในการวิเคราะห์และแก้ไขงานเสมอ |
| 9 | **บังคับใช้ 127.0.0.1** | ห้ามใช้ `localhost` บน Win32 ให้ใช้ `127.0.0.1` เพื่อป้องกันปัญหา Network/Socket |

---

## 🧠 Smart Testing Matrix & Tooling (ระบบทดสอบอัจฉริยะ)

เพื่อรับประกันคุณภาพ ความถูกต้องของภาษาไทย และความเสถียรบนสภาพแวดล้อม Windows (Win32) ห้ามข้ามขั้นตอน **Smart Testing Matrix** 5 ขั้นตอนนี้เด็ดขาด:

1. **Sequential Thinking (คิดวิเคราะห์เชิงลึก):** ใช้สำหรับวางแผนการพัฒนา วิเคราะห์ผลกระทบ (Blast Radius) และครอบคลุม Edge Cases ทั้งหมดก่อนเริ่มลงมือเขียนโค้ด
2. **Postgres MCP (ตรวจสอบ DB จริง):** ทำการทดสอบแบบ **No-Mock** ตรวจสอบและแก้ไขข้อมูลในฐานข้อมูลจริงเพื่อจำลองสภาวะแวดล้อมที่แม่นยำ รวมถึงจัดการข้อมูล Seed Data
3. **Puppeteer + Domscribe (Visual Integrity):** ทำการตรวจสอบความถูกต้องของหน้าจอ (Antigravity Check) โดยเฉพาะการแสดงผลภาษาไทย การจัดเลย์เอาต์ และความยืดหยุ่นของ Responsive UI
4. **Filesystem MCP (วิเคราะห์ข้อมูล/Log):** ตรวจสอบไฟล์การตั้งค่า (Config) รันคำสั่งวิเคราะห์ Log เพื่อหาสาเหตุที่แท้จริงของปัญหา
5. **Memory MCP (บันทึกและจดจำรูปแบบ Bug):** จดจำรูปแบบ Bug และ Best Practices สำคัญในการพัฒนา เช่น **การบังคับใช้ `127.0.0.1` แทน `localhost` ในระบบเครือข่ายของ Windows (Win32)** เพื่อป้องกันปัญหา Socket และ CORS

### 🛠️ ชุดเครื่องมือ MCP มาตรฐาน 6 ตัวหลัก
เพื่อให้การเข้าถึงข้อมูลและการรันคำสั่งดำเนินไปอย่างถูกต้องภายใต้กรอบสิทธิ์การใช้งานของโปรเจกต์ นักพัฒนาและ AI เอเจนต์จะใช้เครื่องมือมาตรฐานเหล่านี้เป็นหลัก:
- `postgres` (จัดการ/ตรวจสอบ DB)
- `domscribe` (ตรวจสอบและทำความเข้าใจโครงสร้าง UI)
- `puppeteer` (จำลองการทดสอบบน Browser จริง)
- `filesystem` (อ่าน/เขียนและจัดการไฟล์งาน)
- `memory` (จดจำบริบทและกฎเฉพาะทาง)
- `sequential-thinking` (ช่วยวางแผนและวิเคราะห์ปัญหาที่ซับซ้อน)

---

## 🗂️ Index — Rules, Skills & Workflows

### 📐 Rules (`.agents/rules/`)
| ไฟล์ | ใช้เมื่อ |
|------|---------|
| [rules-thai-product.md](.agents/rules/rules-thai-product.md) | จัดการข้อมูลสินค้าไทย, Text Processing |
| [rules-supabase.md](.agents/rules/rules-supabase.md) | สร้าง/แก้ตาราง, Query, Edge Functions |
| [rules-antigravity.md](.agents/rules/rules-antigravity.md) | แก้ไข Frontend/UI, ตรวจสอบ Layout ภาษาไทย |
| [rules-windows.md](.agents/rules/rules-windows.md) | รันคำสั่ง PowerShell, ตั้งค่า Port, **LAN Access** |
| [rules-git-hygiene.md](.agents/rules/rules-git-hygiene.md) | ก่อน git commit/push |
| [rules-ai-agent.md](.agents/rules/rules-ai-agent.md) | กฎของ AI เอเจนต์ ป้องกันการคิดไปเอง / ลืมบริบท / ประเมินผลกระทบ |

### 🥋 Skills (`.agents/skills/`)
| ไฟล์ | ใช้เมื่อ |
|------|---------|
| [thai-taxonomy-expert](.agents/skills/thai-taxonomy-expert/SKILL.md) | ออกแบบ/แก้ไข taxonomy_nodes, keyword_rules |
| [data-cleaner](.agents/skills/data-cleaner/SKILL.md) | Normalize ชื่อสินค้า, จัดการ noise/หน่วยวัด |
| [pgvector-semantic-search](.agents/skills/pgvector-semantic-search/SKILL.md) | pgvector, HNSW index, vector search (⚠️ ใช้ 384-dim) |
| [vercel-react-best-practices](.agents/skills/vercel-react-best-practices/SKILL.md) | เขียน/ปรับปรุง React/Next.js components |

### 🔄 Workflows (`.agents/workflows/`)
| ไฟล์ | ใช้เมื่อ |
|------|---------|
| [smart_impact_workflow.md](.agents/workflows/smart_impact_workflow.md) | ก่อนแก้ไขใดๆ — วิเคราะห์ผลกระทบ |
| [workflow-new-feature.md](.agents/workflows/workflow-new-feature.md) | พัฒนา Feature ใหม่ตั้งแต่ต้น |
| [workflow-analyze-db.md](.agents/workflows/workflow-analyze-db.md) | Dump + วิเคราะห์ Database Schema |
| [workflow-antigravity-verification.md](.agents/workflows/workflow-antigravity-verification.md) | ตรวจสอบ UI ภาษาไทยหลังแก้ Frontend |

---

## 🌐 LAN Access (ใช้งาน 2026-05-30)

**Server IP:** `192.168.1.80` | เข้าใช้งาน: `http://192.168.1.80:3000`

```
เครื่อง LAN (browser)
  └─▶ http://192.168.1.80:3000            [Next.js : bind 0.0.0.0]
        ├─▶ /api/fastapi/*  → :8000       [FastAPI AI Engine]
        └─▶ /api/supabase/* → :54331      [Supabase PostgreSQL]
```

**กฎ:** `NEXT_PUBLIC_*` ต้องเป็น relative path เสมอ → ดูรายละเอียดใน [rules-windows.md](.agents/rules/rules-windows.md#-lan-access-บันทึก-2026-05-30)

---

## 🏗️ System Architecture

```
[User/API Request]
      │
      ▼
[Supabase Edge Function: hybrid-classification-local]
      │
      ├─── Keyword Match (60%) ─── keyword_rules + taxonomy_nodes.keywords
      │
      └─── Embedding Match (40%) ── FastAPI (/api/classify/category)
                                         ├── /api/embed (Generate Embedding 384-dim)
                                         └── pgvector <=> Vector Similarity Search (Supabase)
                                         (Running on 127.0.0.1:8000)
```

## 📂 Key Directory Structure
| Path | ประโยชน์ |
|------|---------|
| `/src/api` | Python API, `api_server.py`, `routers/` |
| `/src/core` | Logic AI หลัก, `fresh_implementations.py`, `models.py` |
| `/src/services` | Service layer, `ml_feedback_learning.py` |
| `/scripts` | CLI สคริปต์สำหรับจัดการข้อมูล |
| `/taxonomy-app` | Next.js App, UI, Supabase Client |
| `/docs` | Architecture, API docs, DB Schema, Reports |
| `/supabase` | Edge Functions, Migrations |
| `/tests` | Pytest Unit/Integration tests |
| `/.agents` | Rules, Skills, Workflows (AI Agent config) |
