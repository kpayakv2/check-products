# 🛒 Thai Product Taxonomy Manager & Similarity Checker
*(ระบบจัดการหมวดหมู่สินค้าไทยและตรวจสอบความคล้ายคลึง)*

AI-powered system สำหรับจัดการหมวดหมู่สินค้าไทย (Taxonomy) และตรวจสอบความซ้ำซ้อนของสินค้า (Deduplication) โดยใช้เทคโนโลยี Hybrid AI (Keyword + Vector Embedding)

---

## 🏗️ System Architecture (สถาปัตยกรรมระบบ)

ระบบประกอบด้วย 3 ส่วนหลักที่ทำงานร่วมกัน:

1.  **Frontend (Next.js):** UI สำหรับจัดการ Taxonomy, Review สินค้า และระบบนำเข้าข้อมูล (Import Wizard)
2.  **Supabase (Backend-as-a-Service):** 
    *   **PostgreSQL:** เก็บข้อมูลสินค้าและหมวดหมู่ พร้อม `pgvector` สำหรับค้นหาความคล้ายคลึง
    *   **Edge Functions:** ประมวลผล AI Logic (Hybrid Classification) และ Search
    *   **Storage:** เก็บไฟล์ CSV ที่นำเข้า
3.  **FastAPI (AI Engine):** ให้บริการ Vector Embeddings (384-dim) โดยใช้โมเดล `paraphrase-multilingual-MiniLM-L12-v2` แบบ Local (ไม่ต้องเสียค่า API)

---

## 🚀 Quick Start (การเริ่มต้นใช้งานอย่างรวดเร็ว)

### 1. ติดตั้ง Dependencies
```bash
# ติดตั้ง Python dependencies (สำหรับ AI Engine)
pip install -r requirements.txt

# ติดตั้ง Node.js dependencies (สำหรับ UI)
cd taxonomy-app
npm install
```

### 2. รันระบบ (Local Development)
1.  **Start Supabase:** `supabase start` (ต้องติดตั้ง Supabase CLI)
2.  **Start AI Engine:** `python api_server.py` (รันที่ port 8000)
3.  **Start Frontend:** `npm run dev` (รันที่ port 3000)

---

## 📊 AI Capabilities (ขีดความสามารถของ AI)

*   **Hybrid Classification:** ผสมผสาน Keyword (60%) และ Embedding (40%) เพื่อความแม่นยำสูงสุด
*   **Accuracy:** รักษามาตรฐานความแม่นยำที่ **~72%** สำหรับสินค้าไทย
*   **Vector Search:** ค้นหาสินค้าที่คล้ายคลึงกันด้วย `pgvector` (Cosine Distance)
*   **Thai Text Processing:** รองรับการ Normalize ชื่อสินค้าไทย (ลบสระลอย, แปลงเลขไทย, ล้างคำขยะ)

---

## 📁 Project Structure (โครงสร้างโปรเจกต์)

*   `/taxonomy-app`: ระบบ Frontend (Next.js) และ Supabase Configurations
*   `/supabase`: Edge Functions และ Database Migrations
*   `api_server.py`: FastAPI สำหรับ Embedding Provider
*   `fresh_implementations.py`: Core Logic สำหรับ Thai Text Processing
*   `/docs`: เอกสารประกอบโปรเจกต์ฉบับเต็ม

---

## 🛠️ Dashboard & URLs

| Service | URL | Description |
|---------|-----|-------------|
| **Web UI** | `http://localhost:3000` | หน้าจอหลักสำหรับใช้งานระบบ |
| **AI Engine** | `http://localhost:8000` | REST API สำหรับ Embeddings |
| **Supabase Studio** | `http://localhost:54323` | จัดการฐานข้อมูลและ Edge Functions |

---

## 📚 Documentation (เอกสารเพิ่มเติม)

*   [📖 Architecture Detail](docs/development/architecture.md)
*   [🔌 API Reference](docs/api/api-reference.md)
*   [🚀 Quick Start Guide](docs/guides/quick-start.md)
*   [⚖️ Project Constitution (GEMINI.md)](GEMINI.md)

---

**พัฒนาโดย:** พยัคฆ์ (Gemini CLI Agent) | **สถานะ:** ใช้งานได้ (Stable)
