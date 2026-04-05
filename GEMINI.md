# GEMINI.md - Thai Product Taxonomy Manager & Similarity Checker
*(ระบบจัดการหมวดหมู่สินค้าไทยและตรวจสอบความคล้ายคลึง)*

## 📋 Project Overview (ภาพรวมโปรเจกต์)
This project is an AI-powered system designed to manage Thai product taxonomies and perform product similarity matching (deduplication). It consists of a Python-based AI backend and a Next.js frontend integrated with Supabase.

โปรเจกต์นี้เป็นระบบ AI สำหรับจัดการโครงสร้างหมวดหมู่สินค้าไทย (Taxonomy) และตรวจสอบการซ้ำซ้อนของสินค้า (Deduplication) โดยประกอบด้วย Backend ภาษา Python สำหรับงาน AI และ Frontend ด้วย Next.js ที่เชื่อมต่อกับ Supabase

### Key Technologies (เทคโนโลยีหลัก)
- **Backend (AI & API):** Python 3.8+, FastAPI, PyTorch, Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`), Pandas
- **Frontend:** Next.js (TypeScript), Tailwind CSS, Framer Motion, Lucide React
- **Database & Auth:** Supabase (PostgreSQL with `pgvector`), Supabase Edge Functions
- **Testing:** Pytest (Python), Jest & Playwright (Frontend)

---

## 🚀 Building and Running (การติดตั้งและรันระบบ)

### Prerequisites (สิ่งที่ต้องเตรียม)
- Python 3.8+
- Node.js & npm
- Docker (สำหรับ Supabase local development)

### 1. Backend (Python AI)
1.  **Setup Virtual Environment (สร้าง Environment):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```
2.  **Start API Server (เริ่มเซิร์ฟเวอร์):**
    ```bash
    python api_server.py
    ```
    - API Docs: `http://localhost:8000/docs`
    - Web Interface (legacy/debug): `http://localhost:8000/web`

### 2. Frontend (Taxonomy App)
1.  **Install Dependencies (ติดตั้ง Libs):**
    ```bash
    cd taxonomy-app
    npm install
    ```
2.  **Start Supabase (Local) (รันฐานข้อมูล):**
    ```bash
    npx supabase start
    ```
3.  **Run Development Server (เริ่ม Frontend):**
    ```bash
    npm run dev
    ```
    - URL: `http://localhost:3000`

---

## 🧪 Testing (การทดสอบ)

### Automated & Browser-in-the-loop Testing
- **Backend Tests**: `pytest`
- **Frontend Tests**: `npm test` (Jest), `npm run test:e2e` (Playwright)
- **Antigravity Verification**: เอเจนต์ต้องตรวจสอบ UI ผ่านเบราว์เซอร์เสมอตามกฎใน `.agents/rules/rules-antigravity.md` เพื่อเช็คความถูกต้องของภาษาไทยและ Layout

---

## 📂 Key Directory Structure (โครงสร้างโฟลเดอร์สำคัญ)
- `/` (Root): โค้ด AI หลัก, `api_server.py` และอัลกอริทึมการจับคู่สินค้า
- `/taxonomy-app`: แอปพลิเคชัน Next.js, การตั้งค่า Supabase และ UI Components
- `/docs`: เอกสารประกอบโปรเจกต์ทั้งภาษาไทยและอังกฤษ (Architecture, API, Guides)
- `/supabase`: Supabase Edge Functions, SQL Migrations และ Schema ฐานข้อมูล
- `/tests`: ชุดทดสอบภาษา Python
- `/model_cache`: ที่เก็บโมเดล AI (Sentence Transformer) แบบ Local

---

## 🛠️ Development Conventions (แนวทางการพัฒนา)
- **Thai Text Processing:** ต้องใช้ `TextPreprocessor` เสมอในการทำความสะอาดและจัดรูปแบบชื่อสินค้าไทย
- **Embeddings:** ใช้โมเดลมาตรฐาน `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions)
- **Classification:** แนะนำให้ใช้แนวทาง "Hybrid" (Keyword + Embedding) เพื่อความแม่นยำสูงสุด
- **Database:** ข้อมูลสินค้าและหมวดหมู่ทั้งหมดเก็บใน Supabase; ใช้ Supabase-js ในการเข้าถึงข้อมูลจาก Frontend
- **Styling:** เน้นใช้ Tailwind CSS ในส่วนของ `taxonomy-app`

## 🛠️ Installed Extensions & Tools (เครื่องมือที่ติดตั้งเพิ่มเติม)
- **Postgres (MCP):** ใช้สำหรับ Query และวิเคราะห์ข้อมูลใน Supabase โดยตรง
- **Domscribe (MCP):** ใช้สำหรับวิเคราะห์และแก้ไข UI ใน Next.js (Pixel-to-code)
- **Code Review (Extension):** ใช้สำหรับตรวจสอบคุณภาพโค้ด Python และ TypeScript
- **Skill Creator (Built-in):** ใช้สำหรับสร้าง Custom Skills เฉพาะทางของโปรเจกต์

---

## 📝 Important Notes (หมายเหตุสำคัญ)
- **Architectural Shift:** ส่วนการจัดหมวดหมู่ (Classification) ถูกย้ายจาก FastAPI ไปยัง **Supabase Edge Functions** (`supabase/functions/hybrid-classification-local/`) เพื่อให้การประมวลผลอยู่ใกล้กับข้อมูลมากขึ้น
- **Embedding Port:** Frontend จะเรียกใช้งาน Service สร้าง Vector ผ่าน `http://localhost:8000/api/embed`
- **Environment Variables:** ตรวจสอบไฟล์ `.env` ใน `taxonomy-app/` ให้เชื่อมต่อกับ Supabase ได้ถูกต้อง
