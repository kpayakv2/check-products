# GEMINI.md - Thai Product Taxonomy Manager & Similarity Checker
*(ระบบจัดการหมวดหมู่สินค้าไทยและตรวจสอบความคล้ายคลึง)*

## 📋 Project Overview (ภาพรวมโปรเจกต์)
This project is an AI-powered system designed to manage Thai product taxonomies and perform product similarity matching (deduplication). It consists of a Python-based AI backend and a Next.js frontend integrated with Supabase.

---

## 🏛️ Project Constitution (รัฐธรรมนูญของโปรเจกต์)
*กฎเหล็กที่เอเจนต์ทุกตัวต้องปฏิบัติตามอย่างเคร่งครัด*

### 1. 🛠️ Tech Stack Mandates
- **Thai Text Processing:** ต้องใช้คลาส `ThaiTextProcessor` (จาก `fresh_implementations.py`) ทุกครั้งที่มีการประมวลผลหรือทำความสะอาดชื่อสินค้าภาษาไทย เพื่อรักษาความแม่นยำและมาตรฐานเดียวกัน
- **Edge Function First:** Logic สำหรับการ Classification หรือการประมวลผล AI ใหม่ๆ **ต้องถูกเขียนใน Supabase Edge Functions เท่านั้น** (เว้นแต่จะเป็นงาน Heavy-lifting ที่ต้องการ GPU) เพื่อรักษาแนวทาง Data-Centric Architecture
- **Vector Standard:** ต้องใช้ `pgvector` ใน PostgreSQL และกำหนดขนาด Embedding เป็น **384 dimensions** (โมเดล `paraphrase-multilingual-MiniLM-L12-v2`) เท่านั้น
- **Styling:** ห้ามใช้ Tailwind CSS นอกโฟลเดอร์ `taxonomy-app/` โดยเด็ดขาด

### 2. 🇹🇭 Thai Language & UI Rules
- **UI Encoding:** ต้องรองรับการแสดงผลภาษาไทยที่ถูกต้อง (UTF-8) และจัดการสระลอย/สระจมให้สมบูรณ์ในทุกหน้าจอ
- **Responsiveness:** หน้าจอ Dashboard และการจัดการ Taxonomy ต้องรองรับ Mobile (Responsive) และตัวหนังสือภาษาไทยต้องไม่อ่านยากหรือตัดบรรทัดเพี้ยน
- **Normalization:** ชื่อสินค้าก่อนเข้าสู่กระบวนการ AI ต้องผ่านการ Normalize (เลขไทยเป็นอารบิก, ลบสระลอย, ล้างคำขยะ) ตามมาตรฐานใน `docs/development/text-preprocessing.md`

### 3. 🧪 Validation & Finality
- **No Manual Fixes:** ห้ามแก้ไขโค้ดหรือฐานข้อมูลโดยไม่มีการรัน Test (Pytest/Jest) เพื่อยืนยันผล
- **Benchmark Driven:** การแก้ไข Algorithm การจับคู่ (Similarity) ใดๆ ต้องรักษาค่าความแม่นยำ (F1-score/Accuracy) ให้ไม่ต่ำกว่า 72% (ค่ามาตรฐานปัจจุบัน)

### 4. 🛠️ MCP & Tooling Mandates (ยุทธศาสตร์การใช้เครื่องมือ)
- **Database First (Postgres MCP):** ต้องใช้ `mcp_postgres_query` เป็นทางเลือกแรกในการตรวจสอบข้อมูล, แก้ไข Schema หรือวิเคราะห์ Data ใน Supabase เสมอ เพื่อความรวดเร็วและประหยัด Context
- **Visual Integrity (Domscribe & Puppeteer):** ต้องใช้ MCP เหล่านี้ในการตรวจสอบ UI เสมอ โดยเฉพาะการเช็ค Layout ภาษาไทยในหน้าจอต่างๆ ตามกฎ Antigravity
- **Strategic Delegation:** งานที่เกี่ยวข้องกับการแก้ไขไฟล์จำนวนมาก (>3 ไฟล์) หรือการรันกระบวนการที่ยาวนาน ต้องส่งต่อให้ Sub-agents (`generalist` หรือ `codebase_investigator`) เพื่อรักษาความกระชับของ Main Session Context
- **Context Management:** ต้องอัปเดตไฟล์ `CURRENT_STATUS.md` ทุกครั้งหลังจบ Task สำคัญ เพื่อรักษา "ความจำระยะสั้น" ของโปรเจกต์ให้แม่นยำที่สุด

### 5. 🥋 Specialized Skills (สารบัญทักษะเฉพาะทาง)
*เรียกใช้งานผ่าน `activate_skill(name)` เมื่อเข้าสู่ Workflow ที่เกี่ยวข้อง*

- **`skill-thai-taxonomy-expert`**: ใช้เมื่อต้องการออกแบบลำดับชั้นหมวดหมู่, แก้ไข `taxonomy_nodes` หรือวางแผนโครงสร้างข้อมูลสินค้าไทย
- **`skill-data-cleaner`**: ใช้เมื่อต้องทำความสะอาดข้อมูลชื่อสินค้า (Normalization), จัดการหน่วยวัด (kg/g/ml) หรือลบข้อความโปรโมชั่นก่อนเข้า AI
- **`skill-vector-optimizer`**: ใช้เมื่อต้องการวิเคราะห์ประสิทธิภาพ `pgvector`, ปรับจูน Indexing หรือตรวจสอบสินค้าที่มีค่า Similarity ต่ำเพื่อหาจุดอ่อนของโมเดล

---

## 🏗️ Verified System Architecture (Verified Oct 2025)

### 1. 🧩 Component Roles
- **FastAPI (`api_server.py`):** Local Embedding Provider (384-dim) via `/api/embed`
- **Supabase Edge Functions:** Orchestrator (e.g., `hybrid-classification-local`)
- **PostgreSQL:** Heavy Logic (RPC Functions like `hybrid_category_classification`)

### 2. ⚖️ Hybrid Classification Logic
- **Weights:** Keyword 60% + Embedding 40%
- **Keyword Source:** `keyword_rules`, `taxonomy_nodes.keywords`, และ `name_match`
- **Embedding:** Cosine Distance (`<=>`) กับ `taxonomy_nodes.embedding`

---

## 📂 Key Directory Structure (โครงสร้างโฟลเดอร์สำคัญ)
- `/` (Root): โค้ด AI หลัก, `api_server.py`
- `/taxonomy-app`: Next.js App, Supabase Integration, UI Components
- `/docs`: เอกสารประกอบโปรเจกต์ (Architecture, API, Guides)
- `/supabase`: Edge Functions, Migrations, Schema
- `/tests`: ชุดทดสอบ Python (Unit/Integration)

---

