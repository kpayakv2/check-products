# Product Requirements Document (PRD) — Thai Product Taxonomy Manager & Similarity Checker
*(เอกสารความต้องการระบบ — ระบบจัดการหมวดหมู่สินค้าไทยและตรวจสอบความคล้ายคลึง)*

**ชื่อโปรเจกต์:** Thai Product Taxonomy Manager & Similarity Checker (ระบบจัดการหมวดหมู่สินค้าไทยและตรวจสอบความคล้ายคลึง)  
**สถานะ:** ฉบับปรับปรุงสำหรับ Repository (Production Local Dev)  
**วันที่ปรับปรุงล่าสุด:** 28 มิถุนายน 2026

---

## 1. ภาพรวมโครงการ (Project Overview)
วัตถุประสงค์เพื่อสร้างระบบจัดการโครงสร้างหมวดหมู่สินค้าไทย (Thai Product Taxonomy) และการจับคู่ความคล้ายคลึงของสินค้า (Deduplication / Similarity Matching) เพื่อตรวจหาความซ้ำซ้อนในคลังข้อมูลเดิม (Internal Catalog Audit) รองรับการประมวลผลขนาดใหญ่สูงสุด 10,000 SKU
* **ความแม่นยำเป้าหมาย (Accuracy / F1-score):** ไม่ต่ำกว่า 72% บนชุดทดสอบ Benchmark
* **สถาปัตยกรรมการประมวลผล:** ไฮบริด (Hybrid Algorithm) แบ่งเป็น Keyword Match 60% และ Embedding Semantic Match 40%
* **รูปแบบการติดตั้ง:** รันเป็นระบบ Local Production บน Windows (Win32) รองรับการเข้าถึงผ่าน LAN ในออฟฟิศ (IP Server: `192.168.1.80`)

---

## 2. ขอบเขตการทำงาน (Scope & Use Cases)
1. **ระบบจัดการโครงสร้างหมวดหมู่ (Taxonomy Management):**
   * ผู้ใช้สามารถเพิ่ม ลบ หรือแก้ไขโครงสร้างหมวดหมู่สินค้า (Taxonomy Tree) ผ่าน Next.js UI
   * กำหนดและจัดการคำสำคัญ (Keywords) และกฎสำหรับจับคู่คำสำคัญ (Keyword Rules) ในแต่ละหมวดหมู่
2. **ระบบวิเคราะห์แนะนำหมวดหมู่สินค้าแบบไฮบริด (Hybrid Classification):**
   * วิเคราะห์หาหมวดหมู่ที่เหมาะสมที่สุดให้กับชื่อสินค้าไทยผ่าน Supabase Edge Function (`hybrid-classification-local`)
   * ผสมคะแนนจากการจับคู่คำสำคัญ (Keyword Rules) 60% และการเปรียบเทียบความคล้ายเชิงความหมายด้วย Vector Embeddings 40%
3. **ระบบตรวจสอบและยุบรวมสินค้าซ้ำ (Product Deduplication & Catalog Audit):**
   * ตรวจสอบสินค้าที่ชื่อคล้ายคลึงกันในระบบเพื่อระบุคู่สินค้าที่ซ้ำซ้อน (Deduplication) โดยใช้ค่า Cosine Distance และแสดงผลบน Audit UI
   * แปลงชื่อสินค้าภาษาไทยเป็น Vector ขนาด 384 มิติ ด้วยโมเดล `paraphrase-multilingual-MiniLM-L12-v2` แล้วเปรียบเทียบในฐานข้อมูลด้วยดัชนี HNSW (pgvector)
4. **ลูปการเรียนรู้ป้อนกลับจากมนุษย์ (ML Feedback Loop System):**
   * เมื่อผู้ใช้กดยืนยันหรือแก้ไขการแนะนำหมวดหมู่ผ่านหน้า UI ระบบจะบันทึกผลการตัดสินใจลงในฐานข้อมูล Supabase (`similarity_matches` / `feedback` table)
   * ระบบนำข้อมูล Feedback ดังกล่าวไป Retrain โมเดล RandomForestClassifier (15 features) ผ่าน FastAPI Backend เพื่อใช้ปรับความเชื่อมั่น (Confidence) ใน Stage-2 Inference ของระบบตรวจจับสินค้าซ้ำ

---

## 3. สถาปัตยกรรมระบบ (System Architecture)
ระบบใช้สถาปัตยกรรมที่รวมศูนย์ตรรกะไว้บนฐานข้อมูลเป็นหลัก (Supabase-First Architecture) โดยรันผ่านบริการโลคอลบน Windows (Win32):

```
[ Next.js Frontend UI (Port 3000) ]
        │ (เรียกใช้ Supabase JS Client / Edge Functions)
        ▼
[ Supabase Local (Port 54331) ] ── (pgvector/HNSW Index ใน PostgreSQL)
        │
        ▼ (ร้องขอผ่าน Http Client / host.docker.internal)
[ FastAPI AI Engine (Port 8000) ] (รันโลคอล แปลงข้อจำเป็น Vector 384-dim)
```

* **Frontend UI (Next.js):** 
  * รันที่โฟลเดอร์ `/taxonomy-app` (Port 3000)
  * ใช้ relative URL สำหรับ Supabase connection เสมอ เพื่อไม่ให้ติดปัญหา CORS เมื่อเข้าใช้งานผ่าน LAN
* **Primary Backend & Orchestrator (Supabase Local):**
  * รันผ่าน Docker Compose ของ Supabase Local (Port 54331)
  * จัดการสิทธิ์การเข้าถึงข้อมูลผ่าน Row Level Security (RLS) ทั้งหมด 18 policies
  * รัน Edge Function `hybrid-classification-local` สำหรับรันโค้ดไฮบริดและประสานงานระหว่าง FastAPI และ PostgreSQL
* **AI Worker (FastAPI Python Engine):**
  * รันที่โฟลเดอร์ `/src/api` (Port 8000)
  * ทำหน้าที่ทำความสะอาดชื่อสินค้าไทย (Text Preprocessing) และคำนวณ Vector Embedding ขนาด 384 มิติ เท่านั้น (ห้ามบรรจุ Business Decision Logic หรือประเมินสิทธิ์ผู้ใช้ที่นี่)

---

## 4. โครงสร้างข้อมูลที่สำคัญ (Core Database Schema)
ระบบใช้ Supabase PostgreSQL เป็น Single Source of Truth เพียงแหล่งเดียว (ไม่มีการใช้ SQLite/human_feedback.db):

* **ตาราง `taxonomy_nodes` (โครงสร้างหมวดหมู่):**
  * `id`: SERIAL / UUID (PK)
  * `name_th`: VARCHAR (ชื่อหมวดหมู่ภาษาไทย)
  * `keywords`: VARCHAR[] (คำสำคัญประจำหมวดหมู่)
  * `embedding`: VECTOR(384) (ค่าเวกเตอร์ของหมวดหมู่)
* **ตาราง `products` (ข้อมูลสินค้า):**
  * `id`: UUID (PK)
  * `name`: VARCHAR (ชื่อสินค้าภาษาไทย)
  * `category_id`: UUID (FK ไปยัง taxonomy_nodes)
  * `is_duplicate`: BOOLEAN (ระบุว่าตัวนี้ซ้ำกับตัวอื่นหรือไม่)
  * `duplicate_of_id`: UUID (FK ชี้ไปยังสินค้าหลักที่เป็นตัวหลักในกรณีซ้ำ)
  * `embedding`: VECTOR(384) (ค่าเวกเตอร์ของสินค้า)
* **ตาราง `keyword_rules` (กฎการจับคู่หมวดหมู่):**
  * `id`: SERIAL (PK)
  * `pattern`: VARCHAR (คำสำคัญ/Pattern ที่ค้นหา)
  * `category_id`: UUID (FK ไปยังหมวดหมู่ที่ต้องการจับคู่)

---

## 5. ความต้องการที่ไม่ใช่ฟังก์ชัน (Non-Functional Requirements - NFR)
* **NFR1 (Windows Local Network Support):**
  * ห้ามระบุ `localhost` ในโค้ดหรือการเชื่อมต่อเครือข่าย ให้ใช้ IP `127.0.0.1` เสมอ เพื่อลดปัญหาระบบ LAN/Socket บน Windows (Win32)
  * หน้าจอ Next.js ต้องเชื่อมโยงแบบ Relative Path (ไม่มีการระบุโดเมนตรงๆ) เพื่อเปิดให้ผู้ใช้ในวง LAN เข้าถึงผ่าน `http://192.168.1.80:3000` ได้ทันที
* **NFR2 (Resource Management):**
  * จำกัดการประมวลผลและการใช้ RAM โลคอลให้เหมาะสม (RAM ของ database docker container ไม่เกิน 1GB และ FastAPI Python Engine ไม่เกิน 512MB)
* **NFR3 (Clean PC & Persistence):**
  * ห้ามเก็บไฟล์อัปโหลดหรือข้อมูลฐานข้อมูลไว้ในระดับ Container ชั่วคราว (ต้อง Map Docker volume ไปเก็บในโฮสต์ภายนอกเพื่อให้ข้อมูลไม่สูญหาย)
* **NFR4 (Smart Testing Enforcement):**
  * โค้ดทั้งหมดต้องผ่านการทดสอบตาม Smart Testing Matrix ทั้ง 5 ขั้นตอน (Sequential Thinking, Postgres checks, Puppeteer visual verification, Log verification, Memory leak check) ก่อน commit เสมอ

---

## 6. เครื่องมือพัฒนาและซอฟต์แวร์ (Tech Stack & Development Tools)
* **Frontend:** Next.js (React), Supabase JS Client, CSS Vanilla/Tailwind (Tailwind จำกัดเฉพาะใน `/taxonomy-app` เท่านั้น)
* **Backend Engine:** Python 3.10+ (FastAPI, Sentence-Transformers, Pytest)
* **Database:** Supabase Local Docker CLI (PostgreSQL 17 + pgvector extension)
* **AI Model:** `paraphrase-multilingual-MiniLM-L12-v2` (384 Dimensions)
* **Testing:** Pytest (Backend API), Jest/Playwright (Frontend/UI testing)
