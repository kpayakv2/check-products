# 🔄 Import System Routes & Flow (Revised April 2026)

เอกสารฉบับนี้อธิบายกระบวนการไหลของข้อมูล (Data Flow) ในระบบนำเข้าสินค้าเวอร์ชันล่าสุด

---

## 🏗️ Core Workflow: The Streaming Architecture

ระบบนำเข้าสินค้าใช้สถาปัตยกรรมแบบ **Streaming** เพื่อรองรับข้อมูลจำนวนมากและแสดงความคืบหน้าแก่ผู้ใช้แบบ Real-time

### **Step 1: Upload & Record Registration**
1.  **Frontend (Wizard):** ผู้ใช้อัปโหลดไฟล์
2.  **Sanitization:** ระบบรัน `sanitizeFileName()` เพื่อแปลงชื่อไฟล์ไทยเป็น ASCII (เช่น `สินค้า.csv` → `sinka.csv`) เพื่อเลี่ยงข้อจำกัดของ Supabase Storage
3.  **Registration:** บันทึกข้อมูลไฟล์ลงตาราง `imports` (เก็บชื่อไทยต้นฉบับไว้) และอัปโหลดไฟล์ขึ้น Storage

### **Step 2: Server-Side Stream (Next.js API)**
1.  **Trigger:** Wizard เรียก `POST /api/import/process` พร้อม `importId` และ `filePath`
2.  **Streaming:** API รันเป็น ReadableStream เพื่อส่งผลลัพธ์แบบ Chunked กลับไปที่ UI
3.  **Row Processing:**
    - ดึงไฟล์จาก Storage มาอ่านทีละบรรทัด (CSV Parsing)
    - **AI Pipeline (Sequential):**
        - เรียก `generate-embeddings-local` (Edge Function) → ได้ Vector
        - เรียก `hybrid-classification-local` (Edge Function) → ได้หมวดหมู่แนะนำ
    - **Database Save:** บันทึกลงตาราง `products` ทันที พร้อมสถานะ `pending_review_category`

### **Step 3: UI Feedback Loop**
- Wizard รับข้อมูลจาก Stream และอัปเดต Progress Bar ทีละรายการ
- แสดงชื่อสินค้าล่าสุดที่ประมวลผลเสร็จ เพื่อให้ผู้ใช้ทราบสถานะ

---

## 🎯 API Endpoints & Actions

### **1. Processing Engine**
- `POST /api/import/process`
    - **Input:** `importId`, `filePath`, `columnMapping`
    - **Output:** SSE Stream (`type: progress`, `type: completed`, `type: error`)

### **2. Approval & Review**
- `GET /api/import/pending`
    - ดึงรายการสินค้าที่รอการตรวจสอบ
- `POST /api/import/pending`
    - **Action: `approve`** → ย้ายสถานะเป็น `approved`
    - **Action: `reject`** → ย้ายสถานะเป็น `rejected`

---

## 🗄️ Database Tables Usage

| Table Name | Role in Import Flow |
| :--- | :--- |
| `imports` | เก็บประวัติการนำเข้า, ชื่อไฟล์จริง (ไทย), และ Storage Path |
| `products` | เก็บข้อมูลสินค้าที่นำเข้า (ทั้งรอตรวจและอนุมัติแล้ว) พร้อมค่า Vector |
| `product_category_suggestions` | เก็บคำแนะนำจาก AI พร้อมเหตุผล (Explanation) และค่า Confidence |
| `taxonomy_nodes` | แหล่งข้อมูลหมวดหมู่ (Hierarchy) สำหรับการ Match |
| `keyword_rules` | กฎคำสำคัญ (60% weight) ที่ใช้ตัดสินหมวดหมู่ |

---

## 🤖 AI Algorithm (Hybrid Strategy)

ระบบตัดสินใจเลือกหมวดหมู่ผ่านกระบวนการ **Hybrid Scoring**:
1.  **Keyword Matching (60%):** ค้นหาคำสำคัญในชื่อสินค้าเทียบกับ Rules ใน DB
2.  **Vector Similarity (40%):** เปรียบเทียบระยะห่างของ Vector สินค้ากับ Vector ของหมวดหมู่ (pgvector)
3.  **Result:** รวมคะแนนและแนะนำหมวดหมู่ที่ได้คะแนนสูงสุด

---

**อัปเดตล่าสุด:** 9 เมษายน 2569 | **สถานะ:** เสถียรและใช้งานจริง
