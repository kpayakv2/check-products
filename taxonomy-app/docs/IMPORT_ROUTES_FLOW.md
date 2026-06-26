# 🔄 Import System Routes & Flow (Revised May 2026)

เอกสารฉบับนี้อธิบายกระบวนการไหลของข้อมูล (Data Flow) ในระบบนำเข้าสินค้าผ่าน **Magic 5-Step Wizard** ล่าสุด

---

## 🏗️ Core Workflow: 5-Step Pipeline Architecture

ระบบนำเข้าสินค้าปรับเป็นสถาปัตยกรรมแบบ **Pipeline Batching** เพื่อให้ผู้ใช้ (Human-in-the-loop) เข้ามามีส่วนร่วมตัดสินใจในจุดสำคัญ

### **Step 1: Upload & Mapping (Frontend)**
1.  ผู้ใช้อัปโหลดไฟล์ CSV/XLSX
2.  ระบบรัน `sanitizeFileName()` และอ่านข้อมูล CSV
3.  จัดทำ Mapping คอลัมน์โดยอัตโนมัติ (เช่น ค้นหา `product_name`)

### **Step 2: Data Cleaning (Backend)**
1.  **Trigger:** ส่งข้อมูลไปทำความสะอาดที่ `POST /api/v1/clean` (FastAPI)
2.  **Processing:** 
    - ตัดคำสแปม (เช่น "พร้อมส่ง", "ลดราคา")
    - ลบ/สกัดหน่วยวัด
    - แปลงเลขไทยเป็นอารบิก (โดยใช้ `ThaiTextProcessor`)
3.  **UI Feedback:** แสดงตารางก่อน/หลัง ล้างข้อมูลให้ผู้ใช้รีวิว

### **Step 3: Deduplication Triage (Backend + UI)**
1.  **Trigger:** ส่งข้อมูลที่ล้างแล้วไปตรวจสอบความซ้ำซ้อนที่ `POST /api/v1/match/batch`
2.  **Processing:** ค้นหาสินค้าที่มีอยู่ใน Database อยู่แล้ว (ทั้งแบบ Exact Match และ Semantic Match ผ่าน pgvector)
3.  **UI Feedback:** แสดง Dashboard สรุปความเสี่ยง (ซ้ำแน่นอน, อาจจะซ้ำ, ไม่ซ้ำ) ผู้ใช้สามารถกด Action เพื่อข้ามการนำเข้าสินค้าที่ซ้ำได้

### **Step 4: AI Categorization Review (Backend + UI)**
1.  **Trigger:** ส่งข้อมูลที่ไม่ซ้ำไปให้ AI วิเคราะห์หาหมวดหมู่ผ่าน Edge Functions (`hybrid-classification-local`)
2.  **Processing:**
    - Keyword Matching (60% weight)
    - Vector Similarity (40% weight - 384 dim)
3.  **UI Feedback:** แสดง UI สำหรับ Review-by-Exception (ไฮไลท์สีตามค่า Confidence) ผู้ใช้ตรวจสอบเฉพาะรายการที่ AI ไม่มั่นใจ (สีเหลือง/แดง)

### **Step 5: Save & Complete (Database)**
1.  บันทึกข้อมูลสุดท้ายลงตาราง `products` และเชื่อมโยง `taxonomy_nodes`
2.  แสดง Confetti และสถิติสรุปการนำเข้า

---

## 🗄️ Database Tables Usage

| Table Name | Role in Import Flow |
| :--- | :--- |
| `products` | เก็บข้อมูลสินค้าที่นำเข้าใหม่ล่าสุด พร้อมค่า Vector |
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

**อัปเดตล่าสุด:** พฤษภาคม 2569 | **สถานะ:** เสถียรและใช้งานจริง (5-Step Pipeline)
