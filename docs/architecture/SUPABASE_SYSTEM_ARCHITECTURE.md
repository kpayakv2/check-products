# 🏗️ Supabase System Architecture (Verified & Modern)

**Thai Product Taxonomy Manager - สถาปัตยกรรมระบบล่าสุด**

---

## 📊 1. Database Layer (PostgreSQL + pgvector)

ระบบใช้ Supabase PostgreSQL เป็นหัวใจหลักในการจัดเก็บข้อมูลและการประมวลผลความคล้ายคลึง

### **โครงสร้างตารางหลัก (Core Tables):**
- `taxonomy_nodes`: โครงสร้างหมวดหมู่สินค้า (Hierarchy) พร้อมเก็บ Embeddings ประจำหมวดหมู่
- `products`: ข้อมูลสินค้าหลัก พร้อมเก็บ **384-dim Vector Embeddings**
- `imports`: ประวัติและสถานะการนำเข้าข้อมูล (สนับสนุนชื่อไฟล์ภาษาไทย)
- `keyword_rules`: กฎสำหรับจำแนกหมวดหมู่ด้วยคำสำคัญ (Keyword Matching)
- `product_category_suggestions`: ผลลัพธ์การแนะนำหมวดหมู่จาก AI (Hybrid Score)

---

## 🚀 2. Business Logic Layer (Edge Functions)

ตามกฎ **"Edge Function First"** ตรรกะการประมวลผลทั้งหมดต้องอยู่ที่นี่ เพื่อความง่ายในการขยายระบบและรักษาความเป็น Data-Centric

### **ฟังก์ชันสำคัญที่ใช้งานอยู่:**
1.  **`hybrid-classification-local`**: 
    - **Logic:** Keyword (60%) + Embedding (40%)
    - **Workflow:** เรียก AI Engine → เปรียบเทียบกับ Taxonomy → คืนค่า Top Suggestion
2.  **`generate-embeddings-local`**: 
    - ประสานงานกับ FastAPI เพื่อเปลี่ยนข้อความเป็น Vector
3.  **`hybrid-search`**: 
    - ค้นหาสินค้าโดยใช้ทั้งข้อความปกติและ Vector Similarity ร่วมกัน
4.  **`product-deduplication`**: 
    - ตรวจหาสินค้าที่ซ้ำกันก่อนบันทึกลงฐานข้อมูล

---

## 🧠 3. AI Engine Layer (FastAPI)

FastAPI ทำหน้าที่เป็น **Worker Service** สำหรับงานที่ใช้ทรัพยากรเครื่องสูง (Heavy-lifting)

### **หน้าที่หลัก:**
- สร้าง **Vector Embeddings (384 Dimensions)** โดยใช้โมเดล `paraphrase-multilingual-MiniLM-L12-v2`
- ทำงานบน Local Environment (ไม่ต้องเสียค่า API ภายนอก)
- **ห้ามเก็บ Business Logic:** Logic การตัดสินใจต้องส่งกลับมาให้ Edge Functions เป็นผู้จัดการ

---

## 🔄 4. Data Flow (ลำดับการไหลของข้อมูล)

### **กระบวนการนำเข้าสินค้า (Import Flow):**
1.  **Frontend:** Upload CSV → Supabase Storage
2.  **Next.js API:** อ่านข้อมูลจาก CSV และส่งต่อให้ระบบประมวลผล
3.  **Edge Function (`hybrid-classification-local`):**
    a. ส่งชื่อสินค้าไปให้ **FastAPI** เพื่อรับ Vector
    b. ดึง **Keyword Rules** จาก DB มา Match
    c. ดึง **Taxonomy Embeddings** จาก DB มาเปรียบเทียบ (pgvector `<=>`)
    d. คำนวณ Hybrid Score
4.  **DB:** บันทึกผลลัพธ์ลงตาราง `products` และ `product_category_suggestions`
5.  **Frontend:** แสดงผลให้ผู้ใช้ตรวจสอบ (Review UI)

---

## 📡 5. การตั้งค่า Environment

เพื่อให้ Edge Functions เชื่อมต่อกับ FastAPI ได้ ต้องตั้งค่า Secret ใน Supabase:

```bash
# ในเครื่อง Local Dev (ใช้ host.docker.internal เพื่อให้ Container คุยกับ Host ได้)
supabase secrets set FASTAPI_URL=http://host.docker.internal:8000
```

---

## 🎯 สรุปจุดเด่น
- **Free & Fast:** ใช้ Local Embedding Model ไม่เสียค่าใช้จ่ายรายครั้ง และทำงานรวดเร็ว
- **Single Source of Truth:** Logic การจัดหมวดหมู่รวมอยู่ที่ Edge Functions ที่เดียว
- **Scalable:** สามารถเพิ่ม Worker (FastAPI) ได้อิสระเมื่อข้อมูลมีปริมาณมาก
- **Precise:** ผสมผสานจุดแข็งของทั้ง Keyword และ Semantic Search

---

**อัปเดตสถานะล่าสุด (เม.ย. 2569):** ระบบทำงานเสถียรบน Local Dev และรองรับการนำเข้าไฟล์ชื่อไทยสมบูรณ์แบบ
