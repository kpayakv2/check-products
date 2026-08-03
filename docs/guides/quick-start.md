# 🚀 Quick Start Guide (Modern Hybrid Stack v3.0)

คู่มือการเริ่มต้นใช้งานระบบ **Thai Product Taxonomy Manager & Similarity Checker** ฉบับปรับปรุงล่าสุด

---

## ✅ ลำดับการรันระบบ (Correct Startup Sequence)

เพื่อให้ระบบทำงานได้สมบูรณ์ กรุณารันตามลำดับดังนี้:

### 1. เปิดระบบฐานข้อมูล (Supabase)
*   เปิดโปรแกรม **Docker Desktop**
*   ใช้คำสั่ง: `supabase start` ใน Terminal (รันที่โฟลเดอร์ที่มี config ของ supabase)

### 2. รัน AI Engine (FastAPI) - **หัวใจหลัก**
*   เปิด Terminal ใหม่ (PowerShell หรือ CMD)
*   เปิด Virtual Env: `.\.venv\Scripts\activate` (Windows)
*   รัน Server: `python -m src.api.api_server`
*   **ตรวจสอบ**: ต้องเห็นข้อความรันอยู่ที่ `http://127.0.0.1:8000`

### 3. รันหน้าเว็บจัดการข้อมูล (Next.js)
*   เปิด Terminal ใหม่
*   ไปที่โฟลเดอร์: `cd taxonomy-app`
*   รันเว็บ: `npm run dev`
*   **เข้าใช้งาน**: [http://127.0.0.1:3000](http://127.0.0.1:3000) (หรือเข้าจากอุปกรณ์อื่นในวง LAN ที่ `http://192.168.1.80:3000`)

---

## 💻 การใช้งานหลัก (Core Workflows)

### **A. การจัดหมวดหมู่สินค้าใหม่ (Classification)**
1.  เข้าหน้าเว็บพอร์ต 3000 ไปที่เมนู **Import** หรือ **Taxonomy**
2.  ระบบจะส่งชื่อสินค้าไปประมวลผลผ่าน Supabase Edge Function `hybrid-classification-local` ซึ่งจะติดต่อ FastAPI เพื่อแปลงข้อความเป็นเวกเตอร์ 384 มิติ
3.  ระบบประมวลผลจับคู่คะแนนไฮบริด (Keyword 60% + Embedding 40%) และให้ผลลัพธ์คำแนะนำหมวดหมู่
4.  ตรวจสอบผลลัพธ์บน UI ได้ทันที

### **B. การตรวจซ้ำสินค้า (Deduplication)**
ตรวจหาสินค้าที่ซ้ำกันเชิงเวกเตอร์ (Cosine Similarity) ในคลังสินค้าและประวัติการตรวจสอบ:
1. เข้าใช้งานหน้าจอ **Data Quality Center** และเลือกแท็บ **Deduplication**
2. กดปุ่มสแกนสินค้าซ้ำ ระบบจะประมวลผลคู่ที่คาดว่าซ้ำโดยเปรียบเทียบเชิงเวกเตอร์ (pgvector `vector(384)`) ควบคู่กับ ML model (RandomForestClassifier) ใน Stage-2 Inference เพื่อคัดแยกคู่ที่มั่นใจว่าเป็นคนละชิ้นออก
3. สามารถคลิกยืนยันการยุบรวมสินค้า (Merge หรือ Ignore) ซึ่งผลการรีวิวจะบันทึกลงฐานข้อมูล Supabase และใช้สำหรับ Retrain ML model เพื่อเรียนรู้ป้อนกลับอย่างต่อเนื่อง

---

## 🔌 การเชื่อมต่อ (Integration Details)
*   **API Local (FastAPI)**: `http://127.0.0.1:8000`
*   **Supabase API Local**: `http://127.0.0.1:54331` (ผ่าน Edge Functions)
*   **Interactive API Docs**: `http://127.0.0.1:8000/docs` (Swagger UI)

---

## 💡 ทางเลือกที่รวดเร็ว (Quick Startup Script)
คุณสามารถใช้งานสคริปต์ **`START_PHAYAK.bat`** ในโฟลเดอร์หลักของโปรเจกต์ ซึ่งจะรันคำสั่งเปิด Firewall, รัน Backend, รัน Frontend และเปิดบราวเซอร์ให้อัตโนมัติในคลิกเดียว

---

**📅 Last Updated**: 28 มิถุนายน 2569 (Verified for Local Dev Stack)
