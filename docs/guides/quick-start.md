# 🚀 Quick Start Guide (Modern Hybrid Stack v3.0)

คู่มือการเริ่มต้นใช้งานระบบ **Thai Product Taxonomy Manager** ฉบับปรับปรุงล่าสุด

---

## ✅ ลำดับการรันระบบ (Correct Startup Sequence)

เพื่อให้ระบบทำงานได้สมบูรณ์ กรุณารันตามลำดับดังนี้:

### 1. เปิดระบบฐานข้อมูล (Supabase)
*   เปิดโปรแกรม **Docker Desktop**
*   ใช้คำสั่ง: `supabase start` ใน Terminal

### 2. รัน AI Engine (FastAPI) - **หัวใจหลัก**
*   เปิด Terminal ใหม่
*   เปิด Virtual Env: `venv\Scripts\activate` (Windows)
*   รัน Server: `python api_server.py`
*   **ตรวจสอบ**: ต้องเห็นข้อความรันอยู่ที่ `http://127.0.0.1:8000`

### 3. รันหน้าเว็บจัดการข้อมูล (Next.js)
*   เปิด Terminal ใหม่
*   ไปที่โฟลเดอร์: `cd taxonomy-app`
*   รันเว็บ: `npm run dev`
*   **เข้าใช้งาน**: [http://localhost:3000](http://localhost:3000)

---

## 💻 การใช้งานหลัก (Core Workflows)

### **A. การจัดหมวดหมู่สินค้าใหม่**
1.  เข้าหน้าเว็บ Port 3000
2.  ไปที่เมนู **Import** หรือ **Classification**
3.  ระบบจะส่งชื่อสินค้าไปที่ API Port 8000 เพื่อคำนวณ Hybrid Score (60/40)
4.  ตรวจสอบผลลัพธ์: 🟢 Auto-Approve (>0.9) หรือ 🟡 Needs Review (0.7-0.9)

### **B. การตรวจซ้ำสินค้า (Deduplication)**
ใช้สคริปต์ Pipeline เพื่อประสิทธิภาพสูงสุด:
```bash
# วิเคราะห์ไฟล์สินค้า
python complete_deduplication_pipeline.py --input data.csv --mode analyze
```

---

## 🔌 การเชื่อมต่อ (Integration Details)
*   **API Local**: `http://127.0.0.1:8000/api/classify/category`
*   **Vector Provider**: `http://127.0.0.1:8000/api/embed`
*   **Interactive API Docs**: `http://127.0.0.1:8000/docs` (Swagger UI)

---

**📅 Last Updated**: 16 เมษายน 2569 (Verified for FastAPI)
