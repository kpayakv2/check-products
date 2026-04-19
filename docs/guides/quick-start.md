# 🚀 Quick Start Guide (Modern Stack)

ยินดีต้อนรับสู่คู่มือการเริ่มต้นใช้งานระบบ **Thai Product Taxonomy Manager & Similarity Checker** ฉบับปรับปรุงใหม่ที่รองรับสถาปัตยกรรม Next.js และ Supabase

---

## ✅ ข้อกำหนดเบื้องต้น (Prerequisites)

- **Node.js 18+** และ npm
- **Python 3.9+**
- **Docker Desktop** (สำหรับรัน Supabase Local)
- **Supabase CLI** (ติดตั้งผ่าน `npm install supabase --save-dev` หรือตาม [คู่มือ](https://supabase.com/docs/guides/cli))

---

## ⚙️ ขั้นตอนการติดตั้ง (Installation)

### 1. Clone Repository
```bash
git clone <repository-url>
cd check-products
```

### 2. ตั้งค่า AI Engine (FastAPI)
AI Engine ทำหน้าที่สร้าง Vector Embeddings สำหรับภาษาไทย
```bash
# สร้าง virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# ติดตั้ง dependencies
pip install -r requirements.txt

# รัน AI Engine
python api_server.py
# ระบบจะรันที่ http://localhost:8000
```

### 3. ตั้งค่า Database & Backend (Supabase)
```bash
# เริ่มต้น Supabase (ต้องเปิด Docker Desktop ก่อน)
supabase start

# รัน Migration (ถ้ามี)
supabase db reset
```

### 4. ตั้งค่า Frontend (Next.js)
```bash
cd taxonomy-app
npm install

# รัน Frontend
npm run dev
# ระบบจะรันที่ http://localhost:3000
```

---

## 💻 การใช้งานระบบ (Usage)

### 1. เข้าใช้งาน Web UI
เปิดเบราว์เซอร์ไปที่ [http://localhost:3000](http://localhost:3000) เพื่อเข้าสู่ Dashboard หลัก

### 2. นำเข้าข้อมูล (Import Wizard)
- ไปที่เมนู **"Import"**
- อัปโหลดไฟล์ CSV (รองรับชื่อไฟล์ภาษาไทย)
- ระบบจะส่งข้อมูลไปประมวลผลผ่าน AI Engine และแสดงข้อแนะนำหมวดหมู่แบบ Real-time

### 3. ตรวจสอบและยืนยัน (Review Process)
- ตรวจสอบความถูกต้องของหมวดหมู่ที่ AI แนะนำ (Hybrid Score)
- กด **"Approve"** เพื่อบันทึกลงฐานข้อมูลหลัก หรือแก้ไขหากไม่ถูกต้อง
- ระบบจะนำ Feedback ไปปรับปรุงความแม่นยำโดยอัตโนมัติ

---

## 🔌 API & Integration

### **AI Embedding API**
- `POST http://localhost:8000/api/embed`
- ใช้สำหรับสร้าง 384-dim vector จากข้อความภาษาไทย

### **Supabase Edge Functions**
- `category-suggestions` - แนะนำหมวดหมู่ด้วย Keyword
- `hybrid-classification-local` - แนะนำหมวดหมู่ด้วย Hybrid Algorithm (Keyword 60% + Embedding 40%)

---

## 🧪 การทดสอบ (Testing)

```bash
# ทดสอบ Python Logic
pytest

# ทดสอบ Frontend (Next.js)
cd taxonomy-app
npm test
```

---

## ⚠️ ข้อควรระวัง
- ตรวจสอบให้แน่ใจว่า **Docker Desktop** กำลังทำงานอยู่ก่อนรัน `supabase start`
- **FastAPI** ต้องทำงานอยู่ที่ port 8000 เพื่อให้ Edge Functions สามารถดึง Embeddings ได้
- ข้อมูลในตาราง `imports` บน Local Dev อาจต้องใช้ `fix_imports_rls.sql` หากพบปัญหา Permission

---

**พร้อมใช้งานแล้ว!** หากพบปัญหา สามารถศึกษาเพิ่มเติมได้ที่ [`GEMINI.md`](../../GEMINI.md) หรือสอบถามผ่านทีมพัฒนาครับ 🚀
