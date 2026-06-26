# Product Similarity Checker - Documentation

ยินดีต้อนรับสู่ศูนย์รวมเอกสารของโปรเจกต์ **Thai Product Taxonomy Manager & Similarity Checker** ระบบจำแนกหมวดหมู่สินค้าไทยด้วย AI

## 📚 โครงสร้างเอกสาร

### 🚀 [User Guides](guides/)
- **[Quick Start Guide](guides/quick-start.md)** - เริ่มต้นติดตั้งและใช้งานระบบ Modern Stack (Next.js + Supabase)
- **[Human Feedback System](guides/human-feedback.md)** - คู่มือการใช้งานระบบ Review และการให้ Feedback กับ AI

### 🔧 [API Documentation](api/)
- **[API Reference](api/api-reference.md)** - รายละเอียด API Endpoints ทั้ง FastAPI และ Edge Functions
- **[API Architecture](../API_ARCHITECTURE.md)** - สถาปัตยกรรมการเชื่อมต่อระหว่าง Service ต่างๆ

### 👩‍💻 [Development](development/)
- **[Architecture Overview](development/architecture.md)** - รายละเอียดโครงสร้างโมดูลและ Data Pipeline
- **[Text Preprocessing](development/text-preprocessing.md)** - ขั้นตอนการทำความสะอาดชื่อสินค้าภาษาไทย
- **[Contributing Guide](development/contributing.md)** - แนวทางการพัฒนาและส่งต่องาน

### 📊 [Reports & Analysis](reports/)
- **[Test Results Summary](reports/test-results-summary.md)** - สรุปผลการทดสอบล่าสุด
- **[Sentence Transformer Analysis](reports/sentence-transformer-analysis.md)** - วิเคราะห์ประสิทธิภาพของโมเดล ML (384-dim)

---

## 🎯 **การนำทางด่วน (Quick Navigation)**

### **สำหรับผู้ใช้งานทั่วไป:**
1. เริ่มต้นที่ [`README.md`](../README.md) เพื่อทำความเข้าใจภาพรวม
2. ทำตามขั้นตอนใน [`Quick Start Guide`](guides/quick-start.md)
3. เข้าใช้งานระบบผ่านหน้าจอ [http://localhost:3000](http://localhost:3000)

### **สำหรับนักพัฒนา:**
1. ศึกษา [`Architecture Overview`](development/architecture.md)
2. ตรวจสอบกฎเหล็กใน [`GEMINI.md`](../GEMINI.md)
3. ศึกษาสถาปัตยกรรม Supabase ใน [`SUPABASE_SYSTEM_ARCHITECTURE.md`](../SUPABASE_SYSTEM_ARCHITECTURE.md)

---

## 🔄 **Document Status**

### **✅ ปรับปรุงล่าสุด (เมษายน 2569):**
- 🎯 เอกสารทุกฉบับถูกปรับปรุงให้รองรับสถาปัตยกรรม **Next.js + Supabase**
- 🧹 นำเนื้อหาที่ล้าสมัย (Port 5000, 8000/web) ออกจากเอกสารหลัก
- 📋 อ้างอิงความแม่นยำมาตรฐานที่ **72%** ตามการทดสอบจริง

---

**🏗️ เอกสารนี้ได้รับการดูแลโดย พยัคฆ์ (Gemini CLI Agent)**
