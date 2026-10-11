# 🎯 **API Capabilities - Hybrid AI System (v3.0)**

## 📊 **ภาพรวมระบบวิเคราะห์**

**Endpoint หลัก**: `/api/classify/category` & `/api/embed`  
**จุดประสงค์**: จัดหมวดหมู่สินค้าและตรวจซ้ำ (Deduplication) ด้วยความแม่นยำสูง  
**เวอร์ชัน**: v3.0 - Hybrid Architecture  
**เทคโนโลยีหลัก**: FastAPI + Supabase + pgvector (384-dim)

---

## 🔧 **คุณสมบัติเด่น (Core Features)**

### **1. 🤖 Hybrid Classification Algorithm**
- **Weights**: **Keyword (60%)** + **Embedding (40%)**
- **Keyword Source**: กฎจาก `keyword_rules` และ `taxonomy_nodes.keywords`
- **Embedding Source**: คำนวณผ่านโมเดล `paraphrase-multilingual-MiniLM-L12-v2` (Local Provider)
- **Accuracy**: เป้าหมายความแม่นยำไม่ต่ำกว่า **72%** ตามมาตรฐานโปรเจกต์

### **2. 🧠 Human-in-the-Loop & Machine Learning**
- ระบบรองรับการเรียนรู้จาก Feedback ของผู้ใช้ผ่านสคริปต์ `complete_deduplication_pipeline.py`
- เก็บข้อมูลการตรวจสอบลงใน `product_deduplication.db` (SQLite) เพื่อใช้สอนโมเดลให้ฉลาดขึ้นตามสไตล์การตรวจของคุณกาน

### **3. 🇹🇭 Thai Language Optimization**
- ใช้ `ThaiTextProcessor` ในการลบสระลอย, ล้างคำขยะ, และจัดการหน่วยวัด (kg/g/ml) ก่อนเข้า AI
- รองรับการค้นหาแบบ Semantic Search ที่เข้าใจความหมายภาษาไทยแม้สะกดต่างกัน

---

## 🔄 **Workflow การทำงานปัจจุบัน**

```
1. Input Product Name
   └─ ผ่านการ Normalize โดย ThaiTextProcessor

2. AI Analysis (Local API Server)
   ├─ หา Keyword Match (60% weight)
   └─ คำนวณ Vector Embedding (40% weight)

3. PostgreSQL / Supabase Logic
   ├─ เปรียบเทียบกับฐานข้อมูลเดิม (pgvector)
   └─ สรุปคะแนนความคล้าย (Hybrid Score)

4. Classification Result
   ├─ Auto-Approve: ถ้าคะแนนสูงกว่า Threshold (เช่น > 0.8)
   └─ Pending Review: ถ้าคะแนนกึ่งกลาง (เช่น 0.6 - 0.79) เพื่อให้มนุษย์ตรวจ

5. Learning Loop
   └─ บันทึกผลตรวจเข้า ML Model (joblib) เพื่อพัฒนาความแม่นยำในอนาคต
```

---

## 📊 **Performance Metrics**

- **Embedding Speed**: < 0.05 วินาทีต่อสินค้า
- **Classification Time**: ~0.1 - 0.2 วินาที (รวม Database Query)
- **Memory Usage**: เสถียรที่ ~300-500 MB (ขึ้นกับขนาดโมเดล SBERT)

---

**📅 Last Updated**: 16 เมษายน 2569  
**🔖 Version**: 3.0 - Verified Hybrid Implementation
