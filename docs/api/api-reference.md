# 🔌 API Reference & Testing Guide (v4.0 — Modular Architecture)

> 📖 **Project Constitution**: ระบบนี้ทำงานบนฐานของ FastAPI และใช้ Hybrid Algorithm (Keyword 60% + Embedding 40%) ตามที่ระบุใน [GEMINI.md](../../GEMINI.md)

## 🚀 **API Server Setup**

### **Starting the Server**
```bash
# วิธีแนะนำ: ดับเบิลคลิก
START_PHAYAK.bat

# หรือรันตรง
.venv\Scripts\python src\api\api_server.py

# Server จะรันที่:
# - API Home:        http://127.0.0.1:8000
# - Swagger UI:      http://127.0.0.1:8000/docs
# - Health Check:    http://127.0.0.1:8000/api/v1/health
```

---

## 📋 **Embedding API** — `src/api/routers/embed.py`

### **1. Single Embedding**
แปลงชื่อสินค้าเป็น Vector 384 มิติ ใช้โดย Supabase Edge Functions
```http
POST /api/embed
Content-Type: application/json
```
```json
{ "text": "กล่องล็อค 560 มล" }
```
**Response:**
```json
{
  "embedding": [0.234, -0.567, ...],
  "dimension": 384,
  "model": "paraphrase-multilingual-MiniLM-L12-v2",
  "processing_time": 0.045
}
```

### **2. Batch Embedding**
```http
POST /api/embed/batch
Content-Type: application/json
```
```json
{ "texts": ["สินค้า A", "สินค้า B", "สินค้า C"] }
```

---

## 📋 **Matching API** — `src/api/routers/match.py`

### **3. Single Match**
```http
POST /api/v1/match/single
Content-Type: application/json
```
```json
{
  "query_product": "กล่องล็อค 560",
  "reference_products": ["กล่อง 560 มล", "ถัง 1L", "ขวด 500"],
  "threshold": 0.6,
  "top_k": 5
}
```

### **4. Batch Match (Background Job)**
```http
POST /api/v1/match/batch
Content-Type: application/json
```
ส่งคืน `job_id` สำหรับติดตามสถานะผ่าน `/api/v1/jobs/{job_id}`

---

## 📋 **System API** — `src/api/routers/system.py`

### **5. ล้างชื่อสินค้าภาษาไทย**
```http
POST /api/v1/clean
Content-Type: application/json
```
```json
{ "texts": ["มาม่า(หมู) 60กรัม", "ข้าวสาร ๕ กก"] }
```

### **6. Health Check**
```http
GET /api/v1/health
```

---

## 📋 **Jobs API** — `src/api/routers/jobs.py`

### **7. เช็คสถานะ Job**
```http
GET /api/v1/jobs/{job_id}
```

### **8. ดึงผลลัพธ์ Job**
```http
GET /api/v1/results/{job_id}
```

---

## 📋 **Learning API** — `src/api/routers/learn.py`

### **9. สอน AI จากการยืนยันหมวดหมู่**
```http
POST /api/v1/learn/verify
Content-Type: application/json
```
```json
{
  "product_name": "มาม่าหมูสับ 60ก",
  "category_id": "cat_001"
}
```

---

## 🏗️ **Deduplication Workflow**

```bash
python scripts/complete_deduplication_pipeline.py --input data.csv --mode analyze
python scripts/complete_deduplication_pipeline.py --input data.csv --mode review
python scripts/complete_deduplication_pipeline.py --mode train
```

---

**📅 Last Updated**: 24 พฤษภาคม 2569 (v4.0 — แยก Routers, ลบ embed_service.py ที่ไม่มีจริง)
