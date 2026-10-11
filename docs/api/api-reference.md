# 🔌 API Reference (Single Source of Truth)

**อัปเดตล่าสุด:** 2026-10-11
**Project Constitution:** อ้างอิงตาม [AGENTS.md](../../AGENTS.md) และ [ADR 0004](../adr/0004-two-pillar-single-source-of-truth-types.md)

ระบบ API ของโปรเจกต์แบ่งออกเป็น 2 เสาหลัก:
1. **FastAPI AI Engine (Port 8000):** ให้บริการโมเดล Machine Learning, การประมวลผลข้อความไทย, และ Vector Embeddings (384-dim)
2. **Next.js Administrative Routes (Port 3000):** จัดการสิทธิ์, อัปเดตฐานข้อมูลผ่าน Service Role (`supabaseAdmin`), และทำหน้าที่เป็น Proxy Gateway

---

## 🧠 เสาหลักที่ 1: FastAPI Endpoints (`http://127.0.0.1:8000`)

### 1. หมวด System & Health (`src/api/routers/system.py`)

#### `GET /api/v1/health`
ตรวจสอบสถานะความพร้อมของเซิร์ฟเวอร์และ AI Components
- **Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2026-10-11T16:00:00.000Z",
  "version": "5.0.0",
  "uptime": 124.5,
  "components": {
    "pipeline": "operational",
    "embedding_model": "operational",
    "taxonomy_service": "operational"
  }
}
```

#### `POST /api/v1/clean`
ทำความสะอาดและ Normalize ชื่อสินค้าภาษาไทย (ตัดโปรโมชั่น, แปลงเลขไทย, จัดการหน่วยวัด)
- **Request Body:**
```json
{
  "texts": ["มาม่า(หมู) 60กรัม", "ข้าวสาร ๕ กก"]
}
```
- **Response (200 OK):**
```json
{
  "results": [
    { "original": "มาม่า(หมู) 60กรัม", "cleaned": "มาม่า หมู 60 กรัม" },
    { "original": "ข้าวสาร ๕ กก", "cleaned": "ข้าวสาร 5 กก." }
  ],
  "processing_time": 0.005
}
```

---

### 2. หมวด Vector Embeddings (`src/api/routers/embed.py`)

โมเดลที่ใช้งานคือ `paraphrase-multilingual-MiniLM-L12-v2` ให้ Vector ขนาด **384 มิติ**

#### `POST /api/embed`
สร้าง Vector สำหรับข้อความเดี่ยว
- **Request Body:**
```json
{
  "text": "กล่องล็อค 560 มล"
}
```
- **Response (200 OK):**
```json
{
  "embedding": [0.0234, -0.0567, 0.0123],
  "dimension": 384,
  "model": "paraphrase-multilingual-MiniLM-L12-v2",
  "processing_time": 0.015
}
```

#### `POST /api/embed/batch`
สร้าง Vectors สำหรับรายการข้อความ
- **Request Body:**
```json
{
  "texts": ["สินค้า ก", "สินค้า ข"]
}
```
- **Response (200 OK):**
```json
{
  "embeddings": [[0.0234, ...], [0.0112, ...]],
  "count": 2,
  "dimension": 384,
  "model": "paraphrase-multilingual-MiniLM-L12-v2",
  "processing_time": 0.028
}
```

---

### 3. หมวด Classification (`src/api/routers/classify.py`)

ใช้อัลกอริทึม Hybrid: **Keyword Rules 60% + Cosine Vector 40%**

#### `POST /api/classify/category`
จำแนกหมวดหมู่สินค้าเดี่ยว
- **Request Body:**
```json
{
  "product_name": "กาแฟปรุงสำเร็จชนิดผง 3in1",
  "method": "hybrid",
  "top_k": 3
}
```
- **Response (200 OK):**
```json
{
  "product_name": "กาแฟปรุงสำเร็จชนิดผง 3in1",
  "category_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "category_name": "เครื่องดื่มชง",
  "confidence": 0.85,
  "suggestions": [
    {
      "category_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
      "category_name": "เครื่องดื่มชง",
      "confidence": 0.85,
      "method": "hybrid",
      "matched_keyword": "กาแฟ"
    }
  ],
  "processing_time": 0.042
}
```

#### `POST /api/classify/batch`
จำแนกหมวดหมู่สินค้าเป็นชุด
- **Request Body:**
```json
{
  "products": ["สินค้า 1", "สินค้า 2"],
  "method": "hybrid",
  "top_k": 3
}
```

---

### 4. หมวด Deduplication & Matching (`src/api/routers/internal_match.py`)

#### `POST /api/v1/match/scan-internal`
เริ่มงานสแกนหาคู่สินค้าที่ซ้ำซ้อนภายในคลังสินค้า (Background Asynchronous Task)
- **Request Body:**
```json
{
  "threshold": 0.95,
  "limit": 100
}
```
- **Response (200 OK):**
```json
{
  "task_id": "4f9d2d41-1122-3344-5566-778899aabbcc",
  "status": "pending",
  "message": "กำลังยืนยันคิวงานสแกน..."
}
```

#### `GET /api/v1/match/scan-internal/status/{task_id}`
ตรวจสอบความคืบหน้าหรือผลลัพธ์ของงานสแกน
- **Response (200 OK - Processing / Completed):**
```json
{
  "task_id": "4f9d2d41-1122-3344-5566-778899aabbcc",
  "status": "completed",
  "progress": 100,
  "message": "สแกนตรวจสอบเสร็จสิ้นเรียบร้อยแล้ว!",
  "result": {
    "total_scanned": 3103,
    "pairs_found": 12,
    "results": [
      {
        "id_a": "uuid-1",
        "name_a": "กะละมัง 004 58 สี sp",
        "sku_a": "SKU-001",
        "category_a": "เครื่องใช้ในบ้าน",
        "id_b": "uuid-2",
        "name_b": "กะละมัง 004 58 สี SP",
        "sku_b": "SKU-002",
        "category_b": "เครื่องใช้ในบ้าน",
        "similarity": 0.985,
        "ml_prediction": "similar",
        "confidence": 0.985,
        "reason": "ความคล้าย 98.5%"
      }
    ],
    "processing_time": 1.25,
    "fallback_active": false,
    "system_warnings": []
  },
  "error": null
}
```

#### `POST /api/v1/match/import-dedup`
ตรวจจับความซ้ำซ้อนของสินค้านำเข้าเทียบกับฐานข้อมูลสินค้าจริง
- **Request Body:**
```json
{
  "products": ["แปรงสีฟันคอลเกต"],
  "threshold": 0.75
}
```
- **Response (200 OK):** List of `ProductMatchResult` (พร้อมฟิลด์ `identity_verdict`)

#### `POST /api/v1/match/file-dedup`
เปรียบเทียบไฟล์ CSV สินค้าเก่าและใหม่ที่อัปโหลดไว้ใน Supabase Storage Bucket `uploads`

---

### 5. หมวด Continuous Learning (`src/api/routers/learn.py`)

#### `POST /api/v1/learn/verify`
เรียนรู้หมวดหมู่เมื่อมนุษย์ตรวจสอบยืนยันหมวดหมู่สินค้าใน UI
- **Request Body:**
```json
{
  "product_name": "ผงซักฟอกบรีส เอกเซล 800ก",
  "category_id": "cat-uuid"
}
```

#### `GET /api/v1/learn/status`
ดึงความแม่นยำ (Accuracy) และ Feature Importance ของโมเดล ML ล่าสุด

#### `POST /api/v1/learn/retrain`
ทริกเกอร์การเทรนโมเดล ML ใหม่จากข้อมูลรีวิวใน `similarity_matches`

#### `GET /api/v1/learn/history`
ดึงประวัติการเทรนโมเดลจากตาราง `ml_training_history`

---

## 🛡️ เสาหลักที่ 2: Next.js API Routes (`http://localhost:3000/api`)

ทุก Route ได้รับการปกป้องโดย `middleware.ts` (ยกเว้น `/unlock`) และส่งคืน JSON Envelope: `{ success: boolean, data?: T, error?: string, message?: string }`

| Route Path | Methods | หน้าที่และความรับผิดชอบ |
|---|---|---|
| `/api/unlock` | POST | ปลดล็อก Session Cookie (30 วัน) ด้วย `INTERNAL_API_SECRET` |
| `/api/lock` | POST | ลบคุกกี้ Session เพื่อล็อกหน้าระบบ |
| `/api/import/history` | GET | ดึงประวัติการนำเข้าไฟล์จากตาราง `imports` ผ่าน Service Role |
| `/api/import/commit` | POST | บันทึกชุดสินค้านำเข้าเข้าสู่ฐานข้อมูล (`products`, `imports`, `similarity_matches`) |
| `/api/taxonomy` | POST | สร้างโหนดหมวดหมู่ใหม่ใน `taxonomy_nodes` |
| `/api/taxonomy/[id]` | PUT, DELETE | แก้ไขหรือลบโหนดหมวดหมู่ |
| `/api/synonyms` | POST | สร้างชุดคำพ้องเสียง/คำพ้องรูป (`synonym_lemmas`, `synonym_terms`) |
| `/api/synonyms/[id]` | PUT, DELETE | แก้ไขหรือลบคำพ้อง |
| `/api/keyword-rules/[id]` | DELETE | ลบกฎ Keyword Rule จากตาราง `keyword_rules` อย่างปลอดภัย |
| `/api/similarity/resolve` | POST | จัดการคู่สินค้าซ้ำ (Action: `merge` หรือ `keep_both`) |
| `/api/recheck` | GET, POST | ดึงคู่สินค้าสำหรับตรวจซ้ำ (Recheck) และบันทึกผลการยืนยัน |
| `/api/verify` | POST | ยืนยันการจัดหมวดหมู่สินค้าและส่งต่อการเรียนรู้ไปยัง FastAPI `/learn/verify` |
| `/api/pos-export` | POST | Export ข้อมูลสินค้าที่อนุมัติแล้วในรูปแบบ POS พร้อมบันทึกประวัติ |
| `/api/settings` | GET, POST | อ่านและบันทึกการตั้งค่าระบบลงในตาราง `system_settings` |
