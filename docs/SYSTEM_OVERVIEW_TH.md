# 📚 ภาพรวมระบบ - Thai Product Taxonomy Manager

**สรุปการทำงาน Database, API และสิ่งที่ต้องแก้ไข**

---

## 🗄️ ตารางในฐานข้อมูล (14 ตาราง)

### **ตารางหลักสำหรับการนำเข้าสินค้า:**

#### **1. taxonomy_nodes** - หมวดหมู่สินค้า
```
📂 มีข้อมูล: 67 หมวดหมู่ (12 หลัก + 55 ย่อย)

โครงสร้าง:
├── id (UUID)                 ← Primary Key
├── name_th (ชื่อไทย)         ← "กล่อง/ที่เก็บของ"
├── name_en (ชื่ออังกฤษ)     ← "Storage Box"
├── parent_id                 ← หมวดหมู่แม่
├── level (ระดับ)             ← 0=หลัก, 1=ย่อย
├── keywords (คำหลัก[])       ← ["กล่อง", "เก็บของ"]
└── metadata (ข้อมูลเพิ่มเติม)

ใช้เพื่อ:
✅ เก็บโครงสร้างหมวดหมู่
✅ ให้ AI จับคู่กับสินค้า
✅ แสดงในระบบ dropdown
```

#### **2. products** - สินค้าที่นำเข้า
```
📦 มีข้อมูล: สินค้าที่ import ทั้งหมด

โครงสร้าง:
├── id (UUID)
├── name_th                   ← "กล่องล็อค 560 มล"
├── description               ← ข้อความที่ทำความสะอาดแล้ว
├── category_id               ← FK → taxonomy_nodes (หมวดหมู่ที่เลือก)
├── embedding (vector[384])   ← AI embeddings สำหรับหาความคล้าย
├── keywords (text[])         ← ["กล่อง", "ล็อค", "560"]
├── confidence_score          ← ความมั่นใจของ AI (0-1)
├── status                    ← 'pending', 'approved', 'rejected'
├── import_batch_id           ← FK → imports (รอบการนำเข้า)
└── metadata                  ← {units, colors, sizes, etc.}

ใช้เพื่อ:
✅ เก็บสินค้าทั้งหมดที่นำเข้า
✅ เก็บ AI embeddings สำหรับค้นหา
✅ ติดตามสถานะการอนุมัติ
```

#### **3. imports** - รอบการนำเข้า
```
📊 ติดตาม progress ของแต่ละรอบ import

โครงสร้าง:
├── id (UUID)
├── name                      ← "Product Import - 04/10/2025"
├── file_name                 ← "products.csv"
├── total_records             ← 296 (จำนวนทั้งหมด)
├── processed_records         ← 296 (ประมวลผลแล้ว)
├── success_records           ← 290 (สำเร็จ)
├── error_records             ← 6 (ล้มเหลว)
├── status                    ← 'pending', 'processing', 'completed'
├── started_at
└── completed_at

ใช้เพื่อ:
✅ ติดตามความคืบหน้า
✅ แสดง progress bar
✅ สรุปผลการนำเข้า
```

#### **4. product_category_suggestions** - คำแนะนำจาก AI
```
🤖 เก็บคำแนะนำหมวดหมู่จาก AI

โครงสร้าง:
├── id (UUID)
├── product_id                ← FK → products
├── suggested_category_id     ← FK → taxonomy_nodes (AI แนะนำ)
├── confidence_score          ← 0.72 (ความมั่นใจ)
├── suggestion_method         ← 'keyword', 'embedding', 'hybrid'
├── metadata                  ← {explanation, matched_keywords}
├── is_accepted               ← true/false/null (user approved?)
└── reviewed_at

ใช้เพื่อ:
✅ เก็บคำแนะนำของ AI
✅ ให้ user approve/reject
✅ เรียนรู้จาก feedback
```

#### **5. product_attributes** - คุณสมบัติสินค้า
```
🏷️ เก็บคุณสมบัติที่สกัดได้

โครงสร้าง:
├── product_id                ← FK → products
├── attribute_name            ← "color", "size", "unit"
├── attribute_value           ← "แดง", "L", "500ml"
└── attribute_type            ← "text", "number"

ใช้เพื่อ:
✅ เก็บสี, ขนาด, หน่วย
✅ ใช้ในการค้นหา filter
✅ แสดงใน product detail
```

#### **6. keyword_rules** - กฎคำหลัก
```
🔑 กฎสำหรับจับคู่คำหลักกับหมวดหมู่

โครงสร้าง:
├── category_id               ← FK → taxonomy_nodes
├── keywords (text[])         ← ["กล่อง", "ล็อค", "เก็บของ"]
├── priority                  ← 1-10 (ความสำคัญ)
└── is_active

ใช้เพื่อ:
✅ Keyword matching algorithm
✅ เพิ่มความแม่นยำของ AI
✅ Admin สามารถจัดการได้
```

---

## 🔄 Data Flow - ขั้นตอนการนำเข้าสินค้า

### **ภาพรวม:**

```
1. USER ──(Upload CSV)──► 2. FRONTEND ──(API Call)──► 3. NEXT.JS API
                                                          │
                                                          ▼
                                                    4. PYTHON AI
                                                    (Embedding + 
                                                     Classifier)
                                                          │
                                                          ▼
                                                    5. DATABASE
                                                    (Save Results)
                                                          │
                                                          ▼
                                                    6. USER REVIEW
                                                    (Approve/Reject)
```

### **รายละเอียดแต่ละขั้นตอน:**

#### **STEP 1: Frontend Upload** 📤
```typescript
// File: components/Import/UploadAndMappingStep.tsx (ขั้นที่ 1 ของ WizardTab)

Input:
  • CSV/Excel file จาก user
  • Column mapping (ชื่อสินค้า / SKU / ราคาขาย)

Process:
  1. อ่านไฟล์ในเบราว์เซอร์ (ยังไม่แตะฐานข้อมูล)
  2. ให้ผู้ใช้จับคู่คอลัมน์ แล้วส่งต่อเป็น state ให้ขั้นถัดไป

Output:
  • rawRows[] + columnMapping (อยู่ใน state ของ WizardTab)
```

> **หมายเหตุ (29 ส.ค. 2569):** ก่อนหน้านี้ขั้นตอนนี้อยู่ใน `ProcessingStep.tsx` ที่ยิง
> `POST /api/import/process` — ทั้งคอมโพเนนต์และ route ถูกลบไปแล้ว ของจริงตอนนี้คือ
> วิซาร์ด 5 ขั้นใน `WizardTab.tsx` ที่บันทึกครั้งเดียวตอนจบผ่าน `/api/import/commit`

#### **STEP 2: ทำความสะอาดข้อความ** 🔤
```typescript
// File: components/Import/DataCleaningStep.tsx → POST {FASTAPI}/api/v1/clean

Input:
  • rawRows[] + columnMapping จากขั้นที่ 1

Process (สำหรับแต่ละสินค้า):
  1. Clean Text:
     "กล่องล็อค 560 มล!!!" → "กล่องล็อค 560 มล"
     
  2. Tokenize:
     "กล่องล็อค 560 มล" → ["กล่อง", "ล็อค", "560", "มล"]
     
  3. Extract Units:
     → ["560 มล"]
     
  4. Extract Attributes:
     → {colors: ["แดง"], sizes: ["L"]}

Output:
  • cleaned_text
  • tokens[]
  • units[]
  • attributes{}
```

#### **STEP 3: Python AI - Embedding** 🧠
```python
# Service: src/api/routers/embed.py (FastAPI - Port 8000)

Input:
  • text: "กล่องล็อค 560 มล"

Process:
  1. Load model: paraphrase-multilingual-MiniLM-L12-v2
  2. Generate embedding vector (384 dimensions)
  3. Normalize (L2 normalization)

Output:
  • embedding: [0.234, -0.567, 0.123, ...]  # 384 numbers

ใช้เพื่อ:
  ✅ หาสินค้าที่คล้ายกัน (cosine similarity)
  ✅ จัดกลุ่มสินค้า
  ✅ ค้นหาแบบ semantic
```

#### **STEP 4: Python AI - Category Classification** 🎯
```python
# Service: src/api/api_server.py (FastAPI - Port 8000) — Modular v4.0

Input:
  • product_name: "กล่องล็อค 560 มล"
  • method: "hybrid" (keyword + embedding)

Process:
  1. Keyword Method (60% weight):
     - Match กับ keyword_rules
     - "กล่อง" พบใน category "กล่อง/ที่เก็บของ"
     - Confidence: 0.9
  
  2. Embedding Method (40% weight):
     - คำนวณ similarity กับทุก category
     - "กล่อง/ที่เก็บของ" similarity: 0.52
     - Confidence: 0.52
  
  3. Hybrid (Combine):
     - Final = 0.9 * 0.6 + 0.52 * 0.4
     - Final = 0.54 + 0.21 = 0.75

Output:
  • category_id: "abc-123"
  • category_name: "กล่อง/ที่เก็บของ"
  • confidence: 0.75
  • explanation: "พบคำที่ตรงกัน: กล่อง"
```

#### **STEP 5: Database Save** 💾
```typescript
// File: app/api/import/commit/route.ts (เรียกจาก WizardTab ขั้นสุดท้าย)

Input:
  • items[] (ผ่านการทำความสะอาด + เทียบซ้ำ + จัดหมวดมาแล้ว)
  • run_id — กันกดบันทึกซ้ำ (unique index บน imports)

Process:
  1. สร้างรอบ import ใน `imports` (status='processing')
  2. INSERT INTO products แบบเป็นชุด พร้อม embedding และ metadata
     (clean_name / similarity_score / duplicate_of — ชื่อ key ต้องตรงกับที่ VerifyTab อ่าน)
     status ตั้งตามผลเทียบซ้ำ:
       duplicate → 'rejected'
       review    → 'pending_review_dedup'   (ไปด่าน 1 ในหน้า /data-quality)
       new       → 'pending_review_category' (ไปด่าน 2)
  3. INSERT INTO product_category_suggestions — เก็บหมวดที่ AI เสนอ
  4. UPDATE imports SET processed/success/error records, status='completed'

Output:
  • products มีของใหม่รอคนตรวจที่ /data-quality
  • imports.status = 'completed' (โชว์ในแท็บประวัติของหน้า /import)
```

> **หมายเหตุ (29 ส.ค. 2569):** งานตรวจของคนอยู่ที่ `/data-quality` ที่เดียว และบันทึกผ่าน
> `POST /api/verify` (service role) — เขียนตรงจากเบราว์เซอร์ด้วย anon key ไม่ได้ RLS จะกรองทิ้งเงียบๆ

#### **STEP 6: User Review** 👤
```typescript
// User Interface

Display:
  ┌─────────────────────────────────────┐
  │ สินค้า: กล่องล็อค 560 มล           │
  │ AI แนะนำ: กล่อง/ที่เก็บของ (75%)   │
  │                                     │
  │ [✓ อนุมัติ]  [✗ ปฏิเสธ]  [⏭ ข้าม]  │
  └─────────────────────────────────────┘

User Actions:
  1. อนุมัติ (Approve):
     UPDATE products 
     SET category_id = 'abc-123', status = 'approved'
     
     UPDATE product_category_suggestions
     SET is_accepted = true
  
  2. ปฏิเสธ (Reject):
     → แสดง dropdown เลือกหมวดหมู่ใหม่
     UPDATE products SET category_id = <user_selected>
     
     UPDATE product_category_suggestions
     SET is_accepted = false
     
     INSERT INTO human_feedback (...)
     → เก็บไว้เรียนรู้
  
  3. ข้าม (Skip):
     → status ยัง 'pending'
```

---

## ⚠️ สิ่งที่ต้องแก้ไข

### **1. Backend Services** (ต้องเพิ่ม/แก้)

#### **A. `src/api/api_server.py`** - FastAPI Entry Point (Port 8000, Modular v4.0)

```
✅ Endpoints ที่มีอยู่แล้ว (ครบถ้วน):
  - POST /api/embed               → src/api/routers/embed.py
  - POST /api/embed/batch         → src/api/routers/embed.py
  - POST /api/v1/match/single     → src/api/routers/match.py
  - POST /api/v1/match/batch      → src/api/routers/match.py
  - POST /api/v1/clean            → src/api/routers/system.py
  - GET  /api/v1/health           → src/api/routers/system.py
  - GET  /api/v1/jobs/{id}        → src/api/routers/jobs.py
  - POST /api/v1/learn/verify     → src/api/routers/learn.py
  - GET  /docs                    → Swagger UI

✅ รันด้วย: START_PHAYAK.bat
```

---

### **2. การจัดหมวดหมู่ตอนนำเข้า** (แก้แล้ว)

เอกสารเดิมตรงนี้เป็นรายการสิ่งที่ต้องแก้ใน `app/api/import/process/route.ts` ซึ่ง
**ลบทิ้งไปแล้ว** พร้อม `ProcessingStep.tsx` และสิ่งที่มันขอให้ทำก็ทำเสร็จแล้วเช่นกัน:
ขั้นจัดหมวดของวิซาร์ดเรียก hybrid classifier ของ FastAPI ตรงๆ

```typescript
// File: components/Import/CategorizationStep.tsx
const res = await fetch(`${apiBase}/api/classify/category`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ product_name: name, method: 'hybrid', top_k: 5 })
})
```

---

### **3. Embedding Generation** (ใช้งานได้แล้ว)

**ปัจจุบัน:**
```typescript
// ✅ ถูกต้องแล้ว: เรียก FastAPI port 8000
const response = await fetch('http://127.0.0.1:8000/api/embed', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text }),
  signal: AbortSignal.timeout(10000)
})
```

**หมายเหตุ:**
- `/api/embed` อยู่ใน `src/api/routers/embed.py` (FastAPI Port 8000)
- ใช้ SentenceTransformerModel (paraphrase-multilingual-MiniLM-L12-v2)
- Return 384-dimensional vector

---

## 🚀 วิธีเริ่มระบบ (Development)

### **ลำดับการ start services:**

```bash
# วิธีแนะนำ: ดับเบิลคลิกไฟล์เดียว
START_PHAYAK.bat
# ✅ Backend:  http://127.0.0.1:8000
# ✅ Frontend: http://localhost:3000

# --- หรือรันแยก ---

# 1. Start Supabase (Database)
cd d:\product_checker\check-products\taxonomy-app
npx supabase start
# ✅ Database: http://127.0.0.1:54331

# 2. Start FastAPI Backend (Port 8000)
cd d:\product_checker\check-products
.venv\Scripts\python src\api\api_server.py
# ✅ API: http://127.0.0.1:8000
# ✅ Swagger: http://127.0.0.1:8000/docs

# 3. Start Next.js Frontend (Port 3000)
cd taxonomy-app
npm run dev
# ✅ Frontend: http://localhost:3000
```

---

## 🧪 วิธีทดสอบ

### **Test 1: Embedding API**
```bash
curl -X POST http://127.0.0.1:8000/api/embed \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"กล่องล็อค 560 มล\"}"

# Expected: 
# {
#   "embedding": [0.234, -0.567, ...],
#   "dimension": 384,
#   "model": "paraphrase-multilingual-MiniLM-L12-v2",
#   "processing_time": 0.045
# }
```

### **Test 2: Category Classifier**
```bash
curl -X POST http://127.0.0.1:8000/api/classify/category \
  -H "Content-Type: application/json" \
  -d "{\"product_name\": \"กล่องล็อค 560 มล\", \"method\": \"hybrid\"}"

# Expected:
# {
#   "suggestions": [...],
#   "top_suggestion": {
#     "category_name": "กล่อง/ที่เก็บของ",
#     "confidence": 0.75
#   }
# }
```

### **Test 3: Full Import Flow**
```bash
# 1. Open http://localhost:3000/import
# 2. Upload CSV file
# 3. Watch console for:
#    ✅ File uploaded
#    ✅ Products parsed
#    ✅ Embeddings generated
#    ✅ Categories suggested
#    ✅ Saved to database
# 4. Check database:
SELECT * FROM products WHERE import_batch_id = '<batch-id>';
```

---

## 📊 สรุปความสัมพันธ์ของ Tables

```
┌─────────────────┐
│ imports         │ ← สร้างตอน upload CSV
│ (batch)         │
└────┬────────────┘
     │
     │ One-to-Many
     │
     ▼
┌─────────────────┐
│ products        │ ← สร้างหลัง AI ประมวลผล
│ (สินค้า)        │
└────┬────────────┘
     │
     ├──► product_category_suggestions  ← AI แนะนำ
     │    (คำแนะนำ)
     │
     └──► product_attributes            ← คุณสมบัติ
          (สี, ขนาด, หน่วย)

┌─────────────────┐
│ taxonomy_nodes  │ ← หมวดหมู่ (67 รายการ)
│ (หมวดหมู่)      │
└────┬────────────┘
     │
     │ Referenced by
     │
     ├──► products.category_id
     ├──► product_category_suggestions.suggested_category_id
     └──► keyword_rules.category_id
```

---

## 🎯 Next Steps

1. **Backend พร้อมแล้ว (v4.0):**
   - `src/api/api_server.py` มี Endpoints ครบทุกตัว
   - รันผ่าน `START_PHAYAK.bat` ได้ทันที

2. **ทดสอบ Full Import Flow:**
   - Start ระบบผ่าน `START_PHAYAK.bat`
   - Upload CSV ที่ http://localhost:3000/import
   - ตรวจสอบผลลัพธ์ใน database

3. **Monitor:**
   - Swagger UI: http://127.0.0.1:8000/docs
   - Health Check: http://127.0.0.1:8000/api/v1/health

---

## 📚 เอกสารอ้างอิง

- **DATABASE_SCHEMA.md** - รายละเอียด tables ทั้งหมด
- **INTEGRATION_STEPS.md** - ขั้นตอนการ integrate แบบละเอียด
- **FINAL_REPORT.md** - สรุปผลการทดสอบ algorithm
- **TEST_SUMMARY.md** - สรุปแบบย่อ

---

**สรุป: ระบบ Architecture ครบแล้ว — ใช้ `START_PHAYAK.bat` รันได้ทันที** ✅

*Last Updated: 24 พฤษภาคม 2569 (v4.0 — Modular FastAPI)*
