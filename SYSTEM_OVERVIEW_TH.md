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
// File: ProcessingStep.tsx

Input:
  • CSV file จาก user
  • Column mapping (ระบุว่าคอลัมน์ไหนคือชื่อสินค้า)

Process:
  1. อัปโหลดไฟล์ไปยัง Supabase Storage
  2. สร้างรอบ import ใน table `imports`
  3. เรียก API: POST /api/import/process

Output:
  • import_batch_id (UUID)
  • File path in storage
```

#### **STEP 2: Next.js API - Text Processing** 🔤
```typescript
// File: taxonomy-app/app/api/import/process/route.ts

Input:
  • CSV file (FormData)

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
# Service: embed_service.py (Flask - Port 5000)

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
# Service: api_server.py (FastAPI - Port 8000)
# ⚠️ ต้องเพิ่ม endpoint นี้!

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
// File: ProcessingStep.tsx → saveProductsToDatabase()

Input:
  • ProcessedProduct[] (จาก AI)
  • import_batch_id

Process:
  For each product:
    1. INSERT INTO products (...)
       → สร้างสินค้า
    
    2. INSERT INTO product_category_suggestions (...)
       → บันทึกคำแนะนำของ AI
    
    3. INSERT INTO product_attributes (...)
       → บันทึกคุณสมบัติ (สี, ขนาด, หน่วย)
  
  Update imports SET 
    processed_records = 296,
    success_records = 290,
    status = 'completed'

Output:
  • products table มีข้อมูลใหม่
  • product_category_suggestions มีคำแนะนำ
  • imports.status = 'completed'
```

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

#### **A. api_server.py** - FastAPI Service (Port 8000)

```python
# ✅ มี Embed API อยู่แล้ว:
  - POST /api/embed          (single text)
  - POST /api/embed/batch    (multiple texts)

# ⚠️ ปัจจุบัน: ไม่มี endpoint สำหรับ category classification
# ✅ ต้องเพิ่ม: /api/classify/category

# สิ่งที่ต้องทำ:
1. Import CategoryClassifier จาก test_category_algorithm.py
2. Initialize classifier ตอน startup
3. สร้าง endpoint: POST /api/classify/category
4. Return: category suggestions (hybrid method)

# ไฟล์อ้างอิง:
- test_category_algorithm.py ← Logic ที่ทดสอบแล้ว (72% accuracy)
- INTEGRATION_STEPS.md ← มี code ตัวอย่างครบ
```

---

### **2. Next.js API Route** (ต้องแก้)

#### **File: taxonomy-app/app/api/import/process/route.ts**

**ปัญหา:**
```typescript
// ❌ ปัจจุบัน: ใช้ keyword matching แบบง่าย
async function suggestCategory(tokens, attributes, embedding) {
  // Simple keyword matching only
  // Confidence ไม่แม่นยำ
  // ไม่ได้ใช้ hybrid algorithm
}
```

**ต้องแก้เป็น:**
```typescript
// ✅ ใหม่: เรียก Python API (Hybrid Algorithm)
async function suggestCategory(
  productName: string,
  tokens: string[], 
  attributes: Record<string, any>, 
  embedding: number[]
) {
  // Call FastAPI Category Classifier
  const response = await fetch('http://localhost:8000/api/classify/category', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      product_name: productName,
      method: 'hybrid',
      top_k: 5
    })
  })
  
  return response.json()
}
```

**ตำแหน่งที่ต้องแก้:**
```typescript
// บรรทัดที่ ~295 ใน route.ts
const suggestion = await suggestCategory(
  productName,  // ← เพิ่ม parameter นี้
  tokens, 
  attributes, 
  embedding
)
```

---

### **3. Embedding Generation** (ใช้งานได้แล้ว)

**ปัจจุบัน:**
```typescript
// ✅ ถูกต้องแล้ว: เรียก FastAPI port 8000
const response = await fetch('http://localhost:8000/api/embed', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text }),
  signal: AbortSignal.timeout(10000)
})
```

**หมายเหตุ:**
- api_server.py มี /api/embed และ /api/embed/batch อยู่แล้ว
- ใช้ SentenceTransformerModel (paraphrase-multilingual-MiniLM-L12-v2)
- Return 384-dimensional vector

---

## 🚀 วิธีเริ่มระบบ (Development)

### **ลำดับการ start services:**

```bash
# 1. Start Supabase (Database)
cd d:\product_checker\check-products\taxonomy-app
npx supabase start
# ✅ Database: http://localhost:54321

# 2. Start FastAPI Service (Port 8000)
# มี Embed API + ต้องเพิ่ม Category Classifier
cd d:\product_checker\check-products
python api_server.py
# ✅ API: http://localhost:8000
# - /api/embed (มีแล้ว)
# - /api/embed/batch (มีแล้ว)
# - /api/classify/category (ต้องเพิ่ม)

# 3. Start Next.js Frontend (Port 3000)
cd taxonomy-app
npm run dev
# ✅ Frontend: http://localhost:3000
```

---

## 🧪 วิธีทดสอบ

### **Test 1: Embedding API**
```bash
curl -X POST http://localhost:8000/api/embed \
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
curl -X POST http://localhost:8000/api/classify/category \
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

1. **แก้ไข api_server.py:**
   - เพิ่ม `/api/classify/category` endpoint
   - ใช้ logic จาก test_category_algorithm.py
   - ดู code ตัวอย่างใน INTEGRATION_STEPS.md

2. **แก้ไข route.ts:**
   - Update `suggestCategory()` function
   - Update `generateEmbedding()` function
   - เปลี่ยน port จาก 8000 → 5000 สำหรับ embeddings

3. **ทดสอบ:**
   - Start ทั้ง 4 services
   - Upload CSV ทดสอบ
   - ตรวจสอบผลลัพธ์ใน database

4. **Deploy:**
   - ใช้ Docker Compose (แนะนำ)
   - หรือ deploy แยก service

---

## 📚 เอกสารอ้างอิง

- **DATABASE_SCHEMA.md** - รายละเอียด tables ทั้งหมด
- **INTEGRATION_STEPS.md** - ขั้นตอนการ integrate แบบละเอียด
- **FINAL_REPORT.md** - สรุปผลการทดสอบ algorithm
- **TEST_SUMMARY.md** - สรุปแบบย่อ

---

**สรุป: ระบบ Architecture ครบแล้ว ต้องเชื่อมต่อ API ให้ถูกต้อง** ✅
