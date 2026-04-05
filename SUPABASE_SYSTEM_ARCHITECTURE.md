# 🏗️ Supabase System Architecture

**Thai Product Taxonomy Manager - Complete System Overview**

---

## 📊 1. Database Tables (14 ตาราง)

### **✅ ตารางที่มีอยู่:**

```sql
-- Core Tables
1. taxonomy_nodes              -- หมวดหมู่ (67 รายการ)
2. products                    -- สินค้า + embeddings
3. imports                     -- รอบการนำเข้า
4. keyword_rules               -- กฎคำหลัก
5. synonym_lemmas              -- คำหลักพ้องความหมาย
6. synonym_terms               -- คำพ้องความหมายแต่ละคำ

-- AI & Suggestions
7. product_category_suggestions  -- คำแนะนำจาก AI
8. similarity_matches           -- สินค้าที่คล้ายกัน

-- Metadata
9. product_attributes          -- คุณสมบัติสินค้า
10. review_history             -- ประวัติการตรวจสอบ
11. human_feedback             -- Feedback จากผู้ใช้

-- System
12. regex_rules                -- กฎ regex
13. audit_logs                 -- Log การทำงาน
14. system_settings            -- การตั้งค่า
```

### **ตรวจสอบตาราง:**

```bash
# เชื่อมต่อ Supabase Docker
docker exec supabase_db_product_checker psql -U postgres

# แสดงตารางทั้งหมด
\dt

# ดูโครงสร้างตาราง
\d products
\d taxonomy_nodes
\d product_category_suggestions
```

---

## 🚀 2. Supabase Edge Functions (6 Functions)

### **ที่ตั้ง:** `taxonomy-app/supabase/functions/`

### **2.1 category-suggestions** ✅
```typescript
// File: supabase/functions/category-suggestions/index.ts

// จุดประสงค์: แนะนำหมวดหมู่โดยใช้ keyword matching
// Method: Keyword-based (ไม่ใช้ AI embeddings)

Input:
{
  "text": "กล่องล็อค 560 มล",
  "options": {
    "maxSuggestions": 5,
    "minConfidence": 0.3,
    "includeExplanation": true
  }
}

Output:
{
  "suggestions": [
    {
      "categoryId": "abc-123",
      "categoryName": "กล่อง/ที่เก็บของ",
      "confidence": 0.75,
      "matchedKeywords": ["กล่อง", "ล็อค"],
      "explanation": "Matched 2 keywords: กล่อง, ล็อค"
    }
  ],
  "processingTime": 45,
  "tokensUsed": 4
}
```

**วิธีเรียกใช้:**
```typescript
// From Next.js
import { EdgeFunctionAPI } from '@/utils/api'

const result = await EdgeFunctionAPI.getCategorySuggestions({
  text: "กล่องล็อค 560 มล",
  options: { maxSuggestions: 5 }
})
```

---

### **2.2 generate-embeddings** ✅
```typescript
// File: supabase/functions/generate-embeddings/index.ts

// จุดประสงค์: สร้าง vector embeddings
// Method: Local Model (Sentence Transformers) ผ่าน FastAPI

Input:
{
  "texts": ["กล่องล็อค 560", "ถังน้ำ 1000", "ขวดพลาสติก"],
  "model": "sentence-transformer"
}

Output:
{
  "embeddings": [
    [0.234, -0.567, ...],  // 384 dimensions (Local Model)
    [0.123, 0.456, ...],
    [0.789, -0.012, ...]
  ],
  "model": "paraphrase-multilingual-MiniLM-L12-v2",
  "usage": {
    "count": 3
  },
  "processingTime": 45
}
```

**วิธีเรียกใช้:**
```typescript
const result = await EdgeFunctionAPI.generateEmbeddings({
  texts: ["กล่องล็อค 560", "ถังน้ำ"],
  model: "sentence-transformer"
})
```

**⚠️ ต้องตั้งค่า URL ของ FastAPI:**
```bash
# ตั้งค่าใน Supabase Dashboard > Project Settings > Secrets
FASTAPI_URL=http://host.docker.internal:8000
```

---

### **2.3 hybrid-search** ✅
```typescript
// File: supabase/functions/hybrid-search/index.ts

// จุดประสงค์: ค้นหาแบบ Hybrid (Vector + Text)
// Method: Vector similarity + Text search

Input:
{
  "query": "กล่องล็อค",
  "type": "hybrid",  // 'vector' | 'text' | 'hybrid'
  "filters": {
    "categories": ["abc-123"],
    "priceRange": [0, 1000],
    "confidence": [0.5, 1.0]
  },
  "limit": 50,
  "offset": 0
}

Output:
{
  "results": [
    {
      "id": "prod-001",
      "name_th": "กล่องล็อค 560 มล",
      "category": {
        "id": "abc-123",
        "name_th": "กล่อง/ที่เก็บของ"
      },
      "similarity": 0.89,
      "matchType": "hybrid"
    }
  ],
  "total": 42,
  "processingTime": 234
}
```

**วิธีเรียกใช้:**
```typescript
const result = await EdgeFunctionAPI.hybridSearch({
  query: "กล่องล็อค",
  type: "hybrid",
  limit: 20
})
```

---

### **2.4 product-deduplication** ✅
```typescript
// File: supabase/functions/product-deduplication/index.ts

// จุดประสงค์: หาสินค้าที่ซ้ำกัน
// Method: Embedding similarity + Text matching

Input:
{
  "productName": "กล่องล็อค 560 มล",
  "threshold": 0.85
}

Output:
{
  "duplicates": [
    {
      "id": "prod-002",
      "name_th": "กล่อง ล็อค 560มล",
      "similarity": 0.92,
      "isDuplicate": true
    }
  ],
  "count": 1
}
```

---

### **2.5 exec-sql** ✅
```typescript
// File: supabase/functions/exec-sql/index.ts

// จุดประสงค์: Execute SQL queries (สำหรับ admin)
// ⚠️ ใช้ด้วยความระมัดระวัง!

Input:
{
  "query": "SELECT COUNT(*) FROM products WHERE status = 'pending'"
}

Output:
{
  "data": [...],
  "rowCount": 42
}
```

---

### **2.6 Functions ที่อ้างอิงใน code แต่ยังไม่มี:**

```typescript
// ใน utils/api.ts มีการเรียกใช้ แต่ยังไม่ได้สร้าง:

❌ similarity-matching      // ยังไม่มี
❌ batch-processing         // ยังไม่มี  
❌ thai-text-processing     // ยังไม่มี
```

---

## 🔌 3. API Endpoints Status

### **3.1 FastAPI (api_server.py) - Port 8000**

#### **✅ Endpoints ที่มีอยู่:**

```python
# Embed API
POST /api/embed
  Input:  {"text": "กล่องล็อค 560"}
  Output: {"embedding": [...], "dimension": 384}

POST /api/embed/batch
  Input:  {"texts": ["กล่อง", "ถัง"]}
  Output: {"embeddings": [[...], [...]], "count": 2}

# Health Check
GET /health
  Output: {"status": "healthy", "embeddings_ready": true}

# Root
GET /
  Output: {"name": "Thai Product Taxonomy API", ...}
```

#### **❌ Endpoints ที่ยังไม่มี (ต้องเพิ่ม):**

```python
# Category Classification
POST /api/classify/category
  Input:  {"product_name": "กล่องล็อค", "method": "hybrid"}
  Output: {
    "suggestions": [...],
    "top_suggestion": {
      "category_name": "กล่อง/ที่เก็บของ",
      "confidence": 0.72
    }
  }

# Batch Classification
POST /api/classify/batch
  Input:  {"products": [...], "method": "hybrid"}
  Output: [...]
```

---

### **3.2 Next.js API Routes - Port 3000**

```typescript
// taxonomy-app/app/api/

✅ /api/import/process
   - Upload CSV
   - Process products
   - Return stream (real-time updates)

✅ /api/import/approve
   - Approve products
   - Update database
```

---

## 📡 วิธีเรียกใช้ Edge Functions

### **Method 1: จาก Next.js (Recommended)**

```typescript
// File: utils/api.ts

import { supabase } from '@/utils/supabase'

// 1. Category Suggestions
const { data, error } = await supabase.functions.invoke('category-suggestions', {
  body: {
    text: "กล่องล็อค 560 มล",
    options: { maxSuggestions: 5 }
  }
})

// 2. Generate Embeddings
const { data, error } = await supabase.functions.invoke('generate-embeddings', {
  body: {
    texts: ["กล่อง", "ถัง"],
    model: "text-embedding-ada-002"
  }
})

// 3. Hybrid Search
const { data, error } = await supabase.functions.invoke('hybrid-search', {
  body: {
    query: "กล่องล็อค",
    type: "hybrid",
    limit: 20
  }
})
```

### **Method 2: จาก External API (cURL)**

```bash
# Set variables
SUPABASE_URL="http://localhost:54321"
ANON_KEY="your-anon-key"

# 1. Category Suggestions
curl -X POST \
  "$SUPABASE_URL/functions/v1/category-suggestions" \
  -H "Authorization: Bearer $ANON_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "กล่องล็อค 560 มล",
    "options": {"maxSuggestions": 5}
  }'

# 2. Hybrid Search
curl -X POST \
  "$SUPABASE_URL/functions/v1/hybrid-search" \
  -H "Authorization: Bearer $ANON_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "กล่องล็อค",
    "type": "hybrid"
  }'
```

---

## 🔄 Complete Data Flow with Services

### **Scenario: Import Products**

```
┌──────────────────────────────────────────────────────────┐
│ 1. FRONTEND (Next.js)                                    │
│    Upload CSV → /api/import/process                      │
└────────────────┬─────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────┐
│ 2. NEXT.JS API ROUTE                                     │
│    Parse CSV → Clean text → Tokenize                     │
└────────────────┬─────────────────────────────────────────┘
                 │
                 ├──► Option A: Call FastAPI (Python)
                 │    POST http://localhost:8000/api/embed
                 │    → Get embeddings (384-dim)
                 │
                 └──► Option B: Call Edge Function (Supabase)
                      supabase.functions.invoke('generate-embeddings-local')
                      → Get embeddings (384-dim Local Model)
                 
                 │
                 ▼
┌──────────────────────────────────────────────────────────┐
│ 3. CATEGORY CLASSIFICATION                               │
│                                                          │
│    Option A: Python FastAPI ✅ (แนะนำ - แม่นยำที่สุด)   │
│    POST http://localhost:8000/api/classify/category      │
│    → Hybrid Algorithm (72% accuracy)                     │
│                                                          │
│    Option B: Supabase Edge Function ✅ (มีแล้ว)         │
│    supabase.functions.invoke('hybrid-classification-local')
│    → Hybrid matching (Keyword 60% / Embedding 40%)       │
└────────────────┬─────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────┐
│ 4. SAVE TO DATABASE                                      │
│                                                          │
│    INSERT INTO products (...)                            │
│    INSERT INTO product_category_suggestions (...)        │
│    INSERT INTO product_attributes (...)                  │
│    UPDATE imports SET status = 'completed'               │
└────────────────┬─────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────┐
│ 5. USER REVIEW                                           │
│    Display suggestions → User approve/reject             │
│    UPDATE products SET category_id = ...                 │
└──────────────────────────────────────────────────────────┘
```

---

## 🎯 สรุปสถานะปัจจุบัน

### **✅ สิ่งที่มีและใช้งานได้:**

1. **Database Tables:** 14 ตาราง ครบถ้วน
2. **Supabase Edge Functions:**
   - ✅ hybrid-classification-local (Keyword 60% / Embedding 40%)
   - ✅ generate-embeddings-local (Local Model 384-dim)
   - ✅ hybrid-search (vector + text)
   - ✅ product-deduplication
   - ✅ exec-sql
3. **FastAPI Endpoints:**
   - ✅ POST /api/embed
   - ✅ POST /api/embed/batch
   - ✅ POST /api/classify/category (Hybrid AI)
4. **Next.js API:**
   - ✅ /api/import/process
   - ✅ /api/import/approve

### **❌ สิ่งที่ยังต้องปรับปรุง:**

1. **Next.js Route:**
   - ❌ ปรับปรุง `app/api/import/process/route.ts` ให้เรียกใช้ Hybrid API แทน Simple Logic

2. **Edge Functions (ที่อ้างอิงแต่ยังไม่มี):**
   - ❌ similarity-matching
   - ❌ batch-processing
   - ❌ thai-text-processing

---

## 🔧 แนะนำการใช้งาน

### **สำหรับ Category Classification:**

```typescript
// ✅ แนะนำ: ใช้ Hybrid AI (เรียก Python FastAPI)
const response = await fetch('http://localhost:8000/api/classify/category', {
  method: 'POST',
  body: JSON.stringify({ 
    product_name: productName, 
    method: 'hybrid' 
  })
})
```

### **สำหรับ Embeddings:**

```typescript
// ✅ แนะนำ: ใช้ Local Model (384-dim, FREE!)
const response = await fetch('http://localhost:8000/api/embed', {
  method: 'POST',
  body: JSON.stringify({ text: productName })
})
```

---

## 📚 ไฟล์อ้างอิง

- `taxonomy-app/supabase/functions/` - Edge Functions
- `taxonomy-app/utils/api.ts` - API calling utilities
- `api_server.py` - FastAPI service
- `test_category_algorithm.py` - Hybrid algorithm (ทดสอบแล้ว 72%)
- `INTEGRATION_STEPS.md` - วิธีเพิ่ม endpoints

---

**สรุป: ระบบมี 2 ทางเลือกสำหรับ AI Processing**
1. **FastAPI (Python)** - ต้องเพิ่ม category endpoint
2. **Supabase Edge Functions** - มี category-suggestions แล้ว (keyword-based)

**แนะนำ:** ใช้ Supabase Edge Functions สำหรับ production (serverless, scalable)
