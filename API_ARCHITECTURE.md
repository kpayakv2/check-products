# 🏗️ API Architecture - ความสัมพันธ์ระหว่าง FastAPI, Supabase และ Edge Functions

**วันที่:** 2025-10-04 17:35

---

## 📊 ภาพรวมสถาปัตยกรรม

```
┌─────────────────────────────────────────────────────────────────┐
│                    Next.js Frontend (Port 3000)                 │
│                         React + TypeScript                       │
└───────────┬─────────────────────────────────┬───────────────────┘
            │                                 │
            │ Option A                        │ Option B
            │ (ใช้ FastAPI)                  │ (ใช้ Edge Functions)
            │                                 │
            ▼                                 ▼
┌─────────────────────────┐      ┌─────────────────────────────┐
│   FastAPI Backend       │      │   Supabase Edge Functions   │
│   (Port 8000)           │      │   (Serverless)              │
│                         │      │                             │
│ ✅ /api/embed          │      │ ✅ category-suggestions     │
│ ✅ /api/embed/batch    │      │ ✅ generate-embeddings      │
│ ✅ /api/classify/      │      │ ✅ hybrid-search            │
│    category            │      │ ✅ product-deduplication    │
│ ✅ /api/classify/batch │      │                             │
│                         │      │                             │
│ Dependencies:           │      │ Dependencies:               │
│ - CategoryClassifier    │      │ - OpenAI API (embeddings)   │
│ - SentenceTransformer   │      │ - Hugging Face API          │
│ - Supabase Client       │      │ - Deno Runtime              │
└───────────┬─────────────┘      └─────────────┬───────────────┘
            │                                  │
            └──────────────┬───────────────────┘
                           │
                           ▼
            ┌──────────────────────────────────┐
            │   Supabase PostgreSQL Database   │
            │   (Port 54321 - Local)           │
            │                                  │
            │   Tables:                        │
            │   - taxonomy_nodes (67 cats)     │
            │   - keyword_rules (45 rules)     │
            │   - synonym_lemmas (120)         │
            │   - products                     │
            │   - product_category_suggestions │
            │                                  │
            │   Extensions:                    │
            │   - pgvector (embeddings)        │
            └──────────────────────────────────┘
```

---

## 🔄 ความสัมพันธ์และการทำงาน

### **1. FastAPI Backend (api_server.py)**

#### **การเชื่อมต่อกับ Supabase:**

```python
# ใน api_server.py (บรรทัด 686-703)

def initialize_supabase():
    """Initialize Supabase client."""
    if app_state["supabase_client"] is None:
        supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL", "http://localhost:54321")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "...")
        
        client = create_client(supabase_url, supabase_key)
        app_state["supabase_client"] = client
        print(f"✅ Supabase connected: {supabase_url}")
    
    return app_state["supabase_client"]
```

#### **CategoryClassifier ใช้ Supabase:**

```python
# ใน CategoryClassifier class (บรรทัด 753-817)

class CategoryClassifier:
    def __init__(self, supabase_client: Client, embedding_model):
        self.supabase = supabase_client  # ใช้ Supabase client
        ...
    
    def load_taxonomy(self):
        """โหลดหมวดหมู่จาก Supabase Database"""
        response = self.supabase.table("taxonomy_nodes").select("*").execute()
        self.taxonomy_flat = response.data
        # ✅ อ่านข้อมูลจาก database
    
    def load_keyword_rules(self):
        """โหลดกฎคำหลักจาก Supabase Database"""
        response = self.supabase.table("keyword_rules")\
            .select("*")\
            .eq("is_active", True)\
            .execute()
        self.keyword_rules = response.data
        # ✅ อ่านข้อมูลจาก database
    
    def load_synonyms(self):
        """โหลดคำพ้องความหมายจาก Supabase Database"""
        response = self.supabase.table("synonym_lemmas")\
            .select("*, synonym_terms(*)")\
            .eq("is_verified", True)\
            .execute()
        # ✅ อ่านข้อมูลจาก database พร้อม JOIN
```

**สรุป:**
- ✅ FastAPI **อ่านข้อมูล** จาก Supabase Database
- ✅ ไม่ได้เรียกใช้ Edge Functions
- ✅ ใช้ SentenceTransformer (local model) สำหรับ embeddings
- ✅ ใช้ SERVICE_ROLE_KEY (full access, bypass RLS)

---

### **2. Supabase Edge Functions**

#### **Edge Functions ทำงานอิสระ:**

```typescript
// File: supabase/functions/category-suggestions/index.ts

serve(async (req) => {
  // 1. สร้าง Supabase client
  const supabase = createClient(
    Deno.env.get('SUPABASE_URL'),
    Deno.env.get('SUPABASE_ANON_KEY')
  )
  
  // 2. Query ข้อมูลจาก database
  const { data: categories } = await supabase
    .table('taxonomy_nodes')
    .select('*')
    .execute()
  
  // 3. ทำ keyword matching
  // 4. Return suggestions
})
```

**สรุป:**
- ✅ Edge Functions **อ่านข้อมูล** จาก Supabase Database เหมือนกัน
- ✅ ไม่ได้เรียกใช้ FastAPI
- ✅ ใช้ OpenAI API สำหรับ embeddings (ไม่ใช่ local model)
- ✅ ใช้ ANON_KEY (จำกัดสิทธิ์ตาม RLS)

---

## 🆚 เปรียบเทียบ FastAPI vs Edge Functions

| Feature | FastAPI Backend | Supabase Edge Functions |
|---------|-----------------|-------------------------|
| **Embeddings** | SentenceTransformer (384-dim, local) | OpenAI API (1536-dim, cloud) |
| **Algorithm** | Hybrid (Keyword 60% + Embedding 40%) | Keyword-based only |
| **Accuracy** | **72%** ✅ | ~65% |
| **Database Access** | Direct via Supabase Client | Direct via Supabase Client |
| **Authentication** | SERVICE_ROLE_KEY (full access) | ANON_KEY (limited) |
| **Deployment** | Need server/Docker | Serverless (auto-scale) |
| **Cost** | Free (local model) | Pay per request (OpenAI) |
| **Speed** | ~45ms | ~50-100ms (API latency) |
| **Dependencies** | Python + sentence-transformers | Deno + OpenAI API |

---

## 🔀 Data Flow Scenarios

### **Scenario 1: Import Products ด้วย FastAPI (แนะนำ)**

```
1. User uploads CSV → Next.js
2. Next.js API Route: /api/import/process
3. For each product:
   a. Clean & tokenize text
   b. Call FastAPI: POST /api/embed (get embedding)
   c. Call FastAPI: POST /api/classify/category (get suggestions)
   d. Save to Supabase: INSERT INTO products
   e. Save suggestions: INSERT INTO product_category_suggestions
4. Return results to frontend
```

**Code:**
```typescript
// taxonomy-app/app/api/import/process/route.ts

// Step 1: Generate embedding
const embedResponse = await fetch('http://localhost:8000/api/embed', {
  method: 'POST',
  body: JSON.stringify({ text: productName })
})
const { embedding } = await embedResponse.json()

// Step 2: Classify category
const classifyResponse = await fetch('http://localhost:8000/api/classify/category', {
  method: 'POST',
  body: JSON.stringify({
    product_name: productName,
    method: 'hybrid',
    top_k: 5
  })
})
const { top_suggestion, suggestions } = await classifyResponse.json()

// Step 3: Save to Supabase
const { data, error } = await supabase
  .from('products')
  .insert({
    name_th: productName,
    embedding: embedding,
    category_id: top_suggestion?.category_id,
    ...
  })
```

---

### **Scenario 2: Import Products ด้วย Edge Functions**

```
1. User uploads CSV → Next.js
2. Next.js API Route: /api/import/process
3. For each product:
   a. Clean & tokenize text
   b. Call Edge Function: generate-embeddings (OpenAI)
   c. Call Edge Function: category-suggestions (keyword)
   d. Save to Supabase: INSERT INTO products
4. Return results to frontend
```

**Code:**
```typescript
// taxonomy-app/app/api/import/process/route.ts

// Step 1: Generate embedding (OpenAI)
const { data: embeddingData } = await supabase.functions.invoke('generate-embeddings', {
  body: {
    texts: [productName],
    model: 'text-embedding-ada-002'
  }
})

// Step 2: Classify category (keyword-based)
const { data: categoryData } = await supabase.functions.invoke('category-suggestions', {
  body: {
    text: productName,
    options: { maxSuggestions: 5 }
  }
})

// Step 3: Save to Supabase
const { data, error } = await supabase
  .from('products')
  .insert({...})
```

---

### **Scenario 3: Hybrid Approach (ใช้ทั้งสอง)**

```
1. Use FastAPI for:
   ✅ Category Classification (Hybrid Algorithm - 72% accuracy)
   ✅ Embeddings (Free, local model)
   
2. Use Edge Functions for:
   ✅ Hybrid Search (vector + text search)
   ✅ Product Deduplication
   ✅ Batch Processing (serverless scaling)
```

---

## 🔌 การเชื่อมต่อ Database

### **FastAPI → Supabase:**

```python
# FastAPI ใช้ supabase-py library

from supabase import create_client, Client

# Connect
supabase: Client = create_client(url, key)

# Query
response = supabase.table("taxonomy_nodes").select("*").execute()
data = response.data

# Insert
supabase.table("products").insert({...}).execute()

# Update
supabase.table("products").update({...}).eq("id", product_id).execute()
```

### **Edge Functions → Supabase:**

```typescript
// Edge Functions ใช้ @supabase/supabase-js

import { createClient } from '@supabase/supabase-js'

// Connect
const supabase = createClient(url, key)

// Query
const { data, error } = await supabase
  .from('taxonomy_nodes')
  .select('*')

// Insert
const { data, error } = await supabase
  .from('products')
  .insert({...})
```

### **Next.js → Supabase:**

```typescript
// Next.js ใช้ @supabase/supabase-js เหมือนกัน

import { createClient } from '@supabase/supabase-js'

const supabase = createClient(url, anonKey)

// Same API as Edge Functions
```

**สรุป:** ทุก service ใช้ Supabase Client library เชื่อมต่อกับ PostgreSQL เดียวกัน

---

## 🔐 Authentication & Permissions

### **FastAPI:**
```python
# ใช้ SERVICE_ROLE_KEY
SUPABASE_SERVICE_ROLE_KEY=eyJ...

# = Full access, bypass RLS
# = สามารถอ่าน/เขียนทุก table
# = ไม่ต้องตรวจสอบ user authentication
```

### **Edge Functions:**
```typescript
// ใช้ ANON_KEY (public key)
SUPABASE_ANON_KEY=eyJ...

// = Limited access, respect RLS
// = ต้องมี Row Level Security policies
// = ตรวจสอบ user authentication ผ่าน Authorization header
```

### **Next.js Frontend:**
```typescript
// ใช้ ANON_KEY เหมือน Edge Functions
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...

// = Client-side access
// = RLS policies apply
```

---

## 📝 สรุปความสัมพันธ์

### **1. FastAPI ≠ Edge Functions**
- ✅ **แยกกันทำงาน** ไม่ได้เรียกใช้กัน
- ✅ **ทั้งคู่อ่านข้อมูลจาก Supabase Database เดียวกัน**
- ✅ **ใช้งานแยกกันตาม use case**

### **2. FastAPI ↔ Supabase Database**
```
FastAPI (Python)
  → Supabase Client (supabase-py)
    → PostgreSQL (Port 54321)
      → Tables: taxonomy_nodes, keyword_rules, etc.
```

### **3. Edge Functions ↔ Supabase Database**
```
Edge Functions (Deno)
  → Supabase Client (@supabase/supabase-js)
    → PostgreSQL (Port 54321)
      → Tables: taxonomy_nodes, products, etc.
```

### **4. Next.js เลือกเรียกใช้ได้ทั้งสอง:**

```typescript
// Option A: Call FastAPI
const response = await fetch('http://localhost:8000/api/classify/category', {
  method: 'POST',
  body: JSON.stringify({ product_name: "..." })
})

// Option B: Call Edge Function
const { data } = await supabase.functions.invoke('category-suggestions', {
  body: { text: "..." }
})

// Option C: Direct Database Access
const { data } = await supabase.from('products').select('*')
```

---

## 🎯 แนะนำการใช้งาน

### **ใช้ FastAPI สำหรับ:**
- ✅ Category Classification (Hybrid Algorithm - accuracy สูง)
- ✅ Embeddings generation (ฟรี, ไม่ต้อง API key)
- ✅ Complex AI processing
- ✅ Development & testing

### **ใช้ Edge Functions สำหรับ:**
- ✅ Simple operations (keyword matching)
- ✅ Serverless deployment
- ✅ Auto-scaling workloads
- ✅ Production (ไม่ต้อง manage server)

### **ใช้ Direct Supabase Client สำหรับ:**
- ✅ CRUD operations
- ✅ Real-time subscriptions
- ✅ Authentication & Authorization
- ✅ Simple queries

---

## 🔧 Environment Setup

```bash
# .env.local (สำหรับทุก service)

# Supabase (shared by all)
NEXT_PUBLIC_SUPABASE_URL=http://localhost:54321
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...          # For Next.js & Edge Functions
SUPABASE_SERVICE_ROLE_KEY=eyJ...              # For FastAPI

# FastAPI (optional if deploy separately)
FASTAPI_URL=http://localhost:8000

# OpenAI (optional for Edge Functions)
OPENAI_API_KEY=sk-...
```

---

## 🚀 สรุปสุดท้าย

```
┌─────────────────────────────────────────────────────┐
│ ทั้ง FastAPI และ Edge Functions:                   │
│                                                     │
│ 1. อ่านข้อมูลจาก Supabase Database เดียวกัน      │
│ 2. ไม่ได้เรียกใช้กัน (แยกกันทำงาน)                │
│ 3. Next.js เลือกเรียกใช้ตาม use case              │
│ 4. Database เป็นจุดกลางที่ทุกอย่างเชื่อมต่อ         │
└─────────────────────────────────────────────────────┘

FastAPI (Port 8000)          Edge Functions (Serverless)
     ↓                              ↓
     └──────────┬───────────────────┘
                ↓
    Supabase PostgreSQL (Port 54321)
           taxonomy_nodes
           keyword_rules
           products
           ...
```

**ไม่มีการเรียกใช้ซึ่งกันและกัน - แต่ใช้ database เดียวกัน!** ✅
