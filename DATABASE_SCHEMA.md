# 🗄️ Database Schema & Data Flow Documentation

**Thai Product Taxonomy Manager**  
**Date:** 2025-10-04  
**Database:** Supabase (PostgreSQL + pgvector)

---

## 📊 Database Tables Overview

### **14 Tables in System:**

```
┌─────────────────────────┐
│  Core Tables (6)        │
├─────────────────────────┤
│ 1. taxonomy_nodes       │ ← หมวดหมู่สินค้า (67 รายการ)
│ 2. products             │ ← สินค้า + embeddings
│ 3. imports              │ ← รอบการนำเข้า
│ 4. keyword_rules        │ ← กฎคำหลัก (25 rules)
│ 5. synonym_lemmas       │ ← คำที่มีความหมายเหมือนกัน
│ 6. synonym_terms        │ ← คำพ้องความหมาย
└─────────────────────────┘

┌─────────────────────────┐
│  Suggestion Tables (2)  │
├─────────────────────────┤
│ 7. product_category_    │ ← คำแนะนำหมวดหมู่ AI
│    suggestions          │
│ 8. similarity_matches   │ ← สินค้าที่คล้ายกัน
└─────────────────────────┘

┌─────────────────────────┐
│  Metadata Tables (3)    │
├─────────────────────────┤
│ 9. product_attributes   │ ← คุณสมบัติสินค้า
│ 10. review_history      │ ← ประวัติการตรวจสอบ
│ 11. human_feedback      │ ← Feedback จากผู้ใช้
└─────────────────────────┘

┌─────────────────────────┐
│  System Tables (3)      │
├─────────────────────────┤
│ 12. regex_rules         │ ← กฎ regex
│ 13. audit_logs          │ ← Log การทำงาน
│ 14. system_settings     │ ← การตั้งค่าระบบ
└─────────────────────────┘
```

---

## 🔄 Data Flow: Import Process

### **ภาพรวมการทำงาน:**

```
┌──────────────┐
│   USER       │
│ Upload CSV   │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│ STEP 1: Frontend (ProcessingStep.tsx)           │
│                                                  │
│ 1. Upload file to Supabase Storage              │
│ 2. Create import batch record                   │
│ 3. Call API: POST /api/import/process           │
└──────┬───────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│ STEP 2: Next.js API (route.ts)                  │
│                                                  │
│ Input:  CSV file                                 │
│ Process:                                         │
│   • Parse CSV → Extract product names           │
│   • Clean text → Remove special chars           │
│   • Tokenize → Split into keywords              │
│   • Extract attributes → Colors, sizes, units   │
│   • Generate embeddings → Call FastAPI          │
│   • Suggest category → Keyword matching         │
│ Output: Stream JSON (real-time updates)         │
└──────┬───────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│ STEP 3: AI Processing                           │
│                                                  │
│ For each product:                                │
│   1. Text Cleaning                               │
│   2. Tokenization                                │
│   3. Attribute Extraction                        │
│   4. Embedding Generation (384-dim vector)      │
│   5. Category Suggestion (keyword match)        │
└──────┬───────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│ STEP 4: Database Storage                        │
│                                                  │
│ Write to tables:                                 │
│   • products → Insert product data               │
│   • product_category_suggestions → AI prediction│
│   • product_attributes → Extracted features     │
│   • imports → Update batch status                │
└──────┬───────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│ STEP 5: User Review (Review Interface)          │
│                                                  │
│ User actions:                                    │
│   • Approve AI suggestion → Update category_id  │
│   • Reject & select manually → Store feedback   │
│   • Skip for later review                       │
└──────┬───────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│   DATABASE   │
│  (Updated)   │
└──────────────┘
```

---

## 🗃️ Detailed Table Schemas

### **1. taxonomy_nodes** - หมวดหมู่สินค้า

```sql
CREATE TABLE taxonomy_nodes (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  code TEXT,                    -- รหัสหมวดหมู่ เช่น "TOOL001"
  name_th TEXT NOT NULL,        -- ชื่อไทย เช่น "เครื่องมือ_ฮาร์ดแวร์"
  name_en TEXT,                 -- ชื่ออังกฤษ
  description TEXT,             -- คำอธิบาย
  parent_id UUID,               -- หมวดหมู่แม่ (NULL = root)
  level INTEGER DEFAULT 0,      -- ระดับ: 0=หลัก, 1=ย่อย, 2=ย่อยๆ
  sort_order INTEGER,           -- ลำดับการแสดงผล
  path TEXT,                    -- "/1/2/3" - Path for tree
  keywords TEXT[],              -- คำหลักสำหรับจับคู่ ["กรรไกร", "ตัด"]
  metadata JSONB,               -- ข้อมูลเพิ่มเติม
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
)

-- Index for performance
CREATE INDEX idx_taxonomy_parent ON taxonomy_nodes(parent_id);
CREATE INDEX idx_taxonomy_level ON taxonomy_nodes(level);
CREATE INDEX idx_taxonomy_keywords ON taxonomy_nodes USING GIN(keywords);
```

**Data Example:**
```json
{
  "id": "3114d46b-c1e8-4e3d-9a59-6d0ffd8d7a85",
  "code": "HOME_TOOL_001",
  "name_th": "กรรไกร/มีดคัตเตอร์",
  "name_en": "Scissors/Cutters",
  "parent_id": "abc-123-parent-uuid",
  "level": 1,
  "keywords": ["กรรไกร", "มีด", "คัตเตอร์", "ตัด"]
}
```

---

### **2. products** - สินค้า

```sql
CREATE TABLE products (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name_th TEXT NOT NULL,        -- ชื่อสินค้าไทย
  name_en TEXT,                 -- ชื่อสินค้าอังกฤษ
  description TEXT,             -- คำอธิบาย (cleaned text)
  category_id UUID,             -- FK → taxonomy_nodes.id
  brand TEXT,                   -- ยี่ห้อ
  model TEXT,                   -- รุ่น
  sku TEXT UNIQUE,              -- รหัสสินค้า
  price NUMERIC(10,2),          -- ราคา
  
  -- AI Fields
  embedding VECTOR(384),        -- pgvector: Sentence embeddings
  keywords TEXT[],              -- Extracted keywords/tokens
  confidence_score FLOAT,       -- AI confidence (0-1)
  
  -- Metadata
  metadata JSONB,               -- { units, attributes, etc. }
  status TEXT DEFAULT 'pending', -- 'pending', 'approved', 'rejected'
  
  -- Import tracking
  import_batch_id UUID,         -- FK → imports.id
  
  -- Audit
  reviewed_by UUID,
  reviewed_at TIMESTAMP,
  created_by UUID,
  updated_by UUID,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  
  FOREIGN KEY (category_id) REFERENCES taxonomy_nodes(id),
  FOREIGN KEY (import_batch_id) REFERENCES imports(id)
)

-- Indexes
CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_products_status ON products(status);
CREATE INDEX idx_products_import ON products(import_batch_id);
CREATE INDEX idx_products_embedding ON products USING ivfflat(embedding vector_cosine_ops);
```

**Data Example:**
```json
{
  "id": "prod-001",
  "name_th": "กล่องล็อค 560 หูหิ้ว W",
  "description": "กล่องล็อค 560 หูหิ้ว w",
  "category_id": "cat-001",
  "keywords": ["กล่อง", "ล็อค", "560", "หูหิ้ว"],
  "embedding": [0.234, -0.567, ...], // 384 dimensions
  "confidence_score": 0.72,
  "metadata": {
    "units": ["560"],
    "attributes": { "size": "medium" },
    "original_text": "กล่องล็อค 560 หูหิ้ว W"
  },
  "status": "pending",
  "import_batch_id": "batch-001"
}
```

---

### **3. imports** - รอบการนำเข้า

```sql
CREATE TABLE imports (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT NOT NULL,           -- "Product Import - 2025-10-04"
  description TEXT,             -- คำอธิบาย
  file_name TEXT,               -- "products.csv"
  file_size BIGINT,             -- ขนาดไฟล์ (bytes)
  file_type TEXT,               -- "text/csv"
  
  -- Progress tracking
  total_records INTEGER,        -- จำนวนทั้งหมด
  processed_records INTEGER,    -- ประมวลผลแล้ว
  success_records INTEGER,      -- สำเร็จ
  error_records INTEGER,        -- ล้มเหลว
  
  -- Status
  status TEXT DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
  error_details JSONB,          -- ข้อมูล errors
  metadata JSONB,               -- ข้อมูลเพิ่มเติม
  
  -- Timestamps
  created_by UUID,
  started_at TIMESTAMP,
  completed_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW()
)

CREATE INDEX idx_imports_status ON imports(status);
CREATE INDEX idx_imports_created_at ON imports(created_at DESC);
```

**Data Example:**
```json
{
  "id": "batch-001",
  "name": "Product Import - 04/10/2025 10:30:00",
  "file_name": "products_20251004.csv",
  "total_records": 296,
  "processed_records": 296,
  "success_records": 290,
  "error_records": 6,
  "status": "completed",
  "started_at": "2025-10-04T10:30:00Z",
  "completed_at": "2025-10-04T10:35:00Z"
}
```

---

### **4. product_category_suggestions** - คำแนะนำหมวดหมู่จาก AI

```sql
CREATE TABLE product_category_suggestions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  product_id UUID NOT NULL,     -- FK → products.id
  suggested_category_id UUID,   -- FK → taxonomy_nodes.id
  confidence_score FLOAT,       -- 0.0 - 1.0
  suggestion_method TEXT,       -- 'keyword_rule', 'embedding', 'hybrid'
  rule_id UUID,                 -- FK → keyword_rules.id (ถ้าใช้ keyword)
  
  -- Explanation
  metadata JSONB,               -- { explanation, matched_tokens, etc. }
  
  -- Review status
  is_accepted BOOLEAN,          -- true = approved, false = rejected
  reviewed_by UUID,
  reviewed_at TIMESTAMP,
  
  created_at TIMESTAMP DEFAULT NOW(),
  
  FOREIGN KEY (product_id) REFERENCES products(id),
  FOREIGN KEY (suggested_category_id) REFERENCES taxonomy_nodes(id)
)

CREATE INDEX idx_suggestions_product ON product_category_suggestions(product_id);
CREATE INDEX idx_suggestions_category ON product_category_suggestions(suggested_category_id);
CREATE INDEX idx_suggestions_accepted ON product_category_suggestions(is_accepted);
```

**Data Example:**
```json
{
  "id": "sugg-001",
  "product_id": "prod-001",
  "suggested_category_id": "cat-001",
  "confidence_score": 0.72,
  "suggestion_method": "hybrid",
  "metadata": {
    "explanation": "พบคำที่ตรงกัน: กล่อง (1 คำ)",
    "matched_tokens": ["กล่อง", "ล็อค"],
    "keyword_confidence": 0.9,
    "embedding_confidence": 0.52
  },
  "is_accepted": null, // รอการตรวจสอบ
  "created_at": "2025-10-04T10:30:15Z"
}
```

---

### **5. product_attributes** - คุณสมบัติสินค้า

```sql
CREATE TABLE product_attributes (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  product_id UUID NOT NULL,     -- FK → products.id
  attribute_name TEXT NOT NULL, -- "color", "size", "unit", "material"
  attribute_value TEXT,         -- "แดง", "L", "500ml", "พลาสติก"
  attribute_type TEXT,          -- "text", "number", "boolean"
  created_at TIMESTAMP DEFAULT NOW(),
  
  FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
)

CREATE INDEX idx_attributes_product ON product_attributes(product_id);
CREATE INDEX idx_attributes_name ON product_attributes(attribute_name);
```

**Data Example:**
```json
{
  "id": "attr-001",
  "product_id": "prod-001",
  "attribute_name": "size",
  "attribute_value": "560",
  "attribute_type": "number"
}
```

---

### **6. keyword_rules** - กฎคำหลักสำหรับจับคู่

```sql
CREATE TABLE keyword_rules (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  category_id UUID NOT NULL,    -- FK → taxonomy_nodes.id
  keywords TEXT[] NOT NULL,     -- คำหลักที่ต้องการจับ
  priority INTEGER DEFAULT 1,   -- ลำดับความสำคัญ (1-10)
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  
  FOREIGN KEY (category_id) REFERENCES taxonomy_nodes(id)
)

CREATE INDEX idx_keyword_rules_category ON keyword_rules(category_id);
CREATE INDEX idx_keyword_rules_keywords ON keyword_rules USING GIN(keywords);
```

**Data Example:**
```json
{
  "id": "rule-001",
  "category_id": "cat-001",
  "keywords": ["กล่อง", "ล็อค", "เก็บของ"],
  "priority": 8,
  "is_active": true
}
```

---

## 🔄 Data Relationships

### **Entity Relationship Diagram:**

```
┌─────────────────────┐
│  taxonomy_nodes     │ (67 categories)
│  ─────────────────  │
│  • id (PK)          │◄──┐
│  • parent_id (FK)   │   │
│  • name_th          │   │ Many-to-One
│  • keywords[]       │   │
└─────────────────────┘   │
         ▲                │
         │                │
         │ One-to-Many    │
         │                │
┌─────────────────────┐   │
│  products           │   │
│  ─────────────────  │   │
│  • id (PK)          │   │
│  • category_id (FK) ├───┘
│  • name_th          │◄──┐
│  • embedding[384]   │   │
│  • import_batch_id  │   │ One-to-Many
└──────┬──────────────┘   │
       │                  │
       │ One-to-Many      │
       │                  │
       ▼                  │
┌─────────────────────────┴─┐
│  product_category_        │
│  suggestions              │
│  ───────────────────────  │
│  • id (PK)                │
│  • product_id (FK)        │
│  • suggested_category_id  │
│  • confidence_score       │
│  • is_accepted            │
└───────────────────────────┘

┌─────────────────────┐
│  imports            │ (Import batches)
│  ─────────────────  │
│  • id (PK)          │◄──┐
│  • total_records    │   │
│  • success_records  │   │ Many-to-One
└─────────────────────┘   │
                          │
                          │
                    ┌─────┴────┐
                    │ products │
                    │ ──────── │
                    │ import_  │
                    │ batch_id │
                    └──────────┘

┌─────────────────────┐
│  keyword_rules      │ (Matching rules)
│  ─────────────────  │
│  • id (PK)          │
│  • category_id (FK) ├───► taxonomy_nodes
│  • keywords[]       │
│  • priority         │
└─────────────────────┘

┌─────────────────────┐
│  product_attributes │ (Extracted features)
│  ─────────────────  │
│  • id (PK)          │
│  • product_id (FK)  ├───► products
│  • attribute_name   │
│  • attribute_value  │
└─────────────────────┘
```

---

## 🎯 Query Examples

### **1. Get products with AI suggestions:**
```sql
SELECT 
  p.id,
  p.name_th,
  p.status,
  p.confidence_score,
  tn.name_th as suggested_category,
  pcs.confidence_score as ai_confidence,
  pcs.suggestion_method
FROM products p
LEFT JOIN product_category_suggestions pcs ON p.id = pcs.product_id
LEFT JOIN taxonomy_nodes tn ON pcs.suggested_category_id = tn.id
WHERE p.import_batch_id = 'batch-001'
ORDER BY pcs.confidence_score DESC;
```

### **2. Get import batch summary:**
```sql
SELECT 
  i.id,
  i.name,
  i.total_records,
  i.success_records,
  i.error_records,
  i.status,
  COUNT(p.id) as product_count,
  COUNT(CASE WHEN p.status = 'approved' THEN 1 END) as approved_count,
  COUNT(CASE WHEN p.status = 'pending' THEN 1 END) as pending_count
FROM imports i
LEFT JOIN products p ON i.id = p.import_batch_id
WHERE i.id = 'batch-001'
GROUP BY i.id;
```

### **3. Find similar products by embedding:**
```sql
SELECT 
  p2.name_th,
  1 - (p1.embedding <=> p2.embedding) as similarity
FROM products p1
CROSS JOIN products p2
WHERE p1.id = 'prod-001'
  AND p2.id != p1.id
ORDER BY p1.embedding <=> p2.embedding
LIMIT 10;
```

---

## 📦 Data Storage Sizes (Updated 2025-10-05)

```
Table                          Rows    Size (estimated)
────────────────────────────────────────────────────────
taxonomy_nodes                 67      ~50 KB
products                       11      ~500 KB (with 384-dim embeddings)
synonym_lemmas                 28      ~20 KB
synonym_terms                  97      ~50 KB
imports                        4       ~10 KB
keyword_rules                  25      ~25 KB
product_category_suggestions   0       ~0 KB (empty)
product_attributes             0       ~0 KB (empty)
similarity_matches             0       ~0 KB (empty)
review_history                 0       ~0 KB (empty)
────────────────────────────────────────────────────────
Total (active data)                    ~655 KB
```

**Note**: ข้อมูลจริงจากการทดสอบ End-to-End Workflow  
**Empty Tables**: ตารางที่มีโครงสร้างแต่ยังไม่มีข้อมูล

---

## 🎯 Summary

**ความสัมพันธ์หลัก:**
1. **taxonomy_nodes** ← หมวดหมู่ทั้งหมด (67 รายการ)
2. **products** ← สินค้าที่นำเข้า + embeddings + category_id (11 รายการ)
3. **imports** ← รอบการนำเข้า → ติดตาม progress (4 รอบ)
4. **keyword_rules** ← กฎคำหลักสำหรับ AI classification (25 rules)
5. **synonym_lemmas + synonym_terms** ← ระบบคำพ้องความหมาย (28 lemmas, 97 terms)

**Data Flow:**
```
CSV → API → AI Processing → Database → User Review → Approved Products
```
