# 🗄️ Database Schema & Data Flow Documentation

**Thai Product Taxonomy Manager**
**Date:** 2026-08-25 (แทนที่เวอร์ชัน 2025-10-04 ที่ล้าสมัย)
**Database:** Supabase (PostgreSQL + pgvector)
**แหล่งอ้างอิง:** `taxonomy-app/schema_export.sql` (export จริงจากฐานข้อมูล) + migrations ล่าสุดถึง `20260822000003` + โค้ดจริงใน `src/api/`, `src/core/`, `taxonomy-app/app/`

> **หมายเหตุ:** เอกสารนี้แทนที่เวอร์ชันเดิมที่นับตารางได้แค่ 14 และมีรายละเอียด FK/column บางส่วนไม่ตรงกับฐานข้อมูลจริง (ตรวจสอบซ้ำผ่าน `schema_export.sql` และโค้ด backend จริงแล้ว)

---

## 📊 Database Tables Overview

### **15 Tables ในระบบ:**

```
┌─────────────────────────────┐
│  Taxonomy & Rules (5)       │
├─────────────────────────────┤
│ 1. taxonomy_nodes           │ ← หมวดหมู่สินค้า (67 รายการ, self-referencing tree)
│ 2. keyword_rules            │ ← กฎคำหลัก (25 rules)
│ 3. regex_rules              │ ← กฎ regex (คู่ขนานกับ keyword_rules)
│ 4. synonym_lemmas           │ ← กลุ่มคำพ้องความหมาย (28 รายการ)
│ 5. synonym_terms            │ ← คำพ้องความหมายแต่ละคำ (97 รายการ)
└─────────────────────────────┘

┌─────────────────────────────┐
│  Products & Review (4)      │
├─────────────────────────────┤
│ 6. products                 │ ← สินค้า + embeddings (11 รายการ)
│ 7. product_attributes       │ ← คุณสมบัติสินค้า (ยังว่าง)
│ 8. product_category_        │ ← คำแนะนำหมวดหมู่ AI (ยังว่าง)
│    suggestions              │
│ 9. similarity_matches       │ ← ผลตรวจสินค้าซ้ำ (ยังว่าง)
└─────────────────────────────┘

┌─────────────────────────────┐
│  Users, Audit & History (3) │
├─────────────────────────────┤
│ 10. review_history          │ ← ประวัติการตรวจสอบ/อนุมัติ (ยังว่าง)
│ 11. human_feedback          │ ← Feedback จากผู้ใช้เรื่องความคล้าย
│ 12. audit_logs              │ ← Log การเปลี่ยนแปลงทุกตาราง (ยังว่าง)
└─────────────────────────────┘

┌─────────────────────────────┐
│  Import & ML (2)            │
├─────────────────────────────┤
│ 13. imports                 │ ← รอบการนำเข้าไฟล์
│ 14. ml_training_history     │ ← ประวัติการเทรนโมเดล (สร้างล่าสุด 2026-05-30)
└─────────────────────────────┘

┌─────────────────────────────┐
│  System (1)                 │
├─────────────────────────────┤
│ 15. system_settings         │ ← การตั้งค่าระบบ (โครงสร้างปนสองยุค — ดูหมายเหตุ)
└─────────────────────────────┘
```

**ตารางที่เอกสารเดิมไม่เคยกล่าวถึง:** `regex_rules`, `human_feedback`, `ml_training_history` (3 ตาราง)

---

## 🔄 Data Flow: Import Process (ยืนยันจากโค้ดจริง)

```
┌──────────────┐
│   USER       │
│ Upload CSV   │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│ STEP 0: Auth Gate (middleware.ts — เพิ่มใหม่ 22 ส.ค. 2026) │
│                                                    │
│ ทุก request ที่ไม่ใช่ GET ต้องมี session cookie      │
│ ได้จาก POST /api/unlock ด้วย shared secret ก่อน      │
└──────┬─────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│ STEP 1: Next.js API Route                        │
│   /api/import/process, /process-local,           │
│   /process-storage (แล้วแต่แหล่งไฟล์)              │
│                                                    │
│ • สร้างแถวใน imports (track progress)              │
│ • Parse CSV → ดึงชื่อสินค้า                        │
└──────┬───────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│ STEP 2: ประมวลผลทีละสินค้า                          │
│                                                    │
│ 1. ThaiTextProcessor: clean + tokenize            │
│ 2. Generate embedding (384-dim)                   │
│    → Edge Function generate-embeddings-local      │
│    → FastAPI /api/embed                           │
│    → โมเดล paraphrase-multilingual-MiniLM-L12-v2  │
│ 3. Hybrid classification (keyword 60% + embed 40%)│
│    → Edge Function hybrid-classification-local    │
│    → FastAPI /api/classify/category               │
│    → เทียบกับ taxonomy_nodes.embedding เท่านั้น    │
│      (ไม่ได้เทียบกับ products อื่น)                │
└──────┬───────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│ STEP 3: บันทึกลง Database                          │
│                                                    │
│ • products → insert พร้อม category_id, embedding  │
│ • product_category_suggestions → insert ตัวเลือกสำรอง│
│ • audit_logs → บันทึกอัตโนมัติผ่าน trigger         │
└──────┬───────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│ STEP 4: Dedup Check (แยก pipeline คนละเรื่องกับ classify)│
│                                                    │
│ เทียบ embedding สินค้าใหม่กับสินค้าที่ status='approved'│
│ → match_products_by_embedding() (SQL, เพิ่มใหม่)   │
│   หรือ internal_match.py /import-dedup (FastAPI)   │
│ → บันทึกคู่ที่คล้ายกันลง similarity_matches         │
└──────┬───────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│ STEP 5: User Review (ออกแบบไว้ แต่ยังไม่มีข้อมูลจริง)│
│                                                    │
│ ตามทฤษฎี: approve/reject → review_history          │
│ สถานะจริง: product_category_suggestions และ        │
│   review_history มี 0 แถวมาตลอด — ยังไม่ยืนยันว่า   │
│   UI เชื่อมเข้า flow นี้จริงหรือยัง                 │
└────────────────────────────────────────────────────┘
```

**สิ่งสำคัญที่ต่างจากเอกสารเดิม:** classification เทียบกับ `taxonomy_nodes` (หมวดหมู่) เท่านั้น ส่วนการเทียบกับสินค้าอื่น (`products` ↔ `products`) เป็นคนละ pipeline ที่ทำเพื่อ **หาสินค้าซ้ำ** ไม่ใช่เพื่อจัดหมวดหมู่

---

## 🗃️ Detailed Table Schemas (ตรงกับ schema_export.sql)

### **1. taxonomy_nodes** — หมวดหมู่สินค้า

```sql
CREATE TABLE taxonomy_nodes (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  code TEXT UNIQUE,
  name_th TEXT NOT NULL,
  name_en TEXT,
  description TEXT,
  parent_id UUID REFERENCES taxonomy_nodes(id) ON DELETE CASCADE,
  level INTEGER DEFAULT 0,
  sort_order INTEGER,
  path TEXT,
  keywords TEXT[],
  metadata JSONB,
  embedding VECTOR(384),        -- ✅ ถูกต้องตามสเปกมาตั้งแต่แรก
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

RLS: มี policy กำกับ (SELECT เปิดสาธารณะ)

---

### **2. products** — สินค้า

```sql
CREATE TABLE products (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name_th TEXT NOT NULL,
  name_en TEXT,
  description TEXT,
  category_id UUID REFERENCES taxonomy_nodes(id),   -- ON DELETE ไม่ระบุ = RESTRICT
  brand TEXT,
  model TEXT,
  sku TEXT UNIQUE,
  price NUMERIC(10,2),
  embedding VECTOR(384),        -- ✅ แก้จาก 768 → 384 แล้ว (migration 20260822000000)
  keywords TEXT[],
  confidence_score FLOAT,
  metadata JSONB,
  status TEXT DEFAULT 'pending',   -- 'pending' | 'approved' | 'rejected'
  import_batch_id UUID,         -- ⚠️ ไม่มี FK constraint จริงไปยัง imports.id (ยังไม่แก้)
  reviewed_by UUID,
  reviewed_at TIMESTAMP,
  created_by UUID,
  updated_by UUID,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);
```

**ประวัติสำคัญ:** คอลัมน์ `embedding` เคยเป็น `vector(768)` มาก่อน (schema drift จากที่ไหนไม่ทราบแน่ชัด) ขัดกับ `taxonomy_nodes.embedding` ที่เป็น `vector(384)` ตลอด ทำให้ `internal_match.py` (dedup endpoint) crash เมื่อมีคน import ผ่าน endpoint นั้น — **แก้แล้วเมื่อ 22 ส.ค. 2026** ผ่าน `migrations/20260822000000_fix_embedding_dimension.sql` พร้อมล็อก signature ฟังก์ชัน classification ให้รับแค่ `vector(384)` กัน regression

RLS: มี policy กำกับ (`SELECT USING (true)` — อ่านสาธารณะ)

---

### **3. imports** — รอบการนำเข้า

```sql
CREATE TABLE imports (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT NOT NULL,
  description TEXT,
  file_name TEXT,
  file_size BIGINT,
  file_type TEXT,
  total_records INTEGER,
  processed_records INTEGER,
  success_records INTEGER,
  error_records INTEGER,
  status TEXT DEFAULT 'pending',
  error_details JSONB,
  metadata JSONB,
  created_by UUID,
  started_at TIMESTAMP,
  completed_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT NOW()
);
```

⚠️ **ไม่มี FK ขาเข้าจากตารางอื่นจริง** — `products.import_batch_id` เป็นแค่ UUID เปล่า ไม่ใช่ FK ที่บังคับระดับฐานข้อมูล (ยังเป็น known issue ที่ยังไม่แก้)

RLS: เปิดแต่ **ไม่มี policy** — ใช้งานได้เพราะ backend พึ่ง service-role key

---

### **4. keyword_rules** / **5. regex_rules** — กฎการจัดหมวดหมู่

```sql
CREATE TABLE keyword_rules (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  code TEXT UNIQUE,
  name TEXT,
  category_id UUID REFERENCES taxonomy_nodes(id) ON DELETE CASCADE,
  keywords TEXT[] NOT NULL,
  priority INTEGER DEFAULT 5,
  confidence_score FLOAT,
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE regex_rules (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  category_id UUID REFERENCES taxonomy_nodes(id),   -- ON DELETE ไม่ระบุ = RESTRICT (ต่างจาก keyword_rules ที่ CASCADE)
  pattern TEXT,
  created_by UUID REFERENCES auth.users(id),
  updated_by UUID REFERENCES auth.users(id),
  ...
);
```

⚠️ **`ON DELETE` ไม่สม่ำเสมอกัน:** `keyword_rules.category_id` ใช้ CASCADE แต่ `regex_rules.category_id` ไม่ระบุ (RESTRICT) — ยังไม่มีการทบทวนว่าตั้งใจหรือไม่

RLS: ทั้งคู่เปิดแต่ **ไม่มี policy**

---

### **6. synonym_lemmas** / **7. synonym_terms** — ระบบคำพ้องความหมาย

```sql
CREATE TABLE synonym_lemmas (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  code TEXT UNIQUE,
  name_th TEXT,
  category_id UUID REFERENCES taxonomy_nodes(id) ON DELETE SET NULL,
  ...
);

CREATE TABLE synonym_terms (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  lemma_id UUID REFERENCES synonym_lemmas(id) ON DELETE CASCADE,
  term TEXT NOT NULL,
  is_primary BOOLEAN,
  confidence_score FLOAT,
  UNIQUE(lemma_id, term)
);
```

RLS: ทั้งคู่มี policy กำกับ

---

### **8. product_category_suggestions**

```sql
CREATE TABLE product_category_suggestions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  product_id UUID REFERENCES products(id) ON DELETE CASCADE,
  suggested_category_id UUID REFERENCES taxonomy_nodes(id) ON DELETE CASCADE,
  rule_id UUID REFERENCES keyword_rules(id) ON DELETE SET NULL,
  confidence FLOAT,
  method TEXT,
  matched_keyword TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);
```

**สถานะจริง:** 0 แถวมาตลอด — โครงสร้างพร้อม แต่ยังไม่ยืนยันว่า UI เชื่อมเข้ามาใช้จริง

RLS: เปิดแต่ **ไม่มี policy**

---

### **9. product_attributes**

```sql
CREATE TABLE product_attributes (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  product_id UUID REFERENCES products(id) ON DELETE CASCADE,
  attribute_name TEXT NOT NULL,
  attribute_value TEXT,
  attribute_type TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);
```

RLS: เปิดแต่ **ไม่มี policy** · สถานะจริง: 0 แถว

---

### **10. similarity_matches** — ผลตรวจสินค้าซ้ำ

```sql
CREATE TABLE similarity_matches (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  product_a_id UUID REFERENCES products(id) ON DELETE CASCADE,
  product_b_id UUID REFERENCES products(id) ON DELETE CASCADE,
  similarity_score FLOAT NOT NULL,
  match_type TEXT,
  is_duplicate BOOLEAN DEFAULT false,
  reviewed BOOLEAN,
  UNIQUE(product_a_id, product_b_id)
);
```

เติมข้อมูลผ่าน `match_products_by_embedding()` (SQL function ใหม่) หรือ `internal_match.py` ฝั่ง FastAPI — ใช้งานได้จริงแล้วหลังแก้ embedding dimension

RLS: เปิดแต่ **ไม่มี policy** · สถานะจริง: 0 แถว (ยังไม่เคยรัน dedup จริง)

---

### **11. review_history**

```sql
CREATE TABLE review_history (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  product_id UUID REFERENCES products(id) ON DELETE CASCADE,
  old_category_id UUID REFERENCES taxonomy_nodes(id),
  new_category_id UUID REFERENCES taxonomy_nodes(id),
  reviewed_by UUID,
  reviewed_at TIMESTAMP DEFAULT NOW()
);
```

RLS: เปิดแต่ **ไม่มี policy** · สถานะจริง: 0 แถว

---

### **12. human_feedback**

```sql
CREATE TABLE human_feedback (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  reviewer_id UUID REFERENCES auth.users(id),
  human_decision TEXT CHECK (human_decision IN ('similar','different','duplicate','uncertain')),
  similarity_score FLOAT,
  -- ⚠️ ไม่มีคอลัมน์ FK ไปยัง products หรือ similarity_matches เลย
  ...
);
```

⚠️ **ช่องโหว่เชิง schema:** constraint บ่งชัดว่าออกแบบมาเพื่อตัดสินคู่สินค้าที่กำลังเทียบกัน แต่ไม่มีทางเชื่อมกลับไปยัง `similarity_matches` หรือ `products` ได้เลย — ยังไม่แก้

RLS: มี policy กำกับ

---

### **13. audit_logs**

```sql
CREATE TABLE audit_logs (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  table_name TEXT,
  record_id TEXT,
  action TEXT,   -- INSERT | UPDATE | DELETE
  old_values JSONB,
  new_values JSONB,
  user_id UUID REFERENCES auth.users(id),
  created_at TIMESTAMP DEFAULT NOW()
);
```

เติมข้อมูลอัตโนมัติผ่าน `audit_trigger_function()` · RLS: มี policy กำกับ · สถานะจริง: 0 แถว (ยังไม่มี event เกิดขึ้น)

---

### **14. ml_training_history**

ตารางใหม่สุด (สร้าง 2026-05-30) เก็บ accuracy/feature importance ของแต่ละรอบเทรนโมเดลเป็น JSONB **ไม่มี FK เชื่อมกับตารางไหนเลย** — โดดเดี่ยวสนิทในเชิงโครงสร้าง

RLS: มี policy กำกับ

---

### **15. system_settings**

โครงสร้างปนกันสองยุค: คอลัมน์ JSONB เดิม (`search`, `processing`, `ai`, `ui`) อยู่ร่วมกับคอลัมน์ key-value ใหม่ (`setting_key` UNIQUE, `setting_value`) — ยังไม่ได้ migrate ให้เหลือแบบเดียว มี FK `updated_by → auth.users.id`

---

## 🔒 Row Level Security — สรุปสถานะ

**มี policy กำกับ (8 ตาราง):** `audit_logs`, `human_feedback`, `ml_training_history`, `products`, `synonym_lemmas`, `synonym_terms`, `taxonomy_nodes`

**เปิด RLS แต่ไม่มี policy เลย (7 ตาราง — ยังไม่แก้):** `imports`, `keyword_rules`, `product_attributes`, `product_category_suggestions`, `regex_rules`, `review_history`, `similarity_matches`

ตารางกลุ่มหลังใช้งานได้ทุกวันนี้เพราะ backend ยิงผ่าน service-role key ที่ bypass RLS ทั้งหมด

---

## 🛡️ Security Fixes ที่ทำไปแล้ว (22-24 ส.ค. 2026)

| ปัญหาเดิม | การแก้ |
|---|---|
| `exec_sql()` function รัน SQL ใดก็ได้แบบไม่ auth (`SECURITY DEFINER`) | ลบฟังก์ชันและ Edge Function ทิ้งทั้งหมด (`migrations/20260822000003_drop_exec_sql_function.sql`) |
| Edge Function `hybrid-search` ไม่มี auth + ใช้ OpenAI embedding ผิดมิติ (1536 vs 768) | ลบ Edge Function ทิ้งทั้งหมด |
| API routes ใน `taxonomy-app/app/api/` ไม่มีชั้น auth เลย | เพิ่ม `middleware.ts` กันทุก request ที่ไม่ใช่ GET ด้วย session cookie (shared secret ปลดล็อกผ่าน `/api/unlock`) |

---

## 📦 Data Storage (ข้อมูลล่าสุดที่ยืนยันได้ — ณ 2025-10-05, บางตารางยังไม่เช็คซ้ำ)

```
Table                          Rows    หมายเหตุ
──────────────────────────────────────────────────────────
taxonomy_nodes                 67      
products                       11      embedding แก้เป็น 384-dim แล้ว
synonym_lemmas                 28      
synonym_terms                  97      
keyword_rules                  25      
product_category_suggestions   0       ยังไม่เคยใช้งานจริง
product_attributes             0       
similarity_matches             0       dedup ยังไม่เคยรันจริง
review_history                 0       ยังไม่มีใคร approve ผ่านระบบ
audit_logs                     0       ยังไม่มี event
regex_rules                    ไม่ทราบ  ยังไม่เคยนับ
imports                        ไม่ทราบ  ยังไม่เคยนับ (เดิม 4 รอบ ณ 2025-10-05)
ml_training_history             ไม่ทราบ  
human_feedback                 ไม่ทราบ  
system_settings                ~1      
```

---

## 🎯 Summary

**ความสัมพันธ์หลัก (verified):**
1. `taxonomy_nodes` — ศูนย์กลาง self-referencing tree, ถูกอ้างอิงจาก 6 ตาราง
2. `products` — ศูนย์กลางที่สอง, ถูกอ้างอิงจาก 4 ตาราง (attributes, suggestions, review_history, similarity_matches)
3. `synonym_lemmas → synonym_terms` — ห่วงโซ่ 2 ชั้น
4. `auth.users` — ถูกอ้างอิงจาก 4 ตาราง (audit_logs, human_feedback, regex_rules, system_settings)
5. `imports`, `ml_training_history` — โดดเดี่ยว ไม่มี FK เชื่อมกับระบบหลัก

**Data Flow:**
```
CSV → Auth Gate (middleware.ts) → Thai Text Processing → Embedding (384-dim)
    → Hybrid Classification (vs taxonomy_nodes) → บันทึก products
    → Dedup Check (vs products ที่ approved แล้ว) → similarity_matches
    → [ควรมี] User Review → review_history (ยังไม่ยืนยันว่าทำงานจริง)
```

**งานที่ยังเหลือ:** ดู `PRD_database_schema_fixes.md` สำหรับรายการปัญหาที่ยังไม่แก้พร้อม acceptance criteria (RLS 7 ตาราง, FK เชื่อม feedback loop, FK `import_batch_id`, ตรวจสอบ human review workflow)
