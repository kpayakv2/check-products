---
name: rules-supabase
description: |
  กฎมาตรฐานสำหรับการทำงานกับ Supabase และฐานข้อมูล PostgreSQL
  ครอบคลุม Client Usage, Type Safety, RLS และ Vector Search

triggers:
  - สร้างหรือแก้ไขตารางใน Supabase
  - เขียน Supabase Query หรือ RPC Function
  - สร้างหรือแก้ไข Edge Functions
  - งานที่เกี่ยวกับ pgvector หรือ Embedding
  - Migration หรือ Schema changes
---

# ⚡ Supabase & Database Rules

## Context
ใช้เมื่อมีการสร้างหรือแก้ไขตารางฐานข้อมูล, เขียน Query ผ่าน Supabase Client, หรือสร้าง Edge Functions ที่เกี่ยวข้องกับ AI/Vector Search

## Port Configuration (Windows Local Dev)
- **Supabase API Gateway:** พอร์ต `54331` (เนื่องจาก `54321` ถูก Windows Reserve)
- **Database Port:** พอร์ต `54325`
- **Connection:** ใช้ `127.0.0.1` แทน `localhost` ใน Python/FastAPI เสมอ (Win32 quirk)

## Standards
- **Client Usage**: ใช้ Supabase Client เสมอ ห้ามเขียน SQL Direct ใน Application Code
- **Type Safety**: ใช้ TypeScript Generics ในทุก Query
- **Row Level Security (RLS)**: ต้องเปิดใช้งาน RLS ในทุกตาราง และกำหนด Policy ให้ชัดเจน
- **Vector Search**: ใช้ค่า Cosine Distance (`<=>`) เป็นหลักสำหรับงาน Similarity
- **Embedding Dimension**: **384 dims** (`paraphrase-multilingual-MiniLM-L12-v2`) เท่านั้น

## 🔭 Schema-Aware Development (Socraticode)

ก่อนเพิ่มหรือแก้ไข Column/Table ใหม่ ต้องทำขั้นตอนนี้:

1. **ตรวจสอบ schema artifacts ที่มีอยู่:**
   ```
   codebase_context { projectPath: "d:\\product_checker\\check-products" }
   ```
   → ดูว่า `DATABASE_SCHEMA.md` เป็นเวอร์ชันล่าสุดหรือยัง

2. **ค้นหาว่า Column/Table ที่จะเพิ่มมีการใช้ที่ไหนบ้าง:**
   ```
   codebase_search { query: "ชื่อตารางหรือ column ที่จะแก้" }
   ```

3. **วิเคราะห์ blast radius ของ migration:**
   ```
   codebase_impact { target: "ชื่อไฟล์ migration" }
   ```

## Examples


### ✅ Good: การ Query พร้อมกำหนด Type
```typescript
// ใช้ Generics เพื่อความถูกต้องของข้อมูล
const { data, error } = await supabase
  .from<Product>('products')
  .select('id, name')
  .eq('id', 123);
```

### ❌ Bad: การ Query แบบไม่ระบุ Type
```typescript
// ไม่รู้ว่า data ที่ได้มามีหน้าตาเป็นอย่างไร (Risk)
const { data } = await supabase
  .from('products')
  .select();
```

### ✅ Good: การทำ Semantic Search (SQL RPC) — 384 dim
```sql
-- สร้าง Function สำหรับคำนวณความคล้าย
CREATE OR REPLACE FUNCTION match_products (
  query_embedding vector(384),
  match_threshold float,
  match_count int
)
RETURNS TABLE (id uuid, name text, similarity float)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT id, name, 1 - (embedding <=> query_embedding) AS similarity
  FROM products
  WHERE 1 - (embedding <=> query_embedding) > match_threshold
  ORDER BY embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
```

### ❌ Bad: การดึงข้อมูล Vector ทั้งหมดมาคำนวณใน Application
```typescript
// สิ้นเปลือง Memory และ Bandwidth อย่างมาก
const { data: all_vectors } = await supabase.from('products').select('embedding');
// แล้วมาวน loop คำนวณใน JS (ห้ามทำเด็ดขาด)
```
