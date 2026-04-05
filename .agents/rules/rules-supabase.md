# ⚡ Supabase & Database Rules

## Context
ใช้เมื่อมีการสร้างหรือแก้ไขตารางฐานข้อมูล, เขียน Query ผ่าน Supabase Client, หรือสร้าง Edge Functions ที่เกี่ยวข้องกับ AI/Vector Search

## Standards
- **Client Usage**: ใช้ Supabase Client เสมอ ห้ามเขียน SQL Direct ใน Application Code
- **Type Safety**: ใช้ TypeScript Generics ในทุก Query
- **Row Level Security (RLS)**: ต้องเปิดใช้งาน RLS ในทุกตาราง และกำหนด Policy ให้ชัดเจน
- **Vector Search**: ใช้ค่า Cosine Distance (`<=>`) เป็นหลักสำหรับงาน Similarity

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

### ✅ Good: การทำ Semantic Search (SQL RPC)
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
