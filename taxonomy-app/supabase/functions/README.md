# Supabase Edge Functions

This folder contains Supabase Edge Functions for the taxonomy application.

## Functions

### 1. category-suggestions
ฟังก์ชันสำหรับแนะนำหมวดหมู่สินค้าตามข้อความที่ป้อนเข้ามา
- **Endpoint**: `/functions/v1/category-suggestions`
- **Method**: POST
- **Input**: 
  ```json
  {
    "text": "iPhone 15 Pro Max",
    "context": {
      "category": "electronics",
      "brand": "Apple"
    },
    "options": {
      "maxSuggestions": 5,
      "minConfidence": 0.3,
      "includeExplanation": true
    }
  }
  ```

### 2. generate-embeddings
ฟังก์ชันสำหรับสร้าง embeddings จากข้อความ
- **Endpoint**: `/functions/v1/generate-embeddings`
- **Method**: POST
- **Input**:
  ```json
  {
    "texts": ["iPhone 15", "Samsung Galaxy"],
    "model": "text-embedding-ada-002"
  }
  ```

### 3. hybrid-search
ฟังก์ชันสำหรับค้นหาสินค้าแบบ hybrid (vector + text search)
- **Endpoint**: `/functions/v1/hybrid-search`
- **Method**: POST
- **Input**:
  ```json
  {
    "query": "smartphone Apple",
    "type": "hybrid",
    "filters": {
      "categories": ["electronics"],
      "priceRange": [10000, 50000]
    },
    "limit": 20
  }
  ```

### 4. exec-sql
ฟังก์ชันสำหรับรัน SQL queries โดยตรง (ใช้สำหรับ admin เท่านั้น)
- **Endpoint**: `/functions/v1/exec-sql`
- **Method**: POST
- **Input**:
  ```json
  {
    "query": "SELECT * FROM taxonomy_nodes WHERE is_active = true",
    "params": []
  }
  ```

## การ Deploy

```bash
# Deploy ทุกฟังก์ชัน
supabase functions deploy

# Deploy ฟังก์ชันเฉพาะ
supabase functions deploy category-suggestions

# ทดสอบ local
supabase functions serve
```

## Environment Variables ที่ต้องตั้ง

- `SUPABASE_URL`: URL ของ Supabase project
- `SUPABASE_ANON_KEY`: Anonymous key ของ Supabase
- `SUPABASE_SERVICE_ROLE_KEY`: Service role key (สำหรับ exec-sql)
- `OPENAI_API_KEY`: API key ของ OpenAI (สำหรับ embeddings)
- `HUGGINGFACE_API_KEY`: API key ของ Hugging Face (optional)

## ปัญหาที่แก้ไขแล้ว

1. **Import paths**: แก้ไข import map ให้ใช้งานได้ถูกต้อง
2. **Error handling**: เพิ่มการจัดการ error ที่ดีขึ้น
3. **Null checks**: เพิ่มการตรวจสอบ null/undefined
4. **API key validation**: เพิ่มการตรวจสอบ API keys ก่อนใช้งาน
5. **Performance**: ปรับปรุงการคำนวณ similarity และ filtering
6. **Type safety**: เพิ่ม type definitions และ error handling

## การใช้งาน

1. Deploy functions ไปยัง Supabase
2. ตั้งค่า environment variables
3. เรียกใช้ผ่าน HTTP requests หรือ Supabase client
4. ตรวจสอบ logs ใน Supabase dashboard หากมีปัญหา
