# 🚀 Deployment Guide: Local Model Integration

**ใช้ Local Model แบบเดียวกับ FastAPI (FREE!)**

Model: **paraphrase-multilingual-MiniLM-L12-v2** (384-dim)  
Algorithm: **Hybrid (Keyword 60% + Embedding 40%)** - 72% accuracy  
Cost: **FREE!** (ไม่ต้องเสีย OpenAI API)

---

## 📋 สิ่งที่สร้างแล้ว

### **1. Edge Functions**
```
✅ supabase/functions/generate-embeddings-local/
   → สร้าง embeddings ด้วย local model

✅ supabase/functions/hybrid-classification-local/
   → จัดหมวดหมู่ด้วย Hybrid algorithm
```

### **2. Database Functions**
```
✅ migrations/20250105010000_local_model_functions.sql
   → match_categories_by_embedding()
   → hybrid_category_classification()
   → batch_category_classification()
```

### **3. Scripts**
```
✅ scripts/generate-category-embeddings.ts
   → สร้าง embeddings สำหรับ taxonomy_nodes
```

### **4. API Routes**
```
✅ app/api/import/process-local/route.ts
   → Import products ด้วย local model
```

---

## 🔧 Setup Instructions

### **Step 1: Apply Database Migration**

```bash
cd taxonomy-app

# Apply migration
npx supabase migration up

# หรือ reset database
npx supabase db reset
```

**ตรวจสอบ:**
```sql
-- Check functions exist
SELECT routine_name 
FROM information_schema.routines 
WHERE routine_schema = 'public' 
  AND routine_name LIKE '%category%';

-- Should see:
-- - match_categories_by_embedding
-- - hybrid_category_classification
-- - batch_category_classification
-- - generate_category_embedding_text
```

---

### **Step 2: Start FastAPI**

FastAPI ต้องรันเพื่อให้ local model พร้อมใช้:

```bash
cd d:/product_checker/check-products

# Start FastAPI
python api_server.py
```

**ตรวจสอบ:**
```bash
curl http://localhost:8000/api/v1/health

# Should return:
# {"status":"healthy","version":"5.0.0",...}
```

---

### **Step 3: Deploy Edge Functions**

```bash
cd taxonomy-app

# Deploy generate-embeddings-local
npx supabase functions deploy generate-embeddings-local

# Deploy hybrid-classification-local
npx supabase functions deploy hybrid-classification-local

# Or deploy all
npx supabase functions deploy
```

**Set Environment Variables:**
```bash
# ตั้งค่า FASTAPI_URL สำหรับ Edge Functions
npx supabase secrets set FASTAPI_URL=http://host.docker.internal:8000

# Note: ใช้ host.docker.internal สำหรับ Docker
# ใน production ให้ใส่ public URL ของ FastAPI
```

**ตรวจสอบ:**
```bash
# Test generate-embeddings-local
curl -X POST http://localhost:54321/functions/v1/generate-embeddings-local \
  -H "Authorization: Bearer ${SUPABASE_ANON_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"texts":["กล่องล็อค 560 มล"],"model":"sentence-transformer"}'

# Should return:
# {"embeddings":[[...]],"dimension":384,"backend":"FastAPI",...}
```

---

### **Step 4: Generate Category Embeddings**

สร้าง embeddings สำหรับ taxonomy_nodes:

```bash
cd taxonomy-app

# Install dependencies
npm install tsx @supabase/supabase-js dotenv

# Run script
npx tsx scripts/generate-category-embeddings.ts
```

**Output:**
```
🚀 Starting Category Embedding Generation
============================================================
📡 Supabase: http://localhost:54321
🤖 FastAPI: http://localhost:8000
📊 Model: paraphrase-multilingual-MiniLM-L12-v2 (384-dim)
============================================================

1️⃣  Checking FastAPI connection...
   ✅ FastAPI is running (v5.0.0)

2️⃣  Loading taxonomy categories...
   ✅ Loaded 67 categories

   📊 Summary:
      - Has embeddings: 0
      - Needs embeddings: 67

3️⃣  Ready to generate 67 embeddings
   ⏱️  Estimated time: 34 seconds

4️⃣  Generating embeddings...

[1/67] อาหารและเครื่องดื่ม
  📝 Text: "อาหารและเครื่องดื่ม..."
  🎯 Embedding: 384 dimensions
  ✅ Saved to database

...

============================================================
✅ Embedding Generation Complete!
============================================================
📊 Results:
   - Success: 67
   - Errors: 0
   - Total: 67

🎉 Category embeddings are ready!
💡 You can now use hybrid-classification-local Edge Function
```

---

### **Step 5: Test Hybrid Classification**

```bash
# Test classification
curl -X POST http://localhost:54321/functions/v1/hybrid-classification-local \
  -H "Authorization: Bearer ${SUPABASE_ANON_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "กล่องล็อค 560 มล",
    "method": "hybrid",
    "top_k": 3
  }'
```

**Expected Response:**
```json
{
  "product_name": "กล่องล็อค 560 มล",
  "suggestions": [
    {
      "category_id": "abc-123",
      "category_name": "กล่อง/ที่เก็บของ",
      "confidence": 0.72,
      "method": "keyword+embedding"
    }
  ],
  "top_suggestion": {
    "category_id": "abc-123",
    "category_name": "กล่อง/ที่เก็บของ",
    "confidence": 0.72
  },
  "processing_time": 45,
  "model": "local (paraphrase-multilingual-MiniLM-L12-v2)",
  "algorithm": "Keyword 60% + Embedding 40%",
  "cost": 0
}
```

---

### **Step 6: Update Next.js Import Route**

```typescript
// File: app/api/import/process/route.ts
// เปลี่ยนจาก:
import { POST as OriginalPOST } from './route'

// เป็น:
export { POST } from './process-local/route'

// หรือเปลี่ยนชื่อไฟล์:
// mv app/api/import/process/route.ts app/api/import/process/route.old.ts
// mv app/api/import/process-local/route.ts app/api/import/process/route.ts
```

---

### **Step 7: Test Import Workflow**

```bash
cd taxonomy-app

# Start Next.js
npm run dev
```

**Test:**
1. เปิด http://localhost:3000/import
2. Upload CSV file with products
3. ตรวจสอบ console logs:
   ```
   📦 Processing 10 products with local model...
     ✅ [1/10] กล่องล็อค 560 มล → กล่อง/ที่เก็บของ (0.72)
     ✅ [2/10] ถังน้ำ 1000 ลิตร → ที่เก็บน้ำ (0.68)
     ...
   ```
4. ตรวจสอบ database:
   ```sql
   SELECT name_th, category_id, confidence_score, metadata 
   FROM products 
   ORDER BY created_at DESC 
   LIMIT 10;
   ```

---

## 📊 Architecture

```
User Upload CSV
      ↓
Next.js (3000)
      ↓
API Route: /api/import/process-local
      ↓
      ├─→ Edge Function: generate-embeddings-local
      │      ↓
      │   FastAPI (8000) - Local Model
      │      ↓
      │   Return 384-dim embedding
      │
      ├─→ Edge Function: hybrid-classification-local
      │      ↓
      │   Database Function: hybrid_category_classification()
      │      ├─ Keyword matching (60%)
      │      └─ Embedding matching (40% via pgvector)
      │      ↓
      │   Return category suggestions
      │
      └─→ Save to Supabase Database
             ├─ products table
             └─ product_category_suggestions table
```

---

## ✅ Verification Checklist

### **Database:**
- [ ] Migration applied successfully
- [ ] Functions exist and executable
- [ ] Taxonomy categories have embeddings
- [ ] Indexes created (ivfflat)

### **FastAPI:**
- [ ] Running on port 8000
- [ ] Health check passes
- [ ] /api/embed endpoint works
- [ ] /api/embed/batch endpoint works

### **Edge Functions:**
- [ ] generate-embeddings-local deployed
- [ ] hybrid-classification-local deployed
- [ ] FASTAPI_URL environment variable set
- [ ] Test calls return valid responses

### **Next.js:**
- [ ] Import route updated
- [ ] CSV upload works
- [ ] Products saved with embeddings
- [ ] Category suggestions saved

---

## 🎯 Performance Comparison

| Feature | OpenAI API | Local Model |
|---------|-----------|-------------|
| **Model** | text-embedding-ada-002 | paraphrase-multilingual-MiniLM-L12-v2 |
| **Dimension** | 1536 | 384 |
| **Speed** | ~50-100ms | ~40-50ms ✅ |
| **Cost** | $0.0001/1K tokens | FREE! ✅ |
| **Accuracy** | High | 72% ✅ |
| **Scalability** | API limits | Server capacity |

---

## 🔧 Troubleshooting

### **Edge Function ไม่สามารถเชื่อมต่อ FastAPI:**

```bash
# ตรวจสอบ FastAPI running
curl http://localhost:8000/api/v1/health

# ตรวจสอบ Edge Function สามารถเข้าถึง FastAPI
# ใน Docker ต้องใช้: http://host.docker.internal:8000

# Set environment variable
npx supabase secrets set FASTAPI_URL=http://host.docker.internal:8000
```

### **Database Function ไม่มี:**

```bash
# Check migration applied
npx supabase migration list

# Re-apply migration
npx supabase migration up

# Or reset
npx supabase db reset
```

### **Category ไม่มี embeddings:**

```bash
# Re-run embedding generation
npx tsx scripts/generate-category-embeddings.ts

# Or manually update one category
curl -X POST http://localhost:8000/api/embed \
  -H "Content-Type: application/json" \
  -d '{"text":"กล่อง ที่เก็บของ พลาสติก"}' \
  | jq -r '.embedding' > embedding.json

# Then update in database:
# UPDATE taxonomy_nodes SET embedding = '[...]' WHERE id = '...';
```

---

## 🎉 Success!

ตอนนี้ระบบพร้อมใช้งานแล้ว ด้วย:
- ✅ Local model (FREE!)
- ✅ Hybrid algorithm (72% accuracy)
- ✅ Same model as FastAPI (consistency)
- ✅ Supabase Edge Functions + Database Functions
- ✅ No OpenAI API cost!

**Next Steps:**
- ทดสอบ import products
- Monitor performance
- Tune classification threshold
- Add more keyword rules
- Expand taxonomy categories

---

📚 **See also:**
- `README_INTEGRATION.md` - Integration overview
- `MIGRATION_PLAN.md` - Full migration plan
- `SUMMARY_ARCHITECTURE.md` - Architecture summary
