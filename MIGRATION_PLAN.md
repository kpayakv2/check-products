# 🚀 แผนย้ายจาก FastAPI ไปใช้ Supabase-First Architecture

**วันที่:** 2025-10-04 18:04

---

## 📋 Overview

**เป้าหมาย:** บูรณาการระบบให้ใช้ Supabase เป็นหลัก พร้อม Edge Functions, Database Functions, RLS Policies  
**ผลลัพธ์:** Architecture ที่ชัดเจน, ปลอดภัย, scalable, และไม่ซ้ำซ้อน  
**เวลาโดยประมาณ:** 2-3 วัน  

---

## 📊 Step-by-Step Migration Plan

### **Phase 1: เพิ่ม Database Functions (2-3 ชม.)**

#### **1.1 สร้าง match_categories_by_embedding()**

```sql
-- File: supabase/migrations/YYYYMMDD_category_matching_functions.sql

-- Function สำหรับหา category ที่ match กับ embedding
CREATE OR REPLACE FUNCTION match_categories_by_embedding(
  query_embedding vector(384),  -- OpenAI embedding dimension
  match_threshold float DEFAULT 0.5,
  match_count int DEFAULT 5
)
RETURNS TABLE (
  category_id uuid,
  category_name text,
  category_level int,
  similarity float,
  keywords text[]
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT 
    tn.id,
    tn.name_th,
    tn.level,
    1 - (tn.embedding <=> query_embedding) as similarity,
    tn.keywords
  FROM taxonomy_nodes tn
  WHERE 
    tn.embedding IS NOT NULL
    AND tn.is_active = true
    AND (1 - (tn.embedding <=> query_embedding)) >= match_threshold
  ORDER BY tn.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION match_categories_by_embedding TO anon, authenticated;
```

#### **1.2 สร้าง hybrid_category_classification()**

```sql
-- Function สำหรับ Hybrid Classification (Keyword + Embedding)
CREATE OR REPLACE FUNCTION hybrid_category_classification(
  product_name text,
  product_embedding vector(384),
  top_k int DEFAULT 3
)
RETURNS TABLE (
  category_id uuid,
  category_name text,
  confidence float,
  method text,
  matched_keyword text
)
LANGUAGE plpgsql
AS $$
DECLARE
  keyword_weight float := 0.6;
  embedding_weight float := 0.4;
BEGIN
  RETURN QUERY
  WITH keyword_matches AS (
    -- Keyword matching
    SELECT 
      kr.category_id,
      tn.name_th as category_name,
      kr.confidence_score * keyword_weight as confidence,
      'keyword' as method,
      unnest(kr.keywords) as matched_keyword
    FROM keyword_rules kr
    JOIN taxonomy_nodes tn ON kr.category_id = tn.id
    WHERE 
      kr.is_active = true
      AND tn.is_active = true
      AND product_name ~* ANY(kr.keywords)
    
    UNION
    
    -- Category name matching
    SELECT 
      tn.id as category_id,
      tn.name_th as category_name,
      0.95 * keyword_weight as confidence,
      'name_match' as method,
      tn.name_th as matched_keyword
    FROM taxonomy_nodes tn
    WHERE 
      tn.is_active = true
      AND product_name ILIKE '%' || tn.name_th || '%'
  ),
  embedding_matches AS (
    -- Embedding matching
    SELECT 
      tn.id as category_id,
      tn.name_th as category_name,
      (1 - (tn.embedding <=> product_embedding)) * embedding_weight as confidence,
      'embedding' as method,
      NULL::text as matched_keyword
    FROM taxonomy_nodes tn
    WHERE 
      tn.embedding IS NOT NULL
      AND tn.is_active = true
      AND (1 - (tn.embedding <=> product_embedding)) >= 0.3
    ORDER BY tn.embedding <=> product_embedding
    LIMIT 10
  ),
  combined AS (
    SELECT 
      category_id,
      category_name,
      SUM(confidence) as total_confidence,
      string_agg(DISTINCT method, '+' ORDER BY method) as methods,
      MAX(matched_keyword) as matched_keyword
    FROM (
      SELECT * FROM keyword_matches
      UNION ALL
      SELECT * FROM embedding_matches
    ) all_matches
    GROUP BY category_id, category_name
  )
  SELECT 
    c.category_id,
    c.category_name,
    c.total_confidence as confidence,
    c.methods as method,
    c.matched_keyword
  FROM combined c
  ORDER BY c.total_confidence DESC
  LIMIT top_k;
END;
$$;

GRANT EXECUTE ON FUNCTION hybrid_category_classification TO anon, authenticated;
```

#### **1.3 Apply Migration**

```bash
cd taxonomy-app
npx supabase db reset  # หรือ
npx supabase migration up
```

---

### **Phase 2: สร้าง/ปรับปรุง Edge Functions (3-4 ชม.)**

#### **2.1 สร้าง hybrid-classification Function**

```typescript
// File: supabase/functions/hybrid-classification/index.ts

import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface ClassificationRequest {
  product_name: string
  method?: 'keyword' | 'embedding' | 'hybrid'
  top_k?: number
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const supabaseUrl = Deno.env.get('SUPABASE_URL')!
    const supabaseKey = Deno.env.get('SUPABASE_ANON_KEY')!
    
    const supabase = createClient(supabaseUrl, supabaseKey, {
      global: {
        headers: { Authorization: req.headers.get('Authorization')! },
      },
    })

    const { product_name, method = 'hybrid', top_k = 3 }: ClassificationRequest 
      = await req.json()

    if (!product_name?.trim()) {
      return new Response(
        JSON.stringify({ error: 'Product name is required' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    const startTime = Date.now()
    let suggestions: any[] = []

    if (method === 'keyword') {
      // Keyword-only classification
      const { data, error } = await supabase
        .from('keyword_rules')
        .select('*, category:taxonomy_nodes(*)')
        .eq('is_active', true)
      
      if (!error) {
        // Filter by keyword matching
        suggestions = data
          .filter(rule => 
            rule.keywords.some(kw => 
              product_name.toLowerCase().includes(kw.toLowerCase())
            )
          )
          .map(rule => ({
            category_id: rule.category_id,
            category_name: rule.category.name_th,
            category_level: rule.category.level,
            confidence: rule.confidence_score,
            method: 'keyword',
            matched_keyword: rule.keywords.find(kw => 
              product_name.toLowerCase().includes(kw.toLowerCase())
            )
          }))
          .slice(0, top_k)
      }

    } else if (method === 'embedding' || method === 'hybrid') {
      // Generate embedding first
      const embeddingResponse = await supabase.functions.invoke('generate-embeddings', {
        body: {
          texts: [product_name],
          model: 'text-embedding-ada-002'
        }
      })

      if (embeddingResponse.error) {
        throw new Error('Failed to generate embedding')
      }

      const embedding = embeddingResponse.data.embeddings[0]

      if (method === 'embedding') {
        // Embedding-only classification
        const { data, error } = await supabase.rpc('match_categories_by_embedding', {
          query_embedding: embedding,
          match_threshold: 0.3,
          match_count: top_k
        })

        if (!error) {
          suggestions = data.map(m => ({
            category_id: m.category_id,
            category_name: m.category_name,
            category_level: m.category_level,
            confidence: m.similarity,
            method: 'embedding'
          }))
        }

      } else {
        // Hybrid classification (call database function)
        const { data, error } = await supabase.rpc('hybrid_category_classification', {
          product_name: product_name,
          product_embedding: embedding,
          top_k: top_k
        })

        if (!error) {
          suggestions = data
        }
      }
    }

    const processingTime = Date.now() - startTime

    return new Response(
      JSON.stringify({
        product_name,
        suggestions,
        top_suggestion: suggestions[0] || null,
        processing_time: processingTime,
        method
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200,
      }
    )

  } catch (error) {
    console.error('Classification error:', error)
    return new Response(
      JSON.stringify({ error: error.message }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    )
  }
})
```

#### **2.2 Deploy Edge Function**

```bash
cd taxonomy-app

# Deploy single function
npx supabase functions deploy hybrid-classification

# หรือ deploy ทั้งหมด
npx supabase functions deploy
```

---

### **Phase 3: อัปเดต Next.js API Routes (2-3 ชม.)**

#### **3.1 แก้ไข import/process/route.ts**

```typescript
// File: taxonomy-app/app/api/import/process/route.ts

import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
)

// เปลี่ยนจาก FastAPI เป็น Edge Functions
async function generateEmbedding(text: string): Promise<number[]> {
  try {
    // เรียก Edge Function แทน FastAPI
    const { data, error } = await supabase.functions.invoke('generate-embeddings', {
      body: {
        texts: [text],
        model: 'text-embedding-ada-002'  // หรือ paraphrase-multilingual
      }
    })

    if (error) {
      console.error('Embedding generation failed:', error)
      throw error
    }

    return data.embeddings[0]

  } catch (error) {
    console.error('Error generating embedding:', error)
    // Fallback: return zero vector หรือ skip
    return []
  }
}

async function suggestCategory(
  productName: string,
  tokens: string[],
  attributes: Record<string, any>,
  embedding: number[]
): Promise<{
  category_id: string | null
  category_name: string
  confidence: number
  method: string
  all_suggestions: any[]
}> {
  try {
    // เรียก Edge Function แทน FastAPI
    const { data, error } = await supabase.functions.invoke('hybrid-classification', {
      body: {
        product_name: productName,
        method: 'hybrid',
        top_k: 5
      }
    })

    if (error) {
      console.error('Category classification failed:', error)
      throw error
    }

    return {
      category_id: data.top_suggestion?.category_id || null,
      category_name: data.top_suggestion?.category_name || 'ไม่ระบุ',
      confidence: data.top_suggestion?.confidence || 0,
      method: data.top_suggestion?.method || 'hybrid',
      all_suggestions: data.suggestions || []
    }

  } catch (error) {
    console.error('Error suggesting category:', error)
    // Fallback: simple keyword matching
    return fallbackCategoryMatching(tokens)
  }
}

// Main processing function
export async function POST(request: Request) {
  try {
    const formData = await request.formData()
    const file = formData.get('file') as File
    
    if (!file) {
      return Response.json({ error: 'No file provided' }, { status: 400 })
    }

    // Parse CSV
    const text = await file.text()
    const rows = parseCSV(text)

    // Process each product
    const results = []
    for (const row of rows) {
      const productName = row['product_name'] || row['name_th']
      
      // 1. Clean & tokenize
      const cleaned = cleanText(productName)
      const tokens = tokenize(cleaned)
      
      // 2. Extract attributes
      const attributes = extractAttributes(tokens)
      
      // 3. Generate embedding (Edge Function)
      const embedding = await generateEmbedding(productName)
      
      // 4. Suggest category (Edge Function + Database Function)
      const suggestion = await suggestCategory(
        productName,
        tokens,
        attributes,
        embedding
      )
      
      // 5. Save to database
      const { data: product, error } = await supabase
        .from('products')
        .insert({
          name_th: productName,
          embedding: embedding,
          category_id: suggestion.category_id,
          confidence_score: suggestion.confidence,
          keywords: tokens,
          metadata: {
            attributes,
            processing_method: suggestion.method
          },
          status: suggestion.confidence > 0.7 ? 'auto-approved' : 'pending'
        })
        .select()
        .single()

      if (!error && product) {
        // Save all suggestions
        const suggestionRecords = suggestion.all_suggestions.map(s => ({
          product_id: product.id,
          category_id: s.category_id,
          confidence: s.confidence,
          method: s.method,
          matched_keyword: s.matched_keyword
        }))

        await supabase
          .from('product_category_suggestions')
          .insert(suggestionRecords)
      }

      results.push({
        product: productName,
        category: suggestion.category_name,
        confidence: suggestion.confidence,
        status: error ? 'failed' : 'success'
      })
    }

    return Response.json({
      success: true,
      total: rows.length,
      processed: results.length,
      results
    })

  } catch (error) {
    console.error('Import processing error:', error)
    return Response.json(
      { error: 'Failed to process import' },
      { status: 500 }
    )
  }
}
```

---

### **Phase 4: ปรับปรุง FastAPI (1-2 ชม.)**

#### **4.1 ลดขนาด api_server.py**

```python
# File: api_server.py (ปรับปรุง)

# ลบออก:
# - CategoryClassifier class → ย้ายไป Edge Function แล้ว
# - /api/classify/category → ใช้ Edge Function แทน
# - /api/classify/batch → ใช้ Edge Function แทน
# - initialize_category_classifier() → ไม่ต้องใช้แล้ว
# - initialize_supabase() → ไม่ต้องใช้แล้ว

# เหลือเฉพาะ:
# 1. /api/embed - สำหรับ web_server.py (Flask)
# 2. /api/embed/batch - สำหรับ web_server.py (Flask)
# 3. Health check endpoints

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "Product Similarity API (Legacy)",
        "version": "5.0.0",
        "status": "operational",
        "note": "This API is only for web_server.py (Flask). Use Supabase Edge Functions for new features.",
        "endpoints": {
            "health": "/api/v1/health",
            "embeddings": "/api/embed",
            "batch_embeddings": "/api/embed/batch",
            "docs": "/docs"
        }
    }

@app.post("/api/embed", response_model=EmbeddingResponse)
async def embed_single(request: EmbeddingRequest):
    """
    Generate embedding for a single text.
    Used by web_server.py (Flask) only.
    """
    try:
        start_time = time.time()
        model = initialize_embedding_model()
        
        embeddings = model.encode([request.text])
        embedding = embeddings[0].tolist()
        
        processing_time = time.time() - start_time
        
        return EmbeddingResponse(
            embedding=embedding,
            dimension=len(embedding),
            model=model.model_name,
            processing_time=round(processing_time, 3)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

# ส่วนที่เหลือเหมือนเดิม...
```

---

### **Phase 5: ทดสอบระบบ (2-3 ชม.)**

#### **5.1 Test Database Functions**

```sql
-- Test match_categories_by_embedding
SELECT * FROM match_categories_by_embedding(
  '[-0.123, 0.456, ...]'::vector,  -- sample embedding
  0.5,
  5
);

-- Test hybrid_category_classification
SELECT * FROM hybrid_category_classification(
  'กล่องล็อค 560 มล',
  '[-0.123, 0.456, ...]'::vector,
  3
);
```

#### **5.2 Test Edge Functions**

```bash
# Test locally
npx supabase functions serve hybrid-classification

# Test with curl
curl -X POST http://localhost:54321/functions/v1/hybrid-classification \
  -H "Authorization: Bearer ${SUPABASE_ANON_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "กล่องล็อค 560 มล",
    "method": "hybrid",
    "top_k": 3
  }'
```

#### **5.3 Test Next.js Integration**

```bash
cd taxonomy-app
npm run dev

# Upload test CSV via http://localhost:3000/import
# ตรวจสอบ:
# - Embeddings ถูกสร้าง
# - Category suggestions ถูกต้อง
# - Products ถูกบันทึกใน database
# - Suggestions ถูกบันทึกใน product_category_suggestions
```

#### **5.4 Test RLS Policies**

```typescript
// Test with different users
const anonClient = createClient(url, anonKey)
const authClient = createClient(url, anonKey).auth.signIn(...)

// ทดสอบว่า RLS policies ทำงาน:
// - Anonymous users อ่านได้เฉพาะข้อมูลสาธารณะ
// - Authenticated users อ่าน/เขียนได้ตามสิทธิ์
// - Service role bypass RLS (ใช้เฉพาะ admin tasks)
```

---

### **Phase 6: Documentation & Cleanup (1 ชม.)**

#### **6.1 อัปเดต README**

```markdown
# Architecture

## Components

1. **Next.js Frontend** (Port 3000)
   - Main UI
   - Import workflow
   - Uses Supabase Client (ANON_KEY)

2. **Supabase Backend**
   - Database (PostgreSQL + pgvector)
   - Edge Functions (AI processing)
   - Database Functions (business logic)
   - RLS Policies (security)
   - Triggers (automation)

3. **FastAPI** (Port 8000) - Legacy
   - Only for web_server.py (Flask)
   - Embeddings generation
   - Will be deprecated

4. **Flask** (Port 5000)
   - Product Similarity Checker
   - Human review workflow
   - Uses FastAPI for embeddings

## Data Flow

```
User → Next.js → Supabase Edge Functions → Database Functions → PostgreSQL
                     ↓
                 (Optional) FastAPI (legacy)
```
```

#### **6.2 อัปเดต Environment Variables**

```bash
# .env.local (Next.js)
NEXT_PUBLIC_SUPABASE_URL=http://localhost:54321
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...

# Edge Functions use OpenAI by default
# To use local models, see supabase/functions/README.md

# FastAPI (legacy, only for web_server.py)
FASTAPI_URL=http://localhost:8000  # Optional
```

---

## ✅ Checklist

### **Phase 1: Database Functions**
- [ ] สร้าง match_categories_by_embedding()
- [ ] สร้าง hybrid_category_classification()
- [ ] Apply migration
- [ ] ทดสอบ functions

### **Phase 2: Edge Functions**
- [ ] สร้าง hybrid-classification function
- [ ] Deploy to Supabase
- [ ] ทดสอบ locally
- [ ] ทดสอบ deployed version

### **Phase 3: Next.js**
- [ ] แก้ไข generateEmbedding()
- [ ] แก้ไข suggestCategory()
- [ ] ทดสอบ import workflow
- [ ] ตรวจสอบ database records

### **Phase 4: FastAPI**
- [ ] ลบ CategoryClassifier
- [ ] ลบ /api/classify endpoints
- [ ] อัปเดต documentation
- [ ] ทดสอบ web_server.py ยังทำงานได้

### **Phase 5: Testing**
- [ ] Database functions work
- [ ] Edge functions work
- [ ] Next.js integration works
- [ ] RLS policies work
- [ ] Triggers fire correctly
- [ ] Performance acceptable

### **Phase 6: Documentation**
- [ ] อัปเดต README
- [ ] อัปเดต ARCHITECTURE.md
- [ ] สร้าง Migration Guide
- [ ] อัปเดต API docs

---

## 📊 ผลลัพธ์ที่คาดหวัง

### **Before (FastAPI-Centric)**
```
❌ RLS bypassed
❌ No audit trail
❌ Duplicate logic (FastAPI + Edge Functions)
❌ 3 web servers running
❌ Complex architecture
```

### **After (Supabase-First)**
```
✅ RLS respected
✅ Complete audit trail
✅ Single source of truth (Edge Functions + Database Functions)
✅ 3 web servers (but clear separation of concerns)
✅ Clean architecture
```

---

## 🎯 Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| Phase 1 | 2-3 ชม. | Database Functions |
| Phase 2 | 3-4 ชม. | Edge Functions |
| Phase 3 | 2-3 ชม. | Next.js Update |
| Phase 4 | 1-2 ชม. | FastAPI Cleanup |
| Phase 5 | 2-3 ชม. | Testing |
| Phase 6 | 1 ชม. | Documentation |
| **Total** | **11-16 ชม.** | **~2-3 วัน** |

---

**เริ่มได้เลย! Migration Plan พร้อมแล้ว** 🚀
