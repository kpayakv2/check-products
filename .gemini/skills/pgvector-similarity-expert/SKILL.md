---
name: pgvector-similarity-expert
description: |
  Expert guidance for managing pgvector in PostgreSQL for high-accuracy product matching.
  ใช้เมื่อต้องการ tune similarity algorithm, จัดการ embeddings, หรือวิเคราะห์ผล classification

  **Trigger when user asks to:**
  - Tuning the similarity matching algorithm
  - Managing product embeddings and indexing performance
  - Investigating low-accuracy classification results
  - ปรับ match_threshold หรือ HNSW parameters

  **Keywords:** pgvector, similarity, embedding, HNSW, cosine, threshold, vector search
---

# 🔍 pgvector Similarity Expert Skill

## ⚠️ Project-Specific Settings (check-products)

| Setting | **โปรเจกต์นี้** |
|---------|-----------------|
| Dimension | **`384`** (paraphrase-multilingual-MiniLM-L12-v2) |
| Column Type | **`vector(384)`** |
| Index ops | **`vector_cosine_ops`** |
| Distance operator | `<=>` (Cosine) |
| Embedding Provider | FastAPI `http://127.0.0.1:8000/api/embed` |
| Target Accuracy | **≥ 72%** F1-score |

## Hybrid Algorithm (60/40)
- **Keyword 60%** — `keyword_rules` + `taxonomy_nodes.keywords` + `name_match`
- **Embedding 40%** — Cosine Distance (`<=>`) กับ `taxonomy_nodes.embedding`

## Key Procedures

### 1. Index Tuning
```sql
-- HNSW Index สำหรับโปรเจกต์นี้ (vector ไม่ใช่ halfvec)
CREATE INDEX ON taxonomy_nodes
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- Query-time recall
SET hnsw.ef_search = 100;
```

### 2. Similarity Search Pattern
```sql
SELECT id, name, 1 - (embedding <=> $1::vector(384)) AS similarity
FROM taxonomy_nodes
ORDER BY embedding <=> $1::vector(384)
LIMIT 10;
```

### 3. Similarity Analysis (หา Weak Points)
```sql
-- สินค้าที่ได้คะแนน Similarity ต่ำ — อาจต้องเพิ่ม keyword
SELECT p.name, t.name as category, p.similarity_score
FROM products p
JOIN taxonomy_nodes t ON p.category_id = t.id
WHERE p.similarity_score < 0.72
ORDER BY p.similarity_score ASC
LIMIT 20;
```

### 4. Threshold Tuning
- ค่าเริ่มต้น: `match_threshold = 0.72`
- เพิ่ม threshold → Precision ขึ้น, Recall ลด (strict)
- ลด threshold → Recall ขึ้น, Precision ลด (lenient)
- ทดสอบบน Local เสมอก่อน Production

### 5. Performance Monitoring
```sql
-- ตรวจสอบขนาด index
SELECT pg_size_pretty(pg_relation_size('taxonomy_nodes_embedding_idx'));

-- Debug query performance
EXPLAIN (ANALYZE, BUFFERS)
  SELECT id FROM taxonomy_nodes
  ORDER BY embedding <=> $1::vector(384) LIMIT 10;
```

## ⚖️ Mandates
- ห้ามรัน Query ซับซ้อนบน Production โดยไม่ทดสอบ Local ก่อน
- ต้องรักษาขนาด Embedding **384 dimensions** เสมอ
- การแก้ Algorithm ต้องรักษา F1-score ≥ 72%
