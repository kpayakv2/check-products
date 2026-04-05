-- Thai Product Taxonomy - Category Matching Functions
-- Migration: Add database functions for category classification
-- Date: 2025-01-04 18:00

-- Ensure vector extension exists
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop existing functions to handle return type changes or parameter changes
DROP FUNCTION IF EXISTS match_categories_by_embedding(vector(384), float, int);
DROP FUNCTION IF EXISTS match_categories_by_embedding(vector(768), float, int);
DROP FUNCTION IF EXISTS hybrid_category_classification(text, vector(384), int);
DROP FUNCTION IF EXISTS hybrid_category_classification(text, vector(768), int);
DROP FUNCTION IF EXISTS batch_category_classification(jsonb);
DROP FUNCTION IF EXISTS batch_category_classification(jsonb, int);

-- =============================================================================
-- 1. Function: match_categories_by_embedding
-- Purpose: Vector similarity search สำหรับหาหมวดหมู่ที่ตรงกับ embedding
-- =============================================================================

CREATE OR REPLACE FUNCTION match_categories_by_embedding(
  query_embedding vector(384),  -- SentenceTransformer dimension (384 for local model)
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
STABLE
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

COMMENT ON FUNCTION match_categories_by_embedding(vector(384), float, int) IS 
'Vector similarity search for category matching using pgvector cosine distance (384-dim)';

-- Grant execute permission to anon and authenticated users
GRANT EXECUTE ON FUNCTION match_categories_by_embedding(vector(384), float, int) TO anon, authenticated;


-- =============================================================================
-- 2. Function: hybrid_category_classification
-- Purpose: Hybrid classification (Keyword 60% + Embedding 40%)
-- Based on test_category_algorithm.py with 72% accuracy
-- =============================================================================

CREATE OR REPLACE FUNCTION hybrid_category_classification(
  product_name text,
  product_embedding vector(384),
  top_k int DEFAULT 3
)
RETURNS TABLE (
  category_id uuid,
  category_name text,
  category_level int,
  confidence float,
  method text,
  matched_keyword text,
  methods text[]
)
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
  keyword_weight float := 0.6;  -- 60% weight for keyword matching
  embedding_weight float := 0.4; -- 40% weight for embedding similarity
BEGIN
  RETURN QUERY
  WITH keyword_matches AS (
    -- 1. Keyword rule matching
    SELECT 
      kr.category_id,
      tn.name_th as category_name,
      tn.level as category_level,
      kr.confidence_score * keyword_weight as confidence,
      'keyword_rule' as method,
      unnest(kr.keywords) as matched_keyword,
      ARRAY['keyword_rule']::text[] as methods
    FROM keyword_rules kr
    JOIN taxonomy_nodes tn ON kr.category_id = tn.id
    WHERE 
      kr.is_active = true
      AND tn.is_active = true
      AND EXISTS (
        SELECT 1 FROM unnest(kr.keywords) kw
        WHERE product_name ILIKE '%' || kw || '%'
      )
    
    UNION
    
    -- 2. Taxonomy keyword matching
    SELECT 
      tn.id as category_id,
      tn.name_th as category_name,
      tn.level as category_level,
      0.7 * keyword_weight as confidence,
      'taxonomy_keyword' as method,
      (
        SELECT kw FROM unnest(tn.keywords) kw 
        WHERE product_name ILIKE '%' || kw || '%' 
        LIMIT 1
      ) as matched_keyword,
      ARRAY['taxonomy_keyword']::text[] as methods
    FROM taxonomy_nodes tn
    WHERE 
      tn.is_active = true
      AND tn.keywords IS NOT NULL
      AND EXISTS (
        SELECT 1 FROM unnest(tn.keywords) kw
        WHERE product_name ILIKE '%' || kw || '%'
      )
    
    UNION
    
    -- 3. Category name matching
    SELECT 
      tn.id as category_id,
      tn.name_th as category_name,
      tn.level as category_level,
      0.95 * keyword_weight as confidence,
      'name_match' as method,
      tn.name_th as matched_keyword,
      ARRAY['name_match']::text[] as methods
    FROM taxonomy_nodes tn
    WHERE 
      tn.is_active = true
      AND product_name ILIKE '%' || tn.name_th || '%'
  ),
  
  embedding_matches AS (
    -- 4. Embedding similarity matching
    SELECT 
      tn.id as category_id,
      tn.name_th as category_name,
      tn.level as category_level,
      (1 - (tn.embedding <=> product_embedding)) * embedding_weight as confidence,
      'embedding' as method,
      NULL::text as matched_keyword,
      ARRAY['embedding']::text[] as methods
    FROM taxonomy_nodes tn
    WHERE 
      tn.embedding IS NOT NULL
      AND tn.is_active = true
      AND (1 - (tn.embedding <=> product_embedding)) >= 0.3
    ORDER BY tn.embedding <=> product_embedding
    LIMIT 10
  ),
  
  combined AS (
    -- Combine keyword and embedding matches
    SELECT 
      category_id,
      category_name,
      category_level,
      SUM(confidence) as total_confidence,
      MAX(method) as primary_method,
      MAX(matched_keyword) as matched_keyword,
      array_agg(DISTINCT m ORDER BY m) as methods
    FROM (
      SELECT 
        category_id,
        category_name,
        category_level,
        confidence,
        method,
        matched_keyword,
        unnest(methods) as m
      FROM keyword_matches
      
      UNION ALL
      
      SELECT 
        category_id,
        category_name,
        category_level,
        confidence,
        method,
        matched_keyword,
        unnest(methods) as m
      FROM embedding_matches
    ) all_matches
    GROUP BY category_id, category_name, category_level
  )
  
  SELECT 
    c.category_id,
    c.category_name,
    c.category_level,
    LEAST(c.total_confidence, 0.99) as confidence, -- Cap at 0.99
    CASE 
      WHEN array_length(c.methods, 1) > 1 THEN 'hybrid'
      ELSE c.primary_method
    END as method,
    c.matched_keyword,
    c.methods
  FROM combined c
  ORDER BY c.total_confidence DESC, c.category_level ASC
  LIMIT top_k;
END;
$$;

COMMENT ON FUNCTION hybrid_category_classification(text, vector(384), int) IS 
'Hybrid category classification: 60% keyword matching + 40% embedding similarity (384-dim, 72% accuracy)';

-- Grant execute permission
GRANT EXECUTE ON FUNCTION hybrid_category_classification(text, vector(384), int) TO anon, authenticated;


-- =============================================================================
-- 3. Function: batch_category_classification
-- Purpose: Classify multiple products in one call (performance optimization)
-- =============================================================================

CREATE OR REPLACE FUNCTION batch_category_classification(
  product_data jsonb,  -- [{"name": "...", "embedding": [...]}, ...]
  top_k int DEFAULT 3
)
RETURNS TABLE (
  product_name text,
  category_id uuid,
  category_name text,
  confidence float,
  method text
)
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
  product jsonb;
BEGIN
  FOR product IN SELECT * FROM jsonb_array_elements(product_data)
  LOOP
    RETURN QUERY
    SELECT 
      product->>'name' as product_name,
      h.category_id,
      h.category_name,
      h.confidence,
      h.method
    FROM hybrid_category_classification(
      product->>'name',
      (product->'embedding')::vector(384),
      1  -- top 1 only for batch
    ) h
    LIMIT 1;
  END LOOP;
END;
$$;

COMMENT ON FUNCTION batch_category_classification(jsonb, int) IS 
'Batch classify multiple products (optimized for performance, 384-dim)';

GRANT EXECUTE ON FUNCTION batch_category_classification(jsonb, int) TO anon, authenticated;


-- =============================================================================
-- 4. Indexes for Performance
-- =============================================================================

-- Vector index for faster similarity search (if not exists)
CREATE INDEX IF NOT EXISTS idx_taxonomy_nodes_embedding_vector 
ON taxonomy_nodes 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Keyword search index
CREATE INDEX IF NOT EXISTS idx_taxonomy_nodes_keywords_gin 
ON taxonomy_nodes 
USING gin (keywords);

-- Active categories index
CREATE INDEX IF NOT EXISTS idx_taxonomy_nodes_active 
ON taxonomy_nodes (is_active) 
WHERE is_active = true;

-- Keyword rules index
CREATE INDEX IF NOT EXISTS idx_keyword_rules_active 
ON keyword_rules (is_active, category_id) 
WHERE is_active = true;


-- =============================================================================
-- 5. Test Functions (Optional - for debugging)
-- =============================================================================

-- Test function: Get sample categories with embeddings
CREATE OR REPLACE FUNCTION get_sample_categories_with_embeddings(
  sample_size int DEFAULT 5
)
RETURNS TABLE (
  id uuid,
  name_th text,
  has_embedding boolean,
  embedding_dimension int
)
LANGUAGE sql
STABLE
AS $$
  SELECT 
    id,
    name_th,
    embedding IS NOT NULL as has_embedding,
    384 as embedding_dimension
  FROM taxonomy_nodes
  WHERE is_active = true
  ORDER BY created_at DESC
  LIMIT sample_size;
$$;

GRANT EXECUTE ON FUNCTION get_sample_categories_with_embeddings(int) TO anon, authenticated;


-- =============================================================================
-- Migration Complete
-- =============================================================================

-- Log migration
DO $$
BEGIN
  RAISE NOTICE 'Migration 20250104180000_category_matching_functions.sql completed successfully';
END $$;
