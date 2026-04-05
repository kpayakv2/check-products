-- Migration: Support Local Model (384-dim) for Category Classification
-- Date: 2025-01-05
-- Model: paraphrase-multilingual-MiniLM-L12-v2 (Same as FastAPI)

-- Ensure vector extension exists
CREATE EXTENSION IF NOT EXISTS vector;

-- =====================================================
-- Function 1: Match Categories by Embedding (384-dim)
-- =====================================================

CREATE OR REPLACE FUNCTION match_categories_by_embedding(
  query_embedding vector(384),  -- Local model dimension
  match_threshold float DEFAULT 0.3,
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

-- Grant permissions
GRANT EXECUTE ON FUNCTION match_categories_by_embedding TO anon, authenticated;

COMMENT ON FUNCTION match_categories_by_embedding IS 
'Match categories using local model embeddings (384-dim). 
Model: paraphrase-multilingual-MiniLM-L12-v2 (same as FastAPI)';

-- =====================================================
-- Function 2: Hybrid Category Classification
-- =====================================================

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
  matched_keyword text
)
LANGUAGE plpgsql
AS $$
DECLARE
  keyword_weight float := 0.6;  -- 60% weight
  embedding_weight float := 0.4; -- 40% weight
BEGIN
  RETURN QUERY
  WITH keyword_matches AS (
    -- Keyword matching from keyword_rules
    SELECT 
      kr.category_id,
      tn.name_th as category_name,
      tn.level as category_level,
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
      tn.level as category_level,
      0.95 * keyword_weight as confidence,
      'name_match' as method,
      tn.name_th as matched_keyword
    FROM taxonomy_nodes tn
    WHERE 
      tn.is_active = true
      AND product_name ILIKE '%' || tn.name_th || '%'
    
    UNION
    
    -- Keyword matching from taxonomy keywords
    SELECT 
      tn.id as category_id,
      tn.name_th as category_name,
      tn.level as category_level,
      0.7 * keyword_weight as confidence,
      'taxonomy_keyword' as method,
      unnest(tn.keywords) as matched_keyword
    FROM taxonomy_nodes tn
    WHERE 
      tn.is_active = true
      AND tn.keywords IS NOT NULL
      AND product_name ~* ANY(tn.keywords)
  ),
  embedding_matches AS (
    -- Embedding matching using local model (384-dim)
    SELECT 
      tn.id as category_id,
      tn.name_th as category_name,
      tn.level as category_level,
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
    -- Combine both methods
    SELECT 
      category_id,
      category_name,
      category_level,
      SUM(confidence) as total_confidence,
      string_agg(DISTINCT method, '+' ORDER BY method) as methods,
      MAX(matched_keyword) as matched_keyword
    FROM (
      SELECT * FROM keyword_matches
      UNION ALL
      SELECT * FROM embedding_matches
    ) all_matches
    GROUP BY category_id, category_name, category_level
  )
  SELECT 
    c.category_id,
    c.category_name,
    c.category_level,
    c.total_confidence as confidence,
    c.methods as method,
    c.matched_keyword
  FROM combined c
  ORDER BY c.total_confidence DESC, c.category_level ASC
  LIMIT top_k;
END;
$$;

-- Grant permissions
GRANT EXECUTE ON FUNCTION hybrid_category_classification TO anon, authenticated;

COMMENT ON FUNCTION hybrid_category_classification IS 
'Hybrid classification (Keyword 60% + Embedding 40%) using local model.
Algorithm: Same as FastAPI backend (72% accuracy)
Model: paraphrase-multilingual-MiniLM-L12-v2 (384-dim)';

-- =====================================================
-- Function 3: Batch Category Classification
-- =====================================================

CREATE OR REPLACE FUNCTION batch_category_classification(
  products jsonb  -- [{"name": "...", "embedding": [...]}, ...]
)
RETURNS TABLE (
  product_name text,
  category_id uuid,
  category_name text,
  confidence float,
  method text
)
LANGUAGE plpgsql
AS $$
DECLARE
  product_record jsonb;
  result_record record;
BEGIN
  FOR product_record IN SELECT * FROM jsonb_array_elements(products)
  LOOP
    FOR result_record IN
      SELECT 
        product_record->>'name' as product_name,
        hcc.category_id,
        hcc.category_name,
        hcc.confidence,
        hcc.method
      FROM hybrid_category_classification(
        product_record->>'name',
        (product_record->>'embedding')::vector(384),
        1
      ) hcc
      LIMIT 1
    LOOP
      RETURN QUERY SELECT 
        result_record.product_name,
        result_record.category_id,
        result_record.category_name,
        result_record.confidence,
        result_record.method;
    END LOOP;
  END LOOP;
END;
$$;

GRANT EXECUTE ON FUNCTION batch_category_classification TO anon, authenticated;

COMMENT ON FUNCTION batch_category_classification IS 
'Batch process multiple products for category classification';

-- =====================================================
-- COMMENTED OUT: The following parts will be handled 
-- in the main init_hybrid_schema.sql migration
-- =====================================================

/*
-- Update products table to support 384-dim embeddings
COMMENT ON COLUMN products.embedding IS 
'Product embedding vector (384 dimensions).
Generated by: paraphrase-multilingual-MiniLM-L12-v2 (local model via FastAPI)
Same model as FastAPI backend for consistency.';

-- Create index for faster vector search
CREATE INDEX IF NOT EXISTS products_embedding_idx 
ON products USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS taxonomy_nodes_embedding_idx 
ON taxonomy_nodes USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 20);
*/

-- =====================================================
-- Helper function: Generate category embeddings
-- =====================================================

-- Note: This might still fail if taxonomy_nodes is not yet defined
-- We wrap it in a DO block to check if table exists first
DO $$
BEGIN
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'taxonomy_nodes') THEN
        EXECUTE 'CREATE OR REPLACE FUNCTION generate_category_embedding_text(category_row taxonomy_nodes)
        RETURNS text
        LANGUAGE plpgsql
        AS $func$
        DECLARE
          text_parts text[];
        BEGIN
          text_parts := ARRAY[category_row.name_th];
          IF category_row.name_en IS NOT NULL THEN
            text_parts := array_append(text_parts, category_row.name_en);
          END IF;
          IF category_row.keywords IS NOT NULL AND array_length(category_row.keywords, 1) > 0 THEN
            text_parts := text_parts || category_row.keywords;
          END IF;
          IF category_row.description IS NOT NULL THEN
            text_parts := array_append(text_parts, category_row.description);
          END IF;
          RETURN array_to_string(text_parts, '' '');
        END;
        $func$;';
        
        EXECUTE 'GRANT EXECUTE ON FUNCTION generate_category_embedding_text TO anon, authenticated;';
    END IF;
END
$$;
