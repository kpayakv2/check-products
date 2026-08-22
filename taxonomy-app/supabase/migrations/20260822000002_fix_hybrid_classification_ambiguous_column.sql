-- Fix "column reference \"category_id\" is ambiguous" in hybrid_category_classification.
--
-- RETURNS TABLE(category_id uuid, category_name text, category_level int, ...) implicitly
-- declares plpgsql variables with those same names. The `combined` CTE referenced
-- category_id/category_name/category_level unqualified, which Postgres cannot resolve
-- between the subquery column and the OUT-parameter variable. Qualifying every reference
-- to the `all_matches` subquery alias removes the ambiguity.

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
      all_matches.category_id,
      all_matches.category_name,
      all_matches.category_level,
      SUM(all_matches.confidence) as total_confidence,
      MAX(all_matches.method) as primary_method,
      MAX(all_matches.matched_keyword) as matched_keyword,
      array_agg(DISTINCT all_matches.m ORDER BY all_matches.m) as methods
    FROM (
      SELECT
        keyword_matches.category_id,
        keyword_matches.category_name,
        keyword_matches.category_level,
        keyword_matches.confidence,
        keyword_matches.method,
        keyword_matches.matched_keyword,
        unnest(keyword_matches.methods) as m
      FROM keyword_matches

      UNION ALL

      SELECT
        embedding_matches.category_id,
        embedding_matches.category_name,
        embedding_matches.category_level,
        embedding_matches.confidence,
        embedding_matches.method,
        embedding_matches.matched_keyword,
        unnest(embedding_matches.methods) as m
      FROM embedding_matches
    ) all_matches
    GROUP BY all_matches.category_id, all_matches.category_name, all_matches.category_level
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

GRANT EXECUTE ON FUNCTION hybrid_category_classification(text, vector(384), int) TO anon, authenticated;
