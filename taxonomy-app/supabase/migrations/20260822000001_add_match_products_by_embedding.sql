-- Add missing match_products_by_embedding function
-- Referenced by app/api/import/process/route.ts for dedup checks during import,
-- but was never defined as a migration (function did not exist in the DB).
-- Mirrors the pattern of match_categories_by_embedding (384-dim, cosine similarity).

CREATE OR REPLACE FUNCTION match_products_by_embedding(
  query_embedding vector(384),
  match_threshold float DEFAULT 0.5,
  match_count int DEFAULT 5
)
RETURNS TABLE (
  id uuid,
  name_th text,
  category_id uuid,
  status text,
  similarity float
)
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
  RETURN QUERY
  SELECT
    p.id,
    p.name_th,
    p.category_id,
    p.status,
    1 - (p.embedding <=> query_embedding) as similarity
  FROM products p
  WHERE
    p.embedding IS NOT NULL
    AND (1 - (p.embedding <=> query_embedding)) >= match_threshold
  ORDER BY p.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION match_products_by_embedding(vector(384), float, int) IS
'Vector similarity search for product deduplication using pgvector cosine distance (384-dim)';

GRANT EXECUTE ON FUNCTION match_products_by_embedding(vector(384), float, int) TO anon, authenticated, service_role;
