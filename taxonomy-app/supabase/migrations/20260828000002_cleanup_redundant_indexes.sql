-- Clean up redundant indexes found in ER/schema audit (2026-08-28)
--
-- taxonomy_nodes.embedding ended up with TWO ivfflat cosine indexes at once:
--   idx_taxonomy_nodes_embedding_vector (lists=100, from
--     20250928180000_category_matching_functions.sql)
--   taxonomy_nodes_embedding_idx (lists=20, from
--     20260822000000_fix_embedding_dimension.sql)
-- The dimension-fix migration dropped the original idx_taxonomy_nodes_embedding
-- but never touched the differently-named _vector duplicate created earlier.
-- Keeping lists=20: with 134 rows, pgvector's own guidance for ivfflat is
-- roughly lists ~= sqrt(rows) (~12 here) for small tables; lists=100
-- over-partitions and hurts ANN recall for no benefit.
DROP INDEX IF EXISTS idx_taxonomy_nodes_embedding_vector;

-- taxonomy_nodes.keywords also has two identical GIN indexes.
-- Keeping the earlier, non-suffixed one for minimal diff / least surprise.
DROP INDEX IF EXISTS idx_taxonomy_nodes_keywords_gin;

-- Optional low-risk cleanup: idx_taxonomy_nodes_code duplicates the index
-- Postgres already maintains internally for the taxonomy_nodes_code_key
-- UNIQUE constraint.
DROP INDEX IF EXISTS idx_taxonomy_nodes_code;
