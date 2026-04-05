-- Migration: Add Local Model Functions (Legacy/Duplicate)
-- Date: 2025-10-05
-- Note: Commented out because a more advanced version of these functions 
-- was already applied in 20250928180000_category_matching_functions.sql

/*
-- Ensure vector extension exists
CREATE EXTENSION IF NOT EXISTS vector;

-- Function 1: Match Categories by Embedding (384-dim)
CREATE OR REPLACE FUNCTION match_categories_by_embedding(
  query_embedding vector(384),
  match_threshold float DEFAULT 0.3,
  match_count int DEFAULT 5
)
...
*/
