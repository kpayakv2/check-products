-- Test all database functions
-- 1. Test match_categories_by_embedding
SELECT 'Testing match_categories_by_embedding:' as test_name;
SELECT * FROM match_categories_by_embedding(
  (SELECT embedding FROM taxonomy_nodes WHERE embedding IS NOT NULL LIMIT 1),
  0.3,
  3
) LIMIT 3;

-- 2. Test hybrid_category_classification with real embedding
SELECT 'Testing hybrid_category_classification:' as test_name;
SELECT * FROM hybrid_category_classification(
  'กรรไกรตัดหญ้า',
  (SELECT embedding FROM taxonomy_nodes WHERE embedding IS NOT NULL LIMIT 1),
  3
);

-- 3. Test generate_category_embedding_text
SELECT 'Testing generate_category_embedding_text:' as test_name;
SELECT generate_category_embedding_text(tn.*) as embedding_text
FROM taxonomy_nodes tn 
WHERE tn.embedding IS NOT NULL 
LIMIT 3;
