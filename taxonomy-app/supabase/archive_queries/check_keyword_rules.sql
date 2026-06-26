-- Check keyword rules data
SELECT 
  kr.id,
  kr.code,
  kr.name,
  kr.keywords,
  kr.confidence_score,
  kr.is_active,
  tn.name_th as category_name
FROM keyword_rules kr
LEFT JOIN taxonomy_nodes tn ON kr.category_id = tn.id
ORDER BY kr.confidence_score DESC
LIMIT 10;
