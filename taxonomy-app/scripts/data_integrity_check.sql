-- Data Integrity Check Script
-- Purpose: Comprehensive check of database consistency and data quality

\echo '=== THAI PRODUCT TAXONOMY SYSTEM - DATA INTEGRITY REPORT ==='
\echo ''

-- 1. TAXONOMY STRUCTURE CHECK
\echo '1. TAXONOMY STRUCTURE ANALYSIS:'
\echo '================================'
SELECT 
    'Main Categories' as category_type,
    COUNT(*) as count,
    STRING_AGG(code, ', ' ORDER BY code) as codes
FROM taxonomy_nodes WHERE level = 0
UNION ALL
SELECT 
    'Sub Categories' as category_type,
    COUNT(*) as count,
    CASE WHEN COUNT(*) > 20 THEN CONCAT(COUNT(*), ' items (showing first 10): ', 
         (SELECT STRING_AGG(code, ', ' ORDER BY code) FROM 
          (SELECT code FROM taxonomy_nodes WHERE level = 1 ORDER BY code LIMIT 10) x))
    ELSE STRING_AGG(code, ', ' ORDER BY code) 
    END as codes
FROM taxonomy_nodes WHERE level = 1;

\echo ''
\echo '2. SHORT CODE COVERAGE:'
\echo '======================='
SELECT 
    level,
    COUNT(*) as total_nodes,
    COUNT(short_code) as with_short_code,
    COUNT(*) - COUNT(short_code) as missing_short_code,
    ROUND(COUNT(short_code)::numeric / COUNT(*) * 100, 2) as coverage_percentage
FROM taxonomy_nodes 
GROUP BY level 
ORDER BY level;

\echo ''
\echo '3. PARENT-CHILD RELATIONSHIPS:'
\echo '=============================='
-- Check orphaned subcategories
SELECT 
    'Orphaned Subcategories' as issue_type,
    COUNT(*) as count,
    COALESCE(STRING_AGG(code, ', '), 'None') as problematic_codes
FROM taxonomy_nodes 
WHERE level = 1 AND parent_id NOT IN (SELECT id FROM taxonomy_nodes WHERE level = 0)

UNION ALL

-- Check main categories without subcategories
SELECT 
    'Main Cats Without Subs' as issue_type,
    COUNT(*) as count,
    COALESCE(STRING_AGG(code, ', '), 'None') as problematic_codes
FROM taxonomy_nodes main
WHERE level = 0 
AND id NOT IN (SELECT DISTINCT parent_id FROM taxonomy_nodes WHERE level = 1 AND parent_id IS NOT NULL);

\echo ''
\echo '4. SYNONYM SYSTEM INTEGRITY:'
\echo '============================'
SELECT 
    'Total Synonym Lemmas' as metric,
    COUNT(*) as count,
    '' as details
FROM synonym_lemmas
UNION ALL
SELECT 
    'Total Synonym Terms' as metric,
    COUNT(*) as count,
    '' as details  
FROM synonym_terms
UNION ALL
SELECT 
    'Lemmas with Category Links' as metric,
    COUNT(*) as count,
    ROUND(COUNT(*)::numeric / (SELECT COUNT(*) FROM synonym_lemmas) * 100, 2)::text || '%' as details
FROM synonym_lemmas WHERE category_id IS NOT NULL
UNION ALL
SELECT 
    'Orphaned Synonym Terms' as metric,
    COUNT(*) as count,
    CASE WHEN COUNT(*) > 0 THEN 'ISSUE DETECTED' ELSE 'OK' END as details
FROM synonym_terms 
WHERE lemma_id NOT IN (SELECT id FROM synonym_lemmas);

\echo ''
\echo '5. KEYWORD RULES INTEGRITY:'
\echo '==========================='
SELECT 
    'Total Keyword Rules' as metric,
    COUNT(*) as count,
    '' as details
FROM keyword_rules
UNION ALL
SELECT 
    'Total Regex Rules' as metric,
    COUNT(*) as count,
    '' as details
FROM regex_rules  
UNION ALL
SELECT 
    'Active Keyword Rules' as metric,
    COUNT(*) as count,
    ROUND(COUNT(*)::numeric / (SELECT COUNT(*) FROM keyword_rules) * 100, 2)::text || '%' as details
FROM keyword_rules WHERE is_active = true
UNION ALL
SELECT 
    'Rules with Invalid Categories' as metric,
    COUNT(*) as count,
    CASE WHEN COUNT(*) > 0 THEN 'ISSUE DETECTED' ELSE 'OK' END as details
FROM keyword_rules 
WHERE category_id NOT IN (SELECT id FROM taxonomy_nodes);

\echo ''
\echo '6. SYSTEM SETTINGS:'
\echo '=================='
SELECT 
    setting_key,
    setting_value,
    description
FROM system_settings 
ORDER BY setting_key;

\echo ''  
\echo '7. MISSING SUBCATEGORIES VERIFICATION:'
\echo '===================================='
-- Check if we now have all expected subcategories from original dataset
WITH expected_subcats AS (
  SELECT unnest(ARRAY[
    'cat_001_001', 'cat_001_002', 'cat_001_003', 'cat_001_004', 'cat_001_005', 'cat_001_006', 'cat_001_007',
    'cat_002_001', 'cat_002_002', 'cat_002_003', 'cat_002_004', 'cat_002_005', 'cat_002_006', 'cat_002_007', 'cat_002_008', 'cat_002_009', 'cat_002_010', 'cat_002_011', 'cat_002_012', 'cat_002_013',
    'cat_003_001', 'cat_003_002', 'cat_003_003', 'cat_003_004', 'cat_003_005', 'cat_003_006', 'cat_003_007', 'cat_003_008', 'cat_003_009', 'cat_003_010', 'cat_003_011', 'cat_003_012', 'cat_003_013', 'cat_003_014',
    'cat_004_001', 'cat_004_002', 'cat_004_003', 'cat_004_004',
    'cat_005_001', 'cat_005_002', 'cat_005_003',
    'cat_006_001', 'cat_006_002', 'cat_006_003',
    'cat_007_001', 'cat_007_002', 'cat_007_003',
    'cat_008_001',
    'cat_009_001',
    'cat_010_001', 'cat_010_002',
    'cat_011_001',
    'cat_012_001', 'cat_012_002', 'cat_012_003'
  ]) as expected_code
),
missing_check AS (
  SELECT 
    expected_code,
    CASE WHEN EXISTS (SELECT 1 FROM taxonomy_nodes WHERE code = expected_code) 
         THEN 'EXISTS' ELSE 'MISSING' END as status
  FROM expected_subcats
)
SELECT 
  status,
  COUNT(*) as count,
  CASE WHEN status = 'MISSING' 
       THEN STRING_AGG(expected_code, ', ' ORDER BY expected_code)
       ELSE 'All expected subcategories present' 
  END as details
FROM missing_check 
GROUP BY status;

\echo ''
\echo '8. DATABASE SUMMARY:'
\echo '==================='
SELECT 
    'taxonomy_nodes' as table_name,
    COUNT(*) as total_records,
    COUNT(*) FILTER (WHERE level = 0) as main_categories,
    COUNT(*) FILTER (WHERE level = 1) as subcategories,
    COUNT(*) FILTER (WHERE is_active = true) as active_records
FROM taxonomy_nodes
UNION ALL
SELECT 
    'synonym_lemmas' as table_name,
    COUNT(*) as total_records,
    COUNT(*) FILTER (WHERE is_verified = true) as verified,
    COUNT(*) FILTER (WHERE is_active = true) as active,
    COUNT(*) FILTER (WHERE category_id IS NOT NULL) as with_category
FROM synonym_lemmas
UNION ALL
SELECT 
    'synonym_terms' as table_name,
    COUNT(*) as total_records,
    COUNT(*) FILTER (WHERE is_primary = true) as primary_terms,
    COUNT(*) FILTER (WHERE is_verified = true) as verified,
    COUNT(*) FILTER (WHERE confidence_score >= 0.8) as high_confidence
FROM synonym_terms
UNION ALL
SELECT 
    'keyword_rules' as table_name,
    COUNT(*) as total_records,
    COUNT(*) FILTER (WHERE is_active = true) as active,
    COUNT(*) FILTER (WHERE confidence_score >= 0.8) as high_confidence,
    COUNT(*) FILTER (WHERE priority >= 8) as high_priority
FROM keyword_rules
UNION ALL
SELECT 
    'regex_rules' as table_name,
    COUNT(*) as total_records,
    COUNT(*) FILTER (WHERE is_active = true) as active,
    COUNT(*) FILTER (WHERE confidence_score >= 0.8) as high_confidence,
    COUNT(*) FILTER (WHERE priority >= 5) as high_priority
FROM regex_rules;

\echo ''
\echo '=== END OF REPORT ==='
