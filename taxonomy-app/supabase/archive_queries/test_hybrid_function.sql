-- Test hybrid classification function directly
SELECT * FROM hybrid_category_classification(
  'กรรไกรตัดหญ้า',
  '[0.1, 0.2, 0.3]'::vector(384),  -- dummy embedding
  3
);
