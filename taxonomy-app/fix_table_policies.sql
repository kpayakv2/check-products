-- ===================================
-- RLS Policies for Development Mode
-- Allow all operations without authentication
-- ===================================

-- 1. IMPORTS table policies
DROP POLICY IF EXISTS "Allow public insert on imports" ON public.imports;
DROP POLICY IF EXISTS "Allow public select on imports" ON public.imports;
DROP POLICY IF EXISTS "Allow public update on imports" ON public.imports;
DROP POLICY IF EXISTS "Allow public delete on imports" ON public.imports;

CREATE POLICY "Allow public insert on imports"
ON public.imports FOR INSERT
TO public
WITH CHECK (true);

CREATE POLICY "Allow public select on imports"
ON public.imports FOR SELECT
TO public
USING (true);

CREATE POLICY "Allow public update on imports"
ON public.imports FOR UPDATE
TO public
USING (true);

CREATE POLICY "Allow public delete on imports"
ON public.imports FOR DELETE
TO public
USING (true);

-- 2. PRODUCT_CATEGORY_SUGGESTIONS table policies
DROP POLICY IF EXISTS "Allow public insert on product_category_suggestions" ON public.product_category_suggestions;
DROP POLICY IF EXISTS "Allow public select on product_category_suggestions" ON public.product_category_suggestions;
DROP POLICY IF EXISTS "Allow public update on product_category_suggestions" ON public.product_category_suggestions;
DROP POLICY IF EXISTS "Allow public delete on product_category_suggestions" ON public.product_category_suggestions;

CREATE POLICY "Allow public insert on product_category_suggestions"
ON public.product_category_suggestions FOR INSERT
TO public
WITH CHECK (true);

CREATE POLICY "Allow public select on product_category_suggestions"
ON public.product_category_suggestions FOR SELECT
TO public
USING (true);

CREATE POLICY "Allow public update on product_category_suggestions"
ON public.product_category_suggestions FOR UPDATE
TO public
USING (true);

CREATE POLICY "Allow public delete on product_category_suggestions"
ON public.product_category_suggestions FOR DELETE
TO public
USING (true);

-- 3. PRODUCT_ATTRIBUTES table policies
DROP POLICY IF EXISTS "Allow public insert on product_attributes" ON public.product_attributes;
DROP POLICY IF EXISTS "Allow public select on product_attributes" ON public.product_attributes;
DROP POLICY IF EXISTS "Allow public update on product_attributes" ON public.product_attributes;
DROP POLICY IF EXISTS "Allow public delete on product_attributes" ON public.product_attributes;

CREATE POLICY "Allow public insert on product_attributes"
ON public.product_attributes FOR INSERT
TO public
WITH CHECK (true);

CREATE POLICY "Allow public select on product_attributes"
ON public.product_attributes FOR SELECT
TO public
USING (true);

CREATE POLICY "Allow public update on product_attributes"
ON public.product_attributes FOR UPDATE
TO public
USING (true);

CREATE POLICY "Allow public delete on product_attributes"
ON public.product_attributes FOR DELETE
TO public
USING (true);

-- 4. REVIEW_HISTORY table policies
DROP POLICY IF EXISTS "Allow public insert on review_history" ON public.review_history;
DROP POLICY IF EXISTS "Allow public select on review_history" ON public.review_history;

CREATE POLICY "Allow public insert on review_history"
ON public.review_history FOR INSERT
TO public
WITH CHECK (true);

CREATE POLICY "Allow public select on review_history"
ON public.review_history FOR SELECT
TO public
USING (true);

-- 5. SIMILARITY_MATCHES table policies
DROP POLICY IF EXISTS "Allow public insert on similarity_matches" ON public.similarity_matches;
DROP POLICY IF EXISTS "Allow public select on similarity_matches" ON public.similarity_matches;

CREATE POLICY "Allow public insert on similarity_matches"
ON public.similarity_matches FOR INSERT
TO public
WITH CHECK (true);

CREATE POLICY "Allow public select on similarity_matches"
ON public.similarity_matches FOR SELECT
TO public
USING (true);

-- Verify all policies
SELECT schemaname, tablename, policyname, roles, cmd
FROM pg_policies 
WHERE tablename IN (
  'imports', 
  'products', 
  'product_category_suggestions', 
  'product_attributes',
  'review_history',
  'similarity_matches'
)
ORDER BY tablename, policyname;
