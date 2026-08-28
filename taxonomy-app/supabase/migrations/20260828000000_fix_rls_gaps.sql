-- Fix RLS gaps found in ER/schema audit (2026-08-28)
--
-- system_settings currently has RLS disabled entirely, with full
-- INSERT/SELECT/UPDATE/DELETE/TRUNCATE granted to anon (unauthenticated) —
-- verified live against local DB. This is the highest-severity item: anyone
-- with the public anon key can read/overwrite/wipe the AI provider, search,
-- and processing config the whole classification pipeline depends on.
-- Masked today only because the app exclusively uses the service-role key.
--
-- The other 7 tables have RLS enabled but zero policies — same masking
-- failure mode: the day anything talks to Supabase with a non-service-role
-- key, these silently return empty results or fail instead of erroring loudly.
--
-- Policy convention follows 20250924120000_init_hybrid_schema.sql:
--   SELECT: USING (true)                                     [public read]
--   INSERT/UPDATE: WITH CHECK / USING auth.role() IN ('taxonomy_editor','taxonomy_admin')
--   DELETE: USING auth.role() = 'taxonomy_admin'
--
-- Deviations, reasoned per-table:
--   product_attributes, imports: SELECT restricted to editor/admin — nothing
--     in the reachable app needs these publicly readable, least-privilege default.
--   system_settings: SELECT also restricted to editor/admin (not just write) —
--     this is the table the whole migration exists to lock down; leaving read
--     public would still expose AI-provider/threshold config to anon.
--     INSERT/UPDATE restricted to admin only (not editor+admin like everywhere
--     else) — a mis-set global config affects every product's classification,
--     higher blast radius than a single row elsewhere.

-- ============================================================
-- system_settings: RLS was never enabled at all
-- ============================================================
ALTER TABLE system_settings ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "system_settings_read" ON system_settings;
CREATE POLICY "system_settings_read" ON system_settings
    FOR SELECT USING (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));

DROP POLICY IF EXISTS "system_settings_insert" ON system_settings;
CREATE POLICY "system_settings_insert" ON system_settings
    FOR INSERT WITH CHECK (auth.role() = 'taxonomy_admin');

DROP POLICY IF EXISTS "system_settings_update" ON system_settings;
CREATE POLICY "system_settings_update" ON system_settings
    FOR UPDATE USING (auth.role() = 'taxonomy_admin');

DROP POLICY IF EXISTS "system_settings_delete" ON system_settings;
CREATE POLICY "system_settings_delete" ON system_settings
    FOR DELETE USING (auth.role() = 'taxonomy_admin');

-- ============================================================
-- keyword_rules — RLS enabled, 0 policies
-- ============================================================
DROP POLICY IF EXISTS "keyword_rules_read" ON keyword_rules;
CREATE POLICY "keyword_rules_read" ON keyword_rules FOR SELECT USING (true);

DROP POLICY IF EXISTS "keyword_rules_insert" ON keyword_rules;
CREATE POLICY "keyword_rules_insert" ON keyword_rules
    FOR INSERT WITH CHECK (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));

DROP POLICY IF EXISTS "keyword_rules_update" ON keyword_rules;
CREATE POLICY "keyword_rules_update" ON keyword_rules
    FOR UPDATE USING (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));

DROP POLICY IF EXISTS "keyword_rules_delete" ON keyword_rules;
CREATE POLICY "keyword_rules_delete" ON keyword_rules
    FOR DELETE USING (auth.role() = 'taxonomy_admin');

-- ============================================================
-- regex_rules — RLS enabled, 0 policies
-- ============================================================
DROP POLICY IF EXISTS "regex_rules_read" ON regex_rules;
CREATE POLICY "regex_rules_read" ON regex_rules FOR SELECT USING (true);

DROP POLICY IF EXISTS "regex_rules_insert" ON regex_rules;
CREATE POLICY "regex_rules_insert" ON regex_rules
    FOR INSERT WITH CHECK (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));

DROP POLICY IF EXISTS "regex_rules_update" ON regex_rules;
CREATE POLICY "regex_rules_update" ON regex_rules
    FOR UPDATE USING (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));

DROP POLICY IF EXISTS "regex_rules_delete" ON regex_rules;
CREATE POLICY "regex_rules_delete" ON regex_rules
    FOR DELETE USING (auth.role() = 'taxonomy_admin');

-- ============================================================
-- product_category_suggestions — RLS enabled, 0 policies
-- ============================================================
DROP POLICY IF EXISTS "product_category_suggestions_read" ON product_category_suggestions;
CREATE POLICY "product_category_suggestions_read" ON product_category_suggestions
    FOR SELECT USING (true);

DROP POLICY IF EXISTS "product_category_suggestions_insert" ON product_category_suggestions;
CREATE POLICY "product_category_suggestions_insert" ON product_category_suggestions
    FOR INSERT WITH CHECK (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));

DROP POLICY IF EXISTS "product_category_suggestions_update" ON product_category_suggestions;
CREATE POLICY "product_category_suggestions_update" ON product_category_suggestions
    FOR UPDATE USING (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));

DROP POLICY IF EXISTS "product_category_suggestions_delete" ON product_category_suggestions;
CREATE POLICY "product_category_suggestions_delete" ON product_category_suggestions
    FOR DELETE USING (auth.role() = 'taxonomy_admin');

-- ============================================================
-- product_attributes — RLS enabled, 0 policies
-- SELECT restricted (not public) — nothing in the reachable app needs it public
-- ============================================================
DROP POLICY IF EXISTS "product_attributes_read" ON product_attributes;
CREATE POLICY "product_attributes_read" ON product_attributes
    FOR SELECT USING (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));

DROP POLICY IF EXISTS "product_attributes_insert" ON product_attributes;
CREATE POLICY "product_attributes_insert" ON product_attributes
    FOR INSERT WITH CHECK (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));

DROP POLICY IF EXISTS "product_attributes_update" ON product_attributes;
CREATE POLICY "product_attributes_update" ON product_attributes
    FOR UPDATE USING (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));

DROP POLICY IF EXISTS "product_attributes_delete" ON product_attributes;
CREATE POLICY "product_attributes_delete" ON product_attributes
    FOR DELETE USING (auth.role() = 'taxonomy_admin');

-- ============================================================
-- similarity_matches — RLS enabled, 0 policies
-- ============================================================
DROP POLICY IF EXISTS "similarity_matches_read" ON similarity_matches;
CREATE POLICY "similarity_matches_read" ON similarity_matches FOR SELECT USING (true);

DROP POLICY IF EXISTS "similarity_matches_insert" ON similarity_matches;
CREATE POLICY "similarity_matches_insert" ON similarity_matches
    FOR INSERT WITH CHECK (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));

DROP POLICY IF EXISTS "similarity_matches_update" ON similarity_matches;
CREATE POLICY "similarity_matches_update" ON similarity_matches
    FOR UPDATE USING (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));

DROP POLICY IF EXISTS "similarity_matches_delete" ON similarity_matches;
CREATE POLICY "similarity_matches_delete" ON similarity_matches
    FOR DELETE USING (auth.role() = 'taxonomy_admin');

-- ============================================================
-- review_history — RLS enabled, 0 policies
-- ============================================================
DROP POLICY IF EXISTS "review_history_read" ON review_history;
CREATE POLICY "review_history_read" ON review_history FOR SELECT USING (true);

DROP POLICY IF EXISTS "review_history_insert" ON review_history;
CREATE POLICY "review_history_insert" ON review_history
    FOR INSERT WITH CHECK (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));

DROP POLICY IF EXISTS "review_history_update" ON review_history;
CREATE POLICY "review_history_update" ON review_history
    FOR UPDATE USING (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));

DROP POLICY IF EXISTS "review_history_delete" ON review_history;
CREATE POLICY "review_history_delete" ON review_history
    FOR DELETE USING (auth.role() = 'taxonomy_admin');

-- ============================================================
-- imports — RLS enabled, 0 policies
-- SELECT restricted (not public) — operational metadata (file names/sizes/
-- error details of supplier CSV uploads), not for public read
-- ============================================================
DROP POLICY IF EXISTS "imports_read" ON imports;
CREATE POLICY "imports_read" ON imports
    FOR SELECT USING (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));

DROP POLICY IF EXISTS "imports_insert" ON imports;
CREATE POLICY "imports_insert" ON imports
    FOR INSERT WITH CHECK (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));

DROP POLICY IF EXISTS "imports_update" ON imports;
CREATE POLICY "imports_update" ON imports
    FOR UPDATE USING (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));

DROP POLICY IF EXISTS "imports_delete" ON imports;
CREATE POLICY "imports_delete" ON imports
    FOR DELETE USING (auth.role() = 'taxonomy_admin');
