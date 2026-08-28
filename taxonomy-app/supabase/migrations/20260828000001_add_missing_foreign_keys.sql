-- Add missing foreign keys found in ER/schema audit (2026-08-28)
--
-- Verified live against local DB before writing this migration:
--   - products.import_batch_id: 0 orphaned rows (3,106 products, all
--     reference an existing imports row across 2 batches) — safe to add
--     with a plain ADD CONSTRAINT, no backfill needed.
--   - Every *_by / *_id attribution column below is 100% NULL across all
--     12 affected tables (auth.users itself has 0 rows locally) — safe to
--     add with no backfill needed.

-- ============================================================
-- products.import_batch_id -> imports.id
-- RESTRICT (explicit, not relying on the implicit default): products are
-- the core business asset and must never silently disappear or lose
-- lineage because an import record was cleaned up. RESTRICT forces a
-- conscious reassignment/decision before an import batch can be deleted
-- while products still reference it.
-- ============================================================
ALTER TABLE products
    ADD CONSTRAINT products_import_batch_id_fkey
    FOREIGN KEY (import_batch_id) REFERENCES imports(id) ON DELETE RESTRICT;

-- ============================================================
-- *_by / *_id -> auth.users(id), all ON DELETE SET NULL
--
-- Deliberate deviation from the 4 existing auth.users FKs in this schema
-- (regex_rules.created_by/updated_by, system_settings.updated_by,
-- audit_logs.user_id, human_feedback.reviewer_id), which all use the
-- default NO ACTION (blocks deleting a user who ever touched a row).
-- These 15 columns are pure attribution/audit metadata on otherwise-valid
-- business data (products, rules, review history, similarity matches) —
-- NO ACTION here would make it impossible to ever delete/offboard a
-- Supabase auth user who touched even one row across 11 tables. SET NULL
-- anonymizes the "who did this" pointer without touching the row itself.
-- ============================================================

ALTER TABLE products
    ADD CONSTRAINT products_reviewed_by_fkey
    FOREIGN KEY (reviewed_by) REFERENCES auth.users(id) ON DELETE SET NULL;
ALTER TABLE products
    ADD CONSTRAINT products_created_by_fkey
    FOREIGN KEY (created_by) REFERENCES auth.users(id) ON DELETE SET NULL;
ALTER TABLE products
    ADD CONSTRAINT products_updated_by_fkey
    FOREIGN KEY (updated_by) REFERENCES auth.users(id) ON DELETE SET NULL;

ALTER TABLE taxonomy_nodes
    ADD CONSTRAINT taxonomy_nodes_created_by_fkey
    FOREIGN KEY (created_by) REFERENCES auth.users(id) ON DELETE SET NULL;
ALTER TABLE taxonomy_nodes
    ADD CONSTRAINT taxonomy_nodes_updated_by_fkey
    FOREIGN KEY (updated_by) REFERENCES auth.users(id) ON DELETE SET NULL;

ALTER TABLE keyword_rules
    ADD CONSTRAINT keyword_rules_created_by_fkey
    FOREIGN KEY (created_by) REFERENCES auth.users(id) ON DELETE SET NULL;
ALTER TABLE keyword_rules
    ADD CONSTRAINT keyword_rules_updated_by_fkey
    FOREIGN KEY (updated_by) REFERENCES auth.users(id) ON DELETE SET NULL;

ALTER TABLE synonym_lemmas
    ADD CONSTRAINT synonym_lemmas_created_by_fkey
    FOREIGN KEY (created_by) REFERENCES auth.users(id) ON DELETE SET NULL;
ALTER TABLE synonym_lemmas
    ADD CONSTRAINT synonym_lemmas_updated_by_fkey
    FOREIGN KEY (updated_by) REFERENCES auth.users(id) ON DELETE SET NULL;

ALTER TABLE synonym_terms
    ADD CONSTRAINT synonym_terms_created_by_fkey
    FOREIGN KEY (created_by) REFERENCES auth.users(id) ON DELETE SET NULL;

ALTER TABLE imports
    ADD CONSTRAINT imports_created_by_fkey
    FOREIGN KEY (created_by) REFERENCES auth.users(id) ON DELETE SET NULL;

ALTER TABLE product_attributes
    ADD CONSTRAINT product_attributes_created_by_fkey
    FOREIGN KEY (created_by) REFERENCES auth.users(id) ON DELETE SET NULL;

ALTER TABLE product_category_suggestions
    ADD CONSTRAINT product_category_suggestions_reviewed_by_fkey
    FOREIGN KEY (reviewed_by) REFERENCES auth.users(id) ON DELETE SET NULL;

ALTER TABLE review_history
    ADD CONSTRAINT review_history_reviewer_id_fkey
    FOREIGN KEY (reviewer_id) REFERENCES auth.users(id) ON DELETE SET NULL;

ALTER TABLE similarity_matches
    ADD CONSTRAINT similarity_matches_reviewed_by_fkey
    FOREIGN KEY (reviewed_by) REFERENCES auth.users(id) ON DELETE SET NULL;
