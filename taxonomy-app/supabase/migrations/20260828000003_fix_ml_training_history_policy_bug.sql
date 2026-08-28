-- Fix ml_training_history RLS policy bug found in ER/schema audit (2026-08-28)
--
-- The INSERT/DELETE policies' own comments say "เขียนได้เฉพาะ service_role"
-- ("write allowed only for service_role") but the predicates are USING(true)
-- / WITH CHECK(true) — service_role bypasses RLS entirely (policies aren't
-- evaluated for it at all), so the correct way to express "service_role
-- only" for the roles RLS *does* apply to (anon, authenticated) is false,
-- not true. The current true predicate actually lets anon/authenticated
-- insert and delete training-history rows, contradicting the stated intent.
--
-- ml_training_history_read_all (SELECT USING(true)) is untouched — its
-- comment ("อ่านได้สาธารณะ" / publicly readable) matches its predicate.

DROP POLICY IF EXISTS "ml_training_history_insert_service" ON ml_training_history;
CREATE POLICY "ml_training_history_insert_service" ON ml_training_history
    FOR INSERT WITH CHECK (false);

DROP POLICY IF EXISTS "ml_training_history_delete_service" ON ml_training_history;
CREATE POLICY "ml_training_history_delete_service" ON ml_training_history
    FOR DELETE USING (false);
