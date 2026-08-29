-- Let the dashboard actually call recheck_agreement_stats() (2026-08-29)
--
-- 20260829000000 granted EXECUTE to authenticated and service_role only.
-- Verified in a real browser: the dashboard is a client component and talks to
-- Supabase with the public anon key, so every call came back
-- "42501 permission denied for function recheck_agreement_stats" and the page
-- fell back to showing no figure at all.
--
-- Granting anon adds no exposure: the function is SECURITY INVOKER and only
-- aggregates products and product_category_suggestions, both of which already
-- have SELECT policies of USING (true) — anon can read every row it counts,
-- one row at a time, today. What it returns is two integers over data that is
-- already readable.

GRANT EXECUTE ON FUNCTION recheck_agreement_stats() TO anon;
