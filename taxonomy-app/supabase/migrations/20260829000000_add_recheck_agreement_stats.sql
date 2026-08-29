-- Recheck agreement, computed in the database (2026-08-29)
--
-- The dashboard needs to show how often the AI's category matches the human
-- one on the 3,103 legacy products. Before this, the only "accuracy" on
-- screen was the 99.8% hardcoded into /reports, which was never measured
-- against anything.
--
-- Doing this from the client would mean pulling 3,103 suggestion rows plus
-- their products on every dashboard load just to compare two uuids, so it is
-- a function instead: two counts, one round trip.
--
-- SECURITY INVOKER (the default) on purpose — the caller's RLS still applies
-- to both tables, so this cannot be used to read past a policy.

CREATE OR REPLACE FUNCTION recheck_agreement_stats()
RETURNS TABLE (total BIGINT, agreed BIGINT)
LANGUAGE sql
STABLE
AS $$
    SELECT
        COUNT(*) AS total,
        COUNT(*) FILTER (WHERE s.suggested_category_id = p.category_id) AS agreed
    FROM product_category_suggestions s
    JOIN products p ON p.id = s.product_id
    WHERE s.suggestion_method = 'recheck_legacy';
$$;

COMMENT ON FUNCTION recheck_agreement_stats() IS
    'จำนวนรายการที่ AI ตรวจซ้ำ และจำนวนที่ AI เห็นตรงกับหมวดที่คนจัดไว้ (suggestion_method = recheck_legacy)';

-- อ่านอย่างเดียวและไม่รับพารามิเตอร์ แต่ยังไม่ให้ anon เรียก
-- ตามแนวเดียวกับ 20260828000000_fix_rls_gaps.sql ที่ปิดสิทธิ์ anon ไว้ก่อน
REVOKE ALL ON FUNCTION recheck_agreement_stats() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION recheck_agreement_stats() TO authenticated, service_role;
