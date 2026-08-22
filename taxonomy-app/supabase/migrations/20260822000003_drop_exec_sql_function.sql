-- Security fix: remove exec_sql, an unauthenticated SECURITY DEFINER function
-- that executed arbitrary caller-supplied SQL with service_role privileges.
-- The corresponding exec-sql edge function has been deleted; confirmed unused
-- in production (only ad-hoc dev test scripts referenced it).

DROP FUNCTION IF EXISTS public.exec_sql(TEXT, JSONB);
