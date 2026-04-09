-- Fix RLS for imports table (local dev: allow anon access)
ALTER TABLE public.imports ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Allow public insert on imports" ON public.imports;
DROP POLICY IF EXISTS "Allow public select on imports" ON public.imports;
DROP POLICY IF EXISTS "Allow public update on imports" ON public.imports;
DROP POLICY IF EXISTS "Allow public delete on imports" ON public.imports;

CREATE POLICY "Allow public insert on imports"
  ON public.imports FOR INSERT TO public WITH CHECK (true);

CREATE POLICY "Allow public select on imports"
  ON public.imports FOR SELECT TO public USING (true);

CREATE POLICY "Allow public update on imports"
  ON public.imports FOR UPDATE TO public USING (true);

CREATE POLICY "Allow public delete on imports"
  ON public.imports FOR DELETE TO public USING (true);

-- Verify
SELECT policyname, roles, cmd FROM pg_policies WHERE tablename = 'imports' ORDER BY policyname;
