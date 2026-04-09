DROP POLICY IF EXISTS "Allow public insert on synonym_lemmas" ON public.synonym_lemmas;
DROP POLICY IF EXISTS "Allow public select on synonym_lemmas" ON public.synonym_lemmas;
DROP POLICY IF EXISTS "Allow public update on synonym_lemmas" ON public.synonym_lemmas;
 DROP POLICY IF EXISTS "Allow public delete on synonym_lemmas" ON public.synonym_lemmas;

CREATE POLICY "Allow public insert on synonym_lemmas" ON public.synonym_lemmas FOR INSERT TO
public WITH CHECK (true);
CREATE POLICY "Allow public select on synonym_lemmas" ON public.synonym_lemmas FOR SELECT TO
public USING (true);
CREATE POLICY "Allow public update on synonym_lemmas" ON public.synonym_lemmas FOR UPDATE TO
 public USING (true);
 CREATE POLICY "Allow public delete on synonym_lemmas" ON public.synonym_lemmas FOR DELETE TO
 public USING (true);

-- 2. synonym_terms table policies
 DROP POLICY IF EXISTS "Allow public insert on synonym_terms" ON public.synonym_terms;
 DROP POLICY IF EXISTS "Allow public select on synonym_terms" ON public.synonym_terms;        
 DROP POLICY IF EXISTS "Allow public delete on synonym_terms" ON public.synonym_terms;        

 CREATE POLICY "Allow public insert on synonym_terms" ON public.synonym_terms FOR INSERT TO   
 public WITH CHECK (true);
   CREATE POLICY "Allow public select on synonym_terms" ON public.synonym_terms FOR SELECT TO   
 public USING (true);
   CREATE POLICY "Allow public delete on synonym_terms" ON public.synonym_terms FOR DELETE TO   
public USING (true);