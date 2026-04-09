 DROP POLICY IF EXISTS "Allow public insert on products" ON public.products;
  DROP POLICY IF EXISTS "Allow public select on products" ON public.products;
     DROP POLICY IF EXISTS "Allow public update on products" ON public.products;
     DROP POLICY IF EXISTS "Allow public delete on products" ON public.products;
    
     CREATE POLICY "Allow public insert on products" ON public.products FOR INSERT TO public WITH CHECK
      (true);
    CREATE POLICY "Allow public select on products" ON public.products FOR SELECT TO public USING
      (true);
     CREATE POLICY "Allow public update on products" ON public.products FOR UPDATE TO public USING
      (true);
    CREATE POLICY "Allow public delete on products" ON public.products FOR DELETE TO public USING
      (true);