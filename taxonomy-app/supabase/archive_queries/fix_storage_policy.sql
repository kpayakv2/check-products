-- Drop existing restrictive policy
DROP POLICY IF EXISTS "Allow authenticated uploads" ON storage.objects;

-- Allow all operations for development (no auth required)
CREATE POLICY "Allow public insert for development"
ON storage.objects FOR INSERT
TO public
WITH CHECK (bucket_id = 'uploads');

CREATE POLICY "Allow public select for development"
ON storage.objects FOR SELECT
TO public
USING (bucket_id = 'uploads');

CREATE POLICY "Allow public delete for development"
ON storage.objects FOR DELETE
TO public
USING (bucket_id = 'uploads');

-- Verify policies
SELECT policyname, roles, cmd FROM pg_policies 
WHERE tablename = 'objects' AND schemaname = 'storage';
