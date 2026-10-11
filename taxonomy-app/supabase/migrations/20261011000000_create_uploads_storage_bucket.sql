-- Migration: 20261011000000_create_uploads_storage_bucket.sql
-- Create 'uploads' storage bucket for file imports and downloads (F-024)

INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'uploads',
    'uploads',
    false,
    52428800, -- 50 MB
    ARRAY['text/csv', 'text/plain', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']
)
ON CONFLICT (id) DO UPDATE SET
    file_size_limit = EXCLUDED.file_size_limit,
    allowed_mime_types = EXCLUDED.allowed_mime_types;

-- Enable public read access for objects in the 'uploads' bucket
DROP POLICY IF EXISTS "Allow read on uploads bucket" ON storage.objects;
CREATE POLICY "Allow read on uploads bucket"
ON storage.objects
FOR SELECT
TO public
USING (bucket_id = 'uploads');

-- Enable public insert access for objects in the 'uploads' bucket
DROP POLICY IF EXISTS "Allow upload to uploads bucket" ON storage.objects;
CREATE POLICY "Allow upload to uploads bucket"
ON storage.objects
FOR INSERT
TO public
WITH CHECK (bucket_id = 'uploads');
