 -- 1. สร้าง Bucket ชื่อ 'uploads' (ถ้ายังไม่มี)
     INSERT INTO storage.buckets (id, name, public) 
     VALUES ('uploads', 'uploads', true)
     ON CONFLICT (id) DO NOTHING;
    
     -- 2. ปลดล็อก RLS ให้ใครก็ได้ (Public) สามารถอัปโหลดไฟล์เข้าถังนี้ได้ (สำหรับ Dev Mode)
     DROP POLICY IF EXISTS "Allow Public Upload" ON storage.objects;
     CREATE POLICY "Allow Public Upload" ON storage.objects
     FOR INSERT TO public WITH CHECK (bucket_id = 'uploads');
   
    DROP POLICY IF EXISTS "Allow Public Select" ON storage.objects;
    CREATE POLICY "Allow Public Select" ON storage.objects
    FOR SELECT TO public USING (bucket_id = 'uploads');