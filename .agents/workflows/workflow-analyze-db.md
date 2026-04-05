# 📊 Database Analysis Workflow

## Description
ใช้สำหรับดึงโครงสร้างฐานข้อมูล (Schema) จาก Supabase Local มาวิเคราะห์และสร้างไฟล์สรุปข้อมูลเพื่อให้ AI เข้าใจความสัมพันธ์ของตารางต่างๆ

## Steps
1. **Dump Schema**: รันคำสั่ง `npx supabase db dump --local > schema.sql` เพื่อดึง SQL ล่าสุด
2. **Read File**: อ่านเนื้อหาในไฟล์ `schema.sql`
3. **Analyze**: วิเคราะห์โครงสร้างตาราง, Primary Keys, และ Foreign Keys
4. **Generate Summary**: สร้างหรืออัปเดตไฟล์ `DATABASE_SCHEMA.md` เพื่อสรุปความสัมพันธ์ของข้อมูล
5. **Update Types**: ตรวจสอบว่า `db-types.ts` (ถ้ามี) สอดคล้องกับ Schema ใหม่หรือไม่
