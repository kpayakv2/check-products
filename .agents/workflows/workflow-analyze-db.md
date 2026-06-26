# 📊 Database Analysis Workflow

## Description
ใช้สำหรับดึงโครงสร้างฐานข้อมูล (Schema) จาก Supabase Local มาวิเคราะห์และสร้างไฟล์สรุปข้อมูลเพื่อให้ AI เข้าใจความสัมพันธ์ของตารางต่างๆ

## Steps

0. **Socraticode Graph Analysis (ก่อน Dump Schema)**:
   - Build dependency graph เพื่อ map ไฟล์ที่เกี่ยวกับ DB ทั้งหมด:
     ```
     codebase_graph_build { projectPath: "d:\\product_checker\\check-products" }
     ```
   - ดูสถิติ graph หลัง build เสร็จ:
     ```
     codebase_graph_stats { projectPath: "d:\\product_checker\\check-products" }
     ```
   - ตรวจสอบ context artifacts ที่มีอยู่แล้ว (อาจไม่ต้อง dump ใหม่):
     ```
     codebase_context { projectPath: "d:\\product_checker\\check-products" }
     ```

1. **Dump Schema**: รันคำสั่ง `npx supabase db dump --local > schema.sql` เพื่อดึง SQL ล่าสุด

2. **Read File**: อ่านเนื้อหาในไฟล์ `schema.sql`

3. **Analyze**: วิเคราะห์โครงสร้างตาราง, Primary Keys, และ Foreign Keys

4. **Generate Summary**: สร้างหรืออัปเดตไฟล์ `DATABASE_SCHEMA.md` เพื่อสรุปความสัมพันธ์ของข้อมูล

5. **Update Types**:
   - ตรวจสอบว่า `db-types.ts` (ถ้ามี) สอดคล้องกับ Schema ใหม่หรือไม่
   - ค้นหาไฟล์ทั้งหมดที่ import DB types เพื่อตรวจว่าต้องอัปเดต:
     ```
     codebase_symbols { query: "Database" }
     codebase_search { query: "import db types supabase schema" }
     ```
