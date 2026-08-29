-- กันการบันทึกซ้ำจาก Import Wizard ที่ระดับฐานข้อมูล
--
-- ฝั่ง API เช็คก่อนแล้วว่ารอบนี้เคยบันทึกหรือยัง แต่ลำพังการเช็คก่อน insert ยังแข่งกันได้
-- เพราะระหว่างนั้นมีการขอ embedding จาก FastAPI คั่นอยู่ซึ่งกินเวลาหลายวินาที
-- ผู้ใช้ที่กดย้อน step แล้วเดินหน้าใหม่ในช่วงนั้นจะสร้าง batch ซ้ำได้ทัน
--
-- index บังคับเฉพาะแถวที่มีคีย์นี้ ของเดิมที่นำเข้าด้วยวิธีอื่นจึงไม่กระทบ
create unique index if not exists imports_wizard_run_id_key
  on public.imports ((metadata->>'wizard_run_id'))
  where metadata ? 'wizard_run_id';
