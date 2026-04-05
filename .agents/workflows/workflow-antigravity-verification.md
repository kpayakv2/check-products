# 🔍 Antigravity Verification Workflow

## Description
ขั้นตอนการตรวจสอบการทำงานของระบบผ่านเบราว์เซอร์ เพื่อให้มั่นใจว่าผู้ใช้จะได้รับประสบการณ์ที่ดีที่สุด (Browser-in-the-loop Verification)

## Steps
1. **Prepare Environment**:
   - รัน Backend (`python api_server.py`)
   - รัน Frontend (`npm run dev` ใน `taxonomy-app`)
2. **Execution**:
   - ใช้ Playwright หรือเครื่องมือเบราว์เซอร์เพื่อเข้าถึง URL เป้าหมาย
   - จำลองพฤติกรรมผู้ใช้ (คลิก, พิมพ์ข้อมูลสินค้าไทย, อัปโหลดไฟล์)
3. **Inspection**:
   - ตรวจสอบว่า UI ตอบสนองตามที่คาดหวัง (เช่น Modal เด้งขึ้นมา, ตารางอัปเดตข้อมูล)
   - เช็ค Console สำหรับ Error (Red flags 🚩)
4. **Evidence Gathering**:
   - จับภาพหน้าจอ (Screenshots) ในจุดที่สำคัญ
   - บันทึก Network Response เพื่อดูว่า API ส่งข้อมูลมาถูกต้องไหม
5. **Refinement**:
   - หากพบปัญหา (เช่น ปุ่มกดไม่ติด, Layout พัง) ให้กลับไปแก้ไขโค้ดและเริ่มขั้นตอนที่ 1 ใหม่
6. **Summary**: สรุปผลการทดสอบพร้อมแนบหลักฐานประกอบการรายงาน (Screenshots/Logs)
