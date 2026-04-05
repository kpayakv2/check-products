# 🆕 New Feature Workflow

## Description
ขั้นตอนมาตรฐานสำหรับการพัฒนาฟีเจอร์ใหม่ เพื่อให้มั่นใจว่าโค้ดมีคุณภาพและมีการทดสอบอย่างครบถ้วน

## Steps
1. **Branch Out**: สร้าง Branch ใหม่ `feature/[ชื่อฟีเจอร์]`
2. **Draft Design**: สร้างเอกสารดีไซน์เบื้องต้น (หรืออัปเดต ARCHITECTURE.md)
3. **Implementation**:
   - เขียนโค้ดตามกฎใน `.windsurf/rules/`
   - ตรวจสอบว่าได้ใช้ Component ที่มีอยู่แล้ว (Reuse) หรือไม่
4. **Testing**:
   - สร้าง Unit Test ไฟล์ใหม่
   - รัน `npm test` หรือ `pytest` (ตามภาษาที่ใช้)
5. **Browser Verification (Antigravity)**:
   - ตรวจสอบ UI ผ่านเบราว์เซอร์ตาม `workflow-antigravity-verification.md`
   - ตรวจสอบการแสดงผลภาษาไทยและความ Responsive
6. **Linting**: รันคำสั่ง Lint เพื่อเช็คคุณภาพโค้ด
7. **PR Creation**: เปิด Pull Request และสรุปการเปลี่ยนแปลงทั้งหมด
