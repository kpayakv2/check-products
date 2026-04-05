# 🚀 Antigravity Browser Verification Rules

## Context
ใช้เมื่อมีการแก้ไข Frontend (Next.js/Tailwind), การพัฒนา UI Components, หรือการทดสอบ Integration ระหว่าง Web และ AI Backend

## Standards
- **Browser-in-the-loop**: ทุกครั้งที่มีการเปลี่ยนแปลง UI ที่สำคัญ เอเจนต์ต้องทำการตรวจสอบผ่านเบราว์เซอร์ (Headless หรือ Managed)
- **Visual Consistency**:
  - ตรวจสอบการแสดงผลภาษาไทย (สระจม/ลอย, ฟอนต์)
  - ตรวจสอบความสวยงามของ Layout และความ Responsive (Mobile/Desktop)
- **Error Monitoring**: ต้องตรวจสอบ Console Log และ Network Tab (API Status) เสมอ
- **Artifact Generation**: การรายงานผลต้องแนบหลักฐาน เช่น Screenshot หรือ Log จาก Playwright

## Examples

### ✅ Good: การตรวจสอบหลังแก้ UI
1. แก้ไขโค้ดใน `components/ProductCard.tsx`
2. รัน `npm run test:e2e` หรือสั่งให้เอเจนต์เปิดเบราว์เซอร์ดูหน้าจอ
3. แคปภาพหน้าจอมาเปรียบเทียบกับดีไซน์

### ❌ Bad: การแก้โค้ดโดยไม่รันเซิร์ฟเวอร์ดูผลลัพธ์
"แก้ไขโค้ดเรียบร้อยแล้วครับ น่าจะทำงานได้ปกติ" (โดยไม่มีการรันจริง)
