---
name: rules-antigravity
description: |
  กฎการตรวจสอบ UI ผ่านเบราว์เซอร์ (Antigravity Visual Integrity)
  ใช้เมื่อมีการแก้ไข Frontend, UI Components, หรือต้องการทดสอบ Integration

triggers:
  - แก้ไขไฟล์ใน taxonomy-app/
  - แก้ไข page.tsx, components/, globals.css
  - งานที่เกี่ยวกับ UI, layout, หรือ responsive design
  - ต้องการตรวจสอบการแสดงผลภาษาไทย
  - รันหลัง workflow-new-feature.md
---

# 🚀 Antigravity Browser Verification Rules

## Context
ใช้เมื่อมีการแก้ไข Frontend (Next.js/Tailwind), การพัฒนา UI Components, หรือการทดสอบ Integration ระหว่าง Web และ AI Backend

## Standards
- **Browser-in-the-loop**: ทุกครั้งที่มีการเปลี่ยนแปลง UI ที่สำคัญ เอเจนต์ต้องทำการตรวจสอบผ่านเบราว์เซอร์ (Headless หรือ Managed)
- **Visual Consistency**:
  - ตรวจสอบการแสดงผลภาษาไทย (สระจม/ลอย, ฟอนต์)
  - ตรวจสอบความสวยงามของ Layout และความ Responsive (Mobile 375px / Desktop 1280px)
- **Error Monitoring**: ต้องตรวจสอบ Console Log และ Network Tab (API Status) เสมอ
- **Artifact Generation**: การรายงานผลต้องแนบหลักฐาน เช่น Screenshot หรือ Log จาก Playwright

## Audit Checklist
- [ ] Mobile (375px): ไม่มี Horizontal Scroll
- [ ] Console: ไม่มี `Failed to fetch` errors
- [ ] ภาษาไทย: สระไม่จม/ลอย, ข้อความยาวมี `truncate` หรือ `break-words`
- [ ] Supabase: ต่อได้จริง (ไม่ใช่ Mock)

## Examples

### ✅ Good: การตรวจสอบหลังแก้ UI
1. แก้ไขโค้ดใน `components/ProductCard.tsx`
2. รัน `npx playwright test e2e/antigravity-specialist.spec.ts`
3. แคปภาพหน้าจอมาเปรียบเทียบกับดีไซน์

### ❌ Bad: การแก้โค้ดโดยไม่รันเซิร์ฟเวอร์ดูผลลัพธ์
"แก้ไขโค้ดเรียบร้อยแล้วครับ น่าจะทำงานได้ปกติ" (โดยไม่มีการรันจริง)
