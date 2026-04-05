# 🛰️ Workflow: Antigravity UI Verification

## 🎯 Objective
ตรวจสอบความเป๊ะของ UI ในระดับพิกเซล โดยเฉพาะภาษาไทยและการล้น (Overflow) ในหน้าจอขนาดต่างๆ

## 🔄 Steps
1. **Trigger:** ทุกครั้งที่มีการแก้ไข `page.tsx`, `components/`, หรือ `globals.css`
2. **Action:** รัน Specialist ผ่าน Playwright
   ```bash
   npx playwright test e2e/antigravity-specialist.spec.ts
   ```
3. **Audit Criteria:**
   - **Mobile (375px):** ต้องไม่เกิด Horizontal Scroll (Zero Overflow)
   - **Console:** ต้องไม่มี Error `Failed to fetch` (ต้องต่อ Supabase ติดจริง)
   - **Thai Text:** สระต้องไม่จม/ลอย และข้อความยาวๆ ต้องถูกคุมด้วย `truncate` หรือ `break-words`
4. **Result:** หาก FAIL ต้องกลับไปแก้ไข Layout จนกว่า Specialist จะให้ PASS
