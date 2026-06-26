# 🆕 New Feature Workflow

## Description
ขั้นตอนมาตรฐานสำหรับการพัฒนาฟีเจอร์ใหม่ เพื่อให้มั่นใจว่าโค้ดมีคุณภาพและมีการทดสอบอย่างครบถ้วน

## Steps

1. **Branch Out**: สร้าง Branch ใหม่ `feature/[ชื่อฟีเจอร์]`

2. **Draft Design**: สร้างเอกสารดีไซน์เบื้องต้น (หรืออัปเดต `docs/` หรือ `ARCHITECTURE.md`)

3. **Socraticode Exploration (ก่อนลงมือเสมอ)**:
   - ค้นหาโค้ดที่เกี่ยวข้อง:
     ```
     codebase_search { query: "คำอธิบาย feature ที่จะทำ" }
     ```
   - ดู symbols ในไฟล์ที่จะแก้:
     ```
     codebase_symbols { file: "ไฟล์ที่เกี่ยวข้อง" }
     ```
   - Trace execution flow ของ feature ที่เกี่ยวข้อง:
     ```
     codebase_flow { entrypoint: "ชื่อ function entry point" }
     ```

4. **Impact Analysis**: วิเคราะห์ผลกระทบ **ทั้งสองระดับ** ก่อนลงมือแก้:
   - **โค้ด Blast Radius:**
     ```
     codebase_impact { target: "ไฟล์หรือ symbol ที่จะแก้" }
     ```
   - **ข้อมูล:** รัน SQL เช็คว่าการเปลี่ยนแปลงจะกระทบสินค้ากี่รายการ (ดู `smart_impact_workflow.md`)

5. **Implementation**:
   - เขียนโค้ดตามกฎใน `.agents/rules/`
   - ก่อนเรียกใช้ function ที่ไม่คุ้นเคย → ดู signature ด้วย:
     ```
     codebase_symbol { name: "ชื่อ function" }
     ```
   - ตรวจสอบว่าได้ใช้ Component ที่มีอยู่แล้ว (Reuse) หรือไม่
   - Python: ใช้ `ThaiTextProcessor` จาก `fresh_implementations.py`
   - TypeScript: กำหนด Interface ให้ชัดเจน ห้ามใช้ `any`

6. **Testing**:
   - สร้าง Unit Test ไฟล์ใหม่
   - รัน `npm test` หรือ `pytest` (ตามภาษาที่ใช้)
   - ตรวจสอบค่า F1-score/Accuracy ≥ 72%

7. **Browser Verification (Antigravity)**:
   - ตรวจสอบ UI ผ่านเบราว์เซอร์ตาม `.agents/workflows/workflow-antigravity-verification.md`
   - ตรวจสอบการแสดงผลภาษาไทยและความ Responsive (375px / 1280px)

8. **Linting & Circular Dep Check**:
   - รันคำสั่ง Lint เพื่อเช็คคุณภาพโค้ด
   - ตรวจ Circular Dependencies:
     ```
     codebase_graph_circular { projectPath: "d:\\product_checker\\check-products" }
     ```

9. **Update Docs**: อัปเดต `CURRENT_STATUS.md` หลังจบงาน

10. **PR Creation**: เปิด Pull Request และสรุปการเปลี่ยนแปลงทั้งหมด
