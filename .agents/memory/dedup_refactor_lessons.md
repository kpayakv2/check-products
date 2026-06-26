# 🧠 Memory MCP: บทเรียนการ Refactor ระบบ Deduplication & Bug Fixes
*บันทึกข้อมูล ณ วันที่: 2026-05-31 (พยัคฆ์)*

## 1. การ Refactor & จัดขอบเขตของ Component (Separation of Concerns)
* **ปัญหาดั้งเดิม**: หน้าจอ `/data-quality` ในแท็บ Deduplication (`DeduplicationTab.tsx`) มีทั้งโหมดอัปโหลดไฟล์เทียบ CSV และโหมดสแกนคลังสินค้าเดิมภายในระบบ (ซึ่งโหมดอัปโหลดไฟล์เทียบ CSV นั้นไปทับซ้อนและพึ่งพา Supabase Edge Function ชุดเก่า และทำงานซ้ำซ้อนกับหน้าจอหลักของ Import Wizard `/import`)
* **บทเรียนการแก้ไข**: 
  - การลบโหมดที่ไม่จำเป็นออกทั้งหมด (โหมดอัปโหลดไฟล์ CSV เก่า) ทำให้ระบบ Next.js คลีนขึ้น และผู้ใช้งานไม่สับสน
  - การขจัดระบบจำลอง SQLite ท้องถิ่น (`human_feedback.db`, `HumanFeedbackDatabase` handler) ทำให้อุปกรณ์พึ่งพา Supabase single source of truth โดยสมบูรณ์ ไม่มี dead file และ dead sqlite connections หลุดค้าง
  - **สำคัญมาก**: เมื่อทำการลบโค้ดหรือคลาสเก่าที่ไม่มีประโยชน์ใน Backend (เช่น ใน `human_feedback_system.py`) **ต้องไม่ลบคลาสโมเดลหลักที่ระบบข้างเคียงยังจำเป็นต้องอ้างอิง** ได้แก่ `FeedbackType` (Enum) และ `ProductComparison` (Dataclass) เพราะระบบ ML Training (`ml_feedback_learning.py`) และ API endpoints ยังใช้งานคลาสเหล่านี้ในการรับส่งและวิเคราะห์ข้อมูลอยู่

---

## 2. การจัดการ Lucide React Icons ใน Next.js
* **ข้อผิดพลาดพบบ่อย**: `ReferenceError: ZapIcon is not defined`
* **สาเหตุ**: ระหว่างการทำความสะอาด imports ด้านบนของไฟล์ `DeduplicationTab.tsx` ตัวแปร `ZapIcon` ได้ถูกลบออกจากกลุ่ม imports ของ `lucide-react` แต่มีปุ่มสแกนในส่วนควบคุมที่ยังคงอ้างอิงตัวแปรดังกล่าวอยู่
* **บทเรียนการแก้ไข**:
  - ตรวจสอบรายการเรียกใช้งานตัวแปรภาพ (Icons) ใน JSX เสมอก่อนตัดสินใจล้าง imports ออก
  - ในไลบรารี `lucide-react` รุ่นใหม่ ชื่อมาตรฐานของสายฟ้าคือ `Zap` แทนที่จะใช้คำว่า `ZapIcon` (ห้อยท้าย Icon มักพบในไลบรารีของบางแพลตฟอร์ม แต่ใน Lucide มาตรฐานจะส่งออกแบบไม่มีคำว่า Icon เช่น `Zap`, `Check`, `CheckCircle` ...)
  - การเปลี่ยนมานำเข้าและใช้งาน `Zap` แทน `ZapIcon` เป็นการแก้ปัญหาที่ตรงจุดและมีเสถียรภาพที่สุด

---

## 3. ปัญหาการรัน Python Print และการเข้ารหัส (Encoding) บน Windows Terminal
* **ข้อผิดพลาดพบบ่อย**: `UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f527' in position 2: character maps to <undefined>`
* **สาเหตุ**: เมื่อเราเขียนฟังก์ชัน `print()` ในสคริปต์ Python ที่รันบน Windows (PowerShell/cmd) และในประโยคนั้นมี **High-Unicode Characters** เช่น Emojis แฟนซี (🔧, ❌, 🔴, 🟡) สคริปต์จะพังด้วยข้อผิดพลาด UnicodeEncodeError เนื่องจากหน้าจอ Terminal บนระบบปฏิบัติการ Windows ภาษาไทยมักจะเข้ารหัสแบบ CP874 ซึ่งไม่รองรับ Emojis เหล่านั้น
* **บทเรียนการแก้ไข**:
  - หลีกเลี่ยงการใช้ Emoji แฟนซีในข้อความพิมพ์ (`print`) ของตัวทดสอบ Unit/Integration Tests หรือ CLI scripts ที่อาจนำไปรันบน Windows Environment
  - ใช้ข้อความแบบ ASCII-Safe/English หรือข้อความธรรมดาที่ปลอดภัย (เช่น `[INFO]`, `[ERROR]`, `[SUCCESS]`) เพื่อให้มั่นใจว่าชุดทดสอบสามารถรันผ่าน 100% ในทุก ๆ สภาพแวดล้อมโดยไม่มีปัญหา Encoding พังระบบทดสอบ
