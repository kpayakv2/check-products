# แผนแก้ไข WCAG 2.2 AA — มอบให้ Antigravity ทำทั้งหมด

**ที่มา:** รายงาน "ผลตรวจหน้าจอระบบตามมาตรฐาน WCAG 2.2 ระดับ AA" (Claude Design canvas, 31 ส.ค. 2026) — ตรวจ 7 หน้าของระบบจากซอร์สโค้ดจริง พบ 31 ข้อบกพร่อง วิกฤต 5 ข้อ ตกเกณฑ์ AA 18 เกณฑ์ กระทบทั้ง 7/7 หน้า

**สถานะ:** งานนี้มอบให้ Antigravity ทำทั้งหมดตัวเดียว (ไม่แบ่งกับ Claude)

**ขอบเขต:** 7 หน้าที่ถูกตรวจคือ `app/page.tsx` (แดชบอร์ด), `app/products/page.tsx` (คลังสินค้า), `app/taxonomy/page.tsx`, `app/settings/page.tsx`, `app/unlock/page.tsx`, `app/data-quality/page.tsx`, `app/import/page.tsx` — **`app/ai-brain/page.tsx` ไม่อยู่ในรายงานตรวจ ไม่อยู่ในขอบเขตงานนี้**

**ไม่ต้องแตะ:** `taxonomy-app/components/data-quality/PosExportTab.tsx` — ทำเสร็จและผ่านมาตรฐานนี้แล้ว (focus ring, modal, ขนาดตัวอักษร) ใช้เป็นตัวอย่างอ้างอิงได้เลยสำหรับ pattern ด้านล่าง

---

## ขั้นที่ 0 — ฐานร่วม (ทำก่อนไล่แก้ทีละหน้า)

1. **Modal component กลาง** — สร้าง `taxonomy-app/components/ui/Modal.tsx` แล้วใช้แทน modal เดิมทุกจุดในระบบ
   - `role="dialog"`, `aria-modal="true"`, `aria-labelledby` ชี้ไปที่หัวเรื่องจริง
   - ปิดด้วย Esc, คลิก backdrop ปิดได้
   - Focus trap ภายใน (Tab/Shift+Tab วนในกรอบ), `autoFocus` บนปุ่มหลักตอนเปิด
   - คืนโฟกัสกลับไปที่ปุ่มที่เปิด modal ตอนปิด (ยกเว้น flow ที่ปิดแล้วเปลี่ยนหน้าจอไปเลย ให้ย้ายโฟกัสไปหัวข้อของหน้าจอใหม่แทน)
   - **อ้างอิงโค้ดต้นแบบ:** `taxonomy-app/components/data-quality/PosExportTab.tsx` (ส่วน confirm dialog — มี `FOCUS_RING`, focus trap, focus restoration ที่รีวิวผ่านมาตรฐานแล้ว) ดึง pattern นี้มาทำเป็น component กลาง

2. **Convention กลาง** (ใช้ทุกไฟล์ที่แก้จากนี้ไป):
   - Focus ring ต้องมองเห็นได้จริง — ห้ามใช้ `focus:outline-none` โดยไม่มีตัวแทนที่มองเห็นชัด (`focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2` หรือเทียบเท่า)
   - ขนาดตัวอักษรเนื้อหาขั้นต่ำ **12px** (`text-xs` ขึ้นไป) — ห้ามใช้ `text-[10px]`/`text-[11px]` กับข้อความที่มีความหมาย
   - เคารพ `prefers-reduced-motion` — ภาพเคลื่อนไหววนไม่หยุด (`animate-pulse`, `animate-ping`) ต้องมีทางปิด/ลดได้

---

## สรุปปัญหาที่พบซ้ำทุกหน้า (ใช้เป็นเช็คลิสต์หลักตอนไล่แก้)

| อาการ | ความรุนแรง | เกณฑ์ WCAG |
|---|---|---|
| ปุ่มไอคอนไม่มีชื่อที่โปรแกรมอ่านหน้าจอเข้าถึงได้ (ปิด/ลบ/แก้ไข/เปิดเมนู) | วิกฤต | 4.1.2, 1.1.1 |
| หน้าต่างซ้อน (modal) ไม่มี role, ไม่ล็อกโฟกัส, ปิดด้วย Esc ไม่ได้ | วิกฤต | 4.1.2, 2.4.3, 2.1.2 |
| ช่องกรอกใช้ label ที่ไม่ผูกกับ id หรือใช้ placeholder แทน label | วิกฤต | 1.3.1, 3.3.2, 4.1.2 |
| กล่องคลิกได้จาก `<div onClick>`/`<tr onClick>` ไม่มี role/tabIndex | วิกฤต | 2.1.1, 4.1.2 |
| แท็บสร้างจากปุ่มธรรมดา ไม่มี `role="tab"`/`aria-selected` | สูง | 1.3.1, 4.1.2 |
| การโหลด/เปลี่ยนผลลัพธ์ไม่มี live region ประกาศ | สูง | 4.1.3 |
| ปุ่มลบซ่อนใต้ `opacity-0 group-hover` กดด้วยคีย์บอร์ด/มือถือไม่ได้ | สูง | 1.4.13, 2.1.1 |
| ชื่อหน้าเว็บซ้ำกันทุกหน้า (ไม่มี metadata รายหน้า) | กลาง | 2.4.2 |
| Animation วนไม่หยุด ไม่รองรับ prefers-reduced-motion | กลาง | 2.2.2, 2.3.3 |
| ข้อผิดพลาดแจ้งผ่าน toast ที่หายใน 4 วิ ไม่ผูกกับช่องที่ผิด | กลาง | 3.3.1 |
| ช่องรหัสผ่านไม่มี `autocomplete="current-password"` | กลาง | 1.3.5 |
| แถบหัวเรื่องตรึงบน (`sticky top-0`) อาจบังจุดโฟกัสตอนกด Tab | กลาง | 2.4.11 |

---

## ไฟล์ที่ต้องไล่แก้ทั้งหมด

**หน้าแดชบอร์ด**
- `taxonomy-app/app/page.tsx`
- `taxonomy-app/components/Layout/Sidebar.tsx`
- `taxonomy-app/components/Layout/Header.tsx`

สิ่งที่ต้องแก้: เมนูหลักอ่านกลับหัว (คำอธิบายอังกฤษอยู่เหนือชื่อเมนูไทย) + ไม่มี skip link ข้ามไปเนื้อหาหลัก · สถานะระบบ ("All Systems Green", build signature) hardcode + จุดกะพริบไม่มีปุ่มหยุด · ช่องค้นหาไม่มี label จริง ใช้ placeholder เป็น label · ปุ่มแจ้งเตือนไม่มีชื่อ/ตัวเลข hardcode/ไม่มี `aria-expanded`/Esc ปิดไม่ได้ · `<h1>` ซ้อนกันสองอันคนละภาษา ข้อความอังกฤษไม่กำกับ `lang="en"` · ป้ายกำกับตัวเลขบนการ์ด contrast ต่ำ (~2.5:1) + `truncate` ตัดคำอธิบายทิ้ง

**หน้าคลังสินค้า**
- `taxonomy-app/app/products/page.tsx`

สิ่งที่ต้องแก้: กรอบโฟกัสถูกปิดด้วย `focus:outline-none` + ring โปร่งใส 10% (พบซ้ำหลายไฟล์ในระบบ — ใช้ convention จากขั้นที่ 0) · ตัวกรองสถานะใช้อีโมจิแทนความหมาย ไม่มี label ผูก · หัวตารางเล็ก/contrast ต่ำ/ไม่มี `scope`/ไม่มี `caption` · แถวคลิกได้ด้วยเมาส์เท่านั้น (`<tr onClick>`) ปุ่มท้ายแถวไม่มี `onClick`/ไม่มีชื่อ · ป้ายสถานะสีตกคอนทราสต์ทุกสี (ต้องได้ ≥4.5:1 ที่ขนาด 10-12px)

**หน้า unlock**
- `taxonomy-app/app/unlock/page.tsx`

สิ่งที่ต้องแก้: เพิ่ม `autocomplete="current-password"` ให้ช่องรหัสผ่าน

**หน้า data-quality**
- `taxonomy-app/app/data-quality/page.tsx`
- `taxonomy-app/components/data-quality/DeduplicationTab.tsx`
- `taxonomy-app/components/data-quality/VerifyTab.tsx`
- `taxonomy-app/components/data-quality/RecheckTab.tsx`
- `taxonomy-app/components/data-quality/AutoLearnTab.tsx`

สิ่งที่ต้องแก้: แท็บ 5 แท็บสร้างจากปุ่มธรรมดา — เพิ่ม `role="tab"`, `aria-selected`, เลื่อนด้วยลูกศรซ้าย-ขวาได้ · live region (`aria-live="polite"`) ที่จุดแสดงสถานะโหลด/ผลตรวจซ้ำทุกจุดที่ยิง API แล้วอัปเดตผล · ปุ่มลบที่ซ่อนใต้ `opacity-0 group-hover` ใน `AutoLearnTab` — เพิ่ม `group-focus-within:opacity-100` · modal ยืนยันต่างๆ เปลี่ยนมาใช้ `components/ui/Modal.tsx`

**หน้า taxonomy**
- `taxonomy-app/app/taxonomy/page.tsx`
- `taxonomy-app/components/Taxonomy/SynonymsPanel.tsx`

สิ่งที่ต้องแก้: modal ในพาเนลผูกคำพ้อง เปลี่ยนมาใช้ `components/ui/Modal.tsx` · ช่องกรอกที่ใช้ placeholder แทน label — ผูก `<label htmlFor>` กับ `id` จริง

**หน้า settings**
- `taxonomy-app/app/settings/page.tsx`

สิ่งที่ต้องแก้: ช่องกรอกที่ใช้ placeholder แทน label — ผูก `<label htmlFor>` กับ `id` จริง · ปุ่มไอคอนไม่มีชื่อ — เพิ่ม `aria-label`

**หน้า import wizard**
- `taxonomy-app/app/import/page.tsx`
- `taxonomy-app/components/Import/WizardLayout.tsx`
- `taxonomy-app/components/Import/WizardTab.tsx`
- `taxonomy-app/components/Import/UploadAndMappingStep.tsx`
- `taxonomy-app/components/Import/ColumnMappingStep.tsx`
- `taxonomy-app/components/Import/DataCleaningStep.tsx`
- `taxonomy-app/components/Import/DeduplicationStep.tsx`
- `taxonomy-app/components/Import/CategorizationStep.tsx`
- `taxonomy-app/components/Import/CompleteStep.tsx`
- `taxonomy-app/components/Import/ImportHistory.tsx`
- `taxonomy-app/components/Import/StorageImport.tsx`

สิ่งที่ต้องแก้: การ์ดเริ่มนำเข้าที่คลิกได้ด้วย `<div onClick>` — เปลี่ยนเป็น `<button>`/`<a>` จริง หรือเพิ่ม `role="button" tabIndex={0}` + handler คีย์บอร์ด (Enter/Space) · `WizardTab` เป็นแท็บ — เพิ่ม `role="tab"`/`aria-selected` เหมือน data-quality · ปุ่มลบที่ซ่อนใต้ hover ใน `ColumnMappingStep` — เพิ่ม `group-focus-within:opacity-100` · error toast ที่หายใน 4 วิ — เปลี่ยนเป็นข้อความ error ผูกกับช่องที่ผิดโดยตรง (`aria-describedby`)

---

## หลังแก้เสร็จ — ตรวจสอบ

- รันชุดเทสต์เดิมให้ไม่มี regression: `npx jest --ci` (baseline ปัจจุบันหลัง merge POS export: 278 passed, 4 suite ล้มเหลวที่รู้อยู่แล้วจาก `.env.local`) และ `npx tsc --noEmit` (baseline: 2 error เดิมใน `synonym.integration.test.ts`)
- ทดสอบด้วยคีย์บอร์ดล้วน (กด Tab ไล่ทั้งหน้าโดยไม่แตะเมาส์) ในทุกหน้าที่แก้
- ตรวจไม่มี horizontal scroll ที่จอ 375px
- อัปเดต `docs/CURRENT_STATUS.md` บันทึกว่าแก้อะไรไปแล้วบ้าง ตามรูปแบบที่ไฟล์นั้นใช้อยู่

## นอกขอบเขตรอบนี้ (ทำทีหลัง ถ้าต้องการ)

- เปลี่ยนคำเรียกในระบบจากอังกฤษประดิษฐ์เป็นไทยที่พนักงานใช้จริง (เช่น "Product Signature" → "ชื่อสินค้า")
- ตั้งชื่อหน้าเว็บแยกรายหน้า (`metadata` ใน `layout.tsx`/`page.tsx` แต่ละหน้า)
- ลบส่วนติดต่อที่ไม่ทำงานจริง (ปุ่ม "ตั้งค่าบัญชี" ที่ไม่มีปลายทาง, อีเมล admin@company.com ที่ไม่มีจริง)
