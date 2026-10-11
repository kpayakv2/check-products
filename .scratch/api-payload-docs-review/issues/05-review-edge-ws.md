# 05: รีวิว Edge Functions และ WebSocket

**Labels:** `wayfinder:research` (AFK — สกิล research ไม่ได้ติดตั้ง ทำ inline)
**Parent:** [แผนที่](../map.md)
**Blocked by:** [01: สำรวจรายการ](01-inventory.md), [02: เกณฑ์ผ่าน](02-pass-criteria.md)
**Assignee:** Antigravity
**Status:** closed

## Question

ทุกแถวชั้น 3 (Edge Functions 3 ตัวใน `taxonomy-app/supabase/functions/`) และชั้น 5 (`src/api/custom_websockets.py`) ใน [`matrix.md`](../matrix.md) ผ่านหรือไม่ผ่านมิติ 1, 3, 4, 5, 6, 7, 8 ตามเกณฑ์ของตั๋ว 02

จุดที่ต้องดูเป็นพิเศษ:
- **มิติ 4:** `verify_jwt` ถูกคอมเมนต์ทิ้งไปตอนลบ `exec-sql` (AGENTS.md) — Edge Function ที่เหลือเปิดโล่งไหม และใช้ service role ข้างในไหม
- **มิติ 6:** ทั้ง 3 ตัวยังมีผู้เรียกจริงไหม หรือถูกแทนด้วย FastAPI ไปแล้ว (ชื่อ `-local` ชวนสงสัย)
- **มิติ 4/5 ของ ws:** เชื่อมต่อโดยไม่ยืนยันตัวตนได้ไหม, message ผิดรูปทำอะไร

## วิธี

- Edge Function: อ่านโค้ด + ถ้า Supabase local รันอยู่และ function อ่านอย่างเดียว ให้ยิงผ่าน `:54331`
- ws: อ่านโค้ด, เชื่อมต่อจริงได้ถ้าไม่มีผลข้างเคียงเชิงเขียน

## เกณฑ์ว่าเสร็จ

- [x] ทุกช่องมิติของแถวชั้น 3 และ 5 มีค่า (กรอกครบทั้ง 10 แถว L3-01..03 และ L5-01..07 ใน `matrix.md` โดยมิติ 2 กำกับ "รอตั๋ว 06")
- [x] ทุกข้อค้นพบอยู่ใน [`findings.md`](../findings.md) ตามรูปแบบเดียวกับตั๋ว 03 (บันทึก F-018 ถึง F-022 พร้อมหลักฐานการทดสอบและคำตัดสินที่เสนอ)
- [x] P0 ถูกรายงานทันที (ไม่พบ P0 ตัวใหม่ในชั้น Edge Functions และ WebSocket)

## Resolution

- **ยิงจริง Supabase Edge Functions (`:54331/functions/v1/...`):**
  - ยิงทดสอบพบ HTTP 503 `{"message":"name resolution failed"}` จาก Kong gateway
  - ตรวจสอบ `docker ps` พบว่าใน local Supabase stack **ไม่มี container edge runtime รันอยู่เลย**
  - ไม่พบผู้เรียกใช้ Edge Functions ในโค้ด frontend แม้แต่จุดเดียว (หน้าเว็บเรียก FastAPI ตรงผ่าน Next.js rewrites ทั้งหมด)
- **ยิงจริง WebSocket (`ws://127.0.0.1:8000/ws`):**
  - ยืนยันเชื่อมต่อจริงได้สำเร็จโดยไม่ต้องใช้ authentication หรือ token ใดๆ
  - ยืนยันการส่ง binary frame ส่งผลให้เกิด unhandled exception (`RuntimeError`) หลุดจาก handler ทำให้ connection ถูกตัดแบบกะทันหัน (`ConnectionClosedError` ขาด close frame) และ connection ค้างในหน่วยความจำเนื่องจากขาด `finally:` block
  - ตรวจสอบโค้ด frontend ใน `taxonomy-app` ทั้งหมด พบว่าไม่มีโค้ดฝั่งผู้รับ (client) แม้แต่จุดเดียว
- **ข้อค้นพบใหม่ (F-018 ถึง F-022):**
  - **F-018 (P1):** Edge Function `generate-embeddings-local` ประกาศ payload ไม่ตรงกับ FastAPI สองด้าน (ส่ง `{texts: [text]}` แต่ router รับ `{text}`, คาดหวัง `data.embeddings[0]` แต่ router คืน `embedding`) พัง 100% หากเรียกใช้งาน
  - **F-019 (P2):** Edge Functions ทั้ง 3 ตัวเป็น proxy ซ้ำซ้อนที่ไม่มี container รันในเครื่อง และไม่มีผู้เรียกใช้งานจริงในแอป
  - **F-020 (P2):** `WS /ws` เปิดรับการเชื่อมต่อโดยไม่มี auth และ binary frame ทำ connection แครชค้างใน memory
  - **F-021 (P3):** ระบบ WebSocket ฝั่งเซิร์ฟเวอร์ทั้ง 7 รายการ (L5-01 ถึง L5-07) เป็นระบบตายที่ไม่มีผู้รับใน frontend เลย
  - **F-022 (P3):** ขาด Automated Tests สำหรับ Edge Functions และ WebSocket ทั้ง 10 รายการ
- **อัปเดตไฟล์:** กรอกผลประเมินครบทุกมิติใน [`matrix.md`](../matrix.md) (L3-01..03, L5-01..07) และบันทึกข้อค้นพบ F-018 ถึง F-022 ลงใน [`findings.md`](../findings.md) เรียบร้อย

