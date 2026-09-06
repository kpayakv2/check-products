# 04: ชะตากรรมของ GEMINI.md

**Labels:** `wayfinder:grilling`
**Parent:** [แผนที่](../map.md)
**Blocked by:** ~~[01: ลำดับศักดิ์ของแหล่งความจริง](01-precedence-order.md)~~ — ปิดแล้ว 2026-09-06 ตั๋วนี้ไม่ติดบล็อกแล้ว
**Assignee:** Claude (จองแล้ว 2026-09-06)
**Status:** closed (resolved 2026-09-06)

## Question

`GEMINI.md` จะถูก **ลบ / ยุบเหลือตัวชี้ไป `AGENTS.md` / เก็บเป็นบันทึกประวัติศาสตร์แบบแช่แข็ง** และกฎ 3 ข้อที่มีเฉพาะในนั้นจะถูกย้ายเข้า `AGENTS.md` หรือทิ้งไปเลย

## อัปเดตจากการปิด[ตั๋ว 01](01-precedence-order.md) (2026-09-06)

หัวโต๊ะของหมวด Rule ไม่ใช่ `CLAUDE.md` แล้ว แต่เป็น **`AGENTS.md`** ไฟล์ใหม่ที่รากโปรเจกต์ (`CLAUDE.md` ตอนนี้เหลือแค่ `@AGENTS.md` import บรรทัดเดียว) — ทุกที่ข้างล่างที่เคยพูดถึง "CLAUDE.md" ให้อ่านเป็น AGENTS.md แทน

พบข้อเท็จจริงใหม่ที่เพิ่มตัวเลือกให้ตั๋วนี้: **Gemini CLI ตั้งค่า `settings.json` → `context.fileName` ให้อ่าน `AGENTS.md` แทน/ควบคู่กับ `GEMINI.md` ได้จริง** (ยืนยันจาก web search) — เปิดตัวเลือกที่ 4 ที่ไม่เคยมีตอนตั้งตั๋ว: "ลบ GEMINI.md ทิ้ง แล้วตั้งค่า Gemini CLI ให้อ่าน AGENTS.md โดยตรง" ซึ่งต่างจาก "ยุบเหลือตัวชี้" ตรงที่ไม่มีไฟล์ GEMINI.md เหลืออยู่เลย

`.agents/rules/rules-ai-agent.md:29` ที่เคยชี้ไป GEMINI.md **ถูกแก้แล้ว** เป็นการชี้ไป AGENTS.md — เหลือแค่ `docs/development/architecture.md:4` ที่ยังชี้ผิดอยู่ (ดูด้านล่าง)

## ทำไมต้องตัดสินแยกจากลำดับศักดิ์

`GEMINI.md` มีเนื้อที่ **ไม่ได้อยู่ใน `AGENTS.md` เลย** จึงตัดสินอัตโนมัติจากลำดับศักดิ์ไม่ได้ ต้องตัดสินทีละข้อ

## เนื้อที่มีเฉพาะใน GEMINI.md (ต้องตัดสินทีละข้อ)

| เนื้อหา | สภาพ | ตัวเลือก |
|---|---|---|
| กฎข้อ 7 — Smart Testing Matrix 5 ขั้น บังคับก่อน commit/deploy ทุกครั้ง | ขั้นที่ 3 ใช้ Puppeteer + Domscribe ที่ **ไม่มีใน `.mcp.json`** | ทิ้ง / เขียนใหม่ให้ใช้ Playwright ที่มีจริง / ย้ายเข้า AGENTS.md ตามสภาพ |
| กฎข้อ 8 — บังคับใช้ MCP มาตรฐาน 6 ตัว | มีจริง 4 ตัว **ทำตามไม่ได้** | ทิ้ง / เขียนใหม่ให้ตรงกับ 4 ตัวที่มี |
| กฎข้อ 6 — Benchmark ≥ 72% F1 | `AGENTS.md` (ย้ายมาจาก CLAUDE.md เดิม) ประกาศยกเลิกแล้ว | ส่งต่อให้ [07](07-accuracy-rule-wording.md) |
| แผนผังสถาปัตยกรรม + LAN Access (IP `192.168.1.80`) | **ยืนยันแล้วว่ายังจริง** (2026-09-06): `taxonomy-app/next.config.js` implement ตรงตาม checklist เป๊ะ, `START_PHAYAK.bat` มีอยู่จริง, IP ตรงกัน 7 จุดรวมถึงในโค้ด (`utils/supabase.ts`) และ `docs/CURRENT_STATUS.md:546` — เนื้อหานี้มีบ้านอยู่แล้วที่ `.agents/rules/rules-windows.md` (ซึ่ง AGENTS.md ชี้ไปหาอยู่แล้ว) ส่วนใน GEMINI.md เป็นแค่สำเนาซ้ำคำต่อคำ | **ทิ้งได้เลย ไม่ต้องย้าย** — ของจริงอยู่ที่ rules-windows.md แล้ว |

## เงื่อนไขที่ต้องแก้พร้อมกัน ไม่งั้นตัดสินแล้วไม่มีผล

จุดที่ยัง **ชี้กลับมาที่ GEMINI.md ในฐานะแหล่งอำนาจ** ที่ยังไม่ได้แก้ — ลบไฟล์เฉยๆ จะทำให้ลิงก์ตาย ปล่อยไว้เฉยๆ ก็ยังส่งคนไปอ่านของผิด:

- ~~`.agents/rules/rules-ai-agent.md:29`~~ — **แก้แล้ว** (2026-09-06 ระหว่างปิดตั๋ว 01) ชี้ไป AGENTS.md แล้ว
- `docs/development/architecture.md:4` — "กฎเหล็ก: ต้องรักษาความแม่นยำไม่ต่ำกว่า 72% ตาม [GEMINI.md]" — **ยังไม่แก้**

(ให้ [03: ทะเบียนข้อขัดแย้ง](03-conflict-register.md) ยืนยันว่ามีจุดอื่นอีกไหม)

## เกณฑ์ว่าตอบครบ

- [x] ตัดสินชะตากรรมของไฟล์แล้ว และเหตุผลบันทึกไว้
- [x] กฎเฉพาะทั้ง 4 กลุ่มถูกตัดสินทีละข้อ — ย้าย เขียนใหม่ หรือทิ้ง ไม่มีข้อไหนค้าง
- [x] ทุกจุดที่ชี้มาที่ GEMINI.md ถูกแก้ในรอบเดียวกัน
- [x] ~~ถ้าเลือก "แช่แข็งเป็นประวัติศาสตร์"...~~ — ไม่เข้าเงื่อนไขนี้ เลือกลบทิ้งแทน

## Resolution (2026-09-06)

**ผู้ใช้เห็นด้วยกับคำแนะนำทั้ง 3 ข้อ (Q1-Q3):**

1. **ชะตากรรมไฟล์:** ลบทั้งคู่ — `GEMINI.md` (ราก) และ `.gemini/GEMINI.md` **ไม่ใช่ยุบเหลือ stub** ตั้งค่า Gemini CLI ให้อ่าน `AGENTS.md` ตรงผ่าน `.gemini/settings.json` → `context.fileName: ["AGENTS.md"]` แทน (ยืนยันวิธีนี้ใช้ได้จริงจาก web search ก่อนหน้านี้)
2. **กฎข้อ 7 (Smart Testing Matrix, ใช้ Puppeteer+Domscribe ที่ไม่มีจริง):** ทิ้ง ไม่ย้ายเข้า AGENTS.md — ซ้ำกับ §Testing ที่มีอยู่แล้วและละเอียดกว่า
3. **กฎข้อ 8 (บังคับ MCP มาตรฐาน 6 ตัว):** ทิ้ง ไม่ย้าย — ซ้ำกับ §MCP Tools Available ที่มีอยู่แล้ว
4. (กฎข้อ 6 เรื่อง 72% → ส่งต่อ [07](07-accuracy-rule-wording.md) ตามเดิม; แผนผังสถาปัตยกรรม+LAN → ทิ้งได้เลยตามที่ตัดสินไว้แล้วในตั๋ว 03)

**พบระหว่างทำ (ไม่เคยอยู่ในทะเบียนตั๋ว 03 มาก่อน):**
- **`.gemini/GEMINI.md` เป็นไฟล์คนละไฟล์กับ `GEMINI.md` ที่ราก** — เนื้อหาสั้นกว่า ("Local Agent Configuration") อ้าง Memory MCP ที่ถูกถอดไปแล้วในตั๋ว 02 และอ้างอิง "GEMINI.md หลักของโปรเจกต์" เป็นแหล่งจริง — subsumed เต็มโดย AGENTS.md เหมือนกัน จึงลบพร้อมกัน
- **`.gemini/settings.json` มี `mcpServers` เป็นคอนฟิกจริงที่ยังใช้งานอยู่** (คนละกลไกจาก `.mcp.json` ของ Claude Code) — มีครบ 6 ตัวเดิมที่กฎข้อ 8 พูดถึง (`filesystem`, `puppeteer`, `memory`, `domscribe`, `postgres`, `sequential-thinking`) บวก `socraticode` นี่คือกฎข้อ 8 เวอร์ชัน "มีผลจริง" ไม่ใช่แค่เอกสาร — แก้ให้เหลือ 3 ตัวตรงกับ `.mcp.json` (`postgres`, `socraticode`, `sequential-thinking`) ตามการตัดสินใน Q3

**ไฟล์ที่แก้ทั้งหมด:**
- ลบ: `GEMINI.md`, `.gemini/GEMINI.md`
- `.gemini/settings.json` — เพิ่ม `context.fileName`, ตัด mcpServers เหลือ 3 ตัว
- `docs/api/api-reference.md:3`, `docs/development/architecture.md:4`, `README.md:74` — ลิงก์ที่เคยชี้ GEMINI.md เปลี่ยนไปชี้ AGENTS.md แล้ว (บรรทัด architecture.md คงคำถาม "72%" ไว้ให้ตั๋ว 07 ตัดสิน ไม่แตะเนื้อกฎ)
- `AGENTS.md` ย่อหน้าเปิด — อัปเดตว่า Gemini CLI อ่าน AGENTS.md ตรงแล้ว ไม่ใช่ "งานค้าง" อีกต่อไป
- `docs/adr/0001-single-canonical-rule-file.md` — เพิ่มบรรทัดอ้างว่าตั๋วนี้ปิดแล้วและทำอะไรไปบ้าง
