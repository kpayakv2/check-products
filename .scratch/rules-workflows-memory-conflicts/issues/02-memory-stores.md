# 02: คลังความทรงจำสามที่ ควรเหลือกี่ที่

**Labels:** `wayfinder:grilling`
**Parent:** [แผนที่](../map.md)
**Blocked by:** ไม่มี — เริ่มได้ทันที
**Assignee:** Claude (จองแล้ว 2026-09-06)
**Status:** closed (resolved 2026-09-06)

## Question

โปรเจกต์นี้มีคลังความทรงจำซ้อนกันสามที่ โดยไม่มีที่ไหนประกาศว่าอันไหนใช้ทำอะไรหรืออันไหนชนะ **ควรเหลือกี่ที่ และแต่ละที่รับผิดชอบอะไร**

| คลัง | สภาพจริง (ตรวจ 2026-09-05) | ใครเขียน |
|---|---|---|
| auto-memory ของ Claude Code<br>`C:\Users\minds\.claude\projects\d--product-checker-check-products\memory\` | 10 ไฟล์ + `MEMORY.md` แก้ล่าสุด 29 ส.ค. | Claude เขียนเอง, `MEMORY.md` **ถูกฉีดเข้าทุกเซสชัน** |
| `.agents/memory/` | 2 ไฟล์ (`bug_numpy_feature_names.md`, `dedup_refactor_lessons.md`) แตะล่าสุด 27 มิ.ย. | คนเขียนมือ, ไม่มีใครโหลดอัตโนมัติ |
| `.mcp-memory/memory.jsonl` | **0 ไบต์** สร้างไว้ 19 ส.ค. ไม่เคยมีอะไรถูกเขียนลงเลย | memory MCP server |

พร้อมกันนั้น `GEMINI.md:36` กำหนดให้ **Memory MCP เป็นขั้นที่ 5 ของ Smart Testing Matrix ที่ "ห้ามข้ามเด็ดขาด"** — หน้าที่ที่ไม่มีใครทำมาสองสัปดาห์กว่า

## คำถามย่อยที่ต้องตอบไปด้วย

- **memory MCP server ควรอยู่ต่อไหม** ในเมื่อผ่านไปกว่าสองสัปดาห์ยังเขียนอะไรไม่ได้เลยสักไบต์ — ถ้าเก็บไว้ ใครมีหน้าที่เขียน และเขียนเมื่อไร ถ้าไม่เก็บ ต้องถอดออกจาก `.mcp.json` และจาก `enabledMcpjsonServers` ใน `.claude/settings.local.json` ด้วย
- **`.agents/memory/` 2 ไฟล์นั้นยังจริงอยู่ไหม** และควรย้ายไปรวมกับ auto-memory หรือกลายเป็นเอกสารใน `docs/`
- **auto-memory อยู่นอกรีโปและเป็นของส่วนตัว** — คนอื่นในทีมไม่เห็น แต่ `MEMORY.md` มีอิทธิพลสูงสุดเพราะเข้าทุกเซสชัน ความไม่สมมาตรนี้รับได้ไหม

## หลักฐานว่าความทรงจำโกหกอยู่ตอนนี้ (ยืนยันแล้ว 2026-09-05)

`MEMORY.md` บรรทัดแรกสุด — ตัวที่เข้าทุกเซสชัน — เขียนว่า:

> แก้ไป 3/4 ข้อ เหลือ jest.config typo (CLAUDE.md ยังเขียนสถานะเก่าอยู่)

**ผิดทั้งสองครึ่ง:**
- `taxonomy-app/jest.config.js:39` เป็น `moduleNameMapper` ถูกต้องแล้ว
- `CLAUDE.md` "Known Unresolved Issues" ขีดฆ่าครบทั้ง 4 ข้อแล้ว

ไฟล์เต็ม `repo-audit-2026-08-19.md:18` ยังเขียน "❌ ยังไม่แก้" และบรรทัด 20 ยังสั่งว่า "ควรอัปเดต CLAUDE.md ด้วยถ้ามีโอกาส" — คำสั่งที่ทำไปแล้ว

อีกจุด: `MEMORY.md` บรรทัด `project-state-2026-08-28` เขียนว่า "11 commit บนสาขา **ยังไม่ merge**" แต่ `fix/status-mismatch-and-page-cleanup` ถูก merge เข้า `main` แล้ว (`328b1467`)

## เกณฑ์ว่าตอบครบ

- [x] ตัดสินแล้วว่าเหลือคลังไหนบ้าง และแต่ละคลังเก็บอะไร
- [x] ถ้า memory MCP ถูกถอด ต้องถอดจาก `.mcp.json` และ `.claude/settings.local.json` พร้อมกัน
- [x] มีวิธีที่ทำให้ความทรงจำที่เน่าถูกจับได้ ไม่ใช่รอให้บังเอิญเจอ

## Resolution (2026-09-06)

**คำตอบ:** เหลือ 2 คลัง ไม่ใช่ 3 — ถอด memory MCP server ทิ้ง

- **Personal Memory** (auto-memory `MEMORY.md` + ไฟล์ย่อย) — นอกรีโป เห็นคนเดียว ฉีดทุกเซสชัน มีน้ำหนักสูงสุดเพราะถูกอ่านบ่อยที่สุด
- **Team Memory** (`.agents/memory/`) — ในรีโป เห็นได้ทุกคน/ทุกเครื่องมือ ไม่ auto-inject ต้องเปิดอ่านเอง — ตรวจแล้วเนื้อหา 2 ไฟล์ยังจริงอยู่ (อ้างอิงโค้ดที่มีอยู่จริง) แค่ไม่มีใครโหลดอัตโนมัติ ไม่ต้องยุบรวมหรือเลื่อนสถานะ
- **Memory MCP** (`.mcp-memory/memory.jsonl`) — **ถอดทิ้ง** เขียน 0 ไบต์มา 2.5 สัปดาห์ ซ้ำซ้อนกับ Personal Memory ที่ทำงานอยู่แล้ว เหตุผลที่เคยอ้างว่าบังคับ (GEMINI.md Smart Testing Matrix) หมดความหมายไปแล้วตั้งแต่ตั๋ว 01

ทั้งสามหมวดนี้ยังอยู่ใน "Memory" ตามนิยามเดิมของ [CONTEXT.md](../../../CONTEXT.md) (ผิดพลาดได้ ไม่ใช่แหล่งอำนาจ) — แค่แยกย่อยเป็น Personal/Team ให้ชัดขึ้น

**กันเน่าซ้ำ:** ปัญหาจริงไม่ใช่ไม่มีกฎ (แพลตฟอร์มเองก็บอกอยู่แล้วว่าต้องตรวจก่อนเชื่อ) แต่คือพอเจอว่าผิดกลางบทสนทนาแล้วไม่มีใครย้อนไปแก้ไฟล์ — บันทึกเป็นความทรงจำแบบ feedback ใหม่ `feedback-fix-stale-memory-immediately.md` ใน Personal Memory (อยู่นอกรีโป จึงลิงก์ตรงจากที่นี่ไม่ได้)

**สิ่งที่ทำไปแล้ว:**
1. ถอด `memory` server ออกจาก `.mcp.json` และ `enabledMcpjsonServers` ใน `.claude/settings.local.json`
2. ลบโฟลเดอร์ `.mcp-memory/` ทิ้ง (ว่างเปล่า ไม่มีอะไรเสีย)
3. แก้ `AGENTS.md` § MCP Tools — เอาบรรทัด `memory` ออก บันทึกไว้ว่าทำไมถอด
4. แก้ `CONTEXT.md` — แตก "Memory" เป็น "Personal Memory" / "Team Memory"
5. แก้ auto-memory ที่รู้แล้วว่าผิด: `repo-audit-2026-08-19.md` (jest.config ปิดครบ 4/4), `project-state-2026-08-28.md` (branch merge แล้ว), และดัชนี `MEMORY.md` ทั้งสองบรรทัด
6. เขียนความทรงจำแบบ feedback ใหม่ `feedback-fix-stale-memory-immediately.md` + เพิ่มลงดัชนี `MEMORY.md`
