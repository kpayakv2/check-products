# 06: สกิลซ้ำสองที่ — `.agents/skills/` กับ `.claude/skills/`

**Labels:** `wayfinder:grilling`
**Parent:** [แผนที่](../map.md)
**Blocked by:** ~~[01: ลำดับศักดิ์ของแหล่งความจริง](01-precedence-order.md)~~ — ปิดแล้ว 2026-09-06 ตั๋วนี้ไม่ติดบล็อกแล้ว
**Assignee:** (ว่าง — ยังไม่มีใครจอง)
**Status:** open

## Question

สกิล 9 ตัวถูกเก็บซ้ำแบบ **ไบต์ต่อไบต์** ในสองโฟลเดอร์ แต่ Claude Code โหลดแค่ `.claude/skills/` — จะให้ที่ไหนเป็นต้นฉบับ และอีกที่กลายเป็นอะไร (ลบ / symlink / ตัวชี้ / ปล่อยซ้ำต่อไปโดยมีกลไกซิงก์)

## สภาพจริง (ยืนยันแล้ว 2026-09-05)

ซ้ำกันสนิททั้ง 9 ตัว: `code-review`, `diagnosing-bugs`, `find-skills`, `grill-me`, `grill-with-docs`, `implement`, `to-spec`, `to-tickets`, `wayfinder`

`.agents/skills/` มี 13 ตัว = 9 ตัวข้างบน + 4 ตัวที่ **มีเฉพาะที่นั่น** (`thai-taxonomy-expert`, `data-cleaner`, `pgvector-semantic-search`, `vercel-react-best-practices`) — สี่ตัวหลังนี้คือของโปรเจกต์แท้ๆ และ **ไม่ได้ถูกโหลดโดย Claude Code เลย**

`AGENTS.md` (เดิมคือ `CLAUDE.md` ก่อน[ตั๋ว 01](01-precedence-order.md)ย้ายเนื้อหา 2026-09-06) ยังบรรยาย `.agents/skills/` ว่ามีแค่ 4 สกิลนั้น — เขียนไว้ก่อนที่อีก 9 ตัวจะถูกใส่เข้ามา

**อัปเดตจากการปิดตั๋ว 01:** Antigravity IDE มีแนวคิด Rule/Workflow/Skill เป็นของตัวเอง ตรงกับชื่อ `.agents/rules`, `.agents/workflows`, `.agents/skills` แบบเป๊ะ — ยัง **ไม่ยืนยัน** ว่า Antigravity สแกนโฟลเดอร์เหล่านี้อัตโนมัติจริงหรือไม่ ถ้าใช่ `.agents/skills/` ไม่ใช่แค่สำเนาที่ทิ้งได้เฉยๆ แต่เป็นของที่ Antigravity ใช้จริง — ควรตรวจข้อเท็จจริงนี้ก่อนตัดสินว่าอีกฝั่งไหน "ตาย"

`skills-lock.json` ติดตามแค่ 3 ตัว (`pgvector-semantic-search`, `vercel-react-best-practices`, `wayfinder`) พร้อม hash — อีก 10 ตัวไม่มี lock

ทั้ง `.claude/` และสกิลใหม่ใน `.agents/skills/` ยัง **untracked ใน git**

## ทำไมนี่คือข้อขัดแย้งจริง ไม่ใช่แค่รก

สำเนาสองชุดที่ตอนนี้เหมือนกันเป๊ะ **รับประกันว่าจะแตกต่างกันในอนาคต** ใครสักคนแก้ฝั่งที่ไม่ถูกโหลด แล้วสงสัยว่าทำไมแก้แล้วไม่มีผล — เป็นบั๊กชนิดเดียวกับที่ `CLAUDE.md` เตือนไว้เรื่อง FastAPI cache `keyword_rules`

และปัญหาที่หนักกว่า: **สกิลของโปรเจกต์ 4 ตัวที่มีค่าที่สุด อยู่ในฝั่งที่ไม่ถูกโหลด**

## เกณฑ์ว่าตอบครบ

- [ ] ตัดสินแล้วว่าโฟลเดอร์ไหนเป็นต้นฉบับ และอีกฝั่งกลายเป็นอะไร
- [ ] ตัดสินแล้วว่าสกิลโปรเจกต์ 4 ตัว (`thai-taxonomy-expert` ฯลฯ) จะถูกทำให้โหลดได้จริงไหม
- [ ] `skills-lock.json` สอดคล้องกับสิ่งที่เหลืออยู่จริง
- [ ] ตัดสินแล้วว่าอะไรควรเข้า git อะไรควร ignore
- [ ] คำบรรยาย `.agents/skills/` ใน `AGENTS.md` ตรงกับของจริง
- [ ] ยืนยันแล้วว่า Antigravity สแกน `.agents/{rules,workflows,skills}/` อัตโนมัติหรือไม่ ก่อนตัดสินว่าฝั่งไหนคือสำเนาที่ทิ้งได้
