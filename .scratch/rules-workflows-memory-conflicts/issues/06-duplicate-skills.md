# 06: สกิลซ้ำสองที่ — `.agents/skills/` กับ `.claude/skills/`

**Labels:** `wayfinder:grilling`
**Parent:** [แผนที่](../map.md)
**Blocked by:** ~~[01: ลำดับศักดิ์ของแหล่งความจริง](01-precedence-order.md)~~ — ปิดแล้ว 2026-09-06 ตั๋วนี้ไม่ติดบล็อกแล้ว
**Assignee:** Claude (จองแล้ว 2026-09-06)
**Status:** closed (resolved 2026-09-06)

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

- [x] ตัดสินแล้วว่าโฟลเดอร์ไหนเป็นต้นฉบับ และอีกฝั่งกลายเป็นอะไร — **พลิกสมมติฐานเดิมทั้งหมด ดู Resolution**
- [x] ตัดสินแล้วว่าสกิลโปรเจกต์ 4 ตัวจะถูกทำให้โหลดได้จริงไหม — คัดลอกเข้า `.claude/skills/` แล้ว
- [x] `skills-lock.json` สอดคล้องกับสิ่งที่เหลืออยู่จริงเท่าที่ตรวจได้ (7/15 มี lock, ที่เหลือไม่ทราบต้นทางแน่ชัด — ไม่เดา)
- [x] ตัดสินแล้วว่าอะไรควรเข้า git อะไรควร ignore
- [x] คำบรรยาย `.agents/skills/` ใน `AGENTS.md` ตรงกับของจริง
- [x] ยืนยันแล้วว่า Antigravity สแกน `.agents/{rules,skills}/` อัตโนมัติจริง (`.agents/workflows/` ไม่ใช่ — Antigravity เองกำลังเลิกใช้ concept workflow แล้วหันไปใช้ skill แทนภายใน 1 พ.ย. 2026)

## ⚠️ อุบัติเหตุระหว่างทำตั๋วนี้ (2026-09-06)

ระหว่างตรวจสภาพจริง พบว่า `.claude/skills/{9 ตัว}` **ไม่ใช่สำเนาไบต์ต่อไบต์อย่างที่ตั๋วสันนิษฐานไว้ตอนตั้ง — เป็น symlink ชี้ไป `.agents/skills/{name}`** (ของจริงอยู่ที่ `.agents/skills/` ฝั่งเดียว) เขียนสคริปต์ `scripts/sync_agent_skills.py` โดยไม่ตรวจว่าเป็น symlink ก่อน สคริปต์ทำ `rmtree(target)` แล้ว `copytree(source, target)` — สำหรับ 2 ตัว (`grill-me`, `to-spec`) ลำดับนี้ไปลบของจริงใน `.agents/skills/` ทิ้งก่อนที่จะ copy กลับ ทำให้ copy ล้มเหลว (source กลายเป็น symlink ที่ชี้ไปที่ที่ถูกลบไปแล้ว) **ไฟล์จริงหายไปทั้งคู่**

กู้คืนโดยใช้สกิล `find-skills` ค้นหาต้นทาง → ยืนยันว่ามาจาก `mattpocock/skills` (แหล่งเดียวกับ wayfinder/grilling/domain-modeling) → `npx skills add mattpocock/skills@grill-me -y` และ `@to-spec -y` ติดตั้งใหม่จากต้นฉบับจริง ไม่ใช่เดาเนื้อหาจากความจำ — ทั้งสองกลับมาตรงกับต้นฉบับ 100% (ยืนยันว่า `to-spec` ตัวจริงยาวกว่าที่ผมจำได้จาก `head -8` ก่อนหน้ามาก ถ้าพิมพ์คืนจากความจำเองจะสูญเนื้อหาจริง)

**บทเรียน:** ลบสคริปต์ `scripts/sync_agent_skills.py` ทิ้งแล้ว — ไม่จำเป็นอีกต่อไป (ดู Resolution ข้อ 1)

## Resolution (2026-09-06)

**ข้อเท็จจริงที่พลิกสมมติฐานเดิมของตั๋วนี้:** `.agents/skills/` ไม่ใช่สำเนาที่ทิ้งได้ — Antigravity สแกนมันจริง (ยืนยันจาก Antigravity docs) และเครื่องมือติดตั้งสกิลจริง (`npx skills` / `npx skills add`) **จัดการความซ้ำให้เองอยู่แล้วด้วย symlink**: `.agents/skills/{name}` คือของจริง, `.claude/skills/{name}` เป็น symlink ชี้กลับไป สร้างโดยอัตโนมัติทุกครั้งที่ `npx skills add`/`update` ทำงาน — **นี่ไม่ใช่ข้อขัดแย้งที่ต้องแก้ แต่เป็นสถาปัตยกรรมที่ตั้งใจแล้วและทำงานถูกต้องอยู่แล้ว** สำหรับ 11 สกิลที่มาจาก marketplace

**คำตอบต่อคำถามเดิมทั้ง 4 ข้อ (ปรับตามข้อเท็จจริงใหม่ที่พบระหว่างทำ ผู้ใช้เห็นด้วยกับแนวทางที่ปรับแล้ว):**

1. **ต้นฉบับ/สำเนา:** ไม่ต้องเขียนสคริปต์ซิงก์เอง — `npx skills` จัดการ 11 ตัวที่มาจาก marketplace ด้วย symlink อยู่แล้ว (ลบ `scripts/sync_agent_skills.py` ทิ้ง) ส่วน 4 สกิลโปรเจกต์ที่เขียนเอง (ไม่ได้ติดตั้งผ่าน CLI) เก็บเป็น **สำเนาธรรมดา** (ไม่ symlink — พยายาม `ln -s` ด้วยมือแล้วพบว่า Git Bash บน Windows สร้างผลลัพธ์ที่ไม่แน่นอน จึงเลือกความปลอดภัยที่ตรวจสอบได้มากกว่า) ต้องแก้ทั้งสองชุดถ้าจะแก้สกิลกลุ่มนี้ — เอกสารไว้ใน AGENTS.md แล้ว
2. **สกิลโปรเจกต์ 4 ตัว:** คัดลอกเข้า `.claude/skills/` แล้ว ใช้ได้จาก Claude Code แล้วตอนนี้
3. **`skills-lock.json`:** ตอนนี้ track 7 ตัว (`domain-modeling`, `grill-me`, `grilling`, `pgvector-semantic-search`, `to-spec`, `vercel-react-best-practices`, `wayfinder`) — รันแล้วทั้ง `npx skills update` (รีเฟรช 5 ตัวที่ track อยู่) ยังเหลือ **6 ตัวที่ไม่มี lock และไม่ทราบต้นทางแน่ชัด** (`code-review`, `diagnosing-bugs`, `find-skills`, `grill-with-docs`, `implement`, `to-tickets`) — ไม่เดา/ไม่เขียน entry มั่ว ถ้าต้องการ lock ครบ ต้องรัน `npx skills add mattpocock/skills@<name> -y` ทีละตัว (คาดว่ามาจากที่เดียวกันเพราะพฤติกรรม/สไตล์เขียนตรงกัน แต่ยังไม่ยืนยัน)
4. **Git tracking (แก้ไขระหว่างคอมมิต — พบเพิ่มอีกจุด):** commit `.agents/skills/*` ทั้งหมดปกติ (ของจริง ไม่ใช่ symlink) แต่ตอน `git add .claude/skills/` พบว่า repo นี้ตั้ง `core.symlinks=false` — ทำให้ `git add` เก็บ **เนื้อหาที่ถูก dereference เป็นไฟล์จริงแยกต่างหาก** แทนที่จะเก็บ symlink pointer (ตรวจด้วย `git ls-files -s` เจอ mode `100644` ไม่ใช่ `120000`) ถ้าปล่อยให้ commit แบบนั้นจะ**สร้างปัญหาเดิมที่ตั๋วนี้เพิ่งแก้กลับมาใหม่แบบเงียบๆ** (สำเนาที่ git คิดว่าเป็นไฟล์อิสระ จะไม่ sync กับของจริงอีกเลยหลัง commit) — แก้โดย `git restore --staged` เอา 11 ตัวที่เป็น symlink ออก แล้วเพิ่มลง `.gitignore` แทน (คอมเมนต์อธิบายเหตุผลไว้ในไฟล์) เหลือ commit เฉพาะ 4 สกิลโปรเจกต์ที่เป็นไฟล์จริงใน `.claude/skills/` เท่านั้น — ใครโคลนรีโปใหม่ต้องรัน `npx skills update` เองเพื่อสร้าง symlink 11 ตัวกลับมา ignore `.claude/settings.local.json` เหมือนเดิม (มี absolute path เฉพาะเครื่องอยู่จริง)
