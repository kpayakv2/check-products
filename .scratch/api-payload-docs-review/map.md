# แผนที่: รีวิว API, ประเภท payload และเอกสารที่อ้าง ให้ครบทุกมิติ

**Labels:** `wayfinder:map`
**Tracker:** local-markdown (ไม่มี issue tracker จริงในรีโปนี้ — ถ้าจะย้ายขึ้น GitHub/Linear ให้รัน `/setup-matt-pocock-skills` ก่อน)
**ตั๋วทั้งหมด:** `./issues/` — ตั๋วเปิดคือไฟล์ที่ `Status:` ยังไม่ใช่ `closed` · ตั๋วว่าง (frontier) คือเปิด + ทุกตั๋วใน `Blocked by:` ปิดแล้ว + `Assignee:` ว่าง
**ไฟล์ผลงาน:** [`matrix.md`](matrix.md) (เมทริกซ์ endpoint × มิติ — สร้างโดยตั๋ว 01) · [`findings.md`](findings.md) (ทะเบียนข้อค้นพบ — สร้างโดยตั๋วรีวิวตั๋วแรกที่เจอข้อค้นพบ)

---

## Destination

**ทะเบียนข้อค้นพบ + เมทริกซ์ (ทุก endpoint × 8 มิติ) ที่ครอบ 5 ชั้นของ API** — ทุกช่องในเมทริกซ์มีค่า (`ผ่าน` / `มีข้อค้นพบ #n` / `ไม่เกี่ยว`) ทุกข้อค้นพบมีระดับ P0–P3 และ**คำตัดสินที่ผู้ใช้ยืนยันแล้ว** (แก้ / ลบ / ปล่อยไว้) ปิดท้ายด้วยชุดงานแก้ที่จัดกลุ่มตามไฟล์/ชั้นพร้อมส่งต่อ

ถึงปลายทางเมื่อ: เมทริกซ์ไม่มีช่องว่างแม้แต่ช่องเดียว, ไม่มีข้อค้นพบที่ไม่มีคำตัดสิน, และคนที่จะไปลงมือแก้หยิบชุดงานไปทำได้โดยไม่ต้องกลับมาถามว่า "ข้อนี้ต้องแก้ไหม"

## Notes

**โหมด: วางแผนอย่างเดียว** (ค่าตั้งต้นของ wayfinder — ผู้ใช้ยืนยัน 2026-10-11) ห้ามแก้โค้ด/เอกสาร/DB ในแผนที่นี้ ข้อยกเว้นเดียวคือ P0 ด้านล่าง

**ขอบเขต (ผู้ใช้ยืนยัน 2026-10-11):**

5 ชั้นของ "API":
1. FastAPI `:8000` — `src/api/routers/*`, `src/api/models.py`
2. Next.js `/api/*` — `taxonomy-app/app/api/**/route.ts`
3. Supabase Edge Functions — `taxonomy-app/supabase/functions/*`
4. ฝั่งผู้เรียก — โค้ด frontend ที่ `fetch` เข้า 3 ชั้นบน **และ** ที่คุย PostgREST ตรงด้วย anon key
5. WebSocket — `src/api/custom_websockets.py`

"ประเภท payload" = โครงสร้าง request/response ว่า type ที่ประกาศ (Pydantic / TS interface) ตรงกับที่วิ่งจริงไหม **และ** คำถามว่าควรมีแหล่งความจริงของ type ที่เดียวหรือไม่ (ตั๋ว 08) — ชนิด payload (JSON/multipart/stream/ws) จดเป็นคุณสมบัติต่อแถวในเมทริกซ์

"เอกสารที่อ้าง" = **ทุกไฟล์ในรีโป**ที่กล่าวอ้างเรื่อง endpoint/payload/พอร์ต — `docs/api/*`, `AGENTS.md`, `README.md`, `docs/CURRENT_STATUS.md`, `.agents/rules/*`, docstring และคอมเมนต์ในโค้ด เอกสารภายนอก (FastAPI/Supabase/Next.js) ตรวจเฉพาะจุดที่น่าสงสัย ไม่กวาดทั้งหมด

**8 มิติ:**
1. เอกสาร vs โค้ด
2. ความสอดคล้องของ type ข้ามรอยต่อ
3. การตรวจ input ที่ขอบระบบ
4. สิทธิ์เข้าถึง (`middleware.ts` / FastAPI / Edge Functions)
5. สัญญาเรื่อง error — รวม "200 แต่จริงๆ ล้มเหลว"
6. เส้นตาย / ซ้ำซ้อน
7. ความสม่ำเสมอเชิงรูปแบบ (ชื่อ, `camelCase`/`snake_case`, envelope)
8. ความครอบคลุมของเทสต์ต่อ endpoint

**ระดับการยืนยัน:** อ่านโค้ด + ยิง request จริงกับ stack ในเครื่องสำหรับ endpoint **อ่านอย่างเดียว** · endpoint ที่เขียนข้อมูลยืนยันจากโค้ด + เทสต์ที่มีอยู่ **ห้ามยิงจริง** · ทุกข้อค้นพบติดป้าย `ยิงจริงแล้ว` หรือ `อ่านโค้ดอย่างเดียว`

**ระดับความรุนแรง:**
- **P0** — ความปลอดภัย หรือข้อมูลหาย/เสียแบบแก้คืนไม่ได้
- **P1** — ผิดแบบเงียบ: UI บอกสำเร็จแต่ไม่สำเร็จ, field หายโดยไม่มี error
- **P2** — เอกสารโกหกจนคน/เอเจนต์ทำผิดทาง แต่ระบบยังทำงานถูก
- **P3** — ไม่สม่ำเสมอเชิงรูปแบบ, เส้นตาย, เทสต์ขาด

**คำตัดสิน:** เอเจนต์เสนอพร้อมเหตุผลทุกข้อ ผู้ใช้ยืนยัน **ทีละข้อสำหรับ P0/P1** (ตั๋ว 09) และ **เป็นชุดสำหรับ P2/P3** (ตั๋ว 10)

**ข้อยกเว้น P0:** เจอ P0 ที่โจมตีได้จริงระหว่างรีวิว → **หยุดแล้วรายงานผู้ใช้ทันที** ถ้าผู้ใช้สั่งแก้ ให้แก้เป็นงานแยกนอกแผนที่ แล้วจดลิงก์กลับมาในทะเบียน ห้ามแก้เองโดยไม่ถาม

**P0 ที่เจอแล้วและเลื่อนไว้ (ผู้ใช้ตัดสิน 2026-10-11):** [F-001](findings.md) อ่านทั้ง DB จากอินเทอร์เน็ตผ่าน Cloudflare Tunnel + `/api/supabase/*` · [F-002](findings.md) FastAPI `0.0.0.0` เขียนข้อมูลได้จากเครื่องในเครือข่าย — ทั้งคู่รอตั๋ว 09 โดยมีเงื่อนไขว่า **tunnel ต้องปิดไว้ระหว่างรอ** · เวลารีวิวมิติ 4 ให้ถือว่าแอปเปิดสู่อินเทอร์เน็ต ไม่ใช่แค่ LAN

**หั่นตั๋วรีวิวตามชั้น** (03–05) แยกรอยต่อ (06) และเอกสาร (07) ถ้าตั๋ว 01 พบว่าชั้นไหนใหญ่เกินหนึ่งเซสชัน (~100K tokens) ให้แตกตั๋วชั้นนั้น

**ข้อควรระวัง:** ห้ามเชื่อเอกสาร (รวม AGENTS.md และ auto-memory) โดยไม่ตรวจของจริง — แผนที่ก่อนหน้า (`../rules-workflows-memory-conflicts/`) พิสูจน์แล้วว่าเอกสารในรีโปนี้โกหกได้

**สกิลที่ควรเรียก:** ตั๋ว grilling → `grilling` + `domain-modeling` · ก่อนอ้างว่าตั๋วไหนเสร็จ → `superpowers:verification-before-completion` · ตั๋ว 11 → `to-tickets` (อยู่ที่ `.agents/skills/to-tickets/` เท่านั้น ไม่มีใน `.claude/skills/` — อ่าน SKILL.md ตรง) · สกิล `research` **ไม่ได้ติดตั้ง** — ตั๋ว research ให้ subagent ทำ inline
**MCP ที่ใช้ได้:** `postgres` (ดู schema/RLS จริง), `socraticode` (หาผู้เรียกของแต่ละ endpoint — ใช้ path `D:\` ตัวใหญ่)

## Decisions so far

<!-- หนึ่งบรรทัดต่อตั๋วที่ปิดแล้ว -->

- [01: สำรวจรายการ endpoint ผู้เรียก และข้อกล่าวอ้างในเอกสารทั้งหมด](issues/01-inventory.md): [`matrix.md`](matrix.md) มี 103 แถว (FastAPI 24 · Next.js 25 · Edge 3 · PostgREST ตรง 44 ซึ่ง 22 แถวเป็นเมธอดที่ไม่มีผู้เรียก · WS 7 ที่ไม่มีผู้รับเลย) ไม่ต้องแตกตั๋วรีวิว · เอกสารอ้าง endpoint ที่ไม่มีอยู่ 16 กลุ่ม · **สงสัย P0:** FastAPI เปิด `0.0.0.0:8000` ไม่มี auth ทำให้เส้นที่เขียนข้อมูลข้าม `middleware.ts` ได้ (ยังไม่ยืนยันเรื่อง firewall)

- [02: เกณฑ์ "ผ่าน" ของแต่ละมิติ](issues/02-pass-criteria.md): เกณฑ์ 8 มิติอยู่ในหัว [`matrix.md`](matrix.md) · error สม่ำเสมอภายในชั้น · wire เป็น `snake_case` · เช็ก input มือนับได้ถ้าครบ · read ผ่าน `supabaseAdmin` ต้องอยู่ใน `GATED_READ_PATHS` · ช่องเทสต์ล้มเหลว = P3

- [03: รีวิวชั้น FastAPI](issues/03-review-fastapi.md): กรอกครบ 24 แถวใน [`matrix.md`](matrix.md) (มิติ 2 รอ 06) · ยืนยัน Firewall สำหรับ F-002 (P0) ว่า Allow บน Public Wi-Fi จริง · ยิงจริงครบ 10 เส้นอ่านอย่างเดียว · พบ 8 ข้อค้นพบใหม่ F-003 ถึง F-010 (P1: batch 400 ตกเป็น 500, upload เสี่ยง path/size, mutate singleton config · P2: GET / โฆษณา /web 404, str(e) รั่วใน 500 detail 14 เส้น · P3: เส้นตาย legacy 9 เส้น, camelCase/envelope ไม่ตรงแบบ, ขาดเทสต์ 8 เส้น)

- [04: รีวิวชั้น Next.js `/api/*`](issues/04-review-nextjs-api.md): กรอกครบ 25 แถวใน [`matrix.md`](matrix.md) (มิติ 2 รอ 06) · ยิงจริง 8 เส้นอ่านอย่างเดียว · พบ 7 ข้อค้นพบใหม่ F-011 ถึง F-017 (P1: GET /api/recheck หลุดจาก `GATED_READ_PATHS` อ่าน unreviewed ได้โดยไม่ต้อง unlock, POST /api/similarity/resolve update ไม่เช็ก affected rows คืน 200 false success · P2: ขาด Zod 6 เส้น แครช 500 เมื่อ UUID ผิด, 6 เส้นใช้ DatabaseService anon key ขัดกับ RLS · P3: เส้นตาย 6 เส้น, envelope ไม่ตรงแบบ 5 เส้น, ขาดเทสต์ 4 เส้น)

- [05: รีวิว Edge Functions และ WebSocket](issues/05-review-edge-ws.md): กรอกครบ 10 แถวใน [`matrix.md`](matrix.md) (Edge 3, WS 7, มิติ 2 รอ 06) · ยิงจริง Edge คืน 503 name resolution failed (ไม่มี container ใน Docker) · ยิงจริง WS ต่อติดโดยไม่มี auth และ binary frame ทำ connection แครชหลุดจาก handler ค้างใน memory · พบ 5 ข้อค้นพบใหม่ F-018 ถึง F-022 (P1: Edge generate-embeddings schema ไม่ตรงกับ FastAPI สองด้านพัง 100% · P2: Edge 3 ตัวเป็น proxy ซ้ำซ้อนที่ไม่มี container รันและไม่มีผู้เรียกในแอป, WS /ws ไม่มี auth และไม่ตรวจ frame · P3: WS 7 ตัวเป็น dead code ไม่มีผู้รับใน frontend เลย, ขาดเทสต์ทั้ง 10 ตัว)

- [06: รีวิวรอยต่อระหว่างผู้เรียกกับผู้รับ](issues/06-review-boundaries.md): เติมมิติ 2 ครบทุกแถว และตรวจชั้น 4 (PostgREST ตรง) ทั้ง 44 แถวครบ 8 มิติ · ตรวจ schema และ RLS จริงผ่าน MCP postgres · พบ 5 ข้อค้นพบใหม่ F-023 ถึง F-027 (P1: StorageImport.tsx อ่านตาราง imports โดน RLS บล็อกเงียบคืน [] · P2: bucket uploads ไม่มีใน DB ดาวน์โหลดพัง, DatabaseService มี 20 write methods anon ขัด RLS ซึ่ง 11 จุดพังเงียบและ 9 จุดโดน 403 · P3: 7 read methods ใน DatabaseService เป็น dead code, types ซ้ำกระจัดกระจาย 11 ชุดใน 25 ไฟล์) · นับชั้น 4 ได้ 44 จุด (active UI 16 จุด, พังเงียบรวม 12 จุด) เสนอคำตอบให้ประเด็น Fog: ไม่ต้องย้าย SELECT ที่ RLS true ไปหลัง route, ย้ายเฉพาะ L4-06 ไป GET /api/import/history, และลบ dead code ใน DatabaseService ออกทั้งหมด

- [07: ตรวจข้อกล่าวอ้างในเอกสารทุกข้อเทียบของจริง](issues/07-review-docs-claims.md): ปรับมิติ 1 ทุกแถวที่ไม่มีเอกสารอ้าง (`—`) เป็น `ไม่เกี่ยว` (29 แถว) · ตรวจเอกสารใน `docs/api/*` 5 ไฟล์ (มีชีวิต 2 ไฟล์คือ `api-reference.md` และ `API_ARCHITECTURE.md` แต่มีข้อมูลเท็จ/ขัดแย้ง, อีก 3 ไฟล์เป็น dead/archive) · ตรวจข้อกล่าวอ้างผี 16 กลุ่มใน Section 2 ของ `matrix.md` · ยืนยัน `AGENTS.md` และ `.agents/rules/*` ตรงความจริง 100% · พบ 4 ข้อค้นพบใหม่ F-028 ถึง F-031 (P2: docs/api ขัดแย้งสถาปัตยกรรม/payload ผิด, Next.js docs โฆษณา route ผี 6 เส้น, Edge README อ้าง 4 ฟังก์ชันผีและ OpenAI · P3: สคริปต์/โค้ดตัวอย่างเรียกฟังก์ชันผี) เสนอแนวทางระดับไฟล์: Keep+Rewrite `api-reference.md`, Rewrite `API_ARCHITECTURE.md`, Delete `quick-reference.md`, Archive/Delete `API_CORRECTION.md` และ `analyze-capabilities.md`

- [08: ควรมีแหล่งความจริงของ payload type ที่เดียวไหม](issues/08-single-type-source.md): Grilling ได้ข้อสรุปและผู้ใช้ยืนยันเลือกแนวทาง (d) Two-Pillar Single Source of Truth (Postgres สำหรับ DB schema ผ่าน `supabase gen types`, Pydantic สำหรับ AI contracts ผ่าน `openapi-typescript`, และ Next.js รวมศูนย์ View types ไว้ที่ `taxonomy-app/types/`) · ใช้ on-demand script `npm run gen:types` ไม่ผูก `prebuild` · กำจัด duplicate types 11 รายการใน 25 ไฟล์ (#F-027) · บันทึก [ADR 0004](docs/adr/0004-two-pillar-single-source-of-truth-types.md)

- [09: ตัดสินข้อค้นพบ P0/P1 ทีละข้อ](issues/09-verdict-p0-p1.md): ผู้ใช้ยืนยันคำตัดสินครบทั้ง 9 ข้อ (P0 2 ข้อ, P1 7 ข้อ) · F-001 ปล่อยไว้แก้ทีหลัง · F-002 ปล่อยโค้ดไว้ / แก้กฎข้อ 7 ใน AGENTS.md (ลดระดับเป็น P2 ด้านเอกสาร) · F-003, F-004, F-005, F-018 ลบทิ้ง (dead code) · F-011, F-012, F-023 แก้ไข (fix) ครบถ้วนตามเกณฑ์

- [10: ตัดสินข้อค้นพบ P2/P3 เป็นชุด](issues/10-verdict-p2-p3.md): ผู้ใช้ยืนยันคำตัดสินครบทั้ง 22 ข้อใน 5 ชุด (ลบ dead code ใน FastAPI/Next.js/Edge/WS/DatabaseService, แก้ไขเอกสารและกฎข้อ 7 ใน AGENTS.md, แก้ไข Active routes error/input validation, จัดทำ Types ตาม ADR 0004, และเขียนเทสต์เฉพาะเส้นที่ใช้งานจริง)

## Not yet specified

- **frontend ที่คุย PostgREST ตรงควรย้ายไปหลัง route ไหม** — ได้รับคำตัดสินจากผู้ใช้ครบถ้วนในตั๋ว 09/10: ไม่ย้าย SELECT ที่ RLS true, ย้ายเฉพาะ `StorageImport.tsx` (L4-06) ไปใช้ `GET /api/import/history`, และลบ 20 write methods กับ 7 read methods ใน `DatabaseService` ทิ้ง
- **กลไกกันสัญญา API เน่าซ้ำ** (contract test, gen type อัตโนมัติ, เช็กใน CI) — สรุปแล้วในตั๋ว 08 และ ADR 0004: ใช้ Two-Pillar Type Generation (`npm run gen:types`) ควบคู่กับ typecheck `tsc --noEmit` ใน CI/test suite โดยไม่ต้องทำ contract test รันไทม์เพิ่มเติมนอกเหนือจากนี้

## Out of scope

- **ประสิทธิภาพ** (มิติ 9 เดิม: N+1, payload ใหญ่, pagination) — ต้องวัดโหลดจริง เป็นอีกแผนที่หนึ่ง ถ้าเจอปัญหาชัดระหว่างทางให้จดเป็นหมายเหตุในทะเบียน ไม่ใช่ข้อค้นพบ
- **ลงมือแก้** — แผนที่นี้ส่งต่อชุดงาน ไม่ได้แก้เอง (ยกเว้นเส้นทาง P0 ใน Notes)
- **สร้างระบบล็อกอินจริง** — AGENTS.md ระบุว่า "เป็นโปรเจกต์ ไม่ใช่แพตช์" ข้อค้นพบมิติ 4 ที่ต้องพึ่งล็อกอินจริงให้บันทึกและตัดสินว่า "ปล่อยไว้ — รอโปรเจกต์ auth" ไม่ใช่ออกแบบ auth ในแผนที่นี้
- **กวาดเอกสารภายนอกทั้งหมด** — ตรวจเฉพาะจุดที่น่าสงสัยรายตั๋ว
