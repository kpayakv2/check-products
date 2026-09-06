# ทะเบียนข้อขัดแย้งฉบับสมบูรณ์

สร้างโดย[ตั๋ว 03](issues/03-conflict-register.md), กวาดครบทั้งสี่ชั้นตาม Notes ของ[แผนที่](map.md) ทุกแถวยืนยันกับของจริงแล้ว ไม่ใช่อ่านเอกสารเทียบเอกสาร วันที่ตรวจ: 2026-09-06

## เกณฑ์การนับขอบเขต (ตัดสินใจในตั๋วนี้)

**เอกสารรายงาน/แผนงานที่ลงวันที่หรืออ่านออกว่าเป็นบันทึกจุดหนึ่งในอดีต ไม่นับเป็นข้อขัดแย้ง** แม้จะพูดถึงตัวเลข/สถาปัตยกรรมที่เปลี่ยนไปแล้ว ตราบใดที่ไม่มีจุดใดใน **สายทางเดินที่ยังมีชีวิต** (README → `docs/INDEX.md` → ไฟล์ที่ INDEX ชี้ไป, หรือ `AGENTS.md`/`.agents/rules/`/`.agents/workflows/`) ชี้กลับไปอ้างมันเป็นความจริงปัจจุบัน

**เหตุผล:** ระหว่างกวาดพบว่า `docs/development/`, `docs/architecture/`, `docs/reports/` มีเอกสารอย่างน้อย **~25 ไฟล์** ที่บรรยายสถาปัตยกรรมรุ่นเก่าทั้งดุ้น (`api_server.py`/`web_server.py` ที่ระดับราก, Flask Web UI พอร์ต 5000, ก่อนย้ายเป็น `src/api/` + Next.js + Supabase) — เช่น `INTEGRATION_STRATEGY.md`, `MIGRATION_PLAN.md`, `PROJECT_STRUCTURE.md`, `EMBEDDING_INTEGRATION.md`, `module-relationships-analysis.md`, `python-files-inventory.md`, `code-duplication-analysis.md`, `code-usage-analysis.md`, `cleanup-results.md`, `migration-*.md`, `capabilities-summary.md`, `STATUS_CHECK_FINAL.md`, `FINAL_REPORT.md`, `BACKEND_UPDATED.md`, `README_CLASSIFIER.md`, `API_CORRECTION.md`, `INTEGRATION_STEPS.md`, `TEST_SUMMARY.md`, `CATEGORY_ALGORITHM_TEST_REPORT.md` เนื้อหานี้ไม่มีจุดไหนถูกเชื่อมจาก `docs/INDEX.md` เลย (ตรวจแล้วว่าทุกลิงก์ใน INDEX ชี้ไปไฟล์ที่ยังมีจริง และไม่มีลิงก์ไหนชี้เข้าไปในกองนี้) จึงเป็น**ซากประวัติศาสตร์ที่ไม่มีใครเดินผ่าน** ไม่ใช่ข้อขัดแย้งที่ทำร้ายใคร การแจกแจงทีละบรรทัดในกองนี้เกินขนาดของตั๋วนี้และไม่มีประโยชน์ต่อตั๋วอื่น — ถ้าจะจัดการทั้งกอง (ย้ายเข้า archive/ หรือลบ) ควรเป็นตั๋วใหม่ต่างหาก ไม่ใช่ส่วนหนึ่งของทะเบียนนี้

---

## A. แก้ไปแล้ว (ระหว่างตั๋ว 01/02/03)

| # | ไฟล์:บรรทัด | ข้ออ้างเดิม | ของจริง | สถานะ |
|---|---|---|---|---|
| 1 | `.agents/rules/rules-ai-agent.md:29` | สั่งอ่าน `GEMINI.md` เป็นรัฐธรรมนูญ | ชี้ไป `AGENTS.md` แล้ว | ✅ แก้ (ตั๋ว 01) |
| 2 | `.agents/rules/rules-antigravity.md` (ตัวอย่าง Good) | อ้าง `e2e/antigravity-specialist.spec.ts` | ไฟล์ถูกลบ 2026-08-30 (ไม่มี `expect()`) — เปลี่ยนเป็นชี้ไป AGENTS.md §Testing | ✅ แก้ (ตั๋ว 01) |
| 3 | `docs/INDEX.md` | ชี้ `GEMINI.md`, ลิงก์ตาย `../API_ARCHITECTURE.md`, `../SUPABASE_SYSTEM_ARCHITECTURE.md`, ลายเซ็น Gemini CLI Agent | แก้ path, ชี้ AGENTS.md, ลบลิงก์ตาย/ลายเซ็นเก่า | ✅ แก้ (ตั๋ว 01) |
| 4 | `docs/PRD.md:85` (NFR1) | "ห้าม localhost ทุกที่ ใช้ 127.0.0.1 เสมอ" | ขัดกับกฎ frontend-URL จริง — ใส่เงื่อนไข + ลิงก์ AGENTS.md | ✅ แก้ (ตั๋ว 01) |
| 5 | `MEMORY.md` + `repo-audit-2026-08-19.md` | jest.config ยังไม่แก้, CLAUDE.md เก่า | ทั้งคู่แก้จริงแล้วนานแล้ว | ✅ แก้ (ตั๋ว 02) |
| 6 | `MEMORY.md` + `project-state-2026-08-28.md` | branch ยังไม่ merge | merge เข้า main แล้ว (`328b1467`) | ✅ แก้ (ตั๋ว 02) |
| 7 | `.mcp.json`, `.claude/settings.local.json`, `AGENTS.md` | มี `memory` MCP server | ถอดแล้ว — 0 ไบต์มา 2.5 สัปดาห์ | ✅ แก้ (ตั๋ว 02) |
| 8 | `.agents/workflows/smart_impact_workflow.md` (ข้อ 3) | "บันทึกบทเรียนลงใน Memory MCP" | Memory MCP ถูกถอดแล้ว — เปลี่ยนเป็น Team Memory | ✅ แก้ (ตั๋ว 03) |
| 9 | `.agents/workflows/smart_impact_workflow.md` (ข้อ 4) | "ใช้ Puppeteer/Domscribe" | ไม่เคยติดตั้งจริง — เปลี่ยนเป็น Playwright | ✅ แก้ (ตั๋ว 03) |
| 10 | `.agents/workflows/workflow-antigravity-verification.md` | อ้าง `e2e/antigravity-specialist.spec.ts` | ไฟล์เดียวกับ #2 — แก้ให้ชี้ AGENTS.md §Testing แทน | ✅ แก้ (ตั๋ว 03) |
| 11 | `AGENTS.md` ตาราง Key Directories | `supabase` ที่ระดับราก | ไม่มีจริง แก้เป็น `taxonomy-app/supabase` แล้ว | ✅ แก้ (ตั๋ว 03) |

## B. ยังเปิดอยู่ — รอตั๋วที่เกี่ยวข้อง

| # | ไฟล์:บรรทัด | ข้ออ้าง | ของจริง (ยืนยันแล้ว) | ชั้น | รอตั๋ว |
|---|---|---|---|---|---|
| 12 | `GEMINI.md:21` กฎข้อ 6 | "Benchmark ≥ 72% F1" | ตัวเลขปลอม, เทสต์ต้นตอถูกลบแล้ว | Rule (พ้นวาระ) | [04](issues/04-gemini-md-fate.md) → [07](issues/07-accuracy-rule-wording.md) |
| 13 | `GEMINI.md:23` กฎข้อ 8 | "บังคับ MCP มาตรฐาน 6 ตัว" (รวม domscribe/puppeteer/filesystem/memory) | `.mcp.json` เหลือ 3 ตัวจริง (postgres, socraticode, sequential-thinking) — memory เพิ่งถูกถอด | Rule (พ้นวาระ) | [04](issues/04-gemini-md-fate.md) |
| 14 | `GEMINI.md:30-36` กฎข้อ 7 | Smart Testing Matrix 5 ขั้น "ห้ามข้าม" ขั้น 3 ใช้ Puppeteer+Domscribe | เครื่องมือไม่มีจริง | Rule (พ้นวาระ) | [04](issues/04-gemini-md-fate.md) |
| 15 | `GEMINI.md:80-86` LAN Access | สำเนาซ้ำของ `.agents/rules/rules-windows.md` | ของจริงยังถูกต้อง แต่มีบ้านอยู่แล้วที่ rules-windows.md — สำเนานี้ทิ้งได้เลย | Rule (ซ้ำ) | [04](issues/04-gemini-md-fate.md) |
| 16 | `GEMINI.md:119` ตาราง Key Directories | `supabase/` ที่ระดับราก | ไม่มีจริง คือ `taxonomy-app/supabase/` | Rule (พ้นวาระ) | [04](issues/04-gemini-md-fate.md) |
| 17 | `GEMINI.md:69-75`, `:61-67` สารบัญ | workflows 4 ไฟล์ (จริง 5), skills 4 ตัว (จริง 13) | ตกหล่นเพราะเขียนไว้ก่อนไฟล์ใหม่จะถูกใส่เข้ามา | Rule (พ้นวาระ) | [04](issues/04-gemini-md-fate.md) |
| 18 | `.agents/workflows/workflow-new-feature.md:46` | "ตรวจสอบ F1-score/Accuracy ≥ 72%" | เกณฑ์เดียวกับ #12 | Rule (ล้าสมัย) | [07](issues/07-accuracy-rule-wording.md) |
| 19 | `docs/development/architecture.md:4` | "กฎเหล็ก ... ตาม GEMINI.md" | GEMINI.md ไม่ใช่แหล่งกฎแล้ว + ตัวเลข 72% ปลอม | Rule อ้างผิดที่ + เลขผิด | [07](issues/07-accuracy-rule-wording.md) |
| 20 | `docs/PRD.md:12` | "ไม่ต่ำกว่า 72% บน Benchmark" | เลขปลอมเดียวกับ #12 | Rule/Wayfinding ปน | [07](issues/07-accuracy-rule-wording.md) |
| 21 | `docs/INDEX.md:50`, `docs/api/analyze-capabilities.md:18`, `docs/README_CLASSIFIER.md:9,133` | อ้าง "72%" เป็นมาตรฐานปัจจุบัน | เลขปลอมเดียวกับ #12 (README_CLASSIFIER.md อยู่ในกองประวัติศาสตร์ที่ยกเว้นแล้ว แต่ INDEX.md/analyze-capabilities.md อยู่ในสายทางเดินที่ยังมีชีวิต) | Wayfinding พูดเป็น Rule | [07](issues/07-accuracy-rule-wording.md) |
| 22 | `START_HERE.md` | อ้างเอกสาร 6 ไฟล์ที่ไม่มีจริง, สั่ง `python api_server.py` ที่ราก, อ้าง `web_server.py` | ไฟล์ทั้งหมดไม่มีจริง/ย้ายที่แล้ว | Wayfinding พ้นวาระ | [05](issues/05-start-here-fate.md) |
| 23 | `.agents/skills/` vs `.claude/skills/` | สกิลซ้ำไบต์ต่อไบต์ 9 ตัว, `AGENTS.md` บรรยาย `.agents/skills/` ว่ามี 4 ตัว (จริง 13) | โหลดจริงแค่ `.claude/skills/` | Rule/config ซ้ำ | [06](issues/06-duplicate-skills.md) |

## C. วินิจฉัยแล้วว่าไม่ใช่ข้อขัดแย้ง (Status ที่ถูกต้อง)

- **`docs/reports/*.md` ที่ลงวันที่ชัดเจน** (`REPO_AUDIT_2026-08-19.md` และรายงานที่คล้ายกัน) — บันทึกค่าที่วัดได้ ณ เวลานั้นถูกต้องแล้ว ไม่ใช่ข้อขัดแย้ง แม้ตัวเลขจะต่างจากปัจจุบัน
- **กองเอกสารสถาปัตยกรรมเก่า ~25 ไฟล์** ใน `docs/development/`, `docs/architecture/`, `docs/reports/` ที่ระบุไว้ในเกณฑ์ขอบเขตด้านบน — ยืนยันแล้วว่า `docs/INDEX.md` ไม่ได้ชี้เข้าไปเลย จึงไม่ทำร้ายใครที่เดินตามสายทางเดินปกติ

## สรุปจำนวน

- แก้ไปแล้ว: 11 จุด
- ยังเปิดอยู่ — มีตั๋วรออยู่แล้วทุกจุด ไม่มีจุดไหนกำพร้า: 12 จุด
- วินิจฉัยว่าไม่ใช่ข้อขัดแย้ง: 2 กลุ่ม (รายงานลงวันที่ + กองเอกสารเก่า ~25 ไฟล์)
