# Smart & Impact-Aware Workflow
*(อ้างอิงจากหลักการ SocratiCode AGENTS.md)*

เวิร์กโฟลว์นี้ออกแบบมาเพื่อเพิ่มความแม่นยำในการจัดการ Thai Product Taxonomy โดยเน้นการสืบค้นข้อมูลและการวิเคราะห์ผลกระทบก่อนการตัดสินใจ

---

## 0. Socraticode Setup (ทำครั้งแรกของ session)

ก่อนเริ่มงานใดๆ ตรวจสอบว่า index พร้อม:

```
1. codebase_health {}               → ตรวจ infra
2. codebase_status { projectPath }  → ดูว่า index ครบ 100% หรือยัง
3. [ถ้ายังไม่ index] codebase_index { projectPath: "d:\\product_checker\\check-products" }
4. codebase_watch { action: "start", projectPath }  → เปิด auto-update
```

> ดูรายละเอียดใน [workflow-socraticode-setup.md](workflow-socraticode-setup.md)

---

## 1. Search-First Strategy (ค้นหาเชิงลึก)

ห้ามเดาสุ่มหรือไล่อ่านไฟล์ทั้งหมด ให้เริ่มด้วยเครื่องมือสืบค้น:

**Code Exploration (ใช้ Socraticode ก่อน — แม่นยำกว่า grep):**
- `codebase_search { query: "คำอธิบายสิ่งที่ต้องการ" }` → ค้นหาด้วยภาษาธรรมชาติ (Semantic)
- `codebase_symbols { file: "ไฟล์ที่สนใจ" }` → List functions/classes ในไฟล์
- `codebase_symbol { name: "ชื่อ function" }` → ดู definition + callers + callees แบบ 360°
- `grep_search` → ใช้เมื่อต้องการค้นหา literal string ที่แม่นยำ

**Data Exploration:**
- `mcp_postgres_query` → วิเคราะห์ข้อมูลจริงใน Supabase
- `codebase_context { projectPath }` → ดู project knowledge artifacts (DB schema, API spec)

**Semantic Mapping:** เชื่อมโยงบริบทของปัญหาเข้ากับ Codebase ก่อนลงมือแก้

---

## 2. Impact Analysis & Blast Radius (วิเคราะห์ผลกระทบ)

ก่อนทำการเปลี่ยนแปลง (Modify/Delete) — **ต้องทำทั้งสองระดับ:**

**ระดับโค้ด (Socraticode):**
```
codebase_impact {
  target: "ชื่อไฟล์หรือ symbol ที่จะแก้",
  projectPath: "d:\\product_checker\\check-products"
}
```
→ รู้ทันทีว่าไฟล์ไหนบ้างที่จะพัง ("blast radius")

**ระดับข้อมูล (Postgres):**
- **Impact Query:** รัน SQL เพื่อประเมินว่าการแก้กฎหรือหมวดหมู่จะส่งผลต่อสินค้ากี่รายการ
- **Risk Assessment:** หากความเสี่ยงสูง (>10% ของข้อมูล) ต้องทำ Snapshot หรือ Backup ก่อน

---

## 3. Dynamic Context & Memory (การจัดการความรู้)

รักษา "สมอง" ของโปรเจกต์ให้เป็นปัจจุบัน:

- **Project Artifacts:** ใช้ `codebase_context` ดู `DATABASE_SCHEMA.md` และ `API_ARCHITECTURE.md`
- **Living Documentation:** อ้างอิงและอัปเดตเอกสารเหล่านั้นเมื่อ schema เปลี่ยน
- **Session Continuity:** บันทึกความคืบหน้าใน `CURRENT_STATUS.md` เพื่อให้เอเจนต์ตัวถัดไปทำงานต่อได้ทันที
- **Fact Storage:** บันทึกบทเรียนจาก Bug ลงใน Memory MCP

---

## 4. Rigorous Validation (การตรวจสอบขั้นสูง)

- **Circular Dependency Check:** `codebase_graph_circular { projectPath }` → ตรวจก่อน commit
- **Hybrid Scoring Check:** ตรวจสอบว่าผลลัพธ์ของ Hybrid Algorithm (60/40) ยังอยู่ในเกณฑ์ที่รับได้
- **Visual Integrity:** ใช้ Puppeteer/Domscribe ตรวจสอบ UI ภาษาไทยตามกฎ Antigravity
- **Regression Testing:** รัน Pytest/Jest เพื่อยืนยันว่าการแก้ไขไม่ทำให้ส่วนอื่นพัง
