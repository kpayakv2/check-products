# 📊 Current Project Status (Compact Context)
*Last Updated: 2026-08-29 (รอบสาม)*

## 🎯 Current Focus
- **งานจริงของระบบ:** รับไฟล์รายการสินค้าใหม่ → เทียบกับสินค้าในสตอก 3,103 รายการ → **เอาเฉพาะตัวที่ยังไม่เคยมี**
- สินค้าเก่า 3,103 รายการที่คนจัดหมวดไว้ = ข้อมูลตั้งต้นให้ ML ฝึกทั้งการจัดหมวดและการตรวจซ้ำ
- ในสตอกเองก็ยังมีของซ้ำอยู่ การเก็บกวาดสตอกจึงป้อนข้อมูลฝึกให้ ML ไปในตัว — สองงานนี้เป็นวงจรเดียวกัน
- User Workflow: **Import (Wizard) → Dedup → Review (Verify/Recheck) → AI Learns → Better Dedup**

## 📈 ตัวเลขที่วัดได้จริง (2026-08-28)
| ตัวชี้วัด | ค่า |
|---|---|
| accuracy การจัดหมวด (top-1 หมวดย่อย) | **72.3%** วัดจาก test set 595 รายการที่กันไว้ |
| AI ตรวจซ้ำหมวดเก่าแล้วเห็นตรงกับคน | 79.5% (เหลือให้คนดู 635 รายการ) |
| ไฟล์สินค้าใหม่ 405 รายการ | มีในสตอกแล้ว 37 / ก้ำกึ่ง 146-147 / **ของใหม่ 221-222** (wizard ขั้นจัดหมวดตอนนี้ทำเฉพาะกลุ่มนี้แล้ว ไม่ทำทั้ง 405) |

> ⚠️ ตัวเลข "72%" ที่เคยอ้างในเอกสารเก่า**ไม่เคยถูกวัดจริง** มาจาก `tests/benchmark_similarity.py` ที่ print ค่า hardcode (ลบไฟล์นั้นแล้ว) ตัวเลขข้างบนมาจาก `tests/integration/test_classification_accuracy.py` ที่วัดของจริง

---

## ✅ Session 29 ส.ค. 2569 (รอบสาม) — บั๊กหน้า /taxonomy และ /ai-brain

เจ้าของงานแจ้งว่าสองหน้านี้มีปัญหา ตรวจแล้วเจอของจริง 3 เรื่อง (ไม่ใช่ของใหม่จากรอบสอง)

### 🔴 /taxonomy เขียนอะไรไม่ได้เลย — และ "ลบ" หลอกว่าสำเร็จ
ทุกหน้าคุยกับ Supabase ด้วย **anon key** แต่ policy ของ `taxonomy_nodes` /
`synonym_lemmas` / `synonym_terms` / `system_settings` ให้เขียนได้เฉพาะ
`taxonomy_editor` / `taxonomy_admin` (มาตั้งแต่ migration แรก `20250924120000` ไม่ใช่ของรอบที่แล้ว)
ยิงจริงกับ DB แล้วได้: `INSERT` → **42501**, `UPDATE` → **200 พร้อม array ว่าง**, `DELETE` → **204 ที่ไม่ลบอะไร**
สองอันหลัง**ไม่มี error** หน้าเว็บจึงขึ้น "ลบเรียบร้อยแล้ว" ทั้งที่ข้อมูลยังอยู่ครบ
- [x] `utils/admin-db.ts` (ใหม่) — create/update/delete/upsert ผ่าน `supabaseAdmin`
      คืน `null`/`false` เมื่อไม่มีแถวถูกแตะ แทนที่จะเงียบแล้วดูเหมือนสำเร็จ
- [x] `/api/taxonomy`, `/api/synonyms` (+ `[id]`) เขียนผ่าน service role และตอบ **404** เมื่อไม่มีแถวตรง
- [x] `/api/settings` (ใหม่) — หน้า `/settings` เดิมอ่าน `system_settings` ตรง ๆ ได้ **406 ทุกครั้งที่โหลด**
- [x] อีกบั๊กที่ซ้อนอยู่: ฟอร์มส่ง `parent_id: ''` ไปให้คอลัมน์ uuid → `22P02` สร้างหมวดไม่ได้ตั้งแต่แรก
      แก้ด้วย `z.preprocess` ที่ route + ตัดคีย์ค่าว่างทิ้งใน `admin-db`
- [x] `POST /api/synonyms` บังคับให้ส่ง `terms` มา **แล้วไม่เคยบันทึก terms เลย** — คำพ้องที่พิมพ์หายเงียบ ๆ
- [x] ตัวนับ "Global Nodes" นับเฉพาะโหนดบนสุด (12) → นับทั้งต้น = **134**

### 🔴 /ai-brain โชว์ "ความมั่นใจในการปัดตก 85%" ที่ฝังไว้ในโค้ด
`src/api/routers/learn.py` เขียนว่า `average_confidence = 85.0 # Fixed or from training history`
ทั้งที่ `ml_feedback_learning.py` คำนวณค่าจริงได้อยู่แล้วแต่ไม่มีใครเรียกใช้
- [x] ตอนเทรนเก็บ `average_confidence` (ค่าเฉลี่ยความน่าจะเป็นสูงสุดบนชุดทดสอบ) ลง training history
- [x] `/learn/status` ส่งค่าจริง และส่ง `null` เมื่อโมเดลถูกเทรนไว้ก่อนมีฟิลด์นี้
- [x] หน้าเว็บ**ซ่อนการ์ดไปเลย**เมื่อไม่มีค่า (ไม่โชว์ 0% ไม่เดา) — โมเดลปัจจุบันเข้าเคสนี้จนกว่าจะกด retrain

### ⚠️ เรื่องที่เกิดจากเซสชันนี้เอง
`npm run build` ตอนตรวจงานรอบสอง **ทับ `.next` ของ dev server ที่เปิดค้างไว้** ทำให้หน้าเว็บ 404
ทุก chunk จนกว่าจะรัน `npm run dev` ใหม่ — ไม่ใช่บั๊กของโค้ด

### ผลการตรวจสอบ (รอบสาม)
| อย่าง | ผล |
|---|---|
| `pytest` | **163 passed / 7 skipped / 0 failed** |
| `npx jest --ci` | **23 suite ผ่าน / 4 พัง (env เดิม), 168 เทสต์ผ่าน 0 ตก** |
| `npx tsc --noEmit` | 9 error ตาม baseline |
| ยิง API จริงกับ DB จริง | create 201 · update 200 · update id ที่ไม่มี 404 · delete 200 · delete ซ้ำ **404** · synonym สร้างพร้อม term · settings โหลดได้ไม่มี 406 · **ลบ probe ออกครบ ไม่มีขยะค้างใน DB** |

---

## ✅ Session 29 ส.ค. 2569 (รอบสอง) — แก้ status mismatch, ลบ /reports, เก็บกวาดของตาย

ทำตามลำดับที่รอบก่อนเสนอไว้ครบทั้ง 4 ข้อ บนสาขา `fix/status-mismatch-and-page-cleanup`
(commit งานค้าง 24 ไฟล์ของรอบก่อนเป็น 5 commit ก่อนเริ่ม)

### 1. status mismatch — จุดเดียว แก้ได้ 3 หน้า
- [x] `utils/product-status.ts` (ใหม่) — `ProductStatus` + `PENDING_REVIEW_STATUSES`
      **แยกไฟล์เพราะ `utils/supabase.ts` สร้าง client ตอน import** โค้ดฝั่ง server และเทสต์
      จึงอ้างค่าคงที่ได้โดยไม่ต้องมี env ของ client
- [x] `getDashboardStats` นับสองด่านจริง (368) · `getProducts` รับ array แล้วใช้ `.in()`
- [x] `POST /api/products` เคยสร้างสินค้าเป็น `'pending'` = มองไม่เห็นทุกหน้า → `pending_review_category`
- [x] `__tests__/api/products/route.test.ts` เดิมทดสอบแต่ fixture ของตัวเอง เขียนใหม่ให้ยิง route จริง
- [x] `jest.setup.js` mock framer-motion เป็นรายการแท็กตายตัว เจอ `motion.tr` แล้วพัง → เปลี่ยนเป็น Proxy

### 2. `/products` เป็นหน้าดูสตอกอย่างเดียว
- [x] ตัดปุ่ม approve/reject ออก สินค้าที่ยังรอตรวจแสดงป้ายและลิงก์ไป `/data-quality` แทน
- [x] **เจอบั๊กที่ใหญ่กว่าที่คิด:** `getProducts` ดึง 50 แถวมากรองในเบราว์เซอร์ = ค้นได้แค่ 50 ตัวล่าสุด
      จาก 3,103 → เพิ่ม `searchProducts()` (ค้น/กรอง/แบ่งหน้าที่ Postgres) + `getProductStatusCounts()`
- [x] คำค้นถูกตัด `, ( ) * % \` ทิ้งก่อนเข้า `.or()` เพราะอักขระพวกนี้เป็นไวยากรณ์ของ PostgREST

### 3. ลบ `/reports` ยุบเข้าหน้าแรก — ไม่เหลือเลขปลอมสักตัว
- [x] ลบทั้งหน้า (229 บรรทัด) + เมนู + `e2e/report-dashboard.spec.ts`
- [x] หน้าแรกเองก็มีเลขปลอม: `Accuracy 99.8%`, `Latency 12ms`, ป้าย `+8.2%/+4.1%/+12.5%` → ลบหมด
- [x] ตัวเลขใหม่มาจาก DB ทั้งหมด: งานค้างแยกด่าน (147/221), คู่ซ้ำ 1,781 (คนตัดสินแล้ว 1,381),
      ตรวจวันนี้นับจาก `review_history`, และ **AI ตรงกับคน 79.6%** จาก function ใหม่
      `recheck_agreement_stats()` (migration `20260829000000`)
- [x] **ตั้งใจไม่โชว์ 72.3%** — มาจากการรันเทสต์กับ test split ไม่ใช่ค่าที่ query ได้ ถ้า hardcode ไว้ก็เน่าซ้ำรอยเดิม
- [x] แก้ลิงก์ตายบนหน้าแรก 2 จุด (`/synonyms` ที่ยุบไปแล้ว, ปุ่มรีวิวที่ชี้ไป `/products`)

### 4. ลบของตาย
- [x] `PendingTab` + `ApprovalStep` + `CategorySelector` (757 บรรทัด) และ route `pending` /
      `process` / `process-local` / `process-storage` / `approve` — เหลือ `commit` เส้นเดียว
- [x] `/import` เหลือ Wizard + ประวัติการนำเข้า

### 🔍 สองบั๊กที่เจอจากการเปิดเบราว์เซอร์จริง (เทสต์จับไม่ได้ เพราะมันเงียบสนิท)
- **`recheck_agreement_stats()` เรียกไม่ได้จากหน้าเว็บ** — migration แรก grant ให้แค่
  `authenticated`/`service_role` แต่หน้าแรกเป็น client component ใช้ anon key → `42501 permission denied`
  แก้ด้วย migration `20260829000001` (grant anon — ไม่เพิ่มการเปิดเผย เพราะ function เป็น SECURITY INVOKER
  บนสองตารางที่ anon อ่านทีละแถวได้อยู่แล้ว)
- **ประวัติการนำเข้าว่างเปล่าถาวร** — `ImportHistory` อ่านตาราง `imports` ตรงๆ ด้วย anon key
  แต่ migration `20260828000000` จำกัด SELECT ไว้ที่ editor/admin → ได้ array ว่าง **ไม่มี error**
  แก้ด้วย `GET /api/import/history` ที่ใช้ service role

### ⚠️ ยังค้าง (รากเดียวกัน แต่อยู่นอกขอบเขตที่ตกลงกันรอบนี้)
~~`/settings` อ่าน/เขียน `system_settings` ด้วย anon key~~ — **แก้แล้วในรอบสาม** (`/api/settings`)
(ตารางที่จำกัดสิทธิ์เหลืออีก: `product_attributes` — `getProducts`/`getProduct` ยัง embed มาด้วย
จะได้ array ว่างเงียบๆ เหมือนกัน แต่ตอนนี้ยังไม่มีหน้าไหนแสดงผลจริง)

### ผลการตรวจสอบ
| อย่าง | ผล |
|---|---|
| `npx jest --ci` | **19 suite ผ่าน / 4 พัง (env เดิม), 152 เทสต์ผ่าน 0 ตก** (ก่อนเริ่ม: 14 ผ่าน/4 พัง 134 เทสต์) |
| `npx tsc --noEmit` | **9 error ตาม baseline** (เหลือแต่ `e2e/` + `__tests__/integration/`) |
| `npm run build` | ผ่าน |
| เบราว์เซอร์จริง + DB จริง | หน้าแรกขึ้น 368 (147/221), 3,103, 1,781, **79.6%** · `/products` ค้น "ยาสีฟัน" ได้ 46 จาก 3,508 (ตรงกับ SQL) · `/import` มีประวัติจริง · ไม่มี console error |

> ทุกอย่างเขียนแบบ TDD — เทสต์ที่แดงก่อนแล้วค่อยแก้โค้ด (ยกเว้นการลบไฟล์ตาย)

---

## ✅ Session 29 ส.ค. 2569 — ตรวจหน้าเพจ/คอมโพเนนต์ทั้งแอป แล้วแก้บั๊ก Import Wizard

ตรวจ 9 หน้าและ 26 คอมโพเนนต์ พบบั๊กจริง 4 ข้อ กระจุกอยู่ใน Import Wizard ซึ่งเป็นเส้นทางหลักที่เขียนข้อมูลลงฐานข้อมูล

### 🔴 กันบันทึกซ้ำ (ข้อที่หนักที่สุด — เขียนข้อมูลผิดลง DB จริง)
แถบขั้นตอนให้กดย้อนกลับได้ กลับไปขั้นตรวจของซ้ำแล้วเดินหน้าใหม่ = `commitDedup` ยิงรอบสอง
และ route ใช้ `.insert()` เปล่า ไม่มี guard → ได้ `imports` ใหม่ทั้งใบ + `products` ซ้ำทุกแถว
ซึ่งย้อนกลับไปทำให้การตรวจของซ้ำรอบหน้าเพี้ยนตามไปด้วย
- [x] `WizardTab.tsx` — `wizard_run_id` หนึ่งค่าต่อหนึ่งรอบ ส่งไปกับทุก commit เปลี่ยนใหม่เมื่อกดเริ่มใหม่เท่านั้น
      **มี fallback ให้ `crypto.randomUUID`** เพราะ LAN เข้าผ่าน `http://192.168.x.x` ไม่ใช่ secure context ฟังก์ชันนี้จะเป็น `undefined`
- [x] `app/api/import/commit/route.ts` — เช็ค `metadata->>wizard_run_id` **ก่อน** `embedAll` (ซึ่งกินเวลานาน เป็นช่องให้กดซ้ำ)
      เจอแล้วคืนผลของ batch เดิมพร้อมคีย์ `products` ครบ (ขั้นจัดหมวดใช้ทำ map ชื่อ → id)
- [x] migration `20260828000004` — unique index บน `(metadata->>'wizard_run_id')` เพราะลำพัง SELECT-แล้ว-INSERT ยังแข่งกันได้
      route ดัก `23505` แล้วคืนผลของ batch ที่ชนะแทนการโยน error
- [x] **กับดักที่เกือบพลาด:** `.update()` ตอนจบเขียนทับ `metadata` ทั้งก้อน ถ้าไม่ใส่ `wizard_run_id` กลับไปด้วย คีย์จะหายและ guard ตายตั้งแต่รอบแรก
- [x] `product_category_suggestions` ก็ insert ซ้ำได้ → ล้างของ batch เดียวกันก่อน insert

### 🔴 หน้าสรุปโชว์ผลของรอบก่อน / ล้มเหลวแบบเงียบ
- [x] `onReset` ไม่ได้ล้าง `saveResult`/`saveError` → รอบถัดไปที่บันทึกพลาดจะเอาหมวดหมู่ไปแปะสินค้าของ batch เก่า
- [x] `commitCategories` เคย `return` เงียบ ๆ 2 จุด ผู้ใช้เห็น "บันทึกเข้าฐานข้อมูลแล้ว" ทั้งที่หมวดหมู่ไม่ลง DB เลย → ขึ้น `saveError` ทั้งสองกรณี

### 🔴 ปุ่มรีวิวของซ้ำเป็นปุ่มหลอก
`handleBulkAction` มีบรรทัดเดียวคือล้างการเลือกทิ้ง ปุ่ม "รวมเป็นของชิ้นเดียวกัน"/"แยกเป็นของใหม่" ไม่เปลี่ยน `_bucket` เลย
และไม่มี action รายตัว → ทุกอย่างในโซนรีวิวออกไปเป็น `'review'` เหมือนเดิมหมด ช่องค้นหาก็ไม่มี `onChange`
- [x] `DeduplicationStep.tsx` — เปลี่ยนเป็นตรวจทีละรายการ + คีย์ลัด `A`/`D`/`S` และลูกศร **รวมผังแป้นไทย `ฟ`/`ก`/`ห`**
      ยกแพตเทิร์นมาจาก `DeduplicationTab.tsx` (หน้าตรวจของซ้ำในคลัง) ให้ผู้ใช้ไม่ต้องจำสองชุด
- [x] **ต่างจากต้นแบบตรงที่ยิง API รายตัวไม่ได้** — สินค้ายังไม่มีอยู่ใน DB เพิ่งถูกสร้างตอน commit จึงเก็บผลตัดสินใน state ก่อน

### 🧹 ล้างบ้าน
- [x] `types/import.ts` — `WizardItem` / `DedupResults` / `SaveResult` คุมข้อมูลที่ไหลผ่าน 5 ขั้น
      เลิกใช้ `}: any)` ใน 3 คอมโพเนนต์ **พอใส่ type แล้วเจอจุดหละหลวมที่ `any` บังไว้ทันที 8 จุด** (เช่น `_confidence` อาจ undefined)
      `DedupBucket` ดึงมาจาก `utils/price.ts` ที่มีอยู่แล้ว ไม่ประกาศซ้ำ
- [x] ลบ dead component 5 ไฟล์ (2,337 บรรทัด) ที่ถูกเขียนแทนลงในหน้าไปแล้ว — **เก็บ `HybridSearch.tsx` ไว้** เพราะไม่ได้ถูกแทน
      เป็นฟีเจอร์ค้างที่ยังไม่ต่อเข้าแอป และ `settings/page.tsx` ยังมีตัวตั้งค่า `hybridSearchEnabled` รออยู่

### 🧪 เทสต์ที่เคยเชื่อไม่ได้
- [x] ลบ 4 suite ที่ "ผ่าน" แต่ **นิยาม mock component ในไฟล์เทสต์เองแล้ว render อันนั้น** (แก้โค้ดจริงพังยังไงก็ยังเขียว)
- [x] เขียน `ColumnMappingStep.test.tsx` ใหม่ทั้งไฟล์ และแก้ `WizardLayout.test.tsx` (assert ข้อความ UI ที่ถูกรื้อไปแล้ว)
- [x] เพิ่ม `WizardTab.test.tsx` (6), `Import/DeduplicationStep.test.tsx` (10), `api/import/commit.test.ts` (5) คุมบั๊กทั้ง 4 ข้อ
- [x] `jest.setup.js` — polyfill `AbortSignal.timeout` (jsdom ไม่มี ทำให้ `DeduplicationStep` ตกไปทาง fallback สุ่ม เทสต์เลยผ่านบ้างไม่ผ่านบ้าง)
      และกัน `window` ให้เทสต์ API route รันบน environment `node` ได้

### ผลการตรวจสอบ
| อย่าง | ก่อน | หลัง |
|---|---|---|
| `npx jest --ci` | 12 ผ่าน / 8 พัง, 20 เทสต์ตก | **14 ผ่าน / 4 พัง, 0 เทสต์ตก** (4 ที่เหลือต้องใช้ Supabase credentials) |
| `npx tsc --noEmit` | 11 error | **9 error** (เหลือแต่ใน `e2e/` + `__tests__/integration/` ตาม baseline) |
| ยิง commit ซ้ำด้วย run id เดิมบน DB จริง | ได้ 2 batch / 4 แถว | **1 batch / 2 แถว, ครั้งที่สองคืน `reused: true`** |

### ⚠️ ยังค้างอยู่ (ไม่ได้เกิดจากงานรอบนี้)
- **ชุด e2e เน่าทั้งชุด** — `npx playwright test` ผ่าน 2 จาก 20 specs assert ข้อความอย่าง
  `เลือกวิธีการ Import` ที่**ไม่มีอยู่ในโค้ดแล้ว** และ `real-user-workflows.spec.ts` collect ไม่ผ่านเลย
  เพราะ import `__tests__/setup/database-setup.ts` ที่ throw เมื่อไม่มี env — เป็นปัญหาเดียวกับ 4 jest suite ที่เหลือ
  (playwright ไม่ได้โหลด `.env.local`) **ควรเป็นงานรอบถัดไป**
- `__tests__/api/products/route.test.ts` ยังเป็นเทสต์ที่ทดสอบแต่ฟังก์ชันสร้าง fixture ไม่ได้แตะ route จริง
- `any` ที่เหลือ ~56 จุดใน API routes และ `icon: any` (ตั้งใจไม่แตะ เพื่อให้ diff รีวิวได้)

---

## 🔎 ผลตรวจ 9 หน้า (29 ส.ค. 2569) — ✅ **แก้ครบแล้วในรอบสอง (ดูหัวข้อบนสุด)** เก็บไว้เป็นบันทึกว่าเจออะไร

ตรวจครบทั้ง 9 หน้าหลังแก้บั๊ก wizard เสร็จ ข้อสรุป: **จำนวนหน้าไม่ใช่ปัญหา
แต่ 4 หน้าแสดงข้อมูลผิดหรือว่างเปล่าถาวร** ซึ่งอันตรายกว่า เพราะคนตัดสินใจจากตัวเลขที่ไม่จริง

### 🔥 ปัญหาราก: `status` ไม่ตรงกัน 2 ชุด (จุดเดียว พัง 3 หน้า)

UI เก่าเชื่อว่าสถานะคือ `'pending'` แต่ pipeline จริงที่ `commit/route.ts` เขียนใช้
`pending_review_category` / `pending_review_dedup` — **ไม่มีสินค้าสักตัวที่ status เป็น `'pending'` ในฐานข้อมูล**

| จุดที่ยังใช้ `'pending'` | ผลที่เกิด |
|---|---|
| `app/products/page.tsx:56` (default filter) | เปิดหน้ามาว่างเปล่า |
| `app/products/page.tsx:206` (ตัวนับ) | ขึ้น 0 เสมอ |
| `app/products/page.tsx:413` (เงื่อนไขปุ่ม) | ปุ่ม approve/reject ไม่มีวันโผล่ |
| `utils/supabase.ts:753` (`getDashboardStats`) | หน้าแรกนับงานค้าง 0 ทั้งที่จริงมี **368** |

### หน้าที่มีปัญหา เรียงตามความอันตราย

1. **🔴 `/reports` ปลอมทั้งหน้า** — ไม่มี `fetch` สักบรรทัด มีแต่ `setTimeout(800)` ที่คอมเมนต์เขียนว่า
   *"Artificial delay for premium loading feel"* แล้วตามด้วยตัวเลข hardcode 4 ตัวที่ `reports/page.tsx:85-88`
   **โชว์ Overall Accuracy 99.8% ทั้งที่ค่าจริงที่วัดได้คือ 72.3%**
   → บั๊กชนิดเดียวกับ `benchmark_similarity.py` ที่เคยลบไป แต่รอบนี้อยู่บนหน้าจอที่ผู้ใช้เห็นทุกวัน
2. **🔴 `/products` เปิดมาว่างเสมอ** (ราก: status mismatch)
3. **🟠 หน้าแรกนับงานค้าง 0** ทั้งที่มี 368 รายการรอจริง (ราก: เดียวกัน)
4. **🟠 `/import` → แท็บ "รอการอนุมัติ" ตายสนิท** — `api/import/pending/route.ts:27` กรอง
   `suggestion_method='hybrid_ai_preview'` ซึ่ง**คนเขียนค่านี้มีที่เดียวคือ `/api/import/process`**
   ที่ไม่มี UI เรียกแล้ว (เป็นของ `ProcessingStep` ที่ลบไปแล้ว) เช็ค DB แล้วมี **0 แถว**

### ของตายที่ยังค้าง
| อย่าง | ขนาด |
|---|---|
| `PendingTab` + `ApprovalStep` + `CategorySelector` | 757 บรรทัด |
| API routes `process` / `process-local` / `process-storage` / `approve` | 4 เส้น ไม่มี UI เรียก |
| `/reports` ทั้งหน้า | 229 บรรทัด |

> เหลือ import route ที่ใช้จริงเส้นเดียวคือ `commit` — ตรงกับที่เคยบันทึกไว้ว่า "import route ซ้ำ 4 เส้น"

### ✅ ทิศทางที่เจ้าของงานตัดสินใจแล้ว (29 ส.ค.)
- **`/reports`** → **ลบหน้าทิ้ง ยุบเข้าหน้าแรก** ใช้ตัวเลขจริงชุดเดียว
- **`/products`** → เปลี่ยนเป็น **หน้าค้น/ดูสตอกอย่างเดียว ไม่ทำ review** (งานตรวจอยู่ที่ `/data-quality` ที่เดียว)
- ขอบเขตตอนนั้น: **ขอรายงานก่อน ยังไม่ลงมือแก้**

### โครงที่เสนอ (9 → 7 หน้า + `/unlock` ที่ไม่อยู่ในเมนู)
```
/              แดชบอร์ด (ยุบ /reports เข้ามา ตัวเลขจริงชุดเดียว)
/import        Wizard อย่างเดียว (ตัดแท็บที่ตายออก)
/data-quality  ศูนย์รวมงานตรวจ 368 รายการ ← "ที่ทำงาน" จริงของระบบ
/products      ค้น/ดูสตอก 3,103 ตัว อ่านอย่างเดียว
/taxonomy      หมวดหมู่ + synonyms
/ai-brain      ML
/settings      กฎ keyword/regex
```
ที่สำคัญกว่าจำนวนหน้าคือ **งานตรวจมีบ้านเดียว ไม่กระจาย 3 ที่**

### ลำดับที่แนะนำให้ทำต่อ
1. **แก้ status mismatch** — จุดเดียวแก้ได้ 3 หน้า คุ้มที่สุด งานน้อยสุด
2. **จัดการ `/reports`** — ตัวเลขปลอมอยู่บนจอทุกวัน อันตรายเชิงตัดสินใจ
3. **ลบของตาย** — PendingTab + 4 routes เก็บกวาดตามหลัง

### ❓ 2 ข้อที่ยังไม่มีคำตอบ (ต้องถามเจ้าของงาน ตอบจากโค้ดอย่างเดียวไม่ได้)
- `/ai-brain` ควรยุบเป็นแท็บที่ 5 ของ `/data-quality` ไหม — ขึ้นกับว่าใช้เป็นหน้าคุมโมเดลแยกจริงหรือเปล่า
- `/import` เหลือ wizard อย่างเดียวแล้ว ยังต้องมี `ImportHistory` ไหม

---

## ⚠️ สถานะ git ตอนส่งต่อ (29 ส.ค. 2569 รอบสอง)

งานค้าง 24 ไฟล์ของรอบก่อน **commit แล้ว** (5 commit) และงานรอบสองอีก 5 commit
ทั้งหมดอยู่บนสาขา **`fix/status-mismatch-and-page-cleanup`** ยังไม่ merge เข้า `main` และยังไม่ push

migration ที่ apply ลง local DB แล้วและ commit แล้ว: `20260828000004`, `20260829000000`, `20260829000001`

`.mcp.json` มีการแก้ path ของ socraticode จาก `D:\SocratiCode-main\` ไปเป็น path ใน nvm
**ไม่ใช่งานของเซสชันนี้** ไม่ได้แตะ ปล่อยไว้ตามเดิม

---

## ✅ Recently Completed (Session 25-28 ส.ค. 2569 — Classification Accuracy & Working Import Pipeline)

### 📏 สร้างเครื่องวัดที่เชื่อถือได้ (เดิมไม่มีเลย)
- [x] **`src/utils/legacy_dataset.py`** — โหลดข้อมูลเก่า (ไฟล์เป็น UTF-16 หุ้ม cp874 อ่านตรงๆ จะได้ตัวขยะเงียบๆ) + แบ่ง train/test แบบ stratified seed คงที่
- [x] **`tests/integration/test_classification_accuracy.py`** — วัด accuracy จริง และ **skip อัตโนมัติ** ถ้ากฎถูกสกัดจากข้อมูลทั้งหมด (เห็น test set แล้ว) เพื่อไม่รายงานตัวเลขลวง
- [x] **ลบ `tests/benchmark_similarity.py`** — ต้นตอตัวเลข 72% ปลอม

### 🇹🇭 แก้การจัดหมวดหมู่ (25.5% → 72.3%)
- [x] **`src/core/fresh_implementations.py`** — เพิ่ม `tokenize_thai` / `tokens_contain_phrase` / `merge_short_token_runs` (pythainlp) แก้ปัญหา keyword สั้น match กลางคำ เช่น "สี" ใน "ยาสีฟัน" ทำให้ยาสีฟันถูกจัดเป็นสีทาบ้าน
- [x] **`scripts/mine_keywords_from_legacy.py`** — สกัดคีย์เวิร์ดจากข้อมูลที่คนจัดหมวดไว้ เข้า `keyword_rules` **ตัวนี้ให้ผลมากที่สุด (+43 จุด)** เพราะ 58% ของสินค้าอยู่ในหมวดที่ชื่อหมวดไม่ปรากฏในชื่อสินค้าเลย (เช่นหมวดแชมพูมีแต่ชื่อแบรนด์)
- [x] **`src/services/taxonomy_service.py`** — `extract_auto_keywords` เดิมใช้ `.split()` ได้ token ก้อนเดียวติดขนาดมาด้วย ระบบเรียนจาก UI แล้วไม่ได้อะไรเลย
- [x] **migration `20260825000000`** — เพิ่ม taxonomy 4 หมวดหลัก + 63 หมวดย่อยที่ขาด สินค้าเก่า map ได้ครบ 3,103/3,103 (เดิม 45% map ไม่ได้)

### 🔁 ระบบตรวจซ้ำหมวดหมู่ของเก่า
- [x] **`scripts/import_legacy_products.py`**, **`scripts/recheck_legacy_categories.py`**
- [x] **`app/api/recheck/route.ts`** + **`components/data-quality/RecheckTab.tsx`** — แสดงหมวดที่คนจัดคู่กับหมวดที่ AI เสนอ ยืนยันแล้วอัปเดต `products.category_id` ของแถวเดิม เขียน `review_history` (ตารางนี้ไม่เคยมี UI ไหนเขียนเลย) และเรียก `/learn/verify` ให้เรียนต่อ
- [x] **`e2e/recheck-legacy.spec.ts`** — ทดสอบผ่าน Playwright จริง

### 🤖 ML ตรวจซ้ำ — แก้ 4 บั๊กที่ทำให้ใช้งานไม่ได้จริง
- [x] `similarity_matches` **ไม่เคยมีข้อมูลเลย** — `internal_match.py` มีแต่ read ผลสแกนอยู่ใน dict หน่วยความจำ หายทุกครั้งที่รีสตาร์ท → เพิ่ม `scripts/build_similarity_training_data.py`
- [x] `word_overlap` แยกคำด้วยช่องว่าง (ไทยไม่เว้นวรรค) → ใช้ตัดคำจริง
- [x] `brand_similarity` ใช้ลิสต์แบรนด์อิเล็กทรอนิกส์อังกฤษ (iphone, samsung) คืน 0.5 คงที่ทุกคู่ → เทียบ token นำหน้า
- [x] `_fetch_training_data` โดนลิมิต 1000 แถวของ Supabase เสียตัวอย่างไป 344 จาก 1,344 แบบเงียบๆ → ทำ pagination
- [x] model path สัมพัทธ์ ขึ้นกับ CWD → ย้ายไป `model_cache/feedback_model.joblib`

### 📦 Import Wizard — แก้บั๊กที่ทำให้ไม่บันทึกอะไรเลย
- [x] **`app/api/import/commit/route.ts`** — บันทึกจริง แยกสองจังหวะ (`dedup` / `categorize`) **บันทึกตั้งแต่จบขั้นตรวจของซ้ำ** ไม่รอขั้นสุดท้าย เพื่อให้ปิดเบราว์เซอร์กลางคันแล้วไปทำต่อที่หน้า Verify ได้
- [x] `ColumnMappingStep` parse ด้วย `maxRows: 10` เพื่อพรีวิว แต่ส่ง object เดิมไปใช้เป็นข้อมูลจริง → **ไฟล์ 405 รายการถูกประมวลผลแค่ 10** (หน้าจอยังโชว์ 405 เพราะอ่านคนละฟิลด์)
- [x] `DeduplicationStep.onComplete` ส่งข้อมูลดิบกลับ ทิ้งผลแบ่งกลุ่มทั้งหมด → ขั้นถัดไปจึงจัดหมวดสินค้าที่มีในสตอกอยู่แล้วด้วย
- [x] `ProductMatchResult` ไม่เคยคืน id ของสินค้าในคลัง (`id` เป็นเลขลำดับรีวิว) → เพิ่ม `oldProductId` ไม่งั้นเขียน FK ของ `similarity_matches` ไม่ได้
- [x] ทั้งสอง route เดิม**ไม่ใส่ embedding** สินค้าที่เพิ่มผ่าน UI จะมองไม่เห็นในการสแกนครั้งหน้าและถูกนำเข้าซ้ำได้เรื่อยๆ → commit route คำนวณ embedding เป็นชุดก่อนบันทึก
- [x] `CompleteStep` ขึ้นว่า "บันทึกเรียบร้อยแล้ว" ทุกครั้งทั้งที่ไม่เคยเขียน DB เลย → แสดงตัวเลขจริงจากผลตอบกลับ และเตือนถ้าล้มเหลว
- [x] fallback ของขั้นตรวจของซ้ำใช้ `Math.random()` → ใส่การ์ดกันข้อมูลจำลองไม่ให้ลง DB

### 🐛 บั๊กร้ายแรงที่สุดที่เจอ
`internal_match.py` เทียบผลทำนายกับ `FeedbackType.SIMILAR.value` (`'similar'`) แต่โมเดลเทรนจาก `similarity_matches` ซึ่งมีแค่ `'duplicate'`/`'different'` → **เงื่อนไขเป็นเท็จเสมอ ทุกคู่ถูกรายงานว่า "different"** รวมถึงคู่ที่ต่างกันแค่ช่องว่าง (`แขวนเสื้อลวด+หนีบ 99 SM` vs `แขวนเสื้อลวด + หนีบ 99 SM` = 0.96) ถ้าไม่แก้ ผู้ใช้จะนำเข้าสินค้าซ้ำเข้าสตอก มีบั๊กนี้อยู่ 5 จุดใน 3 endpoint

### 🔧 Follow-up หลัง push (28 ส.ค. ต่อเนื่อง)
- [x] **`components/Import/WizardTab.tsx`** — ขั้นจัดหมวด (step 4) เดิมส่งข้อมูลทั้ง 405 แถวเข้า `CategorizationStep` ทั้งที่ 37 แถวถูก reject เป็นของซ้ำไปแล้วและ 146 แถวยังไม่รู้ชะตากรรม (รอคนตัดสินที่ Verify ด่าน 1) แก้ให้กรองเฉพาะ `_bucket === 'new'` ก่อนส่ง — ยืนยันด้วยการอัปโหลดไฟล์ 405 แถวเดิมผ่าน browser จริงคุยกับ backend จริง ได้ผล 37/146/222 ตรงกับที่คาด (ตัวเลขต่างจากรอบก่อน 37/147/221 อยู่ 1 แถวเพราะ threshold model ไม่ deterministic 100% ไม่ใช่บั๊ก)
- [x] **`jest.config.js`** — คีย์ `moduleNameMapping` (พิมพ์ผิด) นอนอยู่ข้าง `moduleNameMapper` (ตัวจริง) โดยไม่มีผลอะไรเลย เพราะ Jest เมิน key ที่ไม่รู้จักเงียบๆ และ `next/jest` ก็ derive alias `@/` จาก `tsconfig.json` ให้อยู่แล้ว รวมเป็นก้อนเดียว — รันเทียบก่อน/หลัง (8 failed / 11 passed ทั้งคู่) ยืนยันว่าไม่กระทบพฤติกรรม เป็นแค่ความสะอาดของ config
- [x] **`docs/architecture/DATABASE_SCHEMA.md`, `REAL_DATABASE_RELATIONSHIPS.md`** — เขียนใหม่ให้ตรงกับ `schema_export.sql` จริง (เดิมค้างที่ 2025-10-04 นับตาราง 14 ทั้งที่จริงมี 15)
- Push แล้วที่ `8e8ab5c0` (4 คอมมิตแยกตามหัวข้อ: wizard fix, jest cleanup, docs rewrite, tsbuildinfo)

### 💰 ราคาขาย: เก็บถึง DB จริง + ใช้ช่วยตรวจของซ้ำ (28 ส.ค. ต่อเนื่อง)
ก่อนหน้านี้ผู้ใช้เลือกคอลัมน์ "ราคาขาย" ตอน map ได้ตั้งแต่ขั้นที่ 1 ของ wizard แต่ค่าหายไปกลางทาง — ยืนยันด้วย query ว่าราคาสินค้าทั้ง 3,508 แถวเป็น NULL 100%
- [x] **`components/Import/DataCleaningStep.tsx`** — เดิมสร้างแค่ `_cleaned_name` ไม่เคยดึงราคาจาก raw row เลย เพิ่ม `price: parsePrice(row[columnMapping.price])` เข้าไปในทุกแถวที่ merge (ทั้ง flow ปกติและ fallback)
- [x] **`components/Import/WizardTab.tsx`** — `commitDedup()` เดิมไม่ใส่ `price` ในสิ่งที่ส่งไป `/api/import/commit` ทั้งที่ backend (`app/api/import/commit/route.ts`) รองรับ field นี้สมบูรณ์อยู่แล้วตั้งแต่แรก (schema + insert มีให้ครบ ไม่เคยต้องแก้ backend เลย)
- [x] **`utils/price.ts`** (ใหม่) — `parsePrice()` (แปลงสตริงราคาที่มีคอมม่า/สัญลักษณ์บาทให้เป็นตัวเลข), `isPriceMismatch()` (ราคาต่างกันเกิน 2 เท่า = mismatch, ข้ามการเช็คถ้าราคาใดราคาหนึ่งไม่มี), `classifyDedupBucket()` (ตัดสิน bucket duplicate/review/new)
- [x] **`components/Import/DeduplicationStep.tsx`** — เพิ่ม `oldPrice`/`_new_price`/`_old_price`/`_price_mismatch` และเปลี่ยนมาเรียก `classifyDedupBucket()` แทน if/else เดิม นโยบาย: ราคาไม่เคยบล็อกการจับคู่แบบเด็ดขาด ใช้แค่ลดชั้นความมั่นใจเฉพาะโซน auto-merge (≥95%) — ถ้าราคาต่างกันเกิน 2 เท่า ดึงลงมาเป็น "ต้องตรวจสอบ" แทน ส่วนโซน 80-94% โชว์ราคาคู่กันเป็นข้อมูลช่วยตรวจ (ไม่เปลี่ยน bucket เพราะคนตรวจอยู่แล้ว) โซน <80% ไม่แตะเลย
- [x] **`src/api/models.py`, `src/api/routers/internal_match.py`** — เพิ่ม `oldPrice` ใน `ProductMatchResult` และ select ราคาสินค้าฝั่งสต๊อกมาด้วย พบและแก้เพิ่ม: จุด fallback ตอน ML ยังไม่ trained ขาด `oldProductId` (บั๊กเดิมที่ไม่มีใครเจอเพราะปกติ ML trained อยู่แล้ว)
- ตั้งใจไม่ทำ: backfill ราคา 3,103 รายการเก่า, แก้ให้ `sku` เหมือนกัน, ใช้ราคาเป็น ML feature จริงจัง, ค้นหาคู่ซ้ำในโซน <80% ด้วยราคา — ทั้งหมดเก็บไว้เป็นงานแยกอนาคตตามที่ตกลงกับผู้ใช้

**การตรวจสอบ:** ยืนยัน Part A (เก็บราคา) จริงผ่าน browser+backend จริง (อัปโหลด CSV ราคา 300/55 → เห็น `products.price = 300.00/55.00` ใน DB ก่อนลบทิ้ง) เนื่องจากบั๊ก 1,000-row (ดูหัวข้อถัดไป) ทำให้ทดสอบ Part B (bucket demotion) ผ่าน live endpoint ไม่เสถียรตอนแรก จึงพิสูจน์ policy ด้วย unit test 16 เคสใน `__tests__/utils/price.test.ts` แทน (ครอบคลุมทุก edge case ของนโยบายที่ตกลงกัน) — `pytest` 160 passed, `jest` ไม่มี suite ใหม่พัง (8 failed เดิมที่รู้อยู่แล้ว)

### 🐛 แก้บั๊ก: ตรวจของซ้ำเทียบกับสต๊อกแค่ ~32% แบบสุ่ม (28 ส.ค. ต่อเนื่อง)
เจอระหว่างทดสอบ Part B ข้างบน — endpoint `/api/v1/match/import-dedup` ดึงสินค้าฝั่งสต๊อกด้วย `.select(...).eq("status","approved")` **ไม่มี `ORDER BY`** และ Supabase/PostgREST จำกัดไว้ที่ 1,000 แถวต่อ request เสมอ ทั้งที่สต๊อกมี 3,103 แถว — พิสูจน์แล้วว่าสินค้าที่ควรแมตช์ตัวเองได้ 100% (`cosine similarity` 0.985) กลับไม่เจอเลยเพราะไม่ได้อยู่ใน 1,000 แถวที่สุ่มมาในการเรียกครั้งนั้น
- [x] เพิ่ม `fetch_all_approved_products()` ใน `internal_match.py` — แบ่งหน้าด้วย `.range()` พร้อม `.order("id")` (จำเป็น! ไม่งั้น Postgres ไม่การันตีลำดับแถวเดิมข้าม request ทำให้แบ่งหน้าแล้วข้ามหรือได้แถวซ้ำได้)
- [x] ใช้ helper นี้แทนของเดิมทั้ง 2 จุด: `deduplicate_imports` (endpoint หลักที่มีบั๊ก ไม่เคยแบ่งหน้าเลย) และ `scan_internal_duplicates` (เดิมแบ่งหน้าอยู่แล้วแต่ไม่มี `.order()` เหมือนกัน เลยรวมให้ใช้โค้ดเดียวกัน)
- **ยืนยันจริง:** เรียก `fetch_all_approved_products()` ตรงๆ ได้ 3,103 แถว unique ครบ (เดิม endpoint ดึงได้แค่ 1,000) และยิง `/api/v1/match/import-dedup` ซ้ำด้วยชื่อเดิมที่เคยหาไม่เจอ → เจอแมตช์ถูกต้องทันที (0.985, 0.901) `pytest` 160 passed ไม่มี regression
- ยังไม่แก้ (พบแต่ไม่ใช่บั๊กแบบเดียวกัน เสี่ยงต่ำกว่า เก็บไว้ก่อน): `ml_feedback_learning._fetch_training_data()` และ `scripts/build_similarity_training_data.py` มีการแบ่งหน้าอยู่แล้วแต่ก็ไม่มี `.order()` เหมือนกัน — ควรเพิ่มให้ครบเป็นงานเล็กๆ ถัดไป

### 🔐 ER/Schema Audit → 4 Migration แก้ RLS/FK/Index (28 ส.ค. ต่อเนื่อง — เซสชันคู่ขนาน check-products-db)
สำรวจ schema ทั้ง 15 ตารางแบบ read-only (2 Explore agent ไล่ migration ทั้งหมด + ไล่โค้ดจริงว่าตารางไหนถูกใช้งานจริง/ตายแล้ว) สรุปว่าตารางหลัก (`products`/`taxonomy_nodes`/`similarity_matches`) ออกแบบดีตอบโจทย์ธุรกิจจริง แต่เจอช่องโหว่ RLS ที่ยังไม่มีใครรู้:
- [x] **พบร้ายแรงสุด:** `system_settings` **ไม่เปิด RLS เลย** + ให้สิทธิ์ `anon` (ไม่ต้อง login) `INSERT/SELECT/UPDATE/DELETE/TRUNCATE` เต็ม — ยืนยันกับ DB จริง ปลอดภัยอยู่ตอนนี้เพราะแอปใช้ service-role key เท่านั้น แต่เป็นระเบิดเวลา
- [x] พบ 8/15 ตาราง เปิด RLS แต่ **0 policy** (`keyword_rules`, `regex_rules`, `product_category_suggestions`, `product_attributes`, `similarity_matches`, `review_history`, `imports` + `system_settings` ข้างบน) — วันที่มีอะไรเรียก Supabase ด้วย key ที่ไม่ใช่ service-role ตารางพวกนี้จะว่างเปล่าเงียบๆ หรือใช้งานไม่ได้เลย
- [x] พบ 16 คอลัมน์ attribution (`created_by`/`updated_by`/`reviewed_by`) ไม่มี FK ไป `auth.users`, และ `products.import_batch_id` ไม่มี FK ไป `imports.id` (ตรวจกับ DB จริง: ไม่มีข้อมูลขาดหาย เพิ่มได้ทันทีไม่ต้อง backfill)
- [x] พบ index ซ้ำซ้อน 2 คู่บน `taxonomy_nodes` (ivfflat embedding ซ้อนกัน 2 ตัว lists=100/20, GIN keywords ซ้ำ) + code index ซ้ำกับ unique constraint
- [x] พบบั๊ก `ml_training_history` policy insert/delete comment เขียนว่า "service_role only" แต่เงื่อนไขจริงเป็น `USING(true)` — เปิดกว้างจริง ขัดกับ comment
- [x] สร้าง 4 migration แก้ครบ: `20260828000000_fix_rls_gaps.sql`, `20260828000001_add_missing_foreign_keys.sql`, `20260828000002_cleanup_redundant_indexes.sql`, `20260828000003_fix_ml_training_history_policy_bug.sql`
- **ยืนยันแล้ว:** apply เข้า local DB สำเร็จ (`npx supabase migration up --local --include-all`), query `pg_policies`/`pg_constraint`/`pg_indexes` ตรงตามที่ตั้งใจทุกจุด (ไม่มีตารางไหนเปิด RLS แล้วไม่มี policy อีกเลย, FK รวม 36 ตัวถูกต้องครบ, index เหลือตัวเดียวต่อชนิดตามแผน), `EXPLAIN` ยืนยัน query vector ยังใช้ index ที่เหลือถูกต้อง, `pytest tests/integration/` ผ่าน 20 passed/6 skipped/1 failed — ตัวที่ fail (`test_internal_scan`) เป็น test timeout 60s ของ scan ที่ใช้เวลาจริง 74s ไม่เกี่ยวกับ schema ที่แก้ (เช็ค task status ตรงๆ พบว่า scan เสร็จสมบูรณ์ถูกต้อง 114 คู่)
- ไม่ทำ (นอกขอบเขต แจ้งผู้ใช้ไว้แล้ว): จัดระเบียบ `system_settings` ที่ปนสองแบบ (JSONB blob + key/value) รอ audit โค้ดแอปที่อ่านคอลัมน์ blob ก่อน; cleanup 4 route insert สินค้าเก่าที่ตายแล้ว (`process`, `process-local`, `process-storage`, `approve`) เป็นหนี้เทคนิคแยกจาก schema
- ก่อนลงมือ เช็คกับเซสชันคู่ขนาน `check-products-e3` แล้วว่าไม่ชนไฟล์กัน (เขาไม่แตะ migrations/RLS) — ยังไม่ commit ไฟล์ migration ทั้ง 4 นี้

---

## ✅ Recently Completed (Session 31 พ.ค. 2569 — Deduplication Clean Refactor & Bug Fixes)

### 🧹 Clean Refactor & Separation of Concerns (Deduplication)
- [x] **`DeduplicationTab.tsx` (Frontend)** — ลบโหมดอัปโหลดไฟล์ (โหมด `'file'`) และความปลอดภัยในการเรียก Supabase Edge function ออกทั้งหมด ปรับปรุงหน้านี้ให้โฟกัสที่การทำความสะอาดคลังสินค้าเดิมภายในระบบ (Internal Catalog Audit) แบบ 100% ป้องกัน UX ทับซ้อนกับหน้า Import Wizard
- [x] **`scripts/complete_deduplication_pipeline.py`** — ลบสคริปต์ CLI ตัวนี้ออกอย่างถาวรตามความประสงค์ของคุณกานเพื่อความสะอาดเรียบร้อย
- [x] **`src/services/human_feedback_system.py`** — ลบคลาส SQLite (`HumanFeedbackDatabase`), คลาส Batch CLI (`ProductDeduplicationSystem`, `HumanReviewInterface`) และลบฐานข้อมูล SQLite ท้องถิ่น `human_feedback.db` ทิ้งทั้งหมด เพื่อหันมาพึ่งพา Supabase standard database เต็มตัวเป็น Single Source of Truth
- [x] **คงรักษา Enums/Dataclasses** — คงเหลือ `FeedbackType`, `ProductComparison` และ `UniqueProduct` ใน `human_feedback_system.py` เพื่อให้ระบบ ML และ API ในโปรเจกต์เรียกใช้งานร่วมกันได้อย่างสมบูรณ์แบบ

### 🐛 Bug Fixes & Stable Testing Matrix
- [x] **แก้ Bug** `ImportError` ใน `src/services/ml_feedback_learning.py` โดยลบ import ตกค้างของคลาสที่ถูกลบออกไปเรียบร้อยแล้ว 100%
- [x] **แก้ Bug** `ReferenceError: ZapIcon is not defined` ใน Next.js `DeduplicationTab.tsx` ด้วยการนำเข้า `Zap` จาก `lucide-react` และปรับการใช้งานปุ่มสแกนให้แสดงผลถูกต้อง
- [x] **ปรับปรุง Unit/Integration Tests** — แก้ไข `test_cleaned_text_system.py` และ `test_deduplication_system.py` เพื่อให้ทดสอบเฉพาะ Component ที่มีอยู่จริง (NLP clean และ Enums) และลบ High-Unicode Emojis ออกทั้งหมดเพื่อป้องกัน Error การเข้ารหัส CP874 บน terminal Windows PowerShell รันผ่านได้สำเร็จ 100%

---

## ✅ Recently Completed (Session 30 พ.ค. 2569 — ML Integration)

### 🧠 ML Continuous Learning Pipeline
- [x] **`src/services/ml_feedback_learning.py`** — Refactor `ContinuousLearningSystem` ให้ดึงข้อมูลจาก Supabase `similarity_matches` (reviewed=true) แทน SQLite เดิม
- [x] **แก้ Bug** `ThaiTextProcessor.preprocess()` → `process()` ใน ML service และ internal_match router
- [x] **`src/api/dependencies.py`** — เพิ่ม `get_ml_learning_system()` เป็น Global Singleton ใน `app_state`
- [x] **`src/api/routers/learn.py`** — เพิ่ม `GET /api/v1/learn/status` (ดึงสถิติโมเดล + Feature Importance) และแก้ `POST /api/v1/learn/retrain` ให้ใช้ Singleton
- [x] **`src/api/routers/internal_match.py`** — เพิ่ม Stage-2 ML Inference: กรองคู่ที่ ML มั่นใจว่าเป็นคนละชิ้น (confidence > 0.6) ออกจากผลการสแกน
- [x] **รัน Migration SQL:** `20260530100000_add_ml_training_history.sql` สร้างตารางประวัติการเทรน

### 🖥️ AI Brain Dashboard (Frontend)
- [x] **`taxonomy-app/components/Layout/Sidebar.tsx`** — เพิ่มเมนู "สมองกล AI" (BrainIcon) ลิงก์ `/ai-brain`
- [x] **`taxonomy-app/app/ai-brain/page.tsx`** — **[NEW]** หน้า Dashboard พรีเมียม 3 ส่วน: Top Stats, Feature Importance (Animated Bar), Control Center
- [x] **`taxonomy-app/components/data-quality/DeduplicationTab.tsx`** — เพิ่มปุ่ม "เริ่มสอน AI จากประวัติ (Retrain Model)" ใต้ปุ่มสแกน

---

## ✅ Previously Completed (Session 26 พ.ค. 2569)

### 🗂️ Route Consolidation & Tab Container Pattern
- [x] **Sidebar** ลดเมนูหลักให้เหลือ **7 เมนู** (Dashboard, Products, Taxonomy, Data Quality, Import, Reports, Settings)
- [x] **Data Quality Center (`/data-quality`):** รวม 3 หน้าย่อยเป็น Tab เดียว
  - `VerifyTab.tsx` (จากหน้า `/verify`)
  - `DeduplicationTab.tsx` (จากหน้า `/deduplication`)
  - `AutoLearnTab.tsx` (จากหน้า `/auto-learn`)
- [x] **Taxonomy Center (`/taxonomy`):** รวม `/synonyms` เข้าเป็น `SynonymsPanel.tsx` ภายใน Tab
- [x] **Import Pipeline (`/import`):** รวม `/import/wizard` → `WizardTab.tsx` และ `/import/pending` → `PendingTab.tsx`
- [x] **UX Auto-redirect:** เมื่อ Deduplication เสร็จ ระบบพาผู้ใช้ไปยัง Verify อัตโนมัติ
- [x] **Production Build ผ่าน 100%** (`npm run build` ไม่มี Error หลงเหลือ)

### 🐛 TypeScript Bug Fixes ที่แก้ไประหว่าง Build
- [x] `process-local/route.ts` → แก้ `never[]` type และ explicit type annotation ให้ `bestMatch`
- [x] `process-storage/route.ts` → แก้ `error.message` จาก `unknown` type + แก้ Duplicate property key `errors`
- [x] `synonyms/route.ts` → แก้ `name` → `name_th` ให้ตรงกับ `Synonym` Interface
- [x] `SynonymsPanel.tsx` → แก้ `null` → `undefined` สำหรับ `category_id`
- [x] `WizardTab.tsx` → แก้ `ColumnMapping` type ที่ไม่ได้ import
- [x] **Case Sensitivity Fixes:** แก้ `components/import` → `components/Import` และ `components/taxonomy` → `components/Taxonomy`

---

## ✅ Previously Completed (Session 24 พ.ค. 2569)

### 🚀 Magic Import Wizard 5-Step (Frontend Components)
- [x] `UploadAndMappingStep.tsx`, `DataCleaningStep.tsx`, `DeduplicationStep.tsx`, `CategorizationStep.tsx`, `CompleteStep.tsx`

### 🏗️ API Server Modular Refactoring (v4.0)
- [x] แตก `api_server.py` เป็น 10 ไฟล์ใน `src/api/`, แก้ Event Loop Blocking, WebSocket Bug, Memory Leak

---

## 🚧 In Progress / ยังไม่ได้ทำ
- [ ] **ทดสอบ E2E:** `pytest tests/integration/test_ml_e2e.py -v` (ต้องการ FastAPI server รันอยู่)

---

## 📋 Next Steps
1. ทดสอบ E2E: กด Retrain → รอ → เปิดหน้า AI Brain → ตรวจสอบสถิติโชว์ครบ
2. พิจารณาเพิ่มตาราง `ml_training_history` ใน Supabase เพื่อให้ประวัติการสอนไม่หายเมื่อ Restart Server
3. อัปเดต `API_ARCHITECTURE.md` ให้สะท้อน ML Layer ใหม่
4. 🚀 **ประสิทธิภาพและการสเกลระบบในอนาคต (Scalability Improvement):**
   - [ ] ย้ายการคำนวณและค้นหาคู่สินค้าซ้ำเชิงเวกเตอร์ (Vector Cosine Similarity) ไปประมวลผลที่ฐานข้อมูล Supabase (PostgreSQL) แทนการโหลดขึ้นมาทำที่ Python RAM
   - [ ] เปิดใช้งาน `pgvector` extension และสร้างดัชนี **HNSW (Hierarchical Navigable Small World) Index** เพื่อปรับปรุงประสิทธิภาพการค้นหาแบบ ANN ลดความซับซ้อนในการเปรียบเทียบจาก $O(N^2)$ เหลือ $O(\log N)$ รองรับสินค้าขนาด 100,000+ SKU

---

## 💡 System State Summary
- **Frontend (Next.js):** http://127.0.0.1:3000 — **8 เมนู** (เพิ่ม AI Brain), Build ✅
- **Backend (FastAPI):** http://127.0.0.1:8000
- **ML Model:** `RandomForestClassifier` (15 features) — เทรนจาก `similarity_matches` ใน Supabase
- **Hybrid Algorithm:** Keyword 60% + Embedding 40% → Target Accuracy ≥ 72%
- **Embedding Model:** `paraphrase-multilingual-MiniLM-L12-v2` (384-dim)
- **Model File:** `feedback_model.joblib` (root dir ของ Backend)
- **LAN Access:** http://192.168.1.80:3000
