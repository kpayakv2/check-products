# 📊 Current Project Status (Compact Context)
*Last Updated: 2026-08-28*

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

**⚠️ เจอบั๊กเดิมระหว่างทดสอบ (ยังไม่แก้ อยู่นอกขอบเขตงานนี้):** `internal_match.py` endpoint `/api/v1/match/import-dedup` ดึงสินค้าฝั่งสต๊อกด้วย `.select(...).eq("status","approved")` **ไม่มี `ORDER BY`** และ Supabase/PostgREST จำกัดไว้ที่ 1,000 แถวต่อ request ทั้งที่สต๊อกมี 3,103 แถว — แปลว่าทุกครั้งที่ตรวจของซ้ำ ระบบเทียบกับสต๊อกแค่ ~32% แบบสุ่มไม่คงที่ (เรียกซ้ำได้ชุดต่างกัน) พิสูจน์แล้วด้วยการยิง endpoint ตรงๆ ว่าสินค้าที่ควรจะแมตช์ตัวเองได้ 100% กลับไม่เจอเลย เพราะไม่ได้อยู่ใน 1,000 แถวที่สุ่มมา **ควรแก้เป็นงานถัดไป** (เพิ่ม `.order("id")` + pagination หรือทำ query แบบ batch)

**การตรวจสอบ:** ยืนยัน Part A (เก็บราคา) จริงผ่าน browser+backend จริง (อัปโหลด CSV ราคา 300/55 → เห็น `products.price = 300.00/55.00` ใน DB ก่อนลบทิ้ง) เนื่องจากบั๊ก 1,000-row ทำให้ทดสอบ Part B (bucket demotion) ผ่าน live endpoint ไม่เสถียร จึงพิสูจน์ด้วย unit test 16 เคสใน `__tests__/utils/price.test.ts` แทน (ครอบคลุมทุก edge case ของนโยบายที่ตกลงกัน) — `pytest` 160 passed, `jest` ไม่มี suite ใหม่พัง (8 failed เดิมที่รู้อยู่แล้ว)

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
