# 📊 Current Project Status (Compact Context)
*Last Updated: 2026-06-07 (By Phayak)*

## 🎯 Current Focus
- ระบบ **Deduplication Refactor**: ปรับโครงสร้างระบบตรวจจับสินค้าซ้ำให้คลีน 100% จัดสรรขอบเขตงานของหน้าจอให้ชัดเจน และถอน dead code
- ระบบ **ML Continuous Learning Pipeline** ทำงานปกติ: AI เรียนรู้จากการรีวิวของผู้ใช้ใน Supabase
- User Workflow: **Import (Wizard) → Dedup (Catalog Audit) → Review → AI Learns → Better Dedup**

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
