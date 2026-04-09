# 📁 โครงสร้างโปรเจกต์ Product Checker

> **เอกสารนี้อธิบายโครงสร้างโปรเจกต์ทั้งหมด เพื่อให้สามารถค้นหาไฟล์ได้ง่ายและเข้าใจการทำงานของระบบ**

---

## 🎯 ภาพรวมโปรเจกต์

โปรเจกต์นี้ประกอบด้วย **2 ระบบหลัก** ที่ทำงานเป็นลำดับขั้นตอน:

1. **Product Similarity Checker** (Python + Web Frontend) - หาสินค้าที่ไม่ซ้ำและอนุมัติสินค้า
2. **Thai Product Taxonomy Manager** (Next.js Frontend) - จัดการหมวดหมู่สินค้าที่ได้รับอนุมัติแล้ว

---

## 📂 โครงสร้างไดเรกทอรีหลัก

```
check-products/
├── 🐍 PYTHON BACKEND (Product Similarity Checker)
│   ├── main.py                              # Entry point หลัก
│   ├── complete_deduplication_pipeline.py   # ระบบหาสินค้าที่ไม่ซ้ำ
│   ├── api_server.py                        # FastAPI REST API
│   ├── fresh_implementations.py             # Core algorithms
│   ├── fresh_architecture.py                # Architecture components
│   └── ...
│
├── ⚛️ NEXT.JS FRONTEND (Taxonomy Manager)
│   └── taxonomy-app/
│       ├── app/                             # Next.js App Router
│       ├── components/                      # React Components
│       ├── utils/                           # Utility functions
│       └── ...
│
├── 📄 DOCUMENTATION
│   └── docs/
│       ├── INDEX.md                         # เอกสารหลัก
│       ├── guides/                          # คู่มือการใช้งาน
│       ├── api/                             # API Documentation
│       └── development/                     # Development guides
│
├── 🧪 TESTING
│   └── tests/
│       ├── integration/                     # Integration tests
│       ├── config/                          # Test configuration
│       └── ...
│
└── 📊 DATA & OUTPUT
    ├── input/                               # ไฟล์ input
    ├── output/                              # ผลลัพธ์
    └── uploads/                             # ไฟล์ที่อัปโหลด
```

---

## 🐍 PART 1: Python Backend (Product Similarity Checker)

### 🎯 **จุดประสงค์**
หาสินค้าใหม่ที่**ไม่ซ้ำกับสินค้าเก่า** เพื่อนำเข้าระบบ (Product Deduplication)

### 📁 **ไฟล์หลักและหน้าที่**

#### **1. Entry Points (จุดเริ่มต้นการใช้งาน)**

| ไฟล์ | หน้าที่ | การใช้งาน |
|------|---------|-----------|
| **`main.py`** | Entry point หลัก (Phase 4 Enhanced) | `python main.py old.csv new.csv` |
| **`complete_deduplication_pipeline.py`** | ระบบครบวงจร + Human-in-the-Loop | `python complete_deduplication_pipeline.py --input products.csv --mode analyze` |
| **`api_server.py`** | FastAPI REST API Server | `python api_server.py` → http://localhost:8000 |
| **`cli.py`** | Command Line Interface | `python cli.py --help` |
| **`quick_start.py`** | Quick start script | `python quick_start.py` |

#### **2. Core Components (ส่วนประกอบหลัก)**

| ไฟล์ | หน้าที่ | ใช้โดย |
|------|---------|--------|
| **`fresh_implementations.py`** | Core algorithms:<br>- TextProcessor (Thai text cleaning)<br>- EmbeddingModel (TF-IDF, Sentence Transformers)<br>- SimilarityCalculator | ทุกไฟล์หลัก |
| **`fresh_architecture.py`** | Architecture components:<br>- ProductMatcher<br>- ProductSimilarityPipeline<br>- Config | main.py, api_server.py |
| **`ensemble_models.py`** | Ensemble learning models | advanced_models.py |
| **`advanced_models.py`** | Advanced ML models | complete_deduplication_pipeline.py |

#### **3. Human Feedback System**

| ไฟล์ | หน้าที่ |
|------|---------|
| **`human_feedback_system.py`** | ระบบรับ feedback จากมนุษย์:<br>- ProductDeduplicationSystem<br>- HumanReviewInterface<br>- HumanFeedbackDatabase |
| **`ml_feedback_learning.py`** | ML learning จาก human feedback:<br>- ContinuousLearningSystem<br>- FeedbackLearningModel |
| **`human_feedback.db`** | SQLite database สำหรับเก็บ feedback |

#### **4. Utility Scripts**

| ไฟล์ | หน้าที่ | การใช้งาน |
|------|---------|-----------|
| **`clean_csv_products.py`** | ทำความสะอาดไฟล์ CSV | `python clean_csv_products.py input.csv` |
| **`filter_matched_products.py`** | กรองผลลัพธ์ตาม threshold | `python filter_matched_products.py --threshold 0.8` |
| **`run_analysis.py`** | วิเคราะห์ผลลัพธ์ | `python run_analysis.py` |
| **`show_uniqueness_criteria.py`** | แสดงเกณฑ์การหาสินค้าที่ไม่ซ้ำ | `python show_uniqueness_criteria.py` |
| **`download_models.py`** | ดาวน์โหลด ML models | `python download_models.py` |
| **`model_cache_manager.py`** | จัดการ model cache | Auto-used |

#### **5. Web Interface**

| ไฟล์ | หน้าที่ | URL |
|------|---------|-----|
| **`web/human_review.html`** | Web UI สำหรับ human review | http://localhost:8000/web |
| **`web/product_deduplication.html`** | Web UI สำหรับ deduplication | http://localhost:8000/dedup |
| **`web_server.py`** | Web server (legacy) | - |

#### **6. Configuration & Dependencies**

| ไฟล์ | หน้าที่ |
|------|---------|
| **`requirements.txt`** | Python dependencies สำหรับ production |
| **`requirements-dev.txt`** | Python dependencies สำหรับ development |
| **`pyproject.toml`** | Project configuration (Black, isort, pytest) |
| **`pytest.ini`** | Pytest configuration |
| **`conftest.py`** | Pytest fixtures |
| **`Makefile`** | Build automation commands |

---

## ⚛️ PART 2: Next.js Frontend (Taxonomy Manager)

### 🎯 **จุดประสงค์**
จัดการ **Taxonomy, Synonym และ Product Review** แบบ Premium SaaS

### 📁 **โครงสร้างไดเรกทอรี**

```
taxonomy-app/
├── app/                          # Next.js App Router
│   ├── page.tsx                  # 🏠 Dashboard หลัก
│   ├── layout.tsx                # Root layout
│   │
│   ├── taxonomy/
│   │   └── page.tsx              # 🌳 Taxonomy Management
│   │
│   ├── synonyms/
│   │   └── page.tsx              # 📚 Synonym Manager
│   │
│   ├── products/
│   │   └── page.tsx              # 🛍️ Product Review
│   │
│   ├── import/
│   │   └── page.tsx              # 📤 Import System
│   │
│   └── api/                      # API Routes
│       ├── taxonomy/
│       │   └── route.ts          # Taxonomy CRUD API
│       ├── synonyms/
│       │   └── route.ts          # Synonym CRUD API
│       ├── products/
│       │   └── route.ts          # Product CRUD API
│       └── similarity/
│           └── route.ts          # Similarity API
│
├── components/                   # React Components
│   ├── Layout/
│   │   ├── Navbar.tsx            # Navigation bar
│   │   ├── Sidebar.tsx           # Sidebar menu
│   │   └── Footer.tsx            # Footer
│   │
│   ├── Taxonomy/
│   │   ├── TaxonomyTree.tsx      # Tree view component
│   │   └── EnhancedTaxonomyTree.tsx  # Enhanced version
│   │
│   ├── Synonym/
│   │   ├── SynonymManager.tsx    # Basic manager
│   │   └── EnhancedSynonymManager.tsx  # Enhanced version
│   │
│   ├── Product/
│   │   ├── ProductReview.tsx     # Basic review
│   │   └── EnhancedProductReview.tsx  # Enhanced version
│   │
│   └── Search/
│       └── HybridSearch.tsx      # Hybrid search component
│
├── utils/                        # Utility Functions
│   ├── supabase.ts               # 🗄️ Supabase client & DB operations
│   ├── rate-limit.ts             # Rate limiting
│   ├── error-handler.ts          # Error handling
│   ├── validation.ts             # Input validation (Zod)
│   └── logger.ts                 # Logging system
│
├── supabase/
│   └── schema.sql                # 🗄️ Database schema
│
├── __tests__/                    # Test files
│   ├── components/               # Component tests
│   ├── api/                      # API tests
│   └── utils/                    # Utility tests
│
├── public/                       # Static files
├── styles/                       # CSS files
│
├── .env.local                    # 🔒 Environment variables (gitignored)
├── .env.local.example            # Environment template
├── next.config.js                # Next.js configuration
├── tailwind.config.js            # Tailwind CSS config
├── tsconfig.json                 # TypeScript config
├── package.json                  # npm dependencies
└── README.md                     # 📖 Documentation
```

### 🗄️ **Database Schema (Supabase)**

| ตาราง | หน้าที่ |
|-------|---------|
| **`taxonomy_nodes`** | โครงสร้างหมวดหมู่แบบ Tree |
| **`synonyms`** | คำพ้องความหมาย (Lemma) |
| **`synonym_terms`** | คำที่เป็น synonym ของแต่ละ lemma |
| **`products`** | ข้อมูลสินค้า + Vector embeddings |
| **`similarity_matches`** | ผลการเปรียบเทียบความคล้าย |
| **`review_history`** | ประวัติการตรวจสอบ |

---

## 📊 DATA & OUTPUT Directories

### **input/** - ไฟล์ข้อมูลเข้า

```
input/
├── new_product/
│   └── products.xlsx             # สินค้าใหม่
├── old_product/
│   └── products.xlsx             # สินค้าเก่า (reference)
└── รายการสินค้าพร้อมหมวดหมู่_AI.txt
```

### **output/** - ผลลัพธ์

```
output/
├── matched_products.csv          # ผลการจับคู่สินค้า
├── matched_products_phase4.csv   # ผลลัพธ์ Phase 4
├── performance_report.json       # Performance metrics
└── unique_products.csv           # สินค้าที่ไม่ซ้ำ
```

### **uploads/** - ไฟล์ที่อัปโหลดผ่าน API

```
uploads/
└── new_products.xlsx
```

---

## 🧪 TESTING Structure

```
tests/
├── config/
│   ├── __init__.py
│   └── test_config.py            # Configuration tests
│
├── integration/
│   ├── test_api_client.py        # API client tests
│   ├── test_api_endpoints.py     # API endpoint tests
│   ├── test_complete_pipeline.py # Pipeline integration tests
│   └── ...
│
├── examples/
│   └── test_refactored_example.py
│
└── fixtures/                     # Test fixtures
```

---

## 📚 DOCUMENTATION Structure

```
docs/
├── INDEX.md                      # 📖 เอกสารหลัก (เริ่มที่นี่)
│
├── guides/                       # 🚀 คู่มือการใช้งาน
│   ├── quick-start.md            # เริ่มต้นใช้งาน
│   ├── human-feedback.md         # Human feedback system
│   ├── embedding-models.md       # Embedding models guide
│   └── model-download.md         # ดาวน์โหลด models
│
├── api/                          # 🔌 API Documentation
│   ├── api-reference.md          # API reference ฉบับเต็ม
│   ├── quick-reference.md        # API quick reference
│   └── analyze-capabilities.md   # API capabilities
│
├── development/                  # 👩‍💻 Development Guides
│   ├── architecture.md           # System architecture
│   ├── text-preprocessing.md     # Thai text processing
│   ├── contributing.md           # How to contribute
│   ├── testing.md                # Testing guide
│   └── changelog.md              # Version history
│
└── archive/                      # 📦 Archived documents
    ├── file-deletion-report.md
    ├── irrelevant-code-analysis.md
    └── ...
```

---

## 🔍 วิธีค้นหาไฟล์ที่ต้องการ

### **ต้องการทำอะไร?** → **ไปที่ไฟล์ไหน?**

#### **🐍 Python Backend**

| ต้องการ | ไฟล์ | คำสั่ง |
|---------|------|--------|
| **เปรียบเทียบสินค้า** | `main.py` | `python main.py old.csv new.csv` |
| **หาสินค้าที่ไม่ซ้ำ** | `complete_deduplication_pipeline.py` | `python complete_deduplication_pipeline.py --input products.csv --mode extract` |
| **เริ่ม API Server** | `api_server.py` | `python api_server.py` |
| **Human Review** | `complete_deduplication_pipeline.py` | `python complete_deduplication_pipeline.py --mode review` |
| **เทรน ML Model** | `complete_deduplication_pipeline.py` | `python complete_deduplication_pipeline.py --mode train` |
| **ทำความสะอาด CSV** | `clean_csv_products.py` | `python clean_csv_products.py input.csv` |
| **แก้ไข Text Processing** | `fresh_implementations.py` | (Lines 28-87: TextProcessor) |
| **แก้ไข Embedding Model** | `fresh_implementations.py` | (Lines 89-200: EmbeddingModel) |
| **แก้ไข Similarity Calculation** | `fresh_implementations.py` | (Lines 202-280: SimilarityCalculator) |

#### **⚛️ Next.js Frontend**

| ต้องการ | ไฟล์ | URL |
|---------|------|-----|
| **Dashboard** | `taxonomy-app/app/page.tsx` | http://localhost:3000 |
| **จัดการ Taxonomy** | `taxonomy-app/app/taxonomy/page.tsx` | http://localhost:3000/taxonomy |
| **จัดการ Synonym** | `taxonomy-app/app/synonyms/page.tsx` | http://localhost:3000/synonyms |
| **ตรวจสอบสินค้า** | `taxonomy-app/app/products/page.tsx` | http://localhost:3000/products |
| **Import สินค้า** | `taxonomy-app/app/import/page.tsx` | http://localhost:3000/import |
| **แก้ไข Database** | `taxonomy-app/utils/supabase.ts` | - |
| **แก้ไข API** | `taxonomy-app/app/api/*/route.ts` | - |
| **แก้ไข Components** | `taxonomy-app/components/` | - |

#### **📚 Documentation**

| ต้องการ | ไฟล์ |
|---------|------|
| **เริ่มต้นใช้งาน** | `docs/INDEX.md` |
| **API Documentation** | `docs/api/api-reference.md` |
| **System Architecture** | `docs/development/architecture.md` |
| **Thai Text Processing** | `docs/development/text-preprocessing.md` |
| **Human Feedback Guide** | `docs/guides/human-feedback.md` |

---

## 🚀 Quick Start Commands

### **Python Backend**

```bash
# 1. ติดตั้ง dependencies
pip install -r requirements.txt

# 2. เปรียบเทียบสินค้า (Basic)
python main.py input/old_product/products.xlsx input/new_product/products.xlsx

# 3. หาสินค้าที่ไม่ซ้ำ (Advanced)
python complete_deduplication_pipeline.py --input input/new_product/products.xlsx --mode extract

# 4. เริ่ม API Server
python api_server.py
# เปิด http://localhost:8000/docs

# 5. Human Review
python complete_deduplication_pipeline.py --input input/new_product/products.xlsx --mode review --reviewer "Your Name"
```

### **Next.js Frontend**

```bash
# 1. ติดตั้ง dependencies
cd taxonomy-app
npm install

# 2. ตั้งค่า environment
cp .env.local.example .env.local
# แก้ไข .env.local ใส่ Supabase credentials

# 3. รัน development server
npm run dev
# เปิด http://localhost:3000

# 4. Build production
npm run build
npm start

# 5. รัน tests
npm test
```

---

## 🔗 ความสัมพันธ์ระหว่าง 2 ระบบ (Workflow ที่ถูกต้อง)

```
┌─────────────────────────────────────────────────────────────┐
│                    USER WORKFLOW                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  1. อัปโหลดสินค้าใหม่ + สินค้าเก่า     │
        │     → Product Similarity Checker      │
        │     → web_server.py (Flask Web UI)    │
        └───────────────────┬───────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  2. AI ประมวลผลหาสินค้าที่ไม่ซ้ำ      │
        │     → fresh_implementations.py        │
        │     → complete_deduplication_pipeline │
        └───────────────────┬───────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  3. Human Review & Approval           │
        │     → Flask Web Interface             │
        │     → ตัดสินใจอนุมัติสินค้า            │
        └───────────────────┬───────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  4. Export approved_products_*.csv    │
        │     → ไฟล์สินค้าที่ผ่านการอนุมัติ      │
        └───────────────────┬───────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  5. Import ไปยัง Taxonomy Manager    │
        │     → taxonomy-app/import/page.tsx    │
        │     → จัดหมวดหมู่สินค้าที่อนุมัติแล้ว   │
        └───────────────────────────────────────┘
```

### **การเชื่อมต่อ**

1. **Product Similarity Checker** (`web_server.py`) ทำหน้าที่หาสินค้าที่ไม่ซ้ำและให้ Human อนุมัติ
2. **ผลลัพธ์** คือไฟล์ `approved_products_*.csv` ที่มีสินค้าที่ผ่านการอนุมัติ
3. **Taxonomy Manager** นำไฟล์ `approved_products_*.csv` มาจัดหมวดหมู่
4. **Human Feedback** ถูกบันทึกใน **SQLite** และใช้เทรน **ML Model**
5. **ข้อมูลหมวดหมู่** ถูกบันทึกใน **Supabase** และแสดงผลใน **Taxonomy Manager**

---

## 📝 Best Practices

### **การจัดระเบียบโค้ด**

1. **Python Backend**:
   - ใช้ `fresh_implementations.py` สำหรับ core algorithms
   - ใช้ `fresh_architecture.py` สำหรับ architecture components
   - แยก utility functions ไปยัง `utils/`

2. **Next.js Frontend**:
   - ใช้ App Router (`app/`) สำหรับ pages
   - แยก components ตามฟีเจอร์ (`components/Taxonomy/`, `components/Synonym/`)
   - ใช้ `utils/supabase.ts` สำหรับทุก database operations

3. **Testing**:
   - เขียน tests ใน `tests/` (Python) และ `__tests__/` (Next.js)
   - ใช้ `pytest` สำหรับ Python และ `Jest` สำหรับ Next.js

4. **Documentation**:
   - อัปเดต `docs/` เมื่อเพิ่มฟีเจอร์ใหม่
   - ใช้ `docs/INDEX.md` เป็นจุดเริ่มต้น

---

## 🎯 สรุป (ความเข้าใจที่ถูกต้อง)

โปรเจกต์นี้มี **2 ระบบหลัก** ที่ทำงานเป็นลำดับขั้นตอน:

1. **Product Similarity Checker** (Python + Flask Web Frontend)
   - หาสินค้าที่ไม่ซ้ำกัน
   - AI-powered matching
   - Human Review & Approval System
   - **ผลลัพธ์**: `approved_products_*.csv`

2. **Thai Product Taxonomy Manager** (Next.js Frontend)
   - รับไฟล์ `approved_products_*.csv` จากระบบแรก
   - จัดการหมวดหมู่สินค้าที่อนุมัติแล้ว
   - Taxonomy และ Synonym Management
   - Premium SaaS UI/UX

**ไฟล์สำคัญที่สุด**:
- 🐍 `web_server.py` - Flask Web Frontend สำหรับ Human Review
- 🐍 `complete_deduplication_pipeline.py` - Deduplication system
- 🐍 `fresh_implementations.py` - Core algorithms
- 📄 `output/approved_products_*.csv` - ไฟล์ผลลัพธ์สำคัญ
- ⚛️ `taxonomy-app/app/import/page.tsx` - รับไฟล์ approved products
- ⚛️ `taxonomy-app/utils/supabase.ts` - Database operations
- 📖 `docs/INDEX.md` - Documentation hub

**เริ่มต้นที่นี่**:
1. อ่าน `README.md` (root และ taxonomy-app)
2. อ่าน `docs/INDEX.md`
3. ทดลองรัน `python main.py` และ `npm run dev`

---

**Made with ❤️ for Thai E-commerce**
