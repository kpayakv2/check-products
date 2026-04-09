# ⚡ Quick Reference - Product Checker

> **คู่มือค้นหาไฟล์แบบรวดเร็ว - สำหรับ Developer**

---

## 🎯 ต้องการทำอะไร? → ไปที่ไฟล์ไหน?

### 🔍 **การค้นหาแบบรวดเร็ว**

| ฉันต้องการ... | ไฟล์ที่ต้องแก้ | บรรทัด/ส่วน |
|---------------|---------------|------------|
| **🌐 เริ่ม Human Review Web UI** | `web_server.py` | Flask Web Frontend |
| **เปรียบเทียบสินค้า 2 ไฟล์** | `main.py` | Entry point |
| **หาสินค้าที่ไม่ซ้ำ + อนุมัติ** | `complete_deduplication_pipeline.py` | `extract_unique_products()` |
| **แก้วิธีทำความสะอาดข้อความไทย** | `fresh_implementations.py` | Lines 28-87 (ThaiTextProcessor) |
| **เปลี่ยน Embedding Model** | `fresh_implementations.py` | Lines 89-200 (EmbeddingModel) |
| **แก้การคำนวณ Similarity** | `fresh_implementations.py` | Lines 202-280 (SimilarityCalculator) |
| **แก้ Human Review UI** | `web/human_review.html` | HTML/JS |
| **ดูไฟล์ผลลัพธ์** | `output/approved_products_*.csv` | ไฟล์สำคัญ! |
| **เทรน ML Model** | `ml_feedback_learning.py` | ContinuousLearningSystem |
| **Import สินค้าที่อนุมัติแล้ว** | `taxonomy-app/app/import/page.tsx` | รับไฟล์ approved_products |
| **จัดการ Taxonomy** | `taxonomy-app/app/taxonomy/page.tsx` | React Component |
| **จัดการ Synonym** | `taxonomy-app/app/synonyms/page.tsx` | React Component |
| **แก้ Database Schema** | `taxonomy-app/supabase/schema.sql` | SQL |
| **แก้ Database Operations** | `taxonomy-app/utils/supabase.ts` | TypeScript |
| **เพิ่ม API Route (Next.js)** | `taxonomy-app/app/api/*/route.ts` | Next.js API |

---

## 🐍 Python Backend - File Map

### **Entry Points (เริ่มต้นที่นี่)**

```python
# 1. 🌐 Human Review Web UI (สำคัญที่สุด!)
python web_server.py  # → http://localhost:5000

# 2. เปรียบเทียบสินค้าพื้นฐาน
python main.py old.csv new.csv

# 3. ระบบครบวงจร
python complete_deduplication_pipeline.py --input products.csv --mode analyze

# 4. API Server (FastAPI)
python api_server.py  # → http://localhost:8000
```

### **Core Algorithms (แก้ไขที่นี่)**

```
fresh_implementations.py
├── Lines 28-87:   ThaiTextProcessor
│   ├── clean_text()           # ทำความสะอาดข้อความ
│   ├── normalize_thai()       # Normalize ภาษาไทย
│   └── tokenize()             # แยกคำ
│
├── Lines 89-200:  EmbeddingModel
│   ├── TfidfEmbeddingModel    # TF-IDF
│   ├── SentenceBertModel      # Sentence BERT
│   └── encode()               # สร้าง embeddings
│
└── Lines 202-280: SimilarityCalculator
    ├── cosine_similarity()    # Cosine similarity
    └── calculate_batch()      # Batch processing
```

### **Architecture Components**

```
fresh_architecture.py
├── ProductMatcher             # Core matching logic
├── ProductSimilarityPipeline  # Complete pipeline
└── Config                     # Configuration
```

### **Human Feedback System**

```
human_feedback_system.py
├── ProductDeduplicationSystem # หาสินค้าซ้ำ
├── HumanReviewInterface       # UI สำหรับ review
└── HumanFeedbackDatabase      # SQLite database

ml_feedback_learning.py
├── ContinuousLearningSystem   # เทรนโมเดลจาก feedback
└── FeedbackLearningModel      # ML model
```

---

## ⚛️ Next.js Frontend - File Map

### **Pages (หน้าจอหลัก)**

```
taxonomy-app/app/
├── page.tsx                   # 🏠 Dashboard
├── taxonomy/page.tsx          # 🌳 Taxonomy Tree
├── synonyms/page.tsx          # 📚 Synonym Manager
├── products/page.tsx          # 🛍️ Product Review
└── import/page.tsx            # 📤 Import System
```

### **Components (UI Components)**

```
taxonomy-app/components/
├── Taxonomy/
│   ├── TaxonomyTree.tsx              # Basic tree
│   └── EnhancedTaxonomyTree.tsx      # Enhanced (Drag & Drop)
│
├── Synonym/
│   ├── SynonymManager.tsx            # Basic manager
│   └── EnhancedSynonymManager.tsx    # Enhanced (CSV import)
│
├── Product/
│   ├── ProductReview.tsx             # Basic review
│   └── EnhancedProductReview.tsx     # Enhanced (Keyboard shortcuts)
│
└── Search/
    └── HybridSearch.tsx              # Vector + Text search
```

### **API Routes**

```
taxonomy-app/app/api/
├── taxonomy/route.ts          # GET, POST, PUT, DELETE taxonomy
├── synonyms/route.ts          # GET, POST, PUT, DELETE synonyms
├── products/route.ts          # GET, POST products
└── similarity/route.ts        # Similarity matching
```

### **Database & Utilities**

```
taxonomy-app/utils/
├── supabase.ts                # 🗄️ Database operations (ใช้บ่อยที่สุด)
├── rate-limit.ts              # Rate limiting
├── error-handler.ts           # Error handling
├── validation.ts              # Zod validation
└── logger.ts                  # Logging
```

---

## 📊 Data Flow - ข้อมูลไหลผ่านไฟล์ไหนบ้าง?

### **Scenario 1: Human Review Workflow (หลัก)**

```
1. web_server.py (Flask Web UI)
   ↓
2. อัปโหลดไฟล์สินค้าใหม่ + เก่า
   ↓
3. fresh_implementations.py (AI Processing)
   ├── ThaiTextProcessor.clean_text()
   ├── EmbeddingModel.encode()
   └── SimilarityCalculator.calculate()
   ↓
4. Human Review & Approval
   ↓
5. output/approved_products_*.csv
```

### **Scenario 2: หาสินค้าที่ไม่ซ้ำ**

```
1. complete_deduplication_pipeline.py --mode extract
   ↓
2. human_feedback_system.py (ProductDeduplicationSystem)
   ↓
3. fresh_implementations.py (Core algorithms)
   ↓
4. ml_feedback_learning.py (ML predictions)
   ↓
5. output/unique_products.csv
```

### **Scenario 3: Human Review**

```
1. complete_deduplication_pipeline.py --mode review
   ↓
2. human_feedback_system.py (HumanReviewInterface)
   ↓
3. web/human_review.html (Web UI)
   ↓
4. human_feedback.db (SQLite)
   ↓
5. ml_feedback_learning.py (เทรนโมเดล)
```

### **Scenario 4: Import สินค้าที่อนุมัติแล้วไปยัง Taxonomy Manager**

```
1. รับไฟล์ approved_products_*.csv จาก web_server.py
   ↓
2. taxonomy-app/app/import/page.tsx (Upload UI)
   ↓
3. taxonomy-app/app/api/products/route.ts (API)
   ↓
4. taxonomy-app/utils/supabase.ts (Database)
   ↓
5. Supabase (PostgreSQL + pgvector)
   ↓
6. taxonomy-app/app/taxonomy/page.tsx (จัดหมวดหมู่)
```

---

## 🔧 Configuration Files

| ไฟล์ | หน้าที่ | แก้ไขเมื่อ |
|------|---------|-----------|
| **`requirements.txt`** | Python dependencies | เพิ่ม Python package |
| **`requirements-dev.txt`** | Dev dependencies | เพิ่ม dev tools |
| **`pyproject.toml`** | Python project config | ตั้งค่า Black, isort, pytest |
| **`pytest.ini`** | Pytest config | ตั้งค่า testing |
| **`taxonomy-app/package.json`** | npm dependencies | เพิ่ม npm package |
| **`taxonomy-app/tsconfig.json`** | TypeScript config | ตั้งค่า TypeScript |
| **`taxonomy-app/tailwind.config.js`** | Tailwind CSS | ตั้งค่า styling |
| **`taxonomy-app/.env.local`** | Environment variables | Supabase credentials |

---

## 🧪 Testing Files

### **Python Tests**

```
tests/
├── integration/
│   ├── test_api_endpoints.py      # API endpoint tests
│   ├── test_complete_pipeline.py  # Pipeline tests
│   └── test_human_feedback.py     # Human feedback tests
│
└── config/
    └── test_config.py             # Configuration tests
```

### **Next.js Tests**

```
taxonomy-app/__tests__/
├── components/
│   ├── TaxonomyTree.test.tsx      # Component tests
│   ├── SynonymManager.test.tsx
│   └── ProductReview.test.tsx
│
├── api/
│   ├── taxonomy.test.ts           # API tests
│   └── products.test.ts
│
└── utils/
    └── supabase.test.ts           # Utility tests
```

---

## 📚 Documentation Files

```
docs/
├── INDEX.md                       # 📖 เริ่มต้นที่นี่
│
├── guides/                        # คู่มือใช้งาน
│   ├── quick-start.md
│   ├── human-feedback.md
│   └── embedding-models.md
│
├── api/                           # API Docs
│   ├── api-reference.md
│   └── quick-reference.md
│
└── development/                   # Dev Guides
    ├── architecture.md
    ├── text-preprocessing.md
    └── contributing.md
```

---

## 🚀 Common Tasks - คำสั่งที่ใช้บ่อย

### **Python Backend**

```bash
# 🌐 เริ่ม Human Review Web UI (สำคัญที่สุด!)
python web_server.py
# เปิด http://localhost:5000

# เปรียบเทียบสินค้า (CLI)
python main.py input/old_product/products.xlsx input/new_product/products.xlsx

# หาสินค้าที่ไม่ซ้ำ
python complete_deduplication_pipeline.py \
  --input input/new_product/products.xlsx \
  --mode extract \
  --output output/approved_products.csv

# Human Review (CLI)
python complete_deduplication_pipeline.py \
  --input input/new_product/products.xlsx \
  --mode review \
  --reviewer "Your Name"

# เทรน ML Model
python complete_deduplication_pipeline.py \
  --mode train

# เริ่ม API Server (FastAPI)
python api_server.py

# ทำความสะอาด CSV
python clean_csv_products.py input/products.csv

# รัน Tests
pytest
pytest tests/integration/test_api_endpoints.py -v
```

### **Next.js Frontend**

```bash
cd taxonomy-app

# Development
npm run dev                        # เริ่ม dev server
npm run build                      # Build production
npm start                          # รัน production

# Testing
npm test                           # รัน tests
npm run test:watch                 # Watch mode
npm run test:coverage              # Coverage report

# Linting
npm run lint                       # ESLint
npm run lint:fix                   # Auto-fix

# Database
npm run db:migrate                 # Run migrations
npm run db:seed                    # Seed data
```

---

## 🔍 วิธีค้นหาโค้ดเฉพาะเจาะจง

### **ต้องการแก้ไข Text Processing**

```bash
# ค้นหาไฟล์ที่เกี่ยวข้อง
grep -r "clean_text" --include="*.py"
grep -r "ThaiTextProcessor" --include="*.py"

# ไฟล์หลัก: fresh_implementations.py (Lines 28-87)
```

### **ต้องการแก้ไข Embedding Model**

```bash
# ค้นหาไฟล์ที่เกี่ยวข้อง
grep -r "EmbeddingModel" --include="*.py"
grep -r "sentence-transformers" --include="*.py"

# ไฟล์หลัก: fresh_implementations.py (Lines 89-200)
```

### **ต้องการแก้ไข Database Schema**

```bash
# Next.js: taxonomy-app/supabase/schema.sql
# Python: human_feedback.db (SQLite)
```

### **ต้องการแก้ไข API Endpoints**

```bash
# Python: api_server.py
# Next.js: taxonomy-app/app/api/*/route.ts
```

---

## 🎯 File Importance Ranking

### ⭐⭐⭐⭐⭐ **ไฟล์สำคัญที่สุด (แก้บ่อย)**

1. **`web_server.py`** - Flask Web UI สำหรับ Human Review
2. **`fresh_implementations.py`** - Core algorithms
3. **`output/approved_products_*.csv`** - ไฟล์เชื่อมต่อระหว่าง 2 ระบบ
4. **`taxonomy-app/app/import/page.tsx`** - รับไฟล์ approved products
5. **`taxonomy-app/utils/supabase.ts`** - Database operations

### ⭐⭐⭐⭐ **ไฟล์สำคัญ**

6. **`complete_deduplication_pipeline.py`** - Deduplication pipeline
7. **`main.py`** - Entry point (CLI)
8. **`human_feedback_system.py`** - Human feedback
9. **`taxonomy-app/app/taxonomy/page.tsx`** - Taxonomy management
10. **`taxonomy-app/components/*/`** - UI components

### ⭐⭐⭐ Supporting Files

11. **`fresh_architecture.py`** - Architecture
12. **`ml_feedback_learning.py`** - ML learning

### **⭐⭐ Utility Files**

12. **`clean_csv_products.py`** - CSV cleaning
13. **`filter_matched_products.py`** - Filtering
14. **`taxonomy-app/utils/*.ts`** - Utilities

### **⭐ Configuration Files**

15. **`requirements.txt`** - Dependencies
16. **`taxonomy-app/package.json`** - npm packages
17. **`.env.local`** - Environment variables

---

## 💡 Pro Tips

### **1. ใช้ VS Code Search**

```
Ctrl+Shift+F (Windows) / Cmd+Shift+F (Mac)
```

ค้นหาคำสำคัญ:
- `def clean_text` - หา text processing
- `class EmbeddingModel` - หา embedding model
- `async function` - หา async functions (Next.js)
- `CREATE TABLE` - หา database schema

### **2. ใช้ Git Blame**

```bash
git blame fresh_implementations.py -L 28,87
```

ดูว่าใครแก้ไขส่วนไหนล่าสุด

### **3. ใช้ Tree Command**

```bash
tree -L 2 -I 'node_modules|.next|__pycache__'
```

ดูโครงสร้างโปรเจกต์

### **4. ใช้ Grep แบบ Smart**

```bash
# ค้นหาเฉพาะ Python files
grep -r "pattern" --include="*.py"

# ค้นหาเฉพาะ TypeScript files
grep -r "pattern" --include="*.ts" --include="*.tsx"

# ค้นหาและแสดงบรรทัดรอบ ๆ
grep -r "pattern" -A 5 -B 5
```

---

## 🔗 Related Files - ไฟล์ที่เกี่ยวข้องกัน

### **Text Processing Chain**

```
fresh_implementations.py (ThaiTextProcessor)
  ↓ ใช้โดย
fresh_architecture.py (ProductMatcher)
  ↓ ใช้โดย
main.py, complete_deduplication_pipeline.py, api_server.py
```

### **Database Chain**

```
taxonomy-app/supabase/schema.sql (Schema)
  ↓ ใช้โดย
taxonomy-app/utils/supabase.ts (Operations)
  ↓ ใช้โดย
taxonomy-app/app/api/*/route.ts (API)
  ↓ ใช้โดย
taxonomy-app/app/*/page.tsx (UI)
```

### **Human Feedback Chain**

```
human_feedback_system.py (Database)
  ↓ ใช้โดย
ml_feedback_learning.py (ML Training)
  ↓ ใช้โดย
complete_deduplication_pipeline.py (Pipeline)
  ↓ แสดงผลใน
web/human_review.html (UI)
```

---

## 📞 Need Help?

1. **อ่าน Documentation**: `docs/INDEX.md`
2. **ดู Examples**: `tests/examples/`
3. **ค้นหาใน Code**: ใช้ VS Code Search
4. **ดู Git History**: `git log --oneline`
5. **เปิด Issue**: GitHub Issues

---

**🎯 Tip**: บุ๊คมาร์กไฟล์นี้ไว้เพื่อค้นหาไฟล์แบบรวดเร็ว!
