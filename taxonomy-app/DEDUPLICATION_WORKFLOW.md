# 🔄 Product Deduplication Workflow

## จุดประสงค์หลัก
**คัดกรองสินค้าที่ไม่ซ้ำ → จัดหมวดหมู่ → ส่งออก CSV สำหรับนำเข้า POS**

## 🎯 กระบวนการทำงาน 5 ขั้นตอน

### 1. 📤 อัปโหลดไฟล์
- **Input**: 
  - `old_products.csv` - สินค้าเก่าที่มีอยู่ในระบบ
  - `new_products.csv` - สินค้าใหม่ที่ต้องการเพิ่ม
- **รองรับ**: CSV, Excel (.xlsx, .xls)
- **การทำงาน**: อัปโหลดไป Supabase Storage

### 2. 🔍 คัดกรองสินค้าซ้ำ
- **Algorithm**: TF-IDF + Jaccard Similarity
- **Threshold**: 0.75 (ปรับได้)
- **การจำแนก**:
  - **Similarity ≥ 95%**: ซ้ำมาก → ตัดออกอัตโนมัติ
  - **75% ≤ Similarity < 95%**: ต้องตรวจสอบด้วยมนุษย์
  - **Similarity < 75%**: ไม่ซ้ำ → ผ่านอัตโนมัติ

### 3. 👤 ตรวจสอบด้วยมนุษย์
- **Human-in-the-Loop Review**
- **ตัวเลือก**:
  - ✅ **แตกต่าง** (ไม่ซ้ำ) - ผ่าน
  - ⚠️ **คล้ายกัน** (ต้องดูเพิ่ม) - ผ่าน
  - ❌ **ซ้ำกัน** (ตัดออก) - ไม่ผ่าน
- **บันทึก**: Human feedback ใน database

### 4. 🏷️ จัดหมวดหมู่
- **AI Category Suggestions**: ใช้ Edge Function
- **Input**: ชื่อสินค้าที่ผ่านการคัดกรอง
- **Output**: หมวดหมู่ + Confidence Score
- **Fallback**: "ไม่ระบุหมวดหมู่" หาก AI ไม่แน่ใจ

### 5. 📄 ส่งออก CSV
- **Format**: CSV พร้อม UTF-8 BOM
- **Columns**:
  - รหัสสินค้า
  - ชื่อสินค้า  
  - หมวดหมู่
  - ความเชื่อมั่น
  - สถานะ
- **Filename**: `products_for_pos_YYYY-MM-DD.csv`

## 🏗️ Technical Architecture

### Frontend (React)
```typescript
/app/deduplication/page.tsx
├── File Upload Interface
├── Workflow Steps Tracker  
├── Human Review Interface
├── Results Table
└── CSV Export Function
```

### Backend (Supabase Edge Function)
```typescript
/functions/product-deduplication/
├── CSV Parsing
├── Similarity Calculation
├── Product Classification
└── Results Formatting
```

### Database Schema
```sql
human_feedback
├── old_product (TEXT)
├── new_product (TEXT)  
├── similarity_score (FLOAT)
├── human_decision (TEXT)
├── ml_prediction (TEXT)
└── reviewer_id (UUID)
```

## 📊 Data Flow

```
CSV Files → Supabase Storage → Edge Function → Similarity Analysis
    ↓
Classification → Human Review → Category Suggestions → CSV Export
    ↓
Database Storage ← Human Feedback ← Review Decisions
```

## 🎛️ Configuration

### Similarity Thresholds
- **High Duplicate**: ≥ 95% (auto-exclude)
- **Review Needed**: 75% - 95% (human review)
- **Unique**: < 75% (auto-approve)

### Category Suggestions
- **Min Confidence**: 0.3
- **Max Suggestions**: 1 per product
- **Fallback**: "ไม่ระบุหมวดหมู่"

## 📈 Performance Metrics

### Processing Stats
- **Total New Products**: จำนวนสินค้าใหม่ทั้งหมด
- **Auto Approved**: สินค้าที่ผ่านอัตโนมัติ
- **Needs Review**: สินค้าที่ต้องตรวจสอบ
- **Excluded Duplicates**: สินค้าที่ตัดออกอัตโนมัติ
- **Processing Time**: เวลาประมวลผล (ms)

### Quality Metrics
- **Human Agreement**: ความสอดคล้องระหว่าง AI กับมนุษย์
- **Category Confidence**: ความเชื่อมั่นในการจัดหมวดหมู่
- **Review Completion**: อัตราการตรวจสอบครบถ้วน

## 🔧 Usage Instructions

### 1. เตรียมไฟล์
```csv
# old_products.csv
name
iPhone 14 Pro Max
Samsung Galaxy S23
MacBook Air M2

# new_products.csv  
name
iPhone 15 Pro Max
Samsung Galaxy S24
iPad Pro 11
```

### 2. เข้าใช้งาน
1. ไปที่ `/deduplication`
2. อัปโหลดไฟล์ทั้งสองไฟล์
3. คลิก "🚀 เริ่มคัดกรองสินค้าซ้ำ"
4. ตรวจสอบสินค้าที่ต้องยืนยัน (ถ้ามี)
5. คลิก "🏷️ จัดหมวดหมู่สินค้า"
6. คลิก "📄 ส่งออก CSV สำหรับ POS"

### 3. นำเข้า POS
- ดาวน์โหลดไฟล์ CSV
- นำเข้าในระบบ POS
- ตรวจสอบข้อมูลหมวดหมู่

## 🚀 Advanced Features

### Batch Processing
- รองรับไฟล์ขนาดใหญ่
- ประมวลผลแบบ streaming
- Progress tracking แบบ real-time

### Machine Learning
- เรียนรู้จาก human feedback
- ปรับปรุง similarity algorithm
- Category suggestion improvement

### Integration
- เชื่อมต่อกับ Taxonomy Manager
- ใช้ Synonym data ในการเปรียบเทียบ
- Real-time updates ผ่าน Supabase

## 🔍 Troubleshooting

### ปัญหาที่พบบ่อย
1. **ไฟล์อ่านไม่ได้**: ตรวจสอบ encoding (UTF-8)
2. **Similarity ต่ำเกินไป**: ลด threshold
3. **Category ไม่ตรง**: เพิ่ม keywords ใน taxonomy
4. **ประมวลผลช้า**: ลดขนาดไฟล์หรือใช้ batch

### Performance Tips
- ใช้ไฟล์ CSV แทน Excel (เร็วกว่า)
- จำกัดจำนวนสินค้าต่อ batch (~1000 รายการ)
- ตรวจสอบคุณภาพข้อมูลก่อนอัปโหลด
- ใช้ taxonomy ที่มี keywords ครบถ้วน

## 📝 Best Practices

### Data Preparation
- ทำความสะอาดชื่อสินค้าก่อนอัปโหลด
- ใช้รูปแบบการตั้งชื่อที่สม่ำเสมอ
- เพิ่มข้อมูล brand/model หากมี

### Review Process
- ตรวจสอบสินค้าที่มี similarity สูงก่อน
- บันทึก feedback ที่ชัดเจน
- ใช้ keyboard shortcuts (A/R/↑↓)

### Quality Control
- ตรวจสอบผลลัพธ์ก่อนส่งออก
- ยืนยันหมวดหมู่ที่ confidence ต่ำ
- เก็บ backup ไฟล์ต้นฉบับ

---

**🎯 เป้าหมาย**: ลดการมีสินค้าซ้ำในระบบ POS และเพิ่มประสิทธิภาพการจัดหมวดหมู่สินค้า
