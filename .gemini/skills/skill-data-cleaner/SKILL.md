---
name: skill-data-cleaner
description: |
  Expert in normalizing and cleaning Thai product names for AI processing.
  ใช้เมื่อต้องการ Normalize ชื่อสินค้า, จัดการหน่วยวัด, หรือลบ Noise ก่อนเข้า AI Pipeline

  **Trigger when user asks to:**
  - ทำความสะอาดชื่อสินค้าภาษาไทย (clean / normalize)
  - แปลงเลขไทยเป็นเลขอารบิก
  - ลบข้อความโปรโมชั่น/โฆษณาออกจากชื่อสินค้า
  - จัดการหน่วยวัด (กก., ก., มล., ลิตร)
  - Import/batch process ข้อมูลสินค้า

  **Keywords:** normalize, clean, preprocessing, เลขไทย, หน่วยวัด, โปรโมชั่น, ThaiTextProcessor, noise, batch
---

# 🧹 Thai Product Data Cleaner Skill
*Expert in normalizing and cleaning Thai product names for AI processing*

## 🎯 Role & Expertise
- เชี่ยวชาญการใช้ `ThaiTextProcessor` จาก `fresh_implementations.py` **(ไม่ใช่ `ProductTextProcessor`)**
- รู้วิธีจัดการกับความซับซ้อนของภาษาไทย (สระลอย, เลขไทย, คำพ้อง)
- เชี่ยวชาญการ Normalize หน่วยวัด (Unit Normalization) และการลบข้อความโปรโมชั่น

## ⚠️ Mandates
- ต้องใช้ **`ThaiTextProcessor`** จาก `fresh_implementations.py` เท่านั้น
- ต้องรักษาความหมายดั้งเดิมของชื่อสินค้าไว้เสมอ (Don't over-clean)
- การทำความสะอาดต้องสอดคล้องกับมาตรฐานใน `docs/development/text-preprocessing.md`

## 🛠️ Key Workflows

### 1. Data Normalization
```python
from fresh_implementations import ThaiTextProcessor

processor = ThaiTextProcessor()
cleaned = processor.clean_text("สบู่  โพรเทคส์ ๑๐๐ก. ซื้อ 1 แถม 1")
# → "สบู่ โพรเทคส์ 100ก."
```

### 2. Normalization Rules (มาตรฐาน)
- แปลงเลขไทย → อารบิก: `๑๐๐` → `100`
- ลบสระลอย / Zero-width Space (`\u200b`, `\u200c`)
- มาตรฐานหน่วยวัด: `100 g` → `100ก.`, `1 kg` → `1กก.`, `500 ml` → `500มล.`
- ลบโปรโมชั่น: `ซื้อ 1 แถม 1`, `ลด 50%`, `ราคาพิเศษ`
- ลบ Brand Prefixes: `แบรนด์`, `ยี่ห้อ`, `Original`

### 3. Batch Cleaning
```python
import pandas as pd

def clean_product_batch(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['name_clean'] = df['name'].apply(processor.clean_text)
    df = df.drop_duplicates(subset=['name_clean'])
    df = df[df['name_clean'].str.len() >= 3]
    return df
```

### 4. Thai Script Correction
- ตรวจสอบและแก้ไขการวางตำแหน่งอักษรภาษาไทยที่ผิดเพี้ยน (สระลอย/สระจม)
- จัดการล้างข้อมูลทีละชุด (Batch) ผ่าน Pandas หรือ SQL อย่างมีประสิทธิภาพ

## Quality Checklist หลัง Clean
- [ ] ไม่มีเลขไทยหลงเหลือ (`[๐-๙]`)
- [ ] ไม่มีช่องว่างซ้ำ
- [ ] ไม่มีข้อความโปรโมชั่น
- [ ] หน่วยวัดเป็นมาตรฐาน (ก./กก./มล./ล.)
- [ ] ชื่อสินค้ายาวอย่างน้อย 3 ตัวอักษร

## Anti-patterns

| ❌ Bad | ✅ Good |
|--------|---------|
| ใช้ `ProductTextProcessor` | ใช้ `ThaiTextProcessor` เท่านั้น |
| เปรียบเทียบชื่อสินค้าดิบ | Clean ก่อนเสมอ |
| ลบวงเล็บทั้งหมด `(แพ็ค 4)` | เก็บข้อมูลปริมาณไว้ |
| Process ทีละรายการใน Loop ใหญ่ | ใช้ `pandas.apply()` |
