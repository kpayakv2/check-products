---
name: data-cleaner
description: |
  ทักษะสำหรับทำความสะอาดและ Normalize ข้อมูลชื่อสินค้าภาษาไทย
  ใช้เมื่อต้องการ Normalize ชื่อสินค้า, จัดการหน่วยวัด, หรือลบ Noise ก่อนเข้า AI Pipeline

  **Trigger when user asks to:**
  - ทำความสะอาดชื่อสินค้าภาษาไทย (clean / normalize)
  - แปลงเลขไทยเป็นเลขอารบิก
  - ลบข้อความโปรโมชั่น/โฆษณาออกจากชื่อสินค้า
  - จัดการหน่วยวัด (กก., ก., มล., ลิตร)
  - Import/batch process ข้อมูลสินค้า

  **Keywords:** normalize, clean, preprocessing, เลขไทย, หน่วยวัด, โปรโมชั่น, ThaiTextProcessor, noise, batch
---

# 🧹 Data Cleaner Skill

## ภาพรวม (Overview)

ชื่อสินค้าภาษาไทยจากร้านค้าปลีกมักมี Noise หลายรูปแบบ:
- เลขไทย: `๑๐๐` แทน `100`
- ช่องว่างซ้ำ: `สบู่  โพรเทคส์`
- สระลอย/ตัวอักษรพิเศษ: `ส​บู่` (มี Zero-width Space)
- ข้อความโปรโมชั่น: `ซื้อ 1 แถม 1`, `ลด 50%`
- หน่วยวัดไม่มาตรฐาน: `100 g`, `100g`, `100ก.`

---

## ใช้ ThaiTextProcessor (มาตรฐานโปรเจกต์)

```python
# นำเข้าจาก fresh_implementations.py เท่านั้น
from fresh_implementations import ThaiTextProcessor

processor = ThaiTextProcessor()
```

### Methods หลัก

| Method | ทำอะไร | Input | Output |
|--------|---------|-------|--------|
| `clean_text(text)` | Normalize ทั้งหมด | `"สบู่  โพรเทคส์ ๑๐๐ก."` | `"สบู่ โพรเทคส์ 100ก."` |
| `remove_promotions(text)` | ลบโปรโมชั่น | `"นม ซื้อ 1 แถม 1 1L"` | `"นม 1L"` |
| `normalize_units(text)` | มาตรฐานหน่วย | `"100 g"` | `"100ก."` |
| `convert_thai_digits(text)` | แปลงเลขไทย | `"๑๒๓"` | `"123"` |

---

## Normalization Rules (มาตรฐาน)

### 1. เลขไทย → อารบิก
```python
thai_digits = str.maketrans('๐๑๒๓๔๕๖๗๘๙', '0123456789')
text = text.translate(thai_digits)
```

### 2. ลบสระลอย / Zero-width Characters
```python
import re
# ลบ Zero-width Space (U+200B) และ Zero-width Non-joiner (U+200C)
text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
```

### 3. มาตรฐานหน่วยวัด
```python
UNIT_MAP = {
    r'\b(\d+)\s*g\b': r'\1ก.',
    r'\b(\d+)\s*kg\b': r'\1กก.',
    r'\b(\d+)\s*ml\b': r'\1มล.',
    r'\b(\d+)\s*l\b': r'\1ล.',
    r'\b(\d+)\s*ก\b': r'\1ก.',
}
for pattern, replacement in UNIT_MAP.items():
    text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
```

### 4. คำ Noise ที่ต้องลบ
```python
NOISE_PATTERNS = [
    r'ซื้อ\s*\d+\s*แถม\s*\d+',  # ซื้อ 1 แถม 1
    r'ลด\s*\d+\s*%',              # ลด 50%
    r'ราคาพิเศษ',
    r'โปรโมชั่น',
    r'สุดคุ้ม',
    r'\(แพ็ค\s*\d+\)',            # (แพ็ค 4) — optional: อาจเก็บไว้
]
```

---

## Batch Processing Pattern

```python
import pandas as pd
from fresh_implementations import ThaiTextProcessor

processor = ThaiTextProcessor()

def clean_product_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    ทำความสะอาดชื่อสินค้าทั้งหมดใน DataFrame
    """
    df = df.copy()
    df['name_clean'] = df['name'].apply(processor.clean_text)
    
    # ลบรายการที่ clean แล้วซ้ำกัน
    df = df.drop_duplicates(subset=['name_clean'])
    
    # ลบรายการที่ชื่อสั้นเกินไป (น้อยกว่า 3 ตัวอักษร)
    df = df[df['name_clean'].str.len() >= 3]
    
    return df
```

---

## Quality Checklist หลัง Clean

- [ ] ไม่มีเลขไทยหลงเหลือ (`[๐-๙]` regex = 0 matches)
- [ ] ไม่มีช่องว่างซ้ำ (ทดสอบด้วย `'  ' in text`)
- [ ] ไม่มีข้อความโปรโมชั่น
- [ ] หน่วยวัดเป็นมาตรฐาน (ก./กก./มล./ล.)
- [ ] ชื่อสินค้ายาวอย่างน้อย 3 ตัวอักษร

---

## Anti-patterns

| ❌ Bad | ✅ Good |
|--------|---------|
| เปรียบเทียบชื่อสินค้าดิบโดยตรง | Clean ก่อนเสมอ |
| ลบวงเล็บทั้งหมด `(แพ็ค 4)` | เก็บข้อมูลปริมาณไว้ |
| แปลงภาษาไทยเป็น ASCII | รักษาภาษาไทยไว้ |
| Process ทีละรายการใน Loop ใหญ่ | ใช้ `pandas.apply()` |
