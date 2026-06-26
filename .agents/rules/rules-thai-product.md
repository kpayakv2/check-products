---
name: rules-thai-product
description: |
  กฎมาตรฐานสำหรับการจัดการข้อมูลสินค้าภาษาไทย
  ครอบคลุม Text Normalization, Type Safety, และ Naming Convention

triggers:
  - งานที่เกี่ยวกับ taxonomy_nodes หรือ classification
  - แก้ไขหรือเพิ่มข้อมูลสินค้าภาษาไทย
  - เขียน Python ที่ประมวลผลชื่อสินค้า
  - เขียน TypeScript สำหรับ Product interface
  - ใช้ร่วมกับ skill-thai-taxonomy-expert และ skill-data-cleaner
---

# 🇹🇭 Thai Product Taxonomy Rules

## Context
ใช้เมื่อมีการจัดการข้อมูลสินค้าภาษาไทย, การพัฒนาอัลกอริทึมแยกหมวดหมู่, หรือการสร้าง UI สำหรับแสดงผลข้อมูลสินค้าไทย

## Standards
- **Normalization**: ต้องใช้ `ThaiTextProcessor` (จาก `fresh_implementations.py`) เสมอ เพื่อให้การเปรียบเทียบข้อความเป็นมาตรฐานเดียวกัน
- **Tokenization**: การตัดคำต้องใช้ชุดพจนานุกรมที่รองรับศัพท์เฉพาะของสินค้า
- **Embedding Dimension**: ต้องใช้ **384 dimensions** (โมเดล `paraphrase-multilingual-MiniLM-L12-v2`) เท่านั้น
- **Naming Convention**:
  - Frontend (TS): `camelCase`
  - Backend (Python): `snake_case`
- **Type Safety**: กำหนด Interface/Type ให้ชัดเจนเสมอ ห้ามใช้ `any`

## 🔭 Code Exploration Before Editing (Socraticode)

ก่อนแก้ไขโค้ดที่เกี่ยวกับ Text Processing ต้องทำขั้นตอนนี้เสมอ:

1. **รู้ blast radius ก่อน** — แก้ `ThaiTextProcessor` อาจกระทบหลายไฟล์:
   ```
   codebase_impact { target: "ThaiTextProcessor" }
   codebase_impact { target: "fresh_implementations.py" }
   ```

2. **ดู callers ทั้งหมดก่อนเปลี่ยน signature:**
   ```
   codebase_symbol { name: "ThaiTextProcessor" }
   codebase_symbol { name: "clean_text" }
   ```

3. **ค้นหา usage pattern ที่คล้าย:**
   ```
   codebase_search { query: "thai text normalization tokenization" }
   ```

## Examples

### ✅ Good: การทำความสะอาดข้อความก่อนประมวลผล
```python
# ใช้ ThaiTextProcessor ที่โปรเจกต์กำหนด (fresh_implementations.py)
from fresh_implementations import ThaiTextProcessor

processor = ThaiTextProcessor()
cleaned_name = processor.clean_text("สบู่ โพรเทคส์  ๑๐๐ก. (แพ็ค 4)")
# ผลลัพธ์: "สบู่ โพรเทคส์ 100ก. (แพ็ค 4)" (แปลงเลขไทย + ตัดช่องว่างซ้ำ)
```

### ❌ Bad: การเปรียบเทียบข้อความดิบ
```python
# ไม่ควรทำ เพราะอาจมีช่องว่างหรืออักขระพิเศษต่างกัน
if product_a.name == product_b.name:
    pass
```

### ✅ Good: การกำหนด Type ใน TypeScript
```typescript
interface Product {
  id: string;
  name: string;
  category_id: number;
  embedding?: number[]; // 384 dimensions
}
```

### ❌ Bad: การใช้ any
```typescript
const handleProduct = (data: any) => {
  console.log(data.name);
}
```
