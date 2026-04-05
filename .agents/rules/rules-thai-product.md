# 🇹🇭 Thai Product Taxonomy Rules

## Context
ใช้เมื่อมีการจัดการข้อมูลสินค้าภาษาไทย, การพัฒนาอัลกอริทึมแยกหมวดหมู่, หรือการสร้าง UI สำหรับแสดงผลข้อมูลสินค้าไทย

## Standards
- **Normalization**: ต้องใช้ `TextPreprocessor` เสมอ เพื่อให้การเปรียบเทียบข้อความเป็นมาตรฐานเดียวกัน
- **Tokenization**: การตัดคำต้องใช้ชุดพจนานุกรมที่รองรับศัพท์เฉพาะของสินค้า
- **Naming Convention**:
  - Frontend (TS): `camelCase`
  - Backend (Python): `snake_case`
- **Type Safety**: กำหนด Interface/Type ให้ชัดเจนเสมอ ห้ามใช้ `any`

## Examples

### ✅ Good: การทำความสะอาดข้อความก่อนประมวลผล
```python
# ใช้ Preprocessor ที่โปรเจกต์กำหนด
from utils.product_data_utils import TextPreprocessor

processor = TextPreprocessor()
cleaned_name = processor.normalize("สบู่ โพรเทคส์  100ก. (แพ็ค 4)")
# ผลลัพธ์: "สบู่ โพรเทคส์ 100ก. (แพ็ค 4)" (ตัดช่องว่างซ้ำ)
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
}
```

### ❌ Bad: การใช้ any
```typescript
const handleProduct = (data: any) => {
  console.log(data.name);
}
```

