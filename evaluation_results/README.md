# Evaluation Results

ผลการทดสอบอัลกอริทึมการจัดหมวดหมู่สินค้า

## 📁 ไฟล์ในโฟลเดอร์นี้

### **JSON Results:**
- `category_eval_keyword_*.json` - ผลลัพธ์จาก Keyword Method
- `category_eval_embedding_*.json` - ผลลัพธ์จาก Embedding Method  
- `category_eval_hybrid_*.json` - ผลลัพธ์จาก Hybrid Method

### **Reports:**
- `category_eval_comparison_*.md` - รายงานเปรียบเทียบทั้ง 3 วิธี

## 📊 สรุปผลลัพธ์

| Method | Coverage | Avg Confidence | Status |
|--------|----------|----------------|--------|
| Keyword | 100% | 0.83 | ✅ Ready |
| Embedding | 100% | 0.49 | ✅ Ready |
| Hybrid | 100% | 0.72 | ✅ **Recommended** |

## 🎯 วิธีดูผลลัพธ์

### **1. ดูรายงานสรุป:**
```
category_eval_comparison_[timestamp].md
```

### **2. ดูผลลัพธ์ละเอียด:**
```python
import json

# อ่านไฟล์ JSON
with open('category_eval_hybrid_*.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# ดูข้อมูลสินค้าแรก
print(results[0])
```

### **3. วิเคราะห์เพิ่มเติม:**
```bash
# รัน visualization script
python ../visualize_results.py
```

## 📈 Visualizations

ดูกราฟใน folder `../visualizations/`:
- Confidence distribution
- Method comparison
- Category distribution
- Performance metrics

## 🔄 รันใหม่

```bash
cd ..
python test_category_algorithm.py
```

ผลลัพธ์จะถูกบันทึกใน folder นี้พร้อม timestamp ใหม่
