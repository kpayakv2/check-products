# 🤖 คู่มือการใช้ Embedding Models

## 📋 ตัวเลือก Embedding Models

### 1. Mock Model (`"mock"`)
- **ความเร็ว**: ⚡⚡⚡ เร็วมาก
- **ความแม่นยำ**: ⭐⭐ พื้นฐาน  
- **เหมาะสำหรับ**: Development, Testing, การทดสอบเร็ว
- **ขนาด**: เล็กมาก
- **เวลาโหลด**: ทันที

### 2. Optimized TF-IDF (`"optimized-tfidf"`)
- **ความเร็ว**: ⚡⚡ เร็ว
- **ความแม่นยำ**: ⭐⭐⭐ ดี
- **เหมาะสำหรับ**: Staging, Production ขนาดกลาง
- **ขนาด**: ปานกลาง
- **เวลาโหลด**: 2-5 วินาที

### 3. Sentence-BERT (`"sentence-bert"`)
- **ความเร็ว**: ⚡ ช้า (ครั้งแรก)
- **ความแม่นยำ**: ⭐⭐⭐⭐⭐ สูงสุด
- **เหมาะสำหรับ**: Production ที่ต้องการความแม่นยำสูง
- **ขนาด**: ใหญ่ (~500MB)
- **เวลาโหลด**: 30-60 วินาที (ครั้งแรก)

## 🚀 วิธีการเปลี่ยน

### ในโค้ด Python
```python
from human_feedback_system import ProductDeduplicationSystem

# สำหรับ Development/Testing (เร็วมาก)
system = ProductDeduplicationSystem(embedding_model_type="mock")

# สำหรับ Production (สมดุล)  
system = ProductDeduplicationSystem(embedding_model_type="optimized-tfidf")

# สำหรับ Production (แม่นยำสูงสุด)
system = ProductDeduplicationSystem(embedding_model_type="sentence-bert")
```

### ใน Web Server (web_server.py)
ไปที่บรรทัด ~489 แล้วเปลี่ยน:

```python
# เปลี่ยนตรงนี้ 👇
embedding_model_type = "mock"  # เปลี่ยนเป็น "optimized-tfidf" หรือ "sentence-bert"

dedup_system = ProductDeduplicationSystem(
    similarity_threshold=0.8,
    embedding_model_type=embedding_model_type
)
```

## 🎯 คำแนะนำการใช้งาน

### Development Phase
```python
embedding_model_type = "mock"
```
- เร็วมาก
- ทดสอบ logic ได้ทันท
- ไม่ต้องรอ download model

### Staging/Testing Phase  
```python
embedding_model_type = "optimized-tfidf"
```
- ประสิทธิภาพดี
- ความแม่นยำยอมรับได้
- โหลดเร็ว

### Production Phase
```python
embedding_model_type = "sentence-bert"
```
- ความแม่นยำสูงสุด
- เหมาะสำหรับ business critical
- ต้องรอ download model ครั้งแรก

## 🔧 Fallback Strategy

สำหรับระบบ Production แนะนำใช้ fallback:

```python
try:
    # พยายามใช้ sentence-bert ก่อน
    system = ProductDeduplicationSystem(embedding_model_type="sentence-bert")
except Exception:
    # หากไม่ได้ ใช้ optimized-tfidf แทน
    system = ProductDeduplicationSystem(embedding_model_type="optimized-tfidf")
```

## 📊 เปรียบเทียบประสิทธิภาพ

| Model | เวลาโหลด | ความเร็วประมวลผล | ความแม่นยำ | RAM Usage |
|-------|----------|------------------|-------------|-----------|
| Mock | < 1s | ⚡⚡⚡ | ⭐⭐ | ~10MB |
| Optimized TF-IDF | 2-5s | ⚡⚡ | ⭐⭐⭐ | ~100MB |
| Sentence-BERT | 30-60s* | ⚡ | ⭐⭐⭐⭐⭐ | ~500MB |

*เฉพาะครั้งแรก หลังจากนั้นจะเร็วขึ้น

## ✅ ตัวอย่างการทดสอบ

รันคำสั่งนี้เพื่อทดสอบ models ต่างๆ:

```bash
cd "d:\product_checker\check-products"
python quick_model_demo.py
```

## 🔄 การเปลี่ยนแปลงแบบ Hot Reload

หากต้องการเปลี่ยน model โดยไม่ restart server:

1. แก้ไข `embedding_model_type` ใน web_server.py
2. Save ไฟล์
3. Restart Flask server
4. Model ใหม่จะถูกโหลดอัตโนมัติ
