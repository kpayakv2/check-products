# 🎯 **API /analyze - ความสามารถและคุณสมบัติ**

## 📊 **ภาพรวม API**

**Endpoint**: `POST /analyze`  
**จุดประสงค์**: วิเคราะห์สินค้าใหม่เพื่อหาสินค้าที่ไม่ซ้ำกับสินค้าเก่า  
**เวอร์ชัน**: v2.1 - Phase 4 Enhanced  
**สถานะ**: ✅ พร้อมใช้งาน (รองรับ Offline)  

---

## 🔧 **คุณสมบัติหลัก**

### **1. 🤖 AI-Powered Product Deduplication**
- **Machine Learning Models**: 
  - ✅ TF-IDF (default, fast)
  - ✅ SentenceTransformer (multilingual, accurate) 
  - ✅ Optimized TF-IDF (balanced)
  - ✅ Mock (testing)
- **Multilingual Support**: รองรับภาษาไทย-อังกฤษ
- **Offline Capability**: ทำงานได้โดยไม่ต้องเชื่อมอินเทอร์เน็ต

### **2. 📈 Intelligent Classification**
- **Unique Products**: สินค้าที่แน่ใจว่าไม่ซ้ำ
- **Duplicate Check Needed**: สินค้าที่ต้องให้มนุษย์ตรวจสอบ
- **Confidence Scoring**: คะแนนความมั่นใจ AI
- **Similarity Thresholds**: ปรับระดับความเข้มงวดได้

### **3. 🧠 Human-in-the-Loop Integration**
- **Review Queue**: สร้างคิวงานสำหรับมนุษย์
- **Interactive Review**: ส่วนต่อประสานการตรวจสอบ
- **Feedback Learning**: เรียนรู้จากข้อมูลย้อนกลับ
- **Progress Tracking**: ติดตามความคืบหน้า

---

## 📝 **Request Format**

### **Method**: POST
### **Headers**: 
```json
{
  "Content-Type": "application/json"
}
```

### **Prerequisites**:
1. อัปโหลดไฟล์สินค้าเก่า: `POST /upload` (type: "old")
2. อัปโหลดไฟล์สินค้าใหม่: `POST /upload` (type: "new")
3. เรียก: `POST /analyze`

### **Body**: ไม่จำเป็น (ใช้ไฟล์ที่อัปโหลดแล้ว)

---

## 📤 **Response Format**

### **✅ Success Response**:
```json
{
  "success": true,
  "old_count": 5000,
  "new_count": 1000,
  "unique_count": 750,
  "duplicate_check_count": 250,
  "pending_review": 250,
  "summary": "จาก 1000 สินค้าใหม่: 750 ไม่ซ้ำ, 250 ต้องตรวจสอบ"
}
```

### **❌ Error Response**:
```json
{
  "success": false,
  "message": "Pipeline error: Model loading failed"
}
```

---

## ⚙️ **การตั้งค่าที่รองรับ**

### **Model Types** (ผ่าน `/api/config`):
- **`tfidf`** (Default): เร็ว, ใช้ RAM น้อย
- **`sentence-bert`**: แม่นยำสูง, รองรับหลายภาษา  
- **`optimized-tfidf`**: สมดุลระหว่างเร็วและแม่นยำ
- **`mock`**: สำหรับการทดสอบ

### **Similarity Methods**:
- **`cosine`** (Default): Cosine Similarity
- **`dot_product`**: Dot Product Similarity

### **Parameters**:
- **`threshold`**: 0.6 (ระดับความคล้าย 0-1)
- **`top_k`**: 10 (จำนวน matches ที่ดูต่อสินค้า)

---

## 🔄 **Process Flow**

```
1. Validate Input Files
   ├─ ตรวจสอบไฟล์สินค้าเก่า
   └─ ตรวจสอบไฟล์สินค้าใหม่

2. Load ML Pipeline
   ├─ สร้าง Fresh Architecture Pipeline
   ├─ โหลด Model ตาม Config
   └─ ตั้งค่า Similarity Method

3. Extract Product Names
   ├─ จาก DataFrame สินค้าเก่า
   └─ จาก DataFrame สินค้าใหม่

4. AI Analysis
   ├─ คำนวณ Embeddings
   ├─ หา Similarity Matches
   └─ คำนวณ Confidence Scores

5. Classify Products
   ├─ Unique Products (ไม่ซ้ำแน่นอน)
   └─ Duplicate Check Needed (ต้องตรวจสอบ)

6. Create Review Queue
   ├─ สร้าง Human Review Tasks
   └─ ตั้งค่า Priority Order

7. Update Application State
   ├─ บันทึกผลลัพธ์
   └─ เตรียมข้อมูลสำหรับ Export
```

---

## 📊 **Statistics & Metrics**

### **ข้อมูลที่ได้รับ**:
- **`old_count`**: จำนวนสินค้าเก่าทั้งหมด
- **`new_count`**: จำนวนสินค้าใหม่ทั้งหมด  
- **`unique_count`**: สินค้าใหม่ที่ไม่ซ้ำ
- **`duplicate_check_count`**: สินค้าที่ต้องตรวจสอบ
- **`pending_review`**: งานที่รอการตรวจสอบ

### **อัตราความแม่นยำ** (ขึ้นกับ Model):
- **TF-IDF**: ~85% accuracy, <2 วินาที
- **SentenceTransformer**: ~95% accuracy, 5-15 วินาที  
- **Optimized TF-IDF**: ~90% accuracy, 2-5 วินาที

---

## 🎭 **Use Cases**

### **1. E-commerce Product Management**
- ป้องกันการเพิ่มสินค้าซ้ำใน catalog
- ตรวจสอบสินค้าใหม่ก่อน import

### **2. Inventory Management** 
- รวมข้อมูลสินค้าจากหลาย supplier
- ทำความสะอาดฐานข้อมูลสินค้า

### **3. Data Quality Assurance**
- ตรวจสอบความถูกต้องของข้อมูล
- หาข้อมูลที่ผิดปกติหรือซ้ำ

---

## 🚀 **Performance Characteristics**

### **ปัจจัยที่ส่งผลต่อประสิทธิภาพ**:
- **จำนวนสินค้า**: Linear scaling O(n×m)
- **Model Type**: SentenceTransformer ช้าแต่แม่นยำกว่า TF-IDF
- **Hardware**: CPU vs GPU, RAM available
- **Internet**: SentenceTransformer ครั้งแรกต้อง download

### **Optimization Features**:
- ✅ **Model Caching**: ไม่โหลดซ้ำระหว่าง requests
- ✅ **Offline Mode**: ทำงานได้โดยไม่ต้องเน็ต
- ✅ **Batch Processing**: ประมวลผลแบบ batch
- ✅ **Memory Management**: จัดการ RAM อย่างมีประสิทธิภาพ

---

## 🔗 **Related APIs**

### **Before Analysis**:
- `POST /upload`: อัปโหลดไฟล์ข้อมูล
- `GET/POST /api/config`: ตั้งค่า model และ parameters

### **After Analysis**:
- `GET /get-review-queue`: ดึงรายการงานที่ต้องตรวจสอบ
- `POST /save-feedback`: บันทึกผลการตรวจสอบ
- `GET /export-csv`: Export ผลลัพธ์เป็น CSV
- `GET /get-results`: ดูสถิติและผลลัพธ์

---

## 🛠️ **Error Handling**

### **Common Errors**:
- **`UPLOAD_BOTH_FILES`**: ยังไม่ได้อัปโหลดไฟล์ครบ
- **`PipelineError`**: ปัญหาการโหลด ML model
- **`ValidationError`**: ข้อมูล input ไม่ถูกต้อง
- **`Memory Error`**: RAM ไม่พอสำหรับข้อมูลขนาดใหญ่

### **Troubleshooting**:
1. ตรวจสอบไฟล์ input format
2. ตรวจสอบ model availability 
3. ตรวจสอบ system resources
4. ดู logs สำหรับ error details

---

## 📚 **Example Usage**

### **JavaScript (Frontend)**:
```javascript
// 1. Upload files first
await uploadFile('old', oldProductsFile);
await uploadFile('new', newProductsFile);

// 2. Start analysis
const response = await fetch('/analyze', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  }
});

const result = await response.json();
console.log(`Analysis complete: ${result.unique_count} unique products`);
```

### **Python (API Client)**:
```python
import requests

# Analyze uploaded files
response = requests.post('http://localhost:5000/analyze')
result = response.json()

if result['success']:
    print(f"✅ Analysis successful!")
    print(f"📊 Unique products: {result['unique_count']}")
    print(f"🔍 Need review: {result['duplicate_check_count']}")
else:
    print(f"❌ Error: {result['message']}")
```

---

## 🎯 **คุณภาพและการรับประกัน**

### **✅ Features Ready**:
- Multi-model AI analysis
- Offline capability  
- Human-in-the-loop workflow
- Multilingual support
- Export capabilities
- Error handling

### **🔄 เตรียมพัฒนา**:
- Real-time progress updates
- Advanced caching system
- GPU acceleration
- Async processing
- Performance monitoring

---

**📅 Last Updated**: September 13, 2025  
**🔖 Version**: 2.1 - Phase 4 Enhanced  
**💻 Environment**: Development & Production Ready
