# 🤖 Embedding Models Guide (v3.0)

คู่มือการจัดการโมเดล AI สำหรับสร้าง Vector ในระบบ Thai Product Taxonomy

---

## 📋 โมเดลมาตรฐานที่ใช้งาน
ระบบคุณกานถูกล็อกให้ใช้โมเดลเดียวเพื่อความแม่นยำและเสถียรภาพ:

*   **Model**: `paraphrase-multilingual-MiniLM-L12-v2`
*   **Dimensions**: 384 numbers
*   **Language**: รองรับ 50+ ภาษา (รวมไทยและอังกฤษ)
*   **Provider**: FastAPI Server (api_server.py) รันที่ Port 8000

---

## 🚀 การตั้งค่าในระบบ (Configuration)

หากต้องการเปลี่ยนพฤติกรรมการประมวลผล ให้ตรวจสอบในไฟล์ **`api_server.py`**:

```python
# ระบบปัจจุบันรองรับ 3 โหมด:
1. "sentence-bert" (Default/High Accuracy) ✅ แนะนำ
2. "optimized-tfidf" (Balanced)
3. "mock" (For Fast Testing only)
```

**⚠️ ข้อควรระวัง**: ห้ามใช้โหมด `mock` ในงานจริง เพราะจะทำให้ค่า Vector เป็นค่าสุ่ม และความแม่นยำจะลดลงต่ำกว่า 10%

---

## 💾 การใช้งานแบบ Offline

คุณกานสามารถรัน AI ได้โดยไม่ต้องต่ออินเทอร์เน็ต หากดาวน์โหลดโมเดลมาไว้ในเครื่องแล้ว:

1.  **ดาวน์โหลด**: รัน `python simple_download.py`
2.  **ตรวจสอบ**: เช็คว่ามีโฟลเดอร์ `model_cache/` ใน Root Directory
3.  **รัน**: เมื่อรัน `python api_server.py` ระบบจะดึงไฟล์จาก Cache มาใช้งานโดยอัตโนมัติ

---

## 📊 ประสิทธิภาพ (Performance)

| โหมด | เวลาโหลด | ความเร็ว/สินค้า | ความแม่นยำ |
|------|----------|----------------|------------|
| **Sentence-BERT** | ~5-10 วินาที | < 0.05s | ⭐⭐⭐⭐⭐ |
| **TF-IDF** | ~1-2 วินาที | < 0.01s | ⭐⭐⭐ |

---

**📅 Last Updated**: 16 เมษายน 2569 (Verified for FastAPI Implementation)
