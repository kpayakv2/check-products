# 🎯 Shared Scoring System (Hybrid Algorithm v3.0)

## 📋 Overview

เอกสารนี้อธิบายสูตรการคำนวณคะแนนความคล้าย (Hybrid Scoring) ที่ใช้จริงในระบบปัจจุบัน เพื่อความสอดคล้องระหว่าง AI Backend และ Database Logic

---

## 🧮 **The Hybrid Formula**

ระบบใช้การผสมผสานระหว่าง **ความแม่นยำทางคำศัพท์** (Deterministic) และ **ความเข้าใจทางภาษ**า (Semantic) ดังนี้:

```
Hybrid_Score = (Keyword_Match_Score × 0.6) + (Embedding_Similarity × 0.4)
```

### **1. Keyword Match Score (60%)**
*   **ที่มา**: นับความถี่ของ Keywords ที่ตรงกับ `taxonomy_nodes.keywords`
*   **ค่าคะแนน**: ปรับจูนให้อยู่ในสเกล 0.0 - 1.0 (ผ่านการ Normalize)
*   **จุดเด่น**: ป้องกันการจัดหมวดผิดในกรณีสินค้าชื่อคล้ายกันแต่คนละประเภท (เช่น "น้ำยาล้างจาน" vs "เครื่องล้างจาน")

### **2. Embedding Similarity (40%)**
*   **ที่มา**: Cosine Similarity จากโมเดล `paraphrase-multilingual-MiniLM-L12-v2`
*   **ค่าคะแนน**: 0.0 - 1.0
*   **จุดเด่น**: ช่วยให้ระบบเข้าใจคำพ้องความหมาย (Synonyms) และคำสะกดผิด (Typos)

---

## 🚦 **Confidence Levels & Thresholds**

เพื่อให้มั่นใจในคุณภาพข้อมูล (Data Integrity) ระบบกำหนดระดับความมั่นใจดังนี้:

| Level | Score Range | Action | UI Color |
|-------|-------------|--------|----------|
| **High** | `> 0.90` | **Auto-Approve** (ยอมรับอัตโนมัติ) | 🟢 เขียว |
| **Medium** | `0.70 - 0.89` | **Review Required** (ต้องให้คนตรวจ) | 🟡 เหลือง |
| **Low** | `< 0.70` | **Uncertain** (ไม่มั่นใจ/ต้องจับคู่ใหม่) | 🔴 แดง |

---

## 🧪 **Implementation Standards**

### **API Response Schema**
```json
{
  "product_name": "มาม่า ต้มยำกุ้ง 60ก",
  "category": "บะหมี่กึ่งสำเร็จรูป",
  "confidence_score": 0.92,
  "confidence_level": "high",
  "method": "hybrid",
  "breakdown": {
    "keyword_score": 0.95,
    "embedding_score": 0.88
  }
}
```

---

## 📈 **Performance Goal**
*   **Auto-Assign Rate**: > 70% ของสินค้าทั้งหมด
*   **Manual Review**: < 25% ของสินค้าใหม่
*   **Latency**: < 200ms ต่อการประมวลผล 1 รายการ

**📅 Last Updated**: 16 เมษายน 2569
