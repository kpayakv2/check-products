# แผนปฏิบัติการ: AI จัดหมวดหมู่สินค้า (Updated v3.0)
อัปเดต: 16 เมษายน 2569

> เป้าหมาย: เมื่อเพิ่มสินค้าใหม่ให้ระบบ **ใส่หมวดอัตโนมัติ** โดยเน้นความแม่นยำสูงสุด (High Precision)

---

## 0) ภาพรวมโครงการ (KPIs)
- **Auto-assign Threshold**: ≥ **0.90** (AI มั่นใจสูงมากถึงจะใส่หมวดให้เอง)
- **Review Threshold**: **0.70 – 0.89** (ต้องให้พนักงานกดเลือกจาก Top-3)
- **Accuracy Target**: ≥ **90%** สำหรับเคส Auto-assign
- **Manual Workload**: เคสที่ต้อง Review ต้องไม่เกิน **25%** ของรายการสินค้าใหม่

---

## 🧠 1. กลไกการประมวลผล (Hybrid Algorithm)

ระบบใช้สูตร **60/40** ตามกฎเหล็กของโปรเจกต์:
1.  **Keyword Match (60%)**: อ้างอิงจากกฎใน `keyword_rules`
2.  **Vector Embedding (40%)**: ใช้โมเดล `paraphrase-multilingual-MiniLM-L12-v2` (384-dim)

---

## 🔄 2. ขั้นตอนเมื่อมีการเพิ่มสินค้าใหม่

1.  **Normalization**: ล้างข้อมูลด้วย `ThaiTextProcessor` (แปลงเลขไทย, ลบสระลอย, มาตรฐานหน่วยวัด)
2.  **Scoring**: คำนวณ Hybrid Score เทียบกับ Taxonomy ทั้งหมด
3.  **Action**:
    *   **Score ≥ 0.90**: ✅ **Auto-Approve** (ระบุหมวดหมู่ทันที)
    *   **Score 0.70 - 0.89**: 🔍 **Needs Review** (แสดงตัวเลือกให้คนคลิกยืนยัน)
    *   **Score < 0.70**: ⚠️ **Unmapped** (ส่งเข้าคิวงาน "รอดำเนินการ")

---

## 📈 3. การพัฒนาต่อเนื่อง (Continuous Learning)

เราใช้ระบบ **Human-in-the-Loop** ผ่านสคริปต์ `complete_deduplication_pipeline.py`:
*   ข้อมูลที่คนตรวจ (Approve/Reject) จะถูกบันทึกลงฐานข้อมูล Feedback
*   ระบบจะใช้ข้อมูลนี้ในการ **Train** โมเดลเสริม (`joblib`) เพื่อปรับจูนน้ำหนักในครั้งถัดไป
*   หากพบคำศัพท์ใหม่บ่อยๆ ระบบจะแนะนำให้เพิ่มเข้าไปใน `synonyms.csv` หรือ `keyword_rules`

---

**📅 Last Updated**: 16 เมษายน 2569  
**🔖 Status**: Deployment Phase (Production Ready)
