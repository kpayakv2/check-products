# 🚀 Quick Start Guide

ยินดีต้อนรับสู่ **Product Similarity Checker** เครื่องมือสำหรับตรวจจับสินค้าซ้ำและคัดกรองสินค้าใหม่ก่อนนำเข้าระบบของคุณ คู่มือนี้สรุปขั้นตอนสำคัญเพื่อให้คุณเริ่มใช้งานได้ภายในไม่กี่นาที

---

## ✅ ก่อนเริ่มใช้งาน

- ติดตั้ง **Python 3.9+**
- ติดตั้งเครื่องมือพื้นฐาน: `git`, `pip`, `virtualenv` (แนะนำ)
- ดาวน์โหลดโมเดลผ่านคำสั่ง `python download_models.py` (ถ้ายังไม่มีไฟล์ใน `model_cache/`)

---

## ⚙️ การติดตั้ง

```bash
git clone <repository-url>
cd check-products

# สร้างและเปิดใช้งาน virtual environment (แนะนำ)
python -m venv venv
venv\Scripts\activate  # Windows
# หรือ
source venv/bin/activate  # macOS/Linux

# ติดตั้ง dependencies
pip install -r requirements.txt
```

---

## 💻 การใช้งานผ่านคำสั่ง (CLI)

```bash
# รันการจับคู่สินค้าแบบ batch
python main.py input/old_product/products.xlsx input/new_product/products.xlsx \
    --output output/matched_products.csv --threshold 0.6

# พารามิเตอร์สำคัญ
# - old_products_file : ไฟล์สินค้าต้นฉบับ (ต้องมีคอลัมน์ `name`)
# - new_products_file : ไฟล์สินค้าใหม่ (ต้องมีคอลัมน์ `รายการ`)
# - --threshold       : เกณฑ์ตัดสินความเหมือน (0.0-1.0)
# - --top-k           : จำนวนสินค้าที่ต้องการให้ระบบแนะนำ (ค่าเริ่มต้น 10)
```

ผลลัพธ์จะถูกบันทึกไว้ใน `output/` พร้อม metadata เช่น confidence score และอันดับของคู่สินค้า

---

## 🌐 Web Interface สำหรับ Human Review

```bash
python web_server.py
# เปิดเบราว์เซอร์ที่ http://localhost:5000
```

จุดเด่น:

- อัปโหลดไฟล์สินค้าเก่า/ใหม่ แสดงผลลัพธ์แบบอินเตอร์แอคทีฟ
- ตรวจสอบสินค้าซ้ำ, ยืนยันหรือปฏิเสธได้แบบเรียลไทม์
- บันทึก human feedback เพื่อนำไปปรับปรุงโมเดลในอนาคต

---

## 🔌 REST API & WebSocket

```bash
python api_server.py
# API Docs: http://localhost:8000/docs
# Web UI (เดียวกับ API server): http://localhost:8000/web
```

Endpoint สำคัญ:

- `POST /api/v1/match/single` – ตรวจจับสินค้าคล้ายสำหรับ 1 สินค้า
- `POST /api/v1/match/batch` – ส่งชุดข้อมูลเพื่อตรวจจับสินค้าคล้ายหลายตัว
- `POST /api/v1/match/upload` – อัปโหลดไฟล์และรอผลลัพธ์แบบ asynchronous
- WebSocket `/ws` – รับอัปเดตสถานะงานแบบเรียลไทม์

---

## 🧪 การทดสอบระบบ

```bash
# รันการทดสอบทั้งหมด
pytest

# รันเฉพาะหมวดตัวอย่าง
pytest tests/examples/test_refactored_example.py

# รันการทดสอบ API integration
pytest tests/integration/test_api_endpoints.py
```

---

## 📚 เอกสารประกอบที่ควรอ่านต่อ

- `README.md` – ภาพรวมของระบบทั้งหมด
- `docs/INDEX.md` – จุดเชื่อมไปยังเอกสารทุกหมวด
- `docs/development/architecture.md` – โครงสร้างระบบและโมดูลที่สำคัญ
- `docs/development/text-preprocessing.md` – รายละเอียด Thai text pipeline
- `docs/api/api-reference.md` – รายละเอียด REST API + ตัวอย่างคำสั่ง

---

## 🎉 พร้อมใช้งาน

- ระบบรองรับทั้ง **Automation (CLI)**, **Integration (API)** และ **Human-in-the-loop Review (Web)**
- สามารถประมวลผลสินค้าหลายพันรายการได้อย่างรวดเร็ว พร้อมระบบคะแนนความเชื่อมั่น
- รองรับภาษาไทยเต็มรูปแบบด้วย Thai Text Preprocessing Pipeline

หากต้องการขยายระบบเพิ่มเติมหรือผนวกกับ Thai Product Taxonomy Manager สามารถดูข้อมูลเพิ่มเติมได้ภายในโฟลเดอร์ `taxonomy-app/` และ `docs/development/`

ขอให้สนุกกับการใช้งาน! 🚀
