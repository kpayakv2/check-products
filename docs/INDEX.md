# Product Similarity Checker - Documentation

Welcome to the comprehensive documentation for the Product Similarity Checker project. This system provides advanced product deduplication using machine learning and human-in-the-loop feedback.

## 📚 Documentation Structure

### 🚀 [User Guides](guides/)
- **[Quick Start Guide](guides/quick-start.md)** - Get up and running quickly
- **[Human Feedback System](guides/human-feedback.md)** - How to use the human-in-the-loop interface
- **[Embedding Models Guide](guides/embedding-models.md)** - Understanding and configuring ML models
- **[Model Download Guide](guides/model-download.md)** - How to download and manage models

### 🔧 [API Documentation](api/)
- **[API Reference](api/api-reference.md)** - Complete API documentation
- **[Quick Reference](api/quick-reference.md)** - Essential API endpoints
- **[Analysis Capabilities](api/analyze-capabilities.md)** - Advanced analysis features

### 👩‍💻 [Development](development/)
- **[Contributing Guide](development/contributing.md)** - How to contribute to the project
- **[Architecture Overview](development/architecture.md)** - System architecture and design
- **[Test Organization](development/test-organization.md)** - Testing framework and structure
- **[Changelog](development/changelog.md)** - Version history and changes
- **[Shared Scoring System](development/shared-scoring.md)** - Scoring algorithm details
- **[Text Preprocessing](development/text-preprocessing.md)** - Text processing pipeline

### 📊 [Reports & Analysis](reports/)
- **[Test Results Summary](reports/test-results-summary.md)** - Latest test execution results
- **[Cleanup Reports](reports/)** - Project cleanup and optimization reports
- **[Sentence Transformer Analysis](reports/sentence-transformer-analysis.md)** - ML model performance analysis
- **[Capabilities Summary](reports/capabilities-summary.md)** - System capabilities overview
- **[Organization Summary](reports/organization-summary.md)** - Project organization details

### 🗄️ [Archive](archive/)
- Historical documentation and deprecated guides

---

## 🎯 **Quick Navigation**

### **สำหรับผู้ใช้ใหม่:**
1. อ่าน [`README.md`](../README.md) ก่อน
2. ดูตัวอย่างใน Quick Start section
3. ทดลองใช้ Web Interface หรือ API

### **สำหรับ Developer:**
1. ศึกษา [`architecture.md`](development/architecture.md) เพื่อเข้าใจโครงสร้าง
2. อ่าน [`text-preprocessing.md`](development/text-preprocessing.md) สำหรับ text processing
3. ใช้ [`test organization`](development/test-organization.md) สำหรับการทดสอบ

### **สำหรับการใช้งาน API:**
1. ดู [`api-reference.md`](api/api-reference.md) สำหรับรายละเอียด endpoint
2. ทดสอบผ่าน Swagger UI: `http://localhost:8000/docs`
3. ใช้ [`test_api_client.py`](../test_api_client.py) เป็นตัวอย่าง

---

## 🔄 **Document Status**

### **✅ Consolidated & Organized:**
- 🎯 เอกสารถูกจัดหมวดหมู่ตามบทบาท (Guides / API / Development / Reports / Archive)
- 🧹 เนื้อหาซ้ำซ้อนถูกย้ายออกจากเอกสารหลักทั้งหมด
- 📋 มี `docs/INDEX.md` เป็นศูนย์กลางการนำทาง
- 🗑️ ไฟล์เก่าที่ไม่จำเป็นถูกย้ายไปไว้ใน `docs/archive/`

### **📂 ปัจจุบัน `docs/` มีโครงสร้างดังนี้:**
```
docs/
├── INDEX.md                  # 🗂️ Navigation index (เอกสารนี้)
├── api/                      # 🔌 REST & WebSocket documentation
├── guides/                   # 🚀 Quick start & how-to guides
├── development/              # 🛠️ Technical references for developers
├── reports/                  # 📊 Project reports & cleanup history
└── archive/                  # 🗄️ เอกสารอ้างอิงเก็บถาวร
```

**🎯 เริ่มต้นที่นี่ แล้วเลือกหมวดที่เหมาะกับงานของคุณได้ทันที!**

---

**🎯 เริ่มต้นด้วย [`README.md`](../README.md) เพื่อการใช้งานที่สมบูรณ์!**
