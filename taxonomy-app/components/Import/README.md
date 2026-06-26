# 📁 Import Components (The Magic Wizard)

โฟลเดอร์นี้เป็นส่วนหลักของ **Import System** แบบ **5-Step Magic Wizard** สำหรับวิเคราะห์ ทำความสะอาด และจัดหมวดหมู่สินค้าด้วย AI

---

## 📂 ชิ้นส่วน Components

### **1. UploadAndMappingStep.tsx** 
**ขั้นตอนที่ 1: อัปโหลดและจับคู่คอลัมน์**
- ลากวางไฟล์ (Drag & Drop) หรือเลือกไฟล์ CSV/XLSX
- จับคู่คอลัมน์ชื่อสินค้า (Product Name) และรายละเอียดอื่นๆ อัตโนมัติ
- *Component รวมร่างจากรุ่นเก่าที่เป็น Upload และ Mapping แยกกัน*

### **2. DataCleaningStep.tsx**
**ขั้นตอนที่ 2: ทำความสะอาดข้อมูลภาษาไทย (Data Cleaning)**
- ส่งข้อมูลไปที่ FastAPI (`/api/v1/clean`) เพื่อใช้ `ThaiTextProcessor`
- สกัดชื่อสินค้าให้บริสุทธิ์ ลบคำสแปม, โฆษณา, และหน่วยวัด
- แสดง Preview เปรียบเทียบชื่อเดิม (Original) กับชื่อที่ผ่านการล้างแล้ว (Cleaned)

### **3. DeduplicationStep.tsx**
**ขั้นตอนที่ 3: คัดกรองของซ้ำ (Deduplication Triage)**
- ส่งข้อมูลไปที่ FastAPI (`/api/v1/match/batch`) เพื่อค้นหาความเหมือน
- แสดง Dashboard วิเคราะห์ความเสี่ยง 3 ระดับ (High, Medium, Low Risk)
- มีปุ่ม Action แบบ Batch ให้ "ข้าม (Skip)" สินค้าที่ซ้ำกัน 100% อย่างรวดเร็ว

### **4. CategorizationStep.tsx**
**ขั้นตอนที่ 4: ตรวจสอบหมวดหมู่จาก AI (AI Categorization)**
- AI แนะนำหมวดหมู่ด้วย Hybrid Algorithm (Keyword 60% + Embedding 40%)
- UI แนว "Review-by-Exception" จัดกลุ่มความมั่นใจ (Confidence) เป็นสีเขียว, เหลือง, แดง
- ผู้ใช้กด Approve แบบกวาดเรียบเฉพาะสีเขียว แล้วค่อยใช้เวลาตัดสินใจกับสีแดง

### **5. CompleteStep.tsx**
**ขั้นตอนที่ 5: สรุปผล (Summary & Celebration)**
- แสดงสถิติการนำเข้า: สำเร็จ, ล้มเหลว, ข้าม
- แผนภูมิหรือสรุปหมวดหมู่ที่นำเข้าบ่อยสุด
- Confetti Animation 🎉

### **6. WizardLayout.tsx**
**แกนกลางของโครงสร้าง (Layout & Navigation)**
- จัดการ State ของแต่ละขั้นตอน
- แสดงแถบความคืบหน้า (Progress Bar / Stepper) แบบ UI ทันสมัย

---

## 🔄 Workflow การทำงาน (Pipeline)

```
┌─────────────────────────────────────────────────────────────┐
│  1. Upload & Mapping (Frontend)                             │
│     - อัปโหลดไฟล์ → หาคอลัมน์ product_name                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  2. Data Cleaning (Backend: ThaiTextProcessor)              │
│     - ลบคำโฆษณา, จัดการหน่วยวัด → Preview ให้ผู้ใช้รีวิว  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  3. Deduplication (Backend: Vector + Exact Match)           │
│     - คัดกรองของซ้ำ (Triage) → ข้ามของเดิม, รับของใหม่    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  4. AI Categorization (Backend: Hybrid Algorithm)           │
│     - แนะนำหมวดหมู่ → ผู้ใช้ Review-by-Exception            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  5. Save & Complete (Database)                              │
│     - บันทึกลงตาราง products → แสดงสถิติ                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 คุณสมบัติเด่น (UI/UX)
- **Glassmorphism & Premium Design:** สไตล์การออกแบบทันสมัย สะอาดตา
- **No-Mock Ready:** เตรียมพร้อมสำหรับการเชื่อมต่อ Real API ในอนาคต
- **State-hoisting:** ส่งผ่านข้อมูล `cleanedData`, `dedupedData`, `categorizedData` ขึ้นไประดับ `page.tsx` อย่างราบรื่น

**อัปเดตล่าสุด:** พฤษภาคม 2569 (Complete Overhaul - 5-Step Pipeline)
