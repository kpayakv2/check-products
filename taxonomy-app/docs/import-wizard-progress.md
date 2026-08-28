# 📊 Import Wizard - Progress Report

**Last Updated:** 2026-08-28
**Status:** ✅ **ใช้งานได้จริง — ทดสอบ E2E ด้วยไฟล์จริง 405 รายการแล้ว**

> **บันทึกสำคัญ:** เวอร์ชันก่อนหน้า (30 พ.ค. 2569) ระบุสถานะว่า *"Production Ready (100% UI Complete, API Connected)"* ซึ่ง**ไม่ตรงกับความจริง** — ตอนนั้นเหลือข้อเดียวที่ยังไม่ติ๊กคือ *"E2E Testing ด้วยข้อมูลจริง ไม่ใช้ข้อมูลจำลอง"* และพอทดสอบจริงเมื่อ 28 ส.ค. ก็พบว่า **wizard ไม่เคยบันทึกอะไรลงฐานข้อมูลเลยสักแถว** ทั้งที่หน้าจอขึ้นว่า "นำเข้าข้อมูลสำเร็จ"
>
> บทเรียน: UI ที่ครบทุกหน้าจอไม่ได้แปลว่าใช้งานได้ ต้องทดสอบจนเห็นข้อมูลเข้า DB จริงเท่านั้น

---

## ✅ **Completed Steps (The Magic 5-Step Flow)**

### **Step 1: Upload & Mapping** ✅
**Component:** `UploadAndMappingStep.tsx`
**Status:** Complete
- ✅ รวมหน้าอัปโหลดไฟล์และจับคู่คอลัมน์ไว้ในหน้าเดียวเพื่อลดความซ้ำซ้อน
- ✅ รองรับ CSV & XLSX (Drag & Drop)
- ✅ Auto-detect คอลัมน์ `product_name` แบบอัตโนมัติ

### **Step 2: Data Cleaning** ✅
**Component:** `DataCleaningStep.tsx`
**Status:** Complete
- ✅ เรียกใช้ Backend `/api/v1/clean` (FastAPI) แบบ Batch
- ✅ ใช้ `ThaiTextProcessor` ในการลบหน่วยวัด, คำสแปม, จัดการเลขไทย
- ✅ แสดง Preview ชื่อสินค้าก่อน-หลังทำความสะอาด

### **Step 3: Deduplication Triage** ✅
**Component:** `DeduplicationStep.tsx`
**Status:** Complete
- ✅ คัดกรองสินค้าซ้ำ (Exact Match / Semantic Match)
- ✅ Dashboard สรุปความเสี่ยง (ซ้ำแน่นอน, อาจจะซ้ำ, ไม่ซ้ำ)
- ✅ Batch Actions: ข้ามรายการที่ซ้ำ (Skip Duplicates) หรือ ดำเนินการต่อทั้งหมด
- ✅ *หมายเหตุ: UI เสร็จสมบูรณ์แล้ว เชื่อมต่อ API `/api/v1/match/batch` แล้ว*

### **Step 4: AI Categorization Review** ✅
**Component:** `CategorizationStep.tsx`
**Status:** Complete
- ✅ แสดงหมวดหมู่ที่ AI แนะนำด้วย Hybrid Algorithm (Keyword 60% + Embedding 40%)
- ✅ แสดง Confidence Score เป็นเปอร์เซ็นต์แบบมีสีสัน (เขียว, เหลือง, แดง)
- ✅ ระบบ "Review-by-Exception" กดยอมรับหมวดหมู่ที่ AI มั่นใจสูงแบบกลุ่มได้ทันที
- ✅ *หมายเหตุ: UI เสร็จสมบูรณ์แล้ว เชื่อมต่อ AI API ท้ายสุดแล้ว*

### **Step 5: Complete & Summary** ✅
**Component:** `CompleteStep.tsx`
**Status:** Complete (แก้ใหม่ 28 ส.ค.)
- ✅ สรุปตัวเลข **จากผลตอบกลับจริงของ `/api/import/commit`** ไม่ใช่นับจาก state ในเบราว์เซอร์
- ✅ ขึ้นสัญลักษณ์เตือนสีเหลืองพร้อมสาเหตุ ถ้าบันทึกไม่สำเร็จ (เดิมขึ้นว่าสำเร็จเสมอ)
- ✅ เตือนถ้ามีรายการที่ไม่มี embedding เพราะจะมองไม่เห็นในการตรวจของซ้ำครั้งหน้า
- ✅ ปุ่มทางลัดไปยังหน้าคลังสินค้า

---

## 🏗️ **Technical Architecture (New Pipeline)**

```
Client (Wizard UI - Next.js) 
    ↓ (1. Upload)
Next.js API
    ↓ (2. Batch Clean)
FastAPI (ThaiTextProcessor)
    ↓ (3. Batch Dedup)
FastAPI (/api/v1/match/batch)
    ↓ (4. Batch Categorize)
Supabase Edge Functions / FastAPI (Hybrid 384-dim)
    ↓ (5. Save)
Database (PostgreSQL)
```

## 🎯 **Next Steps**
- [x] **API Connection:** เชื่อมต่อ `DeduplicationStep` เข้ากับ Endpoint `/api/v1/match/import-dedup` จริง
- [x] **API Connection:** เชื่อมต่อ `CategorizationStep` เข้ากับ Hybrid Algorithm
- [x] **E2E Testing:** ทดสอบตั้งแต่ Step 1 ถึง 5 ด้วยไฟล์จริง ไม่ใช้ข้อมูลจำลอง (28 ส.ค. 2569)
- [x] **การบันทึกลงฐานข้อมูล:** `app/api/import/commit/route.ts`
- [x] ขั้นจัดหมวดยังทำงานกับทั้ง 405 รายการ ควรทำเฉพาะ 221 ตัวที่เป็นของใหม่ (แก้ 28 ส.ค. — กรอง `_bucket === 'new'` ใน `WizardTab.tsx` ก่อนส่งเข้า `CategorizationStep`)
- [x] "ราคาขาย" ที่ map ไว้ตอนขั้นที่ 1 ไม่เคยถึง DB เลย (`products.price` เป็น NULL 100% ทุกแถว) — แก้ 28 ส.ค. ดูรายละเอียดที่ `docs/CURRENT_STATUS.md` หัวข้อ "ราคาขาย: เก็บถึง DB จริง + ใช้ช่วยตรวจของซ้ำ"
- [ ] การจับคู่คอลัมน์ราคาเลือก `ราคา/หน่วย` แทน `ราคาขาย` ต้องเลือกเองในหน้าจอ
- [x] **บั๊กที่เจอ 28 ส.ค.:** `/api/v1/match/import-dedup` ดึงสินค้าฝั่งสต๊อกไม่มี `ORDER BY` และโดน PostgREST cap ที่ 1,000 แถว ทั้งที่สต๊อกมี 3,103 แถว — แก้แล้วด้วย `fetch_all_approved_products()` (แบ่งหน้า + `.order("id")`) ดูรายละเอียดที่ `docs/CURRENT_STATUS.md`

## 🐛 บั๊กที่พบตอนทดสอบ E2E จริง (แก้แล้วทั้งหมด)
1. `ColumnMappingStep` parse ด้วย `maxRows: 10` เพื่อพรีวิว แต่ส่ง object เดิมไปใช้เป็นข้อมูลจริง → ไฟล์ 405 รายการถูกประมวลผลแค่ 10 (หน้าจอยังโชว์ 405 เพราะอ่านจาก `totalCount` คนละฟิลด์กับ `rows`)
2. `DeduplicationStep.onComplete` ส่ง `cleanedData` ดิบกลับ ทิ้งผลแบ่งกลุ่ม 37/147/221 ทั้งหมด
3. `ProductMatchResult` ไม่คืน id ของสินค้าในคลัง (ฟิลด์ `id` เป็นเลขลำดับรีวิว `review_1`) เขียน FK ไม่ได้
4. ทั้ง `/api/import/approve` และ `/api/import/pending` ไม่ใส่ `embedding` สินค้าที่เพิ่มผ่าน UI จึงมองไม่เห็นในการสแกนครั้งหน้า
5. `CompleteStep` ขึ้นว่าบันทึกสำเร็จทุกครั้งทั้งที่ไม่มีโค้ดเขียน DB เลย

---

## 📊 ผลทดสอบจริง (ไฟล์ `input/new_product/POS_เพิ่มสินค้า_*.csv` 405 รายการ)

| กลุ่ม | จำนวน | สถานะที่บันทึก | เห็นได้ที่ |
|---|---:|---|---|
| มีในสตอกแล้ว (>95%) | 37 | `rejected` + คู่ใน `similarity_matches` | ไม่ปนสตอก |
| ก้ำกึ่ง (80-94%) | 147 | `pending_review_dedup` | wizard + Verify ด่าน 1 |
| ของใหม่ (<80%) | 221 | `pending_review_category` | wizard + Verify ด่าน 2 |

บันทึกตั้งแต่จบขั้นที่ 3 ไม่รอขั้นสุดท้าย เพื่อให้ปิดเบราว์เซอร์กลางคันแล้วไปทำต่อที่หน้า Verify ได้ ทุกรายการมี embedding ครบ
