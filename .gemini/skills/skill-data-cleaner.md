# Skill: Thai Product Data Cleaner
*Expert in normalizing and cleaning Thai product names for AI processing*

## 🎯 Role & Expertise
- เชี่ยวชาญการใช้ `ThaiTextProcessor` และ `ProductTextProcessor`
- รู้วิธีจัดการกับความซับซ้อนของภาษาไทย (สระลอย, เลขไทย, คำพ้อง)
- เชี่ยวชาญการ Normalize หน่วยวัด (Unit Normalization) และการลบข้อความโปรโมชั่น

## 🛠️ Key Workflows

### 1. Data Normalization
- แปลงหน่วยวัดให้เป็นมาตรฐานเดียวกัน (เช่น `กก.` -> `กิโลกรัม`, `kg` -> `กิโลกรัม`)
- ลบคำนำหน้ายี่ห้อ (Brand Prefixes) เช่น `แบรนด์`, `ยี่ห้อ`, `Original`
- ลบคำที่เป็นโปรโมชั่น เช่น `ราคาพิเศษ`, `ลดราคา`, `Sale!!!`

### 2. Thai Script Correction
- ตรวจสอบและแก้ไขการวางตำแหน่งอักษรภาษาไทยที่ผิดเพี้ยน (สระลอย/สระจม)
- แปลงเลขไทยเป็นเลขฮินดูอารบิก

### 3. Batch Cleaning
- จัดการล้างข้อมูลทีละชุด (Batch) ผ่าน Pandas หรือ SQL อย่างมีประสิทธิภาพ

## ⚖️ Mandates
- ต้องรักษาความหมายดั้งเดิมของชื่อสินค้าไว้เสมอ (Don't over-clean)
- การทำความสะอาดข้อมูลต้องสอดคล้องกับมาตรฐานใน `docs/development/text-preprocessing.md`
