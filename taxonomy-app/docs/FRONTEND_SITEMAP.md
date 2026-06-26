# 🗺️ แผนที่หน้าเว็บ (Frontend Sitemap / UI Directory)

เอกสารนี้รวบรวม **หน้าจอทั้งหมด (Pages)** และ **ที่อยู่ไฟล์โค้ด (Routes)** ในโปรเจกต์ `taxonomy-app` เพื่อให้ทีมงานและผู้พัฒนาสามารถไล่หาโค้ดและทำความเข้าใจโครงสร้างของระบบได้อย่างรวดเร็ว

**อัปเดตล่าสุด:** พฤษภาคม 2569

---

## 📱 1. ระบบนำเข้าและประมวลผล (Import & Processing)

| URL Path | โฟลเดอร์ในโค้ด (`app/...`) | หน้าที่การทำงาน (Description) | สถานะ (Status) |
| :--- | :--- | :--- | :--- |
| `/import/wizard` | `app/import/wizard/page.tsx` | **ระบบนำเข้าอัจฉริยะ (Magic 5-Step Wizard)**: อัปโหลด, ล้างข้อมูล, เช็คซ้ำ, จัดหมวดหมู่ | 🟡 รอเชื่อม Backend |
| `/import` | `app/import/page.tsx` | หน้าจอหลักของการนำเข้า (รองรับ Storage Mode ดึงไฟล์เก่า) | 🟢 ใช้งานได้ |
| `/import/pending` | `app/import/pending/page.tsx` | หน้ารีวิวรายการสินค้าที่รออนุมัติหมวดหมู่ (แบบเก่า) | 🟢 ใช้งานได้ (อาจจะถูกแทนที่ด้วย Wizard) |
| `/verify` | `app/verify/page.tsx` | ระบบตรวจสอบความถูกต้องของข้อมูลหลังการนำเข้า | 🟢 ใช้งานได้ |
| `/deduplication` | `app/deduplication/page.tsx` | เครื่องมือ **เช็คของซ้ำเฉพาะกิจ** นอกกระบวนการนำเข้า | 🟢 ใช้งานได้ |

---

## 🗃️ 2. ระบบจัดการฐานข้อมูล (Data Management)

| URL Path | โฟลเดอร์ในโค้ด (`app/...`) | หน้าที่การทำงาน (Description) | สถานะ (Status) |
| :--- | :--- | :--- | :--- |
| `/products` | `app/products/page.tsx` | **คลังสินค้า (Product Database)**: ดูและค้นหาสินค้าทั้งหมดในระบบ | 🟢 ใช้งานได้ |
| `/taxonomy` | `app/taxonomy/page.tsx` | **โครงสร้างหมวดหมู่**: เพิ่ม/ลด/แก้ไข โครงสร้างต้นไม้ (Taxonomy Tree) | 🟢 ใช้งานได้ |
| `/synonyms` | `app/synonyms/page.tsx` | **คลังคำพ้องความหมาย**: ดิกชันนารีสอน AI ว่าคำไหนแปลว่าอะไร | 🟢 ใช้งานได้ |

---

## 🤖 3. ระบบวิเคราะห์และ AI (Analytics & AI System)

| URL Path | โฟลเดอร์ในโค้ด (`app/...`) | หน้าที่การทำงาน (Description) | สถานะ (Status) |
| :--- | :--- | :--- | :--- |
| `/auto-learn` | `app/auto-learn/page.tsx` | **AI Feedback Loop**: หน้าจอให้ AI เรียนรู้คำศัพท์ใหม่ๆ อัตโนมัติจากการใช้งานของมนุษย์ | 🟢 ใช้งานได้ |
| `/reports` | `app/reports/page.tsx` | **Dashboard & Reports**: หน้าสรุปสถิติและรายงานการใช้งานระบบ | 🟢 ใช้งานได้ |
| `/settings` | `app/settings/page.tsx` | **System Configuration**: ตั้งค่าการทำงานของระบบ | 🟢 ใช้งานได้ |

---

## 🧩 4. โครงสร้างชิ้นส่วนหลัก (Key Components)
*ไฟล์เหล่านี้อยู่ในโฟลเดอร์ `components/` และถูกนำไปประกอบในหน้าเว็บต่างๆ ด้านบน*

*   **`components/Import/`** : ชิ้นส่วนสำหรับทำหน้าต่างนำเข้า (Wizard) เช่น `DataCleaningStep`, `CategorizationStep`
*   **`components/Taxonomy/`** : ชิ้นส่วนกราฟิกต้นไม้ เช่น `EnhancedTaxonomyTree`
*   **`components/Search/`** : ชิ้นส่วนกล่องค้นหาแบบ Hybrid (Keyword + AI) เช่น `HybridSearch`
*   **`components/Product/`** : ชิ้นส่วนสำหรับรีวิวสินค้าแต่ละชิ้น เช่น `EnhancedProductReview`
*   **`components/Layout/`** : ชิ้นส่วนโครงเว็บ (แถบเมนูข้าง, แถบด้านบน) เช่น `Sidebar`, `Header`

---
*💡 **ทิปส์:** หากต้องการแก้ UI หน้าไหน ให้หา URL ในตารางนี้ แล้วไปที่โฟลเดอร์โค้ดนั้นได้เลย!*
