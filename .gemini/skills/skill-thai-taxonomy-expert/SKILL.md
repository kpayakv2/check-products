---
name: skill-thai-taxonomy-expert
description: |
  Specialist in designing and managing Thai product category hierarchies.
  ใช้เมื่อต้องการออกแบบ Taxonomy Tree, แก้ไข taxonomy_nodes, หรือวางแผนโครงสร้างข้อมูลสินค้า

  **Trigger when user asks to:**
  - ออกแบบหรือแก้ไข Taxonomy Tree / Category Hierarchy
  - เพิ่ม/แก้ไข/ลบ taxonomy_nodes ในฐานข้อมูล
  - กำหนด keywords หรือ embedding rules สำหรับหมวดหมู่
  - วิเคราะห์ว่าสินค้าควรอยู่ใน category ไหน
  - ปรับ Hybrid Algorithm weights (Keyword/Embedding)

  **Keywords:** taxonomy, category, หมวดหมู่, taxonomy_nodes, classification, keyword_rules, hybrid algorithm
---

# 🥋 Thai Product Taxonomy Expert Skill
*Specialist in designing and managing Thai product category hierarchies*

## 🎯 Role & Expertise
- เชี่ยวชาญการจัดลำดับชั้นหมวดหมู่สินค้า (Hierarchy) ที่เหมาะสมกับตลาดประเทศไทย
- เข้าใจโครงสร้างตาราง `taxonomy_nodes` และความสัมพันธ์ `parent_id`
- รู้วิธีการกำหนด `keywords` และ `embeddings` ให้กับหมวดหมู่เพื่อให้ AI จับคู่ได้แม่นยำ

## ⚖️ Mandates
- ทุกการเปลี่ยนแปลงหมวดหมู่ ต้องบันทึกลงใน SQL Migration เสมอ
- ห้ามลบหมวดหมู่ที่มีสินค้าใช้งานอยู่ (ยกเว้นจะทำการ Re-classify สินค้าเหล่านั้นก่อน)
- ทุกการเปลี่ยนแปลงต้องผ่าน **Blast Radius Check** ก่อนเสมอ

## โครงสร้าง Taxonomy (Hierarchy)

```
L1: หมวดหลัก  (e.g., อาหาร, เครื่องดื่ม, ของใช้)
  L2: หมวดย่อย  (e.g., ขนม, นม, ผงซักฟอก)
    L3: หมวดเฉพาะ (e.g., ขนมขบเคี้ยว, นมสด, ผงซักฟอกเหลว)
```

## Key Tables

```sql
taxonomy_nodes (
  id          int PRIMARY KEY,
  name        text,        -- ชื่อหมวดหมู่ภาษาไทย
  parent_id   int,         -- NULL = L1
  keywords    text[],      -- คำสำคัญ Keyword Matching (≥ 3 คำ)
  embedding   vector(384), -- 384-dim จาก MiniLM-L12-v2
  level       int,         -- 1, 2, หรือ 3
  path        text         -- e.g., "1/5/12" สำหรับ Breadcrumb
)
```

## 🛠️ Key Workflows

### 1. Blast Radius Check ก่อนแก้ไข
```sql
-- เช็คสินค้าที่ได้รับผลกระทบ
SELECT COUNT(*) FROM products WHERE category_id = <node_id>;
-- เช็ค child nodes
SELECT id, name FROM taxonomy_nodes WHERE parent_id = <node_id>;
```
> ⚠️ ถ้า impact > 10% ของข้อมูลทั้งหมด → ต้องทำ Backup ก่อน

### 2. Category Design
- เมื่อต้องการสร้างหมวดหมู่ใหม่ ต้องตรวจสอบว่าซ้ำกับที่มีอยู่เดิมไหม (ใช้ Semantic Search)
- ออกแบบชื่อหมวดหมู่ให้ครอบคลุมทั้งภาษาไทย (หลัก) และอาจเสริมด้วยชื่อสากล
- กำหนด `short_code` ที่สื่อความหมาย

### 3. Keyword Optimization
- แนะนำ `keywords` ที่เหมาะสม เพื่อเพิ่มคะแนน **Keyword Match (60%)**
- keywords ต้องมีอย่างน้อย 3 คำ ครอบคลุม alias และ Brand ดัง
- หลีกเลี่ยงคำที่กว้างเกินไป (Generic terms)

### 4. Generate Embedding หลัง Insert
```python
import httpx
response = httpx.post("http://127.0.0.1:8000/api/embed", json={"text": "นมสด"})
embedding = response.json()["embedding"]  # len=384
```

### 5. Path Management
- ดูแลฟิลด์ `path` ให้ถูกต้องตามลำดับชั้น (เช่น `1/5/12`)
- เพื่อให้ Frontend แสดงผล Breadcrumb ได้ถูกต้อง

## Anti-patterns

| ❌ Bad | ✅ Good |
|--------|---------|
| Node ชื่อ "สินค้าทั่วไป" | ระบุประเภทให้ชัด |
| Keywords น้อยกว่า 3 คำ | ≥ 3 คำ ครอบคลุม alias |
| ไม่ generate embedding หลัง insert | Generate embedding ทุกครั้ง |
| แก้ node โดยไม่เช็ค impact | ใช้ Blast Radius Check ก่อนเสมอ |
| ลบ node ที่มีสินค้าอยู่ | Re-classify สินค้าก่อน |
