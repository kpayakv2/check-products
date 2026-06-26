---
name: thai-taxonomy-expert
description: |
  ทักษะสำหรับออกแบบและจัดการระบบหมวดหมู่สินค้าไทย (Thai Product Taxonomy)
  ใช้เมื่อต้องการออกแบบ Taxonomy Tree, แก้ไข taxonomy_nodes, หรือวางแผนโครงสร้างข้อมูลสินค้า

  **Trigger when user asks to:**
  - ออกแบบหรือแก้ไข Taxonomy Tree / Category Hierarchy
  - เพิ่ม/แก้ไข/ลบ taxonomy_nodes ในฐานข้อมูล
  - กำหนด keywords หรือ embedding rules สำหรับหมวดหมู่
  - วิเคราะห์ว่าสินค้าควรอยู่ใน category ไหน
  - ปรับ Hybrid Algorithm weights (Keyword/Embedding)

  **Keywords:** taxonomy, category, หมวดหมู่, taxonomy_nodes, classification, keyword_rules, hybrid algorithm
---

# 🥋 Thai Taxonomy Expert Skill

## ภาพรวมระบบ (System Overview)

โปรเจกต์นี้ใช้ **Hybrid Classification Algorithm**:
- **Keyword 60%** — ค้นจาก `keyword_rules` + `taxonomy_nodes.keywords` + `name_match`
- **Embedding 40%** — Cosine Distance (`<=>`) กับ `taxonomy_nodes.embedding` (384 dims)
- **Endpoint:** `POST /api/classify/category` บน FastAPI
- **Target Accuracy:** ≥ 72% F1-score

---

## โครงสร้าง Taxonomy (Hierarchy)

```
L1: หมวดหลัก (e.g., อาหาร, เครื่องดื่ม, ของใช้)
  L2: หมวดย่อย (e.g., ขนม, นม, ผงซักฟอก)
    L3: หมวดเฉพาะ (e.g., ขนมขบเคี้ยว, นมสด, ผงซักฟอกเหลว)
```

**กฎการออกแบบ Hierarchy:**
- แต่ละ node ต้องมี keywords ≥ 3 คำ
- หลีกเลี่ยง node ที่กว้างเกินไป (ambiguous) เช่น "อื่นๆ"
- ชื่อ node ควรเป็นภาษาไทยล้วน

---

## Key Tables

```sql
-- หมวดหมู่หลัก
taxonomy_nodes (
  id          int PRIMARY KEY,
  name        text,           -- ชื่อหมวดหมู่ภาษาไทย
  parent_id   int,            -- NULL = L1
  keywords    text[],         -- คำสำคัญสำหรับ Keyword Matching
  embedding   vector(384),    -- Embedding ของ node name
  level       int             -- 1, 2, หรือ 3
)

-- กฎ Keyword เพิ่มเติม
keyword_rules (
  id          int PRIMARY KEY,
  pattern     text,           -- Regex หรือ exact match
  category_id int REFERENCES taxonomy_nodes(id),
  priority    int             -- ยิ่งสูงยิ่งชนะ
)
```

---

## Workflow การเพิ่มหมวดหมู่ใหม่

### Step 1: ตรวจสอบ Parent Node
```sql
-- ดู hierarchy ปัจจุบัน
SELECT id, name, parent_id, level
FROM taxonomy_nodes
WHERE level <= 2
ORDER BY level, parent_id, id;
```

### Step 2: กำหนด Keywords
- ใช้ภาษาไทยล้วน (ไม่ต้องมีภาษาอังกฤษ เว้นแต่เป็น Brand)
- ครอบคลุมทั้ง: ชื่อสินค้า, ชื่อ Brand ดัง, หน่วยวัดทั่วไป
- ตัวอย่าง keywords สำหรับ "นมสด": `["นมสด", "นมโค", "นม UHT", "นมพาสเจอร์ไรส์", "โฟร์โมสต์", "ดัชมิลล์"]`

### Step 3: สร้าง Node + Generate Embedding
```sql
-- เพิ่ม node ใหม่ (embedding จะถูก generate ผ่าน FastAPI)
INSERT INTO taxonomy_nodes (name, parent_id, keywords, level)
VALUES ('นมสด', 5, ARRAY['นมสด', 'นมโค', 'นม UHT'], 3);
```

```python
# Generate embedding ผ่าน FastAPI local server
import httpx

response = httpx.post("http://127.0.0.1:8000/api/embed", json={"text": "นมสด"})
embedding = response.json()["embedding"]  # list[float], len=384
```

### Step 4: ทดสอบ Classification
```python
# ทดสอบว่า node ใหม่ทำงานได้จริง
response = httpx.post("http://127.0.0.1:8000/api/classify/category",
    json={"product_name": "นมสดโฟร์โมสต์ 1L"})
assert response.json()["category_id"] == <new_id>
```

---

## Anti-patterns ที่ต้องหลีกเลี่ยง

| ❌ Bad | ✅ Good |
|--------|---------|
| Node ชื่อ "สินค้าทั่วไป" | ระบุประเภทให้ชัด เช่น "ผงซักฟอก" |
| Keywords น้อยกว่า 3 คำ | ≥ 3 คำ ครอบคลุม alias |
| ไม่ generate embedding หลัง insert | Generate embedding ทุกครั้ง |
| แก้ node โดยไม่เช็ค impact | ใช้ `smart_impact_workflow.md` |

---

## Blast Radius Check ก่อนแก้ไข

```sql
-- เช็คว่า node นี้มีสินค้ากี่รายการที่ classify อยู่
SELECT COUNT(*)
FROM products p
WHERE p.category_id = <node_id>;

-- เช็ค child nodes
SELECT id, name FROM taxonomy_nodes WHERE parent_id = <node_id>;
```

> ⚠️ ถ้า impact > 10% ของข้อมูลทั้งหมด → ต้องทำ Backup ก่อน
