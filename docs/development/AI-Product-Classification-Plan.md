# แผนปฏิบัติการ: AI จัดหมวดหมู่สินค้า (สำหรับขั้นตอน “เพิ่มสินค้าใหม่”)
อัปเดต: 2025-09-20

> เป้าหมาย: เมื่อเพิ่มสินค้าใหม่ใน POS ให้ระบบ **ใส่หมวดอัตโนมัติ** (ถ้าไม่ชัวร์ให้เสนอ Top-3) พร้อมเก็บร่องรอยตรวจสอบได้

---

## 0) ภาพรวมโครงการ (Executive Summary)
- **Use case หลัก:** เพิ่มสินค้าใหม่ → AI จัดหมวด → คนตรวจเฉพาะเคสไม่ชัวร์
- **KPI PoC 1–2 เดือน:**  
  - Auto-assign ≥ **70%**  
  - Accuracy (sampling) ≥ **90%**  
  - Review queue ≤ **25%**, Unmapped < **5%**
- **สcopeเริ่ม:** โฟกัส 3–5 หมวดขายดี เพื่อปล่อยใช้เร็วแบบ vibecode

---

## 1) โปรไฟล์ข้อมูลจากไฟล์ที่อัปโหลด (ระบบจริง)
แหล่ง: `รายการสินค้าพร้อมหมวดหมู่_AI.txt`  
- เข้ารหัส: **UTF-16**
- จำนวนสินค้า (แถว): **3,103**
- หมวดหลัก (Main): **16**
- หมวดย่อย (Sub): **116**
- ซับเดียวกันอยู่หลายเมน (conflict): **0**
- ชื่อซับสะกดต่างแต่ความหมายเดียวกัน (near-duplicate): **1** กลุ่ม

> ข้อแนะนำทันที: ล็อก Taxonomy v1 จากชุดนี้ (16 Main / 116 Sub) เป็น “รายการหมวดที่อนุญาต” จากนั้นรวมชื่อซับที่ซ้ำ/เขียนต่างกันให้เหลือชื่อเดียว พร้อมเพิ่ม synonyms ให้ครบ

---

## 2) Roadmap แบบ “เริ่มเร็ว ใช้ได้จริง” (Now → Next → Later)
**Now (สัปดาห์ 1–2)**  
- Freeze `Taxonomy v1` จากไฟล์จริง (16 Main / 116 Sub)  
- เคลียร์ near-duplicate ให้เหลือชื่อเดียว  
- ทำ `synonyms.csv` ชุดแรก 50–200 คำ (แบรนด์/คำบ้าน ๆ)  
- ตั้ง threshold: auto ≥ 0.90 / review 0.60–0.89 / unmapped < 0.60

**Next (สัปดาห์ 3–4)**  
- เปิด API /classify (rule + embedding)  
- เชื่อมหน้า “เพิ่มสินค้าใหม่” ให้เรียก /classify แบบเรียลไทม์  
- เปิด Review Queue ให้พนักงานยืนยันเฉพาะเคสไม่ชัวร์

**Later (สัปดาห์ 5+)**  
- รายงานประจำสัปดาห์: %Auto / %Review / %Unmapped + “คู่หมวดสับสน”  
- เติม synonyms ต่อเนื่อง, ปรับ threshold, วางเวอร์ชันนิ่ง Taxonomy/Synonyms

---

## 3) เช็กลิสต์สิ่งที่ต้องทำ (พร้อม DoD)
### 0) Kickoff
- [ ] ตั้ง KPI (ด้านบน) + ทีม Owner/DE/สินค้า/Backend  
**DoD:** ขอบเขต & เวลา & ผู้รับผิดชอบชัดเจน

### 1) รวบรวม & เตรียมข้อมูล
- [ ] Export products.csv (รหัส, ชื่อ, หมวด, แบรนด์?, ขนาด?, บาร์โค้ด?)  
- [ ] รวมคำค้น POS/เว็บ/แชท + top sellers + catalog ผู้ขาย  
**DoD:** ครอบคลุม ≥95% ของ SKU active, ไฟล์ UTF-8 หรือ UTF-16 พร้อมระบุ encoding

### 2) ออกแบบมาตรฐานการตั้งชื่อ
- [ ] นิยาม Synonym vs Attribute vs Brand  
- [ ] Canonical naming ไทย/อังกฤษ + กฎการตั้งชื่อ (หน่วย/สัญลักษณ์)  
**DoD:** มีตัวอย่างถูก–ผิดอย่างน้อย 10 เคส (ใน governance.md)

### 3) Taxonomy v1
- [ ] สร้าง taxonomy_nodes.csv (Main/Sub + code + description)  
**DoD:** ไม่มีชื่อซ้ำ, ทุก node มี parent, ครอบคลุม ≥98%

### 4) Synonyms v1
- [ ] สกัดคำพ้องจากชื่อสินค้า/queries/vendor catalog → synonyms.csv  
**DoD:** sub แต่ละตัวมี ≥5 คำขึ้นไป, ครอบคลุมแบรนด์ท็อป

### 5) Normalization
- [ ] ทำความสะอาดชื่อ, แยกหน่วย/ขนาดไป attributes  
**DoD:** หน่วย normalized ≥95%

### 6) AI Assist
- [ ] Embedding multilingual + dictionary match → candidate  
- [ ] Closed-set classification + confidence → decision  
**DoD:** Top-1 ≥85% บน dev set 1,000 SKU, P95 latency < 1s

### 7) Human-in-the-loop
- [ ] สร้าง Review Queue (Top-3 candidates ให้คลิกเลือก)  
**DoD:** เคสค้าง review < 300, agreement ≥ 0.9

### 8) Integration
- [ ] เชื่อม /classify กับหน้าฟอร์มเพิ่มสินค้า + logging  
**DoD:** เพิ่มสินค้าแล้วระบบใส่หมวดอัตโนมัติ (หรือโชว์ Top-3)

### 9) ทดสอบ & ยอมรับ
- [ ] วัด precision/recall, UAT ฝ่ายสินค้า/หน้าร้าน  
**DoD:** ผ่าน KPI, มี rollback plan

### 10) เฝ้าระวัง & ปรับปรุง
- [ ] Dashboard & weekly report, refresh embeddings รายเดือน  
**DoD:** มีสรุปสัปดาห์ + เวอร์ชันนิ่ง TAXO/SYN เดินหน้า

---

## 4) สคีม่าไฟล์มาตรฐาน (Template)
### 4.1 taxonomy_nodes.csv
```
node_id,code,name,parent_id,level,description,active,version
1000,1000,เครื่องดื่ม,,main,"หมวดเครื่องดื่มทั้งหมด",true,TAXO_v1
1100,1100,น้ำอัดลม,1000,sub,"โซดาหวาน/โคล่า/น้ำสี ฯลฯ",true,TAXO_v1
...
```

### 4.2 synonyms.csv
```
keyword,canonical,type,context_node_id,version
โค้ก,น้ำอัดลม,alias,1100,SYN_v1
coke,น้ำอัดลม,alias,1100,SYN_v1
downy,น้ำยาปรับผ้านุ่ม,brand,,SYN_v1
uht,นมยูเอชที,alias,,SYN_v1
...
```

### 4.3 taxonomy_mapping.csv (ผลลัพธ์จาก AI/คน)
```
product_id,node_id,confidence,method,status,reviewed_by,reviewed_at,model_version,taxonomy_version
SKU123,1100,0.93,embedding,auto,,,clf_v1.2,TAXO_v1
SKU456,1210,0.68,llm,needs_review,kan,2025-09-20,clf_v1.2,TAXO_v1
...
```

---

## 5) การทำงานตอน “เพิ่มสินค้าใหม่”
1) ฟอร์ม POS ส่ง ข้อมูลไปที่ /classify : product_name, brand?, size?, barcode?  
2) ระบบคืนค่า: node_id, node_name, confidence, top3  
3) ตัดสินใจ:  
   - ≥ 0.90 → ใส่หมวดให้อัตโนมัติ  
   - 0.60–0.89 → แสดง Top-3 ให้คลิกเลือก  
   - < 0.60 → ใส่คิว review  
4) บันทึกผลลง taxonomy_mapping.csv + log

---

## 6) Governance & Versioning
- เวอร์ชัน: TAXO_v1, SYN_v1 → อัปเกรดเป็น v1.1, v1.2 เมื่อมีการปรับ  
- ทุกการเปลี่ยนแปลงต้องอธิบายใน governance.md และทำ diff

---

## 7) ความเสี่ยงที่ต้องระวัง + วิธีแก้
- ชื่อสินค้าไม่บอกชนิด → เติม synonyms ว่าเป็นชนิดใด (เช่น ดาวนี่ → น้ำยาปรับผ้านุ่ม)  
- หมวดย่อยซ้ำ/ใกล้กัน → รวมภายใต้กลุ่มใหญ่ (“อุปกรณ์ขัดถู”) แล้วค่อยแตกละเอียด  
- ไทย-อังกฤษ/สะกดหลากหลาย → ใช้ normalization + synonyms + ตัวอย่างสอนโมเดล

---

## 8) โมดูล (1 โมดูล = 1 หน้า)
**MVP 6 โมดูล:**  
1) เพิ่มสินค้าใหม่  2) Review Queue  3) Taxonomy Manager  
4) Synonyms Manager  5) Config & Threshold  6) Reports & Audit

---

## 9) รายการส่งมอบ
- taxonomy_nodes.csv  
- synonyms.csv  
- taxonomy_mapping.csv  
- spelling_dict.txt  
- attributes_schema.json  
- governance.md  
- evaluation_report.md  
