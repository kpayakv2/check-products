# 📊 Thai Product Taxonomy & Synonyms Dataset

## 🎯 ภาพรวม Dataset

Dataset นี้สร้างจากการวิเคราะห์ไฟล์ `รายการสินค้าพร้อมหมวดหมู่_AI.txt` ที่มีสินค้า **3,105 รายการ** จากร้านค้าจริง

### 📈 สถิติ Dataset:
- **หมวดหมู่หลัก**: 12 หมวด
- **หมวดหมู่ย่อย**: 47 หมวด  
- **Synonym Lemmas**: 26 กลุ่มคำ
- **Synonym Terms**: 180+ คำ
- **Keyword Rules**: 25 กฎ
- **Regex Rules**: 10 รูปแบบ

## 🗂️ โครงสร้าง Taxonomy

### 📋 หมวดหมู่หลัก (Level 0):

1. **เครื่องมือ_ฮาร์ดแวร์** (Hardware & Tools)
   - อุปกรณ์ทำสวน, เครื่องมือช่าง, สีและอุปกรณ์ทาสี
   - อุปกรณ์ประตูและกุญแจ, อุปกรณ์ไฟฟ้า, อุปกรณ์ประปา, วัสดุขัดผิว

2. **ของใช้ในบ้าน** (Household Items)
   - ขันน้ำ, กระติก, กล่อง/ที่เก็บของ, ตะกร้า/กระจาด
   - ตะแกรง/ชั้นวางของ, ถังน้ำ/ถังเอนกประสงค์, กะละมัง
   - กระจก, กระบอก_หัวฉีด_ขวด, ไม้แขวนเสื้อ_อุปกรณ์ตากผ้า
   - อุปกรณ์จัดเก็บ, เก้าอี_โต้ะ, ที่นอน/ฟูก

3. **เครื่องครัว** (Kitchen & Cooking)
   - ภาชนะใส่เครื่องปรุง/ขวดซอส, อุปกรณ์ตวง, เขียง
   - ช้อน/ส้อม/อุปกรณ์รับประทานอาหาร, อุปกรณ์เสิร์ฟ/อุปกรณ์หนีบอาหาร
   - หม้อ/ภาชนะหุงต้ม, มีดทำครัว, ถาดรองอาหาร, แก้วน้ำ/ถ้วย
   - กระชอน/ที่กรอง, อุปกรณ์ลับมีด, กระทะ, กรวย
   - อุปกรณ์ประกอบอาหาร/ตะแกรงครัว

4. **อุปกรณ์ทำความสะอาด** (Cleaning Supplies)
   - ฟองน้ำ/ใยขัด/ฝอยขัด, แปรงขัดต่าง ๆ
   - ผ้าถูพื้น/ผ้าเอนกประสงค์, ถุงขยะ/ถังขยะ/เก็บขยะ

5. **เครื่องเขียน_สำนักงาน** (Stationery & Office)
   - กรรไกร/มีดคัตเตอร์, กาว/เทป, สมุด/กระดาษ

6. **ของเล่น_นันทนาการ** (Toys & Recreation)
   - รถของเล่น, ปืนของเล่น, ของเล่นกิจกรรม

7. **ผลิตภัณฑ์ดูแลส่วนบุคคล** (Personal Care)
   - ผลิตภัณฑ์ระงับกลิ่นกาย, ผลิตภัณฑ์อื่น ๆ สำหรับส่วนบุคคล
   - สบู่เหลว/ครีมอาบน้ำ

8. **ผลิตภัณฑ์ทำความสะอาดในบ้าน** (Household Cleaning)
9. **สินค้าเพื่อสัตว์เลี้ยง** (Pet Supplies)
10. **เครื่องใช้ไฟฟ้า** (Electrical Appliances)
11. **แม่และเด็ก** (Mother & Baby)
12. **เบ็ดเตล็ด** (Miscellaneous)

## 🔤 ระบบ Synonyms

### 📝 โครงสร้าง Lemma + Terms:

**Lemma** = กลุ่มคำหลัก (Main concept)
**Terms** = คำย่อยในกลุ่ม (Related terms)

#### ตัวอย่าง:
```sql
-- Lemma: เครื่องมือทำสวน
Terms: คราด, คราดมือเสือ, เสียม, เสียมมิด, สายยาง, กระถาง

-- Lemma: ภาชนะใส่น้ำ  
Terms: ขัน, ขันน้ำ, ขันปั๊ม, กระติก, กระติกเหลี่ยม, กระบอกน้ำ

-- Lemma: อุปกรณ์ขัดล้าง
Terms: ฟองน้ำ, ฟองน้ำตาข่าย, ใยขัด, ฝอยขัด, ไบร์ท
```

### 🎯 Confidence Scores:
- **Primary Terms**: 0.90-0.95 (คำหลัก)
- **Secondary Terms**: 0.80-0.89 (คำรอง)
- **Variant Terms**: 0.70-0.79 (คำแปรผัน)

## 🤖 AI Rules System

### 🔍 Keyword Rules (25 กฎ):

#### High Priority Rules (Priority 9-10):
- **Garden Tools**: คราด, เสียม, สายยาง, กระถาง
- **Water Containers**: ขัน, กระติก, กระบอกน้ำ
- **Sauce Bottles**: ขวดซอส, ซอส, เครื่องปรุง
- **Sponges**: ฟองน้ำ, ใยขัด, ฝอยขัด

#### Medium Priority Rules (Priority 7-8):
- **Kitchen Knives**: มีด, สับ, เหล็กกล้า
- **Cleaning Brushes**: แปรง, ซักผ้า, ห้องน้ำ
- **Toy Cars**: รถ, จ้าว, ทะเลทราย

### 🎭 Regex Rules (10 รูปแบบ):

#### Pattern Examples:
```regex
# ขนาดและมิติ
\b\d+(\.\d+)?\s*(ซม|cm|นิ้ว|ลิตร|L|มล|ml)\b

# หมายเลขรุ่น  
\b(NO\.|#|เบอร์)\s*[A-Z0-9\-]+\b

# สี
\b(สี)?(แดง|เขียว|น้ำเงิน|เหลือง|ขาว|ดำ|หวาน|สด|ใส)\b

# แบรนด์
\b(ALLWAYS|KIWI|TOA|DORCO|ATM|NCL|SMT|SRT)\b
```

## 📥 การติดตั้ง Dataset

### 1. เตรียม Database:
```sql
-- รัน schema หลักก่อน
\i supabase/schema.sql

-- จากนั้นรัน dataset
\i data/taxonomy_dataset.sql
\i data/synonyms_dataset.sql  
\i data/rules_dataset.sql
```

### 2. ตรวจสอบการติดตั้ง:
```sql
-- ตรวจสอบจำนวน taxonomy nodes
SELECT level, COUNT(*) FROM taxonomy_nodes GROUP BY level ORDER BY level;

-- ตรวจสอบ synonyms
SELECT COUNT(*) as lemmas FROM synonym_lemmas;
SELECT COUNT(*) as terms FROM synonym_terms;

-- ตรวจสอบ rules
SELECT COUNT(*) as keyword_rules FROM keyword_rules WHERE is_active = true;
SELECT COUNT(*) as regex_rules FROM regex_rules WHERE is_active = true;
```

## 🔧 การใช้งาน API

### 1. Category Suggestions:
```typescript
const suggestions = await EdgeFunctionAPI.getCategorySuggestions({
  text: "คราดมือเสือ 023 NCL",
  options: {
    maxSuggestions: 5,
    minConfidence: 0.3,
    includeExplanation: true
  }
})
```

### 2. Hybrid Search:
```typescript
const results = await EdgeFunctionAPI.hybridSearch({
  query: "ขวดซอสแฟนซี",
  type: 'hybrid',
  filters: {
    categories: ['cat_003_001']
  }
})
```

### 3. Synonym Matching:
```typescript
// ค้นหา synonyms ของคำ
const synonyms = await DatabaseService.getSynonymsByTerm("คราด")

// ค้นหา lemma ที่เกี่ยวข้อง
const lemmas = await DatabaseService.getLemmasByCategory("cat_001_001")
```

## 📊 สถิติการใช้งาน

### 🏆 Top Categories (จำนวนสินค้า):
1. **ของใช้ในบ้าน**: ~1,200 รายการ (38%)
2. **เครื่องครัว**: ~800 รายการ (26%)
3. **เครื่องมือ_ฮาร์ดแวร์**: ~600 รายการ (19%)
4. **อุปกรณ์ทำความสะอาด**: ~300 รายการ (10%)
5. **อื่นๆ**: ~205 รายการ (7%)

### 🎯 Keyword Coverage:
- **High Coverage** (>50 สินค้า): กล่อง, ตะกร้า, ถัง, มีด
- **Medium Coverage** (20-50 สินค้า): แปรง, ขัน, เขียง, ฟองน้ำ
- **Low Coverage** (<20 สินค้า): ของเล่น, สัตว์เลี้ยง, เครื่องใช้ไฟฟ้า

## 🚀 การปรับปรุง Dataset

### 1. เพิ่ม Synonyms ใหม่:
```sql
-- เพิ่ม lemma ใหม่
INSERT INTO synonym_lemmas (name_th, name_en, category_id) 
VALUES ('กลุ่มคำใหม่', 'New Term Group', 'cat_xxx');

-- เพิ่ม terms
INSERT INTO synonym_terms (lemma_id, term, confidence_score) 
VALUES ('lemma_xxx', 'คำใหม่', 0.85);
```

### 2. เพิ่ม Rules ใหม่:
```sql
-- เพิ่ม keyword rule
INSERT INTO keyword_rules (name, keywords, category_id, confidence_score) 
VALUES ('New Rule', ARRAY['คำ1', 'คำ2'], 'cat_xxx', 0.80);

-- เพิ่ม regex rule  
INSERT INTO regex_rules (name, pattern, category_id, confidence_score)
VALUES ('New Pattern', '\b(pattern)\b', 'cat_xxx', 0.75);
```

### 3. ปรับปรุง Confidence Scores:
```sql
-- อัปเดตคะแนนตาม performance
UPDATE synonym_terms 
SET confidence_score = 0.90 
WHERE term = 'คำที่ต้องการปรับ';

UPDATE keyword_rules 
SET confidence_score = 0.85 
WHERE name = 'Rule ที่ต้องการปรับ';
```

## 🎯 Best Practices

### 1. การตั้งชื่อ Categories:
- ใช้ภาษาไทยเป็นหลัก
- เพิ่มภาษาอังกฤษเป็น secondary
- ใช้ underscore สำหรับ code

### 2. การจัดการ Synonyms:
- จัดกลุ่มตาม semantic meaning
- ใช้ primary term เป็นตัวแทนกลุ่ม
- ปรับ confidence score ตาม usage

### 3. การเขียน Rules:
- เริ่มจาก high-confidence keywords
- ใช้ regex สำหรับ patterns ที่ซับซ้อน
- ทดสอบ rules กับข้อมูลจริง

### 4. การ Monitor Performance:
- ติดตาม accuracy ของ suggestions
- วิเคราะห์ false positives/negatives
- ปรับปรุง rules ตาม feedback

---

## 📞 การสนับสนุน

หากต้องการความช่วยเหลือ:
1. ตรวจสอบ logs ใน Supabase Dashboard
2. ทดสอบ rules ในหน้า Settings
3. ใช้ JSON Editor สำหรับการแก้ไขขั้นสูง

**Dataset Version**: 1.0  
**Last Updated**: 2024-01-21  
**Total Products Analyzed**: 3,105 รายการ
