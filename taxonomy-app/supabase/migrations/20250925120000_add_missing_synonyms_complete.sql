-- Migration: add_missing_synonyms_complete
-- Purpose: Add all missing synonym lemmas and terms from original dataset
-- Fix category mapping issues and complete synonym system

BEGIN;

-- First, fix incorrect category mappings in existing lemmas
UPDATE synonym_lemmas SET category_id = (SELECT id FROM taxonomy_nodes WHERE code = 'cat_002_011') 
WHERE code = 'lemma_008'; -- Fix อุปกรณ์จัดระเบียบ: cat_002_008 -> cat_002_011

UPDATE synonym_lemmas SET category_id = (SELECT id FROM taxonomy_nodes WHERE code = 'cat_004_003') 
WHERE code = 'lemma_015'; -- Fix ผ้าทำความสะอาด: cat_004_004 -> cat_004_003

UPDATE synonym_lemmas SET category_id = (SELECT id FROM taxonomy_nodes WHERE code = 'cat_003_004') 
WHERE code = 'lemma_011'; -- Fix อุปกรณ์รับประทาน: cat_003_007 -> cat_003_004

-- Add usage_count column to synonym_terms if not exists
ALTER TABLE synonym_terms ADD COLUMN IF NOT EXISTS usage_count INTEGER DEFAULT 0;

-- Add missing synonym lemmas (lemma_016 to lemma_026)
INSERT INTO synonym_lemmas (code, name_th, name_en, description, category_id, is_verified) VALUES
('lemma_016', 'อุปกรณ์ตัด', 'Cutting Supplies', 'กรรไกรและอุปกรณ์ตัดต่างๆ', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_005_001'), true),
 
('lemma_017', 'อุปกรณ์ติด', 'Adhesive Supplies', 'กาวและเทปสำหรับติดและยึด', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_005_002'), true),
 
('lemma_018', 'ของเล่นยานพาหนะ', 'Vehicle Toys', 'รถและยานพาหนะของเล่น', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_006_001'), true),
 
('lemma_019', 'ของเล่นอาวุธ', 'Weapon Toys', 'ปืนและอาวุธของเล่น', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_006_002'), true),
 
('lemma_020', 'ผลิตภัณฑ์อาบน้ำ', 'Bathing Products', 'สบู่และผลิตภัณฑ์อาบน้ำ', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_007_003'), true),
 
('lemma_021', 'อุปกรณ์โกนหนวด', 'Shaving Supplies', 'อุปกรณ์สำหรับโกนหนวดและตกแต่ง', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_007_002'), true),
 
('lemma_022', 'อุปกรณ์สัตว์เลี้ยง', 'Pet Accessories', 'อุปกรณ์สำหรับสัตว์เลี้ยง', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_009_001'), true),
 
('lemma_023', 'อุปกรณ์ไฟฟ้าพกพา', 'Portable Electronics', 'อุปกรณ์ไฟฟ้าขนาดเล็กพกพา', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_010_001'), true),
 
('lemma_024', 'อุปกรณ์ให้นมเด็ก', 'Baby Feeding', 'อุปกรณ์สำหรับให้นมและดูแลเด็ก', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_011_001'), true),
 
('lemma_025', 'อุปกรณ์ความงาม', 'Beauty Accessories', 'อุปกรณ์ตกแต่งและความงาม', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_012_002'), true),
 
('lemma_026', 'วัสดุตกแต่ง', 'Decorative Materials', 'วัสดุสำหรับตกแต่งและประดับ', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_012_003'), true);

-- Add missing terms for existing lemmas (additional terms that were in original)
-- Use INSERT ... ON CONFLICT DO NOTHING to avoid duplicates
INSERT INTO synonym_terms (lemma_id, term, language, is_primary, confidence_score, usage_count) VALUES
-- เครื่องมือทำสวน (lemma_001) - เพิ่มเติม
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_001'), 'กรรไกรตัดหญ้า', 'th', false, 0.80, 1),

-- เครื่องมือช่าง (lemma_002) - เพิ่มเติม
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_002'), 'เอ็นแดง', 'th', false, 0.85, 4),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_002'), 'เอ็นเขียว', 'th', false, 0.85, 8),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_002'), 'ตลับเมตร', 'th', false, 0.80, 1),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_002'), 'ระดับน้ำ', 'th', false, 0.80, 3),

-- อุปกรณ์ทาสี (lemma_003) - เพิ่มเติม
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_003'), 'จานจับกระดาษทราย', 'th', false, 0.75, 1),

-- อุปกรณ์ไฟฟ้า (lemma_004) - เพิ่มเติม
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_004'), 'ปลั๊กเสียบ', 'th', false, 0.85, 2),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_004'), 'ไขควงวัดไฟ', 'th', false, 0.80, 1),

-- ภาชนะใส่น้ำ (lemma_005) - เพิ่มเติม
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_005'), 'ขันด้าม', 'th', false, 0.80, 1),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_005'), 'กระบอกน้ำ', 'th', false, 0.80, 1),

-- ภาชนะเก็บของ (lemma_006) - เพิ่มเติม
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_006'), 'กล่องเหลี่ยม', 'th', false, 0.85, 8),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_006'), 'กล่องล็อค', 'th', false, 0.90, 15),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_006'), 'กล่องล้อ', 'th', false, 0.85, 4),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_006'), 'เก๊ะลิ้นชัก', 'th', false, 0.80, 3),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_006'), 'ตะกร้ากลม', 'th', false, 0.85, 6),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_006'), 'ตะกร้าเหลี่ยม', 'th', false, 0.85, 4),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_006'), 'กระจาด', 'th', true, 0.90, 12),

-- ถังและกะละมัง (lemma_007) - เพิ่มเติม
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_007'), 'ถังน้ำ', 'th', false, 0.90, 8),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_007'), 'ถังฝา', 'th', false, 0.85, 4),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_007'), 'ถังปูน', 'th', false, 0.80, 6),

-- อุปกรณ์จัดระเบียบ (lemma_008) - เพิ่มเติม
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_008'), 'ตะขอตัวเอส', 'th', false, 0.80, 3),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_008'), 'ชั้นวาง', 'th', true, 0.90, 6),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_008'), 'ตะแกรง', 'th', true, 0.85, 8),

-- ภาชนะครัว (lemma_009) - เพิ่มเติม
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_009'), 'ขวดซอสแฟนซี', 'th', false, 0.85, 3),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_009'), 'ขวดโหล', 'th', false, 0.80, 8),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_009'), 'โหลแก้ว', 'th', false, 0.75, 6),

-- อุปกรณ์ตัดหั่น (lemma_010) - เพิ่มเติม
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_010'), 'เขียงอเนกประสงค์', 'th', false, 0.85, 1),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_010'), 'เขียงมะขาม', 'th', false, 0.80, 3),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_010'), 'มีดสับ', 'th', false, 0.85, 2),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_010'), 'มีดเหล็กกล้า', 'th', false, 0.80, 3),

-- อุปกรณ์รับประทาน (lemma_011) - เพิ่มเติม  
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_011'), 'ช้อนกาแฟ', 'th', false, 0.80, 1),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_011'), 'คีบสลัด', 'th', false, 0.80, 2),

-- อุปกรณ์ตวงวัด (lemma_012) - เพิ่มเติม
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_012'), 'ตักน้ำแข็ง', 'th', false, 0.85, 3),

-- อุปกรณ์ขัดล้าง (lemma_013) - เพิ่มเติม (skip existing terms)
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_013'), 'สครับเบอร์', 'th', false, 0.80, 2),

-- แปรงทำความสะอาด (lemma_014) - เพิ่มเติม
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_014'), 'แปรงซักผ้า', 'th', false, 0.85, 8),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_014'), 'แปรงห้องน้ำ', 'th', false, 0.80, 1),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_014'), 'แปรงเตารีด', 'th', false, 0.75, 2),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_014'), 'แปรงลวด', 'th', false, 0.80, 3),

-- ผ้าทำความสะอาด (lemma_015) - เพิ่มเติม
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_015'), 'ผ้าไมโคร', 'th', false, 0.85, 1),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_015'), 'ผ้าขาว', 'th', false, 0.80, 1)
ON CONFLICT (lemma_id, term) DO NOTHING;

-- Add terms for NEW lemmas (lemma_016 to lemma_026)
INSERT INTO synonym_terms (lemma_id, term, language, is_primary, confidence_score, usage_count) VALUES
-- อุปกรณ์ตัด (lemma_016)
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_016'), 'กรรไกร', 'th', true, 0.95, 4),

-- อุปกรณ์ติด (lemma_017)
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_017'), 'กาว', 'th', true, 0.95, 4),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_017'), 'กาวช้าง', 'th', false, 0.85, 1),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_017'), 'เทป', 'th', true, 0.90, 2),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_017'), 'เทปโฟม', 'th', false, 0.80, 1),

-- ของเล่นยานพาหนะ (lemma_018)
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_018'), 'รถ', 'th', true, 0.95, 8),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_018'), 'รถจ้าว', 'th', false, 0.85, 3),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_018'), 'รถตักดิน', 'th', false, 0.80, 2),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_018'), 'รถสามล้อ', 'th', false, 0.75, 1),

-- ของเล่นอาวุธ (lemma_019)
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_019'), 'ปืน', 'th', true, 0.90, 6),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_019'), 'ปืนฉีดน้ำ', 'th', false, 0.85, 6),

-- ผลิตภัณฑ์อาบน้ำ (lemma_020)
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_020'), 'ครีมอาบน้ำ', 'th', true, 0.90, 1),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_020'), 'โฟรเทคส์', 'th', false, 0.75, 1),

-- อุปกรณ์โกนหนวด (lemma_021)
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_021'), 'ใบมีดโกน', 'th', true, 0.90, 1),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_021'), 'โกบหนวด', 'th', false, 0.80, 1),

-- อุปกรณ์สัตว์เลี้ยง (lemma_022)
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_022'), 'ปลอกคอ', 'th', true, 0.95, 3),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_022'), 'ปลอกคอหมา', 'th', false, 0.90, 2),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_022'), 'ปลอกคอแมว', 'th', false, 0.85, 1),

-- อุปกรณ์ไฟฟ้าพกพา (lemma_023)
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_023'), 'ไฟฉาย', 'th', true, 0.90, 1),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_023'), 'หูฟัง', 'th', true, 0.85, 2),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_023'), 'สมอลทอด', 'th', false, 0.70, 2),

-- อุปกรณ์ให้นมเด็ก (lemma_024)
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_024'), 'จุกนม', 'th', true, 0.90, 1),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_024'), 'ขวดนม', 'th', false, 0.85, 1),

-- อุปกรณ์ความงาม (lemma_025)
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_025'), 'แหนบ', 'th', true, 0.90, 8),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_025'), 'แหนบหุ้มผึ้ง', 'th', false, 0.80, 2),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_025'), 'หวี', 'th', true, 0.85, 6),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_025'), 'หวีแปรง', 'th', false, 0.80, 4),

-- วัสดุตกแต่ง (lemma_026)
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_026'), 'ริบบิ้น', 'th', true, 0.85, 1),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_026'), 'กระดาษห่อของขวัญ', 'th', false, 0.80, 1),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_026'), 'เรนโบว์', 'th', false, 0.75, 2);

-- Add lemmas for categories that still don't have any synonyms
-- Add basic lemmas for major missing categories

-- เครื่องมือ categories ที่ยังขาด
INSERT INTO synonym_lemmas (code, name_th, name_en, description, category_id, is_verified) VALUES
('lemma_027', 'อุปกรณ์ประปาและท่อ', 'Plumbing Supplies', 'อุปกรณ์ประปาและการต่อท่อ', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_001_006'), true),
 
('lemma_028', 'วัสดุขัดผิว', 'Abrasive Materials', 'วัสดุสำหรับขัดและเงาผิว', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_001_007'), true);

INSERT INTO synonym_terms (lemma_id, term, language, is_primary, confidence_score, usage_count) VALUES
-- อุปกรณ์ประปาและท่อ (lemma_027)
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_027'), 'ก๊อก', 'th', true, 0.90, 2),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_027'), 'ท่อพลาสติก', 'th', false, 0.80, 1),

-- วัสดุขัดผิว (lemma_028)
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_028'), 'กระดาษทราย', 'th', true, 0.90, 3),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_028'), 'สักหลาด', 'th', false, 0.80, 1);

COMMIT;
