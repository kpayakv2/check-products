-- Migration: load_synonyms
-- Purpose: Load synonym lemmas & terms using hybrid UUID+code pattern
-- Requires taxonomy nodes already loaded for category_id lookups

BEGIN;

-- Clear existing synonyms
DELETE FROM synonym_terms;
DELETE FROM synonym_lemmas;

-- Load Synonym Lemmas with category_id lookups
INSERT INTO synonym_lemmas (code, name_th, name_en, description, category_id, is_verified) VALUES
('lemma_001', 'เครื่องมือทำสวน', 'Gardening Tools', 'อุปกรณ์สำหรับทำสวนและดูแลต้นไม้', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_001_001'), true),
 
('lemma_002', 'เครื่องมือช่าง', 'Hand Tools', 'เครื่องมือช่างทั่วไป', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_001_002'), true),
 
('lemma_003', 'อุปกรณ์ทาสี', 'Painting Supplies', 'อุปกรณ์สำหรับทาสีและตกแต่ง', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_001_003'), true),
 
('lemma_004', 'อุปกรณ์ไฟฟ้า', 'Electrical Tools', 'เครื่องมือและอุปกรณ์ไฟฟ้า', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_001_005'), true),
 
('lemma_005', 'ภาชนะใส่น้ำ', 'Water Containers', 'ภาชนะสำหรับใส่น้ำและเครื่องดื่ม', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_002_001'), true),
 
('lemma_006', 'ภาชนะเก็บของ', 'Storage Containers', 'กล่อง ตะกร้า และภาชนะเก็บของต่างๆ', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_002_003'), true),
 
('lemma_007', 'ถังและกะละมัง', 'Buckets and Basins', 'ภาชนะขนาดใหญ่สำหรับใส่น้ำและของใช้', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_002_006'), true),
 
('lemma_008', 'อุปกรณ์จัดระเบียบ', 'Organization Tools', 'อุปกรณ์สำหรับจัดระเบียบในบ้าน', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_002_008'), true),
 
('lemma_009', 'ภาชนะครัว', 'Kitchen Containers', 'ภาชนะสำหรับใส่เครื่องปรุงและอาหาร', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_003_001'), true),
 
('lemma_010', 'อุปกรณ์ตัดหั่น', 'Cutting Tools', 'มีดและเขียงสำหรับเตรียมอาหาร', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_003_003'), true),
 
('lemma_011', 'อุปกรณ์รับประทาน', 'Eating Utensils', 'ช้อน ส้อม และอุปกรณ์รับประทานอาหาร', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_003_007'), true),
 
('lemma_012', 'อุปกรณ์ตวงวัด', 'Measuring Tools', 'เครื่องมือสำหรับตวงและวัดในครัว', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_003_002'), true),
 
('lemma_013', 'อุปกรณ์ขัดล้าง', 'Scrubbing Tools', 'ฟองน้ำ ใยขัด และอุปกรณ์ขัดล้าง', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_004_001'), true),
 
('lemma_014', 'แปรงทำความสะอาด', 'Cleaning Brushes', 'แปรงสำหรับทำความสะอาดต่างๆ', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_004_002'), true),
 
('lemma_015', 'ผ้าทำความสะอาด', 'Cleaning Cloths', 'ผ้าสำหรับเช็ดและทำความสะอาด', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_004_004'), true);

-- Load Synonym Terms with lemma_id lookups  
INSERT INTO synonym_terms (lemma_id, term, language, is_primary, confidence_score, usage_count) VALUES
-- เครื่องมือทำสวน
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_001'), 'คราด', 'th', true, 0.95, 5),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_001'), 'คราดมือเสือ', 'th', false, 0.90, 2),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_001'), 'เสียม', 'th', true, 0.95, 3),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_001'), 'เสียมมิด', 'th', false, 0.85, 1),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_001'), 'สายยาง', 'th', true, 0.90, 8),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_001'), 'สายยางฟ้า', 'th', false, 0.85, 2),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_001'), 'กระถาง', 'th', true, 0.90, 12),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_001'), 'กระถางต้นไม้', 'th', false, 0.85, 8),

-- เครื่องมือช่าง
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_002'), 'ค้อน', 'th', true, 0.95, 3),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_002'), 'ค้อนหัวยาง', 'th', false, 0.90, 2),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_002'), 'หกเหลี่ยม', 'th', true, 0.85, 4),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_002'), 'เอ็น', 'th', true, 0.90, 15),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_002'), 'เอ็นขาว', 'th', false, 0.85, 8),

-- ภาชนะใส่น้ำ
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_005'), 'ขัน', 'th', true, 0.95, 8),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_005'), 'ขันน้ำ', 'th', false, 0.90, 3),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_005'), 'ขันปั๊ม', 'th', false, 0.85, 6),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_005'), 'กระติก', 'th', true, 0.95, 8),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_005'), 'กระติกเหลี่ยม', 'th', false, 0.85, 3),

-- อุปกรณ์ขัดล้าง
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_013'), 'ฟองน้ำ', 'th', true, 0.95, 8),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_013'), 'ฟองน้ำตาข่าย', 'th', false, 0.85, 3),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_013'), 'ใยขัด', 'th', true, 0.90, 12),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_013'), 'ฝอยขัด', 'th', false, 0.85, 6),
((SELECT id FROM synonym_lemmas WHERE code = 'lemma_013'), 'ไบร์ท', 'th', false, 0.80, 4);

COMMIT;
