-- Migration: load_taxonomy  
-- Purpose: Load taxonomy nodes (main + sub categories)
-- Uses hybrid UUID+code pattern with subquery lookups for parent_id

BEGIN;

-- Clear existing taxonomy (child -> parent)
DELETE FROM taxonomy_nodes WHERE level > 0;
DELETE FROM taxonomy_nodes WHERE level = 0;

-- Load Main Categories (Level 0)
INSERT INTO taxonomy_nodes (code, name_th, name_en, level, parent_id, sort_order, keywords, is_active) VALUES
('cat_001', 'เครื่องมือ_ฮาร์ดแวร์', 'Hardware & Tools', 0, NULL, 1, 
 ARRAY['เครื่องมือ', 'ฮาร์ดแวร์', 'hardware', 'tools', 'คราด', 'เสียม', 'ค้อน', 'แปรง', 'สายยาง'], true),

('cat_002', 'ของใช้ในบ้าน', 'Household Items', 0, NULL, 2, 
 ARRAY['ของใช้ในบ้าน', 'household', 'ขัน', 'กระติก', 'กล่อง', 'ตะกร้า', 'ถัง', 'กะละมัง'], true),

('cat_003', 'เครื่องครัว', 'Kitchen & Cooking', 0, NULL, 3, 
 ARRAY['เครื่องครัว', 'kitchen', 'cooking', 'ขวดซอส', 'เหยือกตวง', 'เขียง', 'มีด', 'ช้อน', 'ส้อม'], true),

('cat_004', 'อุปกรณ์ทำความสะอาด', 'Cleaning Supplies', 0, NULL, 4, 
 ARRAY['ทำความสะอาด', 'cleaning', 'ฟองน้ำ', 'ใยขัด', 'แปรง', 'ผ้า', 'ถังขยะ'], true),

('cat_005', 'เครื่องเขียน_สำนักงาน', 'Stationery & Office', 0, NULL, 5, 
 ARRAY['เครื่องเขียน', 'สำนักงาน', 'stationery', 'office', 'กรรไกร', 'กาว', 'เทป'], true),

('cat_006', 'ของเล่น_นันทนาการ', 'Toys & Recreation', 0, NULL, 6, 
 ARRAY['ของเล่น', 'นันทนาการ', 'toys', 'recreation', 'รถ', 'ปืน', 'ฉีดน้ำ'], true),

('cat_007', 'ผลิตภัณฑ์ดูแลส่วนบุคคล', 'Personal Care', 0, NULL, 7, 
 ARRAY['ดูแลส่วนบุคคล', 'personal care', 'น้ำอบไทย', 'ใบมีดโกน', 'ครีมอาบน้ำ'], true),

('cat_008', 'ผลิตภัณฑ์ทำความสะอาดในบ้าน', 'Household Cleaning', 0, NULL, 8, 
 ARRAY['ทำความสะอาดในบ้าน', 'household cleaning', 'ลูกเหม็น', 'ดับกลิ่น', 'โซดา'], true),

('cat_009', 'สินค้าเพื่อสัตว์เลี้ยง', 'Pet Supplies', 0, NULL, 9, 
 ARRAY['สัตว์เลี้ยง', 'pet supplies', 'ปลอกคอ', 'หมา', 'แมว'], true),

('cat_010', 'เครื่องใช้ไฟฟ้า', 'Electrical Appliances', 0, NULL, 10, 
 ARRAY['เครื่องใช้ไฟฟ้า', 'electrical', 'appliances', 'ไฟฉาย', 'หูฟัง'], true),

('cat_011', 'แม่และเด็ก', 'Mother & Baby', 0, NULL, 11, 
 ARRAY['แม่และเด็ก', 'mother', 'baby', 'จุกนม', 'ขวดนม'], true),

('cat_012', 'เบ็ดเตล็ด', 'Miscellaneous', 0, NULL, 12, 
 ARRAY['เบ็ดเตล็ด', 'miscellaneous', 'ฟิล์ม', 'ผ้าปู', 'แหนบ', 'หวี', 'ริบบิ้น'], true);

-- Load Sub-Categories (Level 1) - using parent_id lookups
INSERT INTO taxonomy_nodes (code, name_th, name_en, level, parent_id, sort_order, keywords, is_active) VALUES
-- เครื่องมือ_ฮาร์ดแวร์ subcategories
('cat_001_001', 'อุปกรณ์ทำสวน', 'Gardening Tools', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_001'), 1, 
 ARRAY['ทำสวน', 'gardening', 'คราด', 'เสียม', 'สายยาง', 'กระถาง', 'กรรไกรตัดหญ้า'], true),

('cat_001_002', 'เครื่องมือช่างอื่น ๆ', 'Other Tools', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_001'), 2, 
 ARRAY['เครื่องมือช่าง', 'tools', 'หกเหลี่ยม', 'เอ็น', 'ค้อน', 'ตลับเมตร', 'ระดับน้ำ'], true),

('cat_001_003', 'สีและอุปกรณ์ทาสี', 'Paint & Painting Tools', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_001'), 3, 
 ARRAY['สี', 'ทาสี', 'paint', 'แปรงทาสี', 'กระดาษทราย'], true),

('cat_001_004', 'อุปกรณ์ประตูและกุญแจ', 'Door & Lock Hardware', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_001'), 4, 
 ARRAY['ประตู', 'กุญแจ', 'door', 'lock', 'โซ่จักรยาน'], true),

('cat_001_005', 'อุปกรณ์ไฟฟ้า', 'Electrical Equipment', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_001'), 5, 
 ARRAY['ไฟฟ้า', 'electrical', 'หัวแร้ง', 'สายพ่วง', 'ปลั๊ก', 'ไขควงวัดไฟ'], true),

-- Continue with more subcategories...
('cat_002_001', 'ขันน้ำ', 'Water Bowls', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_002'), 1, 
 ARRAY['ขัน', 'ขันน้ำ', 'water bowl', 'ขันปั๊ม', 'ลายไทย'], true),

('cat_002_003', 'กล่อง/ที่เก็บของ', 'Storage Boxes', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_002'), 3, 
 ARRAY['กล่อง', 'ที่เก็บของ', 'storage', 'box', 'ล็อค', 'ลิ้นชัก', 'ล้อ'], true),

('cat_002_004', 'ตะกร้า/กระจาด', 'Baskets', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_002'), 4, 
 ARRAY['ตะกร้า', 'กระจาด', 'basket', 'กลม', 'เหลี่ยม', 'เลส'], true),

('cat_002_006', 'ถังน้ำ/ถังเอนกประสงค์', 'Water Tanks & Buckets', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_002'), 6, 
 ARRAY['ถัง', 'ถังน้ำ', 'tank', 'bucket', 'ปากบาน', 'ฝา', 'ปูน'], true),

('cat_002_008', 'กระจก', 'Mirrors', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_002'), 8, 
 ARRAY['กระจก', 'mirror', 'พับ', 'หัวใจ', 'โค้ง'], true),

-- เครื่องครัว subcategories
('cat_003_001', 'ภาชนะใส่เครื่องปรุง / ขวดซอส', 'Condiment Containers', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_003'), 1, 
 ARRAY['ขวดซอส', 'เครื่องปรุง', 'condiment', 'sauce bottle', 'แฟนซี'], true),

('cat_003_002', 'อุปกรณ์ตวง', 'Measuring Tools', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_003'), 2, 
 ARRAY['ตวง', 'measuring', 'เหยือก', 'ตักน้ำแข็ง', 'มิเนียม'], true),

('cat_003_003', 'เขียง', 'Cutting Boards', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_003'), 3, 
 ARRAY['เขียง', 'cutting board', 'อเนกประสงค์', 'มะขาม', 'กลม', 'เหลี่ยม'], true),

('cat_003_007', 'มีดทำครัว', 'Kitchen Knives', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_003'), 7, 
 ARRAY['มีด', 'knife', 'kitchen', 'สับ', 'เหล็กกล้า', 'นกแก้ว', 'KIWI'], true),

-- อุปกรณ์ทำความสะอาด subcategories
('cat_004_001', 'ฟองน้ำ/ใยขัด/ฝอยขัด', 'Sponges & Scrubbers', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_004'), 1, 
 ARRAY['ฟองน้ำ', 'ใยขัด', 'ฝอยขัด', 'sponge', 'scrubber', 'ตาข่าย', 'ไบร์ท'], true),

('cat_004_002', 'แปรงขัดต่าง ๆ', 'Brushes', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_004'), 2, 
 ARRAY['แปรง', 'brush', 'ซักผ้า', 'ห้องน้ำ', 'เตารีด', 'ลวด', 'รองเท้า'], true),

('cat_004_004', 'ถุงขยะ/ถังขยะ/เก็บขยะ', 'Waste Management', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_004'), 4, 
 ARRAY['ถุงขยะ', 'ถังขยะ', 'waste', 'garbage', 'กทม', 'ฝา'], true),

-- เครื่องเขียน subcategories
('cat_005_001', 'กรรไกร/มีดคัตเตอร์', 'Scissors & Cutters', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_005'), 1, 
 ARRAY['กรรไกร', 'scissors', 'มีดคัตเตอร์', 'cutter', 'กุหลาบ'], true),

('cat_005_002', 'กาว/เทป', 'Glue & Tape', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_005'), 2, 
 ARRAY['กาว', 'เทป', 'glue', 'tape', 'ช้าง', 'โฟม', 'TOA'], true),

-- ของเล่น subcategories
('cat_006_001', 'รถของเล่น', 'Toy Cars', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_006'), 1, 
 ARRAY['รถ', 'จ้าว', 'ทะเลทราย', 'ตักดิน', 'สามล้อ', 'มอเตอร์ไซค์'], true),

('cat_006_002', 'ปืนฉีดน้ำ', 'Water Guns', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_006'), 2, 
 ARRAY['ปืน', 'ฉีดน้ำ', 'น้ำ'], true),

-- สัตว์เลี้ยง subcategories
('cat_009_001', 'ปลอกคอสัตว์เลี้ยง', 'Pet Collars', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_009'), 1, 
 ARRAY['ปลอกคอ', 'หมา', 'แมว', 'สัตว์เลี้ยง'], true),

-- แม่และเด็ก subcategories
('cat_011_001', 'สินค้าเด็ก', 'Baby Products', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_011'), 1, 
 ARRAY['จุกนม', 'ขวดนม', 'เด็ก', 'โกกิ'], true),

-- เบ็ดเตล็ด subcategories
('cat_012_002', 'อุปกรณ์ความงาม', 'Beauty Accessories', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_012'), 2, 
 ARRAY['แหนบ', 'หวี', 'แปรง', 'ความงาม'], true);

COMMIT;
