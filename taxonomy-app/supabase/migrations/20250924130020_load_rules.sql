-- Migration: load_rules
-- Purpose: Load keyword & regex rules + system settings using hybrid UUID+code pattern
-- Requires taxonomy nodes present for category references

BEGIN;

-- Clear existing rules & settings
DELETE FROM keyword_rules;
DELETE FROM regex_rules;
DELETE FROM system_settings;

-- Load Keyword Rules with category_id lookups
INSERT INTO keyword_rules (code, name, description, keywords, category_id, match_type, priority, confidence_score, is_active) VALUES
-- เครื่องมือ_ฮาร์ดแวร์ Rules
('rule_kw_001', 'Garden Tools Detection', 'ตรวจจับเครื่องมือทำสวน', 
 ARRAY['คราด', 'เสียม', 'สายยาง', 'กระถาง', 'ทำสวน', 'ต้นไม้', 'กรรไกรตัดหญ้า'], 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_001_001'), 'contains', 10, 0.90, true),

('rule_kw_002', 'Hand Tools Detection', 'ตรวจจับเครื่องมือช่าง', 
 ARRAY['ค้อน', 'หกเหลี่ยม', 'เอ็น', 'ตลับเมตร', 'ระดับน้ำ', 'ไขควง'], 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_001_002'), 'contains', 9, 0.85, true),

('rule_kw_003', 'Paint Tools Detection', 'ตรวจจับอุปกรณ์ทาสี', 
 ARRAY['แปรงทาสี', 'กระดาษทราย', 'ทาสี', 'สี'], 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_001_003'), 'contains', 8, 0.85, true),

('rule_kw_004', 'Electrical Tools Detection', 'ตรวจจับอุปกรณ์ไฟฟ้า', 
 ARRAY['หัวแร้ง', 'สายพ่วง', 'ปลั๊ก', 'ไฟฟ้า', 'วัดไฟ'], 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_001_005'), 'contains', 8, 0.80, true),

-- ของใช้ในบ้าน Rules
('rule_kw_005', 'Water Containers Detection', 'ตรวจจับภาชนะใส่น้ำ', 
 ARRAY['ขัน', 'กระติก', 'กระบอกน้ำ', 'น้ำ'], 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_002_001'), 'contains', 10, 0.90, true),

('rule_kw_006', 'Storage Boxes Detection', 'ตรวจจับกล่องเก็บของ', 
 ARRAY['กล่อง', 'ล็อค', 'ลิ้นชัก', 'เก็บของ', 'ล้อ'], 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_002_003'), 'contains', 9, 0.85, true),

('rule_kw_007', 'Baskets Detection', 'ตรวจจับตะกร้าและกระจาด', 
 ARRAY['ตะกร้า', 'กระจาด', 'กลม', 'เหลี่ยม'], 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_002_004'), 'contains', 8, 0.85, true),

('rule_kw_008', 'Tanks and Buckets Detection', 'ตรวจจับถังและกะละมัง', 
 ARRAY['ถัง', 'กะละมัง', 'ฝา', 'ปูน'], 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_002_006'), 'contains', 8, 0.80, true),

('rule_kw_009', 'Mirrors Detection', 'ตรวจจับกระจก', 
 ARRAY['กระจก', 'พับ', 'หัวใจ', 'โค้ง'], 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_002_008'), 'contains', 7, 0.85, true),

-- เครื่องครัว Rules
('rule_kw_010', 'Sauce Bottles Detection', 'ตรวจจับขวดซอส', 
 ARRAY['ขวดซอส', 'ซอส', 'เครื่องปรุง', 'แฟนซี'], 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_003_001'), 'contains', 10, 0.90, true),

('rule_kw_011', 'Cutting Boards Detection', 'ตรวจจับเขียง', 
 ARRAY['เขียง', 'อเนกประสงค์', 'มะขาม'], 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_003_003'), 'contains', 9, 0.90, true),

('rule_kw_012', 'Kitchen Knives Detection', 'ตรวจจับมีดครัว', 
 ARRAY['มีด', 'สับ', 'เหล็กกล้า', 'นกแก้ว', 'KIWI'], 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_003_007'), 'contains', 9, 0.85, true),

('rule_kw_013', 'Measuring Tools Detection', 'ตรวจจับอุปกรณ์ตวง', 
 ARRAY['เหยือกตวง', 'ตวง', 'ตักน้ำแข็ง', 'วัด'], 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_003_002'), 'contains', 8, 0.80, true),

-- อุปกรณ์ทำความสะอาด Rules
('rule_kw_014', 'Sponges Detection', 'ตรวจจับฟองน้ำและใยขัด', 
 ARRAY['ฟองน้ำ', 'ใยขัด', 'ฝอยขัด', 'ตาข่าย', 'ไบร์ท'], 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_004_001'), 'contains', 10, 0.90, true),

('rule_kw_015', 'Brushes Detection', 'ตรวจจับแปรงทำความสะอาด', 
 ARRAY['แปรง', 'ซักผ้า', 'ห้องน้ำ', 'เตารีด', 'ลวด'], 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_004_002'), 'contains', 9, 0.85, true),

('rule_kw_016', 'Waste Management Detection', 'ตรวจจับถังขยะ', 
 ARRAY['ถังขยะ', 'ขยะ', 'กทม'], 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_004_004'), 'contains', 8, 0.85, true),

-- เครื่องเขียน Rules
('rule_kw_017', 'Scissors Detection', 'ตรวจจับกรรไกร', 
 ARRAY['กรรไกร', 'ตัด'], 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_005_001'), 'contains', 9, 0.90, true),

('rule_kw_018', 'Glue and Tape Detection', 'ตรวจจับกาวและเทป', 
 ARRAY['กาว', 'เทป', 'ช้าง', 'โฟม', 'TOA'], 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_005_002'), 'contains', 8, 0.85, true),

-- ของเล่น Rules
('rule_kw_019', 'Toy Cars Detection', 'ตรวจจับรถของเล่น', 
 ARRAY['รถ', 'จ้าว', 'ทะเลทราย', 'ตักดิน', 'สามล้อ', 'มอเตอร์ไซค์'], 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_006_001'), 'contains', 9, 0.85, true),

('rule_kw_020', 'Water Guns Detection', 'ตรวจจับปืนฉีดน้ำ', 
 ARRAY['ปืน', 'ฉีดน้ำ', 'น้ำ'], 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_006_002'), 'contains', 8, 0.80, true),

-- ผลิตภัณฑ์ส่วนบุคคล Rules
('rule_kw_021', 'Personal Care Detection', 'ตรวจจับผลิตภัณฑ์ส่วนบุคคล', 
 ARRAY['น้ำอบไทย', 'ใบมีดโกน', 'ครีมอาบน้ำ', 'โกบหนวด'], 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_007'), 'contains', 7, 0.75, true),

-- สัตว์เลี้ยง Rules
('rule_kw_022', 'Pet Collars Detection', 'ตรวจจับปลอกคอสัตว์เลี้ยง', 
 ARRAY['ปลอกคอ', 'หมา', 'แมว', 'สัตว์เลี้ยง'], 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_009_001'), 'contains', 8, 0.85, true),

-- เครื่องใช้ไฟฟ้า Rules
('rule_kw_023', 'Electronics Detection', 'ตรวจจับเครื่องใช้ไฟฟ้า', 
 ARRAY['ไฟฉาย', 'หูฟัง', 'ชาร์จ', 'สมอลทอด'], 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_010'), 'contains', 7, 0.75, true),

-- แม่และเด็ก Rules
('rule_kw_024', 'Baby Products Detection', 'ตรวจจับสินค้าเด็ก', 
 ARRAY['จุกนม', 'ขวดนม', 'เด็ก', 'โกกิ'], 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_011_001'), 'contains', 8, 0.80, true),

-- เบ็ดเตล็ด Rules
('rule_kw_025', 'Beauty Accessories Detection', 'ตรวจจับอุปกรณ์ความงาม', 
 ARRAY['แหนบ', 'หวี', 'แปรง', 'ความงาม'], 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_012_002'), 'contains', 6, 0.70, true);

-- Load Regex Rules
INSERT INTO regex_rules (code, name, description, pattern, flags, category_id, priority, confidence_score, is_active, test_cases) VALUES

-- Size and Dimension Patterns
('rule_rx_001', 'Size Detection', 'ตรวจจับขนาดและมิติ', 
 '\\b\\d+(\\.\\d+)?\\s*(ซม|cm|นิ้ว|inch|"|ลิตร|L|l|มล|ml|กรัม|g|kg|กก)\\b', 'gi', 
 NULL, 5, 0.70, true, 
 ARRAY['12 นิ้ว', '3 ลิตร', '5L', '100 มล', '2.5 ซม']),

-- Model Number Patterns
('rule_rx_002', 'Model Number Detection', 'ตรวจจับหมายเลขรุ่น', 
 '\\b(NO\\.|#|เบอร์)\\s*[A-Z0-9\\-]+\\b', 'gi', 
 NULL, 4, 0.60, true, 
 ARRAY['NO.304', '#9859', 'เบอร์ 1']),

-- Color Patterns
('rule_rx_003', 'Color Detection', 'ตรวจจับสี', 
 '\\b(สี)?(แดง|เขียว|น้ำเงิน|เหลือง|ขาว|ดำ|ชมพู|ม่วง|ส้ม|เทา|หวาน|สด|ใส|สีไม้)\\b', 'gi', 
 NULL, 3, 0.50, true, 
 ARRAY['สีแดง', 'เขียว', 'สีหวาน', 'ใส']),

-- Brand Patterns
('rule_rx_004', 'Brand Detection', 'ตรวจจับแบรนด์', 
 '\\b(ALLWAYS|KIWI|TOA|DORCO|ATM|NCL|SMT|SRT|KEY|IMM)\\b', 'gi', 
 NULL, 6, 0.75, true, 
 ARRAY['ALLWAYS', 'KIWI no.171', 'TOA']),

-- Quantity Patterns
('rule_rx_005', 'Quantity Detection', 'ตรวจจับจำนวน', 
 '\\b(\\d+)\\s*(ชิ้น|อัน|ใบ|ตัว|หัว|ชั้น|ช่อง)\\b', 'gi', 
 NULL, 4, 0.60, true, 
 ARRAY['3 ชิ้น', '5 อัน', '2 ชั้น']),

-- Kitchen Specific Patterns
('rule_rx_006', 'Kitchen Items Pattern', 'รูปแบบเฉพาะเครื่องครัว', 
 '\\b(ขวด|เหยือก|ช้อน|ส้อม|มีด|เขียง|กระทะ|หม้อ)\\b', 'gi', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_003'), 8, 0.80, true, 
 ARRAY['ขวดซอส', 'เหยือกตวง', 'มีดสับ']),

-- Cleaning Supplies Pattern
('rule_rx_007', 'Cleaning Supplies Pattern', 'รูปแบบอุปกรณ์ทำความสะอาด', 
 '\\b(ฟองน้ำ|ใยขัด|แปรง|ผ้า|ทำความสะอาด|ขัด|ล้าง)\\b', 'gi', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_004'), 8, 0.80, true, 
 ARRAY['ฟองน้ำตาข่าย', 'แปรงซักผ้า', 'ใยขัดชุด']),

-- Hardware Tools Pattern
('rule_rx_008', 'Hardware Tools Pattern', 'รูปแบบเครื่องมือฮาร์ดแวร์', 
 '\\b(คราด|เสียม|ค้อน|เอ็น|แปรงทาสี|สายยาง|กระถาง)\\b', 'gi', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_001'), 9, 0.85, true, 
 ARRAY['คราดมือเสือ', 'ค้อนหัวยาง', 'เอ็นขาว']),

-- Household Items Pattern
('rule_rx_009', 'Household Items Pattern', 'รูปแบบของใช้ในบ้าน', 
 '\\b(ขัน|กระติก|กล่อง|ตะกร้า|ถัง|กะละมัง|กระจก)\\b', 'gi', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_002'), 8, 0.80, true, 
 ARRAY['ขันน้ำ', 'กล่องล็อค', 'ตะกร้ากลม']),

-- Toys Pattern
('rule_rx_010', 'Toys Pattern', 'รูปแบบของเล่น', 
 '\\b(รถ|ปืน|ของเล่น|เด็ก|ฉีดน้ำ)\\b', 'gi', 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_006'), 7, 0.75, true, 
 ARRAY['รถจ้าว', 'ปืนฉีดน้ำ', 'รถตักดิน']);

-- Load System Settings (UUID only, no code needed)
INSERT INTO system_settings (search, processing, ai, ui) VALUES 
('{
   "vectorSearchEnabled": true,
   "textSearchEnabled": true, 
   "hybridSearchEnabled": true,
   "defaultSearchType": "hybrid",
   "maxResults": 50,
   "confidenceThreshold": 0.3
 }',
 '{
   "batchSize": 100,
   "maxConcurrentJobs": 5,
   "retryAttempts": 3,
   "timeoutSeconds": 30
 }',
 '{
   "embeddingModel": "text-embedding-ada-002",
   "apiProvider": "openai",
   "maxTokens": 4000,
   "temperature": 0.1
 }',
 '{
   "theme": "light",
   "language": "th", 
   "itemsPerPage": 20,
   "enableAnimations": true
 }');

COMMIT;
