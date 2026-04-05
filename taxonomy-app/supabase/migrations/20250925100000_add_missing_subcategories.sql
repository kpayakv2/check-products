-- Migration: add_missing_subcategories
-- Purpose: Add 12 missing subcategories from original dataset + short codes
-- Fix data completeness issue in taxonomy

BEGIN;

-- First, add short_code column to taxonomy_nodes if not exists
ALTER TABLE taxonomy_nodes ADD COLUMN IF NOT EXISTS short_code TEXT;

-- Update existing main categories with short codes
UPDATE taxonomy_nodes SET short_code = 'HW' WHERE code = 'cat_001';
UPDATE taxonomy_nodes SET short_code = 'HH' WHERE code = 'cat_002'; 
UPDATE taxonomy_nodes SET short_code = 'KIT' WHERE code = 'cat_003';
UPDATE taxonomy_nodes SET short_code = 'CL' WHERE code = 'cat_004';
UPDATE taxonomy_nodes SET short_code = 'ST' WHERE code = 'cat_005';
UPDATE taxonomy_nodes SET short_code = 'TOY' WHERE code = 'cat_006';
UPDATE taxonomy_nodes SET short_code = 'PC' WHERE code = 'cat_007';
UPDATE taxonomy_nodes SET short_code = 'CLH' WHERE code = 'cat_008';
UPDATE taxonomy_nodes SET short_code = 'PET' WHERE code = 'cat_009';
UPDATE taxonomy_nodes SET short_code = 'ELC' WHERE code = 'cat_010';
UPDATE taxonomy_nodes SET short_code = 'BB' WHERE code = 'cat_011';
UPDATE taxonomy_nodes SET short_code = 'MIS' WHERE code = 'cat_012';

-- Update existing subcategories with short codes
UPDATE taxonomy_nodes SET short_code = 'HW_GARDEN' WHERE code = 'cat_001_001';
UPDATE taxonomy_nodes SET short_code = 'HW_TOOLS' WHERE code = 'cat_001_002';
UPDATE taxonomy_nodes SET short_code = 'HW_PAINT' WHERE code = 'cat_001_003';
UPDATE taxonomy_nodes SET short_code = 'HW_LOCK' WHERE code = 'cat_001_004';
UPDATE taxonomy_nodes SET short_code = 'HW_ELEC' WHERE code = 'cat_001_005';
UPDATE taxonomy_nodes SET short_code = 'HH_BOWL' WHERE code = 'cat_002_001';
UPDATE taxonomy_nodes SET short_code = 'HH_STORAGE' WHERE code = 'cat_002_003';
UPDATE taxonomy_nodes SET short_code = 'HH_BASKET' WHERE code = 'cat_002_004';
UPDATE taxonomy_nodes SET short_code = 'HH_TANK' WHERE code = 'cat_002_006';
UPDATE taxonomy_nodes SET short_code = 'HH_MIRROR' WHERE code = 'cat_002_008';
UPDATE taxonomy_nodes SET short_code = 'KIT_COND' WHERE code = 'cat_003_001';
UPDATE taxonomy_nodes SET short_code = 'KIT_MEAS' WHERE code = 'cat_003_002';
UPDATE taxonomy_nodes SET short_code = 'KIT_BOARD' WHERE code = 'cat_003_003';
UPDATE taxonomy_nodes SET short_code = 'KIT_KNIFE' WHERE code = 'cat_003_007';
UPDATE taxonomy_nodes SET short_code = 'CL_SPONGE' WHERE code = 'cat_004_001';
UPDATE taxonomy_nodes SET short_code = 'CL_BRUSH' WHERE code = 'cat_004_002';
UPDATE taxonomy_nodes SET short_code = 'CL_WASTE' WHERE code = 'cat_004_004';
UPDATE taxonomy_nodes SET short_code = 'ST_SCISS' WHERE code = 'cat_005_001';
UPDATE taxonomy_nodes SET short_code = 'ST_GLUE' WHERE code = 'cat_005_002';
UPDATE taxonomy_nodes SET short_code = 'TOY_CAR' WHERE code = 'cat_006_001';
UPDATE taxonomy_nodes SET short_code = 'TOY_GUN' WHERE code = 'cat_006_002';
UPDATE taxonomy_nodes SET short_code = 'PET_COLLAR' WHERE code = 'cat_009_001';
UPDATE taxonomy_nodes SET short_code = 'BB_BOTTLE' WHERE code = 'cat_011_001';
UPDATE taxonomy_nodes SET short_code = 'MIS_FASHION' WHERE code = 'cat_012_002';

-- Add missing subcategories from original dataset

-- Missing ของใช้ในบ้าน subcategories
INSERT INTO taxonomy_nodes (code, short_code, name_th, name_en, level, parent_id, sort_order, keywords, is_active) VALUES
('cat_002_002', 'HH_THERM', 'กระติก', 'Thermos & Containers', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_002'), 2, 
 ARRAY['กระติก', 'thermos', 'container', 'เหลี่ยม', 'ลิตร'], true),

('cat_002_005', 'HH_RACK', 'ตะแกรง/ชั้นวางของ', 'Racks & Shelves', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_002'), 5, 
 ARRAY['ตะแกรง', 'ชั้นวาง', 'rack', 'shelf', 'คว่ำจาน'], true),

('cat_002_007', 'HH_BASIN', 'กะละมัง', 'Basins', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_002'), 7, 
 ARRAY['กะละมัง', 'basin', 'เลส', 'เจาะรู'], true),

('cat_002_009', 'HH_BOTTLE', 'กระบอก_หัวฉีด_ขวด', 'Bottles & Sprayers', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_002'), 9, 
 ARRAY['กระบอก', 'ขวด', 'bottle', 'หัวฉีด', 'โหล', 'แก้ว'], true),

('cat_002_010', 'HH_HANGER', 'ไม้แขวนเสื้อ_อุปกรณ์ตากผ้า', 'Hangers & Drying', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_002'), 10, 
 ARRAY['แขวนเสื้อ', 'ตากผ้า', 'hanger', 'drying', 'ห่วง', 'ราว'], true),

('cat_002_011', 'HH_ORG', 'อุปกรณ์จัดเก็บ', 'Organization Tools', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_002'), 11, 
 ARRAY['จัดเก็บ', 'organization', 'ตะขอ', 'เอส'], true),

('cat_002_012', 'HH_FURN', 'เก้าอี_โต้ะ', 'Chairs & Tables', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_002'), 12, 
 ARRAY['เก้าอี้', 'โต๊ะ', 'chair', 'table', 'เตี้ย', 'ซักผ้า'], true),

('cat_002_013', 'HH_BED', 'ที่นอน / ฟูก', 'Mattresses & Bedding', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_002'), 13, 
 ARRAY['ที่นอน', 'ฟูก', 'mattress', 'bedding', 'หมอน', 'ปิกนิก'], true);

-- Missing เครื่องครัว subcategories  
INSERT INTO taxonomy_nodes (code, short_code, name_th, name_en, level, parent_id, sort_order, keywords, is_active) VALUES
('cat_003_004', 'KIT_CUTL', 'ช้อน/ส้อม/อุปกรณ์รับประทานอาหาร', 'Cutlery & Utensils', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_003'), 4, 
 ARRAY['ช้อน', 'ส้อม', 'cutlery', 'utensils', 'กาแฟ', 'เลส'], true),

('cat_003_005', 'KIT_SERV', 'อุปกรณ์เสิร์ฟ/อุปกรณ์หนีบอาหาร', 'Serving & Tongs', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_003'), 5, 
 ARRAY['เสิร์ฟ', 'serving', 'คีบ', 'tongs', 'สลัด', 'ใบไม้'], true),

('cat_003_006', 'KIT_POT', 'หม้อ/ภาชนะหุงต้ม', 'Pots & Cooking Vessels', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_003'), 6, 
 ARRAY['หม้อ', 'pot', 'cooking', 'กาน้ำ', 'จระเข้'], true),

('cat_003_008', 'KIT_TRAY', 'ถาดรองอาหาร', 'Food Trays', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_003'), 8, 
 ARRAY['ถาด', 'tray', 'food', 'วางเค้ก', 'กลม', 'เลส'], true);

-- Missing other subcategories
INSERT INTO taxonomy_nodes (code, short_code, name_th, name_en, level, parent_id, sort_order, keywords, is_active) VALUES
-- Missing อุปกรณ์ทำความสะอาด
('cat_004_003', 'CL_CLOTH', 'ผ้าถูพื้น/ผ้าเอนกประสงค์', 'Cleaning Cloths', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_004'), 3, 
 ARRAY['ผ้า', 'cloth', 'ถูพื้น', 'ไมโคร', 'เอนกประสงค์'], true),

-- Missing เครื่องเขียน  
('cat_005_003', 'ST_PAPER', 'สมุด/กระดาษ', 'Books & Paper', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_005'), 3, 
 ARRAY['สมุด', 'กระดาษ', 'book', 'paper', 'ห่อของขวัญ', 'เรนโบว์'], true),

-- Missing ของเล่น
('cat_006_003', 'TOY_ACT', 'ของเล่นกิจกรรม', 'Activity Toys', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_006'), 3, 
 ARRAY['กิจกรรม', 'activity', 'นกหวีด', 'whistle'], true);

-- Add missing main category subcategories that were completely missing
INSERT INTO taxonomy_nodes (code, short_code, name_th, name_en, level, parent_id, sort_order, keywords, is_active) VALUES
-- ผลิตภัณฑ์ดูแลส่วนบุคคล (was completely missing subcategories)
('cat_007_001', 'PC_DEO', 'ผลิตภัณฑ์ระงับกลิ่นกาย', 'Deodorants', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_007'), 1, 
 ARRAY['ระงับกลิ่น', 'deodorant', 'น้ำอบไทย', 'แม่สาวิตรี'], true),

('cat_007_002', 'PC_OTHER', 'ผลิตภัณฑ์อื่น ๆ สำหรับส่วนบุคคล', 'Other Personal Care', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_007'), 2, 
 ARRAY['ส่วนบุคคล', 'personal care', 'ใบมีดโกน', 'โกบหนวด'], true),

('cat_007_003', 'PC_SOAP', 'สบู่เหลว/ครีมอาบน้ำ', 'Liquid Soap & Shower Cream', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_007'), 3, 
 ARRAY['สบู่เหลว', 'ครีมอาบน้ำ', 'liquid soap', 'shower cream', 'โฟรเทคส์'], true),

-- ผลิตภัณฑ์ทำความสะอาดในบ้าน (was completely missing subcategories)
('cat_008_001', 'CLH_OTHER', 'ผลิตภัณฑ์ทำความสะอาดอื่น ๆ', 'Other Cleaning Products', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_008'), 1, 
 ARRAY['ทำความสะอาด', 'cleaning products', 'ลูกเหม็น', 'ดับกลิ่น', 'โซดาเกล็ด'], true),

-- เครื่องใช้ไฟฟ้า (was completely missing subcategories)
('cat_010_001', 'ELC_FLASH', 'ไฟฉาย', 'Flashlights', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_010'), 1, 
 ARRAY['ไฟฉาย', 'flashlight', 'ชาร์จ'], true),

('cat_010_002', 'ELC_MOBILE', 'อุปกรณ์เสริมมือถือ', 'Mobile Accessories', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_010'), 2, 
 ARRAY['อุปกรณ์เสริม', 'mobile accessories', 'หูฟัง', 'สมอลทอด'], true),

-- เบ็ดเตล็ด additional subcategories
('cat_012_001', 'MIS_OTHER', 'สินค้าเบ็ดเตล็ดอื่น ๆ', 'Other Miscellaneous', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_012'), 1, 
 ARRAY['เบ็ดเตล็ด', 'miscellaneous', 'เคซี่', 'ซุปเปอร์แพค', 'ฟิล์ม', 'ผ้าปู', 'ด้ามกระทะ'], true),

('cat_012_003', 'MIS_DECOR', 'ริบบิ้น/วัสดุตกแต่ง', 'Ribbons & Decorative Materials', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_012'), 3, 
 ARRAY['ริบบิ้น', 'ribbon', 'วัสดุตกแต่ง', 'decorative', 'ดิ้นทอง'], true);

-- Add index on short_code for performance
CREATE INDEX IF NOT EXISTS idx_taxonomy_nodes_short_code ON taxonomy_nodes(short_code) WHERE short_code IS NOT NULL;

COMMIT;
