-- Thai Product Taxonomy Dataset
-- Generated from รายการสินค้าพร้อมหมวดหมู่_AI.txt
-- Total products analyzed: 3,105 items

-- Clear existing data
DELETE FROM synonym_terms WHERE lemma_id IN (SELECT id FROM synonym_lemmas);
DELETE FROM synonym_lemmas;
DELETE FROM taxonomy_nodes;

-- Insert Main Categories (Level 0)
INSERT INTO taxonomy_nodes (id, name_th, name_en, code, level, parent_id, sort_order, keywords, is_active) VALUES
('cat_001', 'เครื่องมือ_ฮาร์ดแวร์', 'Hardware & Tools', 'HW', 0, NULL, 1, 
 ARRAY['เครื่องมือ', 'ฮาร์ดแวร์', 'hardware', 'tools', 'คราด', 'เสียม', 'ค้อน', 'แปรง', 'สายยาง'], true),

('cat_002', 'ของใช้ในบ้าน', 'Household Items', 'HH', 0, NULL, 2, 
 ARRAY['ของใช้ในบ้าน', 'household', 'ขัน', 'กระติก', 'กล่อง', 'ตะกร้า', 'ถัง', 'กะละมัง'], true),

('cat_003', 'เครื่องครัว', 'Kitchen & Cooking', 'KIT', 0, NULL, 3, 
 ARRAY['เครื่องครัว', 'kitchen', 'cooking', 'ขวดซอส', 'เหยือกตวง', 'เขียง', 'มีด', 'ช้อน', 'ส้อม'], true),

('cat_004', 'อุปกรณ์ทำความสะอาด', 'Cleaning Supplies', 'CL', 0, NULL, 4, 
 ARRAY['ทำความสะอาด', 'cleaning', 'ฟองน้ำ', 'ใยขัด', 'แปรง', 'ผ้า', 'ถังขยะ'], true),

('cat_005', 'เครื่องเขียน_สำนักงาน', 'Stationery & Office', 'ST', 0, NULL, 5, 
 ARRAY['เครื่องเขียน', 'สำนักงาน', 'stationery', 'office', 'กรรไกร', 'กาว', 'เทป'], true),

('cat_006', 'ของเล่น_นันทนาการ', 'Toys & Recreation', 'TOY', 0, NULL, 6, 
 ARRAY['ของเล่น', 'นันทนาการ', 'toys', 'recreation', 'รถ', 'ปืน', 'ฉีดน้ำ'], true),

('cat_007', 'ผลิตภัณฑ์ดูแลส่วนบุคคล', 'Personal Care', 'PC', 0, NULL, 7, 
 ARRAY['ดูแลส่วนบุคคล', 'personal care', 'น้ำอบไทย', 'ใบมีดโกน', 'ครีมอาบน้ำ'], true),

('cat_008', 'ผลิตภัณฑ์ทำความสะอาดในบ้าน', 'Household Cleaning', 'CLH', 0, NULL, 8, 
 ARRAY['ทำความสะอาดในบ้าน', 'household cleaning', 'ลูกเหม็น', 'ดับกลิ่น', 'โซดา'], true),

('cat_009', 'สินค้าเพื่อสัตว์เลี้ยง', 'Pet Supplies', 'PET', 0, NULL, 9, 
 ARRAY['สัตว์เลี้ยง', 'pet supplies', 'ปลอกคอ', 'หมา', 'แมว'], true),

('cat_010', 'เครื่องใช้ไฟฟ้า', 'Electrical Appliances', 'ELC', 0, NULL, 10, 
 ARRAY['เครื่องใช้ไฟฟ้า', 'electrical', 'appliances', 'ไฟฉาย', 'หูฟัง'], true),

('cat_011', 'แม่และเด็ก', 'Mother & Baby', 'BB', 0, NULL, 11, 
 ARRAY['แม่และเด็ก', 'mother', 'baby', 'จุกนม', 'ขวดนม'], true),

('cat_012', 'เบ็ดเตล็ด', 'Miscellaneous', 'MIS', 0, NULL, 12, 
 ARRAY['เบ็ดเตล็ด', 'miscellaneous', 'ฟิล์ม', 'ผ้าปู', 'แหนบ', 'หวี', 'ริบบิ้น'], true);

-- Insert Sub-Categories (Level 1) for เครื่องมือ_ฮาร์ดแวร์
INSERT INTO taxonomy_nodes (id, name_th, name_en, code, level, parent_id, sort_order, keywords, is_active) VALUES
('cat_001_001', 'อุปกรณ์ทำสวน', 'Gardening Tools', 'HW_GARDEN', 1, 'cat_001', 1, 
 ARRAY['ทำสวน', 'gardening', 'คราด', 'เสียม', 'สายยาง', 'กระถาง', 'กรรไกรตัดหญ้า'], true),

('cat_001_002', 'เครื่องมือช่างอื่น ๆ', 'Other Tools', 'HW_TOOLS', 1, 'cat_001', 2, 
 ARRAY['เครื่องมือช่าง', 'tools', 'หกเหลี่ยม', 'เอ็น', 'ค้อน', 'ตลับเมตร', 'ระดับน้ำ'], true),

('cat_001_003', 'สีและอุปกรณ์ทาสี', 'Paint & Painting Tools', 'HW_PAINT', 1, 'cat_001', 3, 
 ARRAY['สี', 'ทาสี', 'paint', 'แปรงทาสี', 'กระดาษทราย'], true),

('cat_001_004', 'อุปกรณ์ประตูและกุญแจ', 'Door & Lock Hardware', 'HW_LOCK', 1, 'cat_001', 4, 
 ARRAY['ประตู', 'กุญแจ', 'door', 'lock', 'โซ่จักรยาน'], true),

('cat_001_005', 'อุปกรณ์ไฟฟ้า', 'Electrical Equipment', 'HW_ELEC', 1, 'cat_001', 5, 
 ARRAY['ไฟฟ้า', 'electrical', 'หัวแร้ง', 'สายพ่วง', 'ปลั๊ก', 'ไขควงวัดไฟ'], true),

('cat_001_006', 'อุปกรณ์ประปา', 'Plumbing Supplies', 'HW_PLUMB', 1, 'cat_001', 6, 
 ARRAY['ประปา', 'plumbing', 'ก๊อก', 'พลาสติก'], true),

('cat_001_007', 'วัสดุขัดผิว', 'Abrasive Materials', 'HW_ABRAS', 1, 'cat_001', 7, 
 ARRAY['ขัดผิว', 'abrasive', 'กระดาษทราย', 'สักหลาด'], true);

-- Insert Sub-Categories (Level 1) for ของใช้ในบ้าน
INSERT INTO taxonomy_nodes (id, name_th, name_en, code, level, parent_id, sort_order, keywords, is_active) VALUES
('cat_002_001', 'ขันน้ำ', 'Water Bowls', 'HH_BOWL', 1, 'cat_002', 1, 
 ARRAY['ขัน', 'ขันน้ำ', 'water bowl', 'ขันปั๊ม', 'ลายไทย'], true),

('cat_002_002', 'กระติก', 'Thermos & Containers', 'HH_THERM', 1, 'cat_002', 2, 
 ARRAY['กระติก', 'thermos', 'container', 'เหลี่ยม', 'ลิตร'], true),

('cat_002_003', 'กล่อง/ที่เก็บของ', 'Storage Boxes', 'HH_STORAGE', 1, 'cat_002', 3, 
 ARRAY['กล่อง', 'ที่เก็บของ', 'storage', 'box', 'ล็อค', 'ลิ้นชัก', 'ล้อ'], true),

('cat_002_004', 'ตะกร้า/กระจาด', 'Baskets', 'HH_BASKET', 1, 'cat_002', 4, 
 ARRAY['ตะกร้า', 'กระจาด', 'basket', 'กลม', 'เหลี่ยม', 'เลส'], true),

('cat_002_005', 'ตะแกรง/ชั้นวางของ', 'Racks & Shelves', 'HH_RACK', 1, 'cat_002', 5, 
 ARRAY['ตะแกรง', 'ชั้นวาง', 'rack', 'shelf', 'คว่ำจาน'], true),

('cat_002_006', 'ถังน้ำ/ถังเอนกประสงค์', 'Water Tanks & Buckets', 'HH_TANK', 1, 'cat_002', 6, 
 ARRAY['ถัง', 'ถังน้ำ', 'tank', 'bucket', 'ปากบาน', 'ฝา', 'ปูน'], true),

('cat_002_007', 'กะละมัง', 'Basins', 'HH_BASIN', 1, 'cat_002', 7, 
 ARRAY['กะละมัง', 'basin', 'เลส', 'เจาะรู'], true),

('cat_002_008', 'กระจก', 'Mirrors', 'HH_MIRROR', 1, 'cat_002', 8, 
 ARRAY['กระจก', 'mirror', 'พับ', 'หัวใจ', 'โค้ง'], true),

('cat_002_009', 'กระบอก_หัวฉีด_ขวด', 'Bottles & Sprayers', 'HH_BOTTLE', 1, 'cat_002', 9, 
 ARRAY['กระบอก', 'ขวด', 'bottle', 'หัวฉีด', 'โหล', 'แก้ว'], true),

('cat_002_010', 'ไม้แขวนเสื้อ_อุปกรณ์ตากผ้า', 'Hangers & Drying', 'HH_HANGER', 1, 'cat_002', 10, 
 ARRAY['แขวนเสื้อ', 'ตากผ้า', 'hanger', 'drying', 'ห่วง', 'ราว'], true),

('cat_002_011', 'อุปกรณ์จัดเก็บ', 'Organization Tools', 'HH_ORG', 1, 'cat_002', 11, 
 ARRAY['จัดเก็บ', 'organization', 'ตะขอ', 'เอส'], true),

('cat_002_012', 'เก้าอี_โต้ะ', 'Chairs & Tables', 'HH_FURN', 1, 'cat_002', 12, 
 ARRAY['เก้าอี้', 'โต๊ะ', 'chair', 'table', 'เตี้ย', 'ซักผ้า'], true),

('cat_002_013', 'ที่นอน / ฟูก', 'Mattresses & Bedding', 'HH_BED', 1, 'cat_002', 13, 
 ARRAY['ที่นอน', 'ฟูก', 'mattress', 'bedding', 'หมอน', 'ปิกนิก'], true);

-- Insert Sub-Categories (Level 1) for เครื่องครัว
INSERT INTO taxonomy_nodes (id, name_th, name_en, code, level, parent_id, sort_order, keywords, is_active) VALUES
('cat_003_001', 'ภาชนะใส่เครื่องปรุง / ขวดซอส', 'Condiment Containers', 'KIT_COND', 1, 'cat_003', 1, 
 ARRAY['ขวดซอส', 'เครื่องปรุง', 'condiment', 'sauce bottle', 'แฟนซี'], true),

('cat_003_002', 'อุปกรณ์ตวง', 'Measuring Tools', 'KIT_MEAS', 1, 'cat_003', 2, 
 ARRAY['ตวง', 'measuring', 'เหยือก', 'ตักน้ำแข็ง', 'มิเนียม'], true),

('cat_003_003', 'เขียง', 'Cutting Boards', 'KIT_BOARD', 1, 'cat_003', 3, 
 ARRAY['เขียง', 'cutting board', 'อเนกประสงค์', 'มะขาม', 'กลม', 'เหลี่ยม'], true),

('cat_003_004', 'ช้อน/ส้อม/อุปกรณ์รับประทานอาหาร', 'Cutlery & Utensils', 'KIT_CUTL', 1, 'cat_003', 4, 
 ARRAY['ช้อน', 'ส้อม', 'cutlery', 'utensils', 'กาแฟ', 'เลส'], true),

('cat_003_005', 'อุปกรณ์เสิร์ฟ/อุปกรณ์หนีบอาหาร', 'Serving & Tongs', 'KIT_SERV', 1, 'cat_003', 5, 
 ARRAY['เสิร์ฟ', 'serving', 'คีบ', 'tongs', 'สลัด', 'ใบไม้'], true),

('cat_003_006', 'หม้อ/ภาชนะหุงต้ม', 'Pots & Cooking Vessels', 'KIT_POT', 1, 'cat_003', 6, 
 ARRAY['หม้อ', 'pot', 'cooking', 'กาน้ำ', 'จระเข้'], true),

('cat_003_007', 'มีดทำครัว', 'Kitchen Knives', 'KIT_KNIFE', 1, 'cat_003', 7, 
 ARRAY['มีด', 'knife', 'kitchen', 'สับ', 'เหล็กกล้า', 'นกแก้ว', 'KIWI'], true),

('cat_003_008', 'ถาดรองอาหาร', 'Food Trays', 'KIT_TRAY', 1, 'cat_003', 8, 
 ARRAY['ถาด', 'tray', 'food', 'วางเค้ก', 'กลม', 'เลส'], true),

('cat_003_009', 'แก้วน้ำ/ถ้วย', 'Glasses & Cups', 'KIT_GLASS', 1, 'cat_003', 9, 
 ARRAY['แก้ว', 'ถ้วย', 'glass', 'cup', 'ขีดแดง'], true),

('cat_003_010', 'กระชอน/ที่กรอง', 'Strainers & Filters', 'KIT_STRAIN', 1, 'cat_003', 10, 
 ARRAY['กระชอน', 'ที่กรอง', 'strainer', 'filter', 'พลาสติก'], true),

('cat_003_011', 'อุปกรณ์ลับมีด', 'Knife Sharpeners', 'KIT_SHARP', 1, 'cat_003', 11, 
 ARRAY['ลับมีด', 'knife sharpener', 'โค้ง'], true),

('cat_003_012', 'กระทะ', 'Pans', 'KIT_PAN', 1, 'cat_003', 12, 
 ARRAY['กระทะ', 'pan', 'ม้วนขอบ'], true),

('cat_003_013', 'กรวย', 'Funnels', 'KIT_FUNNEL', 1, 'cat_003', 13, 
 ARRAY['กรวย', 'funnel', 'เลส', 'พลาสติก'], true),

('cat_003_014', 'อุปกรณ์ประกอบอาหาร/ตะแกรงครัว', 'Food Prep & Kitchen Strainers', 'KIT_PREP', 1, 'cat_003', 14, 
 ARRAY['ประกอบอาหาร', 'food prep', 'ขูดมะละกอ', 'ตะแกรงตักทอด', 'เลส'], true);

-- Insert Sub-Categories (Level 1) for อุปกรณ์ทำความสะอาด
INSERT INTO taxonomy_nodes (id, name_th, name_en, code, level, parent_id, sort_order, keywords, is_active) VALUES
('cat_004_001', 'ฟองน้ำ/ใยขัด/ฝอยขัด', 'Sponges & Scrubbers', 'CL_SPONGE', 1, 'cat_004', 1, 
 ARRAY['ฟองน้ำ', 'ใยขัด', 'ฝอยขัด', 'sponge', 'scrubber', 'ตาข่าย', 'ไบร์ท'], true),

('cat_004_002', 'แปรงขัดต่าง ๆ', 'Brushes', 'CL_BRUSH', 1, 'cat_004', 2, 
 ARRAY['แปรง', 'brush', 'ซักผ้า', 'ห้องน้ำ', 'เตารีด', 'ลวด', 'รองเท้า'], true),

('cat_004_003', 'ผ้าถูพื้น/ผ้าเอนกประสงค์', 'Cleaning Cloths', 'CL_CLOTH', 1, 'cat_004', 3, 
 ARRAY['ผ้า', 'cloth', 'ถูพื้น', 'ไมโคร', 'เอนกประสงค์'], true),

('cat_004_004', 'ถุงขยะ/ถังขยะ/เก็บขยะ', 'Waste Management', 'CL_WASTE', 1, 'cat_004', 4, 
 ARRAY['ถุงขยะ', 'ถังขยะ', 'waste', 'garbage', 'กทม', 'ฝา'], true);

-- Insert Sub-Categories (Level 1) for เครื่องเขียน_สำนักงาน
INSERT INTO taxonomy_nodes (id, name_th, name_en, code, level, parent_id, sort_order, keywords, is_active) VALUES
('cat_005_001', 'กรรไกร/มีดคัตเตอร์', 'Scissors & Cutters', 'ST_SCISS', 1, 'cat_005', 1, 
 ARRAY['กรรไกร', 'scissors', 'มีดคัตเตอร์', 'cutter', 'กุหลาบ'], true),

('cat_005_002', 'กาว/เทป', 'Glue & Tape', 'ST_GLUE', 1, 'cat_005', 2, 
 ARRAY['กาว', 'เทป', 'glue', 'tape', 'ช้าง', 'โฟม', 'TOA'], true),

('cat_005_003', 'สมุด/กระดาษ', 'Books & Paper', 'ST_PAPER', 1, 'cat_005', 3, 
 ARRAY['สมุด', 'กระดาษ', 'book', 'paper', 'ห่อของขวัญ', 'เรนโบว์'], true);

-- Insert Sub-Categories (Level 1) for ของเล่น_นันทนาการ
INSERT INTO taxonomy_nodes (id, name_th, name_en, code, level, parent_id, sort_order, keywords, is_active) VALUES
('cat_006_001', 'รถของเล่น', 'Toy Cars', 'TOY_CAR', 1, 'cat_006', 1, 
 ARRAY['รถ', 'toy car', 'จ้าว', 'ทะเลทราย', 'ตักดิน', 'สามล้อ', 'มอเตอร์ไซค์'], true),

('cat_006_002', 'ปืนของเล่น', 'Toy Guns', 'TOY_GUN', 1, 'cat_006', 2, 
 ARRAY['ปืน', 'toy gun', 'ฉีดน้ำ', 'water gun'], true),

('cat_006_003', 'ของเล่นกิจกรรม', 'Activity Toys', 'TOY_ACT', 1, 'cat_006', 3, 
 ARRAY['กิจกรรม', 'activity', 'นกหวีด', 'whistle'], true);

-- Insert Sub-Categories (Level 1) for ผลิตภัณฑ์ดูแลส่วนบุคคล
INSERT INTO taxonomy_nodes (id, name_th, name_en, code, level, parent_id, sort_order, keywords, is_active) VALUES
('cat_007_001', 'ผลิตภัณฑ์ระงับกลิ่นกาย', 'Deodorants', 'PC_DEO', 1, 'cat_007', 1, 
 ARRAY['ระงับกลิ่น', 'deodorant', 'น้ำอบไทย', 'แม่สาวิตรี'], true),

('cat_007_002', 'ผลิตภัณฑ์อื่น ๆ สำหรับส่วนบุคคล', 'Other Personal Care', 'PC_OTHER', 1, 'cat_007', 2, 
 ARRAY['ส่วนบุคคล', 'personal care', 'ใบมีดโกน', 'โกบหนวด'], true),

('cat_007_003', 'สบู่เหลว/ครีมอาบน้ำ', 'Liquid Soap & Shower Cream', 'PC_SOAP', 1, 'cat_007', 3, 
 ARRAY['สบู่เหลว', 'ครีมอาบน้ำ', 'liquid soap', 'shower cream', 'โฟรเทคส์'], true);

-- Insert Sub-Categories (Level 1) for ผลิตภัณฑ์ทำความสะอาดในบ้าน
INSERT INTO taxonomy_nodes (id, name_th, name_en, code, level, parent_id, sort_order, keywords, is_active) VALUES
('cat_008_001', 'ผลิตภัณฑ์ทำความสะอาดอื่น ๆ', 'Other Cleaning Products', 'CLH_OTHER', 1, 'cat_008', 1, 
 ARRAY['ทำความสะอาด', 'cleaning products', 'ลูกเหม็น', 'ดับกลิ่น', 'โซดาเกล็ด'], true);

-- Insert Sub-Categories (Level 1) for สินค้าเพื่อสัตว์เลี้ยง
INSERT INTO taxonomy_nodes (id, name_th, name_en, code, level, parent_id, sort_order, keywords, is_active) VALUES
('cat_009_001', 'ปลอกคอสัตว์เลี้ยง', 'Pet Collars', 'PET_COLLAR', 1, 'cat_009', 1, 
 ARRAY['ปลอกคอ', 'pet collar', 'หมา', 'แมว', 'ล็อค'], true);

-- Insert Sub-Categories (Level 1) for เครื่องใช้ไฟฟ้า
INSERT INTO taxonomy_nodes (id, name_th, name_en, code, level, parent_id, sort_order, keywords, is_active) VALUES
('cat_010_001', 'ไฟฉาย', 'Flashlights', 'ELC_FLASH', 1, 'cat_010', 1, 
 ARRAY['ไฟฉาย', 'flashlight', 'ชาร์จ'], true),

('cat_010_002', 'อุปกรณ์เสริมมือถือ', 'Mobile Accessories', 'ELC_MOBILE', 1, 'cat_010', 2, 
 ARRAY['อุปกรณ์เสริม', 'mobile accessories', 'หูฟัง', 'สมอลทอด'], true);

-- Insert Sub-Categories (Level 1) for แม่และเด็ก
INSERT INTO taxonomy_nodes (id, name_th, name_en, code, level, parent_id, sort_order, keywords, is_active) VALUES
('cat_011_001', 'ขวดนม/อุปกรณ์ให้นม', 'Baby Bottles & Feeding', 'BB_BOTTLE', 1, 'cat_011', 1, 
 ARRAY['ขวดนม', 'baby bottle', 'จุกนม', 'feeding', 'โกกิ'], true);

-- Insert Sub-Categories (Level 1) for เบ็ดเตล็ด
INSERT INTO taxonomy_nodes (id, name_th, name_en, code, level, parent_id, sort_order, keywords, is_active) VALUES
('cat_012_001', 'สินค้าเบ็ดเตล็ดอื่น ๆ', 'Other Miscellaneous', 'MIS_OTHER', 1, 'cat_012', 1, 
 ARRAY['เบ็ดเตล็ด', 'miscellaneous', 'เคซี่', 'ซุปเปอร์แพค', 'ฟิล์ม', 'ผ้าปู', 'ด้ามกระทะ'], true),

('cat_012_002', 'ของใช้ส่วนตัว/แฟชั่น', 'Personal Items & Fashion', 'MIS_FASHION', 1, 'cat_012', 2, 
 ARRAY['ของใช้ส่วนตัว', 'แฟชั่น', 'fashion', 'แหนบ', 'หวี', 'แปรง'], true),

('cat_012_003', 'ริบบิ้น/วัสดุตกแต่ง', 'Ribbons & Decorative Materials', 'MIS_DECOR', 1, 'cat_012', 3, 
 ARRAY['ริบบิ้น', 'ribbon', 'วัสดุตกแต่ง', 'decorative', 'ดิ้นทอง'], true);
