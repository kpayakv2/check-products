-- เพิ่มหมวดหมู่ที่ขาดจากข้อมูลเก่า (input/รายการสินค้าพร้อมหมวดหมู่_AI.txt)
-- หมวดหลักใหม่ 4 + หมวดย่อยใหม่ 63 (ครอบคลุมสินค้าเก่า 1,397 จาก 3,103 รายการ)
-- embedding เว้นเป็น NULL ไว้ก่อน ให้สคริปต์ generate ตามหลัง

BEGIN;

-- ===== หมวดหลักใหม่ 4 หมวด =====
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
VALUES ('cat_013', 'FB', 'อาหาร_เครื่องดื่ม', 0, NULL, 13, true, ARRAY['อาหาร', 'เครื่องดื่ม', 'food', 'beverage']::text[])
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
VALUES ('cat_014', 'RIT', 'ของใช้ในพิธีกรรม', 0, NULL, 14, true, ARRAY['พิธีกรรม', 'ศาสนา', 'ritual', 'พาน']::text[])
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
VALUES ('cat_015', 'MED', 'เวชภัณฑ์_สุขภาพ', 0, NULL, 15, true, ARRAY['เวชภัณฑ์', 'สุขภาพ', 'medical', 'health']::text[])
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
VALUES ('cat_016', 'AUTO', 'ยานยนต์', 0, NULL, 16, true, ARRAY['ยานยนต์', 'รถยนต์', 'automotive', 'car']::text[])
ON CONFLICT (code) DO NOTHING;

-- ===== หมวดย่อยใหม่ =====
-- ผลิตภัณฑ์ดูแลส่วนบุคคล (+10)
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_007_004', 'PC_LOTION', 'โลชั่น/ครีมบำรุงผิว', 1, p.id, 4, true, ARRAY['โลชั่น/ครีมบำรุงผิว', 'โลชั่น', 'ครีมบำรุงผิว']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_007'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_007_005', 'PC_HAIRCARE', 'ผลิตภัณฑ์ดูแลหนังศรีษะ', 1, p.id, 5, true, ARRAY['ผลิตภัณฑ์ดูแลหนังศรีษะ']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_007'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_007_006', 'PC_SOAPBAR', 'สบู่ก้อน', 1, p.id, 6, true, ARRAY['สบู่ก้อน']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_007'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_007_007', 'PC_POWDER', 'แป้งฝุ่น/แป้งเด็ก', 1, p.id, 7, true, ARRAY['แป้งฝุ่น/แป้งเด็ก', 'แป้งฝุ่น', 'แป้งเด็ก']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_007'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_007_008', 'PC_FACECARE', 'ผลิตภัณฑ์ดูแลหน้า', 1, p.id, 8, true, ARRAY['ผลิตภัณฑ์ดูแลหน้า']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_007'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_007_009', 'PC_TOOTHPASTE', 'ยาสีฟัน', 1, p.id, 9, true, ARRAY['ยาสีฟัน']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_007'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_007_010', 'PC_PERFUME', 'น้ำหอม/โคโลญจน์', 1, p.id, 10, true, ARRAY['น้ำหอม/โคโลญจน์', 'น้ำหอม', 'โคโลญจน์']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_007'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_007_011', 'PC_TOOTHBRUSH', 'แปรงสีฟัน', 1, p.id, 11, true, ARRAY['แปรงสีฟัน']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_007'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_007_012', 'PC_COTTONBUD', 'คัทตัลบัท/ก้านสำลี', 1, p.id, 12, true, ARRAY['คัทตัลบัท/ก้านสำลี', 'คัทตัลบัท', 'ก้านสำลี']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_007'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_007_013', 'PC_ORALCARE', 'ผลิตภัณฑ์ดูแลช่องปาก', 1, p.id, 13, true, ARRAY['ผลิตภัณฑ์ดูแลช่องปาก']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_007'
ON CONFLICT (code) DO NOTHING;

-- ของใช้ในบ้าน (+3)
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_002_014', 'HH_INSECT', 'อุปกรณ์ไล่แมลง / กันยุง', 1, p.id, 14, true, ARRAY['อุปกรณ์ไล่แมลง / กันยุง', 'อุปกรณ์ไล่แมลง', 'กันยุง']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_002'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_002_015', 'HH_IRONPAD', 'ผ้ารองรีด', 1, p.id, 15, true, ARRAY['ผ้ารองรีด']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_002'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_002_016', 'HH_TABLECLOTH', 'ผ้าปูโต๊ะ/แผ่นพลาสติกคลุมโต๊ะ', 1, p.id, 16, true, ARRAY['ผ้าปูโต๊ะ/แผ่นพลาสติกคลุมโต๊ะ', 'ผ้าปูโต๊ะ', 'แผ่นพลาสติกคลุมโต๊ะ']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_002'
ON CONFLICT (code) DO NOTHING;

-- เครื่องครัว (+6)
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_003_015', 'KIT_SPATULA', 'ตะหลิว/ทัพพี', 1, p.id, 15, true, ARRAY['ตะหลิว/ทัพพี', 'ตะหลิว', 'ทัพพี']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_003'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_003_016', 'KIT_LADLE', 'กระบวย', 1, p.id, 16, true, ARRAY['กระบวย']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_003'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_003_017', 'KIT_KITBOX', 'กล่องจัดเก็บในครัว', 1, p.id, 17, true, ARRAY['กล่องจัดเก็บในครัว']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_003'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_003_018', 'KIT_PLATE', 'จาน/ชาม', 1, p.id, 18, true, ARRAY['จาน/ชาม', 'จาน', 'ชาม']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_003'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_003_019', 'KIT_PRESERVE', 'อุปกรณ์ถนอมอาหาร', 1, p.id, 19, true, ARRAY['อุปกรณ์ถนอมอาหาร']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_003'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_003_020', 'KIT_FOODBOX', 'กล่องใส่อาหาร', 1, p.id, 20, true, ARRAY['กล่องใส่อาหาร']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_003'
ON CONFLICT (code) DO NOTHING;

-- เครื่องมือ_ฮาร์ดแวร์ (+5)
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_001_008', 'HW_PLIER', 'คีม', 1, p.id, 8, true, ARRAY['คีม']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_001'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_001_009', 'HW_FASTENER', 'น็อต_สกรู_อุปกรณ์ยึด', 1, p.id, 9, true, ARRAY['น็อต_สกรู_อุปกรณ์ยึด', 'น็อต', 'สกรู', 'อุปกรณ์ยึด']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_001'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_001_010', 'HW_PIPEFIT', 'ท่อ_ข้อต่อ_บานพับ', 1, p.id, 10, true, ARRAY['ท่อ_ข้อต่อ_บานพับ', 'ท่อ', 'ข้อต่อ', 'บานพับ']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_001'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_001_011', 'HW_SCREWDRIVER', 'ไขควง', 1, p.id, 11, true, ARRAY['ไขควง']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_001'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_001_012', 'HW_REGULATOR', 'ตัวควบคุมแรงดัน', 1, p.id, 12, true, ARRAY['ตัวควบคุมแรงดัน']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_001'
ON CONFLICT (code) DO NOTHING;

-- เบ็ดเตล็ด (+4)
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_012_004', 'MIS_FASHION', 'ของใช้ส่วนตัว/แฟชั่น', 1, p.id, 4, true, ARRAY['ของใช้ส่วนตัว/แฟชั่น', 'ของใช้ส่วนตัว', 'แฟชั่น']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_012'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_012_005', 'MIS_BAG', 'ถุงใส่ของ', 1, p.id, 5, true, ARRAY['ถุงใส่ของ']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_012'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_012_006', 'MIS_SEWING', 'อุปกรณ์ตัดเย็บผ้า', 1, p.id, 6, true, ARRAY['อุปกรณ์ตัดเย็บผ้า']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_012'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_012_007', 'MIS_BATTERY', 'ถ่านไฟฉาย', 1, p.id, 7, true, ARRAY['ถ่านไฟฉาย']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_012'
ON CONFLICT (code) DO NOTHING;

-- ผลิตภัณฑ์ทำความสะอาดในบ้าน (+7)
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_008_002', 'CLH_SOFTENER', 'น้ำยาปรับผ้านุ่ม', 1, p.id, 2, true, ARRAY['น้ำยาปรับผ้านุ่ม']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_008'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_008_003', 'CLH_DETERGENT', 'ผงซักฟอก/น้ำยาซักผ้า', 1, p.id, 3, true, ARRAY['ผงซักฟอก/น้ำยาซักผ้า', 'ผงซักฟอก', 'น้ำยาซักผ้า']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_008'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_008_004', 'CLH_DISHWASH', 'น้ำยาล้างจาน', 1, p.id, 4, true, ARRAY['น้ำยาล้างจาน']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_008'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_008_005', 'CLH_TISSUE', 'กระดาษชำระ', 1, p.id, 5, true, ARRAY['กระดาษชำระ']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_008'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_008_006', 'CLH_FLOORCLEAN', 'น้ำยาถูพื้น', 1, p.id, 6, true, ARRAY['น้ำยาถูพื้น']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_008'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_008_007', 'CLH_BLEACH', 'น้ำยาฟอกขาว/ขจัดคราบ', 1, p.id, 7, true, ARRAY['น้ำยาฟอกขาว/ขจัดคราบ', 'น้ำยาฟอกขาว', 'ขจัดคราบ']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_008'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_008_008', 'CLH_SCOURPOWDER', 'ผงขัดทำความสะอาด', 1, p.id, 8, true, ARRAY['ผงขัดทำความสะอาด']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_008'
ON CONFLICT (code) DO NOTHING;

-- เครื่องเขียน_สำนักงาน (+5)
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_005_004', 'ST_FOLDER', 'แฟ้ม/อุปกรณ์สำนักงาน', 1, p.id, 4, true, ARRAY['แฟ้ม/อุปกรณ์สำนักงาน', 'แฟ้ม', 'อุปกรณ์สำนักงาน']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_005'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_005_005', 'ST_CLIP', 'คลิปหนีบกระดาษ', 1, p.id, 5, true, ARRAY['คลิปหนีบกระดาษ']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_005'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_005_006', 'ST_STAPLER', 'เครื่องเย็บกระดาษ/ที่เจาะ', 1, p.id, 6, true, ARRAY['เครื่องเย็บกระดาษ/ที่เจาะ', 'เครื่องเย็บกระดาษ', 'ที่เจาะ']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_005'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_005_007', 'ST_PRICETAG', 'ป้ายแสดงราคา/ป้ายโปรโมชั่น', 1, p.id, 7, true, ARRAY['ป้ายแสดงราคา/ป้ายโปรโมชั่น', 'ป้ายแสดงราคา', 'ป้ายโปรโมชั่น']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_005'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_005_008', 'ST_PEN', 'ปากกา/ดินสอ', 1, p.id, 8, true, ARRAY['ปากกา/ดินสอ', 'ปากกา', 'ดินสอ']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_005'
ON CONFLICT (code) DO NOTHING;

-- อุปกรณ์ทำความสะอาด (+5)
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_004_005', 'CL_BROOM', 'ไม้กวาดและที่โกยผง', 1, p.id, 5, true, ARRAY['ไม้กวาดและที่โกยผง']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_004'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_004_006', 'CL_DUSTER', 'อุปกรณ์ทำความสะอาดฝุ่น', 1, p.id, 6, true, ARRAY['อุปกรณ์ทำความสะอาดฝุ่น']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_004'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_004_007', 'CL_MOP', 'ไม้ถูพื้น / ม็อบถูพื้น', 1, p.id, 7, true, ARRAY['ไม้ถูพื้น / ม็อบถูพื้น', 'ไม้ถูพื้น', 'ม็อบถูพื้น']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_004'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_004_008', 'CL_LAUNDRYBAG', 'ถุงซักผ้า/ตาข่ายซักผ้า', 1, p.id, 8, true, ARRAY['ถุงซักผ้า/ตาข่ายซักผ้า', 'ถุงซักผ้า', 'ตาข่ายซักผ้า']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_004'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_004_009', 'CL_SQUEEGEE', 'ยางรีดน้ำ', 1, p.id, 9, true, ARRAY['ยางรีดน้ำ']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_004'
ON CONFLICT (code) DO NOTHING;

-- ของเล่น_นันทนาการ (+4)
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_006_004', 'TOY_TOYOTHER', 'ของเล่นอื่น ๆ', 1, p.id, 4, true, ARRAY['ของเล่นอื่น ๆ', 'ของเล่นอื่น']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_006'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_006_005', 'TOY_ROLEPLAY', 'ชุดของเล่นบทบาทสมมติ', 1, p.id, 5, true, ARRAY['ชุดของเล่นบทบาทสมมติ']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_006'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_006_006', 'TOY_TOYGUN', 'ปืนของเล่น', 1, p.id, 6, true, ARRAY['ปืนของเล่น']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_006'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_006_007', 'TOY_DOLL', 'ตุ๊กตา', 1, p.id, 7, true, ARRAY['ตุ๊กตา']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_006'
ON CONFLICT (code) DO NOTHING;

-- สินค้าเพื่อสัตว์เลี้ยง (+5)
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_009_002', 'PET_PETFEED', 'อุปกรณ์ให้อาหาร', 1, p.id, 2, true, ARRAY['อุปกรณ์ให้อาหาร']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_009'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_009_003', 'PET_LEASH', 'สายจูงสัตว์เลี้ยง', 1, p.id, 3, true, ARRAY['สายจูงสัตว์เลี้ยง']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_009'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_009_004', 'PET_PETBRUSH', 'อุปกรณ์แปรงขน', 1, p.id, 4, true, ARRAY['อุปกรณ์แปรงขน']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_009'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_009_005', 'PET_PETOTHER', 'ของใช้สัตว์เลี้ยง', 1, p.id, 5, true, ARRAY['ของใช้สัตว์เลี้ยง']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_009'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_009_006', 'PET_PETFOOD', 'อาหารสัตว์', 1, p.id, 6, true, ARRAY['อาหารสัตว์']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_009'
ON CONFLICT (code) DO NOTHING;

-- เครื่องใช้ไฟฟ้า (+2)
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_010_003', 'ELC_APPLIANCE', 'เครื่องใช้ไฟฟ้าในบ้านอื่น ๆ', 1, p.id, 3, true, ARRAY['เครื่องใช้ไฟฟ้าในบ้านอื่น ๆ', 'เครื่องใช้ไฟฟ้าในบ้านอื่น']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_010'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_010_004', 'ELC_LIGHTING', 'หลอดไฟ/อุปกรณ์แสงสว่าง', 1, p.id, 4, true, ARRAY['หลอดไฟ/อุปกรณ์แสงสว่าง', 'หลอดไฟ', 'อุปกรณ์แสงสว่าง']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_010'
ON CONFLICT (code) DO NOTHING;

-- แม่และเด็ก (+2)
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_011_002', 'BB_FEEDBOTTLE', 'ขวดนม/อุปกรณ์ให้นม', 1, p.id, 2, true, ARRAY['ขวดนม/อุปกรณ์ให้นม', 'ขวดนม', 'อุปกรณ์ให้นม']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_011'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_011_003', 'BB_BABYCARE', 'ผลิตภัณฑ์ดูแลเด็ก', 1, p.id, 3, true, ARRAY['ผลิตภัณฑ์ดูแลเด็ก']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_011'
ON CONFLICT (code) DO NOTHING;

-- อาหาร_เครื่องดื่ม (+2)
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_013_001', 'FB_RICE', 'ข้าวสาร', 1, p.id, 1, true, ARRAY['ข้าวสาร']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_013'
ON CONFLICT (code) DO NOTHING;
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_013_002', 'FB_DRINK', 'เครื่องดื่ม', 1, p.id, 2, true, ARRAY['เครื่องดื่ม']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_013'
ON CONFLICT (code) DO NOTHING;

-- ของใช้ในพิธีกรรม (+1)
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_014_001', 'RIT_TRAY', 'พาน / พานโตก', 1, p.id, 1, true, ARRAY['พาน / พานโตก', 'พาน', 'พานโตก']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_014'
ON CONFLICT (code) DO NOTHING;

-- เวชภัณฑ์_สุขภาพ (+1)
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_015_001', 'MED_PATIENT', 'สุขภัณฑ์ / อุปกรณ์ผู้ป่วย', 1, p.id, 1, true, ARRAY['สุขภัณฑ์ / อุปกรณ์ผู้ป่วย', 'สุขภัณฑ์', 'อุปกรณ์ผู้ป่วย']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_015'
ON CONFLICT (code) DO NOTHING;

-- ยานยนต์ (+1)
INSERT INTO taxonomy_nodes (code, short_code, name_th, level, parent_id, sort_order, is_active, keywords)
SELECT 'cat_016_001', 'AUTO_CARCARE', 'อุปกรณ์ดูแลรถยนต์', 1, p.id, 1, true, ARRAY['อุปกรณ์ดูแลรถยนต์']::text[]
FROM taxonomy_nodes p WHERE p.code = 'cat_016'
ON CONFLICT (code) DO NOTHING;

COMMIT;
