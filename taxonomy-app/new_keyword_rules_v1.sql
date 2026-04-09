-- New Keyword Rules for Expansion
INSERT INTO keyword_rules (code, name, keywords, category_id, confidence_score, is_active) VALUES 
('rule_kw_026', 'Kitchenware Detection', ARRAY['ตะหลิว', 'ทัพพี', 'ที่คีบ', 'ตะกร้อ', 'ที่เปิดขวด', 'ครก', 'สาก'], 'e8bcd9c6-4e55-43af-90e7-5d188a23cb38', 0.85, true),
('rule_kw_027', 'Water Flask Detection', ARRAY['กระบอกน้ำ', 'คูลเลอร์', 'เก็บความร้อน', 'เก็บความเย็น', 'กระติก'], '55326a0c-d235-40a6-94b5-c4ccde75dbfe', 0.85, true),
('rule_kw_028', 'Flashlight Detection', ARRAY['สปอร์ตไลท์', 'ตะเกียง', 'ถ่านไฟฉาย', 'ชาร์จไฟ', 'ไฟฉาย'], '7b9b5fcd-244a-43d6-bc59-e08a18072ec7', 0.90, true),
('rule_kw_029', 'Cup and Glass Detection', ARRAY['แก้วมัค', 'ถ้วยน้ำ', 'แก้วพลาสติก', 'แก้วเยติ', 'แก้วน้ำ'], '41977e9c-46de-4b5d-8225-0675a195c617', 0.85, true),
('rule_kw_030', 'Cleaning Tools Expansion', ARRAY['ไม้กวาด', 'ไม้ถูพื้น', 'ที่ตักผง', 'แปรงถูพื้น', 'ถังถูพื้น'], '4717eee9-01a1-4e5a-b8e9-36b7efef8fc5', 0.80, true),
('rule_kw_031', 'Baby Products Expansion', ARRAY['ผ้าอ้อม', 'แพมเพิส', 'รถเข็นเด็ก', 'เปล', 'จุกนม'], '482a79dc-361a-4472-82cf-7f914c7e5d45', 0.85, true),
('rule_kw_032', 'Basin Detection', ARRAY['อ่างล้างหน้า', 'อ่างอาบน้ำเด็ก', 'ซักผ้า', 'กะละมัง'], 'af9d3cea-c1d9-4f97-96b3-cd77d7d30122', 0.90, true),
('rule_kw_033', 'Furniture Detection', ARRAY['ม้านั่ง', 'โซฟา', 'โต๊ะพับ', 'โต๊ะทำงาน', 'เก้าอี้', 'โต๊ะ'], 'b5be1680-5924-4911-9b9f-35aed2fc64f0', 0.85, true),
('rule_kw_034', 'Cooking Pot Expansion', ARRAY['กระทะ', 'ลังถึง', 'หม้อซึ้ง', 'หม้อแรงดัน', 'หม้อ'], 'a75e45ec-29e4-4686-86cf-3c3a56f1911e', 0.85, true),
('rule_kw_035', 'Cutlery Detection', ARRAY['ตะเกียบ', 'มีดสเต็ก', 'ช้อนกาแฟ', 'ช้อน', 'ส้อม'], '9b702924-704d-497a-9ae1-a6d10c9e61f9', 0.90, true);
