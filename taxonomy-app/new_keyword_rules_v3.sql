-- New Keyword Rules Batch 3
INSERT INTO keyword_rules (code, name, keywords, category_id, confidence_score, is_active) VALUES 
('rule_kw_046', 'Hanger and Drying Expansion', ARRAY['หนีบผ้า', 'ไม้หนีบ', 'ราวตากผ้า', 'ห่วงตากผ้า', 'ไม้แขวนเสื้อ'], '3a0697f1-7938-4cbb-a97b-c49b8a57f570', 0.85, true),
('rule_kw_047', 'Rack and Shelf Detection', ARRAY['คว่ำจาน', 'ตะแกรงวางของ', 'ที่คว่ำหม้อ', 'ตะแกรง', 'ชั้นวาง'], '8324e2b4-5096-4b11-b1f2-730d3f17e92d', 0.85, true),
('rule_kw_048', 'Bedding Items Detection', ARRAY['หมอนข้าง', 'ผ้าห่ม', 'มุ้งครอบ', 'มุ้งกระโจม', 'เสื่อ', 'ที่นอน'], 'f203581f-9091-4a6f-b79c-e0295fb3d5fc', 0.85, true),
('rule_kw_049', 'Frying Pan Detection', ARRAY['หม้อด้าม', 'กระทะเทฟลอน', 'กระทะก้นลึก', 'กระทะย่าง', 'กระทะ'], '0d8135a3-2aa1-4ed1-ba75-1c319e812111', 0.90, true),
('rule_kw_050', 'Strainer and Filter Detection', ARRAY['กระชอน', 'ที่กรองชา', 'ตะแกรงกรอง', 'แผ่นกรอง', 'ที่กรอง'], 'afdc9541-459d-4926-9772-c24f495f3398', 0.85, true),
('rule_kw_051', 'Food Prep Tools Detection', ARRAY['ขูดมะละกอ', 'ปอกผลไม้', 'มีดปอก', 'ที่สไลด์', 'ตะแกรงครัว'], '7aedad7c-eb97-44a8-a7db-6e70e10338c2', 0.85, true),
('rule_kw_052', 'Paper and Notebook Expansion', ARRAY['สมุดวาดเขียน', 'กระดาษรายงาน', 'ซองจดหมาย', 'กระดาษกาว', 'สมุด', 'กระดาษ'], 'b48e0f77-8cb7-4d4c-92ce-e3b2491499ce', 0.85, true),
('rule_kw_053', 'Mirror Expansion', ARRAY['กระจกเงา', 'กระจกแต่งหน้า', 'กระจกส่องหน้า', 'กระจก'], '26ca3dd8-e240-420c-b467-9ec24bf35f12', 0.95, true),
('rule_kw_054', 'Storage Organization Expansion', ARRAY['กล่องรองเท้า', 'แฟ้มเอกสาร', 'กระเป๋าเก็บของ', 'ที่จัดเก็บ'], '0969fe38-f751-4632-bb60-45772a0fd9ac', 0.80, true),
('rule_kw_055', 'Baby Care Detection', ARRAY['ขวดนม', 'จุกหลอก', 'ผ้าห่อตัว', 'แป้งเด็ก', 'สบู่เด็ก', 'สินค้าเด็ก'], 'f812b103-c27c-41f1-8b97-62e0c5344bf8', 0.85, true);
