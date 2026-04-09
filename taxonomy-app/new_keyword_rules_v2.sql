-- New Keyword Rules Batch 2
INSERT INTO keyword_rules (code, name, keywords, category_id, confidence_score, is_active) VALUES 
('rule_kw_036', 'Hardware Tools Expansion', ARRAY['ค้อน', 'ไขควง', 'คีม', 'เลื่อย', 'ประแจ', 'สว่าน'], '3ffe3a54-5f8e-450d-a138-dfa20cc6aaed', 0.85, true),
('rule_kw_037', 'Organization Tools Expansion', ARRAY['ชั้นวาง', 'ที่แขวน', 'ตะขอ', 'แร็ค', 'ราวแขวน'], '0969fe38-f751-4632-bb60-45772a0fd9ac', 0.80, true),
('rule_kw_038', 'Stationery Detection', ARRAY['ปากกา', 'ดินสอ', 'ยางลบ', 'ไม้บรรทัด', 'สมุดจด', 'แม็กซ์'], 'f3deeb9a-b5bf-4770-bcc1-402fc7c7c0c3', 0.90, true),
('rule_kw_039', 'Spray Bottle Detection', ARRAY['ฟ็อกกี้', 'กระบอกฉีดน้ำ', 'หัวสเปรย์', 'ฟร็อกกี้'], '844c0f9f-6d01-4599-96b4-b6a832e95369', 0.90, true),
('rule_kw_040', 'Miscellaneous Items Expansion', ARRAY['กุญแจมือ', 'นกหวีด', 'เข็มทิศ', 'ไฟแช็ค', 'ร่ม'], 'b99f86d8-4370-4709-a235-76ab37ba910a', 0.75, true),
('rule_kw_041', 'Deodorant Detection', ARRAY['สเปรย์ระงับกลิ่น', 'โรลออน', 'แป้งเต่า', 'สารส้ม', 'ระงับกลิ่นกาย'], '57ef13ea-b5c9-468b-a08c-0b7beb436c44', 0.90, true),
('rule_kw_042', 'Liquid Soap Detection', ARRAY['แชมพู', 'ครีมนวด', 'โฟมล้างหน้า', 'เจลอาบน้ำ', 'สบู่เหลว'], '3f867897-970c-41ff-9c7e-2633edd42f7f', 0.85, true),
('rule_kw_043', 'Plumbing Tools Detection', ARRAY['ก๊อกน้ำ', 'สายยาง', 'ท่อน้ำ', 'เทปพันเกลียว', 'ข้อต่อ'], '10544fe3-db86-4d63-800b-f8d70a3042f5', 0.85, true),
('rule_kw_044', 'Knife Sharpener Detection', ARRAY['หินลับมีด', 'แท่งลับมีด', 'ที่ลับมีดอัตโนมัติ', 'ลับมีด'], '815b00bb-d2c8-4748-8776-b9cecede60e0', 0.95, true),
('rule_kw_045', 'Food Tray Detection', ARRAY['ถาดหลุม', 'ถาดเสิร์ฟ', 'ถาดพลาสติก', 'ถาดสแตนเลส', 'ถาด'], '93a178c4-04d3-46a8-ab17-585687b91f2e', 0.85, true);
