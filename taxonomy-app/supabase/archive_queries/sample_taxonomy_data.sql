-- Insert sample taxonomy data with keywords for testing
INSERT INTO taxonomy_nodes (code, name_th, name_en, keywords, level, is_active) VALUES 
('FASHION', 'แฟชั่น', 'Fashion', ARRAY['เสื้อ', 'กางเกง', 'รองเท้า', 'แฟชั่น', 'เสื้อผ้า'], 0, true),
('FASHION_TOPS', 'เสื้อผ้าบน', 'Tops', ARRAY['เสื้อ', 'เสื้อยืด', 'เสื้อเชิ้ต', 'เสื้อกล้าม', 'บลาวส์'], 1, true),
('FASHION_BOTTOMS', 'เสื้อผ้าล่าง', 'Bottoms', ARRAY['กางเกง', 'กระโปรง', 'ขาสั้น', 'ขายาว', 'ยีนส์'], 1, true),
('SHOES', 'รองเท้า', 'Shoes', ARRAY['รองเท้า', 'รองเท้าผ้าใบ', 'รองเท้าหนัง', 'รองเท้าส้นสูง'], 1, true),
('ELECTRONICS', 'อิเล็กทรอนิกส์', 'Electronics', ARRAY['มือถือ', 'คอมพิวเตอร์', 'แล็ปท็อป', 'หูฟัง', 'อิเล็กทรอนิกส์'], 0, true);
