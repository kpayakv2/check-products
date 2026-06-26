-- Quick test: Insert just a few categories for testing
INSERT INTO taxonomy_nodes (code, name_th, name_en, level, parent_id, sort_order, keywords, is_active) VALUES
('FASHION', 'เสื้อผ้าแฟชั่น', 'Fashion & Clothing', 0, NULL, 1, 
 ARRAY['เสื้อ', 'เสื้อผ้า', 'กางเกง', 'แฟชั่น', 'เสื้อยืด', 'เสื้อเชิ้ต'], true),
 
('ELECTRONICS', 'อุปกรณ์อิเล็กทรอนิกส์', 'Electronics', 0, NULL, 2,
 ARRAY['มือถือ', 'โทรศัพท์', 'คอมพิวเตอร์', 'อิเล็กทรอนิกส์', 'หูฟัง'], true),
 
('KITCHEN', 'เครื่องครัว', 'Kitchen & Cooking', 0, NULL, 3,
 ARRAY['เครื่องครัว', 'หม้อ', 'กระทะ', 'มีด', 'ช้อน', 'ส้อม', 'จาน'], true);
