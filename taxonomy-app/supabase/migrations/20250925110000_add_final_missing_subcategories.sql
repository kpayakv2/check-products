-- Migration: add_final_missing_subcategories
-- Purpose: Add the last 8 missing subcategories from original dataset
-- Complete the taxonomy to match original 47 subcategories

BEGIN;

-- Add remaining missing subcategories from เครื่องมือ_ฮาร์ดแวร์
INSERT INTO taxonomy_nodes (code, short_code, name_th, name_en, level, parent_id, sort_order, keywords, is_active) VALUES
('cat_001_006', 'HW_PLUMB', 'อุปกรณ์ประปา', 'Plumbing Supplies', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_001'), 6, 
 ARRAY['ประปา', 'plumbing', 'ก๊อก', 'พลาสติก'], true),

('cat_001_007', 'HW_ABRAS', 'วัสดุขัดผิว', 'Abrasive Materials', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_001'), 7, 
 ARRAY['ขัดผิว', 'abrasive', 'กระดาษทราย', 'สักหลาด'], true);

-- Add remaining missing subcategories from เครื่องครัว
INSERT INTO taxonomy_nodes (code, short_code, name_th, name_en, level, parent_id, sort_order, keywords, is_active) VALUES
('cat_003_009', 'KIT_GLASS', 'แก้วน้ำ/ถ้วย', 'Glasses & Cups', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_003'), 9, 
 ARRAY['แก้ว', 'ถ้วย', 'glass', 'cup', 'ขีดแดง'], true),

('cat_003_010', 'KIT_STRAIN', 'กระชอน/ที่กรอง', 'Strainers & Filters', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_003'), 10, 
 ARRAY['กระชอน', 'ที่กรอง', 'strainer', 'filter', 'พลาสติก'], true),

('cat_003_011', 'KIT_SHARP', 'อุปกรณ์ลับมีด', 'Knife Sharpeners', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_003'), 11, 
 ARRAY['ลับมีด', 'knife sharpener', 'โค้ง'], true),

('cat_003_012', 'KIT_PAN', 'กระทะ', 'Pans', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_003'), 12, 
 ARRAY['กระทะ', 'pan', 'ม้วนขอบ'], true),

('cat_003_013', 'KIT_FUNNEL', 'กรวย', 'Funnels', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_003'), 13, 
 ARRAY['กรวย', 'funnel', 'เลส', 'พลาสติก'], true),

('cat_003_014', 'KIT_PREP', 'อุปกรณ์ประกอบอาหาร/ตะแกรงครัว', 'Food Prep & Kitchen Strainers', 1, 
 (SELECT id FROM taxonomy_nodes WHERE code = 'cat_003'), 14, 
 ARRAY['ประกอบอาหาร', 'food prep', 'ขูดมะละกอ', 'ตะแกรงตักทอด', 'เลส'], true);

-- Fix system_settings table structure (the integrity check showed it was missing columns)
ALTER TABLE system_settings ADD COLUMN IF NOT EXISTS setting_key TEXT;
ALTER TABLE system_settings ADD COLUMN IF NOT EXISTS setting_value TEXT; 
ALTER TABLE system_settings ADD COLUMN IF NOT EXISTS description TEXT;

-- Add unique constraint on setting_key for proper upsert (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'system_settings_setting_key_key'
    ) THEN
        ALTER TABLE system_settings ADD CONSTRAINT system_settings_setting_key_key UNIQUE (setting_key);
    END IF;
END
$$;

-- Update existing system_settings record or insert if empty
INSERT INTO system_settings (setting_key, setting_value, description)
VALUES ('taxonomy_version', '2.0', 'Current taxonomy system version')
ON CONFLICT (setting_key) DO UPDATE SET 
    setting_value = EXCLUDED.setting_value,
    description = EXCLUDED.description;

INSERT INTO system_settings (setting_key, setting_value, description)
VALUES ('total_categories', '59', 'Total number of taxonomy categories (12 main + 47 sub)')
ON CONFLICT (setting_key) DO UPDATE SET 
    setting_value = EXCLUDED.setting_value,
    description = EXCLUDED.description;

INSERT INTO system_settings (setting_key, setting_value, description)
VALUES ('last_updated', NOW()::text, 'Last time taxonomy was updated')
ON CONFLICT (setting_key) DO UPDATE SET 
    setting_value = NOW()::text,
    description = EXCLUDED.description;

COMMIT;
