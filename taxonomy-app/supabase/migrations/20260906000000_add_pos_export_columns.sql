-- คอลัมน์ที่ไฟล์นำเข้า POS ต้องการ แต่ products ยังไม่มี (2026-09-06)
--
-- ไฟล์ที่ POS กินได้มี 9 คอลัมน์ ตอนนี้ตารางตอบได้แค่ 5
-- (ชื่อ, รหัส, ราคา, และหมวด 2 ชั้นผ่าน category_id) ที่เหลือต้องเพิ่มเข้ามา
--
-- pos_exported_at ไม่ได้มีไว้ทำรายงาน แต่เป็นตัวกันส่งออกซ้ำ:
-- ของที่ส่งเข้า POS แล้วก็คือ "มีในสตอกแล้ว" ไม่ควรโผล่ในไฟล์รอบหน้าอีก
--
-- สเปก: docs/specs/2026-09-06-pos-export.md

ALTER TABLE products
    ADD COLUMN IF NOT EXISTS barcode         TEXT,
    ADD COLUMN IF NOT EXISTS unit_name       TEXT,
    ADD COLUMN IF NOT EXISTS vat_flag        TEXT NOT NULL DEFAULT 'V',
    ADD COLUMN IF NOT EXISTS pos_exported_at TIMESTAMPTZ;

COMMENT ON COLUMN products.barcode IS
    'บาร์โค้ดสำหรับ POS — คนละอย่างกับ sku ถ้าว่าง ตัวส่งออกจะใช้ sku แทนตามไฟล์ต้นแบบ';
COMMENT ON COLUMN products.unit_name IS
    'หน่วยนับที่ POS แสดง เช่น ชิ้น / คู่ / กล่อง';
COMMENT ON COLUMN products.vat_flag IS
    'ธง VAT ในไฟล์นำเข้า POS — ไฟล์ต้นแบบทุกแถวเป็น V จึงตั้ง default ไว้';
COMMENT ON COLUMN products.pos_exported_at IS
    'เวลาที่ถูกใส่ในไฟล์นำเข้า POS ครั้งล่าสุด — NULL แปลว่ายังไม่เคยส่งออก ใช้กันส่งซ้ำ';

-- ตัวส่งออกคัดด้วยสามเงื่อนไขนี้เสมอ (source + dedup_bucket + ยังไม่ส่งออก)
-- partial index เพราะแถวที่ส่งออกไปแล้วไม่ต้องอยู่ในดัชนี
CREATE INDEX IF NOT EXISTS idx_products_pending_pos_export
    ON products ((metadata->>'source'), (metadata->>'dedup_bucket'))
    WHERE pos_exported_at IS NULL;
