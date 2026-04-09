-- Thai Product Taxonomy Manager - Complete Dataset
-- Generated for manual installation
-- Execute this entire file in Supabase SQL Editor

-- ==================================================
-- Schema (Database Structure)
-- Creates tables, indexes, and constraints
-- ==================================================

-- เปิดใช้งาน extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ตาราง taxonomy_nodes (โหนดหมวดหมู่สินค้า)
CREATE TABLE taxonomy_nodes (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name_th TEXT NOT NULL,
    name_en TEXT,
    description TEXT,
    parent_id UUID REFERENCES taxonomy_nodes(id) ON DELETE CASCADE,
    level INTEGER DEFAULT 0,
    sort_order INTEGER DEFAULT 0,
    path TEXT, -- materialized path เช่น /1/2/3/
    keywords TEXT[], -- array ของ keywords สำหรับการค้นหา
    metadata JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_by UUID,
    updated_by UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ตาราง synonyms (กลุ่มคำพ้องความหมาย)
CREATE TABLE synonyms (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name TEXT NOT NULL, -- ชื่อกลุ่ม synonym
    description TEXT,
    category_id UUID REFERENCES taxonomy_nodes(id) ON DELETE SET NULL,
    is_active BOOLEAN DEFAULT true,
    created_by UUID,
    updated_by UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ตาราง synonym_terms (คำในกลุ่ม synonym)
CREATE TABLE synonym_terms (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    synonym_id UUID REFERENCES synonyms(id) ON DELETE CASCADE,
    term TEXT NOT NULL,
    is_primary BOOLEAN DEFAULT false, -- คำหลักของกลุ่ม
    confidence_score FLOAT DEFAULT 1.0,
    source TEXT DEFAULT 'manual', -- manual, auto, imported, ml
    language TEXT DEFAULT 'th', -- th, en, mixed
    is_verified BOOLEAN DEFAULT false,
    created_by UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(synonym_id, term)
);

-- ตาราง synonym_category_map (การเชื่อมโยง synonym กับหมวดหมู่)
CREATE TABLE synonym_category_map (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    synonym_id UUID REFERENCES synonyms(id) ON DELETE CASCADE,
    category_id UUID REFERENCES taxonomy_nodes(id) ON DELETE CASCADE,
    weight FLOAT DEFAULT 1.0, -- น้ำหนักความสำคัญ
    created_by UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(synonym_id, category_id)
);

-- ตาราง products (สินค้า)
CREATE TABLE products (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name_th TEXT NOT NULL,
    name_en TEXT,
    description TEXT,
    category_id UUID REFERENCES taxonomy_nodes(id),
    brand TEXT,
    model TEXT,
    sku TEXT UNIQUE,
    price DECIMAL(12,2),
    embedding vector(768), -- เปลี่ยนเป็น 768 dimensions
    keywords TEXT[], -- keywords ที่สกัดจากชื่อและคำอธิบาย
    metadata JSONB DEFAULT '{}',
    status TEXT DEFAULT 'pending', -- pending, approved, rejected, draft
    confidence_score FLOAT, -- คะแนนความมั่นใจในการจัดหมวดหมู่
    import_batch_id UUID, -- อ้างอิงถึง batch การนำเข้า
    reviewed_by UUID,
    reviewed_at TIMESTAMP WITH TIME ZONE,
    created_by UUID,
    updated_by UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ตาราง keyword_rules (กฎการจัดหมวดหมู่ตาม keyword)
CREATE TABLE keyword_rules (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    keywords TEXT[] NOT NULL, -- keywords ที่ใช้ในการจับคู่
    category_id UUID REFERENCES taxonomy_nodes(id) ON DELETE CASCADE,
    priority INTEGER DEFAULT 0, -- ลำดับความสำคัญ (สูงกว่า = สำคัญกว่า)
    match_type TEXT DEFAULT 'contains', -- contains, exact, regex, fuzzy
    confidence_score FLOAT DEFAULT 0.8,
    is_active BOOLEAN DEFAULT true,
    created_by UUID,
    updated_by UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ตาราง product_category_suggestions (คำแนะนำหมวดหมู่สำหรับสินค้า)
CREATE TABLE product_category_suggestions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    product_id UUID REFERENCES products(id) ON DELETE CASCADE,
    suggested_category_id UUID REFERENCES taxonomy_nodes(id) ON DELETE CASCADE,
    confidence_score FLOAT NOT NULL,
    suggestion_method TEXT NOT NULL, -- keyword_rule, ml_model, similarity, manual
    rule_id UUID REFERENCES keyword_rules(id) ON DELETE SET NULL,
    metadata JSONB DEFAULT '{}', -- เก็บข้อมูลเพิ่มเติมเช่น matched keywords
    is_accepted BOOLEAN,
    reviewed_by UUID,
    reviewed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ตาราง imports (การนำเข้าข้อมูล)
CREATE TABLE imports (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    file_name TEXT,
    file_size BIGINT,
    file_type TEXT, -- csv, xlsx, json
    total_records INTEGER DEFAULT 0,
    processed_records INTEGER DEFAULT 0,
    success_records INTEGER DEFAULT 0,
    error_records INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending', -- pending, processing, completed, failed
    error_details JSONB,
    metadata JSONB DEFAULT '{}',
    created_by UUID,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ตาราง audit_logs (บันทึกการเปลี่ยนแปลง)
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_name TEXT NOT NULL,
    record_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('INSERT', 'UPDATE', 'DELETE')),
    old_values JSONB,
    new_values JSONB,
    changed_fields TEXT[],
    user_id UUID REFERENCES auth.users(id),
    user_agent TEXT,
    ip_address INET,
    session_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ตาราง regex_rules (กฎการตรวจสอบด้วย regex)
CREATE TABLE regex_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    description TEXT,
    pattern TEXT NOT NULL,
    flags TEXT DEFAULT 'gi',
    category_id UUID REFERENCES taxonomy_nodes(id),
    priority INTEGER DEFAULT 0,
    confidence_score DECIMAL(3,2) DEFAULT 0.7 CHECK (confidence_score >= 0 AND confidence_score <= 1),
    is_active BOOLEAN DEFAULT true,
    test_cases TEXT[] DEFAULT '{}',
    created_by UUID REFERENCES auth.users(id),
    updated_by UUID REFERENCES auth.users(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ตาราง system_settings (การตั้งค่าระบบ)
CREATE TABLE system_settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    search JSONB DEFAULT '{
        "vectorSearchEnabled": true,
        "textSearchEnabled": true,
        "hybridSearchEnabled": true,
        "defaultSearchType": "hybrid",
        "maxResults": 50,
        "confidenceThreshold": 0.5
    }',
    processing JSONB DEFAULT '{
        "batchSize": 100,
        "maxConcurrentJobs": 5,
        "retryAttempts": 3,
        "timeoutSeconds": 30
    }',
    ai JSONB DEFAULT '{
        "embeddingModel": "paraphrase-multilingual-MiniLM-L12-v2",
        "apiProvider": "openai",
        "maxTokens": 4000,
        "temperature": 0.1
    }',
    ui JSONB DEFAULT '{
        "theme": "light",
        "language": "th",
        "itemsPerPage": 20,
        "enableAnimations": true
    }',
    updated_by UUID REFERENCES auth.users(id),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ตาราง product_attributes (คุณสมบัติสินค้า)
CREATE TABLE product_attributes (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    product_id UUID REFERENCES products(id) ON DELETE CASCADE,
    attribute_name TEXT NOT NULL,
    attribute_value TEXT NOT NULL,
    attribute_type TEXT DEFAULT 'text', -- text, number, boolean, date
    created_by UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ตาราง review_history (ประวัติการตรวจสอบ)
CREATE TABLE review_history (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    product_id UUID REFERENCES products(id) ON DELETE CASCADE,
    reviewer_id UUID,
    action TEXT NOT NULL, -- approved, rejected, modified, category_changed
    old_category_id UUID REFERENCES taxonomy_nodes(id),
    new_category_id UUID REFERENCES taxonomy_nodes(id),
    comments TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ตาราง similarity_matches (ผลการจับคู่ความคล้าย)
CREATE TABLE similarity_matches (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    product_a_id UUID REFERENCES products(id) ON DELETE CASCADE,
    product_b_id UUID REFERENCES products(id) ON DELETE CASCADE,
    similarity_score FLOAT NOT NULL,
    match_type TEXT DEFAULT 'semantic', -- semantic, exact, fuzzy, keyword
    algorithm TEXT DEFAULT 'cosine', -- cosine, euclidean, jaccard
    is_duplicate BOOLEAN DEFAULT false,
    reviewed BOOLEAN DEFAULT false,
    reviewed_by UUID,
    reviewed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(product_a_id, product_b_id)
);

-- Indexes สำหรับประสิทธิภาพ

-- Taxonomy nodes indexes
CREATE INDEX idx_taxonomy_nodes_parent_id ON taxonomy_nodes(parent_id);
CREATE INDEX idx_taxonomy_nodes_level ON taxonomy_nodes(level);
CREATE INDEX idx_taxonomy_nodes_path ON taxonomy_nodes(path);
CREATE INDEX idx_taxonomy_nodes_keywords ON taxonomy_nodes USING GIN(keywords); -- GIN index สำหรับ array search
CREATE INDEX idx_taxonomy_nodes_active ON taxonomy_nodes(is_active) WHERE is_active = true;

-- Synonyms indexes
CREATE INDEX idx_synonyms_category_id ON synonyms(category_id);
CREATE INDEX idx_synonyms_active ON synonyms(is_active) WHERE is_active = true;

-- Synonym terms indexes
CREATE INDEX idx_synonym_terms_synonym_id ON synonym_terms(synonym_id);
CREATE INDEX idx_synonym_terms_term ON synonym_terms(term);
CREATE INDEX idx_synonym_terms_primary ON synonym_terms(is_primary) WHERE is_primary = true;
CREATE INDEX idx_synonym_terms_verified ON synonym_terms(is_verified);

-- Synonym category map indexes
CREATE INDEX idx_synonym_category_map_synonym ON synonym_category_map(synonym_id);
CREATE INDEX idx_synonym_category_map_category ON synonym_category_map(category_id);

-- Products indexes
CREATE INDEX idx_products_category_id ON products(category_id);
CREATE INDEX idx_products_status ON products(status);
CREATE INDEX idx_products_sku ON products(sku) WHERE sku IS NOT NULL;
CREATE INDEX idx_products_brand ON products(brand) WHERE brand IS NOT NULL;
CREATE INDEX idx_products_keywords ON products USING GIN(keywords); -- GIN index สำหรับ keywords array
CREATE INDEX idx_products_embedding ON products USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100); -- IVFFlat index สำหรับ vector search
CREATE INDEX idx_products_import_batch ON products(import_batch_id) WHERE import_batch_id IS NOT NULL;
CREATE INDEX idx_products_created_at ON products(created_at);

-- Keyword rules indexes
CREATE INDEX idx_keyword_rules_category ON keyword_rules(category_id);
CREATE INDEX idx_keyword_rules_keywords ON keyword_rules USING GIN(keywords);
CREATE INDEX idx_keyword_rules_active ON keyword_rules(is_active) WHERE is_active = true;
CREATE INDEX idx_keyword_rules_priority ON keyword_rules(priority DESC);

-- Product category suggestions indexes
CREATE INDEX idx_product_suggestions_product ON product_category_suggestions(product_id);
CREATE INDEX idx_product_suggestions_category ON product_category_suggestions(suggested_category_id);
CREATE INDEX idx_product_suggestions_confidence ON product_category_suggestions(confidence_score DESC);
CREATE INDEX idx_product_suggestions_method ON product_category_suggestions(suggestion_method);
CREATE INDEX idx_product_suggestions_accepted ON product_category_suggestions(is_accepted) WHERE is_accepted IS NOT NULL;

-- Imports indexes
CREATE INDEX idx_imports_status ON imports(status);
CREATE INDEX idx_imports_created_by ON imports(created_by);
CREATE INDEX idx_imports_created_at ON imports(created_at DESC);

-- Audit logs indexes
CREATE INDEX idx_audit_logs_table_record ON audit_logs(table_name, record_id);
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at DESC);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);

-- Product attributes indexes
CREATE INDEX idx_product_attributes_product_id ON product_attributes(product_id);
CREATE INDEX idx_product_attributes_name ON product_attributes(attribute_name);

-- Review history indexes
CREATE INDEX idx_review_history_product_id ON review_history(product_id);
CREATE INDEX idx_review_history_reviewer ON review_history(reviewer_id) WHERE reviewer_id IS NOT NULL;
CREATE INDEX idx_review_history_action ON review_history(action);
CREATE INDEX idx_review_history_created_at ON review_history(created_at DESC);

-- Similarity matches indexes
CREATE INDEX idx_similarity_matches_product_a ON similarity_matches(product_a_id);
CREATE INDEX idx_similarity_matches_product_b ON similarity_matches(product_b_id);
CREATE INDEX idx_similarity_matches_score ON similarity_matches(similarity_score DESC);
CREATE INDEX idx_similarity_matches_type ON similarity_matches(match_type);
CREATE INDEX idx_similarity_matches_duplicate ON similarity_matches(is_duplicate) WHERE is_duplicate = true;
CREATE INDEX idx_similarity_matches_reviewed ON similarity_matches(reviewed);

-- Functions สำหรับ updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers สำหรับ auto-update timestamps
CREATE TRIGGER update_taxonomy_nodes_updated_at BEFORE UPDATE ON taxonomy_nodes FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_synonyms_updated_at BEFORE UPDATE ON synonyms FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_products_updated_at BEFORE UPDATE ON products FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_keyword_rules_updated_at BEFORE UPDATE ON keyword_rules FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function สำหรับ audit logging
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO audit_logs (table_name, record_id, action, old_values, user_id)
        VALUES (TG_TABLE_NAME, OLD.id, 'DELETE', row_to_json(OLD), current_setting('app.current_user_id', true)::UUID);
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_logs (table_name, record_id, action, old_values, new_values, user_id)
        VALUES (TG_TABLE_NAME, NEW.id, 'UPDATE', row_to_json(OLD), row_to_json(NEW), current_setting('app.current_user_id', true)::UUID);
        RETURN NEW;
    ELSIF TG_OP = 'INSERT' THEN
        INSERT INTO audit_logs (table_name, record_id, action, new_values, user_id)
        VALUES (TG_TABLE_NAME, NEW.id, 'INSERT', row_to_json(NEW), current_setting('app.current_user_id', true)::UUID);
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Audit triggers
CREATE TRIGGER audit_taxonomy_nodes AFTER INSERT OR UPDATE OR DELETE ON taxonomy_nodes FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
CREATE TRIGGER audit_synonyms AFTER INSERT OR UPDATE OR DELETE ON synonyms FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
CREATE TRIGGER audit_products AFTER INSERT OR UPDATE OR DELETE ON products FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
CREATE TRIGGER audit_keyword_rules AFTER INSERT OR UPDATE OR DELETE ON keyword_rules FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

-- Row Level Security (RLS) policies
ALTER TABLE taxonomy_nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE synonyms ENABLE ROW LEVEL SECURITY;
ALTER TABLE synonym_terms ENABLE ROW LEVEL SECURITY;
ALTER TABLE synonym_category_map ENABLE ROW LEVEL SECURITY;
ALTER TABLE products ENABLE ROW LEVEL SECURITY;
ALTER TABLE keyword_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE product_category_suggestions ENABLE ROW LEVEL SECURITY;
ALTER TABLE imports ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE product_attributes ENABLE ROW LEVEL SECURITY;
ALTER TABLE review_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE similarity_matches ENABLE ROW LEVEL SECURITY;

-- Create roles
DO $$
BEGIN
    -- Create roles if they don't exist
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'taxonomy_reader') THEN
        CREATE ROLE taxonomy_reader;
    END IF;
    
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'taxonomy_editor') THEN
        CREATE ROLE taxonomy_editor;
    END IF;
    
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'taxonomy_admin') THEN
        CREATE ROLE taxonomy_admin;
    END IF;
END
$$;

-- Grant permissions to roles

-- Reader role - read-only access
GRANT USAGE ON SCHEMA public TO taxonomy_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO taxonomy_reader;

-- Editor role - can modify data but not structure
GRANT taxonomy_reader TO taxonomy_editor;
GRANT INSERT, UPDATE, DELETE ON taxonomy_nodes TO taxonomy_editor;
GRANT INSERT, UPDATE, DELETE ON synonyms TO taxonomy_editor;
GRANT INSERT, UPDATE, DELETE ON synonym_terms TO taxonomy_editor;
GRANT INSERT, UPDATE, DELETE ON synonym_category_map TO taxonomy_editor;
GRANT INSERT, UPDATE, DELETE ON products TO taxonomy_editor;
GRANT INSERT, UPDATE, DELETE ON keyword_rules TO taxonomy_editor;
GRANT INSERT, UPDATE, DELETE ON product_category_suggestions TO taxonomy_editor;
GRANT INSERT, UPDATE, DELETE ON product_attributes TO taxonomy_editor;
GRANT INSERT, UPDATE, DELETE ON review_history TO taxonomy_editor;
GRANT INSERT, UPDATE, DELETE ON similarity_matches TO taxonomy_editor;
GRANT INSERT, UPDATE ON imports TO taxonomy_editor;

-- Admin role - full access
GRANT taxonomy_editor TO taxonomy_admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO taxonomy_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO taxonomy_admin;

-- RLS Policies

-- Taxonomy nodes policies
CREATE POLICY "taxonomy_nodes_read" ON taxonomy_nodes FOR SELECT USING (true);
CREATE POLICY "taxonomy_nodes_insert" ON taxonomy_nodes FOR INSERT WITH CHECK (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));
CREATE POLICY "taxonomy_nodes_update" ON taxonomy_nodes FOR UPDATE USING (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));
CREATE POLICY "taxonomy_nodes_delete" ON taxonomy_nodes FOR DELETE USING (auth.role() = 'taxonomy_admin');

-- Synonyms policies
CREATE POLICY "synonyms_read" ON synonyms FOR SELECT USING (true);
CREATE POLICY "synonyms_insert" ON synonyms FOR INSERT WITH CHECK (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));
CREATE POLICY "synonyms_update" ON synonyms FOR UPDATE USING (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));
CREATE POLICY "synonyms_delete" ON synonyms FOR DELETE USING (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));

-- Synonym terms policies
CREATE POLICY "synonym_terms_read" ON synonym_terms FOR SELECT USING (true);
CREATE POLICY "synonym_terms_insert" ON synonym_terms FOR INSERT WITH CHECK (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));
CREATE POLICY "synonym_terms_update" ON synonym_terms FOR UPDATE USING (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));
CREATE POLICY "synonym_terms_delete" ON synonym_terms FOR DELETE USING (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));

-- Products policies
CREATE POLICY "products_read" ON products FOR SELECT USING (true);
CREATE POLICY "products_insert" ON products FOR INSERT WITH CHECK (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));
CREATE POLICY "products_update" ON products FOR UPDATE USING (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));
CREATE POLICY "products_delete" ON products FOR DELETE USING (auth.role() = 'taxonomy_admin');

-- Audit logs policies (admin only)
CREATE POLICY "audit_logs_read" ON audit_logs FOR SELECT USING (auth.role() = 'taxonomy_admin');
CREATE POLICY "audit_logs_insert" ON audit_logs FOR INSERT WITH CHECK (true); -- System can always insert

-- Sample data สำหรับ taxonomy nodes
INSERT INTO taxonomy_nodes (name_th, name_en, level, sort_order, path, keywords) VALUES
('อิเล็กทรอนิกส์', 'Electronics', 0, 1, '/1/', ARRAY['อิเล็กทรอนิกส์', 'electronics', 'เครื่องใช้ไฟฟ้า', 'gadget']),
('เสื้อผ้าแฟชั่น', 'Fashion & Clothing', 0, 2, '/2/', ARRAY['เสื้อผ้า', 'แฟชั่น', 'fashion', 'clothing', 'apparel']),
('อาหารและเครื่องดื่ม', 'Food & Beverages', 0, 3, '/3/', ARRAY['อาหาร', 'เครื่องดื่ม', 'food', 'beverage', 'drink']),
('บ้านและสวน', 'Home & Garden', 0, 4, '/4/', ARRAY['บ้าน', 'สวน', 'home', 'garden', 'household']),
('กีฬาและกิจกรรมกลางแจ้ง', 'Sports & Outdoors', 0, 5, '/5/', ARRAY['กีฬา', 'outdoor', 'sports', 'exercise', 'fitness']);

-- Sample synonyms groups
INSERT INTO synonyms (name, description) VALUES
('โทรศัพท์มือถือ', 'กลุ่มคำที่เกี่ยวกับโทรศัพท์มือถือ'),
('คอมพิวเตอร์', 'กลุ่มคำที่เกี่ยวกับคอมพิวเตอร์'),
('เสื้อยืด', 'กลุ่มคำที่เกี่ยวกับเสื้อยืด');

-- Sample synonym terms
INSERT INTO synonym_terms (synonym_id, term, is_primary, confidence_score, source) VALUES
((SELECT id FROM synonyms WHERE name = 'โทรศัพท์มือถือ'), 'โทรศัพท์มือถือ', true, 1.0, 'manual'),
((SELECT id FROM synonyms WHERE name = 'โทรศัพท์มือถือ'), 'มือถือ', false, 1.0, 'manual'),
((SELECT id FROM synonyms WHERE name = 'โทรศัพท์มือถือ'), 'สมาร์ทโฟน', false, 0.9, 'manual'),
((SELECT id FROM synonyms WHERE name = 'โทรศัพท์มือถือ'), 'smartphone', false, 0.9, 'manual'),
((SELECT id FROM synonyms WHERE name = 'คอมพิวเตอร์'), 'คอมพิวเตอร์', true, 1.0, 'manual'),
((SELECT id FROM synonyms WHERE name = 'คอมพิวเตอร์'), 'คอม', false, 0.8, 'manual'),
((SELECT id FROM synonyms WHERE name = 'คอมพิวเตอร์'), 'computer', false, 1.0, 'manual'),
((SELECT id FROM synonyms WHERE name = 'เสื้อยืด'), 'เสื้อยืด', true, 1.0, 'manual'),
((SELECT id FROM synonyms WHERE name = 'เสื้อยืด'), 'เสื้อทีเชิร์ต', false, 1.0, 'manual'),
((SELECT id FROM synonyms WHERE name = 'เสื้อยืด'), 't-shirt', false, 1.0, 'manual');

-- Sample keyword rules
INSERT INTO keyword_rules (name, description, keywords, category_id, priority, confidence_score) VALUES
('iPhone Detection', 'ตรวจจับสินค้า iPhone', ARRAY['iphone', 'ไอโฟน'], (SELECT id FROM taxonomy_nodes WHERE name_th = 'อิเล็กทรอนิกส์'), 10, 0.95),
('Samsung Detection', 'ตรวจจับสินค้า Samsung', ARRAY['samsung', 'galaxy', 'ซัมซุง'], (SELECT id FROM taxonomy_nodes WHERE name_th = 'อิเล็กทรอนิกส์'), 9, 0.9),
('T-Shirt Detection', 'ตรวจจับเสื้อยืด', ARRAY['เสื้อยืด', 'เสื้อทีเชิร์ต', 't-shirt', 'tee'], (SELECT id FROM taxonomy_nodes WHERE name_th = 'เสื้อผ้าแฟชั่น'), 8, 0.85);


-- ==================================================
-- Taxonomy Dataset
-- 12 main categories + 47 sub-categories
-- ==================================================

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


-- ==================================================
-- Synonyms Dataset
-- 26 lemmas + 180+ synonym terms
-- ==================================================

-- Thai Product Synonyms Dataset
-- Generated from รายการสินค้าพร้อมหมวดหมู่_AI.txt
-- Lemma + Terms structure for enhanced synonym management

-- Insert Synonym Lemmas (Main concept groups)
INSERT INTO synonym_lemmas (id, name_th, name_en, description, category_id, is_verified) VALUES

-- เครื่องมือ_ฮาร์ดแวร์ Synonyms
('lemma_001', 'เครื่องมือทำสวน', 'Gardening Tools', 'อุปกรณ์สำหรับทำสวนและดูแลต้นไม้', 'cat_001_001', true),
('lemma_002', 'เครื่องมือช่าง', 'Hand Tools', 'เครื่องมือช่างทั่วไป', 'cat_001_002', true),
('lemma_003', 'อุปกรณ์ทาสี', 'Painting Supplies', 'อุปกรณ์สำหรับทาสีและตกแต่ง', 'cat_001_003', true),
('lemma_004', 'อุปกรณ์ไฟฟ้า', 'Electrical Tools', 'เครื่องมือและอุปกรณ์ไฟฟ้า', 'cat_001_005', true),

-- ของใช้ในบ้าน Synonyms
('lemma_005', 'ภาชนะใส่น้ำ', 'Water Containers', 'ภาชนะสำหรับใส่น้ำและเครื่องดื่ม', 'cat_002_001', true),
('lemma_006', 'ภาชนะเก็บของ', 'Storage Containers', 'กล่อง ตะกร้า และภาชนะเก็บของต่างๆ', 'cat_002_003', true),
('lemma_007', 'ถังและกะละมัง', 'Buckets and Basins', 'ภาชนะขนาดใหญ่สำหรับใส่น้ำและของใช้', 'cat_002_006', true),
('lemma_008', 'อุปกรณ์จัดระเบียบ', 'Organization Tools', 'อุปกรณ์สำหรับจัดระเบียบในบ้าน', 'cat_002_011', true),

-- เครื่องครัว Synonyms
('lemma_009', 'ภาชนะครัว', 'Kitchen Containers', 'ภาชนะสำหรับใส่เครื่องปรุงและอาหาร', 'cat_003_001', true),
('lemma_010', 'อุปกรณ์ตัดหั่น', 'Cutting Tools', 'มีดและเขียงสำหรับเตรียมอาหาร', 'cat_003_003', true),
('lemma_011', 'อุปกรณ์รับประทาน', 'Eating Utensils', 'ช้อน ส้อม และอุปกรณ์รับประทานอาหาร', 'cat_003_004', true),
('lemma_012', 'อุปกรณ์ตวงวัด', 'Measuring Tools', 'เครื่องมือสำหรับตวงและวัดในครัว', 'cat_003_002', true),

-- อุปกรณ์ทำความสะอาด Synonyms
('lemma_013', 'อุปกรณ์ขัดล้าง', 'Scrubbing Tools', 'ฟองน้ำ ใยขัด และอุปกรณ์ขัดล้าง', 'cat_004_001', true),
('lemma_014', 'แปรงทำความสะอาด', 'Cleaning Brushes', 'แปรงสำหรับทำความสะอาดต่างๆ', 'cat_004_002', true),
('lemma_015', 'ผ้าทำความสะอาด', 'Cleaning Cloths', 'ผ้าสำหรับเช็ดและทำความสะอาด', 'cat_004_003', true),

-- เครื่องเขียน Synonyms
('lemma_016', 'อุปกรณ์ตัด', 'Cutting Supplies', 'กรรไกรและอุปกรณ์ตัดต่างๆ', 'cat_005_001', true),
('lemma_017', 'อุปกรณ์ติด', 'Adhesive Supplies', 'กาวและเทปสำหรับติดและยึด', 'cat_005_002', true),

-- ของเล่น Synonyms
('lemma_018', 'ของเล่นยานพาหนะ', 'Vehicle Toys', 'รถและยานพาหนะของเล่น', 'cat_006_001', true),
('lemma_019', 'ของเล่นอาวุธ', 'Weapon Toys', 'ปืนและอาวุธของเล่น', 'cat_006_002', true),

-- ผลิตภัณฑ์ส่วนบุคคล Synonyms
('lemma_020', 'ผลิตภัณฑ์อาบน้ำ', 'Bathing Products', 'สบู่และผลิตภัณฑ์อาบน้ำ', 'cat_007_003', true),
('lemma_021', 'อุปกรณ์โกนหนวด', 'Shaving Supplies', 'อุปกรณ์สำหรับโกนหนวดและตกแต่ง', 'cat_007_002', true),

-- สัตว์เลี้ยง Synonyms
('lemma_022', 'อุปกรณ์สัตว์เลี้ยง', 'Pet Accessories', 'อุปกรณ์สำหรับสัตว์เลี้ยง', 'cat_009_001', true),

-- เครื่องใช้ไฟฟ้า Synonyms
('lemma_023', 'อุปกรณ์ไฟฟ้าพกพา', 'Portable Electronics', 'อุปกรณ์ไฟฟ้าขนาดเล็กพกพา', 'cat_010_001', true),

-- แม่และเด็ก Synonyms
('lemma_024', 'อุปกรณ์ให้นมเด็ก', 'Baby Feeding', 'อุปกรณ์สำหรับให้นมและดูแลเด็ก', 'cat_011_001', true),

-- เบ็ดเตล็ด Synonyms
('lemma_025', 'อุปกรณ์ความงาม', 'Beauty Accessories', 'อุปกรณ์ตกแต่งและความงาม', 'cat_012_002', true),
('lemma_026', 'วัสดุตกแต่ง', 'Decorative Materials', 'วัสดุสำหรับตกแต่งและประดับ', 'cat_012_003', true);

-- Insert Synonym Terms for each Lemma
INSERT INTO synonym_terms (lemma_id, term, language, is_primary, confidence_score, usage_count) VALUES

-- เครื่องมือทำสวน (lemma_001)
('lemma_001', 'คราด', 'th', true, 0.95, 5),
('lemma_001', 'คราดมือเสือ', 'th', false, 0.90, 2),
('lemma_001', 'เสียม', 'th', true, 0.95, 3),
('lemma_001', 'เสียมมิด', 'th', false, 0.85, 1),
('lemma_001', 'สายยาง', 'th', true, 0.90, 8),
('lemma_001', 'สายยางฟ้า', 'th', false, 0.85, 2),
('lemma_001', 'กระถาง', 'th', true, 0.90, 12),
('lemma_001', 'กระถางต้นไม้', 'th', false, 0.85, 8),
('lemma_001', 'กรรไกรตัดหญ้า', 'th', false, 0.80, 1),

-- เครื่องมือช่าง (lemma_002)
('lemma_002', 'ค้อน', 'th', true, 0.95, 3),
('lemma_002', 'ค้อนหัวยาง', 'th', false, 0.90, 2),
('lemma_002', 'หกเหลี่ยม', 'th', true, 0.85, 4),
('lemma_002', 'เอ็น', 'th', true, 0.90, 15),
('lemma_002', 'เอ็นขาว', 'th', false, 0.85, 8),
('lemma_002', 'เอ็นแดง', 'th', false, 0.85, 4),
('lemma_002', 'เอ็นเขียว', 'th', false, 0.85, 8),
('lemma_002', 'ตลับเมตร', 'th', false, 0.80, 1),
('lemma_002', 'ระดับน้ำ', 'th', false, 0.80, 3),

-- อุปกรณ์ทาสี (lemma_003)
('lemma_003', 'แปรงทาสี', 'th', true, 0.95, 2),
('lemma_003', 'กระดาษทราย', 'th', true, 0.90, 3),
('lemma_003', 'จานจับกระดาษทราย', 'th', false, 0.75, 1),

-- อุปกรณ์ไฟฟ้า (lemma_004)
('lemma_004', 'หัวแร้ง', 'th', true, 0.90, 2),
('lemma_004', 'สายพ่วง', 'th', true, 0.85, 1),
('lemma_004', 'ปลั๊ก', 'th', true, 0.90, 3),
('lemma_004', 'ปลั๊กเสียบ', 'th', false, 0.85, 2),
('lemma_004', 'ไขควงวัดไฟ', 'th', false, 0.80, 1),

-- ภาชนะใส่น้ำ (lemma_005)
('lemma_005', 'ขัน', 'th', true, 0.95, 8),
('lemma_005', 'ขันน้ำ', 'th', false, 0.90, 3),
('lemma_005', 'ขันปั๊ม', 'th', false, 0.85, 6),
('lemma_005', 'ขันด้าม', 'th', false, 0.80, 1),
('lemma_005', 'กระติก', 'th', true, 0.95, 8),
('lemma_005', 'กระติกเหลี่ยม', 'th', false, 0.85, 3),
('lemma_005', 'กระบอกน้ำ', 'th', false, 0.80, 1),

-- ภาชนะเก็บของ (lemma_006)
('lemma_006', 'กล่อง', 'th', true, 0.95, 25),
('lemma_006', 'กล่องเหลี่ยม', 'th', false, 0.85, 8),
('lemma_006', 'กล่องล็อค', 'th', false, 0.90, 15),
('lemma_006', 'กล่องล้อ', 'th', false, 0.85, 4),
('lemma_006', 'เก๊ะลิ้นชัก', 'th', false, 0.80, 3),
('lemma_006', 'ตะกร้า', 'th', true, 0.95, 18),
('lemma_006', 'ตะกร้ากลม', 'th', false, 0.85, 6),
('lemma_006', 'ตะกร้าเหลี่ยม', 'th', false, 0.85, 4),
('lemma_006', 'กระจาด', 'th', true, 0.90, 12),

-- ถังและกะละมัง (lemma_007)
('lemma_007', 'ถัง', 'th', true, 0.95, 15),
('lemma_007', 'ถังน้ำ', 'th', false, 0.90, 8),
('lemma_007', 'ถังฝา', 'th', false, 0.85, 4),
('lemma_007', 'ถังปูน', 'th', false, 0.80, 6),
('lemma_007', 'กะละมัง', 'th', true, 0.90, 8),

-- อุปกรณ์จัดระเบียบ (lemma_008)
('lemma_008', 'ตะขอ', 'th', true, 0.85, 3),
('lemma_008', 'ตะขอตัวเอส', 'th', false, 0.80, 3),
('lemma_008', 'ชั้นวาง', 'th', true, 0.90, 6),
('lemma_008', 'ตะแกรง', 'th', true, 0.85, 8),

-- ภาชนะครัว (lemma_009)
('lemma_009', 'ขวดซอส', 'th', true, 0.95, 8),
('lemma_009', 'ขวดซอสแฟนซี', 'th', false, 0.85, 3),
('lemma_009', 'ขวดโหล', 'th', false, 0.80, 8),
('lemma_009', 'โหลแก้ว', 'th', false, 0.75, 6),

-- อุปกรณ์ตัดหั่น (lemma_010)
('lemma_010', 'เขียง', 'th', true, 0.95, 6),
('lemma_010', 'เขียงอเนกประสงค์', 'th', false, 0.85, 1),
('lemma_010', 'เขียงมะขาม', 'th', false, 0.80, 3),
('lemma_010', 'มีด', 'th', true, 0.95, 12),
('lemma_010', 'มีดสับ', 'th', false, 0.85, 2),
('lemma_010', 'มีดเหล็กกล้า', 'th', false, 0.80, 3),

-- อุปกรณ์รับประทาน (lemma_011)
('lemma_011', 'ช้อน', 'th', true, 0.95, 3),
('lemma_011', 'ช้อนกาแฟ', 'th', false, 0.80, 1),
('lemma_011', 'คีบ', 'th', true, 0.85, 4),
('lemma_011', 'คีบสลัด', 'th', false, 0.80, 2),

-- อุปกรณ์ตวงวัด (lemma_012)
('lemma_012', 'เหยือกตวง', 'th', true, 0.90, 2),
('lemma_012', 'ตักน้ำแข็ง', 'th', false, 0.85, 3),
('lemma_012', 'แก้ว', 'th', true, 0.85, 2),

-- อุปกรณ์ขัดล้าง (lemma_013)
('lemma_013', 'ฟองน้ำ', 'th', true, 0.95, 8),
('lemma_013', 'ฟองน้ำตาข่าย', 'th', false, 0.85, 3),
('lemma_013', 'ใยขัด', 'th', true, 0.90, 12),
('lemma_013', 'ฝอยขัด', 'th', false, 0.85, 6),
('lemma_013', 'ไบร์ท', 'th', false, 0.80, 4),

-- แปรงทำความสะอาด (lemma_014)
('lemma_014', 'แปรง', 'th', true, 0.95, 15),
('lemma_014', 'แปรงซักผ้า', 'th', false, 0.85, 8),
('lemma_014', 'แปรงห้องน้ำ', 'th', false, 0.80, 1),
('lemma_014', 'แปรงเตารีด', 'th', false, 0.75, 2),
('lemma_014', 'แปรงลวด', 'th', false, 0.80, 3),

-- ผ้าทำความสะอาด (lemma_015)
('lemma_015', 'ผ้า', 'th', true, 0.90, 4),
('lemma_015', 'ผ้าไมโคร', 'th', false, 0.85, 1),
('lemma_015', 'ผ้าขาว', 'th', false, 0.80, 1),

-- อุปกรณ์ตัด (lemma_016)
('lemma_016', 'กรรไกร', 'th', true, 0.95, 4),

-- อุปกรณ์ติด (lemma_017)
('lemma_017', 'กาว', 'th', true, 0.95, 4),
('lemma_017', 'กาวช้าง', 'th', false, 0.85, 1),
('lemma_017', 'เทป', 'th', true, 0.90, 2),
('lemma_017', 'เทปโฟม', 'th', false, 0.80, 1),

-- ของเล่นยานพาหนะ (lemma_018)
('lemma_018', 'รถ', 'th', true, 0.95, 8),
('lemma_018', 'รถจ้าว', 'th', false, 0.85, 3),
('lemma_018', 'รถตักดิน', 'th', false, 0.80, 2),
('lemma_018', 'รถสามล้อ', 'th', false, 0.75, 1),

-- ของเล่นอาวุธ (lemma_019)
('lemma_019', 'ปืน', 'th', true, 0.90, 6),
('lemma_019', 'ปืนฉีดน้ำ', 'th', false, 0.85, 6),

-- ผลิตภัณฑ์อาบน้ำ (lemma_020)
('lemma_020', 'ครีมอาบน้ำ', 'th', true, 0.90, 1),
('lemma_020', 'โฟรเทคส์', 'th', false, 0.75, 1),

-- อุปกรณ์โกนหนวด (lemma_021)
('lemma_021', 'ใบมีดโกน', 'th', true, 0.90, 1),
('lemma_021', 'โกบหนวด', 'th', false, 0.80, 1),

-- อุปกรณ์สัตว์เลี้ยง (lemma_022)
('lemma_022', 'ปลอกคอ', 'th', true, 0.95, 3),
('lemma_022', 'ปลอกคอหมา', 'th', false, 0.90, 2),
('lemma_022', 'ปลอกคอแมว', 'th', false, 0.85, 1),

-- อุปกรณ์ไฟฟ้าพกพา (lemma_023)
('lemma_023', 'ไฟฉาย', 'th', true, 0.90, 1),
('lemma_023', 'หูฟัง', 'th', true, 0.85, 2),
('lemma_023', 'สมอลทอด', 'th', false, 0.70, 2),

-- อุปกรณ์ให้นมเด็ก (lemma_024)
('lemma_024', 'จุกนม', 'th', true, 0.90, 1),
('lemma_024', 'ขวดนม', 'th', false, 0.85, 1),

-- อุปกรณ์ความงาม (lemma_025)
('lemma_025', 'แหนบ', 'th', true, 0.90, 8),
('lemma_025', 'แหนบหุ้มผึ้ง', 'th', false, 0.80, 2),
('lemma_025', 'หวี', 'th', true, 0.85, 6),
('lemma_025', 'หวีแปรง', 'th', false, 0.80, 4),

-- วัสดุตกแต่ง (lemma_026)
('lemma_026', 'ริบบิ้น', 'th', true, 0.85, 1),
('lemma_026', 'กระดาษห่อของขวัญ', 'th', false, 0.80, 1),
('lemma_026', 'เรนโบว์', 'th', false, 0.75, 2);


-- ==================================================
-- Rules Dataset
-- 25 keyword rules + 10 regex rules + system settings
-- ==================================================

-- AI Rules Dataset for Thai Product Classification
-- Generated from รายการสินค้าพร้อมหมวดหมู่_AI.txt analysis

-- Insert Keyword Rules
INSERT INTO keyword_rules (id, name, description, keywords, category_id, match_type, priority, confidence_score, is_active) VALUES

-- เครื่องมือ_ฮาร์ดแวร์ Rules
('rule_kw_001', 'Garden Tools Detection', 'ตรวจจับเครื่องมือทำสวน', 
 ARRAY['คราด', 'เสียม', 'สายยาง', 'กระถาง', 'ทำสวน', 'ต้นไม้', 'กรรไกรตัดหญ้า'], 
 'cat_001_001', 'contains', 10, 0.90, true),

('rule_kw_002', 'Hand Tools Detection', 'ตรวจจับเครื่องมือช่าง', 
 ARRAY['ค้อน', 'หกเหลี่ยม', 'เอ็น', 'ตลับเมตร', 'ระดับน้ำ', 'ไขควง'], 
 'cat_001_002', 'contains', 9, 0.85, true),

('rule_kw_003', 'Paint Tools Detection', 'ตรวจจับอุปกรณ์ทาสี', 
 ARRAY['แปรงทาสี', 'กระดาษทราย', 'ทาสี', 'สี'], 
 'cat_001_003', 'contains', 8, 0.85, true),

('rule_kw_004', 'Electrical Tools Detection', 'ตรวจจับอุปกรณ์ไฟฟ้า', 
 ARRAY['หัวแร้ง', 'สายพ่วง', 'ปลั๊ก', 'ไฟฟ้า', 'วัดไฟ'], 
 'cat_001_005', 'contains', 8, 0.80, true),

-- ของใช้ในบ้าน Rules
('rule_kw_005', 'Water Containers Detection', 'ตรวจจับภาชนะใส่น้ำ', 
 ARRAY['ขัน', 'กระติก', 'กระบอกน้ำ', 'น้ำ'], 
 'cat_002_001', 'contains', 10, 0.90, true),

('rule_kw_006', 'Storage Boxes Detection', 'ตรวจจับกล่องเก็บของ', 
 ARRAY['กล่อง', 'ล็อค', 'ลิ้นชัก', 'เก็บของ', 'ล้อ'], 
 'cat_002_003', 'contains', 9, 0.85, true),

('rule_kw_007', 'Baskets Detection', 'ตรวจจับตะกร้าและกระจาด', 
 ARRAY['ตะกร้า', 'กระจาด', 'กลม', 'เหลี่ยม'], 
 'cat_002_004', 'contains', 8, 0.85, true),

('rule_kw_008', 'Tanks and Buckets Detection', 'ตรวจจับถังและกะละมัง', 
 ARRAY['ถัง', 'กะละมัง', 'ฝา', 'ปูน'], 
 'cat_002_006', 'contains', 8, 0.80, true),

('rule_kw_009', 'Mirrors Detection', 'ตรวจจับกระจก', 
 ARRAY['กระจก', 'พับ', 'หัวใจ', 'โค้ง'], 
 'cat_002_008', 'contains', 7, 0.85, true),

-- เครื่องครัว Rules
('rule_kw_010', 'Sauce Bottles Detection', 'ตรวจจับขวดซอส', 
 ARRAY['ขวดซอส', 'ซอส', 'เครื่องปรุง', 'แฟนซี'], 
 'cat_003_001', 'contains', 10, 0.90, true),

('rule_kw_011', 'Cutting Boards Detection', 'ตรวจจับเขียง', 
 ARRAY['เขียง', 'อเนกประสงค์', 'มะขาม'], 
 'cat_003_003', 'contains', 9, 0.90, true),

('rule_kw_012', 'Kitchen Knives Detection', 'ตรวจจับมีดครัว', 
 ARRAY['มีด', 'สับ', 'เหล็กกล้า', 'นกแก้ว', 'KIWI'], 
 'cat_003_007', 'contains', 9, 0.85, true),

('rule_kw_013', 'Measuring Tools Detection', 'ตรวจจับอุปกรณ์ตวง', 
 ARRAY['เหยือกตวง', 'ตวง', 'ตักน้ำแข็ง', 'วัด'], 
 'cat_003_002', 'contains', 8, 0.80, true),

-- อุปกรณ์ทำความสะอาด Rules
('rule_kw_014', 'Sponges Detection', 'ตรวจจับฟองน้ำและใยขัด', 
 ARRAY['ฟองน้ำ', 'ใยขัด', 'ฝอยขัด', 'ตาข่าย', 'ไบร์ท'], 
 'cat_004_001', 'contains', 10, 0.90, true),

('rule_kw_015', 'Brushes Detection', 'ตรวจจับแปรงทำความสะอาด', 
 ARRAY['แปรง', 'ซักผ้า', 'ห้องน้ำ', 'เตารีด', 'ลวด'], 
 'cat_004_002', 'contains', 9, 0.85, true),

('rule_kw_016', 'Waste Management Detection', 'ตรวจจับถังขยะ', 
 ARRAY['ถังขยะ', 'ขยะ', 'กทม'], 
 'cat_004_004', 'contains', 8, 0.85, true),

-- เครื่องเขียน Rules
('rule_kw_017', 'Scissors Detection', 'ตรวจจับกรรไกร', 
 ARRAY['กรรไกร', 'ตัด'], 
 'cat_005_001', 'contains', 9, 0.90, true),

('rule_kw_018', 'Glue and Tape Detection', 'ตรวจจับกาวและเทป', 
 ARRAY['กาว', 'เทป', 'ช้าง', 'โฟม', 'TOA'], 
 'cat_005_002', 'contains', 8, 0.85, true),

-- ของเล่น Rules
('rule_kw_019', 'Toy Cars Detection', 'ตรวจจับรถของเล่น', 
 ARRAY['รถ', 'จ้าว', 'ทะเลทราย', 'ตักดิน', 'สามล้อ', 'มอเตอร์ไซค์'], 
 'cat_006_001', 'contains', 9, 0.85, true),

('rule_kw_020', 'Water Guns Detection', 'ตรวจจับปืนฉีดน้ำ', 
 ARRAY['ปืน', 'ฉีดน้ำ', 'น้ำ'], 
 'cat_006_002', 'contains', 8, 0.80, true),

-- ผลิตภัณฑ์ส่วนบุคคล Rules
('rule_kw_021', 'Personal Care Detection', 'ตรวจจับผลิตภัณฑ์ส่วนบุคคล', 
 ARRAY['น้ำอบไทย', 'ใบมีดโกน', 'ครีมอาบน้ำ', 'โกบหนวด'], 
 'cat_007', 'contains', 7, 0.75, true),

-- สัตว์เลี้ยง Rules
('rule_kw_022', 'Pet Collars Detection', 'ตรวจจับปลอกคอสัตว์เลี้ยง', 
 ARRAY['ปลอกคอ', 'หมา', 'แมว', 'สัตว์เลี้ยง'], 
 'cat_009_001', 'contains', 8, 0.85, true),

-- เครื่องใช้ไฟฟ้า Rules
('rule_kw_023', 'Electronics Detection', 'ตรวจจับเครื่องใช้ไฟฟ้า', 
 ARRAY['ไฟฉาย', 'หูฟัง', 'ชาร์จ', 'สมอลทอด'], 
 'cat_010', 'contains', 7, 0.75, true),

-- แม่และเด็ก Rules
('rule_kw_024', 'Baby Products Detection', 'ตรวจจับสินค้าเด็ก', 
 ARRAY['จุกนม', 'ขวดนม', 'เด็ก', 'โกกิ'], 
 'cat_011_001', 'contains', 8, 0.80, true),

-- เบ็ดเตล็ด Rules
('rule_kw_025', 'Beauty Accessories Detection', 'ตรวจจับอุปกรณ์ความงาม', 
 ARRAY['แหนบ', 'หวี', 'แปรง', 'ความงาม'], 
 'cat_012_002', 'contains', 6, 0.70, true);

-- Insert Regex Rules
INSERT INTO regex_rules (id, name, description, pattern, flags, category_id, priority, confidence_score, is_active, test_cases) VALUES

-- Size and Dimension Patterns
('rule_rx_001', 'Size Detection', 'ตรวจจับขนาดและมิติ', 
 '\\b\\d+(\\.\\d+)?\\s*(ซม|cm|นิ้ว|inch|"|ลิตร|L|l|มล|ml|กรัม|g|kg|กก)\\b', 'gi', 
 NULL, 5, 0.70, true, 
 ARRAY['12 นิ้ว', '3 ลิตร', '5L', '100 มล', '2.5 ซม']),

-- Model Number Patterns
('rule_rx_002', 'Model Number Detection', 'ตรวจจับหมายเลขรุ่น', 
 '\\b(NO\\.|#|เบอร์)\\s*[A-Z0-9\\-]+\\b', 'gi', 
 NULL, 4, 0.60, true, 
 ARRAY['NO.304', '#9859', 'เบอร์ 1']),

-- Color Patterns
('rule_rx_003', 'Color Detection', 'ตรวจจับสี', 
 '\\b(สี)?(แดง|เขียว|น้ำเงิน|เหลือง|ขาว|ดำ|ชมพู|ม่วง|ส้ม|เทา|หวาน|สด|ใส|สีไม้)\\b', 'gi', 
 NULL, 3, 0.50, true, 
 ARRAY['สีแดง', 'เขียว', 'สีหวาน', 'ใส']),

-- Brand Patterns
('rule_rx_004', 'Brand Detection', 'ตรวจจับแบรนด์', 
 '\\b(ALLWAYS|KIWI|TOA|DORCO|ATM|NCL|SMT|SRT|KEY|IMM)\\b', 'gi', 
 NULL, 6, 0.75, true, 
 ARRAY['ALLWAYS', 'KIWI no.171', 'TOA']),

-- Quantity Patterns
('rule_rx_005', 'Quantity Detection', 'ตรวจจับจำนวน', 
 '\\b(\\d+)\\s*(ชิ้น|อัน|ใบ|ตัว|หัว|ชั้น|ช่อง)\\b', 'gi', 
 NULL, 4, 0.60, true, 
 ARRAY['3 ชิ้น', '5 อัน', '2 ชั้น']),

-- Kitchen Specific Patterns
('rule_rx_006', 'Kitchen Items Pattern', 'รูปแบบเฉพาะเครื่องครัว', 
 '\\b(ขวด|เหยือก|ช้อน|ส้อม|มีด|เขียง|กระทะ|หม้อ)\\b', 'gi', 
 'cat_003', 8, 0.80, true, 
 ARRAY['ขวดซอส', 'เหยือกตวง', 'มีดสับ']),

-- Cleaning Supplies Pattern
('rule_rx_007', 'Cleaning Supplies Pattern', 'รูปแบบอุปกรณ์ทำความสะอาด', 
 '\\b(ฟองน้ำ|ใยขัด|แปรง|ผ้า|ทำความสะอาด|ขัด|ล้าง)\\b', 'gi', 
 'cat_004', 8, 0.80, true, 
 ARRAY['ฟองน้ำตาข่าย', 'แปรงซักผ้า', 'ใยขัดชุด']),

-- Hardware Tools Pattern
('rule_rx_008', 'Hardware Tools Pattern', 'รูปแบบเครื่องมือฮาร์ดแวร์', 
 '\\b(คราด|เสียม|ค้อน|เอ็น|แปรงทาสี|สายยาง|กระถาง)\\b', 'gi', 
 'cat_001', 9, 0.85, true, 
 ARRAY['คราดมือเสือ', 'ค้อนหัวยาง', 'เอ็นขาว']),

-- Household Items Pattern
('rule_rx_009', 'Household Items Pattern', 'รูปแบบของใช้ในบ้าน', 
 '\\b(ขัน|กระติก|กล่อง|ตะกร้า|ถัง|กะละมัง|กระจก)\\b', 'gi', 
 'cat_002', 8, 0.80, true, 
 ARRAY['ขันน้ำ', 'กล่องล็อค', 'ตะกร้ากลม']),

-- Toys Pattern
('rule_rx_010', 'Toys Pattern', 'รูปแบบของเล่น', 
 '\\b(รถ|ปืน|ของเล่น|เด็ก|ฉีดน้ำ)\\b', 'gi', 
 'cat_006', 7, 0.75, true, 
 ARRAY['รถจ้าว', 'ปืนฉีดน้ำ', 'รถตักดิน']);

-- Insert System Settings
INSERT INTO system_settings (id, search, processing, ai, ui) VALUES 
(gen_random_uuid(), 
 '{
   "vectorSearchEnabled": true,
   "textSearchEnabled": true, 
   "hybridSearchEnabled": true,
   "defaultSearchType": "hybrid",
   "maxResults": 50,
   "confidenceThreshold": 0.3
 }',
 '{
   "batchSize": 100,
   "maxConcurrentJobs": 5,
   "retryAttempts": 3,
   "timeoutSeconds": 30
 }',
 '{
   "embeddingModel": "paraphrase-multilingual-MiniLM-L12-v2",
   "apiProvider": "openai",
   "maxTokens": 4000,
   "temperature": 0.1
 }',
 '{
   "theme": "light",
   "language": "th", 
   "itemsPerPage": 20,
   "enableAnimations": true
 }');


