-- Thai Product Taxonomy System - Hybrid UUID+Code Schema
-- Based on schema.sql with added code columns for human-readable references
-- PK = UUID (internal), Code = TEXT (human stable, external references)

-- เปิดใช้งาน extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto; -- for gen_random_uuid()

-- ตาราง taxonomy_nodes (โหนดหมวดหมู่สินค้า)
CREATE TABLE IF NOT EXISTS taxonomy_nodes (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    code TEXT UNIQUE NOT NULL, -- human-readable code: cat_001, cat_001_001, etc.
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

-- ตาราง synonym_lemmas (กลุ่มคำพ้องความหมาย) - renamed from synonyms for dataset compatibility
CREATE TABLE IF NOT EXISTS synonym_lemmas (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    code TEXT UNIQUE NOT NULL, -- lemma_001, lemma_002, etc.
    name_th TEXT NOT NULL, -- ชื่อกลุ่ม synonym
    name_en TEXT,
    description TEXT,
    category_id UUID REFERENCES taxonomy_nodes(id) ON DELETE SET NULL,
    is_verified BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT true,
    created_by UUID,
    updated_by UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ตาราง synonym_terms (คำในกลุ่ม synonym)
CREATE TABLE IF NOT EXISTS synonym_terms (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    lemma_id UUID REFERENCES synonym_lemmas(id) ON DELETE CASCADE, -- references lemmas not synonyms
    term TEXT NOT NULL,
    is_primary BOOLEAN DEFAULT false, -- คำหลักของกลุ่ม
    confidence_score FLOAT DEFAULT 1.0,
    usage_count INTEGER DEFAULT 0,
    source TEXT DEFAULT 'manual', -- manual, auto, imported, ml
    language TEXT DEFAULT 'th', -- th, en, mixed
    is_verified BOOLEAN DEFAULT false,
    created_by UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(lemma_id, term)
);

-- ตาราง products (สินค้า)
CREATE TABLE IF NOT EXISTS products (
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
CREATE TABLE IF NOT EXISTS keyword_rules (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    code TEXT UNIQUE NOT NULL, -- rule_kw_001, rule_kw_002, etc.
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

-- ตาราง regex_rules (กฎการตรวจสอบด้วย regex)
CREATE TABLE IF NOT EXISTS regex_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code TEXT UNIQUE NOT NULL, -- rule_rx_001, rule_rx_002, etc.
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

-- ตาราง system_settings (การตั้งค่าระบบ) - UUID only, no code needed
CREATE TABLE IF NOT EXISTS system_settings (
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
        "embeddingModel": "text-embedding-ada-002",
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

-- ตาราง product_category_suggestions (คำแนะนำหมวดหมู่สำหรับสินค้า)
CREATE TABLE IF NOT EXISTS product_category_suggestions (
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
CREATE TABLE IF NOT EXISTS imports (
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
CREATE TABLE IF NOT EXISTS audit_logs (
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

-- ตาราง product_attributes (คุณสมบัติสินค้า)
CREATE TABLE IF NOT EXISTS product_attributes (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    product_id UUID REFERENCES products(id) ON DELETE CASCADE,
    attribute_name TEXT NOT NULL,
    attribute_value TEXT NOT NULL,
    attribute_type TEXT DEFAULT 'text', -- text, number, boolean, date
    created_by UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ตาราง review_history (ประวัติการตรวจสอบ)
CREATE TABLE IF NOT EXISTS review_history (
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
CREATE TABLE IF NOT EXISTS similarity_matches (
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
CREATE INDEX idx_taxonomy_nodes_code ON taxonomy_nodes(code); -- UNIQUE already creates index but explicit for clarity

-- Synonym lemmas indexes
CREATE INDEX idx_synonym_lemmas_category_id ON synonym_lemmas(category_id);
CREATE INDEX idx_synonym_lemmas_active ON synonym_lemmas(is_active) WHERE is_active = true;
CREATE INDEX idx_synonym_lemmas_code ON synonym_lemmas(code); -- UNIQUE index

-- Synonym terms indexes
CREATE INDEX idx_synonym_terms_lemma_id ON synonym_terms(lemma_id);
CREATE INDEX idx_synonym_terms_term ON synonym_terms(term);
CREATE INDEX idx_synonym_terms_primary ON synonym_terms(is_primary) WHERE is_primary = true;
CREATE INDEX idx_synonym_terms_verified ON synonym_terms(is_verified);

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
CREATE INDEX idx_keyword_rules_code ON keyword_rules(code); -- UNIQUE index

-- Regex rules indexes
CREATE INDEX idx_regex_rules_category ON regex_rules(category_id);
CREATE INDEX idx_regex_rules_active ON regex_rules(is_active) WHERE is_active = true;
CREATE INDEX idx_regex_rules_priority ON regex_rules(priority DESC);
CREATE INDEX idx_regex_rules_code ON regex_rules(code); -- UNIQUE index

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
CREATE TRIGGER update_synonym_lemmas_updated_at BEFORE UPDATE ON synonym_lemmas FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_products_updated_at BEFORE UPDATE ON products FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_keyword_rules_updated_at BEFORE UPDATE ON keyword_rules FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_regex_rules_updated_at BEFORE UPDATE ON regex_rules FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function สำหรับ audit logging
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO audit_logs (table_name, record_id, action, old_values, user_id)
        VALUES (TG_TABLE_NAME, OLD.id::TEXT, 'DELETE', row_to_json(OLD), current_setting('app.current_user_id', true)::UUID);
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_logs (table_name, record_id, action, old_values, new_values, user_id)
        VALUES (TG_TABLE_NAME, NEW.id::TEXT, 'UPDATE', row_to_json(OLD), row_to_json(NEW), current_setting('app.current_user_id', true)::UUID);
        RETURN NEW;
    ELSIF TG_OP = 'INSERT' THEN
        INSERT INTO audit_logs (table_name, record_id, action, new_values, user_id)
        VALUES (TG_TABLE_NAME, NEW.id::TEXT, 'INSERT', row_to_json(NEW), current_setting('app.current_user_id', true)::UUID);
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Audit triggers
CREATE TRIGGER audit_taxonomy_nodes AFTER INSERT OR UPDATE OR DELETE ON taxonomy_nodes FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
CREATE TRIGGER audit_synonym_lemmas AFTER INSERT OR UPDATE OR DELETE ON synonym_lemmas FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
CREATE TRIGGER audit_products AFTER INSERT OR UPDATE OR DELETE ON products FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
CREATE TRIGGER audit_keyword_rules AFTER INSERT OR UPDATE OR DELETE ON keyword_rules FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

-- Row Level Security (RLS) policies
ALTER TABLE taxonomy_nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE synonym_lemmas ENABLE ROW LEVEL SECURITY;
ALTER TABLE synonym_terms ENABLE ROW LEVEL SECURITY;
ALTER TABLE products ENABLE ROW LEVEL SECURITY;
ALTER TABLE keyword_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE regex_rules ENABLE ROW LEVEL SECURITY;
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
GRANT INSERT, UPDATE, DELETE ON synonym_lemmas TO taxonomy_editor;
GRANT INSERT, UPDATE, DELETE ON synonym_terms TO taxonomy_editor;
GRANT INSERT, UPDATE, DELETE ON products TO taxonomy_editor;
GRANT INSERT, UPDATE, DELETE ON keyword_rules TO taxonomy_editor;
GRANT INSERT, UPDATE, DELETE ON regex_rules TO taxonomy_editor;
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

-- Synonym lemmas policies
CREATE POLICY "synonym_lemmas_read" ON synonym_lemmas FOR SELECT USING (true);
CREATE POLICY "synonym_lemmas_insert" ON synonym_lemmas FOR INSERT WITH CHECK (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));
CREATE POLICY "synonym_lemmas_update" ON synonym_lemmas FOR UPDATE USING (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));
CREATE POLICY "synonym_lemmas_delete" ON synonym_lemmas FOR DELETE USING (auth.role() IN ('taxonomy_editor', 'taxonomy_admin'));

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
